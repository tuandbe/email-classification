"""
Main FastAPI application for email interview classification service.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.endpoints import predictions
from app.core.config import settings
from app.core.security import log_sensitive_data
from app.models.ml_model import MLModelManager
from app.utils.helpers import setup_logging


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global model manager
model_manager: MLModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    global model_manager

    # Startup
    logging.info("Starting Email Interview Classification Service...")

    # Setup logging
    setup_logging()
    logger = structlog.get_logger()

    # Ensure directories exist
    settings.ensure_directories()

    # Initialize model manager
    try:
        model_manager = MLModelManager()
        await model_manager.load_model()
        logger.info("Model loaded successfully", model_path=str(settings.model_path))
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        # Don't fail startup if model is not available
        model_manager = None

    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    if model_manager:
        model_manager.cleanup()
    logger.info("Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A service for classifying emails to determine if they are interview-related",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all HTTP requests and responses.
    """
    start_time = time.time()

    # Log request
    logger = structlog.get_logger()
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=get_remote_address(request),
    )

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=f"{process_time:.4f}s",
    )

    return response


# Include routers
app.include_router(predictions.router, prefix="/v1", tags=["predictions"])


@app.get("/")
async def root():
    """
    Root endpoint with basic service information.
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    model_loaded = model_manager is not None and model_manager.is_loaded()

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": settings.app_version,
        "uptime": "N/A",  # Could be implemented with startup time tracking
    }


# Model manager is now handled in app.api.model_dependencies


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
