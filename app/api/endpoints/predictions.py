"""
Prediction endpoints for email interview classification.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from slowapi import Limiter

from app.api.dependencies import RateLimiterDep, ValidatedEmailDep
from app.api.model_dependencies import ModelManagerDep
from app.core.config import settings
from app.core.security import create_error_response, validate_email_content
from app.schemas.prediction import PredictionRequest, PredictionResponse

# Initialize logger
logger = structlog.get_logger()

# Create router
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify email as interview or non-interview",
    description="Predict whether an email is interview-related or not",
    responses={
        200: {
            "description": "Successful prediction",
            "model": PredictionResponse
        },
        400: {
            "description": "Invalid request format",
            "content": {
                "application/json": {
                    "example": create_error_response(
                        "Invalid request format. Missing 'email_body' field.",
                        400
                    )
                }
            }
        },
        429: {
            "description": "Rate limit exceeded"
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": create_error_response(
                        "Model not found. Please run training first.",
                        500
                    )
                }
            }
        },
        503: {
            "description": "Service unavailable - model not loaded"
        }
    }
)
async def predict_email(
    request: PredictionRequest,
    model_manager: ModelManagerDep,
    rate_limiter: RateLimiterDep
):
    """
    Classify an email as interview-related or not.
    
    Args:
        request: Prediction request containing email body
        model_manager: ML model manager dependency
        rate_limiter: Rate limiter dependency
        
    Returns:
        PredictionResponse with classification result and confidence
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Validate email body
        email_body = validate_email_content(request.email_body)
        
        # Log prediction request (without sensitive data)
        logger.info(
            "Prediction request received",
            email_length=len(email_body),
            has_model=model_manager.is_loaded()
        )
        
        # Make prediction
        prediction_result = await model_manager.predict(email_body)
        
        # Log prediction result
        logger.info(
            "Prediction completed",
            is_interview=prediction_result["is_interview"],
            confidence=prediction_result["confidence"]
        )
        
        return PredictionResponse(
            is_interview=prediction_result["is_interview"],
            confidence=prediction_result["confidence"]
        )
        
    except Exception as e:
        logger.error(
            "Prediction failed",
            error=str(e),
            email_length=len(email_body)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again."
        )


@router.get(
    "/model-info",
    summary="Get model information",
    description="Get information about the loaded model",
    responses={
        200: {
            "description": "Model information retrieved successfully"
        },
        503: {
            "description": "Service unavailable - model not loaded"
        }
    }
)
async def get_model_info(model_manager: ModelManagerDep):
    """
    Get information about the loaded model.
    
    Args:
        model_manager: ML model manager dependency
        
    Returns:
        Model information including metadata and performance metrics
    """
    try:
        model_info = await model_manager.get_model_info()
        
        logger.info("Model info requested", model_loaded=True)
        
        return {
            "model_loaded": True,
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )
