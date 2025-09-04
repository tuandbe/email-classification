"""
Model dependencies for API endpoints.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.models.ml_model import MLModelManager


def get_model_manager() -> MLModelManager:
    """
    Dependency to get the global model manager.

    Returns:
        MLModelManager instance

    Raises:
        HTTPException: If model is not loaded
    """
    # Import here to avoid circular import
    from app.main import model_manager

    if model_manager is None or not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please run training first.",
        )
    return model_manager


# Type alias for dependency injection
ModelManagerDep = Annotated[MLModelManager, Depends(get_model_manager)]
