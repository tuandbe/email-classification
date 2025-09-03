"""
Pydantic schemas for health check endpoints.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    """
    status: str = Field(
        ...,
        description="Service status: 'healthy', 'degraded', or 'unhealthy'",
        example="healthy"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready",
        example=True
    )
    version: str = Field(
        ...,
        description="Service version",
        example="1.0.0"
    )
    uptime: Optional[str] = Field(
        None,
        description="Service uptime",
        example="2h 30m 15s"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "uptime": "2h 30m 15s"
            }
        }


class ModelInfoResponse(BaseModel):
    """
    Response schema for model information endpoint.
    """
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded",
        example=True
    )
    model_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed model information including metadata and performance metrics",
        example={
            "model_type": "LogisticRegression",
            "training_date": "2024-01-15T10:30:00Z",
            "accuracy": 0.89,
            "precision": 0.87,
            "recall": 0.91,
            "f1_score": 0.89,
            "feature_count": 5000,
            "training_samples": 100
        }
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_loaded": True,
                "model_info": {
                    "model_type": "LogisticRegression",
                    "training_date": "2024-01-15T10:30:00Z",
                    "accuracy": 0.89,
                    "precision": 0.87,
                    "recall": 0.91,
                    "f1_score": 0.89,
                    "feature_count": 5000,
                    "training_samples": 100
                }
            }
        }
