"""
Pydantic schemas for prediction requests and responses.
"""

from typing import Literal

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """
    Request schema for email prediction.
    """
    email_body: str = Field(
        ...,
        description="The email body text to classify",
        min_length=1,
        max_length=10000,
        example="We'd love to schedule a 30-minute video conversation to learn more about your experience..."
    )
    
    @validator('email_body')
    def validate_email_body(cls, v):
        """Validate email body content."""
        if not v or not v.strip():
            raise ValueError('Email body cannot be empty')
        return v.strip()


class PredictionResponse(BaseModel):
    """
    Response schema for email prediction.
    """
    is_interview: Literal["Yes", "No"] = Field(
        ...,
        description="Classification result: 'Yes' for interview-related, 'No' for non-interview",
        example="Yes"
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the prediction (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.85
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "is_interview": "Yes",
                "confidence": 0.85
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch email prediction.
    """
    emails: list[str] = Field(
        ...,
        description="List of email body texts to classify",
        min_items=1,
        max_items=100,
        example=[
            "We'd love to schedule a 30-minute video conversation...",
            "Your application has been received and is under review..."
        ]
    )
    
    @validator('emails')
    def validate_emails(cls, v):
        """Validate email list."""
        if not v:
            raise ValueError('Emails list cannot be empty')
        
        validated_emails = []
        for email in v:
            if not email or not email.strip():
                raise ValueError('Email body cannot be empty')
            validated_emails.append(email.strip())
        
        return validated_emails


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch email prediction.
    """
    predictions: list[PredictionResponse] = Field(
        ...,
        description="List of prediction results for each email"
    )
    total_processed: int = Field(
        ...,
        description="Total number of emails processed"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"is_interview": "Yes", "confidence": 0.85},
                    {"is_interview": "No", "confidence": 0.92}
                ],
                "total_processed": 2
            }
        }
