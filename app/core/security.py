"""
Security utilities for the email classification service.
"""

import re
from typing import Any, Dict, Optional

from fastapi import HTTPException, status


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize input text to prevent potential security issues.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Raises:
        HTTPException: If input is invalid
    """
    if not isinstance(text, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input must be a string"
        )
    
    # Check length
    if len(text) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long. Maximum length is {max_length} characters"
        )
    
    # Remove potential script tags and dangerous characters
    # Keep only alphanumeric, spaces, and common punctuation
    sanitized = re.sub(r'[<>"\']', '', text)
    
    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    return sanitized


def validate_email_content(email_body: str) -> str:
    """
    Validate and sanitize email body content.
    
    Args:
        email_body: Email body text
        
    Returns:
        Validated and sanitized email body
        
    Raises:
        HTTPException: If email body is invalid
    """
    if not email_body or not email_body.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email body cannot be empty"
        )
    
    return sanitize_input(email_body.strip())


def create_error_response(error_message: str, error_code: int = 500) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error_message: Error message
        error_code: HTTP status code
        
    Returns:
        Error response dictionary
    """
    return {
        "error": error_message,
        "code": error_code
    }


def log_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove sensitive information from data before logging.
    
    Args:
        data: Data dictionary to sanitize
        
    Returns:
        Sanitized data dictionary
    """
    sensitive_keys = ["password", "token", "secret", "key", "auth"]
    sanitized = data.copy()
    
    for key in sanitized:
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "[REDACTED]"
    
    return sanitized
