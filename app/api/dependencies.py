"""
Shared dependencies for API endpoints.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.security import validate_email_content
from app.api.model_dependencies import ModelManagerDep

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def get_rate_limiter() -> Limiter:
    """
    Get rate limiter instance.

    Returns:
        Limiter instance
    """
    return limiter


def validate_email_input(email_body: str) -> str:
    """
    Validate email input for prediction endpoints.

    Args:
        email_body: Email body text

    Returns:
        Validated email body

    Raises:
        HTTPException: If validation fails
    """
    return validate_email_content(email_body)


# Type aliases for dependency injection
RateLimiterDep = Annotated[Limiter, Depends(get_rate_limiter)]
ValidatedEmailDep = Annotated[str, Depends(validate_email_input)]
