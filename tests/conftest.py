"""
Pytest configuration and fixtures for email classification service tests.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings
from app.models.ml_model import MLModelManager
from app.services.preprocessing import TextPreprocessor


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_model_manager():
    """Create a mock ML model manager."""
    manager = Mock(spec=MLModelManager)
    manager.is_loaded.return_value = True
    manager.predict = AsyncMock(return_value={
        "is_interview": "Yes",
        "confidence": 0.85
    })
    manager.get_model_info = AsyncMock(return_value={
        "model_type": "LogisticRegression",
        "accuracy": 0.89
    })
    return manager


@pytest.fixture
def sample_email_data():
    """Sample email data for testing."""
    return {
        "interview_email": "We'd love to schedule a 30-minute video conversation to learn more about your experience and discuss the role in detail.",
        "non_interview_email": "Thank you for your application. We have received your resume and will review it carefully.",
        "security_code_email": "Your verification code is 123456. Please enter this code to verify your account.",
        "rejection_email": "Thank you for your interest in our company. Unfortunately, we have decided not to proceed with your application at this time."
    }


@pytest.fixture
def text_preprocessor():
    """Create a text preprocessor instance."""
    return TextPreprocessor()


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return """Body,is_interview
"We'd love to schedule a 30-minute video conversation to learn more about your experience",Yes
"Thank you for your application. We have received your resume and will review it carefully",No
"Your verification code is 123456. Please enter this code to verify your account",No
"Thank you for your interest in our company. Unfortunately, we have decided not to proceed with your application",No
"We would like to invite you for a phone interview next week",Yes"""
