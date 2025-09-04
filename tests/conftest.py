"""
Pytest configuration and fixtures for email classification service tests.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import sys
import importlib.util

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import modules with fallback
try:
    from app.core.config import settings
    from app.models.ml_model import MLModelManager
    from app.services.preprocessing import TextPreprocessor
except ImportError as e:
    # If import fails, try alternative approach
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"Project root exists: {project_root.exists()}")
    
    # Try to import using importlib
    try:
        spec = importlib.util.spec_from_file_location("app", project_root / "app" / "__init__.py")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        
        from app.core.config import settings
        from app.models.ml_model import MLModelManager
        from app.services.preprocessing import TextPreprocessor
    except Exception as e2:
        print(f"Alternative import also failed: {e2}")
        raise e


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
    manager.predict = AsyncMock(
        return_value={"is_interview": "Yes", "confidence": 0.85}
    )
    manager.get_model_info = AsyncMock(
        return_value={"model_type": "LogisticRegression", "accuracy": 0.89}
    )
    return manager


@pytest.fixture
def sample_email_data():
    """Sample email data for testing."""
    return {
        "interview_email": "We'd love to schedule a 30-minute video conversation to learn more about your experience and discuss the role in detail.",
        "non_interview_email": "Thank you for your application. We have received your resume and will review it carefully.",
        "security_code_email": "Your verification code is 123456. Please enter this code to verify your account.",
        "rejection_email": "Thank you for your interest in our company. Unfortunately, we have decided not to proceed with your application at this time.",
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
