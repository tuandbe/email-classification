"""
Utility helper functions for the email classification service.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from pythonjsonlogger import jsonlogger


def setup_logging() -> None:
    """
    Setup structured logging for the application.
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # JSON formatter for console
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    console_handler.setFormatter(json_formatter)

    root_logger.addHandler(console_handler)


def ensure_directory(path: Path) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure
    """
    path.mkdir(parents=True, exist_ok=True)


def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage string.

    Args:
        confidence: Confidence score (0.0 to 1.0)

    Returns:
        Formatted confidence string
    """
    return f"{confidence:.1%}"


def validate_file_path(
    file_path: str, allowed_extensions: Optional[list] = None
) -> Path:
    """
    Validate and return a Path object for a file.

    Args:
        file_path: File path string
        allowed_extensions: List of allowed file extensions

    Returns:
        Path object

    Raises:
        ValueError: If file path is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if allowed_extensions:
        if path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"File extension not allowed. Allowed: {allowed_extensions}"
            )

    return path


def safe_json_load(file_path: Path) -> Dict[str, Any]:
    """
    Safely load JSON data from a file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data

    Raises:
        ValueError: If JSON is invalid
        FileNotFoundError: If file doesn't exist
    """
    import json

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")


def safe_json_save(data: Dict[str, Any], file_path: Path) -> None:
    """
    Safely save data as JSON to a file.

    Args:
        data: Data to save
        file_path: Path to save file
    """
    import json

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    return file_path.stat().st_size / (1024 * 1024)


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_response_metadata(
    request_id: Optional[str] = None,
    processing_time: Optional[float] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create metadata for API responses.

    Args:
        request_id: Request identifier
        processing_time: Processing time in seconds
        model_version: Model version

    Returns:
        Metadata dictionary
    """
    metadata = {}

    if request_id:
        metadata["request_id"] = request_id

    if processing_time is not None:
        metadata["processing_time"] = processing_time
        metadata["processing_time_formatted"] = format_duration(processing_time)

    if model_version:
        metadata["model_version"] = model_version

    return metadata
