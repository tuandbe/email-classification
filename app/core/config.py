"""
Configuration settings for the email classification service.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    app_name: str = "Email Interview Classification Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Model settings
    model_dir: Path = Field(default=Path("models"), env="MODEL_DIR")
    model_filename: str = Field(default="classifier.pkl", env="MODEL_FILENAME")
    vectorizer_filename: str = Field(default="vectorizer.pkl", env="VECTORIZER_FILENAME")
    metadata_filename: str = Field(default="metadata.json", env="METADATA_FILENAME")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default="logs/app.log", env="LOG_FILE")
    error_log_file: Optional[str] = Field(default="logs/error.log", env="ERROR_LOG_FILE")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # CORS settings
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: list[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: list[str] = Field(default=["*"], env="CORS_HEADERS")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def model_path(self) -> Path:
        """Get full path to model file."""
        return self.model_dir / self.model_filename
    
    @property
    def vectorizer_path(self) -> Path:
        """Get full path to vectorizer file."""
        return self.model_dir / self.vectorizer_filename
    
    @property
    def metadata_path(self) -> Path:
        """Get full path to metadata file."""
        return self.model_dir / self.metadata_filename
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.model_dir.mkdir(exist_ok=True)
        if self.log_file:
            Path(self.log_file).parent.mkdir(exist_ok=True)
        if self.error_log_file:
            Path(self.error_log_file).parent.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
