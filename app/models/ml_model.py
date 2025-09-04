"""
ML model management for email interview classification.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import structlog
from sklearn.base import BaseEstimator

from app.core.config import settings
from app.services.preprocessing import TextPreprocessor

# Initialize logger
logger = structlog.get_logger()


class MLModelManager:
    """
    Manages ML model loading, prediction, and metadata.
    """

    def __init__(self):
        """Initialize the ML model manager."""
        self.model: Optional[BaseEstimator] = None
        self.vectorizer: Optional[BaseEstimator] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.preprocessor = TextPreprocessor()
        self._is_loaded = False

    async def load_model(self) -> None:
        """
        Load the trained model, vectorizer, and metadata.

        Raises:
            FileNotFoundError: If model files are not found
            Exception: If model loading fails
        """
        try:
            # Check if model files exist
            if not settings.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {settings.model_path}")

            if not settings.vectorizer_path.exists():
                raise FileNotFoundError(
                    f"Vectorizer file not found: {settings.vectorizer_path}"
                )

            if not settings.metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {settings.metadata_path}"
                )

            # Load model
            self.model = joblib.load(settings.model_path)
            logger.info(
                "Model loaded successfully", model_path=str(settings.model_path)
            )

            # Load vectorizer
            self.vectorizer = joblib.load(settings.vectorizer_path)
            logger.info(
                "Vectorizer loaded successfully",
                vectorizer_path=str(settings.vectorizer_path),
            )

            # Load metadata
            with open(settings.metadata_path, "r") as f:
                self.metadata = json.load(f)
            logger.info(
                "Metadata loaded successfully",
                metadata_path=str(settings.metadata_path),
            )

            self._is_loaded = True
            logger.info("ML model manager initialized successfully")

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            self._is_loaded = False
            raise

    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready.

        Returns:
            True if model is loaded, False otherwise
        """
        return (
            self._is_loaded and self.model is not None and self.vectorizer is not None
        )

    async def predict(self, email_body: str) -> Dict[str, Any]:
        """
        Predict if an email is interview-related.

        Args:
            email_body: Email body text to classify

        Returns:
            Dictionary with prediction result and confidence

        Raises:
            RuntimeError: If model is not loaded
            Exception: If prediction fails
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")

        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(email_body)

            # Vectorize text
            text_vector = self.vectorizer.transform([processed_text])

            # Make prediction
            prediction = self.model.predict(text_vector)[0]
            prediction_proba = self.model.predict_proba(text_vector)[0]

            # Get confidence (max probability)
            confidence = float(np.max(prediction_proba))

            # Convert prediction to string format
            is_interview = "Yes" if prediction == 1 else "No"

            result = {"is_interview": is_interview, "confidence": confidence}

            logger.info(
                "Prediction completed",
                is_interview=is_interview,
                confidence=confidence,
                text_length=len(email_body),
            )

            return result

        except Exception as e:
            logger.error("Prediction failed", error=str(e), text_length=len(email_body))
            raise

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")

        info = {
            "model_type": type(self.model).__name__,
            "vectorizer_type": type(self.vectorizer).__name__,
            "is_loaded": True,
        }

        # Add metadata if available
        if self.metadata:
            info.update(self.metadata)

        # Add additional model information
        if hasattr(self.model, "feature_importances_"):
            info["has_feature_importances"] = True

        if hasattr(self.vectorizer, "vocabulary_"):
            info["vocabulary_size"] = len(self.vectorizer.vocabulary_)

        return info

    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self._is_loaded = False
        logger.info("ML model manager cleaned up")
