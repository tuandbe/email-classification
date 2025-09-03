"""
Prediction service for email interview classification.
"""

from typing import Dict, List, Any

import structlog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from app.models.ml_model import MLModelManager
from app.services.preprocessing import TextPreprocessor

# Initialize logger
logger = structlog.get_logger()


class PredictionService:
    """
    Service for handling prediction-related business logic.
    """
    
    def __init__(self, model_manager: MLModelManager):
        """
        Initialize the prediction service.
        
        Args:
            model_manager: ML model manager instance
        """
        self.model_manager = model_manager
        self.preprocessor = TextPreprocessor()
    
    async def predict_single(self, email_body: str) -> Dict[str, Any]:
        """
        Predict classification for a single email.
        
        Args:
            email_body: Email body text
            
        Returns:
            Prediction result with confidence
        """
        return await self.model_manager.predict(email_body)
    
    async def predict_batch(self, email_bodies: List[str]) -> List[Dict[str, Any]]:
        """
        Predict classification for multiple emails.
        
        Args:
            email_bodies: List of email body texts
            
        Returns:
            List of prediction results
        """
        predictions = []
        
        for email_body in email_bodies:
            try:
                prediction = await self.predict_single(email_body)
                predictions.append(prediction)
            except Exception as e:
                logger.error("Batch prediction failed for email", error=str(e))
                # Add error result
                predictions.append({
                    "is_interview": "No",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        return predictions
    
    def evaluate_model_performance(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.model_manager.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        # Make predictions
        y_pred = self.model_manager.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate class-specific metrics
        precision_interview = precision_score(y_test, y_pred, pos_label=1)
        recall_interview = recall_score(y_test, y_pred, pos_label=1)
        f1_interview = f1_score(y_test, y_pred, pos_label=1)
        
        precision_non_interview = precision_score(y_test, y_pred, pos_label=0)
        recall_non_interview = recall_score(y_test, y_pred, pos_label=0)
        f1_non_interview = f1_score(y_test, y_pred, pos_label=0)
        
        return {
            "overall_accuracy": accuracy,
            "overall_precision": precision,
            "overall_recall": recall,
            "overall_f1": f1,
            "interview_precision": precision_interview,
            "interview_recall": recall_interview,
            "interview_f1": f1_interview,
            "non_interview_precision": precision_non_interview,
            "non_interview_recall": recall_non_interview,
            "non_interview_f1": f1_non_interview
        }
    
    def cross_validate_model(self, X, y, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        if not self.model_manager.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model_manager.model,
            X,
            y,
            cv=cv,
            scoring='accuracy'
        )
        
        return {
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": cv_scores.mean(),
            "std_cv_score": cv_scores.std(),
            "cv_folds": cv
        }
    
    def get_prediction_confidence_threshold(self) -> float:
        """
        Get the confidence threshold for predictions.
        
        Returns:
            Confidence threshold value
        """
        # This could be configurable or learned from validation data
        return 0.5
    
    def is_high_confidence_prediction(self, confidence: float) -> bool:
        """
        Check if a prediction has high confidence.
        
        Args:
            confidence: Prediction confidence score
            
        Returns:
            True if confidence is above threshold
        """
        threshold = self.get_prediction_confidence_threshold()
        return confidence >= threshold
    
    def get_prediction_explanation(self, email_body: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for a prediction.
        
        Args:
            email_body: Original email body
            prediction: Prediction result
            
        Returns:
            Explanation of the prediction
        """
        # Extract keywords
        keywords = self.preprocessor.extract_keywords(email_body, top_n=5)
        
        # Get text features
        features = self.preprocessor.get_text_features(email_body)
        
        # Determine explanation based on prediction
        is_interview = prediction["is_interview"] == "Yes"
        confidence = prediction["confidence"]
        
        if is_interview:
            explanation = "This email appears to be interview-related based on keywords and content patterns."
        else:
            explanation = "This email does not appear to be interview-related based on content analysis."
        
        return {
            "prediction": prediction,
            "explanation": explanation,
            "confidence_level": "high" if self.is_high_confidence_prediction(confidence) else "low",
            "key_features": {
                "keywords": keywords,
                "text_features": features
            }
        }
