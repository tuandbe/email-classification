#!/usr/bin/env python3
"""
Training script for email interview classification model.

Usage:
    python scripts/train.py data/Interview_vs_Non-Interview_Training_Emails__100_rows_.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.preprocessing import TextPreprocessor
from app.utils.helpers import ensure_directory, safe_json_save

# Initialize logger
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load training data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Loaded DataFrame

    Raises:
        ValueError: If data format is invalid
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info("Data loaded successfully", rows=len(df), columns=list(df.columns))

        # Validate required columns
        required_columns = ["Body", "is_interview"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for empty values
        if df["Body"].isna().any():
            logger.warning("Found empty email bodies, removing them")
            df = df.dropna(subset=["Body"])

        if df["is_interview"].isna().any():
            logger.warning("Found empty labels, removing them")
            df = df.dropna(subset=["is_interview"])

        logger.info("Data validation completed", final_rows=len(df))
        return df

    except Exception as e:
        logger.error("Failed to load data", error=str(e))
        raise


def preprocess_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the training data.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (X, y) arrays
    """
    logger.info("Starting data preprocessing")

    # Initialize preprocessor
    preprocessor = TextPreprocessor()

    # Preprocess text data
    processed_texts = []
    for text in df["Body"]:
        processed_text = preprocessor.preprocess(str(text))
        processed_texts.append(processed_text)

    # Convert labels to binary (Yes=1, No=0)
    y = (df["is_interview"] == "Yes").astype(int)

    # Convert to numpy arrays
    X = np.array(processed_texts)
    y = np.array(y)

    # Log class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(["No", "Yes"], counts))
    logger.info("Class distribution", distribution=class_distribution)

    return X, y


def create_vectorizer() -> TfidfVectorizer:
    """
    Create and configure TF-IDF vectorizer.

    Returns:
        Configured TfidfVectorizer
    """
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,  # minimum document frequency
        max_df=0.95,  # maximum document frequency
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train the classification model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    logger.info("Starting model training")

    # Create model with class weights to handle imbalance
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",  # Handle class imbalance
        C=1.0,
    )

    # Train model
    model.fit(X_train, y_train)

    logger.info("Model training completed")
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model performance")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)

    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()

    precision_interview = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_interview = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_interview = (
        2
        * (precision_interview * recall_interview)
        / (precision_interview + recall_interview)
        if (precision_interview + recall_interview) > 0
        else 0
    )

    precision_non_interview = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_non_interview = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_non_interview = (
        2
        * (precision_non_interview * recall_non_interview)
        / (precision_non_interview + recall_non_interview)
        if (precision_non_interview + recall_non_interview) > 0
        else 0
    )

    metrics = {
        "accuracy": accuracy,
        "precision_interview": precision_interview,
        "recall_interview": recall_interview,
        "f1_interview": f1_interview,
        "precision_non_interview": precision_non_interview,
        "recall_non_interview": recall_non_interview,
        "f1_non_interview": f1_non_interview,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": report,
    }

    logger.info(
        "Model evaluation completed", accuracy=accuracy, f1_interview=f1_interview
    )
    return metrics


def cross_validate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform cross-validation on the model.

    Args:
        model: Model to validate
        X: Features
        y: Labels

    Returns:
        Cross-validation results
    """
    logger.info("Performing cross-validation")

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    cv_results = {
        "cv_scores": cv_scores.tolist(),
        "mean_cv_score": float(cv_scores.mean()),
        "std_cv_score": float(cv_scores.std()),
        "cv_folds": 5,
    }

    logger.info("Cross-validation completed", mean_score=cv_results["mean_cv_score"])
    return cv_results


def save_model(
    model,
    vectorizer: TfidfVectorizer,
    metrics: dict,
    cv_results: dict,
    training_data_info: dict,
) -> None:
    """
    Save the trained model and metadata.

    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        metrics: Evaluation metrics
        cv_results: Cross-validation results
        training_data_info: Training data information
    """
    logger.info("Saving model and metadata")

    # Ensure model directory exists
    ensure_directory(settings.model_dir)

    # Save model
    joblib.dump(model, settings.model_path)
    logger.info("Model saved", path=str(settings.model_path))

    # Save vectorizer
    joblib.dump(vectorizer, settings.vectorizer_path)
    logger.info("Vectorizer saved", path=str(settings.vectorizer_path))

    # Create metadata
    metadata = {
        "model_type": type(model).__name__,
        "vectorizer_type": type(vectorizer).__name__,
        "training_date": datetime.now().isoformat(),
        "training_data": training_data_info,
        "performance_metrics": metrics,
        "cross_validation": cv_results,
        "model_parameters": {
            "max_features": vectorizer.max_features,
            "ngram_range": vectorizer.ngram_range,
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "class_weight": model.class_weight,
            "C": model.C,
            "max_iter": model.max_iter,
        },
    }

    # Save metadata
    safe_json_save(metadata, settings.metadata_path)
    logger.info("Metadata saved", path=str(settings.metadata_path))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train email interview classification model"
    )
    parser.add_argument("csv_path", type=str, help="Path to training CSV file")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--use-smote", action="store_true", help="Use SMOTE for oversampling"
    )

    args = parser.parse_args()

    try:
        # Load data
        csv_path = Path(args.csv_path)
        df = load_data(csv_path)

        # Preprocess data
        X, y = preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        logger.info(
            "Data split completed", train_size=len(X_train), test_size=len(X_test)
        )

        # Create and fit vectorizer
        vectorizer = create_vectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        logger.info(
            "Text vectorization completed", features=X_train_vectorized.shape[1]
        )

        # Apply SMOTE if requested
        if args.use_smote:
            logger.info("Applying SMOTE oversampling")
            smote = SMOTE(random_state=args.random_state)
            X_train_vectorized, y_train = smote.fit_resample(
                X_train_vectorized, y_train
            )
            logger.info(
                "SMOTE oversampling completed", new_train_size=len(X_train_vectorized)
            )

        # Train model
        model = train_model(X_train_vectorized, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test_vectorized, y_test)

        # Cross-validation
        cv_results = cross_validate_model(model, X_train_vectorized, y_train)

        # Training data info
        training_data_info = {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": X_train_vectorized.shape[1],
            "class_distribution": {
                "interview": int(np.sum(y == 1)),
                "non_interview": int(np.sum(y == 0)),
            },
        }

        # Save model and metadata
        save_model(model, vectorizer, metrics, cv_results, training_data_info)

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Model saved to: {settings.model_path}")
        print(f"Vectorizer saved to: {settings.vectorizer_path}")
        print(f"Metadata saved to: {settings.metadata_path}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Interview F1-Score: {metrics['f1_interview']:.3f}")
        print(f"  Interview Recall: {metrics['recall_interview']:.3f}")
        print(
            f"  Cross-validation Score: {cv_results['mean_cv_score']:.3f} ± {cv_results['std_cv_score']:.3f}"
        )
        print("=" * 50)

    except Exception as e:
        logger.error("Training failed", error=str(e))
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
