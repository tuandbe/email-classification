# Project Structure and Implementation Plan

## Complete Directory Structure (FastAPI Best Practices)

```
decision-making/
├── README.md                               # Main documentation
├── TECHNICAL_SPEC.md                       # Technical specification
├── PROJECT_STRUCTURE.md                    # This file
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore file
├── .env.example                           # Environment variables template
│
├── data/                                  # Training data
│   └── Interview_vs_Non-Interview_Training_Emails__100_rows_.csv
│
├── models/                                # Trained models (generated)
│   ├── classifier.pkl                     # Trained classifier
│   ├── vectorizer.pkl                     # TF-IDF vectorizer
│   └── metadata.json                      # Model metadata
│
├── app/                                   # Main application code (FastAPI best practices)
│   ├── __init__.py
│   ├── main.py                            # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   └── predictions.py             # Prediction endpoints
│   │   └── dependencies.py                # Shared dependencies
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                      # Configuration settings
│   │   └── security.py                    # Security utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── ml_model.py                    # ML model class
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── prediction.py                  # Request/Response schemas
│   │   └── health.py                      # Health check schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── preprocessing.py               # Text preprocessing
│   │   └── prediction_service.py          # Business logic
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                     # Utility functions
│
├── scripts/                               # Utility scripts
│   ├── train.py                          # Main training script
│   ├── evaluate.py                       # Model evaluation script
│   └── test_api.py                       # API testing script
│
├── tests/                                 # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py                       # Pytest configuration
│   ├── test_preprocessing.py             # Preprocessing tests
│   ├── test_model.py                     # Model tests
│   ├── test_api.py                       # API tests
│   └── test_services/                    # Service layer tests
│       ├── __init__.py
│       └── test_prediction_service.py
│
├── alembic/                               # Database migrations (if needed)
│   ├── versions/
│   └── env.py
│
├── logs/                                  # Application logs
│   ├── app.log                           # General application logs
│   └── error.log                         # Error logs
│
├── Dockerfile                             # Docker configuration
├── docker-compose.yml                     # Docker compose for development
└── .github/                              # GitHub workflows (optional)
    └── workflows/
        └── ci.yml                        # CI/CD pipeline
```

## Implementation Order and Priority

### Phase 1: Core Functionality (MVP)
**Timeline: 1-2 days**

1. **Text Preprocessing Module** (`app/services/preprocessing.py`)
   - Implement `preprocess_text()` function
   - Handle basic text cleaning
   - Test with sample data

2. **Model Training Script** (`scripts/train.py`)
   - Load CSV data
   - Train TF-IDF + Logistic Regression
   - Save model artifacts
   - Basic evaluation metrics

3. **FastAPI Application** (`app/main.py`)
   - Basic POST /v1/predict endpoint
   - Model loading on startup
   - Simple prediction logic

4. **Testing**
   - Manual testing with curl commands
   - Verify end-to-end functionality

### Phase 2: Production Ready (Enhancements)
**Timeline: 1 day**

1. **Enhanced Model Training** (`app/models/ml_model.py`)
   - Cross-validation
   - Class imbalance handling
   - Multiple algorithm comparison
   - Comprehensive metrics

2. **API Improvements** (`app/api/endpoints/predictions.py`)
   - Error handling
   - Input validation
   - Health check endpoint
   - Logging

3. **Testing Suite** (`tests/`)
   - Unit tests for all modules
   - API integration tests
   - Edge case testing

### Phase 3: Production Deployment (Optional)
**Timeline: 0.5 day**

1. **Docker Support**
   - Dockerfile
   - Docker compose
   - Health checks

2. **Documentation**
   - API documentation
   - Deployment guide
   - Usage examples

## File-by-File Implementation Guide

### 1. `app/services/preprocessing.py`
```python
"""
Text preprocessing utilities for email classification
"""
import re
from typing import Dict, Any
from app.core.config import Settings

class PreprocessingService:
    """Text preprocessing service"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess email text for classification
        
        Args:
            text: Raw email body text
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        return text.strip()
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract additional features from email text
        
        Args:
            text: Preprocessed email text
            
        Returns:
            Dictionary of extracted features
        """
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "has_interview_keywords": self._has_interview_keywords(text),
            "has_schedule_keywords": self._has_schedule_keywords(text)
        }
    
    def _has_interview_keywords(self, text: str) -> bool:
        """Check for interview-related keywords"""
        keywords = ["interview", "meeting", "discuss", "opportunity"]
        return any(keyword in text for keyword in keywords)
    
    def _has_schedule_keywords(self, text: str) -> bool:
        """Check for scheduling-related keywords"""
        keywords = ["schedule", "time", "date", "available", "calendar"]
        return any(keyword in text for keyword in keywords)
```

### 2. `app/models/ml_model.py`
```python
"""
ML model classes for email classification
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from pathlib import Path

class BaseMLService(ABC):
    """Base class for ML services"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the trained model"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Any) -> Dict[str, Any]:
        """Make prediction"""
        pass
    
    def _validate_model_loaded(self) -> None:
        """Validate that model is loaded"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

class EmailClassifier(BaseMLService):
    """Email interview classification model"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
    
    async def load_model(self) -> None:
        """Load the trained model and vectorizer"""
        try:
            self.model = joblib.load(self.model_path / "classifier.pkl")
            self.vectorizer = joblib.load(self.model_path / "vectorizer.pkl")
            self.is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    async def predict(self, email_text: str) -> Dict[str, Any]:
        """
        Predict if email is interview-related
        
        Args:
            email_text: Raw email body text
            
        Returns:
            Dictionary with prediction results
        """
        self._validate_model_loaded()
        
        # Preprocess text
        processed_text = self._preprocess_text(email_text)
        
        if not processed_text:
            return {
                "is_interview": "No",
                "confidence": 0.0,
                "error": "Empty or invalid input text"
            }
        
        # Vectorize text
        text_vector = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vector)[0]
        confidence = self.model.predict_proba(text_vector)[0].max()
        
        return {
            "is_interview": "Yes" if prediction == 1 else "No",
            "confidence": float(confidence),
            "processed_text": processed_text
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text or not isinstance(text, str):
            return ""
        return text.lower().strip()
```

### 3. `app/schemas/prediction.py`
```python
"""
Pydantic models for prediction API
"""
from pydantic import BaseModel, Field, validator
from typing import Optional

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    email_body: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Email body text to analyze"
    )
    
    @validator('email_body')
    def validate_email_body(cls, v):
        """Validate email body content"""
        if not v or not isinstance(v, str):
            raise ValueError('Email body must be a non-empty string')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "email_body": "We would like to schedule an interview with you"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    is_interview: str = Field(..., description="Prediction result: Yes or No")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "is_interview": "Yes",
                "confidence": 0.85
            }
        }
```

### 4. `app/api/endpoints/predictions.py`
```python
"""
Prediction API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.api.dependencies import get_prediction_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["predictions"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_interview(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Predict if email is interview-related
    
    Args:
        request: Email content to analyze
        prediction_service: Injected prediction service
        
    Returns:
        Prediction result with confidence score
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Make prediction
        result = await prediction_service.predict(request.email_body)
        
        return PredictionResponse(
            is_interview=result["is_interview"],
            confidence=result["confidence"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction service error. Please try again."
        )
```

### 5. `app/api/dependencies.py`
```python
"""
Shared dependencies for API endpoints
"""
from functools import lru_cache
from app.core.config import Settings
from app.services.prediction_service import PredictionService

@lru_cache()
def get_settings() -> Settings:
    """Get application settings"""
    return Settings()

async def get_prediction_service(
    settings: Settings = Depends(get_settings)
) -> PredictionService:
    """Get prediction service instance"""
    service = PredictionService(settings)
    if not service.is_loaded:
        await service.load_model()
    return service
```

### 6. `app/services/prediction_service.py`
```python
"""
Prediction service with business logic
"""
import logging
from typing import Dict, Any
from app.models.ml_model import EmailClassifier
from app.services.preprocessing import PreprocessingService
from app.core.config import Settings

logger = logging.getLogger(__name__)

class PredictionService(EmailClassifier):
    """Email classification prediction service"""
    
    def __init__(self, settings: Settings):
        super().__init__(settings.model_path)
        self.settings = settings
        self.preprocessor = PreprocessingService(settings)
    
    async def predict(self, email_text: str) -> Dict[str, Any]:
        """
        Predict if email is interview-related with comprehensive error handling
        
        Args:
            email_text: Raw email body text
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Validate input
            if not email_text or not isinstance(email_text, str):
                raise ValueError("Email text must be a non-empty string")
            
            # Check model status
            if not self.is_loaded:
                await self.load_model()
            
            # Log prediction attempt
            logger.info(f"Making prediction for email of length: {len(email_text)}")
            
            # Preprocess text
            processed_text = self.preprocessor.preprocess_text(email_text)
            
            if not processed_text:
                logger.warning("Empty text after preprocessing")
                return {
                    "is_interview": "No",
                    "confidence": 0.0,
                    "error": "Empty text after preprocessing"
                }
            
            # Make prediction
            result = await self._make_prediction(processed_text)
            
            # Log successful prediction
            logger.info(f"Prediction completed: {result['is_interview']} (confidence: {result['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    async def _make_prediction(self, processed_text: str) -> Dict[str, Any]:
        """Internal method to make prediction with error handling"""
        try:
            # Vectorize text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(text_vector)[0]
            confidence = self.model.predict_proba(text_vector)[0].max()
            
            return {
                "is_interview": "Yes" if prediction == 1 else "No",
                "confidence": float(confidence),
                "processed_text": processed_text
            }
            
        except Exception as e:
            logger.error(f"Error in prediction model: {e}", exc_info=True)
            raise RuntimeError(f"Model prediction failed: {str(e)}")
```

### 7. `scripts/train.py`
```python
"""
Main training script
"""
import argparse
import pandas as pd
from pathlib import Path
from app.models.ml_model import EmailClassifier
from app.services.preprocessing import PreprocessingService
from app.core.config import Settings

def main():
    parser = argparse.ArgumentParser(description='Train email classifier')
    parser.add_argument('data_path', help='Path to CSV training data')
    parser.add_argument('--output-dir', default='models', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Initialize services
    settings = Settings()
    preprocessor = PreprocessingService(settings)
    
    # Preprocess
    print("Preprocessing text...")
    X = df['Body'].apply(preprocessor.preprocess_text)
    y = df['is_interview']
    
    # Train model
    print("Training model...")
    classifier = EmailClassifier(args.output_dir)
    metrics = classifier.train(X, y)
    
    # Save model
    print(f"Saving model to {args.output_dir}")
    classifier.save(args.output_dir)
    
    # Print results
    print(f"Training completed!")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
    print(f"Interview Recall: {metrics.get('recall_interview', 'N/A'):.3f}")

if __name__ == "__main__":
    main()
```

### 8. `app/main.py`
```python
"""
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import predictions
from app.core.config import Settings
from app.utils.logging import setup_logging
import logging

# Initialize settings
settings = Settings()

# Setup logging
setup_logging(settings)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description="API để phân loại email interview",
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router)

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Starting Email Interview Classifier API")
    logger.info(f"API Version: {settings.api_version}")
    logger.info(f"Debug Mode: {settings.debug}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down Email Interview Classifier API")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Email Interview Classifier",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Email Interview Classifier",
        "version": settings.api_version
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
```

### 9. `app/core/config.py`
```python
"""
Application configuration settings
"""
from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    api_title: str = "Email Interview Classifier"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Model settings
    model_path: str = "models"
    model_accuracy_threshold: float = 0.7
    
    # Logging
    log_level: str = "INFO"
    
    # Security (if needed)
    secret_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

### 10. `app/utils/logging.py`
```python
"""
Logging configuration utilities
"""
import logging
import sys
from pathlib import Path
from app.core.config import Settings

def setup_logging(settings: Settings):
    """Configure application logging"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log"),
            logging.FileHandler(log_dir / "error.log", level=logging.ERROR)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
```

## Development Workflow

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Development Cycle
```bash
# 1. Implement preprocessing
# Edit app/services/preprocessing.py
python -c "from app.services.preprocessing import PreprocessingService; from app.core.config import Settings; p = PreprocessingService(Settings()); print(p.preprocess_text('Test email'))"

# 2. Implement model training
# Edit app/models/ml_model.py and scripts/train.py
python scripts/train.py data/Interview_vs_Non-Interview_Training_Emails__100_rows_.csv

# 3. Implement API
# Edit app/api/endpoints/predictions.py and app/main.py
uvicorn app.main:app --reload

# 4. Test API
curl -X POST "http://localhost:8000/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"email_body": "We would like to schedule an interview"}'
```

### 3. Testing
```bash
# Run unit tests
pytest tests/ -v

# Test specific module
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### 4. Production Deployment
```bash
# Build Docker image
docker build -t email-classifier .

# Run container
docker run -p 8000:8000 email-classifier

# Or use docker-compose
docker-compose up
```

## Quality Checklist

### Before Commit
- [ ] All tests passing
- [ ] Code follows PEP 8 style
- [ ] Functions have docstrings
- [ ] Error handling implemented
- [ ] Logging added for important operations

### Before Production
- [ ] Model achieves target accuracy (>85%)
- [ ] API handles all edge cases
- [ ] Health check endpoint working
- [ ] Docker image builds successfully
- [ ] Documentation updated

### Performance Targets
- [ ] Training completes in <30 seconds
- [ ] API response time <100ms
- [ ] Memory usage <100MB
- [ ] Model accuracy >85%
- [ ] Interview recall >90%

## Common Issues và Solutions

### 1. Import Errors
```bash
# Add app to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports
from app.services.preprocessing import PreprocessingService
```

### 2. Model Not Found Error
```bash
# Ensure models directory exists and contains files
ls -la models/
# Should see: classifier.pkl, vectorizer.pkl, metadata.json
```

### 3. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Use different port
uvicorn app.main:app --port 8001
```

### 4. Memory Issues
```bash
# Reduce TF-IDF features
max_features=3000  # instead of 5000

# Use binary features
binary=True

# Limit model complexity
max_depth=10  # for Random Forest
```

---

This is a complete implementation guide. You can follow each phase and file systematically to build a complete service.
