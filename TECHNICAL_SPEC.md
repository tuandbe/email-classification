# Technical Specification - Email Interview Classifier

## 1. Data Analysis and Preprocessing

### 1.1 Dataset Characteristics
```
Total samples: 100
Features: 1 (email body text)
Target: Binary classification (Yes/No)
Class distribution: Imbalanced (25:75)
Text length: Variable (50-500 words typically)
```

### 1.2 Text Preprocessing Pipeline
```python
def preprocess_text(text: str) -> str:
    """
    Comprehensive text preprocessing for email classification
    """
    # 1. Basic cleaning
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    # 2. Email-specific cleaning
    text = re.sub(r'\b\d+\b', 'NUMBER', text)  # Replace numbers
    text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', 'EMAIL', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    return text
```

### 1.3 Feature Engineering Strategy

#### TF-IDF Configuration
```python
vectorizer = TfidfVectorizer(
    max_features=5000,           # Limit vocabulary size
    ngram_range=(1, 2),          # Unigrams + bigrams
    min_df=2,                    # Minimum document frequency
    max_df=0.95,                 # Maximum document frequency
    stop_words='english',        # Remove common words
    lowercase=True,              # Already handled in preprocessing
    token_pattern=r'\w+',        # Word tokens only
    use_idf=True,               # Use inverse document frequency
    smooth_idf=True,            # Smooth IDF weights
    sublinear_tf=True           # Apply sublinear scaling
)
```

#### Key Features Expected
- **Interview signals**: schedule, interview, phone, video, call, meet, time, calendar
- **Booking signals**: link, book, slot, availability, confirm, invite
- **Rejection signals**: unfortunately, decided, proceed, review, thank
- **Security signals**: code, verification, password, login, secure

## 2. Machine Learning Model Architecture

### 2.1 Model Selection Rationale

#### Primary: Logistic Regression
```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',    # Handle imbalanced data
    solver='liblinear',         # Good for small datasets
    penalty='l2',               # L2 regularization
    C=1.0                       # Regularization strength
)
```

**Advantages**:
- Fast training and inference
- Interpretable coefficients
- Good with text data
- Built-in class weighting

#### Backup: Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    max_depth=10,               # Prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    bootstrap=True
)
```

### 2.2 Class Imbalance Handling

#### Strategy 1: Class Weights (Preferred)
```python
# Automatic balanced weights
class_weights = {
    'No': len(y) / (2 * (y == 'No').sum()),   # ~0.67
    'Yes': len(y) / (2 * (y == 'Yes').sum())  # ~2.0
}
```

#### Strategy 2: SMOTE (Alternative)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    random_state=42,
    k_neighbors=3,              # Adjust for small dataset
    sampling_strategy='auto'    # Balance to 1:1 ratio
)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 2.3 Model Evaluation Framework

#### Cross-Validation Strategy
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,                 # 5-fold CV
    shuffle=True,
    random_state=42
)

# Preserve class distribution in each fold
for train_idx, val_idx in cv.split(X, y):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
```

#### Evaluation Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, y_prob=None):
    """Comprehensive model evaluation"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Yes')
    recall = recall_score(y_true, y_pred, pos_label='Yes')
    f1 = f1_score(y_true, y_pred, pos_label='Yes')
    
    # Class-specific metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['No', 'Yes'])
    
    # AUC-ROC if probabilities available
    if y_prob is not None:
        auc = roc_auc_score(y_true == 'Yes', y_prob[:, 1])
    
    return {
        'accuracy': accuracy,
        'precision_interview': precision,
        'recall_interview': recall,
        'f1_interview': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'auc_roc': auc if y_prob is not None else None
    }
```

## 3. Training Pipeline Implementation

### 3.1 Training Script Architecture
```python
# train.py
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to training CSV file')
    parser.add_argument('--model-dir', default='models', help='Output directory for models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data_path)
    
    # Preprocess
    X = df['Body'].apply(preprocess_text)
    y = df['is_interview']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, 
        random_state=args.random_state,
        stratify=y
    )
    
    # Vectorize
    vectorizer = TfidfVectorizer(...)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(...)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    metrics = evaluate_model(y_test, y_pred)
    
    # Save models
    save_model(model, vectorizer, args.model_dir, metrics)
```

### 3.2 Model Persistence
```python
def save_model(model, vectorizer, model_dir, metrics):
    """Save trained model and associated artifacts"""
    
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, model_dir / 'classifier.pkl')
    joblib.dump(vectorizer, model_dir / 'vectorizer.pkl')
    
    # Save metadata
    metadata = {
        'model_type': type(model).__name__,
        'vectorizer_type': type(vectorizer).__name__,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'feature_count': vectorizer.get_feature_names_out().shape[0]
    }
    
    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_dir}")
    print(f"Test Accuracy: {metrics['accuracy']:.3f}")
    print(f"Interview Recall: {metrics['recall_interview']:.3f}")
```

## 4. Web Service Implementation

### 4.1 FastAPI Application Structure
```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class PredictionRequest(BaseModel):
    email_body: str
    
    class Config:
        schema_extra = {
            "example": {
                "email_body": "We'd love to schedule a 30-minute video conversation..."
            }
        }

class PredictionResponse(BaseModel):
    is_interview: str  # "yes" or "no"
    confidence: float = None  # Optional confidence score
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: str

# Global variables for loaded model
model = None
vectorizer = None
start_time = None

app = FastAPI(
    title="Email Interview Classifier",
    description="API for email interview classification",
    version="1.0.0"
)
```

### 4.2 Model Loading and Startup
```python
@app.on_event("startup")
async def load_model():
    """Load trained model on startup"""
    global model, vectorizer, start_time
    
    start_time = datetime.now()
    model_dir = Path("models")
    
    try:
        model = joblib.load(model_dir / "classifier.pkl")
        vectorizer = joblib.load(model_dir / "vectorizer.pkl")
        
        logger.info("Model loaded successfully")
        
        # Load metadata for logging
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            logger.info(f"Model type: {metadata['model_type']}")
            logger.info(f"Training date: {metadata['training_date']}")
            
    except FileNotFoundError:
        logger.error("Model files not found. Please run training first.")
        model = None
        vectorizer = None
```

### 4.3 Prediction Endpoint
```python
@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_interview(request: PredictionRequest):
    """Predict if email is interview-related"""
    
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please run training first."
        )
    
    try:
        # Preprocess input
        processed_text = preprocess_text(request.email_body)
        
        if not processed_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Email body cannot be empty after preprocessing"
            )
        
        # Vectorize
        text_vector = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # Convert to required format
        is_interview = "yes" if prediction == "Yes" else "no"
        confidence = float(max(probabilities))
        
        # Log prediction
        logger.info(f"Prediction: {is_interview}, Confidence: {confidence:.3f}")
        
        return PredictionResponse(
            is_interview=is_interview,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
```

### 4.4 Health Check and Monitoring
```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    uptime = datetime.now() - start_time if start_time else timedelta(0)
    uptime_str = str(uptime).split('.')[0]  # Remove microseconds
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        uptime=uptime_str
    )

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "Email Interview Classifier",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
```

## 5. Error Handling and Validation

### 5.1 Input Validation
```python
from pydantic import validator

class PredictionRequest(BaseModel):
    email_body: str
    
    @validator('email_body')
    def validate_email_body(cls, v):
        if not v or not v.strip():
            raise ValueError('Email body cannot be empty')
        
        if len(v) > 10000:  # Reasonable limit
            raise ValueError('Email body too long (max 10000 characters)')
            
        return v.strip()
```

### 5.2 Exception Handling
```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "code": 400}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "code": 500}
    )
```

## 6. Testing Strategy

### 6.1 Unit Tests Structure
```python
# tests/test_model.py
import pytest
from app.services.preprocessing import PreprocessingService
from app.models.ml_model import EmailClassifier

class TestPreprocessing:
    def test_basic_cleaning(self):
        text = "HELLO! This is a TEST email."
        preprocessor = PreprocessingService(Settings())
        result = preprocessor.preprocess_text(text)
        assert result == "hello this is a test email"
    
    def test_number_replacement(self):
        text = "Call me at 123-456-7890"
        preprocessor = PreprocessingService(Settings())
        result = preprocessor.preprocess_text(text)
        assert "NUMBER" in result
        assert "123" not in result

class TestModel:
    @pytest.fixture
    def trained_model(self):
        # Setup trained model for testing
        pass
    
    def test_prediction_format(self, trained_model):
        # Test prediction output format
        pass
```

### 6.2 API Integration Tests
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestAPI:
    def test_predict_interview_email(self):
        response = client.post(
            "/v1/predict",
            json={"email_body": "We'd like to schedule an interview"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_interview"] in ["yes", "no"]
        assert "confidence" in data
    
    def test_predict_empty_body(self):
        response = client.post(
            "/v1/predict",
            json={"email_body": ""}
        )
        assert response.status_code == 400
    
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
```

## 7. Performance Optimization

### 7.1 Model Optimization
```python
# Optimize TF-IDF for speed
vectorizer = TfidfVectorizer(
    max_features=3000,           # Reduce feature count
    ngram_range=(1, 2),          # Keep bigrams for context
    min_df=3,                    # Higher min_df for speed
    binary=True,                 # Binary features for speed
    use_idf=True,
    sublinear_tf=False           # Disable for speed
)
```

### 7.2 Caching Strategy
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_preprocess(text: str) -> str:
    """Cache preprocessing results for repeated requests"""
    return preprocess_text(text)

@lru_cache(maxsize=500)
def cached_predict(text: str) -> tuple:
    """Cache predictions for identical inputs"""
    processed = cached_preprocess(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0].max()
    return prediction, probability
```

## 8. Deployment Considerations

### 8.1 Production Configuration
```python
# Production settings
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,              # Multiple workers
        loop="uvloop",          # Faster event loop
        http="httptools",       # Faster HTTP parser
        access_log=True,
        log_level="info"
    )
```

### 8.2 Docker Configuration
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

This is a detailed technical specification for implementation. This document provides sufficient information to start coding and implement each component systematically.
