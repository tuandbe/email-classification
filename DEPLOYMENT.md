# Deployment Guide

## Docker Deployment

### Prerequisites
Before building the Docker image, you need to train the model first:

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data
python scripts/download_nltk_data.py

# 3. Train the model
python scripts/train.py data/Interview_vs_Non-Interview_Training_Emails__100_rows_.csv
```

### Build Image
```bash
docker build -t email-classifier .
```

### Run Container
```bash
docker run -p 8000:8000 email-classifier
```

### Using Docker Compose
```bash
docker-compose up --build
```

## Production Deployment

### Prerequisites
- Python 3.13+
- Docker (optional)

### Steps
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download NLTK data: `python scripts/download_nltk_data.py`
4. Train model: `python scripts/train.py data/Interview_vs_Non-Interview_Training_Emails__100_rows_.csv`
5. Start service: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### Environment Variables
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)

### Health Check
The service provides a health check endpoint at `/health` for monitoring.

### API Endpoints
- `POST /v1/predict`: Classify email
- `GET /health`: Health check
- `GET /docs`: API documentation
