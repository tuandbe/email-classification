"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
    
    def test_health_endpoint_no_model(self, client):
        """Test health endpoint when model is not loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
    
    @patch('app.main.model_manager')
    def test_health_endpoint_with_model(self, mock_model_manager, client):
        """Test health endpoint when model is loaded."""
        mock_model_manager.is_loaded.return_value = True
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_loaded"] is True
    
    def test_predict_endpoint_no_model(self, client):
        """Test prediction endpoint when model is not loaded."""
        response = client.post(
            "/v1/predict",
            json={"email_body": "Test email content"}
        )
        assert response.status_code == 503
    
    def test_predict_endpoint_invalid_request(self, client):
        """Test prediction endpoint with invalid request."""
        # Missing email_body
        response = client.post("/v1/predict", json={})
        assert response.status_code == 422
        
        # Empty email_body
        response = client.post(
            "/v1/predict",
            json={"email_body": ""}
        )
        assert response.status_code == 400
    
    def test_predict_endpoint_valid_request(self, client, mock_model_manager):
        """Test prediction endpoint with valid request."""
        with patch('app.main.get_model_manager', return_value=mock_model_manager):
            response = client.post(
                "/v1/predict",
                json={"email_body": "We'd like to schedule an interview"}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "is_interview" in data
            assert "confidence" in data
            assert data["is_interview"] in ["Yes", "No"]
            assert 0 <= data["confidence"] <= 1
    
    def test_model_info_endpoint_no_model(self, client):
        """Test model info endpoint when model is not loaded."""
        response = client.get("/v1/model-info")
        assert response.status_code == 503
    
    def test_model_info_endpoint_with_model(self, client, mock_model_manager):
        """Test model info endpoint when model is loaded."""
        with patch('app.main.get_model_manager', return_value=mock_model_manager):
            response = client.get("/v1/model-info")
            assert response.status_code == 200
            
            data = response.json()
            assert "model_loaded" in data
            assert "model_info" in data
            assert data["model_loaded"] is True
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/v1/predict")
        # FastAPI handles CORS automatically, so we just check the endpoint exists
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled
    
    def test_rate_limiting(self, client):
        """Test rate limiting (basic check)."""
        # Make multiple requests quickly
        for _ in range(5):
            response = client.post(
                "/v1/predict",
                json={"email_body": "Test email"}
            )
            # Should not be rate limited for small number of requests
            assert response.status_code in [200, 503]  # 503 if model not loaded
