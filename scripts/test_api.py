#!/usr/bin/env python3
"""
API testing script for email interview classification service.

Usage:
    python scripts/test_api.py
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings


async def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://localhost:{settings.port}/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False


async def test_prediction_endpoint():
    """Test the prediction endpoint."""
    print("\nTesting prediction endpoint...")

    # Test cases
    test_cases = [
        {
            "name": "Interview email",
            "email_body": "We'd love to schedule a 30-minute video conversation to learn more about your experience and discuss the role in detail.",
        },
        {
            "name": "Non-interview email",
            "email_body": "Thank you for your application. We have received your resume and will review it carefully.",
        },
        {
            "name": "Security code email",
            "email_body": "Your verification code is 123456. Please enter this code to verify your account.",
        },
        {
            "name": "Rejection email",
            "email_body": "Thank you for your interest in our company. Unfortunately, we have decided not to proceed with your application at this time.",
        },
    ]

    async with httpx.AsyncClient() as client:
        for test_case in test_cases:
            print(f"\n--- {test_case['name']} ---")

            try:
                response = await client.post(
                    f"http://localhost:{settings.port}/v1/predict",
                    json={"email_body": test_case["email_body"]},
                    timeout=30.0,
                )

                print(f"Status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    print(f"Prediction: {result['is_interview']}")
                    print(f"Confidence: {result['confidence']:.3f}")
                else:
                    print(f"Error: {response.text}")

            except Exception as e:
                print(f"Request failed: {e}")


async def test_model_info_endpoint():
    """Test the model info endpoint."""
    print("\nTesting model info endpoint...")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"http://localhost:{settings.port}/v1/model-info"
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"Model info request failed: {e}")
            return False


async def test_invalid_requests():
    """Test invalid requests."""
    print("\nTesting invalid requests...")

    async with httpx.AsyncClient() as client:
        # Test empty email body
        print("\n--- Empty email body ---")
        try:
            response = await client.post(
                f"http://localhost:{settings.port}/v1/predict", json={"email_body": ""}
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

        # Test missing email_body field
        print("\n--- Missing email_body field ---")
        try:
            response = await client.post(
                f"http://localhost:{settings.port}/v1/predict",
                json={"text": "Some text"},
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

        # Test very long email
        print("\n--- Very long email ---")
        long_email = "This is a very long email. " * 1000  # ~30,000 characters
        try:
            response = await client.post(
                f"http://localhost:{settings.port}/v1/predict",
                json={"email_body": long_email},
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")


async def main():
    """Main testing function."""
    print("Email Interview Classification API Test")
    print("=" * 50)

    # Test health endpoint
    health_ok = await test_health_endpoint()

    if not health_ok:
        print("\nHealth check failed. Make sure the server is running.")
        print(
            f"Start the server with: python -m uvicorn app.main:app --host {settings.host} --port {settings.port}"
        )
        return

    # Test model info endpoint
    await test_model_info_endpoint()

    # Test prediction endpoint
    await test_prediction_endpoint()

    # Test invalid requests
    await test_invalid_requests()

    print("\n" + "=" * 50)
    print("API testing completed!")


if __name__ == "__main__":
    asyncio.run(main())
