#!/usr/bin/env python3
"""
Test script to isolate the models import issue
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=== Testing Models Import Isolation ===")

try:
    print("1. Testing app import...")
    import app
    print("✅ app import successful")
except Exception as e:
    print(f"❌ app import failed: {e}")

try:
    print("2. Testing app.models import...")
    import app.models
    print("✅ app.models import successful")
except Exception as e:
    print(f"❌ app.models import failed: {e}")

try:
    print("3. Testing app.models.ml_model import...")
    from app.models import ml_model
    print("✅ app.models.ml_model import successful")
except Exception as e:
    print(f"❌ app.models.ml_model import failed: {e}")

try:
    print("4. Testing MLModelManager class import...")
    from app.models.ml_model import MLModelManager
    print("✅ MLModelManager import successful")
except Exception as e:
    print(f"❌ MLModelManager import failed: {e}")
    import traceback
    traceback.print_exc()

print("=== End Test ===")
