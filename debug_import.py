#!/usr/bin/env python3
"""
Debug script to test imports in CI environment
"""
import sys
import os
from pathlib import Path

print("=== Python Environment Debug ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

print("\n=== Python Path ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

print("\n=== Directory Structure ===")
project_root = Path.cwd()
print(f"Project root: {project_root}")
print(f"Project root exists: {project_root.exists()}")

print("\n=== App Directory ===")
app_dir = project_root / "app"
print(f"App directory: {app_dir}")
print(f"App directory exists: {app_dir.exists()}")
if app_dir.exists():
    print("App directory contents:")
    for item in app_dir.iterdir():
        print(f"  - {item.name}")

print("\n=== Test Directory ===")
test_dir = project_root / "tests"
print(f"Test directory: {test_dir}")
print(f"Test directory exists: {test_dir.exists()}")
if test_dir.exists():
    print("Test directory contents:")
    for item in test_dir.iterdir():
        print(f"  - {item.name}")

print("\n=== Import Tests ===")
try:
    print("Testing: import app")
    import app
    print("✅ app import successful")
    print(f"App module location: {app.__file__}")
except ImportError as e:
    print(f"❌ app import failed: {e}")

try:
    print("Testing: from app.core.config import settings")
    from app.core.config import settings
    print("✅ app.core.config import successful")
except ImportError as e:
    print(f"❌ app.core.config import failed: {e}")

try:
    print("Testing: from app.models.ml_model import MLModelManager")
    from app.models.ml_model import MLModelManager
    print("✅ app.models.ml_model import successful")
except ImportError as e:
    print(f"❌ app.models.ml_model import failed: {e}")

try:
    print("Testing: from app.services.preprocessing import TextPreprocessor")
    from app.services.preprocessing import TextPreprocessor
    print("✅ app.services.preprocessing import successful")
except ImportError as e:
    print(f"❌ app.services.preprocessing import failed: {e}")

print("\n=== Package Installation Check ===")
try:
    import pkg_resources
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    if "email-interview-classifier" in installed_packages:
        print("✅ email-interview-classifier package is installed")
    else:
        print("❌ email-interview-classifier package is NOT installed")
        print("Installed packages:")
        for pkg in sorted(installed_packages):
            if "email" in pkg.lower() or "interview" in pkg.lower():
                print(f"  - {pkg}")
except Exception as e:
    print(f"❌ Error checking package installation: {e}")

print("\n=== End Debug ===")
