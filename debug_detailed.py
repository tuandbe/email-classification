#!/usr/bin/env python3
"""
Detailed debug script to compare local vs CI environments
"""
import sys
import os
import importlib
from pathlib import Path

print("=== DETAILED ENVIRONMENT DEBUG ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

print("\n=== PYTHON PATH ANALYSIS ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")
    if path and Path(path).exists():
        print(f"    ✅ EXISTS")
        if "app" in os.listdir(path) if Path(path).is_dir() else False:
            print(f"    ✅ Contains 'app' directory")
    else:
        print(f"    ❌ NOT EXISTS")

print("\n=== PROJECT STRUCTURE ===")
project_root = Path.cwd()
print(f"Project root: {project_root}")
print(f"Project root exists: {project_root.exists()}")

# Check if we're in the right directory
if (project_root / "app").exists():
    print("✅ Found 'app' directory in project root")
else:
    print("❌ 'app' directory NOT found in project root")
    print("Contents of project root:")
    for item in project_root.iterdir():
        print(f"  - {item.name}")

print("\n=== APP DIRECTORY ANALYSIS ===")
app_dir = project_root / "app"
if app_dir.exists():
    print(f"App directory: {app_dir}")
    print("App directory contents:")
    for item in app_dir.iterdir():
        print(f"  - {item.name}")
        
    # Check models directory specifically
    models_dir = app_dir / "models"
    if models_dir.exists():
        print(f"✅ Models directory exists: {models_dir}")
        print("Models directory contents:")
        for item in models_dir.iterdir():
            print(f"  - {item.name}")
    else:
        print(f"❌ Models directory NOT found: {models_dir}")

print("\n=== IMPORT TESTING ===")

# Test 1: Basic app import
try:
    print("Test 1: import app")
    import app
    print("✅ app import successful")
    print(f"App module location: {app.__file__}")
    print(f"App module path: {app.__path__}")
except ImportError as e:
    print(f"❌ app import failed: {e}")

# Test 2: Direct file imports
try:
    print("\nTest 2: Direct file imports")
    import importlib.util
    
    # Test app.core.config
    config_path = project_root / "app" / "core" / "config.py"
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        print("✅ Direct config.py import successful")
    else:
        print(f"❌ config.py not found: {config_path}")
    
    # Test app.models.ml_model
    ml_model_path = project_root / "app" / "models" / "ml_model.py"
    if ml_model_path.exists():
        spec = importlib.util.spec_from_file_location("ml_model", ml_model_path)
        ml_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ml_model_module)
        print("✅ Direct ml_model.py import successful")
    else:
        print(f"❌ ml_model.py not found: {ml_model_path}")
        
except Exception as e:
    print(f"❌ Direct file import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Module imports
try:
    print("\nTest 3: Module imports")
    from app.core.config import settings
    print("✅ app.core.config import successful")
except ImportError as e:
    print(f"❌ app.core.config import failed: {e}")

try:
    from app.models.ml_model import MLModelManager
    print("✅ app.models.ml_model import successful")
except ImportError as e:
    print(f"❌ app.models.ml_model import failed: {e}")

try:
    from app.services.preprocessing import TextPreprocessor
    print("✅ app.services.preprocessing import successful")
except ImportError as e:
    print(f"❌ app.services.preprocessing import failed: {e}")

print("\n=== PACKAGE INSTALLATION CHECK ===")
try:
    # Try modern importlib.metadata first (Python 3.8+)
    try:
        import importlib.metadata
        installed_packages = [d.metadata['name'] for d in importlib.metadata.distributions()]
        if "email-interview-classifier" in installed_packages:
            print("✅ email-interview-classifier package is installed")
            # Get package info
            pkg = importlib.metadata.distribution("email-interview-classifier")
            print(f"Package location: {pkg.locate_file('')}")
            print(f"Package version: {pkg.version}")
        else:
            print("❌ email-interview-classifier package is NOT installed")
            print("Installed packages (filtered):")
            for pkg in sorted(installed_packages):
                if any(keyword in pkg.lower() for keyword in ["email", "interview", "classifier", "app"]):
                    print(f"  - {pkg}")
    except ImportError:
        # Fallback to pkg_resources for older Python versions
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        if "email-interview-classifier" in installed_packages:
            print("✅ email-interview-classifier package is installed")
            # Get package info
            pkg = pkg_resources.get_distribution("email-interview-classifier")
            print(f"Package location: {pkg.location}")
            print(f"Package version: {pkg.version}")
        else:
            print("❌ email-interview-classifier package is NOT installed")
            print("Installed packages (filtered):")
            for pkg in sorted(installed_packages):
                if any(keyword in pkg.lower() for keyword in ["email", "interview", "classifier", "app"]):
                    print(f"  - {pkg}")
except Exception as e:
    print(f"❌ Error checking package installation: {e}")
    print("This is expected in Python 3.13+ as pkg_resources is deprecated")

print("\n=== END DETAILED DEBUG ===")
