#!/usr/bin/env python3
"""
Script to test CI pipeline locally.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print("✅ SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("STDERR:", e.stderr)
        return False


def main():
    """Run CI pipeline steps locally."""
    print("🚀 Testing CI Pipeline Locally")

    # Change to project root
    project_root = Path(__file__).parent.parent
    import os

    os.chdir(project_root)

    steps = [
        ("black --check --diff app/ scripts/ tests/", "Code Formatting Check"),
        (
            "flake8 app/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503,E501,F401,E402,F841,F541,W293",
            "Linting Check",
        ),
        ("python scripts/download_nltk_data.py", "Download NLTK Data"),
        ("pytest tests/ -v", "Run Tests"),
        (
            "python scripts/train.py data/Interview_vs_Non-Interview_Training_Emails__100_rows_.csv",
            "Train Model",
        ),
    ]

    failed_steps = []

    for i, (command, description) in enumerate(steps, 1):
        print(f"\n📋 Step {i}/{len(steps)}: {description}")

        if not run_command(command, description):
            failed_steps.append((i, description))
            break

    # Summary
    print(f"\n{'='*50}")
    print("🏁 CI Test Summary")
    print(f"{'='*50}")

    if not failed_steps:
        print("✅ All steps passed! Ready for PR.")
        return 0
    else:
        print(f"❌ {len(failed_steps)} step(s) failed:")
        for step_num, description in failed_steps:
            print(f"  - Step {step_num}: {description}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
