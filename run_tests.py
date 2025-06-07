#!/usr/bin/env python3
"""
Test runner script for CharMem RAG tests.
"""
import subprocess
import sys
import os


def run_tests():
    """Run the test suite."""
    print("🧪 Running CharMem RAG Tests...")
    print("=" * 50)

    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    # Run pytest with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--cov=backend",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode


def run_specific_test(test_name):
    """Run a specific test or test class."""
    cmd = [sys.executable, "-m", "pytest", f"tests/test_rag.py::{test_name}", "-v"]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        exit_code = run_specific_test(test_name)
    else:
        # Run all tests
        exit_code = run_tests()

    sys.exit(exit_code)
