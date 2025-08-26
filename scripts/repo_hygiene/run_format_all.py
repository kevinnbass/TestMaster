#!/usr/bin/env python3
"""
Code formatting runner - Python version for cross-platform compatibility
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n=== {description} ===")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        if result.returncode != 0:
            print(f"Warning: {description} returned exit code {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run code formatting tools')
    parser.add_argument('--python-only', action='store_true', help='Only format Python files')
    parser.add_argument('--install-deps', action='store_true', help='Install formatting dependencies')
    
    args = parser.parse_args()
    
    # Check if we're in a virtual environment or can use pip
    if args.install_deps:
        print("Installing Python formatting dependencies...")
        run_command("python -m pip install --upgrade pip", "Upgrading pip")
        run_command("python -m pip install black isort ruff", "Installing Python formatters")
    
    # Python formatting
    success = True
    success &= run_command("python -m black .", "Running Black (Python formatter)")
    success &= run_command("python -m isort .", "Running isort (Python import sorter)")
    success &= run_command("python -m ruff check . --fix", "Running ruff (Python linter with auto-fix)")
    
    # JavaScript/TypeScript formatting (if requested and tools available)
    if not args.python_only:
        # Check if npm/node tools are available
        try:
            subprocess.run(['npm', '--version'], capture_output=True, check=True)
            print("\n=== JavaScript/TypeScript Tools ===")
            run_command("npx prettier --write .", "Running Prettier")
            run_command("npx eslint . --fix", "Running ESLint with auto-fix")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\nSkipping JavaScript/TypeScript formatting (npm tools not available)")
    
    if success:
        print("\n✅ Formatting completed successfully!")
        return 0
    else:
        print("\n⚠️  Formatting completed with some warnings.")
        return 1


if __name__ == "__main__":
    sys.exit(main())