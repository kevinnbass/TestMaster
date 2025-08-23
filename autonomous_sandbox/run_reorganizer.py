#!/usr/bin/env python3
"""
Codebase Reorganizer Launcher
============================

Easy-to-use launcher script for the Codebase Reorganization Tool.
This script automatically finds the TestMaster root directory and runs the reorganizer.

Usage:
    python run_reorganizer.py [options]

Examples:
    python run_reorganizer.py --preview
    python run_reorganizer.py --symlinks
    python run_reorganizer.py --help
"""

import os
import sys
import argparse
from pathlib import Path

def find_testmaster_root() -> Path:
    """Find the TestMaster root directory by looking for key indicators"""
    current_dir = Path.cwd().resolve()

    # Look for common TestMaster indicators
    indicators = [
        'TestMaster', 'PRODUCTION_PACKAGES', 'codebase_reorganizer.py',
        'AGENT_D_HOUR_8-10_PREDICTIVE_INTELLIGENCE_BREAKTHROUGH.md'
    ]

    # Check current directory with bounded loop
    MAX_INDICATORS = 50  # Safety bound for indicators
    for i in range(min(len(indicators), MAX_INDICATORS)):
        indicator = indicators[i]
        if (current_dir / indicator).exists():
            return current_dir

    # Check parent directories with bounded loop
    MAX_PARENT_DIRS = 20  # Safety bound for parent directory checking
    parent_list = list(current_dir.parents)
    for i in range(min(len(parent_list), MAX_PARENT_DIRS)):
        parent = parent_list[i]
        # Bounded loop for indicator checking
        for j in range(min(len(indicators), MAX_INDICATORS)):
            indicator = indicators[j]
            if (parent / indicator).exists():
                return parent

    # Fallback: assume we're in the tools directory
    if 'tools' in str(current_dir) and 'codebase_reorganizer' in str(current_dir):
        return current_dir.parent.parent

    return current_dir

def main() -> None:
    """Main launcher function that finds TestMaster root and executes reorganizer"""
    # Find TestMaster root
    testmaster_root = find_testmaster_root()

    print("Codebase Reorganizer Launcher")
    print("=" * 40)
    print(f"Detected TestMaster root: {testmaster_root}")

    # Verify reorganizer exists
    reorganizer_path = testmaster_root / "tools" / "codebase_reorganizer" / "reorganizer.py"
    if not reorganizer_path.exists():
        print(f"ERROR: Reorganizer not found at {reorganizer_path}")
        print("Please ensure the codebase_reorganizer is properly installed.")
        sys.exit(1)

    # Build command with pre-allocation (Rule 3 compliance)
    MAX_ARGS = 100  # Safety bound for command arguments
    cmd_args = [None] * MAX_ARGS
    cmd_args[0] = sys.executable
    cmd_args[1] = str(reorganizer_path)
    arg_count = 2

    # Add command line arguments with bounded loop
    for i in range(min(len(sys.argv[1:]), MAX_ARGS - arg_count)):
        if arg_count < MAX_ARGS:
            cmd_args[arg_count] = sys.argv[1 + i]
            arg_count += 1

    # Add root directory if not specified
    if '--root' not in sys.argv and arg_count < MAX_ARGS - 2:
        cmd_args[arg_count] = '--root'
        cmd_args[arg_count + 1] = str(testmaster_root)
        arg_count += 2

    # Trim to actual size
    cmd_args = cmd_args[:arg_count]

    print(f"Running: {' '.join(cmd_args)}")
    print("-" * 40)

    # Change to reorganizer directory to ensure proper imports
    os.chdir(reorganizer_path.parent)

    # Execute the reorganizer
    os.execv(sys.executable, cmd_args)

if __name__ == "__main__":
    main()

