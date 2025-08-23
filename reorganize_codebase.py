#!/usr/bin/env python3
"""
TestMaster Codebase Reorganizer
===============================

Simple launcher script for the Codebase Reorganization Tool.
This provides easy access to the reorganizer from the project root.

Usage:
    python reorganize_codebase.py [options]

Examples:
    python reorganize_codebase.py --preview
    python reorganize_codebase.py --symlinks
    python reorganize_codebase.py --help

The tool will:
1. Analyze all Python files in your codebase
2. Exclude research repos and system files
3. Categorize files by functionality
4. Create organized structure with symlinks
5. Update imports automatically
6. Create backups and logs
"""

import sys
import os
from pathlib import Path

def main():
    # Find the reorganizer script
    current_dir = Path(__file__).parent
    reorganizer_path = current_dir / "tools" / "codebase_reorganizer" / "reorganizer.py"

    if not reorganizer_path.exists():
        print("ERROR: Codebase reorganizer not found!")
        print("Expected location:", reorganizer_path)
        print("\nPlease ensure the reorganizer is properly installed in:")
        print("tools/codebase_reorganizer/reorganizer.py")
        sys.exit(1)

    # Build command
    cmd_args = [sys.executable, str(reorganizer_path)]
    cmd_args.extend(sys.argv[1:])  # Pass through all arguments

    # Add root directory if not specified
    if '--root' not in sys.argv:
        cmd_args.extend(['--root', str(current_dir)])

    print("🚀 Starting Codebase Reorganization...")
    print("=" * 50)

    # Execute the reorganizer
    os.chdir(reorganizer_path.parent)
    os.execv(sys.executable, cmd_args)

if __name__ == "__main__":
    main()

