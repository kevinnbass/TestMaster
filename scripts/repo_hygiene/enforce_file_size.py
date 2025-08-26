#!/usr/bin/env python3
"""
File size enforcement script - Python version for cross-platform compatibility
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Enforce file size limits on staged files')
    parser.add_argument('--max-bytes', type=int, default=5242880, help='Maximum file size in bytes (default: 5MB)')
    
    args = parser.parse_args()
    max_bytes = args.max_bytes
    
    try:
        # Get staged files
        result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                              capture_output=True, text=True, check=True)
        staged_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        violations = []
        
        for file_path in staged_files:
            if file_path and os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if size > max_bytes:
                    violations.append((file_path, size))
        
        if violations:
            print("ERROR: Files exceed maximum size limit:")
            for file_path, size in violations:
                size_mb = size / (1024 * 1024)
                max_mb = max_bytes / (1024 * 1024)
                print(f"  {file_path}: {size_mb:.2f}MB (max: {max_mb:.2f}MB)")
            print("\nCommit aborted. Please reduce file sizes or move large files to artifacts directory.")
            return 2
        
        print("All staged files are within size limits.")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        return 1
    except Exception as e:
        print(f"Error checking file sizes: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())