#!/usr/bin/env python3
"""
Simple test runner that shows progress as tests run.
"""

import subprocess
import sys
from pathlib import Path
import time

def main():
    print("="*70)
    print("RUNNING INTELLIGENT TEST SUITE")
    print("="*70)
    
    # Find all intelligent test files
    test_dir = Path("tests/unit")
    intelligent_tests = sorted(test_dir.glob("*_intelligent.py"))
    
    print(f"\nFound {len(intelligent_tests)} intelligent test files")
    print("-"*70)
    
    passed_files = []
    failed_files = []
    error_files = []
    
    for i, test_file in enumerate(intelligent_tests, 1):
        print(f"\n[{i}/{len(intelligent_tests)}] Testing: {test_file.name}")
        
        # Run pytest on the file
        cmd = [
            sys.executable, "-m", "pytest", 
            str(test_file),
            "-xvs",  # Stop on first failure, verbose, no capture
            "--tb=short",  # Short traceback
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout per file
            )
            
            if result.returncode == 0:
                print(f"  ✓ PASSED")
                passed_files.append(test_file.name)
            else:
                if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
                    print(f"  ✗ IMPORT ERROR")
                    error_files.append(test_file.name)
                else:
                    print(f"  ✗ FAILED")
                    failed_files.append(test_file.name)
                    
        except subprocess.TimeoutExpired:
            print(f"  ⚠ TIMEOUT")
            failed_files.append(test_file.name)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            error_files.append(test_file.name)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Files: {len(intelligent_tests)}")
    print(f"Passed: {len(passed_files)} ({len(passed_files)/len(intelligent_tests)*100:.1f}%)")
    print(f"Failed: {len(failed_files)}")
    print(f"Import Errors: {len(error_files)}")
    
    if error_files:
        print(f"\nFiles with import errors:")
        for f in error_files[:10]:  # Show first 10
            print(f"  - {f}")
    
    if failed_files:
        print(f"\nFiles with test failures:")
        for f in failed_files[:10]:  # Show first 10
            print(f"  - {f}")
    
    print("\n✅ Test run complete!")

if __name__ == "__main__":
    main()