#!/usr/bin/env python3
"""
Quick test summary - test each file and report results
"""

import subprocess
import sys
from pathlib import Path

def test_file(test_path):
    """Test a single file and return result."""
    cmd = [sys.executable, "-m", "pytest", str(test_path), "-x", "--tb=no", "-q"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if "passed" in result.stdout:
            if "failed" in result.stdout or "error" in result.stdout:
                return "PARTIAL"
            return "PASS"
        elif "ERROR" in result.stdout or "ERROR" in result.stderr:
            return "ERROR"
        else:
            return "FAIL"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except:
        return "ERROR"

def main():
    test_dir = Path("tests/unit")
    intelligent_tests = sorted(test_dir.glob("*_intelligent.py"))
    
    print("Testing intelligent test files...")
    print("-" * 50)
    
    results = {"PASS": [], "PARTIAL": [], "FAIL": [], "ERROR": [], "TIMEOUT": []}
    
    for i, test_path in enumerate(intelligent_tests[:10], 1):  # Test first 10
        print(f"[{i}/10] {test_path.name}...", end=" ")
        status = test_file(test_path)
        print(status)
        results[status].append(test_path.name)
    
    print("\n" + "=" * 50)
    print("SUMMARY (First 10 files)")
    print("=" * 50)
    print(f"PASS: {len(results['PASS'])}")
    print(f"PARTIAL: {len(results['PARTIAL'])}")
    print(f"ERROR: {len(results['ERROR'])}")
    print(f"FAIL: {len(results['FAIL'])}")
    print(f"TIMEOUT: {len(results['TIMEOUT'])}")
    
    if results['PASS']:
        print(f"\nPassing files:")
        for f in results['PASS']:
            print(f"  [OK] {f}")

if __name__ == "__main__":
    main()