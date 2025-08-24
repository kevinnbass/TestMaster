#!/usr/bin/env python3
"""Simple test for Unified Gamma Dashboard without Unicode characters."""

import sys
import os
from pathlib import Path

# Add web directory to path
sys.path.insert(0, str(Path(__file__).parent / "web"))

def test_import():
    """Test dashboard import."""
    try:
        from unified_gamma_dashboard import UnifiedDashboardEngine
        print("PASS: Dashboard import successful")
        return True
    except Exception as e:
        print(f"FAIL: Import error - {e}")
        return False

def test_initialization():
    """Test dashboard initialization."""
    try:
        from unified_gamma_dashboard import UnifiedDashboardEngine
        dashboard = UnifiedDashboardEngine(port=5999)
        print("PASS: Dashboard initialization successful")
        return True
    except Exception as e:
        print(f"FAIL: Initialization error - {e}")
        return False

def main():
    print("UNIFIED GAMMA DASHBOARD TEST")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_import),
        ("Initialization Test", test_initialization)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Running {name}...")
        success = test_func()
        results.append(success)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("SUMMARY:")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("ALL TESTS PASSED - Dashboard ready!")
    else:
        print("Some tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)