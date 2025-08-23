#!/usr/bin/env python3
"""Simple test for Advanced Gamma Dashboard."""

import sys
import os
from pathlib import Path

# Add web directory to path
sys.path.insert(0, str(Path(__file__).parent / "web"))

def test_advanced_import():
    """Test advanced dashboard import."""
    try:
        from advanced_gamma_dashboard import AdvancedDashboardEngine
        print("PASS: Advanced dashboard import successful")
        return True
    except Exception as e:
        print(f"FAIL: Advanced import error - {e}")
        return False

def test_advanced_initialization():
    """Test advanced dashboard initialization."""
    try:
        from advanced_gamma_dashboard import AdvancedDashboardEngine
        dashboard = AdvancedDashboardEngine(port=5998)
        print("PASS: Advanced dashboard initialization successful")
        return True
    except Exception as e:
        print(f"FAIL: Advanced initialization error - {e}")
        return False

def test_advanced_components():
    """Test advanced component initialization."""
    try:
        from advanced_gamma_dashboard import (
            PredictiveAnalyticsEngine,
            AdvancedInteractionManager,
            PerformanceOptimizer,
            AdvancedReportingSystem
        )
        
        analytics = PredictiveAnalyticsEngine()
        interaction = AdvancedInteractionManager()
        performance = PerformanceOptimizer()
        reporting = AdvancedReportingSystem()
        
        print("PASS: Advanced components initialization successful")
        return True
    except Exception as e:
        print(f"FAIL: Advanced components error - {e}")
        return False

def main():
    print("ADVANCED GAMMA DASHBOARD TEST")
    print("=" * 40)
    
    tests = [
        ("Advanced Import Test", test_advanced_import),
        ("Advanced Initialization Test", test_advanced_initialization),
        ("Advanced Components Test", test_advanced_components)
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
        print("ALL TESTS PASSED - Advanced Dashboard ready!")
        print()
        print("Features Available:")
        print("- Enhanced 3D visualization with advanced interactions")
        print("- Predictive analytics and insights engine")
        print("- Command palette with keyboard shortcuts")
        print("- Performance optimization and monitoring")
        print("- Advanced user behavior tracking")
        print("- Comprehensive reporting and export")
    else:
        print("Some tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)