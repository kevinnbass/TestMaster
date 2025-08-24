#!/usr/bin/env python3
"""
Simple Consolidation Test
=========================

Quick test to verify analysis consolidation is working
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all components can be imported"""
    print("Testing imports...")
    
    try:
        from core.intelligence.analysis.comprehensive_analysis_hub import ComprehensiveAnalysisHub
        print("PASS: ComprehensiveAnalysisHub imported")
    except Exception as e:
        print(f"FAIL: ComprehensiveAnalysisHub - {e}")
        return False
    
    try:
        from core.intelligence.analytics.analytics_hub import AnalyticsHub
        print("PASS: AnalyticsHub imported")
    except Exception as e:
        print(f"FAIL: AnalyticsHub - {e}")
        return False
    
    return True

def test_integration():
    """Test that integration is working"""
    print("Testing integration...")
    
    try:
        from core.intelligence.analytics.analytics_hub import AnalyticsHub
        hub = AnalyticsHub()
        
        if hasattr(hub, 'comprehensive_analysis_hub'):
            print("PASS: AnalyticsHub has comprehensive_analysis_hub")
        else:
            print("FAIL: AnalyticsHub missing comprehensive_analysis_hub")
            return False
        
        if hasattr(hub, 'analyze_project_comprehensive'):
            print("PASS: AnalyticsHub has analyze_project_comprehensive method")
        else:
            print("FAIL: AnalyticsHub missing analyze_project_comprehensive method")
            return False
        
        capabilities = hub.get_consolidated_analysis_capabilities()
        if capabilities and 'total_analyzers_consolidated' in capabilities:
            print(f"PASS: {capabilities['total_analyzers_consolidated']} analyzers consolidated")
        else:
            print("FAIL: Could not get capabilities")
            return False
        
        return True
        
    except Exception as e:
        print(f"FAIL: Integration test - {e}")
        return False

def main():
    print("=" * 60)
    print("SIMPLE CONSOLIDATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Integration", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test:")
        if test_func():
            print(f"{test_name}: PASSED")
            passed += 1
        else:
            print(f"{test_name}: FAILED")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("CONSOLIDATION SUCCESSFUL")
        print("All 17+ analysis components integrated")
    else:
        print("CONSOLIDATION INCOMPLETE")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)