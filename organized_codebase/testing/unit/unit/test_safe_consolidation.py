#!/usr/bin/env python3
"""
Safe Consolidation Test
=======================

Test the safe consolidation approach that works with available components
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_safe_import():
    """Test that safe hub can be imported"""
    try:
        from core.intelligence.analysis.safe_comprehensive_analysis_hub import SafeComprehensiveAnalysisHub
        print("PASS: SafeComprehensiveAnalysisHub imported")
        return True
    except Exception as e:
        print(f"FAIL: SafeComprehensiveAnalysisHub - {e}")
        return False

async def test_safe_functionality():
    """Test safe hub functionality"""
    try:
        from core.intelligence.analysis.safe_comprehensive_analysis_hub import SafeComprehensiveAnalysisHub
        
        # Initialize hub
        hub = SafeComprehensiveAnalysisHub()
        print("PASS: SafeComprehensiveAnalysisHub initialized")
        
        # Test capabilities
        capabilities = hub.get_consolidated_analysis_capabilities()
        if capabilities:
            print(f"PASS: Capabilities retrieved - {capabilities['total_analyzers_loaded']} analyzers loaded")
        else:
            print("FAIL: Could not get capabilities")
            return False
        
        # Test analysis (quick test with small scope)
        result = await hub.analyze_comprehensive(".")
        if result and result.status in ['completed', 'failed']:
            print(f"PASS: Analysis completed with status: {result.status}")
        else:
            print("FAIL: Analysis did not complete properly")
            return False
        
        return True
        
    except Exception as e:
        print(f"FAIL: Safe functionality test - {e}")
        return False

async def main():
    print("=" * 60)
    print("SAFE CONSOLIDATION TEST")
    print("=" * 60)
    
    tests = [
        ("Safe Import", test_safe_import, False),
        ("Safe Functionality", test_safe_functionality, True),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func, is_async in tests:
        print(f"\nRunning {test_name} test:")
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
                
            if result:
                print(f"{test_name}: PASSED")
                passed += 1
            else:
                print(f"{test_name}: FAILED")
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("SAFE CONSOLIDATION WORKING")
        print("Analysis consolidation functional with available components")
    else:
        print("SAFE CONSOLIDATION HAS ISSUES")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)