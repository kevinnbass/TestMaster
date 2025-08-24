"""
Quick verification that all classical analysis modules are importable and functional.
"""

import sys
from pathlib import Path

# Add TestMaster to path
sys.path.insert(0, str(Path(__file__).parent / "testmaster"))

def verify_modules():
    """Verify all classical analysis modules can be imported."""
    print("=" * 60)
    print("CLASSICAL ANALYSIS MODULE VERIFICATION")
    print("=" * 60)
    
    modules_to_verify = [
        ("Supply Chain Security", "testmaster.analysis.comprehensive_analysis.supply_chain_security", "SupplyChainSecurityAnalyzer"),
        ("API Analysis", "testmaster.analysis.comprehensive_analysis.api_analysis", "APIAnalyzer"),
        ("Testing Analysis", "testmaster.analysis.comprehensive_analysis.testing_analysis", "TestingAnalyzer"),
        ("Performance Analysis", "testmaster.analysis.comprehensive_analysis.performance_analysis", "PerformanceAnalyzer"),
        ("Resource I/O Analysis", "testmaster.analysis.comprehensive_analysis.resource_io_analysis", "ResourceIOAnalyzer"),
        ("Memory Analysis", "testmaster.analysis.comprehensive_analysis.memory_analysis", "MemoryAnalyzer"),
        ("Database Analysis", "testmaster.analysis.comprehensive_analysis.database_analysis", "DatabaseAnalyzer"),
        ("Concurrency Analysis", "testmaster.analysis.comprehensive_analysis.concurrency_analysis", "ConcurrencyAnalyzer"),
        ("Error Handling Analysis", "testmaster.analysis.comprehensive_analysis.error_handling_analysis", "ErrorHandlingAnalyzer"),
        ("Main Analyzer", "testmaster.analysis.comprehensive_analysis.main_analyzer", "ComprehensiveCodebaseAnalyzer"),
    ]
    
    success_count = 0
    fail_count = 0
    
    for name, module_path, class_name in modules_to_verify:
        try:
            # Import module
            module = __import__(module_path, fromlist=[class_name])
            
            # Get class
            analyzer_class = getattr(module, class_name)
            
            # Check if it's a class
            if isinstance(analyzer_class, type):
                print(f"[OK] {name:25} - PASS ({class_name} found)")
                success_count += 1
            else:
                print(f"[FAIL] {name:25} - FAIL (Not a class)")
                fail_count += 1
                
        except ImportError as e:
            print(f"[FAIL] {name:25} - FAIL (Import error: {str(e)[:50]})")
            fail_count += 1
        except AttributeError as e:
            print(f"[FAIL] {name:25} - FAIL (Class not found: {str(e)[:50]})")
            fail_count += 1
        except Exception as e:
            print(f"[FAIL] {name:25} - FAIL ({str(e)[:50]})")
            fail_count += 1
    
    print("=" * 60)
    print(f"RESULTS: {success_count} passed, {fail_count} failed")
    
    if fail_count == 0:
        print("[SUCCESS] All classical analysis modules verified successfully!")
    else:
        print(f"[WARNING] {fail_count} module(s) have issues")
    
    print("=" * 60)
    
    # Quick test of main analyzer
    if success_count == len(modules_to_verify):
        print("\nQuick functionality test...")
        try:
            from testmaster.analysis.comprehensive_analysis.main_analyzer import ComprehensiveCodebaseAnalyzer
            
            test_path = Path(__file__).parent / "testmaster"
            analyzer = ComprehensiveCodebaseAnalyzer(test_path)
            categories = analyzer.get_available_categories()
            
            print(f"Available analysis categories: {len(categories)}")
            for i, cat in enumerate(categories[:10], 1):
                print(f"  {i}. {cat}")
            
            if len(categories) > 10:
                print(f"  ... and {len(categories) - 10} more")
            
            print("\n[SUCCESS] Main analyzer is functional!")
            
        except Exception as e:
            print(f"\n[FAIL] Main analyzer test failed: {str(e)}")
    
    return fail_count == 0


if __name__ == "__main__":
    success = verify_modules()
    sys.exit(0 if success else 1)