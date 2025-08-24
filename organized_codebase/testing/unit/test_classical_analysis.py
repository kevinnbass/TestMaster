"""
Test Classical Analysis System
===============================

Comprehensive test to verify all classical analysis modules are working correctly.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add the TestMaster directory to path
sys.path.insert(0, str(Path(__file__).parent / "testmaster"))

from testmaster.analysis.comprehensive_analysis.main_analyzer import ComprehensiveCodebaseAnalyzer


def test_classical_analysis():
    """Test the comprehensive classical analysis system."""
    print("=" * 80)
    print("TESTING CLASSICAL ANALYSIS SYSTEM")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 80)
    
    # Initialize analyzer with TestMaster codebase
    base_path = Path(__file__).parent / "testmaster"
    
    if not base_path.exists():
        print(f"[ERROR] Path does not exist: {base_path}")
        return False
    
    print(f"[INFO] Analyzing codebase at: {base_path}")
    print(f"[INFO] Initializing comprehensive analyzer...")
    
    try:
        analyzer = ComprehensiveCodebaseAnalyzer(base_path)
        available_categories = analyzer.get_available_categories()
        
        print(f"[OK] Analyzer initialized successfully")
        print(f"[INFO] Available analysis categories: {len(available_categories)}")
        for category in available_categories:
            print(f"  - {category}")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize analyzer: {str(e)}")
        return False
    
    print("-" * 80)
    print("[INFO] Testing individual analysis modules...")
    print("-" * 80)
    
    # Test each category individually
    test_results = {}
    failed_categories = []
    
    # Priority categories to test
    priority_categories = [
        'supply_chain_security',
        'api_analysis', 
        'testing_analysis',
        'performance_analysis',
        'resource_io_analysis',
        'memory_analysis',
        'database_analysis',
        'concurrency_analysis',
        'error_handling_analysis'
    ]
    
    for category in priority_categories:
        if category not in available_categories:
            print(f"[WARNING] Category not found: {category}")
            continue
            
        print(f"\n[TEST] Testing {category}...")
        start_time = datetime.now()
        
        try:
            result = analyzer.analyze_category(category)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if 'error' in result.get('results', {}):
                print(f"  [FAIL] {category} - Error: {result['results']['error']}")
                failed_categories.append(category)
                test_results[category] = {
                    'status': 'failed',
                    'error': result['results']['error'],
                    'duration': duration
                }
            else:
                # Check if result has expected structure
                if 'results' in result and isinstance(result['results'], dict):
                    num_aspects = len(result['results'])
                    print(f"  [OK] {category} - Analyzed {num_aspects} aspects in {duration:.2f}s")
                    
                    # Show sample results
                    for key in list(result['results'].keys())[:3]:
                        value = result['results'][key]
                        if isinstance(value, dict):
                            print(f"    - {key}: {len(value)} items")
                        elif isinstance(value, list):
                            print(f"    - {key}: {len(value)} items")
                        else:
                            print(f"    - {key}: {type(value).__name__}")
                    
                    test_results[category] = {
                        'status': 'success',
                        'aspects': num_aspects,
                        'duration': duration
                    }
                else:
                    print(f"  [WARN] {category} - Unexpected result structure")
                    test_results[category] = {
                        'status': 'warning',
                        'duration': duration
                    }
                    
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"  [ERROR] {category} - Exception: {str(e)}")
            failed_categories.append(category)
            test_results[category] = {
                'status': 'error',
                'exception': str(e),
                'duration': duration
            }
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    # Calculate statistics
    total_tested = len(test_results)
    successful = len([r for r in test_results.values() if r['status'] == 'success'])
    warnings = len([r for r in test_results.values() if r['status'] == 'warning'])
    failed = len([r for r in test_results.values() if r['status'] in ['failed', 'error']])
    
    print(f"Total Categories Tested: {total_tested}")
    print(f"Successful: {successful} ({successful/total_tested*100:.1f}%)")
    print(f"Warnings: {warnings} ({warnings/total_tested*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_tested*100:.1f}%)")
    
    if failed_categories:
        print(f"\nFailed Categories:")
        for cat in failed_categories:
            print(f"  - {cat}: {test_results[cat].get('error', test_results[cat].get('exception', 'Unknown error'))}")
    
    # Test comprehensive analysis (all categories together)
    print("\n" + "-" * 80)
    print("[INFO] Testing comprehensive analysis (subset of categories)...")
    
    try:
        # Test with a subset to avoid timeout
        test_categories = ['api_analysis', 'memory_analysis', 'error_handling_analysis']
        print(f"[INFO] Running analysis for: {test_categories}")
        
        start_time = datetime.now()
        comprehensive_result = analyzer.analyze_comprehensive(selected_categories=test_categories)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if comprehensive_result and 'analysis_results' in comprehensive_result:
            num_completed = comprehensive_result['metadata'].get('categories_completed', 0)
            success_rate = comprehensive_result['metadata'].get('success_rate', 0)
            
            print(f"[OK] Comprehensive analysis completed in {duration:.2f}s")
            print(f"  - Categories analyzed: {num_completed}")
            print(f"  - Success rate: {success_rate:.1%}")
            
            # Check for summary and recommendations
            if 'summary' in comprehensive_result:
                print(f"  - Summary generated: Yes")
                
            if 'recommendations' in comprehensive_result:
                num_recommendations = len(comprehensive_result['recommendations'])
                print(f"  - Recommendations: {num_recommendations}")
                
                # Show first few recommendations
                for rec in comprehensive_result['recommendations'][:3]:
                    if isinstance(rec, dict):
                        print(f"    • {rec.get('title', rec.get('description', 'Unknown'))}")
                    else:
                        print(f"    • {rec}")
        else:
            print(f"[WARN] Comprehensive analysis returned unexpected structure")
            
    except Exception as e:
        print(f"[ERROR] Comprehensive analysis failed: {str(e)}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if failed == 0:
        print("✅ ALL TESTS PASSED - Classical Analysis System is fully operational!")
    elif successful > failed:
        print("⚠️ PARTIAL SUCCESS - Most modules working, some issues detected")
    else:
        print("❌ TEST FAILED - Critical issues in Classical Analysis System")
    
    print("=" * 80)
    
    # Save test results
    output_file = Path("classical_analysis_test_results.json")
    try:
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'base_path': str(base_path),
                'test_results': test_results,
                'summary': {
                    'total': total_tested,
                    'successful': successful,
                    'warnings': warnings,
                    'failed': failed,
                    'success_rate': successful / total_tested if total_tested > 0 else 0
                }
            }, f, indent=2)
        print(f"\n[INFO] Test results saved to: {output_file}")
    except Exception as e:
        print(f"\n[WARNING] Could not save test results: {str(e)}")
    
    return failed == 0


def test_specific_module(module_name: str):
    """Test a specific analysis module."""
    print(f"\n[TEST] Testing specific module: {module_name}")
    print("-" * 40)
    
    base_path = Path(__file__).parent / "testmaster"
    analyzer = ComprehensiveCodebaseAnalyzer(base_path)
    
    try:
        result = analyzer.analyze_category(module_name)
        
        if 'error' not in result.get('results', {}):
            print(f"[OK] {module_name} analysis successful")
            
            # Display detailed results
            if 'results' in result:
                for key, value in result['results'].items():
                    if isinstance(value, dict):
                        print(f"\n{key}:")
                        for k, v in list(value.items())[:5]:
                            if isinstance(v, (list, dict)):
                                print(f"  - {k}: {len(v)} items")
                            else:
                                print(f"  - {k}: {v}")
                    elif isinstance(value, list):
                        print(f"\n{key}: {len(value)} items")
                        for item in value[:3]:
                            print(f"  - {item}")
                    else:
                        print(f"\n{key}: {value}")
            
            return True
        else:
            print(f"[FAIL] {module_name} analysis failed: {result['results']['error']}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Exception in {module_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run comprehensive test
    success = test_classical_analysis()
    
    # Optionally test specific modules in detail
    if '--detailed' in sys.argv:
        print("\n" + "=" * 80)
        print("DETAILED MODULE TESTING")
        print("=" * 80)
        
        modules_to_test = [
            'memory_analysis',
            'database_analysis',
            'concurrency_analysis',
            'error_handling_analysis'
        ]
        
        for module in modules_to_test:
            test_specific_module(module)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)