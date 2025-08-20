"""
Verification script for all classical analysis modules
"""

import sys
import traceback
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def verify_modules():
    """Verify all analysis modules can be imported and instantiated"""
    
    modules_to_verify = [
        # Original 10 modules
        ("supply_chain_security", "SupplyChainSecurityAnalyzer"),
        ("api_analysis", "APIAnalyzer"),
        ("testing_analysis", "TestingAnalyzer"),
        ("performance_analysis", "PerformanceAnalyzer"),
        ("resource_io_analysis", "ResourceIOAnalyzer"),
        ("memory_analysis", "MemoryAnalyzer"),
        ("database_analysis", "DatabaseAnalyzer"),
        ("concurrency_analysis", "ConcurrencyAnalyzer"),
        ("error_handling_analysis", "ErrorHandlingAnalyzer"),
        ("security_analysis", "SecurityAnalyzer"),
        # New modules added today
        ("cognitive_load_analysis", "CognitiveLoadAnalyzer"),
        ("technical_debt_analysis", "TechnicalDebtAnalyzer"),
        ("ml_code_analysis", "MLCodeAnalyzer")
    ]
    
    results = []
    
    for module_name, class_name in modules_to_verify:
        try:
            # Try to import the module
            module_path = f"testmaster.analysis.comprehensive_analysis.{module_name}"
            module = __import__(module_path, fromlist=[class_name])
            
            # Try to get the class
            analyzer_class = getattr(module, class_name)
            
            # Try to instantiate it
            analyzer = analyzer_class()
            
            results.append({
                "module": module_name,
                "status": "SUCCESS",
                "message": f"[OK] {class_name} loaded successfully"
            })
            
        except ImportError as e:
            results.append({
                "module": module_name,
                "status": "IMPORT_ERROR",
                "message": f"[FAIL] Import error: {str(e)}"
            })
            
        except AttributeError as e:
            results.append({
                "module": module_name,
                "status": "CLASS_ERROR",
                "message": f"[FAIL] Class not found: {str(e)}"
            })
            
        except Exception as e:
            results.append({
                "module": module_name,
                "status": "OTHER_ERROR",
                "message": f"[FAIL] Unexpected error: {str(e)}"
            })
    
    # Print results
    print("\n" + "="*60)
    print("MODULE VERIFICATION RESULTS")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for result in results:
        print(result["message"])
        if result["status"] == "SUCCESS":
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "-"*60)
    print(f"SUMMARY: {success_count} modules loaded successfully, {fail_count} failed")
    print("-"*60)
    
    # Show module count by category
    print("\nMODULE CATEGORIES:")
    print("- Security & Safety: 3 modules")
    print("- Performance & Optimization: 3 modules")
    print("- Testing & Quality: 2 modules")
    print("- Code Complexity: 2 modules")
    print("- Technical Debt: 1 module")
    print("- Machine Learning: 1 module")
    print("- Infrastructure: 1 module")
    
    print(f"\nTOTAL MODULES IMPLEMENTED: {len(modules_to_verify)}")
    
    return success_count == len(modules_to_verify)

if __name__ == "__main__":
    success = verify_modules()
    sys.exit(0 if success else 1)