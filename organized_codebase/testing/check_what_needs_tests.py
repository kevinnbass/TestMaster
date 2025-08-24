#!/usr/bin/env python3
"""
Check what modules need tests
"""

from pathlib import Path

def check_modules():
    """Check which modules need tests."""
    src_dir = Path("src_new")
    test_dir = Path("tests_new")
    
    # Get all source modules
    src_modules = set()
    for py_file in src_dir.rglob("*.py"):
        if ("__pycache__" not in str(py_file) and 
            py_file.stem != "__init__" and
            not py_file.name.startswith("_")):
            src_modules.add(py_file.stem)
    
    print(f"Total source modules: {len(src_modules)}")
    
    # Get existing tests
    test_modules = set()
    if test_dir.exists():
        for test_file in test_dir.glob("test_*.py"):
            # Extract module name from test file
            name = test_file.stem.replace("test_", "")
            # Remove various suffixes
            for suffix in ["_coverage", "_gemini", "_100coverage", "_api_test", 
                          "_correct_import", "_simple", "_cov_", "_branches", 
                          "_intelligent", "_mega"]:
                if suffix in name:
                    name = name.split(suffix)[0]
            test_modules.add(name)
    
    print(f"Modules with tests: {len(test_modules)}")
    
    # Find modules without tests
    missing = src_modules - test_modules
    print(f"Modules needing tests: {len(missing)}")
    
    if missing:
        print("\nFirst 10 modules needing tests:")
        for i, module in enumerate(sorted(missing)[:10]):
            print(f"  {i+1}. {module}")
    
    return sorted(missing)

if __name__ == "__main__":
    print("="*50)
    print("CHECK WHAT NEEDS TESTS")
    print("="*50)
    missing = check_modules()
    
    if missing:
        print(f"\nTo generate tests, need to process {len(missing)} modules")
        print("At 30 RPM, this would take approximately {:.1f} minutes".format(len(missing) * 2 / 60))