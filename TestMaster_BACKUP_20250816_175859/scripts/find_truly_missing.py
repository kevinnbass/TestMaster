#!/usr/bin/env python3
"""
Find modules that truly have no test files at all
"""

from pathlib import Path

def find_truly_missing():
    """Find modules with absolutely no test coverage."""
    src_dir = Path("src_new")
    test_dir = Path("tests_new")
    
    # Get all source modules
    src_modules = {}
    for py_file in src_dir.rglob("*.py"):
        if ("__pycache__" not in str(py_file) and 
            py_file.stem != "__init__" and
            not py_file.name.startswith("_")):
            src_modules[py_file.stem] = py_file
    
    print(f"Total source modules: {len(src_modules)}")
    
    # Check which modules have ANY test file
    tested_modules = set()
    if test_dir.exists():
        for test_file in test_dir.glob("test_*.py"):
            # Extract base module name
            name = test_file.stem.replace("test_", "")
            
            # Remove all possible suffixes to get base name
            base_name = name
            for suffix in ["_coverage", "_gemini", "_100coverage", "_api_test", 
                          "_correct_import", "_simple", "_working", "_cov_",
                          "_edge_cases", "_branches", "_mega", "_healed",
                          "_intelligent", "_fixed"]:
                if suffix in base_name:
                    base_name = base_name.split(suffix)[0]
            
            # Also check if the base name exists in source
            if base_name in src_modules:
                tested_modules.add(base_name)
    
    print(f"Modules with at least one test: {len(tested_modules)}")
    
    # Find truly missing
    truly_missing = []
    for module_name, module_path in src_modules.items():
        if module_name not in tested_modules:
            truly_missing.append((module_name, module_path))
    
    truly_missing.sort()
    
    print(f"\nModules with NO tests at all: {len(truly_missing)}")
    
    if truly_missing:
        print("\nList of modules needing tests:")
        for i, (name, path) in enumerate(truly_missing, 1):
            rel_path = path.relative_to(src_dir)
            print(f"  {i:3}. {name:30} ({rel_path})")
    
    return truly_missing

if __name__ == "__main__":
    print("="*60)
    print("FINDING MODULES WITH NO TEST COVERAGE")
    print("="*60)
    
    missing = find_truly_missing()
    
    if missing:
        print(f"\n{len(missing)} modules need tests")
    else:
        print("\nALL MODULES HAVE AT LEAST ONE TEST!")
        print("Ready to measure final coverage percentage")