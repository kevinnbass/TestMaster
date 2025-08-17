#!/usr/bin/env python3
"""
Fix Import Issues in All Generated Tests
========================================

Adds proper import path setup to all generated test files.
"""

import sys
from pathlib import Path

def fix_test_imports(test_file: Path) -> bool:
    """Fix imports in a single test file."""
    
    try:
        content = test_file.read_text(encoding='utf-8')
        
        # Check if already fixed
        if "sys.path.insert" in content:
            return False
        
        # Add import fix at the beginning
        import_fix = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src_new"))

"""
        
        # Insert after initial imports but before actual imports
        lines = content.splitlines()
        insert_pos = 0
        
        # Find where to insert (after initial comments/docstrings)
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#') and not line.startswith('"""'):
                if line.startswith('import') or line.startswith('from'):
                    insert_pos = i
                    break
        
        # Insert the fix
        fixed_content = '\n'.join(lines[:insert_pos]) + '\n' + import_fix + '\n'.join(lines[insert_pos:])
        
        # Save the fixed file
        test_file.write_text(fixed_content, encoding='utf-8')
        return True
        
    except Exception as e:
        print(f"  ERROR fixing {test_file.name}: {e}")
        return False


def main():
    """Fix all test files."""
    
    print("="*60)
    print("FIXING IMPORTS IN ALL GENERATED TESTS")
    print("="*60)
    
    test_dir = Path("tests_new/gemini_generated")
    test_files = list(test_dir.glob("test_*.py"))
    
    print(f"\nFound {len(test_files)} test files to fix\n")
    
    fixed_count = 0
    already_fixed = 0
    
    for test_file in test_files:
        print(f"  {test_file.name}...", end=" ")
        if fix_test_imports(test_file):
            print("[FIXED]")
            fixed_count += 1
        else:
            print("[ALREADY OK]")
            already_fixed += 1
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Fixed: {fixed_count}")
    print(f"  Already OK: {already_fixed}")
    print(f"  Total: {len(test_files)}")
    print("="*60)
    
    if fixed_count > 0:
        print("\nNow you can run tests with:")
        print("python -m pytest tests_new/gemini_generated --cov=src_new --cov-report=term")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())