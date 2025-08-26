#!/usr/bin/env python3
"""
Generate pytest test scaffolds for top 100 refactoring targets.
"""

import json
import os
import re
from pathlib import Path

# Configuration
ROOT = Path(__file__).parent.parent.parent  # testmaster root
TARGETS_FILE = ROOT / "tools" / "codebase_monitor" / "outputs" / "refactor_top100.json"
TESTS_DIR = ROOT / "tests" / "refactor_targets"

def sanitize_filename(path_str):
    """Convert a file path to a safe test filename."""
    # Replace path separators and special characters with underscores
    safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', path_str)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name

def generate_test_content(original_path, safe_name):
    """Generate test content for a specific file."""
    return f'''"""
Test scaffolds for refactoring target: {original_path}

This file contains placeholder tests that should be expanded as the target
file is refactored to improve maintainability and reduce complexity.
"""

import pytest
from pathlib import Path


class TestRefactorTarget_{safe_name.replace('.', '_').replace('-', '_')}:
    """Test class for {original_path}"""
    
    @pytest.fixture
    def target_file_path(self):
        """Path to the target file being refactored."""
        return Path(__file__).parent.parent.parent / "{original_path.replace(chr(92), '/')}"
    
    def test_file_exists(self, target_file_path):
        """Test that the target file exists."""
        assert target_file_path.exists(), f"Target file {{target_file_path}} does not exist"
    
    def test_file_is_readable(self, target_file_path):
        """Test that the target file can be read."""
        if target_file_path.exists():
            try:
                content = target_file_path.read_text(encoding='utf-8')
                assert len(content) > 0, "File should not be empty"
            except Exception as e:
                pytest.fail(f"Could not read file: {{e}}")
    
    def test_basic_syntax_valid(self, target_file_path):
        """Test that the file has valid Python syntax."""
        if target_file_path.exists() and target_file_path.suffix == '.py':
            try:
                import ast
                content = target_file_path.read_text(encoding='utf-8')
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {{target_file_path}}: {{e}}")
            except Exception:
                # Skip syntax check for non-Python files
                pass
    
    def test_placeholder_for_complexity_reduction(self):
        """Placeholder test for complexity metrics improvement.
        
        TODO: Add specific tests for:
        - Function length reduction
        - Cyclomatic complexity improvement  
        - Reduced branching
        - Better separation of concerns
        """
        assert True, "Placeholder - implement complexity tests after refactoring"
    
    def test_placeholder_for_maintainability(self):
        """Placeholder test for maintainability improvements.
        
        TODO: Add specific tests for:
        - Clear function/class responsibilities
        - Proper error handling
        - Documentation coverage
        - Code organization
        """
        assert True, "Placeholder - implement maintainability tests after refactoring"


# Additional module-level tests
def test_module_import_safety():
    """Test that this test module can be imported safely."""
    assert __name__ is not None


def test_target_path_reference():
    """Test that the target path is properly referenced."""
    target_path = "{original_path}"
    assert isinstance(target_path, str)
    assert len(target_path) > 0
'''

def main():
    """Main function to generate all test scaffolds."""
    # Ensure the tests directory exists
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the targets list
    if not TARGETS_FILE.exists():
        print(f"Error: Targets file not found: {TARGETS_FILE}")
        print("Please run the top_refactor_picker.py script first.")
        return 1
    
    with open(TARGETS_FILE, 'r', encoding='utf-8') as f:
        targets_data = json.load(f)
    
    top100 = targets_data.get('top100', [])
    
    if not top100:
        print("Error: No targets found in the targets file.")
        return 1
    
    print(f"Generating test scaffolds for {len(top100)} files...")
    
    # Create __init__.py for the test package
    init_file = TESTS_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Test package for refactoring targets."""\n')
    
    generated_count = 0
    skipped_count = 0
    
    for i, target_path in enumerate(top100):
        # Create a safe filename for the test
        safe_name = sanitize_filename(target_path)
        test_filename = f"test_{safe_name}.py"
        test_path = TESTS_DIR / test_filename
        
        # Skip if test already exists (don't overwrite)
        if test_path.exists():
            print(f"Skipping {test_filename} (already exists)")
            skipped_count += 1
            continue
        
        # Generate test content
        test_content = generate_test_content(target_path, safe_name)
        
        # Write the test file
        try:
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            generated_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{len(top100)} test scaffolds...")
                
        except Exception as e:
            print(f"Error generating test for {target_path}: {e}")
    
    print(f"✅ Test scaffold generation complete:")
    print(f"   Generated: {generated_count} new test files")
    print(f"   Skipped: {skipped_count} existing test files")
    print(f"   Total targets: {len(top100)}")
    print(f"   Output directory: {TESTS_DIR}")
    
    # Generate a summary test runner
    runner_path = TESTS_DIR / "run_refactor_tests.py"
    if not runner_path.exists():
        runner_content = '''#!/usr/bin/env python3
"""
Test runner for refactoring target tests.
"""

import pytest
import sys
from pathlib import Path

def main():
    """Run all refactor target tests."""
    test_dir = Path(__file__).parent
    
    print("Running refactor target tests...")
    print(f"Test directory: {test_dir}")
    
    # Run pytest on this directory
    args = [
        str(test_dir),
        "-v",  # verbose
        "--tb=short",  # short traceback format
        "--durations=10",  # show 10 slowest tests
    ]
    
    exit_code = pytest.main(args)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
'''
        runner_path.write_text(runner_content)
        print(f"📝 Generated test runner: {runner_path}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())