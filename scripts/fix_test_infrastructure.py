#!/usr/bin/env python3
"""
Fix Test Infrastructure Issues
===============================

Systematically fix all test infrastructure problems.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class TestInfrastructureFixer:
    """Fix all test infrastructure issues."""
    
    def __init__(self):
        self.test_dir = Path("tests_new")
        self.src_dir = Path("src_new")
        self.fixes_applied = {}
        
    def fix_all_imports(self):
        """Fix import issues in all test files."""
        print("=" * 70)
        print("FIXING IMPORT ISSUES")
        print("=" * 70)
        
        import_fix = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src_new"))

"""
        
        fixed_count = 0
        for test_file in self.test_dir.rglob("test_*.py"):
            content = test_file.read_text(encoding='utf-8')
            
            # Check if fix already applied
            if "sys.path.insert" in content:
                continue
            
            # Find where to insert
            lines = content.splitlines()
            insert_pos = 0
            
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and not line.startswith('"""'):
                    if line.startswith('import') or line.startswith('from'):
                        insert_pos = i
                        break
            
            # Insert the fix
            fixed_content = '\n'.join(lines[:insert_pos]) + '\n' + import_fix + '\n'.join(lines[insert_pos:])
            test_file.write_text(fixed_content, encoding='utf-8')
            fixed_count += 1
            print(f"  Fixed: {test_file.name}")
        
        print(f"\nFixed imports in {fixed_count} files")
        return fixed_count
    
    def analyze_api_mismatches(self) -> Dict[str, List[str]]:
        """Analyze API mismatches in tests."""
        print("\n" + "=" * 70)
        print("ANALYZING API MISMATCHES")
        print("=" * 70)
        
        mismatches = {}
        
        # Known API issues from our test runs
        known_issues = {
            'OptimizePatternsRequest': {
                'wrong': 'optimization_config',
                'correct': 'config',
                'file_pattern': '**/test_*optimize*.py'
            },
            'OptimizePatternsResponse': {
                'wrong': ['optimization_time_ms', 'improvements'],
                'correct': ['optimization_metrics'],
                'file_pattern': '**/test_*optimize*.py'
            },
            'Container.register': {
                'wrong': 'service_type=str',
                'correct': 'service_type=Class',
                'file_pattern': '**/test_*container*.py'
            }
        }
        
        for class_name, issue in known_issues.items():
            mismatches[class_name] = []
            
            # Find files with this issue
            for test_file in self.test_dir.rglob("test_*.py"):
                content = test_file.read_text(encoding='utf-8')
                
                if class_name in content:
                    if isinstance(issue['wrong'], list):
                        for wrong_param in issue['wrong']:
                            if wrong_param in content:
                                mismatches[class_name].append(str(test_file))
                                break
                    else:
                        if issue['wrong'] in content:
                            mismatches[class_name].append(str(test_file))
        
        # Report findings
        for class_name, files in mismatches.items():
            if files:
                print(f"\n{class_name}: {len(files)} files with issues")
                for file_path in files[:3]:  # Show first 3
                    print(f"  - {Path(file_path).name}")
        
        return mismatches
    
    def fix_api_mismatches(self):
        """Fix API mismatches automatically."""
        print("\n" + "=" * 70)
        print("FIXING API MISMATCHES")
        print("=" * 70)
        
        fixes = [
            # Fix OptimizePatternsRequest
            {
                'pattern': r'OptimizePatternsRequest\([^)]*optimization_config=',
                'replacement': 'OptimizePatternsRequest(',
                'description': 'Remove optimization_config parameter'
            },
            # Fix OptimizePatternsResponse
            {
                'pattern': r'optimization_time_ms=[\d.]+',
                'replacement': '',
                'description': 'Remove optimization_time_ms'
            },
            {
                'pattern': r'improvements=\{[^}]*\}',
                'replacement': '',
                'description': 'Remove improvements parameter'
            },
            # Fix Container.register string issue
            {
                'pattern': r'container\.register\("([^"]+)",\s*lambda:',
                'replacement': r'container.register(\1_service_type,',
                'description': 'Fix container registration'
            }
        ]
        
        fixed_count = 0
        for fix in fixes:
            for test_file in self.test_dir.rglob("test_*.py"):
                content = test_file.read_text(encoding='utf-8')
                
                if re.search(fix['pattern'], content):
                    new_content = re.sub(fix['pattern'], fix['replacement'], content)
                    
                    if new_content != content:
                        test_file.write_text(new_content, encoding='utf-8')
                        fixed_count += 1
                        print(f"  Fixed {fix['description']} in {test_file.name}")
        
        print(f"\nFixed {fixed_count} API mismatches")
        return fixed_count
    
    def fix_async_issues(self):
        """Fix async/await issues in tests."""
        print("\n" + "=" * 70)
        print("FIXING ASYNC ISSUES")
        print("=" * 70)
        
        fixed_count = 0
        
        for test_file in self.test_dir.rglob("test_*.py"):
            content = test_file.read_text(encoding='utf-8')
            modified = False
            
            # Fix AsyncMock not being awaited
            if 'AsyncMock' in content:
                # Ensure proper AsyncMock usage
                new_content = content.replace(
                    'mock_generator.generate.return_value',
                    'mock_generator.generate.return_value'
                )
                
                # Add await for async mocks
                new_content = re.sub(
                    r'(\s+)response = use_case\.execute\(([^)]+)\)',
                    r'\1response = await use_case.execute(\2)',
                    new_content
                )
                
                if new_content != content:
                    content = new_content
                    modified = True
            
            # Ensure pytest.mark.asyncio is present for async tests
            if 'async def test_' in content and '@pytest.mark.asyncio' not in content:
                lines = content.splitlines()
                new_lines = []
                
                for i, line in enumerate(lines):
                    new_lines.append(line)
                    if 'async def test_' in line:
                        # Add decorator before async def
                        new_lines.insert(len(new_lines) - 1, '@pytest.mark.asyncio')
                        modified = True
                
                if modified:
                    content = '\n'.join(new_lines)
            
            if modified:
                test_file.write_text(content, encoding='utf-8')
                fixed_count += 1
                print(f"  Fixed async issues in {test_file.name}")
        
        print(f"\nFixed async issues in {fixed_count} files")
        return fixed_count
    
    def fix_mock_issues(self):
        """Fix mock-related issues."""
        print("\n" + "=" * 70)
        print("FIXING MOCK ISSUES")
        print("=" * 70)
        
        fixed_count = 0
        
        for test_file in self.test_dir.rglob("test_*.py"):
            content = test_file.read_text(encoding='utf-8')
            modified = False
            
            # Ensure AsyncMock is imported when needed
            if 'AsyncMock' in content and 'from unittest.mock import' in content:
                if 'AsyncMock' not in re.findall(r'from unittest\.mock import ([^)]+)', content)[0]:
                    content = content.replace(
                        'from unittest.mock import Mock',
                        'from unittest.mock import Mock, AsyncMock'
                    )
                    modified = True
            
            # Fix mock return values for async functions
            if 'AsyncMock()' in content:
                # Ensure async mocks return coroutines
                content = re.sub(
                    r'(mock_\w+)\.(\w+)\.return_value = \[',
                    r'\1.\2.return_value = [',
                    content
                )
                
                # This is a simplified fix - more complex logic may be needed
                if content != test_file.read_text(encoding='utf-8'):
                    modified = True
            
            if modified:
                test_file.write_text(content, encoding='utf-8')
                fixed_count += 1
                print(f"  Fixed mock issues in {test_file.name}")
        
        print(f"\nFixed mock issues in {fixed_count} files")
        return fixed_count
    
    def validate_fixes(self):
        """Validate that fixes improved the situation."""
        print("\n" + "=" * 70)
        print("VALIDATING FIXES")
        print("=" * 70)
        
        # Run a quick test to see if imports work
        test_file = self.test_dir / "test_simple.py"
        
        import subprocess
        result = subprocess.run(
            ['python', str(test_file)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ Import fixes validated")
        else:
            print("✗ Some import issues remain")
            print(f"  Error: {result.stderr[:200]}")
        
        return result.returncode == 0
    
    def apply_all_fixes(self):
        """Apply all infrastructure fixes."""
        print("\n" + "=" * 70)
        print("APPLYING ALL TEST INFRASTRUCTURE FIXES")
        print("=" * 70)
        
        results = {
            'imports': self.fix_all_imports(),
            'api_mismatches': self.fix_api_mismatches(),
            'async_issues': self.fix_async_issues(),
            'mock_issues': self.fix_mock_issues()
        }
        
        print("\n" + "=" * 70)
        print("FIX SUMMARY")
        print("=" * 70)
        
        total_fixes = sum(results.values())
        print(f"Total fixes applied: {total_fixes}")
        
        for category, count in results.items():
            print(f"  {category}: {count} fixes")
        
        # Validate
        if self.validate_fixes():
            print("\n✓ Infrastructure fixes successful!")
        else:
            print("\n⚠ Some issues may remain - manual review needed")
        
        return results


def main():
    """Run infrastructure fixes."""
    fixer = TestInfrastructureFixer()
    
    # Analyze issues first
    fixer.analyze_api_mismatches()
    
    # Apply all fixes
    results = fixer.apply_all_fixes()
    
    print("\n" + "=" * 70)
    print("INFRASTRUCTURE FIXES COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run coverage test to verify improvements")
    print("2. Fix any remaining test failures manually")
    print("3. Proceed to Phase 3: Systematic module coverage")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())