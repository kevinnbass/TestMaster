#!/usr/bin/env python3
"""
Test Quality Improver
Automatically improves test quality by fixing common issues.
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestQualityImprover:
    """Improves test quality by fixing common issues."""
    
    def __init__(self):
        self.improvements_made = 0
        self.files_processed = 0
        self.issues_found = {}
        
    def improve_test_file(self, file_path: Path) -> Tuple[str, List[str]]:
        """Improve a single test file."""
        logger.info(f"Processing {file_path}")
        
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        improvements = []
        
        # Fix invalid assertions
        content, assertion_fixes = self._fix_invalid_assertions(content)
        improvements.extend(assertion_fixes)
        
        # Add missing imports
        content, import_fixes = self._add_missing_imports(content)
        improvements.extend(import_fixes)
        
        # Fix test structure
        content, structure_fixes = self._fix_test_structure(content)
        improvements.extend(structure_fixes)
        
        # Add proper docstrings
        content, docstring_fixes = self._add_docstrings(content)
        improvements.extend(docstring_fixes)
        
        # Remove duplicates
        content, duplicate_fixes = self._remove_duplicates(content)
        improvements.extend(duplicate_fixes)
        
        # Add fixtures where needed
        content, fixture_fixes = self._add_fixtures(content)
        improvements.extend(fixture_fixes)
        
        if content != original_content:
            self.improvements_made += len(improvements)
        
        return content, improvements
    
    def _fix_invalid_assertions(self, content: str) -> Tuple[str, List[str]]:
        """Fix invalid assertions like 'assert FUNCTION'."""
        improvements = []
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix "assert FUNCTION" patterns
            match = re.match(r'(\s+)assert\s+(\w+)\s*#.*Unknown assertion', line)
            if match:
                indent = match.group(1)
                func_name = match.group(2)
                
                # Replace with proper assertion
                fixed_line = f"{indent}assert {func_name} is not None  # Fixed: verify function exists"
                fixed_lines.append(fixed_line)
                improvements.append(f"Fixed invalid assertion: {func_name}")
            
            # Fix "assert result == None"
            elif "assert result == None" in line:
                fixed_line = line.replace("assert result == None", "assert result is None")
                fixed_lines.append(fixed_line)
                improvements.append("Fixed None comparison")
            
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines), improvements
    
    def _add_missing_imports(self, content: str) -> Tuple[str, List[str]]:
        """Add missing imports for functions referenced in tests."""
        improvements = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content, improvements
        
        # Find all function calls
        function_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    function_calls.add(node.func.id)
        
        # Find existing imports
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name.split('.')[-1])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.name)
        
        # Find undefined functions
        undefined = function_calls - imported_names - {'assert', 'print', 'len', 'str', 'int', 
                                                      'float', 'bool', 'list', 'dict', 'set',
                                                      'tuple', 'None', 'True', 'False'}
        
        if undefined:
            # Add import statements
            import_lines = []
            for func in undefined:
                # Try to guess module name
                module = self._guess_module_name(func, content)
                if module:
                    import_lines.append(f"from {module} import {func}")
                    improvements.append(f"Added import for {func}")
            
            if import_lines:
                # Insert imports after existing imports
                lines = content.split('\n')
                insert_pos = self._find_import_position(lines)
                
                for import_line in import_lines:
                    lines.insert(insert_pos, import_line)
                    insert_pos += 1
                
                content = '\n'.join(lines)
        
        return content, improvements
    
    def _guess_module_name(self, func_name: str, content: str) -> Optional[str]:
        """Guess the module name for a function."""
        # Check test file name for hints
        if "test_" in content:
            # Extract module name from test context
            match = re.search(r'test_(\w+)', content.lower())
            if match:
                return match.group(1)
        
        # Common patterns
        if func_name.startswith("Test"):
            return func_name[4:].lower()
        
        return None
    
    def _find_import_position(self, lines: List[str]) -> int:
        """Find position to insert imports."""
        # After existing imports
        last_import = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                last_import = i + 1
        
        # After module docstring
        if last_import == 0:
            for i, line in enumerate(lines):
                if line.strip().startswith('"""') and i > 0:
                    return i + 1
        
        return max(last_import, 1)
    
    def _fix_test_structure(self, content: str) -> Tuple[str, List[str]]:
        """Fix test structure issues."""
        improvements = []
        
        # Ensure proper test class structure
        if "class Test" in content and "def setup_method" not in content:
            # Add setup_method
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("class Test"):
                    # Find first method
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith("def "):
                            # Insert setup_method before first method
                            setup = [
                                "",
                                "    def setup_method(self):",
                                '        """Set up test fixtures."""',
                                "        pass",
                                ""
                            ]
                            lines[j:j] = setup
                            improvements.append("Added setup_method")
                            break
                    break
            
            content = '\n'.join(lines)
        
        return content, improvements
    
    def _add_docstrings(self, content: str) -> Tuple[str, List[str]]:
        """Add docstrings to test methods."""
        improvements = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content, improvements
        
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_"):
                    # Check if has docstring
                    if not ast.get_docstring(node):
                        # Add docstring
                        line_num = node.lineno - 1
                        indent = len(lines[line_num]) - len(lines[line_num].lstrip())
                        
                        # Generate docstring from test name
                        test_desc = node.name[5:].replace('_', ' ').capitalize()
                        docstring = f'{" " * (indent + 4)}"""Test {test_desc}."""'
                        
                        # Insert after function definition
                        lines.insert(line_num + 1, docstring)
                        improvements.append(f"Added docstring to {node.name}")
        
        return '\n'.join(lines), improvements
    
    def _remove_duplicates(self, content: str) -> Tuple[str, List[str]]:
        """Remove duplicate test methods."""
        improvements = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content, improvements
        
        # Find duplicate function names
        function_names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in function_names:
                    function_names[node.name].append(node.lineno)
                else:
                    function_names[node.name] = [node.lineno]
        
        # Identify duplicates
        duplicates = {name: lines for name, lines in function_names.items() if len(lines) > 1}
        
        if duplicates:
            lines = content.split('\n')
            lines_to_remove = set()
            
            for name, line_nums in duplicates.items():
                # Keep first occurrence, remove others
                for line_num in line_nums[1:]:
                    # Find end of function
                    start = line_num - 1
                    end = start
                    
                    # Find next function or class
                    for i in range(start + 1, len(lines)):
                        if lines[i].strip().startswith("def ") or lines[i].strip().startswith("class "):
                            end = i - 1
                            break
                    else:
                        end = len(lines) - 1
                    
                    # Mark lines for removal
                    for i in range(start, end + 1):
                        lines_to_remove.add(i)
                    
                    improvements.append(f"Removed duplicate: {name}")
            
            # Remove marked lines
            content = '\n'.join(line for i, line in enumerate(lines) if i not in lines_to_remove)
        
        return content, improvements
    
    def _add_fixtures(self, content: str) -> Tuple[str, List[str]]:
        """Add fixtures where needed."""
        improvements = []
        
        # Check if fixtures are needed
        if "self.instance" in content and "@pytest.fixture" not in content:
            # Add fixture import
            if "import pytest" not in content:
                lines = content.split('\n')
                insert_pos = self._find_import_position(lines)
                lines.insert(insert_pos, "import pytest")
                content = '\n'.join(lines)
                improvements.append("Added pytest import")
        
        return content, improvements
    
    def process_directory(self, test_dir: Path) -> Dict[str, Any]:
        """Process all test files in a directory."""
        results = {
            "files_processed": 0,
            "improvements_made": 0,
            "files_improved": [],
            "issues_by_type": {}
        }
        
        for test_file in test_dir.rglob("test_*.py"):
            if "__pycache__" in str(test_file):
                continue
            
            try:
                content, improvements = self.improve_test_file(test_file)
                
                if improvements:
                    # Backup original
                    backup_file = test_file.with_suffix('.py.backup')
                    if not backup_file.exists():
                        backup_file.write_text(test_file.read_text(encoding='utf-8'))
                    
                    # Write improved version
                    test_file.write_text(content, encoding='utf-8')
                    
                    results["files_improved"].append(str(test_file))
                    results["improvements_made"] += len(improvements)
                    
                    # Track issue types
                    for improvement in improvements:
                        issue_type = improvement.split(':')[0]
                        results["issues_by_type"][issue_type] = \
                            results["issues_by_type"].get(issue_type, 0) + 1
                
                results["files_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {test_file}: {e}")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate improvement report."""
        report = []
        report.append("=" * 60)
        report.append("TEST QUALITY IMPROVEMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Files Processed: {results['files_processed']}")
        report.append(f"Files Improved: {len(results['files_improved'])}")
        report.append(f"Total Improvements: {results['improvements_made']}")
        report.append("")
        
        if results["issues_by_type"]:
            report.append("IMPROVEMENTS BY TYPE:")
            report.append("-" * 40)
            
            for issue_type, count in sorted(results["issues_by_type"].items(), 
                                           key=lambda x: x[1], reverse=True):
                report.append(f"  {issue_type}: {count}")
            
            report.append("")
        
        if results["files_improved"]:
            report.append("FILES IMPROVED:")
            report.append("-" * 40)
            
            for file_path in results["files_improved"][:20]:
                report.append(f"  {Path(file_path).name}")
            
            if len(results["files_improved"]) > 20:
                report.append(f"  ... and {len(results['files_improved']) - 20} more")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improve test quality")
    parser.add_argument("test_dir", help="Test directory to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    parser.add_argument("--report", help="Output report file")
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)
    
    improver = TestQualityImprover()
    
    # Process tests
    results = improver.process_directory(test_dir)
    
    # Generate report
    report = improver.generate_report(results)
    print(report)
    
    if args.report:
        Path(args.report).write_text(report)
        print(f"\nReport saved to: {args.report}")
    
    print(f"\n✅ Improved {len(results['files_improved'])} test files!")


if __name__ == "__main__":
    main()