#!/usr/bin/env python3
"""
Branch Coverage Analyzer and Test Generator
============================================

Focuses on achieving 100% branch coverage by identifying and testing
all conditional paths, exception handlers, and edge cases.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class BranchInfo:
    """Information about a code branch."""
    line_number: int
    branch_type: str  # 'if', 'elif', 'else', 'try', 'except', 'finally', 'for', 'while'
    condition: str
    covered: bool
    parent_function: str
    complexity: int

@dataclass 
class UncoveredBranch:
    """Represents an uncovered branch that needs testing."""
    file_path: str
    function_name: str
    line_number: int
    branch_type: str
    condition: str
    test_suggestion: str

class BranchCoverageAnalyzer:
    """Analyze and improve branch coverage."""
    
    def __init__(self):
        self.src_dir = Path("src_new")
        self.test_dir = Path("tests_new")
        self.uncovered_branches = []
        
    def analyze_file_branches(self, file_path: Path) -> List[BranchInfo]:
        """Analyze all branches in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                tree = ast.parse(source)
        except:
            return []
        
        branches = []
        
        class BranchVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.branches = []
                
            def visit_FunctionDef(self, node):
                old_func = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_func
                
            def visit_AsyncFunctionDef(self, node):
                old_func = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_func
                
            def visit_If(self, node):
                # Main if branch
                self.branches.append(BranchInfo(
                    line_number=node.lineno,
                    branch_type='if',
                    condition=ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test),
                    covered=False,
                    parent_function=self.current_function or '<module>',
                    complexity=self.calculate_complexity(node.test)
                ))
                
                # elif branches
                for elif_node in node.orelse:
                    if isinstance(elif_node, ast.If):
                        self.branches.append(BranchInfo(
                            line_number=elif_node.lineno,
                            branch_type='elif',
                            condition=ast.unparse(elif_node.test) if hasattr(ast, 'unparse') else str(elif_node.test),
                            covered=False,
                            parent_function=self.current_function or '<module>',
                            complexity=self.calculate_complexity(elif_node.test)
                        ))
                
                # else branch
                if node.orelse and not isinstance(node.orelse[0], ast.If):
                    self.branches.append(BranchInfo(
                        line_number=node.orelse[0].lineno if node.orelse else node.lineno,
                        branch_type='else',
                        condition='not (' + (ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)) + ')',
                        covered=False,
                        parent_function=self.current_function or '<module>',
                        complexity=1
                    ))
                
                self.generic_visit(node)
                
            def visit_Try(self, node):
                # Try block
                self.branches.append(BranchInfo(
                    line_number=node.lineno,
                    branch_type='try',
                    condition='no exception',
                    covered=False,
                    parent_function=self.current_function or '<module>',
                    complexity=1
                ))
                
                # Except handlers
                for handler in node.handlers:
                    exception_type = ast.unparse(handler.type) if handler.type and hasattr(ast, 'unparse') else 'Exception'
                    self.branches.append(BranchInfo(
                        line_number=handler.lineno,
                        branch_type='except',
                        condition=f'raises {exception_type}',
                        covered=False,
                        parent_function=self.current_function or '<module>',
                        complexity=1
                    ))
                
                # Finally block
                if node.finalbody:
                    self.branches.append(BranchInfo(
                        line_number=node.finalbody[0].lineno,
                        branch_type='finally',
                        condition='always',
                        covered=False,
                        parent_function=self.current_function or '<module>',
                        complexity=0
                    ))
                
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.branches.append(BranchInfo(
                    line_number=node.lineno,
                    branch_type='for',
                    condition=f'iterate over {ast.unparse(node.iter) if hasattr(ast, "unparse") else "iterable"}',
                    covered=False,
                    parent_function=self.current_function or '<module>',
                    complexity=1
                ))
                
                # Check for else clause
                if node.orelse:
                    self.branches.append(BranchInfo(
                        line_number=node.orelse[0].lineno,
                        branch_type='for-else',
                        condition='no break',
                        covered=False,
                        parent_function=self.current_function or '<module>',
                        complexity=1
                    ))
                
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.branches.append(BranchInfo(
                    line_number=node.lineno,
                    branch_type='while',
                    condition=ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test),
                    covered=False,
                    parent_function=self.current_function or '<module>',
                    complexity=self.calculate_complexity(node.test)
                ))
                
                self.generic_visit(node)
                
            def calculate_complexity(self, node):
                """Calculate complexity of a condition."""
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.And, ast.Or)):
                        complexity += 1
                    elif isinstance(child, ast.Compare):
                        complexity += len(child.ops) - 1
                return complexity
        
        visitor = BranchVisitor()
        visitor.visit(tree)
        
        return visitor.branches
    
    def get_coverage_data(self) -> Dict:
        """Get current coverage data including branch coverage."""
        print("Running coverage analysis with branch coverage...")
        
        # Run pytest with branch coverage
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new', 
             '--cov=src_new', '--cov-branch', '--cov-report=json',
             '--tb=no', '-q', '--disable-warnings',
             '--timeout=120'],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        # Load coverage data
        coverage_file = Path('coverage.json')
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                return json.load(f)
        
        return {}
    
    def identify_uncovered_branches(self) -> List[UncoveredBranch]:
        """Identify all uncovered branches in the codebase."""
        coverage_data = self.get_coverage_data()
        uncovered = []
        
        for py_file in self.src_dir.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
            
            # Get branches in file
            branches = self.analyze_file_branches(py_file)
            
            # Check coverage data for this file
            file_coverage = None
            for covered_file, data in coverage_data.get('files', {}).items():
                if py_file.name in covered_file:
                    file_coverage = data
                    break
            
            if not file_coverage:
                # File not covered at all
                for branch in branches:
                    uncovered.append(UncoveredBranch(
                        file_path=str(py_file),
                        function_name=branch.parent_function,
                        line_number=branch.line_number,
                        branch_type=branch.branch_type,
                        condition=branch.condition,
                        test_suggestion=self.suggest_test_for_branch(branch)
                    ))
            else:
                # Check which branches are not covered
                missing_branches = file_coverage.get('missing_branches', [])
                for branch in branches:
                    if branch.line_number in missing_branches:
                        uncovered.append(UncoveredBranch(
                            file_path=str(py_file),
                            function_name=branch.parent_function,
                            line_number=branch.line_number,
                            branch_type=branch.branch_type,
                            condition=branch.condition,
                            test_suggestion=self.suggest_test_for_branch(branch)
                        ))
        
        return uncovered
    
    def suggest_test_for_branch(self, branch: BranchInfo) -> str:
        """Generate test suggestion for a specific branch."""
        suggestions = {
            'if': f"Test when {branch.condition} is True",
            'elif': f"Test when {branch.condition} is True (previous conditions False)",
            'else': f"Test when all previous conditions are False",
            'try': "Test successful execution without exceptions",
            'except': f"Test when {branch.condition}",
            'finally': "Test that finally block always executes",
            'for': f"Test iteration with empty and non-empty iterables",
            'for-else': "Test for loop completing without break",
            'while': f"Test when {branch.condition} becomes False"
        }
        
        return suggestions.get(branch.branch_type, f"Test {branch.branch_type} branch")
    
    def generate_branch_test(self, uncovered_branch: UncoveredBranch) -> str:
        """Generate test code for an uncovered branch."""
        test_code = f"""
def test_{uncovered_branch.function_name}_branch_line_{uncovered_branch.line_number}():
    \"\"\"Test {uncovered_branch.branch_type} branch at line {uncovered_branch.line_number}.
    
    Condition: {uncovered_branch.condition}
    Suggestion: {uncovered_branch.test_suggestion}
    \"\"\"
"""
        
        if uncovered_branch.branch_type == 'if':
            test_code += f"""    # Test when condition is True
    mock_obj = Mock()
    # Setup mocks to make condition True
    # Call function and verify True branch executed
    
    # Test when condition is False  
    # Setup mocks to make condition False
    # Call function and verify False branch executed
"""
        
        elif uncovered_branch.branch_type in ['try', 'except']:
            test_code += f"""    # Test exception handling
    mock_obj = Mock()
    
    if '{uncovered_branch.branch_type}' == 'try':
        # Test successful execution
        # Call function without raising exception
        pass
    else:
        # Test exception case
        mock_obj.side_effect = Exception("Test exception")
        with pytest.raises(Exception):
            # Call function that triggers exception
            pass
"""
        
        elif uncovered_branch.branch_type == 'for':
            test_code += f"""    # Test with empty iterable
    result = function_under_test([])
    # Verify behavior with empty list
    
    # Test with non-empty iterable
    result = function_under_test([1, 2, 3])
    # Verify iteration behavior
"""
        
        else:
            test_code += f"""    # Add specific test for {uncovered_branch.branch_type} branch
    # Setup test conditions
    # Call function
    # Verify branch behavior
    pass
"""
        
        return test_code
    
    def generate_branch_coverage_tests(self):
        """Generate tests for all uncovered branches."""
        print("=" * 70)
        print("BRANCH COVERAGE ANALYSIS AND TEST GENERATION")
        print("=" * 70)
        
        # Identify uncovered branches
        uncovered = self.identify_uncovered_branches()
        
        if not uncovered:
            print("No uncovered branches found!")
            return 0
        
        print(f"\nFound {len(uncovered)} uncovered branches")
        
        # Group by file
        by_file = defaultdict(list)
        for branch in uncovered:
            file_name = Path(branch.file_path).stem
            by_file[file_name].append(branch)
        
        # Show summary
        print("\nUncovered branches by module:")
        for module, branches in sorted(by_file.items())[:10]:
            print(f"  {module}: {len(branches)} branches")
            for branch in branches[:3]:
                print(f"    - Line {branch.line_number}: {branch.branch_type} ({branch.test_suggestion})")
        
        # Generate test files
        generated = 0
        for module, branches in by_file.items():
            if generated >= 10:  # Limit for performance
                break
            
            test_file = self.test_dir / f"test_{module}_branches.py"
            
            # Generate test content
            test_content = f'''#!/usr/bin/env python3
"""
Branch coverage tests for {module} module.
Target: 100% branch coverage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src_new"))

import pytest
from unittest.mock import Mock, AsyncMock, patch

# Import module under test
from {module} import *

'''
            
            # Add tests for each uncovered branch
            for branch in branches[:10]:  # Limit tests per file
                test_content += self.generate_branch_test(branch)
                test_content += "\n"
            
            # Save test file
            test_file.write_text(test_content, encoding='utf-8')
            print(f"\nGenerated: {test_file.name}")
            generated += 1
        
        return generated
    
    def measure_branch_coverage(self) -> float:
        """Measure current branch coverage percentage."""
        coverage_data = self.get_coverage_data()
        
        if coverage_data:
            totals = coverage_data.get('totals', {})
            
            # Calculate branch coverage
            num_branches = totals.get('num_branches', 0)
            num_partial_branches = totals.get('num_partial_branches', 0)
            covered_branches = totals.get('covered_branches', 0)
            
            if num_branches > 0:
                branch_coverage = (covered_branches / num_branches) * 100
                print(f"\nBranch Coverage: {branch_coverage:.2f}%")
                print(f"Covered Branches: {covered_branches}/{num_branches}")
                print(f"Partial Branches: {num_partial_branches}")
                return branch_coverage
        
        return 0.0


def main():
    """Run branch coverage analysis and generation."""
    analyzer = BranchCoverageAnalyzer()
    
    # Generate branch coverage tests
    generated = analyzer.generate_branch_coverage_tests()
    
    if generated > 0:
        print(f"\n{generated} branch coverage test files generated")
        
        # Measure improvement
        coverage = analyzer.measure_branch_coverage()
        
        print("\n" + "=" * 70)
        print("BRANCH COVERAGE IMPROVEMENT")
        print("=" * 70)
        print(f"Current branch coverage: {coverage:.2f}%")
        print(f"Target: 100%")
        print(f"Gap: {100 - coverage:.2f}%")
    else:
        print("\nNo branch tests generated - checking current coverage...")
        coverage = analyzer.measure_branch_coverage()
    
    print("\nNext steps:")
    print("1. Run generated branch tests")
    print("2. Fix any test failures")
    print("3. Generate additional tests for remaining branches")
    print("4. Use mutation testing to verify test quality")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())