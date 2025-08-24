#!/usr/bin/env python3
"""
Agent C - AI-Powered Test Generation Enhancement
Enhances existing TestDiscoveryEngine with AI capabilities
Integrates with existing AdvancedTestEngine framework
"""

import os
import re
import ast
import json
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

# Import from existing test framework
import sys
sys.path.append(str(Path(__file__).parent.parent))
from framework.test_engine import (
    TestCase, TestType, TestSuite, FeatureDiscoveryLog
)

@dataclass
class AITestPattern:
    """AI-identified test pattern"""
    pattern_name: str
    pattern_type: str
    confidence_score: float
    applicable_functions: List[str]
    test_template: str
    expected_coverage_increase: float
    complexity_level: int

@dataclass
class TestGenerationContext:
    """Context for AI test generation"""
    source_file: str
    function_name: str
    function_signature: str
    function_body: str
    existing_tests: List[TestCase]
    dependencies: List[str]
    complexity_metrics: Dict[str, Any]

class CodeAnalysisEngine:
    """Analyzes code structure for intelligent test generation"""
    
    def __init__(self):
        self.feature_discovery_log = FeatureDiscoveryLog()
        
    def analyze_function_complexity(self, function_code: str) -> Dict[str, Any]:
        """Analyze function complexity for test generation"""
        try:
            tree = ast.parse(function_code)
            
            complexity_metrics = {
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
                'parameter_count': self._count_parameters(tree),
                'return_paths': self._count_return_paths(tree),
                'exception_handling': self._has_exception_handling(tree),
                'loop_count': self._count_loops(tree),
                'condition_count': self._count_conditions(tree),
                'nested_depth': self._calculate_nested_depth(tree),
                'external_calls': self._count_external_calls(tree)
            }
            
            return complexity_metrics
            
        except Exception as e:
            return {'error': str(e), 'complexity_score': 1}
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
    
    def _count_parameters(self, tree: ast.AST) -> int:
        """Count function parameters"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return len(node.args.args)
        return 0
    
    def _count_return_paths(self, tree: ast.AST) -> int:
        """Count return paths"""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Return)])
    
    def _has_exception_handling(self, tree: ast.AST) -> bool:
        """Check for exception handling"""
        return any(isinstance(node, (ast.Try, ast.Raise, ast.ExceptHandler)) 
                  for node in ast.walk(tree))
    
    def _count_loops(self, tree: ast.AST) -> int:
        """Count loops"""
        return len([node for node in ast.walk(tree) 
                   if isinstance(node, (ast.For, ast.While))])
    
    def _count_conditions(self, tree: ast.AST) -> int:
        """Count conditional statements"""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
    
    def _calculate_nested_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        def traverse(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                depth += 1
            
            for child in ast.iter_child_nodes(node):
                traverse(child, depth)
        
        traverse(tree)
        return max_depth
    
    def _count_external_calls(self, tree: ast.AST) -> int:
        """Count external function calls"""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.Call)])

class TestPatternLearner:
    """Learns from existing test patterns to improve generation"""
    
    def __init__(self):
        self.learned_patterns = []
        self.pattern_effectiveness = {}
        self.feature_discovery_log = FeatureDiscoveryLog()
    
    def analyze_existing_tests(self, test_files: List[str]) -> List[AITestPattern]:
        """Analyze existing tests to learn patterns"""
        patterns = []
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_patterns = self._extract_patterns_from_file(content, test_file)
                patterns.extend(file_patterns)
                
            except Exception as e:
                self.feature_discovery_log.log_discovery_attempt(
                    f"pattern_learning_error_{test_file}",
                    {'error': str(e), 'file': test_file}
                )
        
        return self._consolidate_patterns(patterns)
    
    def _extract_patterns_from_file(self, content: str, file_path: str) -> List[AITestPattern]:
        """Extract test patterns from a single file"""
        patterns = []
        
        # Pattern 1: Arrange-Act-Assert pattern
        if re.search(r'#\s*Arrange[\s\S]*?#\s*Act[\s\S]*?#\s*Assert', content, re.IGNORECASE):
            patterns.append(AITestPattern(
                pattern_name="arrange_act_assert",
                pattern_type="structure",
                confidence_score=0.9,
                applicable_functions=self._extract_tested_functions(content),
                test_template=self._extract_aaa_template(content),
                expected_coverage_increase=0.15,
                complexity_level=1
            ))
        
        # Pattern 2: Parameterized testing pattern
        if re.search(r'@pytest\.mark\.parametrize|@pytest\.fixture', content):
            patterns.append(AITestPattern(
                pattern_name="parameterized_testing",
                pattern_type="data_driven",
                confidence_score=0.8,
                applicable_functions=self._extract_tested_functions(content),
                test_template=self._extract_parametrized_template(content),
                expected_coverage_increase=0.25,
                complexity_level=2
            ))
        
        # Pattern 3: Exception testing pattern
        if re.search(r'pytest\.raises|assertRaises|with.*raises', content):
            patterns.append(AITestPattern(
                pattern_name="exception_testing",
                pattern_type="error_handling",
                confidence_score=0.85,
                applicable_functions=self._extract_exception_tested_functions(content),
                test_template=self._extract_exception_template(content),
                expected_coverage_increase=0.20,
                complexity_level=2
            ))
        
        # Pattern 4: Mock usage pattern
        if re.search(r'@patch|Mock\(|MagicMock', content):
            patterns.append(AITestPattern(
                pattern_name="mocking_pattern",
                pattern_type="isolation",
                confidence_score=0.75,
                applicable_functions=self._extract_mocked_functions(content),
                test_template=self._extract_mock_template(content),
                expected_coverage_increase=0.30,
                complexity_level=3
            ))
        
        return patterns
    
    def _extract_tested_functions(self, content: str) -> List[str]:
        """Extract functions being tested"""
        # Simple regex to find function calls in tests
        function_calls = re.findall(r'(\w+)\s*\(', content)
        # Filter out common test keywords
        excluded = {'assert', 'expect', 'test', 'mock', 'patch', 'setup', 'teardown'}
        return list(set(func for func in function_calls if func not in excluded))
    
    def _extract_exception_tested_functions(self, content: str) -> List[str]:
        """Extract functions tested for exceptions"""
        exception_patterns = re.findall(r'pytest\.raises.*?(\w+)\(', content)
        return list(set(exception_patterns))
    
    def _extract_mocked_functions(self, content: str) -> List[str]:
        """Extract functions that are mocked"""
        mock_patterns = re.findall(r'@patch\(["\']([^"\']+)["\']', content)
        return list(set(mock_patterns))
    
    def _extract_aaa_template(self, content: str) -> str:
        """Extract Arrange-Act-Assert template"""
        return '''def test_{function_name}():
    # Arrange
    {setup_code}
    
    # Act
    result = {function_call}
    
    # Assert
    {assertions}'''
    
    def _extract_parametrized_template(self, content: str) -> str:
        """Extract parameterized test template"""
        return '''@pytest.mark.parametrize("input_data,expected", [
    ({test_data_1}),
    ({test_data_2}),
])
def test_{function_name}_parameterized(input_data, expected):
    result = {function_call}
    assert result == expected'''
    
    def _extract_exception_template(self, content: str) -> str:
        """Extract exception testing template"""
        return '''def test_{function_name}_raises_exception():
    with pytest.raises({exception_type}):
        {function_call}'''
    
    def _extract_mock_template(self, content: str) -> str:
        """Extract mocking template"""
        return '''@patch('{mock_target}')
def test_{function_name}_with_mock(mock_obj):
    # Setup mock
    mock_obj.return_value = {mock_return}
    
    # Execute
    result = {function_call}
    
    # Verify
    assert result == {expected_result}
    mock_obj.assert_called_with({expected_args})'''
    
    def _consolidate_patterns(self, patterns: List[AITestPattern]) -> List[AITestPattern]:
        """Consolidate similar patterns and calculate effectiveness"""
        consolidated = {}
        
        for pattern in patterns:
            key = f"{pattern.pattern_name}_{pattern.pattern_type}"
            if key not in consolidated:
                consolidated[key] = pattern
            else:
                # Merge patterns and update confidence
                existing = consolidated[key]
                existing.confidence_score = max(existing.confidence_score, pattern.confidence_score)
                existing.applicable_functions.extend(pattern.applicable_functions)
                existing.applicable_functions = list(set(existing.applicable_functions))
        
        return list(consolidated.values())

class IntelligentTestGenerator:
    """AI-powered test generator that enhances existing TestDiscoveryEngine"""
    
    def __init__(self):
        self.code_analyzer = CodeAnalysisEngine()
        self.pattern_learner = TestPatternLearner()
        self.feature_discovery_log = FeatureDiscoveryLog()
        self.generated_tests_cache = {}
    
    def enhance_existing_test_discovery(self, existing_test_cases: List[TestCase], 
                                      source_directories: List[str]) -> List[TestCase]:
        """Enhance existing test discovery with AI-generated tests"""
        
        # Log enhancement attempt
        self.feature_discovery_log.log_discovery_attempt(
            "ai_test_generation_enhancement",
            {
                'existing_tests_count': len(existing_test_cases),
                'source_directories': source_directories,
                'enhancement_strategy': 'ENHANCE_EXISTING_DISCOVERY'
            }
        )
        
        enhanced_tests = existing_test_cases.copy()
        
        # Analyze existing test patterns
        existing_test_files = self._find_existing_test_files(source_directories)
        learned_patterns = self.pattern_learner.analyze_existing_tests(existing_test_files)
        
        # Generate additional tests based on coverage gaps
        for source_dir in source_directories:
            source_files = self._find_source_files(source_dir)
            
            for source_file in source_files:
                if self._should_generate_tests_for_file(source_file, existing_test_cases):
                    additional_tests = self._generate_ai_tests_for_file(
                        source_file, learned_patterns, existing_test_cases
                    )
                    enhanced_tests.extend(additional_tests)
        
        return enhanced_tests
    
    def generate_intelligent_test_suite(self, context: TestGenerationContext) -> List[TestCase]:
        """Generate intelligent test suite for a specific function"""
        
        # Analyze function complexity
        complexity_metrics = self.code_analyzer.analyze_function_complexity(context.function_body)
        context.complexity_metrics = complexity_metrics
        
        generated_tests = []
        
        # Generate base functionality tests
        base_tests = self._generate_base_functionality_tests(context)
        generated_tests.extend(base_tests)
        
        # Generate edge case tests based on complexity
        if complexity_metrics.get('complexity_score', 1) > 3:
            edge_case_tests = self._generate_edge_case_tests(context)
            generated_tests.extend(edge_case_tests)
        
        # Generate error condition tests
        if complexity_metrics.get('exception_handling', False):
            error_tests = self._generate_error_condition_tests(context)
            generated_tests.extend(error_tests)
        
        # Generate performance tests for complex functions
        if complexity_metrics.get('cyclomatic_complexity', 1) > 5:
            performance_tests = self._generate_performance_tests(context)
            generated_tests.extend(performance_tests)
        
        return generated_tests
    
    def _find_existing_test_files(self, directories: List[str]) -> List[str]:
        """Find existing test files"""
        test_files = []
        
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists():
                # Common test file patterns
                patterns = ['test_*.py', '*_test.py', 'tests.py']
                for pattern in patterns:
                    test_files.extend(str(f) for f in dir_path.rglob(pattern))
        
        return test_files
    
    def _find_source_files(self, directory: str) -> List[str]:
        """Find source files to analyze"""
        source_files = []
        dir_path = Path(directory)
        
        if dir_path.exists():
            for file_path in dir_path.rglob('*.py'):
                # Skip test files
                if not any(pattern in str(file_path) for pattern in ['test_', '_test.py', 'tests.py']):
                    source_files.append(str(file_path))
        
        return source_files
    
    def _should_generate_tests_for_file(self, source_file: str, existing_tests: List[TestCase]) -> bool:
        """Determine if we should generate tests for a source file"""
        file_path = Path(source_file)
        
        # Check if file already has adequate test coverage
        tested_functions = {test.test_function for test in existing_tests}
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find functions in the file
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            # Check coverage ratio
            coverage_ratio = len([f for f in functions if f in tested_functions]) / len(functions) if functions else 1
            
            return coverage_ratio < 0.8  # Generate tests if coverage is less than 80%
            
        except Exception:
            return False
    
    def _generate_ai_tests_for_file(self, source_file: str, patterns: List[AITestPattern], 
                                  existing_tests: List[TestCase]) -> List[TestCase]:
        """Generate AI tests for a specific file"""
        generated_tests = []
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    # Create generation context
                    context = TestGenerationContext(
                        source_file=source_file,
                        function_name=node.name,
                        function_signature=self._extract_function_signature(node),
                        function_body=ast.unparse(node) if hasattr(ast, 'unparse') else '',
                        existing_tests=[t for t in existing_tests if t.test_function == node.name],
                        dependencies=[],
                        complexity_metrics={}
                    )
                    
                    # Generate tests for this function
                    function_tests = self.generate_intelligent_test_suite(context)
                    generated_tests.extend(function_tests)
        
        except Exception as e:
            self.feature_discovery_log.log_discovery_attempt(
                f"ai_generation_error_{source_file}",
                {'error': str(e), 'file': source_file}
            )
        
        return generated_tests
    
    def _extract_function_signature(self, func_node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        for arg in func_node.args.args:
            args.append(arg.arg)
        
        return f"{func_node.name}({', '.join(args)})"
    
    def _generate_base_functionality_tests(self, context: TestGenerationContext) -> List[TestCase]:
        """Generate basic functionality tests"""
        tests = []
        
        # Happy path test
        test_case = TestCase(
            name=f"test_{context.function_name}_happy_path",
            test_function=f"test_{context.function_name}_happy_path",
            test_file=f"test_{Path(context.source_file).stem}.py",
            test_type=TestType.UNIT,
            description=f"Test {context.function_name} with valid inputs",
            priority=1,
            metadata={'generated_by': 'ai', 'pattern': 'happy_path'}
        )
        tests.append(test_case)
        
        return tests
    
    def _generate_edge_case_tests(self, context: TestGenerationContext) -> List[TestCase]:
        """Generate edge case tests"""
        tests = []
        
        # Boundary value test
        test_case = TestCase(
            name=f"test_{context.function_name}_boundary_values",
            test_function=f"test_{context.function_name}_boundary_values",
            test_file=f"test_{Path(context.source_file).stem}.py",
            test_type=TestType.UNIT,
            description=f"Test {context.function_name} with boundary values",
            priority=2,
            metadata={'generated_by': 'ai', 'pattern': 'boundary_values'}
        )
        tests.append(test_case)
        
        return tests
    
    def _generate_error_condition_tests(self, context: TestGenerationContext) -> List[TestCase]:
        """Generate error condition tests"""
        tests = []
        
        # Exception test
        test_case = TestCase(
            name=f"test_{context.function_name}_error_conditions",
            test_function=f"test_{context.function_name}_error_conditions",
            test_file=f"test_{Path(context.source_file).stem}.py",
            test_type=TestType.UNIT,
            description=f"Test {context.function_name} error handling",
            priority=2,
            metadata={'generated_by': 'ai', 'pattern': 'error_conditions'}
        )
        tests.append(test_case)
        
        return tests
    
    def _generate_performance_tests(self, context: TestGenerationContext) -> List[TestCase]:
        """Generate performance tests"""
        tests = []
        
        # Performance test
        test_case = TestCase(
            name=f"test_{context.function_name}_performance",
            test_function=f"test_{context.function_name}_performance",
            test_file=f"test_{Path(context.source_file).stem}.py",
            test_type=TestType.PERFORMANCE,
            description=f"Test {context.function_name} performance characteristics",
            priority=3,
            timeout=60,
            metadata={'generated_by': 'ai', 'pattern': 'performance'}
        )
        tests.append(test_case)
        
        return tests

def main():
    """Example usage of AI Test Generator"""
    print("🤖 AI-Powered Test Generator - Enhancement Mode")
    print("=" * 60)
    
    # Create AI test generator
    ai_generator = IntelligentTestGenerator()
    
    # Example: Enhance existing test discovery
    existing_tests = []  # Would come from existing TestDiscoveryEngine
    source_dirs = ["./TestMaster", "./tests"]
    
    enhanced_tests = ai_generator.enhance_existing_test_discovery(existing_tests, source_dirs)
    
    print(f"Enhanced test discovery:")
    print(f"Original tests: {len(existing_tests)}")
    print(f"Enhanced tests: {len(enhanced_tests)}")
    print(f"AI-generated tests: {len(enhanced_tests) - len(existing_tests)}")

if __name__ == "__main__":
    main()