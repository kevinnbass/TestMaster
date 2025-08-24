#!/usr/bin/env python3
"""
Testing Framework Analysis Tool - Agent C Hours 44-46
Analyzes testing patterns, framework usage, and test coverage across the codebase.
Identifies testing consolidation opportunities and quality metrics.
"""

import ast
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Optional, Tuple
import sys

class TestingFrameworkAnalyzer(ast.NodeVisitor):
    """Analyzes testing frameworks and patterns across the codebase."""
    
    def __init__(self):
        self.test_files = []
        self.test_functions = []
        self.test_classes = []
        self.testing_frameworks = defaultdict(int)
        self.assertion_patterns = defaultdict(int)
        self.mock_usage = defaultdict(int)
        self.test_patterns = defaultdict(list)
        self.coverage_indicators = []
        self.test_metrics = {}
        self.current_file = None
        self.framework_imports = defaultdict(set)
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for testing patterns."""
        self.current_file = str(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check if this is a test file
            is_test_file = self._is_test_file(file_path, content)
            
            if is_test_file:
                self.test_files.append({
                    'file': str(file_path),
                    'type': self._determine_test_type(file_path, content),
                    'framework': self._detect_testing_framework(content),
                    'patterns': self._extract_test_patterns(content)
                })
            
            # Parse AST for testing patterns
            tree = ast.parse(content, filename=str(file_path))
            self.visit(tree)
            
            # Analyze raw content for additional patterns
            self._analyze_raw_content(content)
            
            return {
                'file': str(file_path),
                'is_test_file': is_test_file,
                'analyzed': True
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'analyzed': False
            }
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements to detect testing frameworks."""
        for alias in node.names:
            self._process_testing_import(alias.name, alias.asname)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statements to detect testing frameworks."""
        module = node.module or ''
        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name
            self._process_testing_import(full_name, alias.asname)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to find test classes."""
        if self._is_test_class(node):
            test_class_info = {
                'name': node.name,
                'file': self.current_file,
                'line': node.lineno,
                'methods': self._extract_test_methods(node),
                'base_classes': [base.id for base in node.bases if isinstance(base, ast.Name)],
                'decorators': [dec.id for dec in node.decorator_list if isinstance(dec, ast.Name)]
            }
            self.test_classes.append(test_class_info)
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to find test functions."""
        if self._is_test_function(node):
            test_func_info = {
                'name': node.name,
                'file': self.current_file,
                'line': node.lineno,
                'args': [arg.arg for arg in node.args.args],
                'decorators': [dec.id for dec in node.decorator_list if isinstance(dec, ast.Name)],
                'assertions': self._count_assertions(node),
                'mocks': self._count_mocks(node),
                'complexity': self._calculate_test_complexity(node)
            }
            self.test_functions.append(test_func_info)
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to detect assertion and mock patterns."""
        if isinstance(node.func, ast.Attribute):
            # Assertion patterns
            if node.func.attr.startswith('assert'):
                self.assertion_patterns[node.func.attr] += 1
            
            # Mock patterns
            if any(mock_word in node.func.attr.lower() for mock_word in ['mock', 'patch', 'spy']):
                self.mock_usage[node.func.attr] += 1
        
        elif isinstance(node.func, ast.Name):
            # Direct assertion calls
            if node.func.id.startswith('assert'):
                self.assertion_patterns[node.func.id] += 1
        
        self.generic_visit(node)
    
    def _is_test_file(self, file_path: Path, content: str) -> bool:
        """Determine if file is a test file."""
        filename = file_path.name.lower()
        
        # Check filename patterns
        test_patterns = [
            filename.startswith('test_'),
            filename.endswith('_test.py'),
            'test' in filename and filename.endswith('.py'),
            '/test' in str(file_path).lower(),
            '/tests/' in str(file_path).lower()
        ]
        
        if any(test_patterns):
            return True
        
        # Check content patterns
        content_patterns = [
            'import unittest',
            'import pytest',
            'from unittest',
            'from pytest',
            'def test_',
            'class Test',
            'assert ',
            '@pytest.mark',
            '@unittest',
            'TestCase'
        ]
        
        for pattern in content_patterns:
            if pattern in content:
                return True
        
        return False
    
    def _determine_test_type(self, file_path: Path, content: str) -> str:
        """Determine the type of test file."""
        if 'unittest' in content:
            return 'unittest'
        elif 'pytest' in content:
            return 'pytest'
        elif 'nose' in content:
            return 'nose'
        elif 'doctest' in content:
            return 'doctest'
        elif 'hypothesis' in content:
            return 'hypothesis'
        elif 'mock' in content or 'patch' in content:
            return 'mock_test'
        elif 'integration' in str(file_path).lower():
            return 'integration_test'
        elif 'unit' in str(file_path).lower():
            return 'unit_test'
        elif 'functional' in str(file_path).lower():
            return 'functional_test'
        else:
            return 'unknown_test'
    
    def _detect_testing_framework(self, content: str) -> List[str]:
        """Detect testing frameworks used in the content."""
        frameworks = []
        
        framework_patterns = {
            'unittest': ['import unittest', 'from unittest', 'TestCase'],
            'pytest': ['import pytest', 'from pytest', '@pytest.mark'],
            'nose': ['import nose', 'from nose', 'nose.tools'],
            'doctest': ['import doctest', 'doctest.testmod'],
            'hypothesis': ['import hypothesis', 'from hypothesis'],
            'mock': ['import mock', 'from mock', 'unittest.mock'],
            'faker': ['import faker', 'from faker'],
            'factory_boy': ['import factory', 'from factory'],
            'responses': ['import responses', '@responses.activate']
        }
        
        for framework, patterns in framework_patterns.items():
            if any(pattern in content for pattern in patterns):
                frameworks.append(framework)
                self.testing_frameworks[framework] += 1
        
        return frameworks
    
    def _extract_test_patterns(self, content: str) -> List[str]:
        """Extract test patterns from content."""
        patterns = []
        
        # Fixture patterns
        if '@pytest.fixture' in content:
            patterns.append('pytest_fixtures')
        if 'setUp' in content or 'tearDown' in content:
            patterns.append('unittest_setup_teardown')
        
        # Parametrized tests
        if '@pytest.mark.parametrize' in content:
            patterns.append('parametrized_tests')
        
        # Mock patterns
        if '@patch' in content or 'mock.patch' in content:
            patterns.append('mock_decorators')
        if 'MagicMock' in content or 'Mock(' in content:
            patterns.append('mock_objects')
        
        # Skip patterns
        if '@pytest.mark.skip' in content or '@unittest.skip' in content:
            patterns.append('skipped_tests')
        
        # Async test patterns
        if 'async def test_' in content:
            patterns.append('async_tests')
        
        # Property-based testing
        if '@given' in content:
            patterns.append('property_based_tests')
        
        return patterns
    
    def _process_testing_import(self, module_name: str, alias: Optional[str]):
        """Process testing-related imports."""
        testing_modules = [
            'unittest', 'pytest', 'nose', 'doctest', 'hypothesis',
            'mock', 'faker', 'factory', 'responses', 'testfixtures'
        ]
        
        base_module = module_name.split('.')[0]
        if base_module in testing_modules:
            self.framework_imports[base_module].add(self.current_file)
    
    def _is_test_class(self, node: ast.ClassDef) -> bool:
        """Check if class is a test class."""
        class_indicators = [
            node.name.startswith('Test'),
            node.name.endswith('Test'),
            'test' in node.name.lower(),
            any(base.id == 'TestCase' for base in node.bases if isinstance(base, ast.Name))
        ]
        
        return any(class_indicators)
    
    def _is_test_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is a test function."""
        function_indicators = [
            node.name.startswith('test_'),
            node.name.startswith('Test'),
            any(dec.id == 'test' for dec in node.decorator_list if isinstance(dec, ast.Name))
        ]
        
        return any(function_indicators)
    
    def _extract_test_methods(self, node: ast.ClassDef) -> List[str]:
        """Extract test method names from a test class."""
        methods = []
        
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                if self._is_test_function(child):
                    methods.append(child.name)
        
        return methods
    
    def _count_assertions(self, node: ast.FunctionDef) -> int:
        """Count assertion statements in a test function."""
        assertion_count = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                assertion_count += 1
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr.startswith('assert'):
                        assertion_count += 1
                elif isinstance(child.func, ast.Name):
                    if child.func.id.startswith('assert'):
                        assertion_count += 1
        
        return assertion_count
    
    def _count_mocks(self, node: ast.FunctionDef) -> int:
        """Count mock usage in a test function."""
        mock_count = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if any(mock_word in child.func.attr.lower() 
                           for mock_word in ['mock', 'patch', 'spy']):
                        mock_count += 1
                elif isinstance(child.func, ast.Name):
                    if any(mock_word in child.func.id.lower() 
                           for mock_word in ['mock', 'patch', 'spy']):
                        mock_count += 1
        
        return mock_count
    
    def _calculate_test_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity of a test function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def _analyze_raw_content(self, content: str) -> None:
        """Analyze raw file content for testing patterns."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Coverage indicators
            if re.search(r'#.*pragma.*no.*cover', line_stripped):
                self.coverage_indicators.append({
                    'type': 'pragma_no_cover',
                    'file': self.current_file,
                    'line': i + 1
                })
            
            # Test configuration patterns
            if re.search(r'pytest\.ini|setup\.cfg|tox\.ini', line_stripped):
                self.test_patterns['config_files'].append({
                    'file': self.current_file,
                    'line': i + 1,
                    'type': 'test_config'
                })
    
    def analyze_directory(self, root_path: Path) -> Dict[str, Any]:
        """Analyze all Python files in directory for testing patterns."""
        results = {
            'files_analyzed': [],
            'total_files': 0,
            'successful_analyses': 0,
            'errors': []
        }
        
        python_files = list(root_path.rglob('*.py'))
        results['total_files'] = len(python_files)
        
        print(f"Analyzing {len(python_files)} Python files for testing patterns...")
        
        for i, file_path in enumerate(python_files):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(python_files)} files analyzed")
            
            analysis = self.analyze_file(file_path)
            results['files_analyzed'].append(analysis)
            
            if analysis.get('analyzed', False):
                results['successful_analyses'] += 1
            else:
                results['errors'].append(analysis)
        
        print(f"Analysis complete: {results['successful_analyses']}/{results['total_files']} files analyzed successfully")
        return results
    
    def detect_testing_opportunities(self) -> Dict[str, Any]:
        """Detect testing improvement and consolidation opportunities."""
        opportunities = {
            'framework_consolidation': self._analyze_framework_consolidation(),
            'test_coverage_gaps': self._identify_coverage_gaps(),
            'test_quality_issues': self._identify_quality_issues(),
            'duplicate_test_patterns': self._find_duplicate_test_patterns(),
            'testing_best_practices': self._analyze_best_practices()
        }
        
        return opportunities
    
    def _analyze_framework_consolidation(self) -> Dict[str, Any]:
        """Analyze opportunities for testing framework consolidation."""
        framework_usage = dict(self.testing_frameworks)
        
        consolidation_analysis = {
            'frameworks_in_use': len(framework_usage),
            'framework_distribution': framework_usage,
            'consolidation_recommendation': 'maintain_current' if len(framework_usage) <= 2 else 'consolidate',
            'primary_framework': max(framework_usage.items(), key=lambda x: x[1])[0] if framework_usage else None
        }
        
        if len(framework_usage) > 2:
            consolidation_analysis['consolidation_benefit'] = f"Reduce from {len(framework_usage)} to 1-2 frameworks"
        
        return consolidation_analysis
    
    def _identify_coverage_gaps(self) -> List[Dict[str, Any]]:
        """Identify potential test coverage gaps."""
        gaps = []
        
        # Files with no coverage indicators
        no_coverage_files = []
        for test_file in self.test_files:
            file_coverage = [c for c in self.coverage_indicators if c['file'] == test_file['file']]
            if not file_coverage:
                no_coverage_files.append(test_file['file'])
        
        if no_coverage_files:
            gaps.append({
                'type': 'no_coverage_indicators',
                'count': len(no_coverage_files),
                'files': no_coverage_files[:10]  # Limit for readability
            })
        
        # Test functions with no assertions
        no_assertion_tests = [tf for tf in self.test_functions if tf['assertions'] == 0]
        if no_assertion_tests:
            gaps.append({
                'type': 'tests_without_assertions',
                'count': len(no_assertion_tests),
                'examples': [f"{t['file']}::{t['name']}" for t in no_assertion_tests[:5]]
            })
        
        return gaps
    
    def _identify_quality_issues(self) -> List[Dict[str, Any]]:
        """Identify test quality issues."""
        issues = []
        
        # Complex test functions
        complex_tests = [tf for tf in self.test_functions if tf['complexity'] > 5]
        if complex_tests:
            issues.append({
                'type': 'complex_test_functions',
                'count': len(complex_tests),
                'examples': [f"{t['file']}::{t['name']} (complexity: {t['complexity']})" 
                           for t in complex_tests[:5]]
            })
        
        # Tests with too many assertions
        assertion_heavy_tests = [tf for tf in self.test_functions if tf['assertions'] > 10]
        if assertion_heavy_tests:
            issues.append({
                'type': 'assertion_heavy_tests',
                'count': len(assertion_heavy_tests),
                'examples': [f"{t['file']}::{t['name']} ({t['assertions']} assertions)" 
                           for t in assertion_heavy_tests[:5]]
            })
        
        return issues
    
    def _find_duplicate_test_patterns(self) -> List[Dict[str, Any]]:
        """Find duplicate test patterns that could be consolidated."""
        pattern_groups = defaultdict(list)
        
        # Group tests by similar names
        for test_func in self.test_functions:
            # Extract base pattern from test name
            base_pattern = re.sub(r'_\d+$|_test$', '', test_func['name'])
            pattern_groups[base_pattern].append(test_func)
        
        duplicates = []
        for pattern, tests in pattern_groups.items():
            if len(tests) > 3:  # More than 3 similar tests
                duplicates.append({
                    'pattern': pattern,
                    'test_count': len(tests),
                    'files': list(set(t['file'] for t in tests)),
                    'consolidation_opportunity': 'parametrize_tests'
                })
        
        return sorted(duplicates, key=lambda x: x['test_count'], reverse=True)[:10]
    
    def _analyze_best_practices(self) -> Dict[str, Any]:
        """Analyze adherence to testing best practices."""
        practices = {
            'fixture_usage': sum(1 for tf in self.test_files if 'pytest_fixtures' in tf.get('patterns', [])),
            'mock_usage': len([tf for tf in self.test_functions if tf['mocks'] > 0]),
            'parametrized_tests': sum(1 for tf in self.test_files if 'parametrized_tests' in tf.get('patterns', [])),
            'async_test_coverage': sum(1 for tf in self.test_files if 'async_tests' in tf.get('patterns', [])),
            'setup_teardown_usage': sum(1 for tf in self.test_files if 'unittest_setup_teardown' in tf.get('patterns', []))
        }
        
        recommendations = []
        if practices['fixture_usage'] < len(self.test_files) * 0.3:
            recommendations.append("Consider using more fixtures for test setup")
        
        if practices['parametrized_tests'] == 0:
            recommendations.append("Consider using parametrized tests to reduce duplication")
        
        if practices['mock_usage'] < len(self.test_functions) * 0.2:
            recommendations.append("Consider using more mocks for unit test isolation")
        
        return {
            'practice_metrics': practices,
            'recommendations': recommendations
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive testing analysis summary."""
        testing_opportunities = self.detect_testing_opportunities()
        
        summary = {
            'testing_statistics': {
                'total_test_files': len(self.test_files),
                'total_test_functions': len(self.test_functions),
                'total_test_classes': len(self.test_classes),
                'testing_frameworks': dict(self.testing_frameworks),
                'assertion_patterns': dict(self.assertion_patterns),
                'mock_usage_patterns': dict(self.mock_usage),
                'average_assertions_per_test': (sum(tf['assertions'] for tf in self.test_functions) / 
                                               len(self.test_functions)) if self.test_functions else 0,
                'average_test_complexity': (sum(tf['complexity'] for tf in self.test_functions) / 
                                          len(self.test_functions)) if self.test_functions else 0
            },
            'framework_analysis': testing_opportunities['framework_consolidation'],
            'quality_analysis': {
                'coverage_gaps': len(testing_opportunities['test_coverage_gaps']),
                'quality_issues': len(testing_opportunities['test_quality_issues']),
                'duplicate_patterns': len(testing_opportunities['duplicate_test_patterns'])
            },
            'testing_opportunities': testing_opportunities,
            'testing_health_score': self._calculate_testing_health_score(),
            'recommendations': self._generate_testing_recommendations()
        }
        
        return summary
    
    def _calculate_testing_health_score(self) -> float:
        """Calculate overall testing health score (0-100)."""
        if not self.test_functions:
            return 0.0
        
        score = 100.0
        
        # Penalize complex tests
        complex_tests = len([tf for tf in self.test_functions if tf['complexity'] > 5])
        if complex_tests > 0:
            score -= min(complex_tests * 2, 20)
        
        # Penalize tests without assertions
        no_assertion_tests = len([tf for tf in self.test_functions if tf['assertions'] == 0])
        if no_assertion_tests > 0:
            score -= min(no_assertion_tests * 5, 30)
        
        # Bonus for good assertion coverage
        avg_assertions = sum(tf['assertions'] for tf in self.test_functions) / len(self.test_functions)
        if 1 <= avg_assertions <= 5:
            score += 10
        
        # Bonus for mock usage
        mock_usage_ratio = len([tf for tf in self.test_functions if tf['mocks'] > 0]) / len(self.test_functions)
        if mock_usage_ratio > 0.2:
            score += 10
        
        # Penalize too many testing frameworks
        if len(self.testing_frameworks) > 2:
            score -= (len(self.testing_frameworks) - 2) * 5
        
        return max(score, 0)
    
    def _generate_testing_recommendations(self) -> List[str]:
        """Generate testing improvement recommendations."""
        recommendations = []
        
        if len(self.testing_frameworks) > 2:
            recommendations.append(f"Consolidate from {len(self.testing_frameworks)} testing frameworks to 1-2")
        
        no_assertion_tests = len([tf for tf in self.test_functions if tf['assertions'] == 0])
        if no_assertion_tests > 0:
            recommendations.append(f"Add assertions to {no_assertion_tests} test functions")
        
        complex_tests = len([tf for tf in self.test_functions if tf['complexity'] > 5])
        if complex_tests > 0:
            recommendations.append(f"Simplify {complex_tests} overly complex test functions")
        
        if len(self.test_files) == 0:
            recommendations.append("Add test files to improve code coverage")
        
        mock_usage_ratio = (len([tf for tf in self.test_functions if tf['mocks'] > 0]) / 
                           len(self.test_functions)) if self.test_functions else 0
        if mock_usage_ratio < 0.1:
            recommendations.append("Consider using more mocks for better unit test isolation")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Testing Framework Analysis Tool')
    parser.add_argument('--root', type=str, required=True, help='Root directory to analyze')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    print("=== Agent C Hours 44-46: Testing Framework Analysis ===")
    print(f"Analyzing directory: {args.root}")
    
    analyzer = TestingFrameworkAnalyzer()
    root_path = Path(args.root)
    
    # Analyze directory
    analysis_results = analyzer.analyze_directory(root_path)
    
    # Generate summary
    summary = analyzer.generate_summary()
    
    # Combine results
    final_results = {
        'analysis_metadata': {
            'tool': 'testing_framework_analyzer',
            'version': '1.0',
            'agent': 'Agent_C',
            'hours': '44-46',
            'phase': 'Utility_Component_Extraction'
        },
        'analysis_results': analysis_results,
        'summary': summary,
        'raw_data': {
            'test_files': analyzer.test_files,
            'test_functions': analyzer.test_functions,
            'test_classes': analyzer.test_classes,
            'testing_frameworks': dict(analyzer.testing_frameworks),
            'assertion_patterns': dict(analyzer.assertion_patterns),
            'mock_usage': dict(analyzer.mock_usage),
            'framework_imports': {k: list(v) for k, v in analyzer.framework_imports.items()}
        }
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== TESTING FRAMEWORK ANALYSIS COMPLETE ===")
    print(f"Files analyzed: {analysis_results['successful_analyses']}/{analysis_results['total_files']}")
    print(f"Test files found: {summary['testing_statistics']['total_test_files']}")
    print(f"Test functions: {summary['testing_statistics']['total_test_functions']}")
    print(f"Test classes: {summary['testing_statistics']['total_test_classes']}")
    print(f"Testing frameworks: {len(summary['testing_statistics']['testing_frameworks'])}")
    print(f"Testing health score: {summary['testing_health_score']:.1f}/100")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()