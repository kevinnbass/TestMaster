"""
Testing and Testability Analysis Module
========================================

Implements comprehensive testing and testability analysis:
- Test coverage potential identification
- Test pyramid analysis (unit/integration/e2e)
- Mock/stub dependency analysis
- Test smell detection
- Mutation testing readiness
- Property-based testing opportunities
- Test data pattern analysis
- Flaky test prediction
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class TestingAnalyzer(BaseAnalyzer):
    """Analyzer for testing patterns and testability."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive testing and testability analysis."""
        print("[INFO] Analyzing Testing and Testability...")
        
        results = {
            "test_coverage_potential": self._analyze_test_coverage_potential(),
            "test_pyramid": self._analyze_test_pyramid(),
            "mock_dependency": self._analyze_mock_dependencies(),
            "test_smells": self._detect_test_smells(),
            "mutation_readiness": self._assess_mutation_testing_readiness(),
            "property_testing": self._identify_property_testing_opportunities(),
            "test_data_patterns": self._analyze_test_data_patterns(),
            "flaky_test_prediction": self._predict_flaky_tests(),
            "testability_metrics": self._calculate_testability_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} testing aspects")
        return results
    
    def _analyze_test_coverage_potential(self) -> Dict[str, Any]:
        """Identify untested code paths and edge cases."""
        coverage_potential = {
            "untested_functions": [],
            "untested_branches": [],
            "untested_exceptions": [],
            "edge_cases": [],
            "boundary_conditions": [],
            "integration_points": []
        }
        
        # Find test files
        test_files = self._get_test_files()
        tested_functions = self._get_tested_functions(test_files)
        
        # Analyze source files
        for py_file in self._get_python_files():
            if self._is_test_file(py_file):
                continue
                
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_full_name = f"{file_key}::{node.name}"
                        
                        # Check if function is tested
                        if func_full_name not in tested_functions and node.name not in tested_functions:
                            coverage_potential["untested_functions"].append({
                                "function": node.name,
                                "file": file_key,
                                "line": node.lineno,
                                "complexity": self._calculate_function_complexity(node),
                                "priority": self._calculate_test_priority(node)
                            })
                        
                        # Analyze branches
                        branches = self._analyze_branches(node)
                        for branch in branches:
                            branch["file"] = file_key
                            branch["function"] = node.name
                            coverage_potential["untested_branches"].append(branch)
                        
                        # Analyze exception paths
                        exceptions = self._analyze_exception_paths(node)
                        for exc in exceptions:
                            exc["file"] = file_key
                            exc["function"] = node.name
                            coverage_potential["untested_exceptions"].append(exc)
                        
                        # Identify edge cases
                        edge_cases = self._identify_edge_cases(node)
                        for edge in edge_cases:
                            edge["file"] = file_key
                            edge["function"] = node.name
                            coverage_potential["edge_cases"].append(edge)
                        
                        # Identify boundary conditions
                        boundaries = self._identify_boundary_conditions(node)
                        for boundary in boundaries:
                            boundary["file"] = file_key
                            boundary["function"] = node.name
                            coverage_potential["boundary_conditions"].append(boundary)
                
                # Identify integration points
                integrations = self._identify_integration_points(tree)
                for integration in integrations:
                    integration["file"] = file_key
                    coverage_potential["integration_points"].append(integration)
                
            except:
                continue
        
        return coverage_potential
    
    def _get_test_files(self) -> List[Path]:
        """Get all test files in the project."""
        test_files = []
        
        # Common test file patterns
        test_patterns = ['test_*.py', '*_test.py', 'tests.py']
        test_dirs = ['tests', 'test', 'testing']
        
        # Find test directories
        for test_dir in test_dirs:
            test_path = self.base_path / test_dir
            if test_path.exists() and test_path.is_dir():
                test_files.extend(test_path.rglob('*.py'))
        
        # Find test files in main directory
        for pattern in test_patterns:
            test_files.extend(self.base_path.rglob(pattern))
        
        return test_files
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if a file is a test file."""
        name = file_path.name.lower()
        return 'test' in name or file_path.parent.name in ['tests', 'test', 'testing']
    
    def _get_tested_functions(self, test_files: List[Path]) -> Set[str]:
        """Extract functions that are being tested."""
        tested = set()
        
        for test_file in test_files:
            try:
                content = self._get_file_content(test_file)
                tree = self._get_ast(test_file)
                
                # Look for test function names
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name.startswith('test_'):
                            # Extract function being tested from name
                            tested_func = node.name.replace('test_', '')
                            tested.add(tested_func)
                    
                    # Look for function calls in tests
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            tested.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            tested.add(node.func.attr)
                
                # Look for imports to understand what's being tested
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            tested.add(alias.name)
                
            except:
                continue
        
        return tested
    
    def _analyze_branches(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Analyze branch coverage opportunities."""
        branches = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                branches.append({
                    "type": "if_branch",
                    "line": node.lineno,
                    "conditions": self._count_conditions(node.test)
                })
            elif isinstance(node, ast.While):
                branches.append({
                    "type": "while_loop",
                    "line": node.lineno,
                    "needs_zero_iteration_test": True,
                    "needs_multiple_iteration_test": True
                })
            elif isinstance(node, ast.For):
                branches.append({
                    "type": "for_loop",
                    "line": node.lineno,
                    "needs_empty_iteration_test": True
                })
        
        return branches
    
    def _count_conditions(self, test_node: ast.AST) -> int:
        """Count logical conditions in a test expression."""
        count = 1
        for node in ast.walk(test_node):
            if isinstance(node, ast.BoolOp):
                count += len(node.values) - 1
        return count
    
    def _analyze_exception_paths(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Analyze exception handling paths."""
        exceptions = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    exc_type = None
                    if handler.type:
                        if isinstance(handler.type, ast.Name):
                            exc_type = handler.type.id
                        elif isinstance(handler.type, ast.Tuple):
                            exc_type = "multiple"
                    
                    exceptions.append({
                        "type": exc_type or "generic",
                        "line": handler.lineno,
                        "has_else": len(node.orelse) > 0,
                        "has_finally": len(node.finalbody) > 0
                    })
            elif isinstance(node, ast.Raise):
                exceptions.append({
                    "type": "raised",
                    "line": node.lineno,
                    "needs_test": True
                })
        
        return exceptions
    
    def _identify_edge_cases(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Identify potential edge cases."""
        edge_cases = []
        
        # Check for None checks
        for node in ast.walk(func_node):
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, ast.Is) and isinstance(comparator, ast.Constant) and comparator.value is None:
                        edge_cases.append({
                            "type": "none_check",
                            "line": node.lineno,
                            "test_suggestion": "Test with None input"
                        })
        
        # Check for empty collection checks
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Call):
                    if isinstance(node.test.func, ast.Name) and node.test.func.id == 'len':
                        edge_cases.append({
                            "type": "empty_collection",
                            "line": node.lineno,
                            "test_suggestion": "Test with empty collection"
                        })
        
        return edge_cases
    
    def _identify_boundary_conditions(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Identify boundary condition test opportunities."""
        boundaries = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                        boundaries.append({
                            "type": "numeric_boundary",
                            "line": node.lineno,
                            "operator": op.__class__.__name__,
                            "test_suggestions": [
                                "Test at boundary value",
                                "Test just below boundary",
                                "Test just above boundary"
                            ]
                        })
        
        # Check for array/list indexing
        for node in ast.walk(func_node):
            if isinstance(node, ast.Subscript):
                boundaries.append({
                    "type": "index_boundary",
                    "line": node.lineno,
                    "test_suggestions": [
                        "Test with index 0",
                        "Test with index -1",
                        "Test with index out of bounds"
                    ]
                })
        
        return boundaries
    
    def _identify_integration_points(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify integration test opportunities."""
        integration_points = []
        
        for node in ast.walk(tree):
            # Database operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['execute', 'query', 'save', 'delete', 'update']:
                        integration_points.append({
                            "type": "database",
                            "operation": node.func.attr,
                            "line": node.lineno
                        })
            
            # HTTP calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['get', 'post', 'put', 'delete', 'patch', 'request']:
                        integration_points.append({
                            "type": "http",
                            "method": node.func.attr,
                            "line": node.lineno
                        })
            
            # File I/O
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    integration_points.append({
                        "type": "file_io",
                        "line": node.lineno
                    })
        
        return integration_points
    
    def _calculate_test_priority(self, func_node: ast.FunctionDef) -> str:
        """Calculate testing priority based on complexity and risk."""
        complexity = self._calculate_function_complexity(func_node)
        
        # Check for critical operations
        has_db_ops = any(isinstance(n, ast.Call) and 
                         isinstance(n.func, ast.Attribute) and 
                         n.func.attr in ['execute', 'query', 'save']
                         for n in ast.walk(func_node))
        
        has_security_ops = any(isinstance(n, ast.Call) and
                              isinstance(n.func, ast.Name) and
                              n.func.id in ['eval', 'exec', 'compile']
                              for n in ast.walk(func_node))
        
        if has_security_ops or complexity > 15:
            return "CRITICAL"
        elif has_db_ops or complexity > 10:
            return "HIGH"
        elif complexity > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _analyze_test_pyramid(self) -> Dict[str, Any]:
        """Analyze test distribution across unit/integration/e2e."""
        pyramid_analysis = {
            "unit_tests": [],
            "integration_tests": [],
            "e2e_tests": [],
            "test_distribution": {},
            "pyramid_health": None,
            "recommendations": []
        }
        
        test_files = self._get_test_files()
        
        for test_file in test_files:
            try:
                tree = self._get_ast(test_file)
                content = self._get_file_content(test_file)
                file_key = str(test_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_type = self._classify_test_type(node, content)
                        test_info = {
                            "name": node.name,
                            "file": file_key,
                            "line": node.lineno,
                            "execution_time_estimate": self._estimate_test_execution_time(node)
                        }
                        
                        if test_type == "unit":
                            pyramid_analysis["unit_tests"].append(test_info)
                        elif test_type == "integration":
                            pyramid_analysis["integration_tests"].append(test_info)
                        else:  # e2e
                            pyramid_analysis["e2e_tests"].append(test_info)
                
            except:
                continue
        
        # Calculate distribution
        total_tests = (len(pyramid_analysis["unit_tests"]) + 
                      len(pyramid_analysis["integration_tests"]) + 
                      len(pyramid_analysis["e2e_tests"]))
        
        if total_tests > 0:
            pyramid_analysis["test_distribution"] = {
                "unit": len(pyramid_analysis["unit_tests"]) / total_tests * 100,
                "integration": len(pyramid_analysis["integration_tests"]) / total_tests * 100,
                "e2e": len(pyramid_analysis["e2e_tests"]) / total_tests * 100
            }
            
            # Assess pyramid health (ideal: 70% unit, 20% integration, 10% e2e)
            unit_pct = pyramid_analysis["test_distribution"]["unit"]
            integration_pct = pyramid_analysis["test_distribution"]["integration"]
            e2e_pct = pyramid_analysis["test_distribution"]["e2e"]
            
            if unit_pct >= 60 and integration_pct <= 30 and e2e_pct <= 15:
                pyramid_analysis["pyramid_health"] = "HEALTHY"
            elif unit_pct >= 40:
                pyramid_analysis["pyramid_health"] = "MODERATE"
            else:
                pyramid_analysis["pyramid_health"] = "UNHEALTHY"
            
            # Generate recommendations
            if unit_pct < 60:
                pyramid_analysis["recommendations"].append("Increase unit test coverage")
            if integration_pct > 30:
                pyramid_analysis["recommendations"].append("Consider converting some integration tests to unit tests")
            if e2e_pct > 15:
                pyramid_analysis["recommendations"].append("Reduce e2e tests to improve test suite speed")
        
        return pyramid_analysis
    
    def _classify_test_type(self, test_node: ast.FunctionDef, content: str) -> str:
        """Classify test as unit, integration, or e2e."""
        # Check for mocking - usually indicates unit test
        has_mocks = any(isinstance(n, ast.Call) and 
                        isinstance(n.func, ast.Attribute) and
                        'mock' in str(n.func.attr).lower()
                        for n in ast.walk(test_node))
        
        if has_mocks:
            return "unit"
        
        # Check for database/network operations - indicates integration
        has_db = any('db' in str(n).lower() or 'database' in str(n).lower()
                    for n in ast.walk(test_node))
        has_network = any('request' in str(n).lower() or 'http' in str(n).lower()
                         for n in ast.walk(test_node))
        
        if has_db or has_network:
            return "integration"
        
        # Check for browser/selenium - indicates e2e
        if 'selenium' in content.lower() or 'browser' in content.lower():
            return "e2e"
        
        # Check for fixtures/setup complexity
        setup_complexity = sum(1 for n in ast.walk(test_node) 
                              if isinstance(n, ast.Call) and
                              isinstance(n.func, ast.Name) and
                              'setup' in n.func.id.lower())
        
        if setup_complexity > 2:
            return "integration"
        
        return "unit"
    
    def _estimate_test_execution_time(self, test_node: ast.FunctionDef) -> str:
        """Estimate test execution time."""
        # Count operations that typically take time
        db_ops = sum(1 for n in ast.walk(test_node)
                    if isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Attribute) and
                    n.func.attr in ['execute', 'query', 'save'])
        
        network_ops = sum(1 for n in ast.walk(test_node)
                         if isinstance(n, ast.Call) and
                         isinstance(n.func, ast.Attribute) and
                         n.func.attr in ['get', 'post', 'request'])
        
        sleep_ops = sum(1 for n in ast.walk(test_node)
                       if isinstance(n, ast.Call) and
                       isinstance(n.func, ast.Name) and
                       n.func.id == 'sleep')
        
        total_time_ops = db_ops + network_ops + (sleep_ops * 10)
        
        if total_time_ops > 5:
            return "slow"
        elif total_time_ops > 1:
            return "moderate"
        else:
            return "fast"
    
    def _analyze_mock_dependencies(self) -> Dict[str, Any]:
        """Analyze mock and stub usage patterns."""
        mock_analysis = {
            "mock_usage": [],
            "over_mocking": [],
            "mock_frameworks": set(),
            "test_doubles": defaultdict(int),
            "mock_smells": []
        }
        
        test_files = self._get_test_files()
        
        for test_file in test_files:
            try:
                tree = self._get_ast(test_file)
                content = self._get_file_content(test_file)
                file_key = str(test_file.relative_to(self.base_path))
                
                # Detect mock framework
                if 'unittest.mock' in content:
                    mock_analysis["mock_frameworks"].add("unittest.mock")
                if 'pytest' in content and 'monkeypatch' in content:
                    mock_analysis["mock_frameworks"].add("pytest-monkeypatch")
                if 'mock' in content:
                    mock_analysis["mock_frameworks"].add("mock")
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        mock_count = self._count_mocks_in_test(node)
                        
                        if mock_count > 0:
                            mock_analysis["mock_usage"].append({
                                "test": node.name,
                                "file": file_key,
                                "mock_count": mock_count,
                                "line": node.lineno
                            })
                            
                            # Check for over-mocking
                            if mock_count > 5:
                                mock_analysis["over_mocking"].append({
                                    "test": node.name,
                                    "file": file_key,
                                    "mock_count": mock_count,
                                    "recommendation": "Consider reducing mock count or splitting test"
                                })
                        
                        # Classify test double types
                        double_types = self._classify_test_doubles(node)
                        for double_type in double_types:
                            mock_analysis["test_doubles"][double_type] += 1
                        
                        # Detect mock smells
                        smells = self._detect_mock_smells(node)
                        for smell in smells:
                            smell["test"] = node.name
                            smell["file"] = file_key
                            mock_analysis["mock_smells"].append(smell)
                
            except:
                continue
        
        mock_analysis["mock_frameworks"] = list(mock_analysis["mock_frameworks"])
        return mock_analysis
    
    def _count_mocks_in_test(self, test_node: ast.FunctionDef) -> int:
        """Count mock objects in a test."""
        mock_count = 0
        
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if 'mock' in node.func.id.lower() or 'patch' in node.func.id.lower():
                        mock_count += 1
                elif isinstance(node.func, ast.Attribute):
                    if 'mock' in node.func.attr.lower() or 'patch' in node.func.attr.lower():
                        mock_count += 1
        
        return mock_count
    
    def _classify_test_doubles(self, test_node: ast.FunctionDef) -> List[str]:
        """Classify types of test doubles used."""
        doubles = []
        
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    name_lower = node.func.id.lower()
                    if 'mock' in name_lower:
                        doubles.append("mock")
                    elif 'stub' in name_lower:
                        doubles.append("stub")
                    elif 'fake' in name_lower:
                        doubles.append("fake")
                    elif 'spy' in name_lower:
                        doubles.append("spy")
                    elif 'dummy' in name_lower:
                        doubles.append("dummy")
        
        return doubles
    
    def _detect_mock_smells(self, test_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Detect mock-related test smells."""
        smells = []
        
        # Check for mock verification without behavior setup
        has_assert_called = any('assert_called' in str(n) for n in ast.walk(test_node))
        has_return_value = any('return_value' in str(n) for n in ast.walk(test_node))
        
        if has_assert_called and not has_return_value:
            smells.append({
                "type": "over_specification",
                "description": "Mock verification without behavior setup",
                "line": test_node.lineno
            })
        
        # Check for mocking concrete classes
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and 'Mock' in node.func.id:
                    if len(node.args) > 0 and isinstance(node.args[0], ast.Name):
                        smells.append({
                            "type": "concrete_class_mock",
                            "description": f"Mocking concrete class: {node.args[0].id}",
                            "line": node.lineno
                        })
        
        return smells
    
    def _detect_test_smells(self) -> Dict[str, Any]:
        """Detect various test smells."""
        test_smells = {
            "assertion_roulette": [],
            "eager_test": [],
            "mystery_guest": [],
            "conditional_logic": [],
            "duplicate_tests": [],
            "slow_tests": [],
            "ignored_tests": []
        }
        
        test_files = self._get_test_files()
        test_signatures = defaultdict(list)
        
        for test_file in test_files:
            try:
                tree = self._get_ast(test_file)
                content = self._get_file_content(test_file)
                file_key = str(test_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        # Assertion roulette - multiple assertions without messages
                        assertions = self._count_assertions(node)
                        if assertions["without_message"] > 2:
                            test_smells["assertion_roulette"].append({
                                "test": node.name,
                                "file": file_key,
                                "assertion_count": assertions["without_message"],
                                "line": node.lineno
                            })
                        
                        # Eager test - testing too much
                        if assertions["total"] > 10 or self._calculate_function_complexity(node) > 10:
                            test_smells["eager_test"].append({
                                "test": node.name,
                                "file": file_key,
                                "complexity": self._calculate_function_complexity(node),
                                "line": node.lineno
                            })
                        
                        # Mystery guest - external file dependencies
                        if self._has_file_dependencies(node):
                            test_smells["mystery_guest"].append({
                                "test": node.name,
                                "file": file_key,
                                "line": node.lineno
                            })
                        
                        # Conditional logic in tests
                        if self._has_conditional_logic(node):
                            test_smells["conditional_logic"].append({
                                "test": node.name,
                                "file": file_key,
                                "line": node.lineno
                            })
                        
                        # Track for duplicate detection
                        sig = self._get_test_signature(node)
                        test_signatures[sig].append({
                            "test": node.name,
                            "file": file_key,
                            "line": node.lineno
                        })
                        
                        # Ignored tests
                        if self._is_ignored_test(node):
                            test_smells["ignored_tests"].append({
                                "test": node.name,
                                "file": file_key,
                                "line": node.lineno
                            })
                
            except:
                continue
        
        # Find duplicate tests
        for sig, tests in test_signatures.items():
            if len(tests) > 1:
                test_smells["duplicate_tests"].append({
                    "signature": sig,
                    "duplicates": tests
                })
        
        return test_smells
    
    def _count_assertions(self, test_node: ast.FunctionDef) -> Dict[str, int]:
        """Count assertions in a test."""
        total = 0
        without_message = 0
        
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id.startswith('assert'):
                        total += 1
                        # Check if assertion has message
                        if len(node.args) < 2 and not any(kw.arg == 'msg' for kw in node.keywords):
                            without_message += 1
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr.startswith('assert'):
                        total += 1
                        if len(node.args) < 2:
                            without_message += 1
        
        return {"total": total, "without_message": without_message}
    
    def _has_file_dependencies(self, test_node: ast.FunctionDef) -> bool:
        """Check if test depends on external files."""
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    return True
                # Check for path operations
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['read', 'write', 'exists', 'join']:
                        return True
        return False
    
    def _has_conditional_logic(self, test_node: ast.FunctionDef) -> bool:
        """Check if test has conditional logic."""
        for node in ast.walk(test_node):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                return True
        return False
    
    def _get_test_signature(self, test_node: ast.FunctionDef) -> str:
        """Get test signature for duplicate detection."""
        # Simple signature based on assertions and calls
        calls = []
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        return '|'.join(sorted(calls))
    
    def _is_ignored_test(self, test_node: ast.FunctionDef) -> bool:
        """Check if test is ignored/skipped."""
        for decorator in test_node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ['skip', 'skipTest', 'skipIf', 'skipUnless']:
                    return True
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in ['skip', 'skipTest', 'skipIf', 'skipUnless']:
                    return True
        return False
    
    def _assess_mutation_testing_readiness(self) -> Dict[str, Any]:
        """Assess readiness for mutation testing."""
        readiness = {
            "mutation_score_estimate": 0,
            "high_value_targets": [],
            "mutation_operators": [],
            "expected_mutants": 0,
            "recommendations": []
        }
        
        # Analyze code for mutation opportunities
        for py_file in self._get_python_files():
            if self._is_test_file(py_file):
                continue
                
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Count mutation opportunities
                        mutations = self._count_mutation_opportunities(node)
                        
                        if mutations["total"] > 5:
                            readiness["high_value_targets"].append({
                                "function": node.name,
                                "file": file_key,
                                "mutation_count": mutations["total"],
                                "operators": mutations["operators"]
                            })
                        
                        readiness["expected_mutants"] += mutations["total"]
                
            except:
                continue
        
        # Determine applicable mutation operators
        readiness["mutation_operators"] = [
            "arithmetic_operator_replacement",
            "comparison_operator_replacement",
            "logical_operator_replacement",
            "constant_replacement",
            "return_value_mutation",
            "exception_swallowing"
        ]
        
        # Estimate mutation score
        if readiness["expected_mutants"] > 0:
            # Rough estimate based on test coverage
            readiness["mutation_score_estimate"] = 60  # Would need actual test execution
        
        # Generate recommendations
        if readiness["expected_mutants"] > 100:
            readiness["recommendations"].append("Start with critical functions only")
        readiness["recommendations"].append("Use mutmut or cosmic-ray for Python mutation testing")
        
        return readiness
    
    def _count_mutation_opportunities(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Count mutation testing opportunities."""
        mutations = {
            "total": 0,
            "operators": []
        }
        
        for node in ast.walk(func_node):
            # Arithmetic operators
            if isinstance(node, ast.BinOp):
                mutations["total"] += 1
                mutations["operators"].append("arithmetic")
            
            # Comparison operators
            elif isinstance(node, ast.Compare):
                mutations["total"] += len(node.ops)
                mutations["operators"].append("comparison")
            
            # Boolean operators
            elif isinstance(node, ast.BoolOp):
                mutations["total"] += 1
                mutations["operators"].append("logical")
            
            # Constants
            elif isinstance(node, ast.Constant):
                mutations["total"] += 1
                mutations["operators"].append("constant")
            
            # Return statements
            elif isinstance(node, ast.Return):
                mutations["total"] += 1
                mutations["operators"].append("return")
        
        mutations["operators"] = list(set(mutations["operators"]))
        return mutations
    
    def _identify_property_testing_opportunities(self) -> Dict[str, Any]:
        """Identify functions suitable for property-based testing."""
        property_testing = {
            "suitable_functions": [],
            "invariants": [],
            "properties": []
        }
        
        for py_file in self._get_python_files():
            if self._is_test_file(py_file):
                continue
                
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if function is pure
                        if self._is_pure_function(node):
                            properties = self._identify_function_properties(node)
                            
                            if properties:
                                property_testing["suitable_functions"].append({
                                    "function": node.name,
                                    "file": file_key,
                                    "line": node.lineno,
                                    "properties": properties,
                                    "hypothesis_strategies": self._suggest_hypothesis_strategies(node)
                                })
                        
                        # Identify invariants
                        invariants = self._identify_invariants(node)
                        for inv in invariants:
                            inv["function"] = node.name
                            inv["file"] = file_key
                            property_testing["invariants"].append(inv)
                
            except:
                continue
        
        return property_testing
    
    def _is_pure_function(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is pure (no side effects)."""
        # Check for side effects
        for node in ast.walk(func_node):
            # I/O operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['print', 'open', 'input']:
                        return False
            
            # Global/nonlocal modifications
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return False
            
            # Attribute assignments (object mutations)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        return False
        
        return True
    
    def _identify_function_properties(self, func_node: ast.FunctionDef) -> List[str]:
        """Identify testable properties of a function."""
        properties = []
        
        # Check for idempotency
        if 'sort' in func_node.name.lower() or 'normalize' in func_node.name.lower():
            properties.append("idempotent")
        
        # Check for commutativity
        if any(isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mult))
              for n in ast.walk(func_node)):
            properties.append("commutative")
        
        # Check for inverse relationships
        if 'encode' in func_node.name.lower() or 'decode' in func_node.name.lower():
            properties.append("has_inverse")
        
        # Check for range constraints
        if any(isinstance(n, ast.Compare) for n in ast.walk(func_node)):
            properties.append("range_constrained")
        
        return properties
    
    def _suggest_hypothesis_strategies(self, func_node: ast.FunctionDef) -> List[str]:
        """Suggest Hypothesis strategies for property testing."""
        strategies = []
        
        # Analyze parameter types
        for arg in func_node.args.args:
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    type_name = arg.annotation.id
                    if type_name == 'int':
                        strategies.append("integers()")
                    elif type_name == 'str':
                        strategies.append("text()")
                    elif type_name == 'float':
                        strategies.append("floats()")
                    elif type_name == 'list':
                        strategies.append("lists()")
            else:
                # Guess from parameter name
                if 'num' in arg.arg or 'count' in arg.arg:
                    strategies.append("integers()")
                elif 'name' in arg.arg or 'text' in arg.arg:
                    strategies.append("text()")
        
        return strategies
    
    def _identify_invariants(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Identify function invariants."""
        invariants = []
        
        # Check for assertions (invariant checks)
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                invariants.append({
                    "type": "assertion",
                    "line": node.lineno,
                    "description": "Assertion-based invariant"
                })
        
        # Check for pre/post conditions in docstring
        docstring = ast.get_docstring(func_node)
        if docstring:
            if 'precondition' in docstring.lower() or 'requires' in docstring.lower():
                invariants.append({
                    "type": "precondition",
                    "description": "Documented precondition"
                })
            if 'postcondition' in docstring.lower() or 'ensures' in docstring.lower():
                invariants.append({
                    "type": "postcondition",
                    "description": "Documented postcondition"
                })
        
        return invariants
    
    def _analyze_test_data_patterns(self) -> Dict[str, Any]:
        """Analyze test data and fixture patterns."""
        test_data_analysis = {
            "fixtures": [],
            "fixture_complexity": {},
            "test_data_duplication": [],
            "factory_patterns": [],
            "builder_patterns": []
        }
        
        test_files = self._get_test_files()
        test_data_hashes = defaultdict(list)
        
        for test_file in test_files:
            try:
                tree = self._get_ast(test_file)
                content = self._get_file_content(test_file)
                file_key = str(test_file.relative_to(self.base_path))
                
                # Detect fixtures
                fixtures = self._detect_fixtures(tree, content)
                for fixture in fixtures:
                    fixture["file"] = file_key
                    test_data_analysis["fixtures"].append(fixture)
                    
                    # Analyze fixture complexity
                    complexity = self._analyze_fixture_complexity(fixture["node"]) if "node" in fixture else 0
                    test_data_analysis["fixture_complexity"][fixture["name"]] = complexity
                
                # Detect test data patterns
                for node in ast.walk(tree):
                    # Look for test data creation
                    if isinstance(node, ast.Assign):
                        # Hash the data structure for duplication detection
                        data_hash = hash(ast.dump(node.value))
                        test_data_hashes[data_hash].append({
                            "file": file_key,
                            "line": node.lineno
                        })
                    
                    # Detect factory patterns
                    if isinstance(node, ast.FunctionDef):
                        if 'factory' in node.name.lower() or 'create' in node.name.lower():
                            test_data_analysis["factory_patterns"].append({
                                "name": node.name,
                                "file": file_key,
                                "line": node.lineno
                            })
                        
                        # Detect builder patterns
                        if 'builder' in node.name.lower() or 'build' in node.name.lower():
                            test_data_analysis["builder_patterns"].append({
                                "name": node.name,
                                "file": file_key,
                                "line": node.lineno
                            })
                
            except:
                continue
        
        # Find duplicated test data
        for data_hash, locations in test_data_hashes.items():
            if len(locations) > 1:
                test_data_analysis["test_data_duplication"].append({
                    "locations": locations,
                    "recommendation": "Consider extracting to shared fixture"
                })
        
        return test_data_analysis
    
    def _detect_fixtures(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect test fixtures."""
        fixtures = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # pytest fixtures
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'fixture':
                        fixtures.append({
                            "name": node.name,
                            "type": "pytest",
                            "line": node.lineno,
                            "node": node
                        })
                
                # setUp/tearDown methods
                if node.name in ['setUp', 'tearDown', 'setUpClass', 'tearDownClass']:
                    fixtures.append({
                        "name": node.name,
                        "type": "unittest",
                        "line": node.lineno,
                        "node": node
                    })
        
        return fixtures
    
    def _analyze_fixture_complexity(self, fixture_node: ast.FunctionDef) -> int:
        """Analyze complexity of a fixture."""
        return self._calculate_function_complexity(fixture_node)
    
    def _predict_flaky_tests(self) -> Dict[str, Any]:
        """Predict tests likely to be flaky."""
        flaky_predictions = {
            "high_risk": [],
            "medium_risk": [],
            "risk_factors": defaultdict(list)
        }
        
        test_files = self._get_test_files()
        
        for test_file in test_files:
            try:
                tree = self._get_ast(test_file)
                content = self._get_file_content(test_file)
                file_key = str(test_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        risk_score = 0
                        risk_factors = []
                        
                        # Time-dependent operations
                        if self._has_time_dependency(node):
                            risk_score += 3
                            risk_factors.append("time_dependency")
                        
                        # Network operations
                        if self._has_network_operations(node):
                            risk_score += 3
                            risk_factors.append("network_operations")
                        
                        # Random data
                        if self._uses_random_data(node):
                            risk_score += 2
                            risk_factors.append("random_data")
                        
                        # Threading/async
                        if self._has_threading(node):
                            risk_score += 3
                            risk_factors.append("threading")
                        
                        # External service dependencies
                        if self._has_external_dependencies(node):
                            risk_score += 2
                            risk_factors.append("external_dependencies")
                        
                        # Test order dependencies
                        if self._has_order_dependency(node, tree):
                            risk_score += 3
                            risk_factors.append("order_dependency")
                        
                        # Floating point comparisons
                        if self._has_float_comparisons(node):
                            risk_score += 1
                            risk_factors.append("float_comparison")
                        
                        if risk_score >= 5:
                            flaky_predictions["high_risk"].append({
                                "test": node.name,
                                "file": file_key,
                                "line": node.lineno,
                                "risk_score": risk_score,
                                "risk_factors": risk_factors
                            })
                        elif risk_score >= 2:
                            flaky_predictions["medium_risk"].append({
                                "test": node.name,
                                "file": file_key,
                                "line": node.lineno,
                                "risk_score": risk_score,
                                "risk_factors": risk_factors
                            })
                        
                        for factor in risk_factors:
                            flaky_predictions["risk_factors"][factor].append(node.name)
                
            except:
                continue
        
        return flaky_predictions
    
    def _has_time_dependency(self, test_node: ast.FunctionDef) -> bool:
        """Check if test has time dependencies."""
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['sleep', 'time', 'now', 'today']:
                        return True
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['sleep', 'time', 'now', 'today', 'utcnow']:
                        return True
        return False
    
    def _has_network_operations(self, test_node: ast.FunctionDef) -> bool:
        """Check if test has network operations."""
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['get', 'post', 'put', 'delete', 'request', 'urlopen']:
                        return True
        return False
    
    def _uses_random_data(self, test_node: ast.FunctionDef) -> bool:
        """Check if test uses random data."""
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if 'random' in node.func.id.lower():
                        return True
                elif isinstance(node.func, ast.Attribute):
                    if 'random' in node.func.attr.lower() or 'rand' in node.func.attr.lower():
                        return True
        return False
    
    def _has_threading(self, test_node: ast.FunctionDef) -> bool:
        """Check if test uses threading or async."""
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['Thread', 'Process', 'asyncio']:
                        return True
            elif isinstance(node, ast.AsyncFunctionDef):
                return True
        return False
    
    def _has_external_dependencies(self, test_node: ast.FunctionDef) -> bool:
        """Check if test has external dependencies."""
        # Check for database or API calls without mocks
        has_external_call = False
        has_mock = False
        
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['execute', 'query', 'connect']:
                        has_external_call = True
                    if 'mock' in node.func.attr.lower():
                        has_mock = True
        
        return has_external_call and not has_mock
    
    def _has_order_dependency(self, test_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if test depends on execution order."""
        # Check for shared state modifications
        for node in ast.walk(test_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if it's a class or module level variable
                        if not any(target.id == arg.arg for arg in test_node.args.args):
                            return True
        return False
    
    def _has_float_comparisons(self, test_node: ast.FunctionDef) -> bool:
        """Check if test has floating point comparisons."""
        for node in ast.walk(test_node):
            if isinstance(node, ast.Compare):
                # Check if comparing floats
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant):
                        if isinstance(comparator.value, float):
                            # Check if using exact equality
                            for op in node.ops:
                                if isinstance(op, (ast.Eq, ast.NotEq)):
                                    return True
        return False
    
    def _calculate_testability_metrics(self) -> Dict[str, Any]:
        """Calculate overall testability metrics."""
        coverage = self._analyze_test_coverage_potential()
        pyramid = self._analyze_test_pyramid()
        mocks = self._analyze_mock_dependencies()
        smells = self._detect_test_smells()
        flaky = self._predict_flaky_tests()
        
        metrics = {
            "untested_function_count": len(coverage["untested_functions"]),
            "test_pyramid_health": pyramid.get("pyramid_health", "UNKNOWN"),
            "average_mock_count": sum(m["mock_count"] for m in mocks["mock_usage"]) / max(len(mocks["mock_usage"]), 1),
            "test_smell_count": sum(len(v) for v in smells.values() if isinstance(v, list)),
            "flaky_test_risk": {
                "high": len(flaky["high_risk"]),
                "medium": len(flaky["medium_risk"])
            },
            "testability_score": 0,
            "recommendations": []
        }
        
        # Calculate testability score
        score = 100
        
        if metrics["untested_function_count"] > 10:
            score -= 20
            metrics["recommendations"].append("Increase test coverage for untested functions")
        
        if metrics["test_pyramid_health"] == "UNHEALTHY":
            score -= 15
            metrics["recommendations"].append("Rebalance test pyramid with more unit tests")
        
        if metrics["average_mock_count"] > 3:
            score -= 10
            metrics["recommendations"].append("Reduce mock usage to improve test maintainability")
        
        if metrics["test_smell_count"] > 20:
            score -= 15
            metrics["recommendations"].append("Address test smells to improve test quality")
        
        if metrics["flaky_test_risk"]["high"] > 5:
            score -= 20
            metrics["recommendations"].append("Fix high-risk flaky tests immediately")
        
        metrics["testability_score"] = max(score, 0)
        
        # Add specific recommendations
        metrics["recommendations"].extend([
            "Implement property-based testing for mathematical functions",
            "Add mutation testing to verify test effectiveness",
            "Use test data builders to reduce duplication"
        ])
        
        return metrics