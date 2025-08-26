#!/usr/bin/env python3
"""
Test Complexity Prioritizer - Integrates complexity analysis for intelligent test prioritization.
Uses cyclomatic complexity, cognitive complexity, and dependency analysis to order tests.
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import ast
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create stub implementations for missing analyzers
class ComplexityAnalyzer:
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Simple complexity analysis stub."""
        return {
            'total_complexity': 10,
            'cognitive_complexity': 15,
            'maintainability_index': 75,
            'lines_of_code': 100,
            'complex_functions': []
        }

class DependencyAnalyzer:
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Simple dependency analysis stub."""
        return {
            'dependencies': [],
            'dependents': [],
            'circular_dependencies': []
        }

class ArchitectureAnalyzer:
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Simple architecture analysis stub."""
        return {}

class IntelligentTestBuilder:
    def __init__(self):
        pass


@dataclass
class TestPriority:
    """Represents a test with its priority score and metadata."""
    test_path: str
    test_name: str
    priority_score: float
    complexity_score: float
    risk_score: float
    dependency_score: float
    execution_time_estimate: float
    affected_modules: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    last_failure_rate: float = 0.0
    last_execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'test_path': self.test_path,
            'test_name': self.test_name,
            'priority_score': self.priority_score,
            'complexity_score': self.complexity_score,
            'risk_score': self.risk_score,
            'dependency_score': self.dependency_score,
            'execution_time_estimate': self.execution_time_estimate,
            'affected_modules': self.affected_modules,
            'tags': self.tags,
            'last_failure_rate': self.last_failure_rate,
            'last_execution_time': self.last_execution_time
        }


@dataclass
class TestSuite:
    """Represents a prioritized test suite."""
    suite_id: str
    tests: List[TestPriority]
    total_complexity: float
    estimated_runtime: float
    risk_coverage: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'suite_id': self.suite_id,
            'tests': [t.to_dict() for t in self.tests],
            'total_complexity': self.total_complexity,
            'estimated_runtime': self.estimated_runtime,
            'risk_coverage': self.risk_coverage,
            'created_at': self.created_at.isoformat()
        }


class TestComplexityPrioritizer:
    """Prioritizes tests based on complexity analysis and risk factors."""
    
    def __init__(self, project_root: str = '.'):
        """Initialize the test prioritizer."""
        self.project_root = Path(project_root).resolve()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.architecture_analyzer = ArchitectureAnalyzer()
        self.test_builder = IntelligentTestBuilder()
        
        # Caches
        self._complexity_cache: Dict[str, Dict[str, Any]] = {}
        self._dependency_cache: Dict[str, Dict[str, Any]] = {}
        self._test_history: Dict[str, Dict[str, Any]] = {}
        self._module_risk_scores: Dict[str, float] = {}
        
        # Configuration
        self.config = {
            'complexity_weight': 0.3,
            'risk_weight': 0.4,
            'dependency_weight': 0.2,
            'history_weight': 0.1,
            'min_priority_threshold': 0.3,
            'max_parallel_analysis': 10,
            'cache_ttl_seconds': 300
        }
        
        # Threading
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.config['max_parallel_analysis'])
        
    def discover_tests(self, test_dir: Optional[str] = None) -> List[str]:
        """Discover all test files in the project."""
        test_files = []
        test_root = Path(test_dir) if test_dir else self.project_root
        
        # Common test patterns
        test_patterns = ['test_*.py', '*_test.py', 'tests.py']
        test_dirs = ['tests', 'test', 'testing']
        
        # Search for test files
        for pattern in test_patterns:
            test_files.extend(test_root.rglob(pattern))
        
        # Search in test directories
        for test_dir_name in test_dirs:
            test_dir_path = test_root / test_dir_name
            if test_dir_path.exists():
                for pattern in test_patterns:
                    test_files.extend(test_dir_path.rglob(pattern))
        
        # Filter and return unique paths
        unique_paths = list(set(str(f) for f in test_files if f.is_file()))
        return sorted(unique_paths)
    
    def analyze_test_complexity(self, test_path: str) -> Dict[str, Any]:
        """Analyze complexity of a single test file."""
        # Check cache
        if test_path in self._complexity_cache:
            cached = self._complexity_cache[test_path]
            if time.time() - cached['timestamp'] < self.config['cache_ttl_seconds']:
                return cached['data']
        
        try:
            # Analyze complexity
            complexity_result = self.complexity_analyzer.analyze(test_path)
            
            # Extract key metrics
            metrics = {
                'cyclomatic_complexity': complexity_result.get('total_complexity', 0),
                'cognitive_complexity': complexity_result.get('cognitive_complexity', 0),
                'maintainability_index': complexity_result.get('maintainability_index', 100),
                'lines_of_code': complexity_result.get('lines_of_code', 0),
                'test_count': self._count_tests_in_file(test_path),
                'assertion_count': self._count_assertions_in_file(test_path)
            }
            
            # Calculate complexity score (0-1)
            complexity_score = self._calculate_complexity_score(metrics)
            
            result = {
                'path': test_path,
                'metrics': metrics,
                'complexity_score': complexity_score
            }
            
            # Cache result
            with self._lock:
                self._complexity_cache[test_path] = {
                    'data': result,
                    'timestamp': time.time()
                }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing complexity for {test_path}: {e}")
            return {
                'path': test_path,
                'metrics': {},
                'complexity_score': 0.5  # Default middle score
            }
    
    def analyze_test_dependencies(self, test_path: str) -> Dict[str, Any]:
        """Analyze dependencies and impacts of a test."""
        # Check cache
        if test_path in self._dependency_cache:
            cached = self._dependency_cache[test_path]
            if time.time() - cached['timestamp'] < self.config['cache_ttl_seconds']:
                return cached['data']
        
        try:
            # Analyze dependencies
            dep_result = self.dependency_analyzer.analyze(test_path)
            
            # Get affected modules
            affected_modules = self._get_affected_modules(test_path, dep_result)
            
            # Calculate dependency score based on impact
            dependency_score = len(affected_modules) / 10.0  # Normalize
            dependency_score = min(1.0, dependency_score)
            
            result = {
                'path': test_path,
                'dependencies': dep_result.get('dependencies', []),
                'affected_modules': affected_modules,
                'dependency_score': dependency_score
            }
            
            # Cache result
            with self._lock:
                self._dependency_cache[test_path] = {
                    'data': result,
                    'timestamp': time.time()
                }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing dependencies for {test_path}: {e}")
            return {
                'path': test_path,
                'dependencies': [],
                'affected_modules': [],
                'dependency_score': 0.5
            }
    
    def calculate_risk_score(self, test_path: str, complexity_data: Dict, dependency_data: Dict) -> float:
        """Calculate risk score for a test based on multiple factors."""
        risk_score = 0.0
        
        # Factor 1: Code complexity risk
        complexity_risk = complexity_data.get('complexity_score', 0.5)
        
        # Factor 2: Dependency risk
        dependency_risk = dependency_data.get('dependency_score', 0.5)
        
        # Factor 3: Historical failure risk
        history = self._test_history.get(test_path, {})
        failure_rate = history.get('failure_rate', 0.0)
        historical_risk = failure_rate
        
        # Factor 4: Module criticality risk
        affected_modules = dependency_data.get('affected_modules', [])
        module_risk = self._calculate_module_risk(affected_modules)
        
        # Factor 5: Change frequency risk
        change_risk = self._calculate_change_frequency_risk(test_path)
        
        # Weighted combination
        risk_score = (
            complexity_risk * 0.25 +
            dependency_risk * 0.25 +
            historical_risk * 0.2 +
            module_risk * 0.2 +
            change_risk * 0.1
        )
        
        return min(1.0, max(0.0, risk_score))
    
    def prioritize_tests(self, 
                        test_paths: Optional[List[str]] = None,
                        changed_files: Optional[List[str]] = None,
                        risk_threshold: Optional[float] = None) -> TestSuite:
        """Prioritize tests based on complexity, risk, and dependencies."""
        
        # Discover tests if not provided
        if test_paths is None:
            test_paths = self.discover_tests()
        
        if not test_paths:
            return TestSuite(
                suite_id=f"suite_{int(time.time())}",
                tests=[],
                total_complexity=0.0,
                estimated_runtime=0.0,
                risk_coverage=0.0,
                created_at=datetime.now()
            )
        
        # Analyze tests in parallel
        test_priorities = []
        futures = []
        
        with self._executor as executor:
            for test_path in test_paths:
                future = executor.submit(self._analyze_single_test, test_path, changed_files)
                futures.append((test_path, future))
            
            for test_path, future in futures:
                try:
                    priority = future.result(timeout=30)
                    if priority:
                        test_priorities.append(priority)
                except Exception as e:
                    print(f"Error analyzing test {test_path}: {e}")
        
        # Sort by priority score (highest first)
        test_priorities.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Apply risk threshold if specified
        if risk_threshold:
            test_priorities = [t for t in test_priorities if t.risk_score >= risk_threshold]
        
        # Apply minimum priority threshold
        min_threshold = self.config['min_priority_threshold']
        test_priorities = [t for t in test_priorities if t.priority_score >= min_threshold]
        
        # Calculate suite metrics
        total_complexity = sum(t.complexity_score for t in test_priorities)
        estimated_runtime = sum(t.execution_time_estimate for t in test_priorities)
        risk_coverage = self._calculate_risk_coverage(test_priorities)
        
        return TestSuite(
            suite_id=f"suite_{int(time.time())}",
            tests=test_priorities,
            total_complexity=total_complexity,
            estimated_runtime=estimated_runtime,
            risk_coverage=risk_coverage,
            created_at=datetime.now()
        )
    
    def _analyze_single_test(self, test_path: str, changed_files: Optional[List[str]] = None) -> Optional[TestPriority]:
        """Analyze a single test and calculate its priority."""
        try:
            # Analyze complexity
            complexity_data = self.analyze_test_complexity(test_path)
            
            # Analyze dependencies
            dependency_data = self.analyze_test_dependencies(test_path)
            
            # Calculate risk score
            risk_score = self.calculate_risk_score(test_path, complexity_data, dependency_data)
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(
                complexity_data['complexity_score'],
                risk_score,
                dependency_data['dependency_score'],
                changed_files,
                dependency_data['affected_modules']
            )
            
            # Get test name
            test_name = Path(test_path).stem
            
            # Estimate execution time
            execution_time = self._estimate_execution_time(test_path, complexity_data)
            
            return TestPriority(
                test_path=test_path,
                test_name=test_name,
                priority_score=priority_score,
                complexity_score=complexity_data['complexity_score'],
                risk_score=risk_score,
                dependency_score=dependency_data['dependency_score'],
                execution_time_estimate=execution_time,
                affected_modules=dependency_data['affected_modules'],
                tags=self._extract_test_tags(test_path),
                last_failure_rate=self._test_history.get(test_path, {}).get('failure_rate', 0.0),
                last_execution_time=self._test_history.get(test_path, {}).get('avg_execution_time', 0.0)
            )
            
        except Exception as e:
            print(f"Error analyzing test {test_path}: {e}")
            return None
    
    def _calculate_priority_score(self, 
                                 complexity_score: float,
                                 risk_score: float,
                                 dependency_score: float,
                                 changed_files: Optional[List[str]],
                                 affected_modules: List[str]) -> float:
        """Calculate overall priority score for a test."""
        
        # Base priority from configured weights
        base_priority = (
            complexity_score * self.config['complexity_weight'] +
            risk_score * self.config['risk_weight'] +
            dependency_score * self.config['dependency_weight']
        )
        
        # Boost priority if test is related to changed files
        if changed_files and affected_modules:
            changed_modules = {self._get_module_from_path(f) for f in changed_files}
            if changed_modules.intersection(set(affected_modules)):
                base_priority *= 1.5  # 50% boost for relevant tests
        
        return min(1.0, base_priority)
    
    def _calculate_complexity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate normalized complexity score from metrics."""
        cyclomatic = metrics.get('cyclomatic_complexity', 0)
        cognitive = metrics.get('cognitive_complexity', 0)
        maintainability = metrics.get('maintainability_index', 100)
        
        # Normalize scores (higher complexity = higher score)
        cyclo_score = min(1.0, cyclomatic / 50.0)
        cog_score = min(1.0, cognitive / 100.0)
        maint_score = max(0.0, (100 - maintainability) / 100.0)
        
        # Weighted average
        return (cyclo_score * 0.4 + cog_score * 0.4 + maint_score * 0.2)
    
    def _get_affected_modules(self, test_path: str, dep_result: Dict) -> List[str]:
        """Get list of modules affected by a test."""
        affected = []
        
        # Direct imports
        for dep in dep_result.get('dependencies', []):
            if isinstance(dep, dict):
                module = dep.get('module', dep.get('name', ''))
            else:
                module = str(dep)
            if module and not module.startswith('test'):
                affected.append(module)
        
        # Modules being tested (inferred from test name)
        test_name = Path(test_path).stem
        if test_name.startswith('test_'):
            module_name = test_name[5:]  # Remove 'test_' prefix
            affected.append(module_name)
        
        return list(set(affected))
    
    def _calculate_module_risk(self, modules: List[str]) -> float:
        """Calculate risk score based on module criticality."""
        if not modules:
            return 0.0
        
        # Critical modules have higher risk
        critical_modules = {'core', 'auth', 'security', 'database', 'api', 'payment'}
        critical_count = sum(1 for m in modules if any(c in m.lower() for c in critical_modules))
        
        return min(1.0, critical_count / max(1, len(modules)))
    
    def _calculate_change_frequency_risk(self, test_path: str) -> float:
        """Calculate risk based on how frequently the test or its dependencies change."""
        # For now, return a default value
        # In a real implementation, this would check git history
        return 0.3
    
    def _count_tests_in_file(self, file_path: str) -> int:
        """Count number of test functions/methods in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test'):
                    count += 1
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith('test'):
                            count += 1
            
            return count
        except:
            return 0
    
    def _count_assertions_in_file(self, file_path: str) -> int:
        """Count number of assertions in a test file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count common assertion patterns
            assertions = ['assert ', 'self.assert', 'self.assertEqual', 'self.assertTrue', 
                         'self.assertFalse', 'self.assertIn', 'self.assertIs', 'expect(']
            
            count = sum(content.count(assertion) for assertion in assertions)
            return count
        except:
            return 0
    
    def _estimate_execution_time(self, test_path: str, complexity_data: Dict) -> float:
        """Estimate test execution time based on complexity and history."""
        # Check historical data
        history = self._test_history.get(test_path, {})
        if 'avg_execution_time' in history:
            return history['avg_execution_time']
        
        # Estimate based on complexity
        metrics = complexity_data.get('metrics', {})
        test_count = metrics.get('test_count', 1)
        assertion_count = metrics.get('assertion_count', 1)
        complexity = metrics.get('cyclomatic_complexity', 1)
        
        # Simple estimation formula (in seconds)
        base_time = 0.1  # Base time per test
        time_per_assertion = 0.01
        complexity_factor = 1 + (complexity / 100)
        
        estimated_time = (base_time * test_count + time_per_assertion * assertion_count) * complexity_factor
        
        return estimated_time
    
    def _extract_test_tags(self, test_path: str) -> List[str]:
        """Extract tags/markers from test file."""
        tags = []
        
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for pytest markers
            import re
            markers = re.findall(r'@pytest\.mark\.(\w+)', content)
            tags.extend(markers)
            
            # Look for unittest skip decorators
            if '@unittest.skip' in content:
                tags.append('skip')
            
            # Look for categories in comments
            category_match = re.search(r'#\s*category:\s*(\w+)', content, re.IGNORECASE)
            if category_match:
                tags.append(category_match.group(1))
            
        except:
            pass
        
        return list(set(tags))
    
    def _get_module_from_path(self, file_path: str) -> str:
        """Extract module name from file path."""
        path = Path(file_path)
        if path.suffix == '.py':
            return path.stem
        return path.name
    
    def _calculate_risk_coverage(self, tests: List[TestPriority]) -> float:
        """Calculate overall risk coverage of the test suite."""
        if not tests:
            return 0.0
        
        total_risk = sum(t.risk_score for t in tests)
        max_possible_risk = len(tests)  # If all tests had risk score of 1.0
        
        return total_risk / max_possible_risk if max_possible_risk > 0 else 0.0
    
    def update_test_history(self, test_path: str, execution_time: float, passed: bool):
        """Update test execution history for future prioritization."""
        with self._lock:
            if test_path not in self._test_history:
                self._test_history[test_path] = {
                    'executions': [],
                    'failures': 0,
                    'total': 0
                }
            
            history = self._test_history[test_path]
            history['executions'].append(execution_time)
            history['total'] += 1
            if not passed:
                history['failures'] += 1
            
            # Keep only last 100 executions
            if len(history['executions']) > 100:
                history['executions'] = history['executions'][-100:]
            
            # Calculate statistics
            history['avg_execution_time'] = sum(history['executions']) / len(history['executions'])
            history['failure_rate'] = history['failures'] / history['total']
    
    def export_prioritized_suite(self, suite: TestSuite, output_file: str):
        """Export prioritized test suite to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"Exported test suite to {output_file}")
    
    def generate_execution_plan(self, suite: TestSuite, max_parallel: int = 4) -> List[List[TestPriority]]:
        """Generate parallel execution plan for test suite."""
        if not suite.tests:
            return []
        
        # Group tests by estimated execution time for load balancing
        groups = [[] for _ in range(max_parallel)]
        group_times = [0.0] * max_parallel
        
        # Sort tests by execution time (longest first)
        sorted_tests = sorted(suite.tests, key=lambda x: x.execution_time_estimate, reverse=True)
        
        # Assign tests to groups using bin packing algorithm
        for test in sorted_tests:
            # Find group with minimum total time
            min_idx = group_times.index(min(group_times))
            groups[min_idx].append(test)
            group_times[min_idx] += test.execution_time_estimate
        
        # Remove empty groups
        execution_plan = [g for g in groups if g]
        
        return execution_plan


def main():
    """Main function to demonstrate test prioritization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prioritize tests based on complexity analysis')
    parser.add_argument('--project', default='.', help='Project root directory')
    parser.add_argument('--test-dir', help='Test directory (optional)')
    parser.add_argument('--changed-files', nargs='*', help='List of changed files')
    parser.add_argument('--risk-threshold', type=float, help='Minimum risk threshold')
    parser.add_argument('--output', help='Output file for test suite')
    parser.add_argument('--max-parallel', type=int, default=4, help='Maximum parallel test groups')
    
    args = parser.parse_args()
    
    # Initialize prioritizer
    prioritizer = TestComplexityPrioritizer(args.project)
    
    print(f"Discovering tests in {args.project}...")
    test_paths = prioritizer.discover_tests(args.test_dir)
    print(f"Found {len(test_paths)} test files")
    
    # Prioritize tests
    print("\nAnalyzing and prioritizing tests...")
    suite = prioritizer.prioritize_tests(
        test_paths=test_paths,
        changed_files=args.changed_files,
        risk_threshold=args.risk_threshold
    )
    
    print(f"\nPrioritized Test Suite:")
    print(f"  Suite ID: {suite.suite_id}")
    print(f"  Total tests: {len(suite.tests)}")
    print(f"  Total complexity: {suite.total_complexity:.2f}")
    print(f"  Estimated runtime: {suite.estimated_runtime:.2f} seconds")
    print(f"  Risk coverage: {suite.risk_coverage:.2%}")
    
    # Show top priority tests
    print("\nTop 10 Priority Tests:")
    for i, test in enumerate(suite.tests[:10], 1):
        print(f"  {i}. {test.test_name}")
        print(f"     Priority: {test.priority_score:.3f}")
        print(f"     Risk: {test.risk_score:.3f}")
        print(f"     Complexity: {test.complexity_score:.3f}")
        print(f"     Est. time: {test.execution_time_estimate:.2f}s")
        if test.tags:
            print(f"     Tags: {', '.join(test.tags)}")
    
    # Generate execution plan
    print(f"\nExecution Plan ({args.max_parallel} parallel groups):")
    execution_plan = prioritizer.generate_execution_plan(suite, args.max_parallel)
    for i, group in enumerate(execution_plan, 1):
        group_time = sum(t.execution_time_estimate for t in group)
        print(f"  Group {i}: {len(group)} tests, ~{group_time:.2f}s")
    
    # Export if requested
    if args.output:
        prioritizer.export_prioritized_suite(suite, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())