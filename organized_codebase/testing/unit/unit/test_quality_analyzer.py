"""
Test Quality Analyzer Module (Part 1 of advanced_testing_intelligence split)
Module size: <300 lines
Analyzes test quality, coverage, and identifies test smells.
"""

import ast
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class TestQuality(Enum):
    """Test quality levels."""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 70-89%
    FAIR = "fair"               # 50-69%
    POOR = "poor"               # 30-49%
    CRITICAL = "critical"       # 0-29%

class TestSmell(Enum):
    """Types of test smells."""
    ASSERTION_ROULETTE = "assertion_roulette"
    EAGER_TEST = "eager_test"
    LAZY_TEST = "lazy_test"
    MYSTERY_GUEST = "mystery_guest"
    SENSITIVE_EQUALITY = "sensitive_equality"
    INAPPROPRIATE_ASSERTION = "inappropriate_assertion"

@dataclass
class TestMetrics:
    """Metrics for a single test."""
    name: str
    assertions: int
    complexity: int
    lines: int
    coverage: float
    quality: TestQuality
    smells: List[TestSmell]
    
class TestQualityAnalyzer:
    """Analyzes test quality and identifies issues."""
    
    def __init__(self):
        self.metrics = {}
        self.quality_thresholds = {
            TestQuality.EXCELLENT: 0.90,
            TestQuality.GOOD: 0.70,
            TestQuality.FAIR: 0.50,
            TestQuality.POOR: 0.30,
            TestQuality.CRITICAL: 0.0
        }
        
    def analyze_test_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single test file for quality metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            test_functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    metrics = self._analyze_test_function(node, content)
                    test_functions.append(metrics)
            
            # Calculate file-level metrics
            file_metrics = self._calculate_file_metrics(test_functions)
            
            return {
                'file': str(file_path),
                'tests': test_functions,
                'file_metrics': file_metrics,
                'quality': self._determine_quality(file_metrics.get('avg_coverage', 0))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {'file': str(file_path), 'error': str(e)}
    
    def _analyze_test_function(self, node: ast.FunctionDef, content: str) -> TestMetrics:
        """Analyze a single test function."""
        # Count assertions
        assertions = self._count_assertions(node)
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        # Count lines
        lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        
        # Detect test smells
        smells = self._detect_test_smells(node, assertions, complexity)
        
        # Estimate coverage (would need actual execution for real coverage)
        coverage = min(1.0, assertions * 0.2)  # Rough estimate
        
        # Determine quality
        quality = self._determine_quality(coverage)
        
        return TestMetrics(
            name=node.name,
            assertions=assertions,
            complexity=complexity,
            lines=lines,
            coverage=coverage,
            quality=quality,
            smells=smells
        )
    
    def _count_assertions(self, node: ast.FunctionDef) -> int:
        """Count assertion statements in test."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                count += 1
            elif isinstance(child, ast.Call):
                if hasattr(child.func, 'attr'):
                    # unittest assertions
                    if child.func.attr.startswith('assert'):
                        count += 1
                elif hasattr(child.func, 'id'):
                    # pytest assertions
                    if child.func.id == 'assert':
                        count += 1
        return count
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of test."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        return complexity
    
    def _detect_test_smells(self, node: ast.FunctionDef, assertions: int, complexity: int) -> List[TestSmell]:
        """Detect common test smells."""
        smells = []
        
        # Assertion Roulette: too many assertions without messages
        if assertions > 5:
            smells.append(TestSmell.ASSERTION_ROULETTE)
        
        # Eager Test: testing too many things
        if complexity > 5:
            smells.append(TestSmell.EAGER_TEST)
        
        # Lazy Test: no assertions
        if assertions == 0:
            smells.append(TestSmell.LAZY_TEST)
        
        # Mystery Guest: external dependencies
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'id') and child.func.id in ['open', 'urlopen']:
                    smells.append(TestSmell.MYSTERY_GUEST)
                    break
        
        return smells
    
    def _determine_quality(self, coverage: float) -> TestQuality:
        """Determine test quality based on coverage."""
        for quality, threshold in self.quality_thresholds.items():
            if coverage >= threshold:
                return quality
        return TestQuality.CRITICAL
    
    def _calculate_file_metrics(self, test_functions: List[TestMetrics]) -> Dict[str, Any]:
        """Calculate file-level metrics from test functions."""
        if not test_functions:
            return {}
        
        total_assertions = sum(t.assertions for t in test_functions)
        total_lines = sum(t.lines for t in test_functions)
        avg_complexity = statistics.mean(t.complexity for t in test_functions)
        avg_coverage = statistics.mean(t.coverage for t in test_functions)
        
        all_smells = []
        for t in test_functions:
            all_smells.extend(t.smells)
        
        return {
            'total_tests': len(test_functions),
            'total_assertions': total_assertions,
            'total_lines': total_lines,
            'avg_complexity': avg_complexity,
            'avg_coverage': avg_coverage,
            'smell_count': len(all_smells),
            'unique_smells': len(set(all_smells))
        }
    
    def generate_quality_report(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        total_tests = sum(r.get('file_metrics', {}).get('total_tests', 0) for r in test_results)
        
        quality_distribution = {}
        for result in test_results:
            quality = result.get('quality', TestQuality.CRITICAL)
            quality_distribution[quality.value] = quality_distribution.get(quality.value, 0) + 1
        
        smell_frequency = {}
        for result in test_results:
            for test in result.get('tests', []):
                for smell in test.smells:
                    smell_frequency[smell.value] = smell_frequency.get(smell.value, 0) + 1
        
        return {
            'summary': {
                'total_files': len(test_results),
                'total_tests': total_tests,
                'quality_distribution': quality_distribution,
                'smell_frequency': smell_frequency
            },
            'recommendations': self._generate_recommendations(quality_distribution, smell_frequency)
        }
    
    def _generate_recommendations(self, quality_dist: Dict, smell_freq: Dict) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Quality-based recommendations
        if quality_dist.get(TestQuality.CRITICAL.value, 0) > 0:
            recommendations.append("CRITICAL: Some tests have very low quality. Immediate attention required.")
        
        if quality_dist.get(TestQuality.POOR.value, 0) > 0:
            recommendations.append("Improve test coverage for poor quality tests.")
        
        # Smell-based recommendations
        if smell_freq.get(TestSmell.LAZY_TEST.value, 0) > 0:
            recommendations.append("Add assertions to tests that currently have none.")
        
        if smell_freq.get(TestSmell.ASSERTION_ROULETTE.value, 0) > 0:
            recommendations.append("Break up tests with too many assertions into focused tests.")
        
        if smell_freq.get(TestSmell.MYSTERY_GUEST.value, 0) > 0:
            recommendations.append("Remove external dependencies from tests. Use mocks instead.")
        
        return recommendations