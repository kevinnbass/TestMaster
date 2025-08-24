"""
Test Metrics Collection Module
Handles collection and calculation of test metrics.
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class TestMetrics:
    """Metrics for a single test."""
    name: str
    file_path: str
    status: str  # passed, failed, skipped
    duration: float
    assertions: int
    coverage: float
    complexity: int
    quality_score: float
    last_modified: datetime
    failure_count: int = 0
    flakiness_score: float = 0.0
    categories: List[str] = field(default_factory=list)


@dataclass
class ModuleMetrics:
    """Metrics for a module."""
    name: str
    path: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    coverage_percentage: float
    average_quality: float
    average_duration: float
    test_density: float  # tests per 100 lines of code
    last_run: datetime
    trend: str = "stable"  # improving, declining, stable


class MetricsCollector:
    """Collects and processes test metrics."""
    
    def __init__(self):
        self.failure_patterns = defaultdict(list)
        self.metrics_cache = {}
        
    def collect_test_metrics(self, test_dir: Path) -> List[TestMetrics]:
        """Collect metrics for all tests in directory."""
        test_metrics = []
        
        for test_file in test_dir.rglob("test_*.py"):
            if "__pycache__" in str(test_file):
                continue
                
            metrics = self._analyze_test_file(test_file)
            test_metrics.extend(metrics)
        
        return test_metrics
    
    def collect_module_metrics(self, source_dir: Path, 
                              test_metrics: List[TestMetrics]) -> List[ModuleMetrics]:
        """Collect metrics for all modules."""
        module_metrics = []
        module_test_map = defaultdict(list)
        
        # Group tests by module
        for test in test_metrics:
            module_name = self._get_module_from_test(test.file_path)
            module_test_map[module_name].append(test)
        
        # Calculate metrics for each module
        for module_name, tests in module_test_map.items():
            module_path = source_dir / f"{module_name}.py"
            
            if module_path.exists():
                metrics = self._calculate_module_metrics(
                    module_name, module_path, tests
                )
                module_metrics.append(metrics)
        
        return module_metrics
    
    def _analyze_test_file(self, test_file: Path) -> List[TestMetrics]:
        """Analyze a single test file."""
        metrics = []
        
        try:
            # Parse test file for test functions
            content = test_file.read_text(encoding='utf-8')
            test_functions = self._extract_test_functions(content)
            
            for func_name in test_functions:
                metric = TestMetrics(
                    name=func_name,
                    file_path=str(test_file),
                    status="unknown",
                    duration=0.0,
                    assertions=self._count_assertions(content, func_name),
                    coverage=0.0,
                    complexity=self._calculate_complexity(content, func_name),
                    quality_score=0.0,
                    last_modified=datetime.fromtimestamp(test_file.stat().st_mtime),
                    categories=self._categorize_test(func_name)
                )
                
                # Calculate quality score
                metric.quality_score = self._calculate_quality_score(metric)
                metrics.append(metric)
                
        except Exception as e:
            logger.warning(f"Error analyzing {test_file}: {e}")
        
        return metrics
    
    def _extract_test_functions(self, content: str) -> List[str]:
        """Extract test function names from file content."""
        import ast
        
        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_"):
                        functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name.startswith("test_"):
                                functions.append(f"{node.name}.{item.name}")
        except:
            pass
        
        return functions
    
    def _count_assertions(self, content: str, func_name: str) -> int:
        """Count assertions in a test function."""
        # Simple heuristic - count assert statements
        count = 0
        in_function = False
        
        for line in content.split('\n'):
            if f"def {func_name}" in line:
                in_function = True
            elif in_function and line.strip().startswith("def "):
                break
            elif in_function and "assert" in line:
                count += 1
        
        return count
    
    def _calculate_complexity(self, content: str, func_name: str) -> int:
        """Calculate cyclomatic complexity of test."""
        # Simple heuristic based on control flow keywords
        complexity = 1
        in_function = False
        
        keywords = ['if ', 'elif ', 'else:', 'for ', 'while ', 'except:', 'finally:']
        
        for line in content.split('\n'):
            if f"def {func_name}" in line:
                in_function = True
            elif in_function and line.strip().startswith("def "):
                break
            elif in_function:
                for keyword in keywords:
                    if keyword in line:
                        complexity += 1
        
        return complexity
    
    def _categorize_test(self, test_name: str) -> List[str]:
        """Categorize test based on name."""
        categories = []
        
        # Common test categories
        if "unit" in test_name.lower():
            categories.append("unit")
        if "integration" in test_name.lower():
            categories.append("integration")
        if "edge" in test_name.lower() or "boundary" in test_name.lower():
            categories.append("edge_case")
        if "exception" in test_name.lower() or "error" in test_name.lower():
            categories.append("error_handling")
        if "performance" in test_name.lower() or "speed" in test_name.lower():
            categories.append("performance")
        
        if not categories:
            categories.append("general")
        
        return categories
    
    def _calculate_quality_score(self, metric: TestMetrics) -> float:
        """Calculate quality score for a test."""
        score = 50.0  # Base score
        
        # Adjust based on assertions
        if metric.assertions > 0:
            score += min(metric.assertions * 5, 20)
        else:
            score -= 20
        
        # Adjust based on complexity
        if metric.complexity <= 3:
            score += 10
        elif metric.complexity > 10:
            score -= 10
        
        # Adjust based on categories
        if "edge_case" in metric.categories:
            score += 5
        if "error_handling" in metric.categories:
            score += 5
        
        # Adjust based on flakiness
        score -= metric.flakiness_score * 20
        
        return max(0, min(100, score))
    
    def _get_module_from_test(self, test_path: str) -> str:
        """Extract module name from test file path."""
        # Convert test_module.py to module
        path = Path(test_path)
        name = path.stem
        
        if name.startswith("test_"):
            return name[5:]
        return name
    
    def _calculate_module_metrics(self, module_name: str, module_path: Path,
                                 tests: List[TestMetrics]) -> ModuleMetrics:
        """Calculate metrics for a module."""
        # Count test statuses
        passed = sum(1 for t in tests if t.status == "passed")
        failed = sum(1 for t in tests if t.status == "failed")
        skipped = sum(1 for t in tests if t.status == "skipped")
        
        # Calculate averages
        avg_quality = sum(t.quality_score for t in tests) / len(tests) if tests else 0
        avg_duration = sum(t.duration for t in tests) / len(tests) if tests else 0
        coverage = sum(t.coverage for t in tests) / len(tests) if tests else 0
        
        # Calculate test density
        lines_of_code = self._count_lines_of_code(module_path)
        test_density = (len(tests) / lines_of_code * 100) if lines_of_code > 0 else 0
        
        return ModuleMetrics(
            name=module_name,
            path=str(module_path),
            total_tests=len(tests),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            coverage_percentage=coverage,
            average_quality=avg_quality,
            average_duration=avg_duration,
            test_density=test_density,
            last_run=datetime.now(),
            trend=self._calculate_trend(module_name, avg_quality)
        )
    
    def _count_lines_of_code(self, file_path: Path) -> int:
        """Count lines of code in a file."""
        if not file_path.exists():
            return 0
        
        try:
            content = file_path.read_text(encoding='utf-8')
            # Count non-empty, non-comment lines
            lines = 0
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines += 1
            return lines
        except:
            return 0
    
    def _calculate_trend(self, module_name: str, current_quality: float) -> str:
        """Calculate quality trend for a module."""
        # Check cache for previous quality
        if module_name in self.metrics_cache:
            prev_quality = self.metrics_cache[module_name]
            if current_quality > prev_quality + 5:
                trend = "improving"
            elif current_quality < prev_quality - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Update cache
        self.metrics_cache[module_name] = current_quality
        return trend