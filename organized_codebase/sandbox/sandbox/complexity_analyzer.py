#!/usr/bin/env python3
"""
Complexity Analyzer Module
=========================

Analyzes code complexity metrics including cyclomatic complexity,
nesting depth, and other complexity-related measurements.
"""

import ast
from typing import Dict, List, Any
from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.sandbox.quality_config import QualityMetric, QualityConfig
from .quality_utils import QualityUtils


class ComplexityAnalyzer:
    """Analyzes code complexity metrics"""

    def __init__(self):
        """Initialize complexity analyzer"""
        assert QualityConfig is not None, "QualityConfig must be available"
        self.config = QualityConfig()
        assert self.config is not None, "Config initialization failed"

    def analyze_complexity(self, tree: ast.Module, content: str) -> List[QualityMetric]:
        """Analyze complexity metrics"""
        # Pre-allocate metrics with known capacity (Rule 3 compliance)
        MAX_METRICS = 10  # Expected number of metrics
        metrics = [None] * MAX_METRICS
        metrics_count = 0

        # Cyclomatic complexity
        complexity = QualityUtils.calculate_cyclomatic_complexity(tree)
        complexity_score = self._calculate_complexity_score(complexity, 'cyclomatic_complexity')
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Cyclomatic Complexity",
                score=complexity_score,
                category="complexity",
                description=f"Cyclomatic complexity: {complexity}",
                recommendations=self._get_complexity_recommendations('cyclomatic_complexity', complexity)
            )
            metrics_count += 1

        # Nesting depth
        nesting_depth = QualityUtils.calculate_nesting_depth(tree)
        nesting_score = self._calculate_complexity_score(nesting_depth, 'nesting_depth')
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Nesting Depth",
                score=nesting_score,
                category="complexity",
                description=f"Maximum nesting depth: {nesting_depth}",
                recommendations=self._get_complexity_recommendations('nesting_depth', nesting_depth)
            )
            metrics_count += 1

        # Function and class counts
        functions, classes = QualityUtils.count_functions_and_classes(tree)
        size_score = self._calculate_size_score(functions, classes)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Code Structure",
                score=size_score,
                category="complexity",
                description=f"Functions: {functions}, Classes: {classes}",
                recommendations=self._get_structure_recommendations(functions, classes)
            )
            metrics_count += 1

        return metrics[:metrics_count]

    def _calculate_complexity_score(self, value: int, metric_type: str) -> float:
        """Calculate complexity score based on thresholds"""
        thresholds = self.config.COMPLEXITY_THRESHOLDS[metric_type]

        if value <= thresholds['good']:
            return 1.0
        elif value <= thresholds['warning']:
            return 0.7
        elif value <= thresholds['critical']:
            return 0.4
        else:
            return 0.1

    def _calculate_size_score(self, functions: int, classes: int) -> float:
        """Calculate code size score"""
        # Simple scoring based on function/class counts
        if functions <= 10 and classes <= 5:
            return 1.0
        elif functions <= 20 and classes <= 10:
            return 0.7
        elif functions <= 30 and classes <= 15:
            return 0.4
        else:
            return 0.1

    def _get_complexity_recommendations(self, metric_type: str, value: int) -> List[str]:
        """Get recommendations for complexity issues"""
        thresholds = self.config.COMPLEXITY_THRESHOLDS[metric_type]

        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 8  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        if value > thresholds['critical']:
            if metric_type == 'cyclomatic_complexity':
                rec_items = [
                    "Break down complex functions into smaller, focused functions",
                    "Extract conditional logic into separate functions",
                    "Use early returns to reduce nested conditions"
                ]
                for i, item in enumerate(rec_items):
                    if rec_count + i < MAX_RECOMMENDATIONS:
                        recommendations[rec_count + i] = item
                rec_count += len(rec_items)
            elif metric_type == 'nesting_depth':
                rec_items = [
                    "Extract nested blocks into separate functions",
                    "Use early returns to avoid deep nesting",
                    "Consider using guard clauses"
                ]
                for i, item in enumerate(rec_items):
                    if rec_count + i < MAX_RECOMMENDATIONS:
                        recommendations[rec_count + i] = item
                rec_count += len(rec_items)
        elif value > thresholds['warning']:
            if metric_type == 'cyclomatic_complexity' and rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = "Consider refactoring to reduce complexity"
                rec_count += 1
            elif metric_type == 'nesting_depth' and rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = "Review nesting structure for simplification"
                rec_count += 1

        return recommendations[:rec_count]

    def _get_structure_recommendations(self, functions: int, classes: int) -> List[str]:
        """Get recommendations for code structure issues"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 4  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        if functions > 30 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Consider breaking large modules into smaller, focused modules"
            rec_count += 1
        if classes > 15 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Review class hierarchy - consider splitting into multiple modules"
            rec_count += 1
        if functions == 0 and classes == 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Consider adding functions or classes for better organization"
            rec_count += 1

        return recommendations[:rec_count]