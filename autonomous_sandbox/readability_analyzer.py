#!/usr/bin/env python3
"""
Readability Analyzer Module
==========================

Analyzes code readability including naming conventions,
comments, documentation, and code formatting.
"""

import re
from typing import Dict, List, Any
from .quality_config import QualityMetric, QualityConfig
from .quality_utils import QualityUtils


class ReadabilityAnalyzer:
    """Analyzes code readability metrics"""

    def __init__(self):
        """Initialize readability analyzer"""
        assert QualityConfig is not None, "QualityConfig must be available"
        self.config = QualityConfig()
        assert self.config is not None, "Config initialization failed"

    def analyze_readability(self, content: str, tree: ast.Module) -> List[QualityMetric]:
        """Analyze readability metrics"""
        # Pre-allocate metrics with known capacity (Rule 3 compliance)
        MAX_METRICS = 10  # Expected number of metrics
        metrics = [None] * MAX_METRICS
        metrics_count = 0

        # Overall readability score
        readability_score = QualityUtils.calculate_readability_score(content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Readability Score",
                score=readability_score,
                category="readability",
                description=".2f",
                recommendations=self._get_readability_recommendations(readability_score)
            )
            metrics_count += 1

        # Code metrics
        code_metrics = QualityUtils.extract_code_metrics(content)
        metrics_score = self._calculate_metrics_score(code_metrics)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Code Metrics",
                score=metrics_score,
                category="readability",
                description=f"Code: {code_metrics['code_lines']}, Comments: {code_metrics['comment_lines']}, Blank: {code_metrics['blank_lines']}",
                recommendations=self._get_metrics_recommendations(code_metrics)
            )
            metrics_count += 1

        # Naming conventions
        naming_results = QualityUtils.check_naming_conventions(content)
        naming_score = self._calculate_naming_score(naming_results)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Naming Conventions",
                score=naming_score,
                category="readability",
                description=f"Functions: {naming_results['snake_case_functions']}, Classes: {naming_results['PascalCase_classes']}",
                recommendations=self._get_naming_recommendations(naming_results)
            )
            metrics_count += 1

        # Identifier lengths
        length_analysis = QualityUtils.analyze_identifier_lengths(tree)
        length_score = self._calculate_length_score(length_analysis)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Identifier Lengths",
                score=length_score,
                category="readability",
                description=f"Long identifiers: {sum(1 for lens in length_analysis.values() for l in lens if l > 25)}",
                recommendations=self._get_length_recommendations(length_analysis)
            )
            metrics_count += 1

        return metrics[:metrics_count]

    def _calculate_metrics_score(self, metrics: Dict[str, int]) -> float:
        """Calculate score based on code metrics"""
        total_lines = metrics['total_lines']
        if total_lines == 0:
            return 0.0

        code_ratio = metrics['code_lines'] / total_lines
        comment_ratio = metrics['comment_lines'] / total_lines
        docstring_ratio = metrics['docstring_lines'] / total_lines

        # Ideal ratios: 60% code, 20% comments, 10% docstrings
        code_score = 1.0 - abs(code_ratio - 0.6)
        comment_score = 1.0 - abs(comment_ratio - 0.2)
        docstring_score = 1.0 - abs(docstring_ratio - 0.1)

        return max(0.0, (code_score + comment_score + docstring_score) / 3)

    def _calculate_naming_score(self, naming_results: Dict[str, int]) -> float:
        """Calculate naming convention score"""
        total_items = sum(naming_results.values()) - naming_results.get('violations', 0)
        if total_items == 0:
            return 1.0

        # Calculate compliance rate
        compliant = naming_results['snake_case_functions'] + naming_results['PascalCase_classes'] + naming_results['UPPER_CASE_constants']
        compliance_rate = compliant / total_items

        return compliance_rate

    def _calculate_length_score(self, length_analysis: Dict[str, List[int]]) -> float:
        """Calculate identifier length score"""
        # Pre-allocate all_lengths with known capacity (Rule 3 compliance)
        MAX_TOTAL_LENGTHS = 1000  # Safety bound for total lengths
        all_lengths = [None] * MAX_TOTAL_LENGTHS
        length_count = 0

        # Bounded loop for length analysis (Rule 2 compliance)
        MAX_CATEGORIES = 10  # Safety bound for categories
        category_items = list(length_analysis.items())
        for i in range(min(len(category_items), MAX_CATEGORIES)):
            category, lengths = category_items[i]
            for j in range(min(len(lengths), MAX_TOTAL_LENGTHS - length_count)):
                if length_count < MAX_TOTAL_LENGTHS:
                    all_lengths[length_count] = lengths[j]
                    length_count += 1

        if length_count == 0:
            return 1.0

        # Count problematic lengths using bounded approach
        too_short = sum(1 for i in range(length_count) if all_lengths[i] and all_lengths[i] < 3)
        too_long = sum(1 for i in range(length_count) if all_lengths[i] and all_lengths[i] > 25)

        penalty = (too_short + too_long) / length_count
        return max(0.0, 1.0 - penalty)

    def _get_readability_recommendations(self, score: float) -> List[str]:
        """Get readability recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 10  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        if score < 0.7:
            rec_items = [
                "Add more comments to explain complex logic",
                "Use descriptive variable and function names",
                "Break long lines into multiple lines for readability",
                "Add proper spacing between logical blocks"
            ]
            for i, item in enumerate(rec_items):
                if rec_count + i < MAX_RECOMMENDATIONS:
                    recommendations[rec_count + i] = item
            rec_count += len(rec_items)

        if score < 0.5:
            rec_items = [
                "Consider adding docstrings to all functions and classes",
                "Use consistent indentation and formatting",
                "Avoid very long functions - break them into smaller functions"
            ]
            for i, item in enumerate(rec_items):
                if rec_count + i < MAX_RECOMMENDATIONS:
                    recommendations[rec_count + i] = item
            rec_count += len(rec_items)

        return recommendations[:rec_count]

    def _get_metrics_recommendations(self, metrics: Dict[str, int]) -> List[str]:
        """Get metrics-based recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 5  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        total_lines = metrics['total_lines']

        if total_lines > 0:
            comment_ratio = metrics['comment_lines'] / total_lines
            if comment_ratio < 0.1 and rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = "Add more comments to explain code logic"
                rec_count += 1
            elif comment_ratio > 0.4 and rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = "Review if all comments are necessary"
                rec_count += 1

            if metrics['blank_lines'] < total_lines * 0.05 and rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = "Add more blank lines to separate logical blocks"
                rec_count += 1

        return recommendations[:rec_count]

    def _get_naming_recommendations(self, naming_results: Dict[str, int]) -> List[str]:
        """Get naming convention recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 5  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        if naming_results['violations'] > 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Review naming conventions - use snake_case for functions, PascalCase for classes"
            rec_count += 1

        if naming_results['snake_case_functions'] == 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Use snake_case for function names (e.g., calculate_score)"
            rec_count += 1

        if naming_results['PascalCase_classes'] == 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Use PascalCase for class names (e.g., CodeAnalyzer)"
            rec_count += 1

        return recommendations[:rec_count]

    def _get_length_recommendations(self, length_analysis: Dict[str, List[int]]) -> List[str]:
        """Get identifier length recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 5  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        for category, lengths in length_analysis.items():
            # Replace complex comprehension with simple loop (Rule 1 compliance)
            long_count = 0
            for i, length in enumerate(lengths):
                if length > 25:
                    long_count += 1
            if long_count > 0 and rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = f"Consider shortening long {category} names"
                rec_count += 1

        return recommendations[:rec_count]
