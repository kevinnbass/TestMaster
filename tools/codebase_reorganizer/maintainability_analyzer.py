#!/usr/bin/env python3
"""
Maintainability Analyzer Coordinator
===================================

Coordinates the maintainability analysis using specialized modules.
"""

from typing import List
from .quality_config import QualityMetric, QualityConfig
from .maintainability_analysis import MaintainabilityAnalysis
from .maintainability_descriptions import MaintainabilityDescriptions
from .maintainability_recommendations import MaintainabilityRecommendations


class MaintainabilityAnalyzer:
    """Coordinates maintainability analysis using specialized modules"""

    def __init__(self):
        """Initialize maintainability analyzer with specialized modules"""
        self.config = QualityConfig()
        self.analysis = MaintainabilityAnalysis(self.config)
        self.descriptions = MaintainabilityDescriptions(self.config)
        self.recommendations = MaintainabilityRecommendations(self.config)

    def analyze_maintainability(self, tree, content: str) -> List[QualityMetric]:
        """Analyze maintainability metrics using specialized modules"""
        # Pre-allocate metrics with known capacity (Rule 3 compliance)
        MAX_METRICS = 10  # Expected number of metrics
        metrics = [None] * MAX_METRICS
        metrics_count = 0

        # Function length analysis
        function_length_score = self.analysis.analyze_function_lengths(tree)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Function Length",
                score=function_length_score,
                category="maintainability",
                description=self.descriptions.get_function_length_description(tree),
                recommendations=self.recommendations.get_function_length_recommendations(tree)
            )
            metrics_count += 1

        # Class complexity analysis
        class_complexity_score = self.analysis.analyze_class_complexity(tree)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Class Complexity",
                score=class_complexity_score,
                category="maintainability",
                description=self.descriptions.get_class_complexity_description(tree),
                recommendations=self.recommendations.get_class_complexity_recommendations(tree)
            )
            metrics_count += 1

        # Coupling analysis
        coupling_score = self.analysis.analyze_coupling(content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Code Coupling",
                score=coupling_score,
                category="maintainability",
                description=self.descriptions.get_coupling_description(content),
                recommendations=self.recommendations.get_coupling_recommendations(content)
            )
            metrics_count += 1

        # Testability analysis
        testability_score = self.analysis.analyze_testability(tree, content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Testability",
                score=testability_score,
                category="maintainability",
                description=self.descriptions.get_testability_description(tree, content),
                recommendations=self.recommendations.get_testability_recommendations(tree, content)
            )
            metrics_count += 1

        return metrics[:metrics_count]