#!/usr/bin/env python3
"""
Code Quality Analyzer
====================

Main coordinator for comprehensive code quality analysis.
Uses specialized analyzers for different aspects of code quality.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import specialized analyzers
from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.sandbox.complexity_analyzer import ComplexityAnalyzer
from .readability_analyzer import ReadabilityAnalyzer
from .maintainability_analyzer import MaintainabilityAnalyzer
from .best_practices_analyzer import BestPracticesAnalyzer
from .quality_config import QualityAnalysis, QualityConfig


class CodeQualityAnalyzer:
    """
    Main code quality analyzer that coordinates specialized analyzers
    """

    def __init__(self) -> None:
        """Initialize the code quality analyzer"""
        self.complexity_analyzer = ComplexityAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.maintainability_analyzer = MaintainabilityAnalyzer()
        self.best_practices_analyzer = BestPracticesAnalyzer()
        self.config = QualityConfig()

    def _parse_file_content(self, file_path: Path) -> tuple:
        """Parse file content and return AST tree (helper function)"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        tree = ast.parse(content)
        return content, tree

    def _run_all_analyses(self, content: str, tree: ast.Module) -> List:
        """Run all specialized analyses (helper function)"""
        # Pre-allocate all_metrics with known capacity (Rule 3 compliance)
        MAX_METRICS = 50  # Expected total number of metrics
        all_metrics = [None] * MAX_METRICS
        metrics_count = 0

        # Complexity analysis
        complexity_metrics = self.complexity_analyzer.analyze_complexity(tree, content)
        for i, metric in enumerate(complexity_metrics):
            if metrics_count + i < MAX_METRICS:
                all_metrics[metrics_count + i] = metric
        metrics_count += len(complexity_metrics)

        # Readability analysis
        readability_metrics = self.readability_analyzer.analyze_readability(content, tree)
        for i, metric in enumerate(readability_metrics):
            if metrics_count + i < MAX_METRICS:
                all_metrics[metrics_count + i] = metric
        metrics_count += len(readability_metrics)

        # Maintainability analysis
        maintainability_metrics = self.maintainability_analyzer.analyze_maintainability(tree, content)
        for i, metric in enumerate(maintainability_metrics):
            if metrics_count + i < MAX_METRICS:
                all_metrics[metrics_count + i] = metric
        metrics_count += len(maintainability_metrics)

        # Best practices analysis
        best_practices_metrics = self.best_practices_analyzer.analyze_best_practices(content)
        for i, metric in enumerate(best_practices_metrics):
            if metrics_count + i < MAX_METRICS:
                all_metrics[metrics_count + i] = metric
        metrics_count += len(best_practices_metrics)

        return all_metrics[:metrics_count]

    def _calculate_quality_metrics(self, all_metrics: List) -> tuple:
        """Calculate overall quality metrics (helper function)"""
        category_scores = self._calculate_category_scores(all_metrics)
        overall_score = self._calculate_overall_score(category_scores)
        quality_grade = self._calculate_grade(overall_score)
        return category_scores, overall_score, quality_grade

    def _extract_analysis_results(self, all_metrics: List) -> tuple:
        """Extract critical issues and recommendations (helper function)"""
        critical_issues = self._extract_critical_issues(all_metrics)
        recommendations = self._extract_recommendations(all_metrics)
        return critical_issues, recommendations

    def analyze_file(self, file_path: Path) -> QualityAnalysis:
        """Analyze a single file for quality metrics (coordinator function)"""
        try:
            # Parse file content using helper
            content, tree = self._parse_file_content(file_path)

            # Run all analyses using helper
            all_metrics = self._run_all_analyses(content, tree)

            # Calculate quality metrics using helper
            category_scores, overall_score, quality_grade = self._calculate_quality_metrics(all_metrics)

            # Extract analysis results using helper
            critical_issues, recommendations = self._extract_analysis_results(all_metrics)

            return QualityAnalysis(
                file_path=str(file_path),
                overall_score=overall_score,
                metrics=all_metrics,
                category_scores=category_scores,
                critical_issues=critical_issues,
                recommendations=recommendations,
                quality_grade=quality_grade
            )

        except Exception as e:
            return QualityAnalysis(
                file_path=str(file_path),
                overall_score=0.0,
                metrics=[],
                category_scores={},
                critical_issues=[f"Analysis failed: {str(e)}"],
                recommendations=["Fix syntax errors before analysis"],
                quality_grade="F"
            )

    def _calculate_category_scores(self, metrics: List) -> Dict[str, float]:
        """Calculate scores by category"""
        category_scores = {}
        # Pre-allocate category storage with known capacity (Rule 3 compliance)
        MAX_CATEGORIES = 10  # Expected number of categories
        category_metrics = {}

        # Group metrics by category with bounded loop (Rule 2 compliance)
        MAX_METRICS_PROCESS = 100  # Safety bound for metrics processing
        for i in range(min(len(metrics), MAX_METRICS_PROCESS)):
            metric = metrics[i]
            if metric.category not in category_metrics:
                category_metrics[metric.category] = []
            if len(category_metrics[metric.category]) < 50:  # Safety bound per category
                category_metrics[metric.category].append(metric.score)

        # Calculate average score per category with bounded loop
        category_items = list(category_metrics.items())
        for i in range(min(len(category_items), MAX_CATEGORIES)):
            category, scores = category_items[i]
            if scores:
                category_scores[category] = sum(scores) / len(scores)
            else:
                category_scores[category] = 0.0

        return category_scores

    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        if not category_scores:
            return 0.0

        weights = self.config.SCORING_WEIGHTS
        weighted_score = 0.0
        total_weight = 0.0

        for category, score in category_scores.items():
            weight = weights.get(category, 0.25)
            weighted_score += score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _calculate_grade(self, score: float) -> str:
        """Calculate quality grade"""
        for grade, threshold in self.config.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"

    def _extract_critical_issues(self, metrics: List) -> List[str]:
        """Extract critical issues from metrics"""
        # Pre-allocate critical_issues with known capacity (Rule 3 compliance)
        MAX_CRITICAL_ISSUES = 20  # Expected number of critical issues
        critical_issues = [None] * MAX_CRITICAL_ISSUES
        issues_count = 0

        # Bounded loop for metrics processing (Rule 2 compliance)
        MAX_METRICS_PROCESS = 100  # Safety bound for metrics processing
        for i in range(min(len(metrics), MAX_METRICS_PROCESS)):
            metric = metrics[i]
            if metric.score < 0.4 and issues_count < MAX_CRITICAL_ISSUES:  # Critical threshold
                critical_issues[issues_count] = f"{metric.name}: {metric.description}"
                issues_count += 1

        return critical_issues[:issues_count]

    def _extract_recommendations(self, metrics: List) -> List[str]:
        """Extract recommendations from all metrics"""
        # Pre-allocate all_recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 50  # Expected total number of recommendations
        all_recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        # Bounded loop for metrics processing (Rule 2 compliance)
        MAX_METRICS_PROCESS = 100  # Safety bound for metrics processing
        for i in range(min(len(metrics), MAX_METRICS_PROCESS)):
            metric = metrics[i]
            if metric.score < 0.7:  # Include recommendations for scores below good
                for j, rec in enumerate(metric.recommendations):
                    if rec_count + j < MAX_RECOMMENDATIONS:
                        all_recommendations[rec_count + j] = rec
                rec_count += len(metric.recommendations)

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = [None] * MAX_RECOMMENDATIONS
        unique_count = 0

        for i in range(min(rec_count, MAX_RECOMMENDATIONS)):
            rec = all_recommendations[i]
            if rec and rec not in seen and unique_count < MAX_RECOMMENDATIONS:
                seen.add(rec)
                unique_recommendations[unique_count] = rec
                unique_count += 1

        return unique_recommendations[:unique_count]


# Legacy functions for backward compatibility
def analyze_quality(content: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    analyzer = CodeQualityAnalyzer()
    if file_path is None:
        file_path = Path("unknown.py")

    analysis = analyzer.analyze_file(file_path)

    return {
        'overall_score': analysis.overall_score,
        'grade': analysis.quality_grade,
        'metrics': [{'name': m.name, 'score': m.score, 'category': m.category,
                    'description': m.description, 'recommendations': m.recommendations}
                   for m in analysis.metrics],
        'recommendations': analysis.recommendations,
        'critical_issues': analysis.critical_issues
    }


def get_quality_score(content: str) -> float:
    """Legacy function for backward compatibility"""
    analyzer = CodeQualityAnalyzer()
    file_path = Path("temp.py")

    # Write content to temporary file for analysis
    with open(file_path, 'w') as f:
        f.write(content)

    analysis = analyzer.analyze_file(file_path)

    # Clean up temp file
    file_path.unlink(missing_ok=True)

    return analysis.overall_score


def get_quality_grade(content: str) -> str:
    """Legacy function for backward compatibility"""
    analyzer = CodeQualityAnalyzer()
    file_path = Path("temp.py")

    # Write content to temporary file for analysis
    with open(file_path, 'w') as f:
        f.write(content)

    analysis = analyzer.analyze_file(file_path)

    # Clean up temp file
    file_path.unlink(missing_ok=True)

    return analysis.quality_grade