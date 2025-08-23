#!/usr/bin/env python3
"""
Best Practices Analyzer Module
============================

Analyzes code for adherence to best practices including
error handling, logging, code structure, and design patterns.
"""

import re
from typing import Dict, List, Any
from .quality_config import QualityMetric, QualityConfig


class BestPracticesAnalyzer:
    """Analyzes adherence to best practices"""

    def __init__(self):
        """Initialize best practices analyzer"""
        assert QualityConfig is not None, "QualityConfig must be available"
        self.config = QualityConfig()
        assert self.config is not None, "Config initialization failed"

    def analyze_best_practices(self, content: str) -> List[QualityMetric]:
        """Analyze best practices compliance"""
        # Pre-allocate metrics with known capacity (Rule 3 compliance)
        MAX_METRICS = 10  # Expected number of metrics
        metrics = [None] * MAX_METRICS
        metrics_count = 0

        # Error handling analysis
        error_handling_score = self._analyze_error_handling(content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Error Handling",
                score=error_handling_score,
                category="best_practices",
                description=self._get_error_handling_description(content),
                recommendations=self._get_error_handling_recommendations(content)
            )
            metrics_count += 1

        # Code structure analysis
        structure_score = self._analyze_code_structure(content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Code Structure",
                score=structure_score,
                category="best_practices",
                description=self._get_structure_description(content),
                recommendations=self._get_structure_recommendations(content)
            )
            metrics_count += 1

        # Logging usage analysis
        logging_score = self._analyze_logging_usage(content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Logging Usage",
                score=logging_score,
                category="best_practices",
                description=self._get_logging_description(content),
                recommendations=self._get_logging_recommendations(content)
            )
            metrics_count += 1

        # Documentation analysis
        documentation_score = self._analyze_documentation(content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Documentation",
                score=documentation_score,
                category="best_practices",
                description=self._get_documentation_description(content),
                recommendations=self._get_documentation_recommendations(content)
            )
            metrics_count += 1

        return metrics[:metrics_count]

    def _analyze_error_handling(self, content: str) -> float:
        """Analyze error handling practices"""
        patterns = self.config.BEST_PRACTICES['error_handling']

        try_blocks = len(re.findall(patterns['try_except_blocks'], content))
        logging_usage = len(re.findall(patterns['logging_usage'], content))
        specific_exceptions = len(re.findall(patterns['exception_specificity'], content))

        # Score based on error handling patterns
        total_patterns = try_blocks + logging_usage + specific_exceptions
        if total_patterns == 0:
            return 0.3  # Minimal error handling

        # More try blocks and specific exceptions = better score
        score = min(1.0, (try_blocks * 0.3 + specific_exceptions * 0.4 + logging_usage * 0.3))
        return score

    def _analyze_code_structure(self, content: str) -> float:
        """Analyze code structure patterns"""
        patterns = self.config.BEST_PRACTICES['code_structure']

        imports_at_top = len(re.findall(patterns['imports_at_top'], content, re.MULTILINE))
        constants_at_top = len(re.findall(patterns['constant_definitions'], content, re.MULTILINE))
        function_spacing = len(re.findall(patterns['function_spacing'], content))
        class_spacing = len(re.findall(patterns['class_spacing'], content))

        # Good structure score
        structure_indicators = imports_at_top + constants_at_top + function_spacing + class_spacing
        return min(1.0, structure_indicators / 10)  # Scale appropriately

    def _analyze_logging_usage(self, content: str) -> float:
        """Analyze logging usage"""
        logging_pattern = r'logging\.'
        logging_usage = len(re.findall(logging_pattern, content))

        if logging_usage == 0:
            return 0.2  # No logging
        elif logging_usage < 3:
            return 0.6  # Some logging
        else:
            return 0.9  # Good logging coverage

    def _analyze_documentation(self, content: str) -> float:
        """Analyze documentation practices"""
        docstring_patterns = [r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"]
        comment_pattern = r'# .*'

        docstring_count = sum(len(re.findall(pattern, content)) for pattern in docstring_patterns)
        comment_count = len(re.findall(comment_pattern, content))

        lines = content.split('\n')
        total_lines = len(lines)

        if total_lines == 0:
            return 0.0

        # Documentation ratio
        doc_ratio = (docstring_count + comment_count) / total_lines
        return min(1.0, doc_ratio * 2)  # Scale up since good documentation is valuable

    def _get_error_handling_description(self, content: str) -> str:
        """Get error handling description"""
        patterns = self.config.BEST_PRACTICES['error_handling']

        try_blocks = len(re.findall(patterns['try_except_blocks'], content))
        logging_usage = len(re.findall(patterns['logging_usage'], content))
        specific_exceptions = len(re.findall(patterns['exception_specificity'], content))

        return f"Try blocks: {try_blocks}, Logging: {logging_usage}, Specific exceptions: {specific_exceptions}"

    def _get_structure_description(self, content: str) -> str:
        """Get code structure description"""
        lines = content.split('\n')

        # Count structural elements
        imports = sum(1 for line in lines if line.strip().startswith(('import ', 'from ')))
        constants = sum(1 for line in lines if re.match(r'^[A-Z_][A-Z0-9_]*\s*=', line.strip()))
        functions = sum(1 for line in lines if line.strip().startswith('def '))
        classes = sum(1 for line in lines if line.strip().startswith('class '))

        return f"Imports: {imports}, Constants: {constants}, Functions: {functions}, Classes: {classes}"

    def _get_logging_description(self, content: str) -> str:
        """Get logging usage description"""
        logging_calls = len(re.findall(r'logging\.', content))
        return f"Logging calls: {logging_calls}"

    def _get_documentation_description(self, content: str) -> str:
        """Get documentation description"""
        lines = content.split('\n')
        total_lines = len(lines)

        docstring_count = len(re.findall(r'"""[\s\S]*?"""', content)) + len(re.findall(r"'''[\s\S]*?'''", content))
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))

        return f"Docstrings: {docstring_count}, Comment lines: {comment_lines}/{total_lines}"

    def _get_error_handling_recommendations(self, content: str) -> List[str]:
        """Get error handling recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 5  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        patterns = self.config.BEST_PRACTICES['error_handling']

        try_blocks = len(re.findall(patterns['try_except_blocks'], content))
        specific_exceptions = len(re.findall(patterns['exception_specificity'], content))

        if try_blocks == 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Add try-except blocks around error-prone operations"
            rec_count += 1
        if specific_exceptions == 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Use specific exception types instead of generic Exception"
            rec_count += 1
        if specific_exceptions < try_blocks and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Specify exception types in except clauses"
            rec_count += 1

        return recommendations[:rec_count]

    def _get_structure_recommendations(self, content: str) -> List[str]:
        """Get structure recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 3  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        lines = content.split('\n')

        has_imports = any(line.strip().startswith(('import ', 'from ')) for line in lines)
        has_constants = any(re.match(r'^[A-Z_][A-Z0-9_]*\s*=', line.strip()) for line in lines)

        if not has_imports and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Consider organizing imports at the top of the file"
            rec_count += 1
        if not has_constants and any(re.match(r'^[A-Z_][A-Z0-9_]*\s*=', line.strip()) for line in lines) and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Group constants at the top after imports"
            rec_count += 1

        return recommendations[:rec_count]

    def _get_logging_recommendations(self, content: str) -> List[str]:
        """Get logging recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 4  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        logging_calls = len(re.findall(r'logging\.', content))

        if logging_calls == 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Add logging statements for important operations"
            rec_count += 1
            if rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = "Use appropriate logging levels (DEBUG, INFO, WARNING, ERROR)"
                rec_count += 1
        elif logging_calls < 3 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Consider adding more logging for better observability"
            rec_count += 1

        return recommendations[:rec_count]

    def _get_documentation_recommendations(self, content: str) -> List[str]:
        """Get documentation recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 3  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        lines = content.split('\n')
        total_lines = len(lines)

        docstring_count = len(re.findall(r'"""[\s\S]*?"""', content)) + len(re.findall(r"'''[\s\S]*?'''", content))
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))

        if docstring_count == 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Add docstrings to functions and classes"
            rec_count += 1
        if comment_lines / total_lines < 0.1 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = "Add more inline comments to explain complex logic"
            rec_count += 1

        return recommendations[:rec_count]
