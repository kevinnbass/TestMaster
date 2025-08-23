#!/usr/bin/env python3
"""
Code Quality Analyzer
====================

Comprehensive code quality analysis module that evaluates various aspects
of code quality including complexity, maintainability, readability, and
best practices compliance.

Key capabilities:
- Complexity analysis (cyclomatic complexity, nesting depth)
- Maintainability assessment
- Readability scoring
- Best practices compliance checking
- Code quality metrics aggregation
- Quality recommendations generation
"""

import ast
import re
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class QualityMetric:
    """Individual quality metric"""
    name: str
    score: float  # 0-1 scale
    category: str  # 'complexity', 'maintainability', 'readability', 'practices'
    description: str
    recommendations: List[str]


@dataclass
class QualityAnalysis:
    """Complete quality analysis results"""
    file_path: Path
    overall_score: float
    metrics: List[QualityMetric]
    category_scores: Dict[str, float]
    critical_issues: List[str]
    recommendations: List[str]
    quality_grade: str  # A, B, C, D, F


class CodeQualityAnalyzer:
    """
    Comprehensive code quality analyzer that evaluates multiple aspects
    of code quality and provides actionable recommendations.
    """

    def __init__(self) -> None:
        """Initialize the code quality analyzer"""
        self._load_quality_rules()

    def _load_quality_rules(self) -> None:
        """Load quality assessment rules and thresholds"""

        # Complexity thresholds
        self.complexity_thresholds = {
            'cyclomatic_complexity': {'good': 10, 'warning': 15, 'critical': 25},
            'nesting_depth': {'good': 3, 'warning': 5, 'critical': 7},
            'function_length': {'good': 30, 'warning': 50, 'critical': 100},
            'class_length': {'good': 200, 'warning': 400, 'critical': 600},
            'parameter_count': {'good': 3, 'warning': 6, 'critical': 10}
        }

        # Readability patterns
        self.readability_patterns = {
            'good': [
                r'# .*',  # Comments
                r'"""[\s\S]*?"""',  # Docstrings
                r"'''[\s\S]*?'''",  # Docstrings
                r'\n\s*\n',  # Proper spacing
                r'class\s+\w+:',  # Class definitions
                r'def\s+\w+\s*\([^)]*\)\s*->\s*\w+:',  # Typed function definitions
            ],
            'bad': [
                r'[A-Z]{3,}',  # ALL CAPS (potential constants without proper naming)
                r'\w{30,}',  # Very long identifiers
                r'[^\w\s]{3,}',  # Multiple consecutive symbols
            ]
        }

        # Best practices
        self.best_practices = {
            'naming_conventions': {
                'snake_case_functions': r'def\s+([a-z_][a-z0-9_]*)\s*\(',
                'PascalCase_classes': r'class\s+([A-Z][a-zA-Z0-9]*)\s*:',
                'UPPER_CASE_constants': r'^[A-Z_][A-Z0-9_]*\s*='
            },
            'error_handling': {
                'try_except_blocks': r'try\s*:',
                'logging_usage': r'logging\.',
                'exception_specificity': r'except\s+[A-Z]\w+Error:'
            },
            'documentation': {
                'docstrings': r'""".*?"""|\'\'\'.*?\'\'\'',
                'inline_comments': r'# .*',
                'type_hints': r'def\s+\w+\s*\([^)]*\)\s*->'
            }
        }

        # Anti-patterns
        self.anti_patterns = {
            'code_smells': [
                r'print\(',  # Debug prints in production
                r'pass$',  # Empty except blocks
                r'except.*:',  # Bare except clauses
                r'global\s+\w+',  # Global variables
                r'eval\(',  # Use of eval
                r'exec\(',  # Use of exec
            ]
        }

    def analyze_quality(self, content: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Perform comprehensive code quality analysis.

        Args:
            content: The code content to analyze
            file_path: Optional path to the file

        Returns:
            Dictionary containing quality analysis results
        """
        try:
            tree = ast.parse(content)

            # Perform various quality analyses
            complexity_metrics = self._analyze_complexity(content, tree)
            maintainability_metrics = self._analyze_maintainability(content, tree)
            readability_metrics = self._analyze_readability(content, tree)
            practice_metrics = self._analyze_best_practices(content, tree)

            # Combine all metrics
            all_metrics = complexity_metrics + maintainability_metrics + readability_metrics + practice_metrics

            # Calculate category scores
            category_scores = self._calculate_category_scores(all_metrics)

            # Calculate overall score
            overall_score = self._calculate_overall_score(category_scores)

            # Generate critical issues and recommendations
            critical_issues = self._identify_critical_issues(all_metrics)
            recommendations = self._generate_recommendations(all_metrics, overall_score)

            # Determine quality grade
            quality_grade = self._determine_quality_grade(overall_score)

            result = {
                'file_path': str(file_path) if file_path else 'unknown',
                'overall_score': overall_score,
                'quality_grade': quality_grade,
                'category_scores': category_scores,
                'metrics': [metric.__dict__ if hasattr(metric, '__dict__') else metric for metric in all_metrics],
                'critical_issues': critical_issues,
                'recommendations': recommendations,
                'total_metrics': len(all_metrics),
                'passing_metrics': len([m for m in all_metrics if getattr(m, 'score', 0) >= 0.7])
            }

            return result

        except SyntaxError as e:
            return self._fallback_quality_analysis(content, file_path, e)
        except Exception as e:
            return {
                'error': f'Quality analysis failed: {e}',
                'file_path': str(file_path) if file_path else 'unknown',
                'overall_score': 0.0,
                'quality_grade': 'F',
                'category_scores': {},
                'metrics': [],
                'critical_issues': ['Unable to analyze due to error'],
                'recommendations': ['Fix syntax errors before quality analysis'],
                'total_metrics': 0,
                'passing_metrics': 0
            }

    def _analyze_complexity(self, content: str, tree: ast.AST) -> List[QualityMetric]:
        """Analyze code complexity metrics"""
        metrics = []

        # Cyclomatic complexity analysis
        total_complexity = 0
        function_count = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
                complexity = self._calculate_cyclomatic_complexity(node)
                total_complexity += complexity

        if function_count > 0:
            avg_complexity = total_complexity / function_count

            complexity_score = self._score_complexity(avg_complexity, 'cyclomatic_complexity')
            metrics.append(QualityMetric(
                name='cyclomatic_complexity',
                score=complexity_score,
                category='complexity',
                description=f'Average cyclomatic complexity: {avg_complexity:.1f}',
                recommendations=self._get_complexity_recommendations('cyclomatic_complexity', avg_complexity)
            ))

        # Nesting depth analysis
        max_nesting = self._calculate_max_nesting(tree)
        nesting_score = self._score_complexity(max_nesting, 'nesting_depth')
        metrics.append(QualityMetric(
            name='nesting_depth',
            score=nesting_score,
            category='complexity',
            description=f'Maximum nesting depth: {max_nesting}',
            recommendations=self._get_complexity_recommendations('nesting_depth', max_nesting)
        ))

        # Function length analysis
        avg_function_length = self._calculate_average_function_length(tree)
        if avg_function_length > 0:
            length_score = self._score_complexity(avg_function_length, 'function_length')
            metrics.append(QualityMetric(
                name='function_length',
                score=length_score,
                category='complexity',
                description=f'Average function length: {avg_function_length:.1f} lines',
                recommendations=self._get_complexity_recommendations('function_length', avg_function_length)
            ))

        return metrics

    def _analyze_maintainability(self, content: str, tree: ast.AST) -> List[QualityMetric]:
        """Analyze code maintainability"""
        metrics = []

        # Calculate lines of code
        lines_of_code = len([line for line in content.split('\n') if line.strip()])

        # Calculate comment ratio
        comment_lines = len(re.findall(r'^\s*#.*', content, re.MULTILINE))
        docstring_lines = len(re.findall(r'"""[\s\S]*?"""', content)) + len(re.findall(r"'''[\s\S]*?'''", content))

        if lines_of_code > 0:
            comment_ratio = (comment_lines + docstring_lines) / lines_of_code
            maintainability_score = min(comment_ratio * 3, 1.0)  # Scale up to make it meaningful

            metrics.append(QualityMetric(
                name='comment_ratio',
                score=maintainability_score,
                category='maintainability',
                description='.1%',
                recommendations=self._get_maintainability_recommendations('comment_ratio', comment_ratio)
            ))

        # Check for TODO/FIXME comments
        todo_count = len(re.findall(r'#\s*(TODO|FIXME|XXX|HACK)', content, re.IGNORECASE))
        if todo_count > 0:
            todo_score = max(0, 1.0 - (todo_count * 0.1))
            metrics.append(QualityMetric(
                name='technical_debt',
                score=todo_score,
                category='maintainability',
                description=f'Found {todo_count} TODO/FIXME comments',
                recommendations=self._get_maintainability_recommendations('technical_debt', todo_count)
            ))

        # Check for duplicated code patterns (simple heuristic)
        duplicate_patterns = self._detect_duplicate_patterns(content)
        if duplicate_patterns > 0:
            duplication_score = max(0, 1.0 - (duplicate_patterns * 0.1))
            metrics.append(QualityMetric(
                name='code_duplication',
                score=duplication_score,
                category='maintainability',
                description=f'Found {duplicate_patterns} potential duplicate patterns',
                recommendations=self._get_maintainability_recommendations('code_duplication', duplicate_patterns)
            ))

        return metrics

    def _analyze_readability(self, content: str, tree: ast.AST) -> List[QualityMetric]:
        """Analyze code readability"""
        metrics = []

        # Calculate average line length
        lines = content.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0

        # Score based on line length (79 chars is PEP8 recommended)
        if avg_line_length <= 79:
            readability_score = 1.0
        elif avg_line_length <= 100:
            readability_score = 0.8
        elif avg_line_length <= 120:
            readability_score = 0.6
        else:
            readability_score = 0.4

        metrics.append(QualityMetric(
            name='line_length',
            score=readability_score,
            category='readability',
            description='.1f',
            recommendations=self._get_readability_recommendations('line_length', avg_line_length)
        ))

        # Check for consistent naming
        naming_score = self._analyze_naming_consistency(tree)
        metrics.append(QualityMetric(
            name='naming_consistency',
            score=naming_score,
            category='readability',
            description=f'Naming consistency score: {naming_score:.2f}',
            recommendations=self._get_readability_recommendations('naming_consistency', naming_score)
        ))

        # Check for proper whitespace usage
        whitespace_score = self._analyze_whitespace_usage(content)
        metrics.append(QualityMetric(
            name='whitespace_usage',
            score=whitespace_score,
            category='readability',
            description=f'Whitespace usage score: {whitespace_score:.2f}',
            recommendations=self._get_readability_recommendations('whitespace_usage', whitespace_score)
        ))

        return metrics

    def _analyze_best_practices(self, content: str, tree: ast.AST) -> List[QualityMetric]:
        """Analyze compliance with best practices"""
        metrics = []

        # Check for type hints usage
        type_hint_score = self._analyze_type_hints(tree)
        metrics.append(QualityMetric(
            name='type_hints',
            score=type_hint_score,
            category='practices',
            description=f'Type hints coverage: {type_hint_score:.2f}',
            recommendations=self._get_practice_recommendations('type_hints', type_hint_score)
        ))

        # Check for proper error handling
        error_handling_score = self._analyze_error_handling(content, tree)
        metrics.append(QualityMetric(
            name='error_handling',
            score=error_handling_score,
            category='practices',
            description=f'Error handling quality: {error_handling_score:.2f}',
            recommendations=self._get_practice_recommendations('error_handling', error_handling_score)
        ))

        # Check for anti-patterns
        anti_pattern_score = self._analyze_anti_patterns(content)
        if anti_pattern_score < 1.0:
            metrics.append(QualityMetric(
                name='anti_patterns',
                score=anti_pattern_score,
                category='practices',
                description=f'Anti-pattern detection: {anti_pattern_score:.2f}',
                recommendations=self._get_practice_recommendations('anti_patterns', anti_pattern_score)
            ))

        return metrics

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and len(child.values) > 2:
                complexity += len(child.values) - 2
                            # Note: ast.Match is only available in Python 3.10+

        return complexity

    def _calculate_max_nesting(self, tree: ast.AST, max_depth: int = 0, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        for node in ast.walk(tree):
                            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                max_depth = max(max_depth, current_depth + 1)
                if hasattr(node, 'body'):
                    max_depth = max(max_depth, self._calculate_max_nesting_from_body(node.body, current_depth + 1))
                if hasattr(node, 'orelse') and node.orelse:
                    max_depth = max(max_depth, self._calculate_max_nesting_from_body(node.orelse, current_depth + 1))

        return max_depth

    def _calculate_max_nesting_from_body(self, body: List[ast.stmt], current_depth: int) -> int:
        """Helper method to calculate nesting from body"""
        max_depth = current_depth
        for stmt in body:
            if hasattr(stmt, 'body'):
                max_depth = max(max_depth, self._calculate_max_nesting_from_body(stmt.body, current_depth + 1))
        return max_depth

    def _calculate_average_function_length(self, tree: ast.AST) -> float:
        """Calculate average function length in lines"""
        function_lengths = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, 'body'):
                    # Estimate lines based on AST nodes (rough approximation)
                    function_length = len(node.body) * 2  # Rough estimate
                    function_lengths.append(function_length)

        return sum(function_lengths) / len(function_lengths) if function_lengths else 0

    def _score_complexity(self, value: float, complexity_type: str) -> float:
        """Score complexity based on thresholds"""
        thresholds = self.complexity_thresholds.get(complexity_type, {})

        good = thresholds.get('good', 0)
        warning = thresholds.get('warning', 0)
        critical = thresholds.get('critical', 0)

        if value <= good:
            return 1.0
        elif value <= warning:
            return 0.7
        elif value <= critical:
            return 0.4
        else:
            return 0.1

    def _detect_duplicate_patterns(self, content: str) -> int:
        """Simple duplicate pattern detection"""
        # Look for repeated patterns (very basic implementation)
        lines = content.split('\n')
                    pattern_count: Dict[str, int] = defaultdict(int)

        for line in lines:
            if len(line.strip()) > 10:  # Only consider substantial lines
                pattern_count[line.strip()] += 1

        # Count patterns that appear more than once
        duplicates = sum(1 for count in pattern_count.values() if count > 1)
        return duplicates

    def _analyze_naming_consistency(self, tree: ast.AST) -> float:
        """Analyze naming consistency"""
        names = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.FunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                names.append(node.name)

        if not names:
            return 1.0

        # Simple heuristic: check for mixed naming conventions
        snake_case = sum(1 for name in names if re.match(r'^[a-z_][a-z0-9_]*$', name))
        camel_case = sum(1 for name in names if re.match(r'^[a-z][a-zA-Z0-9]*$', name))
        pascal_case = sum(1 for name in names if re.match(r'^[A-Z][a-zA-Z0-9]*$', name))

        total_names = len(names)
        dominant_convention = max(snake_case, camel_case, pascal_case)

        return dominant_convention / total_names if total_names > 0 else 1.0

    def _analyze_whitespace_usage(self, content: str) -> float:
        """Analyze whitespace usage consistency"""
        lines = content.split('\n')

        # Check for consistent indentation
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)

        if not indent_levels:
            return 1.0

        # Calculate variance in indentation
        avg_indent = sum(indent_levels) / len(indent_levels)
        variance = sum((x - avg_indent) ** 2 for x in indent_levels) / len(indent_levels)
        std_dev = math.sqrt(variance)

        # Lower standard deviation means more consistent indentation
        if std_dev < 2:
            return 1.0
        elif std_dev < 4:
            return 0.8
        elif std_dev < 8:
            return 0.6
        else:
            return 0.4

    def _analyze_type_hints(self, tree: ast.AST) -> float:
        """Analyze type hints usage"""
        total_functions = 0
        typed_functions = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                has_return_hint = node.returns is not None
                has_arg_hints = all(arg.annotation is not None for arg in node.args.args if arg.arg != 'self')

                if has_return_hint and has_arg_hints:
                    typed_functions += 1

        return typed_functions / total_functions if total_functions > 0 else 1.0

    def _analyze_error_handling(self, content: str, tree: ast.AST) -> float:
        """Analyze error handling quality"""
        score = 0.0
        factors = 0

        # Check for try-except blocks
        try_count = len(re.findall(r'\btry\s*:', content))
        except_count = len(re.findall(r'\bexcept\s+', content))

        if try_count > 0:
            score += min(try_count / 5.0, 1.0)
            factors += 1

        # Check for specific exception handling
        specific_exceptions = len(re.findall(r'except\s+[A-Z]\w+Error:', content))
        if except_count > 0:
            score += specific_exceptions / except_count
            factors += 1

        # Check for logging usage
        if 'logging.' in content or 'logger.' in content:
            score += 0.5
            factors += 1

        return score / factors if factors > 0 else 0.5

    def _analyze_anti_patterns(self, content: str) -> float:
        """Analyze anti-pattern usage"""
        violations = 0

        for pattern in self.anti_patterns['code_smells']:
            matches = len(re.findall(pattern, content))
            violations += matches

        # Scale the score based on violations
        if violations == 0:
            return 1.0
        elif violations <= 3:
            return 0.8
        elif violations <= 7:
            return 0.6
        elif violations <= 10:
            return 0.4
        else:
            return 0.2

    def _calculate_category_scores(self, metrics: List[QualityMetric]) -> Dict[str, float]:
        """Calculate average scores by category"""
        category_scores = defaultdict(list)

        for metric in metrics:
            category_scores[metric.category].append(metric.score)

        return {
            category: sum(scores) / len(scores) if scores else 0.0
            for category, scores in category_scores.items()
        }

    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        if not category_scores:
            return 0.0

        # Weight categories differently
        weights = {
            'complexity': 0.25,
            'maintainability': 0.25,
            'readability': 0.2,
            'practices': 0.3
        }

        weighted_score = 0.0
        total_weight = 0.0

        for category, score in category_scores.items():
            weight = weights.get(category, 0.2)
            weighted_score += score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _identify_critical_issues(self, metrics: List[QualityMetric]) -> List[str]:
        """Identify critical quality issues"""
        issues = []

        for metric in metrics:
            if metric.score < 0.4:
                issues.append(f"Critical: {metric.name} - {metric.description}")

        return issues

    def _determine_quality_grade(self, overall_score: float) -> str:
        """Determine quality grade based on overall score"""
        if overall_score >= 0.9:
            return 'A'
        elif overall_score >= 0.8:
            return 'B'
        elif overall_score >= 0.7:
            return 'C'
        elif overall_score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _generate_recommendations(self, metrics: List[QualityMetric],
                                overall_score: float) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        # Add specific recommendations from metrics
        for metric in metrics:
            if metric.score < 0.7:
                recommendations.extend(metric.recommendations[:2])  # Limit to top 2 per metric

        # Add general recommendations based on overall score
        if overall_score < 0.6:
            recommendations.append("CRITICAL: Major code quality improvements needed")
            recommendations.append("Consider refactoring or rewriting complex sections")
        elif overall_score < 0.7:
            recommendations.append("Significant quality improvements recommended")
            recommendations.append("Focus on reducing complexity and improving readability")
        elif overall_score < 0.8:
            recommendations.append("Moderate quality improvements suggested")
            recommendations.append("Address specific metrics with low scores")

        # Remove duplicates and limit to top 10
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:10]

    def _get_complexity_recommendations(self, complexity_type: str, value: float) -> List[str]:
        """Get recommendations for complexity issues"""
        recommendations = {
            'cyclomatic_complexity': [
                "Break down complex functions into smaller functions",
                "Extract conditional logic into separate methods",
                "Consider using early returns to reduce nesting"
            ],
            'nesting_depth': [
                "Extract deeply nested code into separate functions",
                "Use early returns to reduce nesting levels",
                "Consider using guard clauses"
            ],
            'function_length': [
                "Split long functions into smaller, focused functions",
                "Extract complex logic into helper methods",
                "Consider using the single responsibility principle"
            ]
        }
        return recommendations.get(complexity_type, ["Reduce code complexity"])

    def _get_maintainability_recommendations(self, maintainability_type: str, value: Any) -> List[str]:
        """Get recommendations for maintainability issues"""
        if maintainability_type == 'comment_ratio':
            return [
                "Add more comments and docstrings",
                "Document complex algorithms and business logic",
                "Use descriptive variable and function names"
            ]
        elif maintainability_type == 'technical_debt':
            return [
                "Address TODO and FIXME comments",
                "Schedule time to resolve technical debt",
                "Document known issues and limitations"
            ]
        elif maintainability_type == 'code_duplication':
            return [
                "Extract common code into shared functions",
                "Create utility modules for repeated patterns",
                "Consider using inheritance or composition to reduce duplication"
            ]
        return ["Improve code maintainability"]

    def _get_readability_recommendations(self, readability_type: str, value: Any) -> List[str]:
        """Get recommendations for readability issues"""
        if readability_type == 'line_length':
            return [
                "Break long lines into multiple lines",
                "Use parentheses for line continuation",
                "Extract complex expressions into variables"
            ]
        elif readability_type == 'naming_consistency':
            return [
                "Follow consistent naming conventions",
                "Use descriptive variable and function names",
                "Follow PEP 8 naming guidelines"
            ]
        elif readability_type == 'whitespace_usage':
            return [
                "Use consistent indentation",
                "Add proper spacing between functions and classes",
                "Follow PEP 8 whitespace guidelines"
            ]
        return ["Improve code readability"]

    def _get_practice_recommendations(self, practice_type: str, value: Any) -> List[str]:
        """Get recommendations for best practice issues"""
        if practice_type == 'type_hints':
            return [
                "Add type hints to function parameters and return types",
                "Use Union types for parameters that can have multiple types",
                "Consider using TypedDict for complex data structures"
            ]
        elif practice_type == 'error_handling':
            return [
                "Add proper try-except blocks around risky operations",
                "Use specific exception types instead of bare except",
                "Add logging for error conditions"
            ]
        elif practice_type == 'anti_patterns':
            return [
                "Remove debug print statements",
                "Replace eval/exec with safer alternatives",
                "Fix bare except clauses with specific exceptions"
            ]
        return ["Follow coding best practices"]

    def _fallback_quality_analysis(self, content: str, file_path: Optional[Path],
                                 error: SyntaxError) -> Dict[str, Any]:
        """Fallback quality analysis for files with syntax errors"""
        return {
            'file_path': str(file_path) if file_path else 'unknown',
            'overall_score': 0.0,
            'quality_grade': 'F',
            'category_scores': {},
            'metrics': [],
            'critical_issues': [f'Contains syntax error: {error}'],
            'recommendations': ['Fix syntax errors before quality analysis'],
            'total_metrics': 0,
            'passing_metrics': 0
        }


# Module-level functions for easy integration
def analyze_quality(content: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Module-level function for quality analysis"""
    analyzer = CodeQualityAnalyzer()
    return analyzer.analyze_quality(content, file_path)


def get_quality_score(content: str) -> float:
    """Get overall quality score for code content"""
    analyzer = CodeQualityAnalyzer()
    result = analyzer.analyze_quality(content)
    return result.get('overall_score', 0.0)


def get_quality_grade(content: str) -> str:
    """Get quality grade for code content"""
    analyzer = CodeQualityAnalyzer()
    result = analyzer.analyze_quality(content)
    return result.get('quality_grade', 'F')


if __name__ == "__main__":
    # Example usage
    sample_code = """
import os
import sys
from typing import List, Dict, Optional

def complex_function(a, b, c, d, e, f, g):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                return a + b + c + d + e + f + g
                            else:
                                return a + b + c + d + e + f
                        else:
                            return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0

class VeryLongClassWithManyMethods:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
"""

    analyzer = CodeQualityAnalyzer()
    result = analyzer.analyze_quality(sample_code, Path("sample.py"))

    print("Code Quality Analysis Results:")
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"Quality Grade: {result['quality_grade']}")
    print(f"Category Scores: {result['category_scores']}")
    print(f"Critical Issues: {result['critical_issues'][:3]}")
    print(f"Top Recommendations: {result['recommendations'][:3]}")
