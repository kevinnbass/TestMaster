#!/usr/bin/env python3
"""
Quality Configuration Module
==========================

Configuration data and constants for code quality analysis.
Contains thresholds, patterns, and quality assessment rules.
"""

from dataclasses import dataclass
from typing import Dict, List, Any


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
    file_path: str
    overall_score: float
    metrics: List[QualityMetric]
    category_scores: Dict[str, float]
    critical_issues: List[str]
    recommendations: List[str]
    quality_grade: str  # A, B, C, D, F


class QualityConfig:
    """Configuration constants for quality analysis"""

    # Complexity thresholds
    COMPLEXITY_THRESHOLDS = {
        'cyclomatic_complexity': {'good': 10, 'warning': 15, 'critical': 25},
        'nesting_depth': {'good': 3, 'warning': 5, 'critical': 7},
        'function_length': {'good': 30, 'warning': 50, 'critical': 100},
        'class_length': {'good': 200, 'warning': 400, 'critical': 600},
        'parameter_count': {'good': 3, 'warning': 6, 'critical': 10}
    }

    # Readability patterns
    READABILITY_PATTERNS = {
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
    BEST_PRACTICES = {
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
        'code_structure': {
            'imports_at_top': r'^(import|from)\s+',
            'constant_definitions': r'^[A-Z_][A-Z0-9_]*\s*=',
            'function_spacing': r'\n\ndef\s+',
            'class_spacing': r'\n\nclass\s+'
        }
    }

    # Quality scoring weights
    SCORING_WEIGHTS = {
        'complexity': 0.25,
        'maintainability': 0.25,
        'readability': 0.25,
        'best_practices': 0.25
    }

    # Grade thresholds
    GRADE_THRESHOLDS = {
        'A': 0.9,  # 90%+
        'B': 0.8,  # 80%+
        'C': 0.7,  # 70%+
        'D': 0.6,  # 60%+
        'F': 0.0   # Below 60%
    }
