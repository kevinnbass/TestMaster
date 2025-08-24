#!/usr/bin/env python3
"""
Pattern Data Classes
====================

Data structures for pattern detection results.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DetectedPattern:
    """A detected pattern in the code"""
    pattern_type: str
    pattern_name: str
    confidence: float
    location: str  # class name, function name, or 'module'
    evidence: List[str]
    category: str  # 'design', 'architectural', 'coding', 'anti-pattern'
    recommendations: List[str]


@dataclass
class PatternAnalysis:
    """Complete pattern analysis results"""
    patterns: List[DetectedPattern]
    pattern_categories: Dict[str, List[str]]
    confidence_distribution: Dict[str, int]
    architectural_style: str
    coding_style: str
    recommendations: List[str]


@dataclass
class PatternDefinition:
    """Definition of a detectable pattern"""
    name: str
    category: str
    indicators: List[str]  # Regex patterns to look for
    keywords: List[str]  # Keywords that suggest this pattern
    structure: List[str]  # Structural requirements
    confidence_weight: float  # Base confidence score
    description: str


@dataclass
class PatternDetectionResult:
    """Result of pattern detection in a specific location"""
    pattern_name: str
    confidence: float
    evidence: List[str]
    location: str
    recommendations: List[str]

