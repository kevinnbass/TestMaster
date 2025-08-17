"""
TestMaster Analysis Module

Consolidated coverage analysis, measurement, and improvement capabilities.
Integrates patterns from multiple coverage analysis scripts.
"""

from .coverage_analyzer import CoverageAnalyzer, CoverageReport
from .coverage_improver import CoverageImprover, ImprovementSuggestion  
from .coverage_tracker import CoverageTracker, CoverageMetrics

__all__ = [
    "CoverageAnalyzer",
    "CoverageReport", 
    "CoverageImprover", 
    "ImprovementSuggestion",
    "CoverageTracker",
    "CoverageMetrics"
]
