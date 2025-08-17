"""
TestMaster Breaking Detection Module

Real-time failure detection and analysis inspired by PraisonAI's
performance monitoring and error categorization patterns.
"""

from .failure_detector import FailureDetector, FailureReport
from .regression_tracker import RegressionTracker, RegressionPattern
from .error_categorizer import ErrorCategorizer, ErrorCategory

__all__ = [
    "FailureDetector",
    "FailureReport",
    "RegressionTracker", 
    "RegressionPattern",
    "ErrorCategorizer",
    "ErrorCategory"
]