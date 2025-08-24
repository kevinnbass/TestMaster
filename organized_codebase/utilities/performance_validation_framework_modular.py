#!/usr/bin/env python3
"""
Performance Validation Framework - Modular Implementation
=========================================================

This file provides backward compatibility for the original performance_validation_framework.py
after STEELCLAD modularization into separate components.

All original functionality is preserved through imports from child modules.
"""

# Import all components from modular implementation
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_models import PerformanceTest, PerformanceResult, LoadTestScenario
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_benchmarker import PerformanceBenchmarker
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.cc_1.load_test_executor import LoadTestExecutor
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_regression_detector import PerformanceRegressionDetector
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_validation_core import PerformanceValidationFramework

# Re-export all components for backward compatibility
__all__ = [
    'PerformanceTest',
    'PerformanceResult',
    'LoadTestScenario',
    'PerformanceBenchmarker',
    'LoadTestExecutor',
    'PerformanceRegressionDetector',
    'PerformanceValidationFramework'
]

# Maintain original module docstring for compatibility
__doc__ = """
AGENT BETA - PERFORMANCE VALIDATION FRAMEWORK
Phase 1, Hours 20-25: Initial Performance Validation
===================================================

Comprehensive performance validation system with benchmarking, load testing,
regression detection, and performance measurement reporting.

This module has been modularized via STEELCLAD protocol into:
- performance_models.py: Data structures and type definitions
- performance_benchmarker.py: Core benchmarking engine
- load_test_executor.py: Load testing implementation
- performance_regression_detector.py: Regression analysis
- performance_validation_core.py: Main framework orchestration

All original functionality is preserved and accessible through this module.
"""

# Version information
__version__ = '2.0.0'
__author__ = 'Agent Beta (modularized by Agent C)'
__created__ = '2025-08-23 02:55:00 UTC'
__modularized__ = '2025-08-23 07:50:00 UTC'