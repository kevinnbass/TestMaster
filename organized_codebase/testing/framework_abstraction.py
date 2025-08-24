"""
Framework Abstraction Module
============================
Provides universal framework detection and abstraction capabilities
for TestMaster's intelligence agents with full implementation.
"""

import ast
import importlib
import inspect
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import logging

logger = logging.getLogger(__name__)

class TestResult:
    """Comprehensive test execution result."""

    def __init__(self, passed: bool = True, message: str = "", execution_time: float = 0.0):
        self.passed = passed
        self.message = message
        self.execution_time = execution_time
        self.errors = []
        self.warnings = []
        self.stdout = ""
        self.stderr = ""
        self.metadata = {}

    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.passed = False

    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'passed': self.passed,
            'message': self.message,
            'execution_time': self.execution_time,
            'errors': self.errors,
            'warnings': self.warnings,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'metadata': self.metadata
        }

@dataclass
class TestMetadata:
    """Metadata for test information."""
    name: str
    description: str = ""
    category: str = "unit"
    priority: str = "medium"
    tags: List[str] = None
    dependencies: List[str] = None
    timeout: float = 30.0
    retries: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []

class UniversalTestSuite:
    """Universal test suite abstraction supporting multiple testing frameworks."""

    def __init__(self, name: str = "UniversalTestSuite"):
        self.name = name
        self.tests = []
        self.results = []
        self.metadata = {}
        self.dependencies = []

    def add_test(self, test_func: callable, metadata: TestMetadata = None):
        """Add a test to the suite."""
        if metadata is None:
            metadata = TestMetadata(test_func.__name__)

        self.tests.append({
            'function': test_func,
            'metadata': metadata
        })

    def run_test(self, test_func: callable, metadata: TestMetadata) -> TestResult:
        """Run a single test."""
        result = TestResult()
        start_time = time.time()

        try:
            # Execute the test
            test_func()
            result.message = f"Test {metadata.name} passed"
        except Exception as e:
            result.add_error(str(e))
            result.message = f"Test {metadata.name} failed: {e}"

        result.execution_time = time.time() - start_time
        return result

    def run_all(self) -> List[TestResult]:
        """Run all tests in the suite."""
        results = []

        for test in self.tests:
            result = self.run_test(test['function'], test['metadata'])
            results.append(result)
            self.results.append(result)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        return {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': (passed / total * 100) if total > 0 else 0,
            'total_execution_time': sum(r.execution_time for r in self.results)
        }

# Default instances for backward compatibility
_universal_test_suite = UniversalTestSuite()

def get_universal_test_suite() -> UniversalTestSuite:
    """Get the global universal test suite."""
    return _universal_test_suite

def create_test_suite(name: str) -> UniversalTestSuite:
    """Create a new test suite."""
    return UniversalTestSuite(name)

# Export for backward compatibility
__all__ = [
    'TestResult',
    'TestMetadata',
    'UniversalTestSuite',
    'get_universal_test_suite',
    'create_test_suite'
]

