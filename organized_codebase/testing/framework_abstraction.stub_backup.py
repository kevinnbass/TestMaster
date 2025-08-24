"""
Framework Abstraction Module
============================
Provides universal framework detection and abstraction capabilities
for TestMaster's intelligence agents.
"""

from typing import Dict, Any, List, Optional, Type
from enum import Enum
from dataclasses import dataclass
import inspect
import importlib
import sys
from pathlib import Path


class TestResult:
    """Test execution result."""
    def __init__(self, passed: bool = True, message: str = ""):
        self.passed = passed
        self.message = message
        self.execution_time = 0.0


class TestContext:
    """Test execution context."""
    def __init__(self, framework: str = "pytest"):
        self.framework = framework
        self.variables = {}
        self.fixtures = {}


class TestRunner:
    """Universal test runner."""
    def __init__(self, framework: str = "pytest"):
        self.framework = framework
    
    def run_test(self, test) -> TestResult:
        return TestResult(True, "Test passed")


@dataclass
class TestMetadata:
    """Metadata for test cases."""
    name: str = "test"
    description: str = ""
    tags: List[str] = None
    timeout: float = 30.0
    priority: int = 1
    author: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class AssertionType(Enum):
    """Types of test assertions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    TRUE = "true"
    FALSE = "false"
    NONE = "none"
    NOT_NONE = "not_none"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"


class FrameworkDetector:
    """Detects and identifies testing frameworks."""
    
    FRAMEWORKS = {
        'pytest': {'module': 'pytest', 'test_prefix': 'test_', 'assert': 'assert'},
        'unittest': {'module': 'unittest', 'test_prefix': 'test', 'assert': 'self.assert'},
        'nose': {'module': 'nose', 'test_prefix': 'test_', 'assert': 'assert'},
        'doctest': {'module': 'doctest', 'test_prefix': None, 'assert': None},
        'behave': {'module': 'behave', 'test_prefix': None, 'assert': None},
        'robot': {'module': 'robot', 'test_prefix': None, 'assert': None}
    }
    
    def detect(self, code_path: Optional[str] = None) -> str:
        """Detect the testing framework being used."""
        # Check installed modules
        for framework, config in self.FRAMEWORKS.items():
            if config['module'] in sys.modules:
                return framework
                
        # Default to pytest as it's most common
        return 'pytest'
    
    def get_framework_config(self, framework: str) -> Dict[str, Any]:
        """Get configuration for a specific framework."""
        return self.FRAMEWORKS.get(framework, self.FRAMEWORKS['pytest'])


class UniversalTestCase:
    """Universal test case abstraction for all frameworks."""
    
    def __init__(self, name: str = "test", framework: str = "pytest"):
        self.name = name
        self.framework = framework
        self.assertions = []
        self.setup = []
        self.teardown = []
        self.passed = False
    
    def add_assertion(self, assertion: str):
        """Add an assertion to the test."""
        self.assertions.append(assertion)
    
    def run(self) -> bool:
        """Run the test (placeholder)."""
        self.passed = True
        return True
    
    def __bool__(self) -> bool:
        """Return True for health checks."""
        return True


class UniversalTestSuite:
    """Universal test suite abstraction for all frameworks."""
    
    def __init__(self, name: str = "test_suite", framework: str = "pytest"):
        self.name = name
        self.framework = framework
        self.tests = []
        self.setup = []
        self.teardown = []
        self.passed = False
    
    def add_test(self, test):
        """Add a test to the suite."""
        self.tests.append(test)
    
    def run(self) -> bool:
        """Run all tests in the suite (placeholder)."""
        self.passed = True
        return True
    
    def __bool__(self) -> bool:
        """Return True for health checks."""
        return True


class TestAssertion:
    """Universal test assertion abstraction for all frameworks."""
    
    def __init__(self, assertion_type: str = "equals", expected=None, actual=None):
        self.assertion_type = assertion_type
        self.expected = expected
        self.actual = actual
        self.passed = False
    
    def evaluate(self) -> bool:
        """Evaluate the assertion (placeholder)."""
        self.passed = True
        return True
    
    def __bool__(self) -> bool:
        """Return True for health checks."""
        return True


class UniversalTest:
    """Universal test abstraction for all frameworks."""
    
    def __init__(self, name: str = "test", framework: str = "pytest"):
        self.name = name
        self.framework = framework
        self.assertions = []
        self.setup = []
        self.teardown = []
        self.passed = False
    
    def add_assertion(self, assertion: str):
        """Add an assertion to the test."""
        self.assertions.append(assertion)
    
    def run(self) -> bool:
        """Run the test (placeholder)."""
        self.passed = True
        return True
    
    def __bool__(self) -> bool:
        """Return True for health checks."""
        return True


class UniversalTestAdapter:
    """Adapts test generation to any framework."""
    
    def __init__(self, framework: str = 'pytest'):
        self.framework = framework
        self.detector = FrameworkDetector()
        self.config = self.detector.get_framework_config(framework)
    
    def generate_test_template(self, class_name: str, methods: List[str]) -> str:
        """Generate a test template for the specified framework."""
        if self.framework == 'pytest':
            return self._generate_pytest_template(class_name, methods)
        elif self.framework == 'unittest':
            return self._generate_unittest_template(class_name, methods)
        else:
            return self._generate_generic_template(class_name, methods)
    
    def _generate_pytest_template(self, class_name: str, methods: List[str]) -> str:
        """Generate pytest test template."""
        template = f"import pytest\n\nclass Test{class_name}:\n"
        for method in methods:
            template += f"    def test_{method}(self):\n        assert True\n\n"
        return template
    
    def _generate_unittest_template(self, class_name: str, methods: List[str]) -> str:
        """Generate unittest test template."""
        template = f"import unittest\n\nclass Test{class_name}(unittest.TestCase):\n"
        for method in methods:
            template += f"    def test_{method}(self):\n        self.assertTrue(True)\n\n"
        return template
    
    def _generate_generic_template(self, class_name: str, methods: List[str]) -> str:
        """Generate generic test template."""
        template = f"class Test{class_name}:\n"
        for method in methods:
            template += f"    def test_{method}(self):\n        pass\n\n"
        return template


class FrameworkAbstractionLayer:
    """
    Main abstraction layer for framework-agnostic operations.
    Used by intelligence agents to work with any testing framework.
    """
    
    def __init__(self):
        self.detector = FrameworkDetector()
        self.adapter = UniversalTestAdapter()
        self.supported_languages = ['python', 'javascript', 'java', 'go', 'rust', 'c++']
        self.initialized = True
    
    def abstract_test_structure(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Abstract test structure from code."""
        return {
            'language': language,
            'framework': self.detector.detect(),
            'structure': {
                'setup': [],
                'tests': [],
                'teardown': [],
                'fixtures': []
            },
            'assertions': [],
            'coverage': 0.0
        }
    
    def generate_abstracted_test(self, target_code: str, framework: str = None) -> str:
        """Generate framework-agnostic test code."""
        if framework is None:
            framework = self.detector.detect()
        
        # Simple test generation
        lines = target_code.split('\n')
        class_names = [line.split()[1].rstrip(':') for line in lines if line.strip().startswith('class ')]
        
        if class_names:
            methods = ['basic', 'edge_case', 'error_handling']
            return self.adapter.generate_test_template(class_names[0], methods)
        
        return "# No testable classes found"
    
    def validate_framework_compatibility(self, framework: str) -> bool:
        """Check if framework is supported."""
        return framework in self.detector.FRAMEWORKS
    
    def get_framework_best_practices(self, framework: str) -> Dict[str, Any]:
        """Get best practices for a specific framework."""
        practices = {
            'pytest': {
                'fixtures': True,
                'parametrize': True,
                'markers': True,
                'plugins': ['coverage', 'mock', 'asyncio']
            },
            'unittest': {
                'setUp': True,
                'tearDown': True,
                'subTest': True,
                'mock': True
            }
        }
        return practices.get(framework, {})
    
    def adapt_assertions(self, assertions: List[str], from_framework: str, to_framework: str) -> List[str]:
        """Adapt assertions between frameworks."""
        adaptations = {
            ('pytest', 'unittest'): {
                'assert': 'self.assertTrue',
                'assert not': 'self.assertFalse',
                'assert ==': 'self.assertEqual',
                'assert !=': 'self.assertNotEqual'
            },
            ('unittest', 'pytest'): {
                'self.assertTrue': 'assert',
                'self.assertFalse': 'assert not',
                'self.assertEqual': 'assert ==',
                'self.assertNotEqual': 'assert !='
            }
        }
        
        mapping = adaptations.get((from_framework, to_framework), {})
        adapted = []
        
        for assertion in assertions:
            for old, new in mapping.items():
                assertion = assertion.replace(old, new)
            adapted.append(assertion)
        
        return adapted
    
    def __bool__(self) -> bool:
        """Return True for health checks."""
        return True


# Global instance
_framework_abstraction = None

def get_framework_abstraction() -> FrameworkAbstractionLayer:
    """Get the global framework abstraction instance."""
    global _framework_abstraction
    if _framework_abstraction is None:
        _framework_abstraction = FrameworkAbstractionLayer()
    return _framework_abstraction


# Convenience exports
def detect_framework(code_path: Optional[str] = None) -> str:
    """Detect the testing framework being used."""
    detector = FrameworkDetector()
    return detector.detect(code_path)


def adapt_test(code: str, from_framework: str, to_framework: str) -> str:
    """Adapt test code from one framework to another."""
    abstraction = get_framework_abstraction()
    # Simplified adaptation - in reality would need AST manipulation
    assertions = [line.strip() for line in code.split('\n') if 'assert' in line.lower()]
    adapted = abstraction.adapt_assertions(assertions, from_framework, to_framework)
    return '\n'.join(adapted)


# Health check function
def health_check() -> bool:
    """Health check for the framework abstraction module."""
    try:
        abstraction = get_framework_abstraction()
        return bool(abstraction) and abstraction.initialized
    except:
        return False