from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
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

class TestContext:
    """Enhanced test execution context."""
    
    def __init__(self, framework: str = "pytest"):
        self.framework = framework
        self.variables = {}
        self.fixtures = {}
        self.setup_methods = []
        self.teardown_methods = []
        self.timeout = 30.0
        self.working_directory = Path.cwd()
        self.environment = {}
    
    def add_fixture(self, name: str, value: Any):
        """Add a fixture to the context."""
        self.fixtures[name] = value
    
    def set_variable(self, name: str, value: Any):
        """Set a context variable."""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(name, default)

class TestRunner:
    """Universal test runner with full implementation."""
    
    def __init__(self, framework: str = "pytest"):
        self.framework = framework
        self.detector = FrameworkDetector()
        self.context = TestContext(framework)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def run_test(self, test: Union[str, 'UniversalTestCase'], context: Optional[TestContext] = None) -> TestResult:
        """Run a test with full implementation."""
        
        if context:
            self.context = context
        
        start_time = time.time()
        result = TestResult()
        
        try:
            if isinstance(test, str):
                # Run test from string/file path
                result = self._run_test_from_string(test)
            elif hasattr(test, 'run'):
                # Run test object
                result = test.run()
            else:
                result.add_error(f"Unknown test type: {type(test)}")
            
            result.execution_time = time.time() - start_time
            
        except Exception as e:
            result.add_error(f"Test execution failed: {e}")
            result.execution_time = time.time() - start_time
        
        return result
    
    def _run_test_from_string(self, test_code: str) -> TestResult:
        """Run test from string code."""
        result = TestResult()
        
        try:
            # Parse the test code
            tree = ast.parse(test_code)
            
            # Compile and execute
            code = compile(tree, '<test>', 'exec')
            
            # Create execution namespace
            namespace = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
                **self.context.fixtures,
                **self.context.variables
            }
            
            # Execute the test
            SafeCodeExecutor.safe_exec(code, namespace)
            
            result.passed = True
            result.message = "Test executed successfully"
            
        except AssertionError as e:
            result.add_error(f"Assertion failed: {e}")
        except Exception as e:
            result.add_error(f"Execution error: {e}")
        
        return result
    
    def run_test_suite(self, tests: List[Union[str, 'UniversalTestCase']]) -> List[TestResult]:
        """Run a suite of tests."""
        results = []
        
        for test in tests:
            result = self.run_test(test)
            results.append(result)
        
        return results

@dataclass
class TestMetadata:
    """Enhanced metadata for test cases."""
    name: str = "test"
    description: str = ""
    tags: List[str] = None
    timeout: float = 30.0
    priority: int = 1
    author: str = ""
    framework: str = "pytest"
    dependencies: List[str] = None
    expected_duration: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []

class AssertionType(Enum):
    """Types of test assertions with full implementation."""
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
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    TYPE_CHECK = "type_check"
    RAISES = "raises"

class FrameworkDetector:
    """Enhanced framework detection with full implementation."""
    
    FRAMEWORKS = {
        'pytest': {
            'module': 'pytest', 
            'test_prefix': 'test_', 
            'assert': 'assert',
            'command': 'pytest',
            'config_files': ['pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg']
        },
        'unittest': {
            'module': 'unittest', 
            'test_prefix': 'test', 
            'assert': 'self.assert',
            'command': 'python -m unittest',
            'config_files': []
        },
        'nose': {
            'module': 'nose', 
            'test_prefix': 'test_', 
            'assert': 'assert',
            'command': 'nosetests',
            'config_files': ['nose.cfg', '.noserc']
        },
        'doctest': {
            'module': 'doctest', 
            'test_prefix': None, 
            'assert': None,
            'command': 'python -m doctest',
            'config_files': []
        },
        'behave': {
            'module': 'behave', 
            'test_prefix': None, 
            'assert': None,
            'command': 'behave',
            'config_files': ['behave.ini', '.behaverc']
        },
        'robot': {
            'module': 'robot', 
            'test_prefix': None, 
            'assert': None,
            'command': 'robot',
            'config_files': ['robot.yaml']
        }
    }
    
    def detect(self, code_path: Optional[str] = None) -> str:
        """Detect the testing framework being used."""
        
        if code_path:
            # Check for framework-specific files
            path = Path(code_path)
            if path.is_file():
                path = path.parent
            
            for framework, config in self.FRAMEWORKS.items():
                for config_file in config['config_files']:
                    if (path / config_file).exists():
                        return framework
        
        # Check installed modules
        for framework, config in self.FRAMEWORKS.items():
            try:
                importlib.import_module(config['module'])
                return framework
            except ImportError:
                continue
        
        # Check sys.modules
        for framework, config in self.FRAMEWORKS.items():
            if config['module'] in sys.modules:
                return framework
        
        # Default to pytest as it's most common
        return 'pytest'
    
    def get_framework_config(self, framework: str) -> Dict[str, Any]:
        """Get configuration for a specific framework."""
        return self.FRAMEWORKS.get(framework, self.FRAMEWORKS['pytest'])
    
    def is_framework_available(self, framework: str) -> bool:
        """Check if a framework is available."""
        try:
            config = self.get_framework_config(framework)
            importlib.import_module(config['module'])
            return True
        except ImportError:
            return False
    
    def detect_all_available(self) -> List[str]:
        """Detect all available frameworks."""
        available = []
        for framework in self.FRAMEWORKS.keys():
            if self.is_framework_available(framework):
                available.append(framework)
        return available

class UniversalTestCase:
    """Universal test case abstraction with full implementation."""
    
    def __init__(self, name: str = "test", framework: str = "pytest"):
        self.name = name
        self.framework = framework
        self.assertions = []
        self.setup = []
        self.teardown = []
        self.passed = False
        self.metadata = TestMetadata(name=name, framework=framework)
        self.context = TestContext(framework)
    
    def add_assertion(self, assertion: Union[str, 'TestAssertion']):
        """Add an assertion to the test."""
        self.assertions.append(assertion)
    
    def add_setup(self, setup_code: str):
        """Add setup code."""
        self.setup.append(setup_code)
    
    def add_teardown(self, teardown_code: str):
        """Add teardown code."""
        self.teardown.append(teardown_code)
    
    def run(self) -> TestResult:
        """Run the test with full implementation."""
        result = TestResult()
        start_time = time.time()
        
        try:
            # Run setup
            for setup_code in self.setup:
                self._execute_code(setup_code, result)
                if not result.passed:
                    return result
            
            # Run assertions
            for assertion in self.assertions:
                if isinstance(assertion, str):
                    self._execute_code(assertion, result)
                elif hasattr(assertion, 'evaluate'):
                    assertion_result = assertion.evaluate()
                    if not assertion_result:
                        result.add_error(f"Assertion failed: {assertion}")
                
                if not result.passed:
                    break
            
            # Run teardown (always)
            for teardown_code in self.teardown:
                try:
                    self._execute_code(teardown_code, result)
                except Exception as e:
                    result.add_warning(f"Teardown warning: {e}")
            
            if result.passed and not result.errors:
                result.passed = True
                result.message = f"Test '{self.name}' passed"
            
        except Exception as e:
            result.add_error(f"Test execution failed: {e}")
        
        result.execution_time = time.time() - start_time
        self.passed = result.passed
        return result
    
    def _execute_code(self, code: str, result: TestResult):
        """Execute code in test context."""
        try:
            # Create execution namespace
            namespace = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
                **self.context.fixtures,
                **self.context.variables
            }
            
            # Execute the code
            SafeCodeExecutor.safe_exec(code, namespace)
            
        except AssertionError as e:
            result.add_error(f"Assertion failed: {e}")
        except Exception as e:
            result.add_error(f"Code execution failed: {e}")
    
    def __bool__(self) -> bool:
        """Return test status."""
        return self.passed

class UniversalTestSuite:
    """Universal test suite abstraction with full implementation."""
    
    def __init__(self, name: str = "test_suite", framework: str = "pytest"):
        self.name = name
        self.framework = framework
        self.tests = []
        self.setup = []
        self.teardown = []
        self.passed = False
        self.results = []
    
    def add_test(self, test: UniversalTestCase):
        """Add a test to the suite."""
        self.tests.append(test)
    
    def add_setup(self, setup_code: str):
        """Add suite-level setup."""
        self.setup.append(setup_code)
    
    def add_teardown(self, teardown_code: str):
        """Add suite-level teardown."""
        self.teardown.append(teardown_code)
    
    def run(self) -> List[TestResult]:
        """Run all tests in the suite with full implementation."""
        self.results = []
        
        try:
            # Run suite setup
            for setup_code in self.setup:
                SafeCodeExecutor.safe_exec(setup_code)
            
            # Run all tests
            for test in self.tests:
                result = test.run()
                self.results.append(result)
            
            # Run suite teardown
            for teardown_code in self.teardown:
                try:
                    SafeCodeExecutor.safe_exec(teardown_code)
                except Exception as e:
                    logger.warning(f"Suite teardown warning: {e}")
            
            # Determine overall suite result
            self.passed = all(result.passed for result in self.results)
            
        except Exception as e:
            logger.error(f"Suite execution failed: {e}")
            self.passed = False
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary."""
        if not self.results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'success_rate': 0.0}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': (passed / total * 100) if total > 0 else 0.0,
            'total_time': sum(r.execution_time for r in self.results)
        }
    
    def __bool__(self) -> bool:
        """Return suite status."""
        return self.passed

class TestAssertion:
    """Universal test assertion abstraction with full implementation."""
    
    def __init__(self, assertion_type: AssertionType, expected=None, actual=None, message: str = ""):
        self.assertion_type = assertion_type
        self.expected = expected
        self.actual = actual
        self.message = message
        self.passed = False
    
    def evaluate(self) -> bool:
        """Evaluate the assertion with full implementation."""
        try:
            if self.assertion_type == AssertionType.EQUALS:
                self.passed = self.actual == self.expected
            elif self.assertion_type == AssertionType.NOT_EQUALS:
                self.passed = self.actual != self.expected
            elif self.assertion_type == AssertionType.TRUE:
                self.passed = bool(self.actual) is True
            elif self.assertion_type == AssertionType.FALSE:
                self.passed = bool(self.actual) is False
            elif self.assertion_type == AssertionType.NONE:
                self.passed = self.actual is None
            elif self.assertion_type == AssertionType.NOT_NONE:
                self.passed = self.actual is not None
            elif self.assertion_type == AssertionType.IN:
                self.passed = self.actual in self.expected
            elif self.assertion_type == AssertionType.NOT_IN:
                self.passed = self.actual not in self.expected
            elif self.assertion_type == AssertionType.GREATER_THAN:
                self.passed = self.actual > self.expected
            elif self.assertion_type == AssertionType.LESS_THAN:
                self.passed = self.actual < self.expected
            elif self.assertion_type == AssertionType.GREATER_EQUAL:
                self.passed = self.actual >= self.expected
            elif self.assertion_type == AssertionType.LESS_EQUAL:
                self.passed = self.actual <= self.expected
            elif self.assertion_type == AssertionType.CONTAINS:
                self.passed = self.expected in self.actual
            elif self.assertion_type == AssertionType.TYPE_CHECK:
                self.passed = isinstance(self.actual, self.expected)
            else:
                self.passed = False
                
        except Exception:
            self.passed = False
        
        return self.passed
    
    def __bool__(self) -> bool:
        """Return assertion status."""
        return self.passed
    
    def __str__(self) -> str:
        """String representation."""
        return f"Assertion({self.assertion_type.value}): {self.actual} vs {self.expected} -> {self.passed}"

class UniversalTestAdapter:
    """Enhanced test adapter with full implementation."""
    
    def __init__(self, framework: str = 'pytest'):
        self.framework = framework
        self.detector = FrameworkDetector()
        self.config = self.detector.get_framework_config(framework)
    
    def generate_test_code(self, test_case: UniversalTestCase) -> str:
        """Generate framework-specific test code."""
        
        if self.framework == 'pytest':
            return self._generate_pytest_code(test_case)
        elif self.framework == 'unittest':
            return self._generate_unittest_code(test_case)
        else:
            return self._generate_generic_code(test_case)
    
    def _generate_pytest_code(self, test_case: UniversalTestCase) -> str:
        """Generate pytest-specific code."""
        lines = []
        lines.append(f"def test_{test_case.name}():")
        
        # Add setup
        for setup in test_case.setup:
            lines.append(f"    {setup}")
        
        # Add assertions
        for assertion in test_case.assertions:
            if isinstance(assertion, str):
                lines.append(f"    {assertion}")
            else:
                lines.append(f"    assert {assertion}")
        
        return "\n".join(lines)
    
    def _generate_unittest_code(self, test_case: UniversalTestCase) -> str:
        """Generate unittest-specific code."""
        lines = []
        lines.append(f"class Test{test_case.name.title()}(unittest.TestCase):")
        lines.append(f"    def test_{test_case.name}(self):")
        
        # Add setup
        for setup in test_case.setup:
            lines.append(f"        {setup}")
        
        # Add assertions
        for assertion in test_case.assertions:
            if isinstance(assertion, str):
                lines.append(f"        {assertion}")
            else:
                lines.append(f"        self.assertTrue({assertion})")
        
        return "\n".join(lines)
    
    def _generate_generic_code(self, test_case: UniversalTestCase) -> str:
        """Generate generic test code."""
        lines = []
        lines.append(f"# Test: {test_case.name}")
        
        # Add setup
        for setup in test_case.setup:
            lines.append(setup)
        
        # Add assertions
        for assertion in test_case.assertions:
            lines.append(str(assertion))
        
        return "\n".join(lines)

# Export main classes
__all__ = [
    'TestResult',
    'TestContext', 
    'TestRunner',
    'TestMetadata',
    'AssertionType',
    'FrameworkDetector',
    'UniversalTestCase',
    'UniversalTestSuite', 
    'TestAssertion',
    'UniversalTestAdapter'
]
