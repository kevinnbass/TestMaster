"""
Universal Test Representation

Framework-agnostic test representation that can be converted to any testing framework.
Inspired by OpenAI Swarm's agent abstraction and Agency Swarm's tool patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from datetime import datetime


class AssertionType(Enum):
    """Universal assertion types that exist across testing frameworks."""
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    TRUE = "true"
    FALSE = "false"
    NULL = "null"
    NOT_NULL = "not_null"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    THROWS = "throws"
    NOT_THROWS = "not_throws"
    INSTANCE_OF = "instance_of"
    LENGTH = "length"
    MATCHES = "matches"  # Regex match
    DEEP_EQUAL = "deep_equal"  # Object/Array comparison
    CLOSE_TO = "close_to"  # Float comparison with tolerance
    RESOLVED = "resolved"  # Promise/async resolution
    REJECTED = "rejected"  # Promise/async rejection


@dataclass
class TestParameter:
    """Parameter for parameterized/data-driven tests."""
    name: str
    value: Any
    type_hint: Optional[str] = None
    description: Optional[str] = None


@dataclass
class TestAssertion:
    """Universal test assertion."""
    assertion_type: AssertionType
    actual: str  # Expression or variable name
    expected: Any = None  # Expected value
    message: Optional[str] = None
    tolerance: Optional[float] = None  # For close_to assertions
    exception_type: Optional[str] = None  # For throws assertions
    negate: bool = False  # For negative assertions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert assertion to dictionary."""
        return {
            'type': self.assertion_type.value,
            'actual': self.actual,
            'expected': self.expected,
            'message': self.message,
            'tolerance': self.tolerance,
            'exception_type': self.exception_type,
            'negate': self.negate
        }


@dataclass
class TestSetup:
    """Test setup/beforeEach configuration."""
    code: str
    is_async: bool = False
    timeout: Optional[int] = None  # milliseconds
    description: Optional[str] = None
    scope: str = "function"  # function, class, module, session


@dataclass
class TestTeardown:
    """Test teardown/afterEach configuration."""
    code: str
    is_async: bool = False
    timeout: Optional[int] = None
    description: Optional[str] = None
    scope: str = "function"


@dataclass
class TestFixture:
    """Test fixture definition - adapted from pytest/jest patterns."""
    name: str
    setup_code: str
    teardown_code: Optional[str] = None
    scope: str = "function"  # function, class, module, session
    params: List[TestParameter] = field(default_factory=list)
    is_async: bool = False
    autouse: bool = False
    description: Optional[str] = None


@dataclass
class TestMetadata:
    """Metadata for test tracking and reporting."""
    tags: List[str] = field(default_factory=list)
    priority: int = 1  # 1=highest, 5=lowest
    category: Optional[str] = None  # unit, integration, e2e, performance
    requirements: List[str] = field(default_factory=list)  # Required features/modules
    skip_reason: Optional[str] = None
    expected_duration: Optional[int] = None  # milliseconds
    flaky: bool = False
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    author: Optional[str] = None
    ticket_id: Optional[str] = None  # JIRA/GitHub issue
    description: Optional[str] = None  # Test suite description


@dataclass
class UniversalTest:
    """Universal representation of a single test."""
    name: str
    test_function: str  # The actual test code
    assertions: List[TestAssertion] = field(default_factory=list)
    setup: Optional[TestSetup] = None
    teardown: Optional[TestTeardown] = None
    fixtures: List[str] = field(default_factory=list)  # Fixture names to use
    parameters: List[TestParameter] = field(default_factory=list)  # For parameterized tests
    is_async: bool = False
    timeout: Optional[int] = None  # milliseconds
    description: Optional[str] = None
    metadata: TestMetadata = field(default_factory=TestMetadata)
    skip: bool = False
    skip_condition: Optional[str] = None  # Expression to evaluate
    only: bool = False  # Run only this test (focused test)
    
    def add_assertion(self, assertion: TestAssertion):
        """Add an assertion to the test."""
        self.assertions.append(assertion)
    
    def add_parameter(self, parameter: TestParameter):
        """Add a parameter for parameterized testing."""
        self.parameters.append(parameter)
    
    def mark_skip(self, reason: str = None, condition: str = None):
        """Mark test as skipped."""
        self.skip = True
        if reason:
            self.metadata.skip_reason = reason
        if condition:
            self.skip_condition = condition
    
    def mark_only(self):
        """Mark test to run exclusively."""
        self.only = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test to dictionary representation."""
        return {
            'name': self.name,
            'test_function': self.test_function,
            'assertions': [a.to_dict() for a in self.assertions],
            'setup': self.setup,
            'teardown': self.teardown,
            'fixtures': self.fixtures,
            'parameters': self.parameters,
            'is_async': self.is_async,
            'timeout': self.timeout,
            'description': self.description,
            'metadata': self.metadata,
            'skip': self.skip,
            'skip_condition': self.skip_condition,
            'only': self.only
        }


@dataclass
class UniversalTestCase:
    """Universal test case (class/describe block)."""
    name: str
    tests: List[UniversalTest] = field(default_factory=list)
    setup_class: Optional[TestSetup] = None  # Class-level setup
    teardown_class: Optional[TestTeardown] = None  # Class-level teardown
    setup_method: Optional[TestSetup] = None  # Method-level setup
    teardown_method: Optional[TestTeardown] = None  # Method-level teardown
    fixtures: List[TestFixture] = field(default_factory=list)
    nested_suites: List['UniversalTestCase'] = field(default_factory=list)  # Nested describes
    description: Optional[str] = None
    metadata: TestMetadata = field(default_factory=TestMetadata)
    skip: bool = False
    only: bool = False
    
    def add_test(self, test: UniversalTest):
        """Add a test to this test case."""
        self.tests.append(test)
    
    def add_fixture(self, fixture: TestFixture):
        """Add a fixture to this test case."""
        self.fixtures.append(fixture)
    
    def add_nested_suite(self, suite: 'UniversalTestCase'):
        """Add a nested test suite."""
        self.nested_suites.append(suite)
    
    def get_all_tests(self) -> List[UniversalTest]:
        """Get all tests including nested suites."""
        all_tests = self.tests.copy()
        for suite in self.nested_suites:
            all_tests.extend(suite.get_all_tests())
        return all_tests
    
    def count_tests(self) -> int:
        """Count total tests including nested suites."""
        count = len(self.tests)
        for suite in self.nested_suites:
            count += suite.count_tests()
        return count


@dataclass
class UniversalTestSuite:
    """Universal test suite containing multiple test cases."""
    name: str
    test_cases: List[UniversalTestCase] = field(default_factory=list)
    global_setup: Optional[TestSetup] = None  # Suite-level setup
    global_teardown: Optional[TestTeardown] = None  # Suite-level teardown
    global_fixtures: List[TestFixture] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)  # Required imports
    helper_functions: List[str] = field(default_factory=list)  # Helper function definitions
    config: Dict[str, Any] = field(default_factory=dict)  # Framework-specific config
    metadata: TestMetadata = field(default_factory=TestMetadata)
    
    # Language and framework info
    target_language: str = ""
    target_framework: str = ""
    
    # Coverage and quality metrics
    coverage_percentage: float = 0.0
    assertion_density: float = 0.0  # Assertions per test
    test_quality_score: float = 0.0
    
    def add_test_case(self, test_case: UniversalTestCase):
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
    
    def add_global_fixture(self, fixture: TestFixture):
        """Add a global fixture."""
        self.global_fixtures.append(fixture)
    
    def add_import(self, import_statement: str):
        """Add an import statement."""
        if import_statement not in self.imports:
            self.imports.append(import_statement)
    
    def add_helper_function(self, function_code: str):
        """Add a helper function."""
        self.helper_functions.append(function_code)
    
    def get_all_tests(self) -> List[UniversalTest]:
        """Get all tests from all test cases."""
        all_tests = []
        for test_case in self.test_cases:
            all_tests.extend(test_case.get_all_tests())
        return all_tests
    
    def count_tests(self) -> int:
        """Count total tests in the suite."""
        return sum(tc.count_tests() for tc in self.test_cases)
    
    def count_assertions(self) -> int:
        """Count total assertions in the suite."""
        total = 0
        for test in self.get_all_tests():
            total += len(test.assertions)
        return total
    
    def calculate_metrics(self):
        """Calculate test quality metrics."""
        test_count = self.count_tests()
        assertion_count = self.count_assertions()
        
        if test_count > 0:
            self.assertion_density = assertion_count / test_count
            
            # Calculate quality score based on various factors
            quality_factors = []
            
            # Factor 1: Assertion density (ideal: 3-5 per test)
            if 3 <= self.assertion_density <= 5:
                quality_factors.append(1.0)
            elif 1 <= self.assertion_density < 3:
                quality_factors.append(0.7)
            elif 5 < self.assertion_density <= 10:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
            
            # Factor 2: Test coverage
            quality_factors.append(min(self.coverage_percentage / 100, 1.0))
            
            # Factor 3: Setup/teardown presence
            setup_score = 0.0
            for tc in self.test_cases:
                if tc.setup_method or tc.setup_class:
                    setup_score += 0.5
                if tc.teardown_method or tc.teardown_class:
                    setup_score += 0.5
            quality_factors.append(min(setup_score / len(self.test_cases), 1.0) if self.test_cases else 0)
            
            # Factor 4: Test organization (nested suites)
            has_nested = any(tc.nested_suites for tc in self.test_cases)
            quality_factors.append(1.0 if has_nested else 0.7)
            
            # Factor 5: Metadata completeness
            metadata_score = 0.0
            for test in self.get_all_tests():
                if test.description:
                    metadata_score += 0.33
                if test.metadata.tags:
                    metadata_score += 0.33
                if test.metadata.category:
                    metadata_score += 0.34
            quality_factors.append(min(metadata_score / test_count, 1.0) if test_count > 0 else 0)
            
            # Calculate weighted average
            self.test_quality_score = sum(quality_factors) / len(quality_factors) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert suite to dictionary representation."""
        return {
            'name': self.name,
            'test_cases': [tc.__dict__ for tc in self.test_cases],
            'global_setup': self.global_setup,
            'global_teardown': self.global_teardown,
            'global_fixtures': self.global_fixtures,
            'imports': self.imports,
            'helper_functions': self.helper_functions,
            'config': self.config,
            'metadata': self.metadata,
            'target_language': self.target_language,
            'target_framework': self.target_framework,
            'metrics': {
                'total_tests': self.count_tests(),
                'total_assertions': self.count_assertions(),
                'coverage_percentage': self.coverage_percentage,
                'assertion_density': self.assertion_density,
                'test_quality_score': self.test_quality_score
            }
        }


class TestPatternLibrary:
    """Library of common test patterns across frameworks."""
    
    @staticmethod
    def create_simple_test(name: str, function_name: str, expected_value: Any) -> UniversalTest:
        """Create a simple equality test."""
        test = UniversalTest(
            name=name,
            test_function=f"result = {function_name}()",
            description=f"Test {function_name} returns {expected_value}"
        )
        test.add_assertion(TestAssertion(
            assertion_type=AssertionType.EQUAL,
            actual="result",
            expected=expected_value
        ))
        return test
    
    @staticmethod
    def create_exception_test(name: str, function_name: str, exception_type: str) -> UniversalTest:
        """Create an exception test."""
        test = UniversalTest(
            name=name,
            test_function=f"{function_name}()",
            description=f"Test {function_name} throws {exception_type}"
        )
        test.add_assertion(TestAssertion(
            assertion_type=AssertionType.THROWS,
            actual=function_name,
            exception_type=exception_type
        ))
        return test
    
    @staticmethod
    def create_async_test(name: str, async_function: str, expected_value: Any) -> UniversalTest:
        """Create an async test."""
        test = UniversalTest(
            name=name,
            test_function=f"result = await {async_function}()",
            is_async=True,
            description=f"Test async {async_function}"
        )
        test.add_assertion(TestAssertion(
            assertion_type=AssertionType.EQUAL,
            actual="result",
            expected=expected_value
        ))
        return test
    
    @staticmethod
    def create_parameterized_test(name: str, function_name: str, test_cases: List[Tuple[Any, Any]]) -> UniversalTest:
        """Create a parameterized test."""
        test = UniversalTest(
            name=name,
            test_function=f"result = {function_name}(input_value)",
            description=f"Parameterized test for {function_name}"
        )
        
        for input_val, expected_val in test_cases:
            test.add_parameter(TestParameter(
                name="input_value",
                value=input_val
            ))
            test.add_assertion(TestAssertion(
                assertion_type=AssertionType.EQUAL,
                actual="result",
                expected=expected_val,
                message=f"For input {input_val}"
            ))
        
        return test