"""
Automated Test Generation Models - Core data structures for intelligent test generation

This module provides comprehensive data models for representing test cases, test suites,
and analysis results in the automated test generation system. Includes factory functions
for creating test objects and utilities for test case management.

Enterprise Features:
- Comprehensive test case representation with metadata
- Test suite management with coverage estimation
- Module analysis structures for code understanding  
- Test template management and organization
- Coverage prediction and estimation algorithms
- Test complexity scoring and optimization

Key Components:
- TestCase: Individual test case with metadata and scoring
- TestSuite: Collection of tests with setup/teardown
- ModuleAnalysis: Code structure analysis results
- FunctionInfo: Detailed function analysis with parameters
- ClassInfo: Class structure and method analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Type
from pathlib import Path
from datetime import datetime
import json
from enum import Enum


class TestType(Enum):
    """Test case classification types."""
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error"
    PROPERTY_BASED = "property"
    MOCK_BASED = "mock"
    ASYNC = "async"
    PERFORMANCE = "performance"


class CoverageLevel(Enum):
    """Coverage level targets for test generation."""
    BASIC = "basic"          # 60-70% coverage
    STANDARD = "standard"    # 70-85% coverage  
    COMPREHENSIVE = "comprehensive"  # 85-95% coverage
    EXHAUSTIVE = "exhaustive"       # 95%+ coverage


class ComplexityLevel(Enum):
    """Test complexity classification."""
    SIMPLE = 1      # Basic input/output tests
    MODERATE = 2    # Edge cases and error handling
    COMPLEX = 3     # Mock dependencies and async
    ADVANCED = 4    # Property-based and performance
    EXPERT = 5      # Complex integration scenarios


@dataclass
class ParameterInfo:
    """Information about function parameters for test generation."""
    name: str
    type_hint: str = "Any"
    default_value: Optional[str] = None
    is_optional: bool = False
    is_varargs: bool = False
    is_kwargs: bool = False
    possible_values: List[Any] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionInfo:
    """Comprehensive function analysis for test generation."""
    name: str
    parameters: List[ParameterInfo]
    return_type: str = "Any"
    docstring: str = ""
    is_async: bool = False
    is_property: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    raises_exceptions: List[str] = field(default_factory=list)
    calls_external: bool = False
    uses_io: bool = False
    uses_network: bool = False
    uses_database: bool = False
    complexity_score: int = 1
    cyclomatic_complexity: int = 1
    dependencies: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    source_lines: Tuple[int, int] = (0, 0)  # (start_line, end_line)


@dataclass
class ClassInfo:
    """Comprehensive class analysis for test generation."""
    name: str
    base_classes: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    class_variables: List[str] = field(default_factory=list)
    instance_variables: List[str] = field(default_factory=list)
    has_init: bool = False
    is_abstract: bool = False
    is_dataclass: bool = False
    is_exception: bool = False
    complexity_score: int = 1
    inheritance_depth: int = 0
    source_lines: Tuple[int, int] = (0, 0)


@dataclass
class ModuleAnalysis:
    """Complete module structure analysis for test generation."""
    module_name: str
    file_path: str
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    global_variables: List[str] = field(default_factory=list)
    has_main: bool = False
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    complexity_score: int = 0
    maintainability_index: float = 0.0
    dependencies: Set[str] = field(default_factory=set)
    external_calls: Set[str] = field(default_factory=set)
    analysis_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestCase:
    """Represents a generated test case with comprehensive metadata."""
    name: str
    code: str
    target_function: str
    test_type: TestType
    coverage_increase: float
    complexity_score: ComplexityLevel
    dependencies: List[str] = field(default_factory=list)
    setup_required: bool = False
    teardown_required: bool = False
    mock_requirements: List[str] = field(default_factory=list)
    expected_exceptions: List[str] = field(default_factory=list)
    performance_target: Optional[float] = None  # Max execution time in seconds
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-5, higher = more important
    estimated_execution_time: float = 0.1  # Seconds
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.name.startswith('test_'):
            self.name = f"test_{self.name}"
        
        # Auto-tag based on test type
        self.tags.add(self.test_type.value)
        
        # Set default priority based on test type
        if self.test_type in [TestType.UNIT, TestType.ERROR_HANDLING]:
            self.priority = max(self.priority, 3)
        elif self.test_type in [TestType.INTEGRATION, TestType.EDGE_CASE]:
            self.priority = max(self.priority, 2)


@dataclass
class TestSuite:
    """Collection of test cases for a module with comprehensive metadata."""
    module_name: str
    test_cases: List[TestCase]
    estimated_coverage: float
    setup_code: str = ""
    teardown_code: str = ""
    module_setup_code: str = ""
    module_teardown_code: str = ""
    fixtures: Dict[str, str] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    total_estimated_time: float = 0.0
    target_coverage_level: CoverageLevel = CoverageLevel.STANDARD
    configuration: Dict[str, Any] = field(default_factory=dict)
    generation_timestamp: datetime = field(default_factory=datetime.now)
    generator_version: str = "1.0.0"
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self.total_estimated_time = sum(tc.estimated_execution_time for tc in self.test_cases)
        
        # Collect all dependencies
        for test_case in self.test_cases:
            self.dependencies.update(test_case.dependencies)
            self.dependencies.update(test_case.mock_requirements)
    
    @property
    def test_count(self) -> int:
        """Total number of test cases."""
        return len(self.test_cases)
    
    @property
    def test_types_distribution(self) -> Dict[TestType, int]:
        """Distribution of test types."""
        distribution = {}
        for test_case in self.test_cases:
            distribution[test_case.test_type] = distribution.get(test_case.test_type, 0) + 1
        return distribution
    
    @property
    def average_complexity(self) -> float:
        """Average complexity score of test cases."""
        if not self.test_cases:
            return 0.0
        return sum(tc.complexity_score.value for tc in self.test_cases) / len(self.test_cases)
    
    def get_tests_by_type(self, test_type: TestType) -> List[TestCase]:
        """Get all test cases of a specific type."""
        return [tc for tc in self.test_cases if tc.test_type == test_type]
    
    def get_high_priority_tests(self) -> List[TestCase]:
        """Get test cases with priority >= 3."""
        return [tc for tc in self.test_cases if tc.priority >= 3]


@dataclass
class GenerationConfig:
    """Configuration for automated test generation."""
    target_coverage: float = 85.0
    coverage_level: CoverageLevel = CoverageLevel.STANDARD
    max_tests_per_function: int = 5
    generate_edge_cases: bool = True
    generate_error_tests: bool = True
    generate_mock_tests: bool = True
    generate_async_tests: bool = True
    generate_property_tests: bool = False
    enable_performance_tests: bool = False
    parallel_generation: bool = True
    output_format: str = "pytest"  # pytest, unittest, custom
    include_docstrings: bool = True
    include_type_hints: bool = True
    max_complexity_per_test: ComplexityLevel = ComplexityLevel.COMPLEX
    timeout_per_test: float = 30.0
    custom_templates: Dict[str, str] = field(default_factory=dict)


@dataclass 
class GenerationReport:
    """Comprehensive report of test generation results."""
    total_modules_processed: int
    total_tests_generated: int
    total_coverage_estimated: float
    generation_time_seconds: float
    modules_processed: List[str]
    test_files_created: List[str]
    coverage_by_module: Dict[str, float]
    test_distribution: Dict[TestType, int]
    complexity_distribution: Dict[ComplexityLevel, int]
    errors_encountered: List[Dict[str, str]]
    warnings: List[str]
    recommendations: List[str]
    generation_timestamp: datetime = field(default_factory=datetime.now)
    configuration_used: Optional[GenerationConfig] = None


# Factory Functions

def create_test_case(name: str, 
                    code: str,
                    target_function: str,
                    test_type: TestType = TestType.UNIT,
                    coverage_increase: float = 10.0,
                    complexity: ComplexityLevel = ComplexityLevel.SIMPLE,
                    **kwargs) -> TestCase:
    """
    Create a test case with validation and defaults.
    
    Args:
        name: Test case name
        code: Test code
        target_function: Function being tested
        test_type: Type of test
        coverage_increase: Expected coverage increase
        complexity: Test complexity level
        **kwargs: Additional test case parameters
        
    Returns:
        Configured TestCase instance
    """
    return TestCase(
        name=name,
        code=code,
        target_function=target_function,
        test_type=test_type,
        coverage_increase=coverage_increase,
        complexity_score=complexity,
        **kwargs
    )


def create_test_suite(module_name: str,
                     test_cases: List[TestCase],
                     estimated_coverage: float = 0.0,
                     target_coverage: CoverageLevel = CoverageLevel.STANDARD,
                     **kwargs) -> TestSuite:
    """
    Create a test suite with validation and defaults.
    
    Args:
        module_name: Name of module being tested
        test_cases: List of test cases
        estimated_coverage: Estimated coverage percentage
        target_coverage: Target coverage level
        **kwargs: Additional test suite parameters
        
    Returns:
        Configured TestSuite instance
    """
    return TestSuite(
        module_name=module_name,
        test_cases=test_cases,
        estimated_coverage=estimated_coverage,
        target_coverage_level=target_coverage,
        **kwargs
    )


def create_function_info(name: str,
                        parameters: List[ParameterInfo] = None,
                        **kwargs) -> FunctionInfo:
    """
    Create function information with defaults.
    
    Args:
        name: Function name
        parameters: Function parameters
        **kwargs: Additional function info parameters
        
    Returns:
        Configured FunctionInfo instance
    """
    return FunctionInfo(
        name=name,
        parameters=parameters or [],
        **kwargs
    )


def create_module_analysis(module_name: str,
                          file_path: str,
                          **kwargs) -> ModuleAnalysis:
    """
    Create module analysis with defaults.
    
    Args:
        module_name: Module name
        file_path: Path to module file
        **kwargs: Additional analysis parameters
        
    Returns:
        Configured ModuleAnalysis instance
    """
    return ModuleAnalysis(
        module_name=module_name,
        file_path=file_path,
        **kwargs
    )


def create_generation_config(**kwargs) -> GenerationConfig:
    """
    Create generation configuration with custom settings.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured GenerationConfig instance
    """
    return GenerationConfig(**kwargs)


# Utility Functions

def calculate_coverage_score(test_cases: List[TestCase], 
                           module_complexity: int = 1) -> float:
    """
    Calculate estimated coverage based on test cases and module complexity.
    
    Args:
        test_cases: List of test cases
        module_complexity: Module complexity score
        
    Returns:
        Estimated coverage percentage (0-100)
    """
    if not test_cases:
        return 0.0
    
    total_coverage_points = sum(tc.coverage_increase for tc in test_cases)
    complexity_factor = max(1, module_complexity / 10)
    
    estimated_coverage = min(95.0, total_coverage_points / complexity_factor)
    return max(10.0, estimated_coverage)


def validate_test_case(test_case: TestCase) -> List[str]:
    """
    Validate test case structure and content.
    
    Args:
        test_case: Test case to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not test_case.name.startswith('test_'):
        errors.append("Test name must start with 'test_'")
    
    if not test_case.code.strip():
        errors.append("Test code cannot be empty")
    
    if not test_case.target_function:
        errors.append("Target function must be specified")
    
    if test_case.coverage_increase < 0 or test_case.coverage_increase > 100:
        errors.append("Coverage increase must be between 0 and 100")
    
    return errors


def merge_test_suites(suites: List[TestSuite], 
                     merged_name: str = "merged_suite") -> TestSuite:
    """
    Merge multiple test suites into a single suite.
    
    Args:
        suites: List of test suites to merge
        merged_name: Name for merged suite
        
    Returns:
        Merged test suite
    """
    if not suites:
        return create_test_suite(merged_name, [])
    
    all_test_cases = []
    all_dependencies = set()
    total_coverage = 0.0
    
    for suite in suites:
        all_test_cases.extend(suite.test_cases)
        all_dependencies.update(suite.dependencies)
        total_coverage += suite.estimated_coverage
    
    avg_coverage = total_coverage / len(suites) if suites else 0.0
    
    return create_test_suite(
        module_name=merged_name,
        test_cases=all_test_cases,
        estimated_coverage=min(95.0, avg_coverage),
        dependencies=all_dependencies
    )


# Constants and Configuration

DEFAULT_TYPE_GENERATORS = {
    'str': ['""', '"test"', '"hello world"', '""', '" "*100'],
    'int': ['0', '1', '-1', '42', '999999', '-999999'],
    'float': ['0.0', '1.0', '-1.0', '3.14159', 'float("inf")', 'float("-inf")'],
    'bool': ['True', 'False'],
    'list': ['[]', '[1, 2, 3]', '["a", "b", "c"]', '[None]', 'list(range(1000))'],
    'dict': ['{}', '{"key": "value"}', '{"a": 1, "b": 2}', '{None: None}'],
    'tuple': ['()', '(1, 2, 3)', '("a", "b")', '(None,)'],
    'set': ['set()', '{1, 2, 3}', '{"a", "b", "c"}'],
    'Path': ['Path("test.txt")', 'Path("/tmp/test")', 'Path(".")'],
    'None': ['None'],
    'bytes': ['b""', 'b"test"', 'b"\\x00\\x01\\x02"'],
}

COVERAGE_TARGETS = {
    CoverageLevel.BASIC: 65.0,
    CoverageLevel.STANDARD: 80.0,
    CoverageLevel.COMPREHENSIVE: 90.0,
    CoverageLevel.EXHAUSTIVE: 95.0
}

COMPLEXITY_WEIGHTS = {
    ComplexityLevel.SIMPLE: 1.0,
    ComplexityLevel.MODERATE: 1.5,
    ComplexityLevel.COMPLEX: 2.0,
    ComplexityLevel.ADVANCED: 3.0,
    ComplexityLevel.EXPERT: 5.0
}

# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Automated Generation Team'
__description__ = 'Comprehensive data models for intelligent automated test generation'