"""
Universal Test Generator

Generates tests from Universal AST in any testing framework format.
Integrates patterns from OpenAI Swarm, Agency Swarm, and PraisonAI.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

from ..ast_abstraction import UniversalAST, UniversalFunction, UniversalClass
from .universal_test import (
    UniversalTest, UniversalTestCase, UniversalTestSuite,
    TestAssertion, AssertionType, TestSetup, TestTeardown,
    TestFixture, TestMetadata, TestPatternLibrary
)
from .framework_adapters import FrameworkAdapterRegistry


@dataclass
class TestGenerationConfig:
    """Configuration for test generation."""
    target_framework: str = "pytest"
    target_language: str = "python"
    
    # Test generation options
    generate_async_tests: bool = True
    generate_parameterized_tests: bool = True
    generate_edge_cases: bool = True
    generate_error_tests: bool = True
    generate_integration_tests: bool = False
    
    # Test quality options
    min_assertions_per_test: int = 1
    max_assertions_per_test: int = 5
    test_naming_convention: str = "test_{function_name}_{scenario}"
    
    # Coverage options
    target_coverage: float = 0.80
    include_private_methods: bool = False
    include_helper_functions: bool = True
    
    # Framework-specific options
    use_fixtures: bool = True
    use_mocks: bool = True
    timeout_ms: int = 5000
    
    # Metadata options
    add_descriptions: bool = True
    add_tags: bool = True
    add_requirements: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()


@dataclass
class TestGenerationStrategy:
    """Strategy for generating tests - Adapter pattern from multi-agent frameworks."""
    name: str
    description: str
    applicable_to: List[str] = field(default_factory=list)  # Function types
    
    def generate_tests(self, function: UniversalFunction, config: TestGenerationConfig) -> List[UniversalTest]:
        """Generate tests based on strategy."""
        raise NotImplementedError


@dataclass
class TestGenerationResult:
    """Result of test generation."""
    test_suite: UniversalTestSuite
    framework_code: str
    file_path: str
    
    # Metrics
    total_tests: int = 0
    total_assertions: int = 0
    coverage_estimate: float = 0.0
    generation_time: float = 0.0
    
    # Quality metrics
    test_quality_score: float = 0.0
    assertion_density: float = 0.0
    
    # Issues and warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'file_path': self.file_path,
            'framework': self.test_suite.target_framework,
            'language': self.test_suite.target_language,
            'metrics': {
                'total_tests': self.total_tests,
                'total_assertions': self.total_assertions,
                'coverage_estimate': self.coverage_estimate,
                'test_quality_score': self.test_quality_score,
                'assertion_density': self.assertion_density,
                'generation_time': self.generation_time
            },
            'warnings': self.warnings,
            'errors': self.errors
        }


class UniversalTestGenerator:
    """Generates tests from Universal AST for any framework."""
    
    def __init__(self, config: TestGenerationConfig = None):
        self.config = config or TestGenerationConfig()
        self.strategies: List[TestGenerationStrategy] = []
        self._initialize_strategies()
        
        print(f"Universal Test Generator initialized")
        print(f"   Target framework: {self.config.target_framework}")
        print(f"   Target language: {self.config.target_language}")
    
    def _initialize_strategies(self):
        """Initialize test generation strategies."""
        # Add default strategies
        self.strategies = [
            SimpleTestStrategy(),
            ParameterizedTestStrategy(),
            EdgeCaseTestStrategy(),
            ErrorHandlingTestStrategy(),
            AsyncTestStrategy(),
            IntegrationTestStrategy()
        ]
    
    def generate_tests_from_ast(self, universal_ast: UniversalAST) -> TestGenerationResult:
        """Generate tests from Universal AST."""
        start_time = datetime.now()
        
        print(f"Generating tests for {len(universal_ast.modules)} modules...")
        
        # Create test suite
        test_suite = UniversalTestSuite(
            name=f"TestSuite_{universal_ast.project_path.split('/')[-1]}",
            target_language=self.config.target_language,
            target_framework=self.config.target_framework,
            metadata=TestMetadata(
                created_at=datetime.now(),
                category="unit",
                tags=["auto-generated", "universal-test-framework"]
            )
        )
        
        # Generate tests for each module
        for module in universal_ast.modules:
            test_case = self._generate_module_tests(module)
            if test_case and test_case.tests:
                test_suite.add_test_case(test_case)
        
        # Calculate metrics
        test_suite.calculate_metrics()
        
        # Convert to framework-specific format
        adapter = FrameworkAdapterRegistry.get_adapter(self.config.target_framework)
        if not adapter:
            raise ValueError(f"Unsupported framework: {self.config.target_framework}")
        
        framework_code = adapter.convert_test_suite(test_suite)
        
        # Create result
        generation_time = (datetime.now() - start_time).total_seconds()
        
        result = TestGenerationResult(
            test_suite=test_suite,
            framework_code=framework_code,
            file_path=f"test_{universal_ast.project_path.split('/')[-1]}{adapter.get_file_extension()}",
            total_tests=test_suite.count_tests(),
            total_assertions=test_suite.count_assertions(),
            coverage_estimate=self._estimate_coverage(universal_ast, test_suite),
            generation_time=generation_time,
            test_quality_score=test_suite.test_quality_score,
            assertion_density=test_suite.assertion_density
        )
        
        print(f"Test generation completed:")
        print(f"   Tests generated: {result.total_tests}")
        print(f"   Assertions: {result.total_assertions}")
        print(f"   Coverage estimate: {result.coverage_estimate:.1f}%")
        print(f"   Quality score: {result.test_quality_score:.1f}")
        print(f"   Generation time: {result.generation_time:.2f}s")
        
        return result
    
    def _generate_module_tests(self, module) -> Optional[UniversalTestCase]:
        """Generate tests for a module."""
        if not module.functions and not module.classes:
            return None
        
        test_case = UniversalTestCase(
            name=f"{module.name}_Tests",
            description=f"Tests for {module.name} module",
            metadata=TestMetadata(
                tags=[module.language, "unit"],
                category="unit"
            )
        )
        
        # Add imports
        test_case.setup_method = TestSetup(
            code=f"# Setup for {module.name} tests",
            scope="function"
        )
        
        # Generate tests for functions
        for function in module.functions:
            if self._should_test_function(function):
                tests = self._generate_function_tests(function)
                for test in tests:
                    test_case.add_test(test)
        
        # Generate tests for classes
        for cls in module.classes:
            if self._should_test_class(cls):
                class_test_case = self._generate_class_tests(cls)
                if class_test_case:
                    test_case.add_nested_suite(class_test_case)
        
        return test_case if test_case.tests or test_case.nested_suites else None
    
    def _should_test_function(self, function: UniversalFunction) -> bool:
        """Determine if function should be tested."""
        # Skip private functions if configured
        if not self.config.include_private_methods and function.name.startswith('_'):
            return False
        
        # Skip very simple functions
        if function.lines_of_code < 2 and not function.parameters:
            return False
        
        return True
    
    def _should_test_class(self, cls: UniversalClass) -> bool:
        """Determine if class should be tested."""
        # Skip private classes if configured
        if not self.config.include_private_methods and cls.name.startswith('_'):
            return False
        
        # Skip empty classes
        if not cls.methods and not cls.fields:
            return False
        
        return True
    
    def _generate_function_tests(self, function: UniversalFunction) -> List[UniversalTest]:
        """Generate tests for a function using strategies."""
        tests = []
        
        # Apply each strategy
        for strategy in self.strategies:
            if self._is_strategy_applicable(strategy, function):
                try:
                    strategy_tests = strategy.generate_tests(function, self.config)
                    tests.extend(strategy_tests)
                except Exception as e:
                    print(f"Strategy {strategy.name} failed for {function.name}: {e}")
        
        # If no strategies generated tests, create a simple test
        if not tests and self.config.min_assertions_per_test > 0:
            tests.append(self._create_simple_test(function))
        
        return tests
    
    def _generate_class_tests(self, cls: UniversalClass) -> Optional[UniversalTestCase]:
        """Generate tests for a class."""
        test_case = UniversalTestCase(
            name=f"{cls.name}_Tests",
            description=f"Tests for {cls.name} class",
            metadata=TestMetadata(
                tags=["class", cls.name.lower()],
                category="unit"
            )
        )
        
        # Setup: Create instance
        test_case.setup_method = TestSetup(
            code=f"self.instance = {cls.name}()",
            scope="function"
        )
        
        # Generate tests for each method
        for method in cls.methods:
            if self._should_test_function(method):
                tests = self._generate_function_tests(method)
                for test in tests:
                    # Adjust test to use self.instance
                    test.test_function = test.test_function.replace(
                        method.name + "(", f"self.instance.{method.name}("
                    )
                    test_case.add_test(test)
        
        return test_case if test_case.tests else None
    
    def _is_strategy_applicable(self, strategy: TestGenerationStrategy, function: UniversalFunction) -> bool:
        """Check if strategy is applicable to function."""
        # Check if async strategy should be applied
        if strategy.name == "AsyncTestStrategy" and not function.is_async:
            return False
        
        # Check if parameterized strategy should be applied
        if strategy.name == "ParameterizedTestStrategy" and len(function.parameters) < 1:
            return False
        
        # Check if error strategy should be applied
        if strategy.name == "ErrorHandlingTestStrategy" and not function.throws_exceptions:
            return False
        
        return True
    
    def _create_simple_test(self, function: UniversalFunction) -> UniversalTest:
        """Create a simple test for a function."""
        test_name = f"test_{function.name}_basic"
        
        # Build function call
        params = []
        for param in function.parameters:
            if param.default_value:
                continue  # Skip parameters with defaults
            elif param.type_hint and 'str' in param.type_hint:
                params.append('"test"')
            elif param.type_hint and ('int' in param.type_hint or 'float' in param.type_hint):
                params.append('1')
            elif param.type_hint and 'bool' in param.type_hint:
                params.append('True')
            elif param.type_hint and ('list' in param.type_hint.lower() or 'List' in param.type_hint):
                params.append('[]')
            elif param.type_hint and ('dict' in param.type_hint.lower() or 'Dict' in param.type_hint):
                params.append('{}')
            else:
                params.append('None')
        
        function_call = f"{function.name}({', '.join(params)})"
        
        test = UniversalTest(
            name=test_name,
            test_function=f"result = {function_call}",
            description=f"Basic test for {function.name}",
            is_async=function.is_async
        )
        
        # Add basic assertion
        if function.return_type:
            if 'bool' in function.return_type:
                test.add_assertion(TestAssertion(
                    assertion_type=AssertionType.INSTANCE_OF,
                    actual="result",
                    expected="bool"
                ))
            elif 'str' in function.return_type:
                test.add_assertion(TestAssertion(
                    assertion_type=AssertionType.INSTANCE_OF,
                    actual="result",
                    expected="str"
                ))
            elif 'int' in function.return_type:
                test.add_assertion(TestAssertion(
                    assertion_type=AssertionType.INSTANCE_OF,
                    actual="result",
                    expected="int"
                ))
            else:
                test.add_assertion(TestAssertion(
                    assertion_type=AssertionType.NOT_NULL,
                    actual="result"
                ))
        else:
            # No return type info, just check it doesn't throw
            test.add_assertion(TestAssertion(
                assertion_type=AssertionType.NOT_NULL,
                actual="result",
                message="Function should complete without error"
            ))
        
        return test
    
    def _estimate_coverage(self, ast: UniversalAST, suite: UniversalTestSuite) -> float:
        """Estimate test coverage percentage."""
        total_functions = ast.total_functions
        total_classes = ast.total_classes
        
        tested_items = suite.count_tests()
        total_items = total_functions + total_classes
        
        if total_items == 0:
            return 0.0
        
        coverage = min((tested_items / total_items) * 100, 100.0)
        return coverage


# Test Generation Strategies
class SimpleTestStrategy(TestGenerationStrategy):
    """Generate simple equality tests."""
    
    def __init__(self):
        super().__init__(
            name="SimpleTestStrategy",
            description="Generate basic tests with simple assertions",
            applicable_to=["all"]
        )
    
    def generate_tests(self, function: UniversalFunction, config: TestGenerationConfig) -> List[UniversalTest]:
        """Generate simple test."""
        return [TestPatternLibrary.create_simple_test(
            f"test_{function.name}_returns_value",
            function.name,
            None  # Expected value would be determined by AI or analysis
        )]


class ParameterizedTestStrategy(TestGenerationStrategy):
    """Generate parameterized tests."""
    
    def __init__(self):
        super().__init__(
            name="ParameterizedTestStrategy",
            description="Generate data-driven tests",
            applicable_to=["functions_with_params"]
        )
    
    def generate_tests(self, function: UniversalFunction, config: TestGenerationConfig) -> List[UniversalTest]:
        """Generate parameterized tests."""
        if not config.generate_parameterized_tests or not function.parameters:
            return []
        
        # Generate test cases based on parameter types
        test_cases = []
        for param in function.parameters[:1]:  # Just first param for simplicity
            if 'int' in str(param.type_hint):
                test_cases.extend([(0, None), (1, None), (-1, None)])
            elif 'str' in str(param.type_hint):
                test_cases.extend([("", None), ("test", None), ("long string", None)])
            elif 'bool' in str(param.type_hint):
                test_cases.extend([(True, None), (False, None)])
        
        if test_cases:
            return [TestPatternLibrary.create_parameterized_test(
                f"test_{function.name}_parameterized",
                function.name,
                test_cases
            )]
        return []


class EdgeCaseTestStrategy(TestGenerationStrategy):
    """Generate edge case tests."""
    
    def __init__(self):
        super().__init__(
            name="EdgeCaseTestStrategy",
            description="Generate tests for edge cases",
            applicable_to=["all"]
        )
    
    def generate_tests(self, function: UniversalFunction, config: TestGenerationConfig) -> List[UniversalTest]:
        """Generate edge case tests."""
        if not config.generate_edge_cases:
            return []
        
        tests = []
        
        # Null/None test
        if function.parameters:
            test = UniversalTest(
                name=f"test_{function.name}_handles_none",
                test_function=f"{function.name}(None)",
                description=f"Test {function.name} handles None input"
            )
            test.add_assertion(TestAssertion(
                assertion_type=AssertionType.NOT_THROWS,
                actual=function.name,
                message="Should handle None gracefully"
            ))
            tests.append(test)
        
        # Empty collection test
        for param in function.parameters:
            if 'list' in str(param.type_hint).lower():
                test = UniversalTest(
                    name=f"test_{function.name}_empty_list",
                    test_function=f"result = {function.name}([])",
                    description=f"Test {function.name} with empty list"
                )
                test.add_assertion(TestAssertion(
                    assertion_type=AssertionType.NOT_THROWS,
                    actual=function.name,
                    message="Should handle empty list"
                ))
                tests.append(test)
                break
        
        return tests


class ErrorHandlingTestStrategy(TestGenerationStrategy):
    """Generate error handling tests."""
    
    def __init__(self):
        super().__init__(
            name="ErrorHandlingTestStrategy",
            description="Generate tests for error conditions",
            applicable_to=["functions_with_exceptions"]
        )
    
    def generate_tests(self, function: UniversalFunction, config: TestGenerationConfig) -> List[UniversalTest]:
        """Generate error handling tests."""
        if not config.generate_error_tests:
            return []
        
        tests = []
        
        # Test for each exception type
        for exception_type in function.throws_exceptions:
            test = TestPatternLibrary.create_exception_test(
                f"test_{function.name}_raises_{exception_type.lower()}",
                function.name,
                exception_type
            )
            tests.append(test)
        
        return tests


class AsyncTestStrategy(TestGenerationStrategy):
    """Generate async tests."""
    
    def __init__(self):
        super().__init__(
            name="AsyncTestStrategy",
            description="Generate tests for async functions",
            applicable_to=["async_functions"]
        )
    
    def generate_tests(self, function: UniversalFunction, config: TestGenerationConfig) -> List[UniversalTest]:
        """Generate async tests."""
        if not config.generate_async_tests or not function.is_async:
            return []
        
        return [TestPatternLibrary.create_async_test(
            f"test_{function.name}_async",
            function.name,
            None
        )]


class IntegrationTestStrategy(TestGenerationStrategy):
    """Generate integration tests."""
    
    def __init__(self):
        super().__init__(
            name="IntegrationTestStrategy",
            description="Generate integration tests",
            applicable_to=["functions_with_dependencies"]
        )
    
    def generate_tests(self, function: UniversalFunction, config: TestGenerationConfig) -> List[UniversalTest]:
        """Generate integration tests."""
        if not config.generate_integration_tests:
            return []
        
        # Would generate tests that test function interactions
        return []