"""
Universal Testing Framework Abstraction for TestMaster

Provides framework-agnostic test generation that can output to any testing framework.
Adapted from multi-agent frameworks' tool abstraction patterns.
"""

from .universal_test import (
    UniversalTest,
    UniversalTestCase,
    UniversalTestSuite,
    TestAssertion,
    TestSetup,
    TestTeardown,
    TestFixture,
    TestParameter,
    TestMetadata,
    AssertionType
)

from .framework_adapters import (
    BaseFrameworkAdapter,
    FrameworkAdapterRegistry,
    PytestAdapter,
    UnittestAdapter,
    JestAdapter,
    MochaAdapter,
    JUnitAdapter,
    NUnitAdapter,
    XUnitAdapter,
    GoTestAdapter,
    RustTestAdapter,
    RSpecAdapter,
    MinitestAdapter
)

from .test_generator import (
    UniversalTestGenerator,
    TestGenerationStrategy,
    TestGenerationConfig,
    TestGenerationResult
)

__all__ = [
    # Universal Test Models
    'UniversalTest',
    'UniversalTestCase',
    'UniversalTestSuite',
    'TestAssertion',
    'TestSetup',
    'TestTeardown',
    'TestFixture',
    'TestParameter',
    'TestMetadata',
    'AssertionType',
    
    # Framework Adapters
    'BaseFrameworkAdapter',
    'FrameworkAdapterRegistry',
    'PytestAdapter',
    'UnittestAdapter',
    'JestAdapter',
    'MochaAdapter',
    'JUnitAdapter',
    'NUnitAdapter',
    'XUnitAdapter',
    'GoTestAdapter',
    'RustTestAdapter',
    'RSpecAdapter',
    'MinitestAdapter',
    
    # Test Generation
    'UniversalTestGenerator',
    'TestGenerationStrategy',
    'TestGenerationConfig',
    'TestGenerationResult'
]