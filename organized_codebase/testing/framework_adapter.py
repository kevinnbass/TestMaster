"""
Universal Framework Adapter

Adapts universal test suites to specific testing frameworks across any programming language.
Provides seamless integration with popular testing frameworks while maintaining 
codebase-agnostic architecture.

Adapted from Agency Swarm's framework adaptation patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Union
from enum import Enum
import json
from pathlib import Path

from core.framework_abstraction import (
    UniversalTestSuite, UniversalTestCase, UniversalTest, 
    TestAssertion, AssertionType, TestMetadata
)


class SupportedFramework(Enum):
    """Supported testing frameworks across languages."""
    # Python frameworks
    PYTEST = "pytest"
    UNITTEST = "unittest"
    NOSE = "nose"
    DOCTEST = "doctest"
    
    # JavaScript/TypeScript frameworks
    JEST = "jest"
    MOCHA = "mocha"
    JASMINE = "jasmine"
    AVA = "ava"
    VITEST = "vitest"
    
    # Java frameworks
    JUNIT = "junit"
    TESTNG = "testng"
    SPOCK = "spock"
    
    # C# frameworks
    NUNIT = "nunit"
    XUNIT = "xunit"
    MSTEST = "mstest"
    
    # Go frameworks
    GO_TEST = "go_test"
    GINKGO = "ginkgo"
    TESTIFY = "testify"
    
    # Rust frameworks
    RUST_TEST = "rust_test"
    
    # PHP frameworks
    PHPUNIT = "phpunit"
    PEST = "pest"
    
    # Ruby frameworks
    RSPEC = "rspec"
    MINITEST = "minitest"
    
    # Universal (framework-agnostic)
    UNIVERSAL = "universal"


@dataclass
class TestFrameworkMapping:
    """Mapping configuration for a specific framework."""
    framework: SupportedFramework
    language: str
    file_extension: str
    
    # Test structure templates
    test_file_template: str = ""
    test_class_template: str = ""
    test_method_template: str = ""
    
    # Assertion mappings
    assertion_mappings: Dict[AssertionType, str] = field(default_factory=dict)
    
    # Import statements
    required_imports: List[str] = field(default_factory=list)
    
    # Setup/teardown templates
    setup_template: str = ""
    teardown_template: str = ""
    
    # Framework-specific configurations
    framework_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'framework': self.framework.value,
            'language': self.language,
            'file_extension': self.file_extension,
            'templates': {
                'test_file': self.test_file_template,
                'test_class': self.test_class_template,
                'test_method': self.test_method_template,
                'setup': self.setup_template,
                'teardown': self.teardown_template
            },
            'assertion_mappings': {k.value: v for k, v in self.assertion_mappings.items()},
            'required_imports': self.required_imports,
            'framework_config': self.framework_config
        }


@dataclass
class FrameworkAdapterConfig:
    """Configuration for framework adaptation."""
    # Target frameworks
    target_frameworks: List[str] = field(default_factory=list)
    auto_detect_frameworks: bool = True
    
    # Output settings
    generate_separate_files: bool = True
    maintain_universal_format: bool = True
    
    # Naming conventions
    test_file_naming: str = "test_{module_name}"  # test_module.py, module_test.py, etc.
    test_class_naming: str = "Test{ClassName}"
    test_method_naming: str = "test_{method_name}"
    
    # Framework-specific settings
    framework_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'target_frameworks': self.target_frameworks,
            'auto_detect_frameworks': self.auto_detect_frameworks,
            'output_settings': {
                'generate_separate_files': self.generate_separate_files,
                'maintain_universal_format': self.maintain_universal_format
            },
            'naming_conventions': {
                'test_file_naming': self.test_file_naming,
                'test_class_naming': self.test_class_naming,
                'test_method_naming': self.test_method_naming
            },
            'framework_configs': self.framework_configs
        }


class UniversalFrameworkAdapter:
    """Universal adapter for testing frameworks."""
    
    def __init__(self):
        self.framework_mappings = self._load_framework_mappings()
        
        print(f"Universal Framework Adapter initialized")
        print(f"   Supported frameworks: {len(self.framework_mappings)}")
        for framework in SupportedFramework:
            if framework in self.framework_mappings:
                print(f"      {framework.value}")
    
    def adapt_test_suite(self, 
                        universal_suite: UniversalTestSuite,
                        config: FrameworkAdapterConfig) -> List[UniversalTestSuite]:
        """Adapt universal test suite to target frameworks."""
        
        print(f"\\nAdapting test suite: {universal_suite.name}")
        print(f"   Target frameworks: {config.target_frameworks}")
        
        adapted_suites = []
        
        # If no frameworks specified, return universal format
        if not config.target_frameworks:
            print(f"   No frameworks specified, returning universal format")
            adapted_suites.append(universal_suite)
            return adapted_suites
        
        # Adapt to each target framework
        for framework_name in config.target_frameworks:
            try:
                framework = SupportedFramework(framework_name)
                adapted_suite = self._adapt_to_framework(universal_suite, framework, config)
                adapted_suites.append(adapted_suite)
                print(f"   ✓ Adapted to {framework.value}")
                
            except ValueError:
                print(f"   ❌ Unsupported framework: {framework_name}")
                continue
            except Exception as e:
                print(f"   ❌ Error adapting to {framework_name}: {str(e)}")
                continue
        
        # Always maintain universal format if requested
        if config.maintain_universal_format:
            adapted_suites.append(universal_suite)
            print(f"   ✓ Maintained universal format")
        
        return adapted_suites
    
    def _adapt_to_framework(self, 
                           universal_suite: UniversalTestSuite,
                           framework: SupportedFramework,
                           config: FrameworkAdapterConfig) -> UniversalTestSuite:
        """Adapt test suite to specific framework."""
        
        if framework not in self.framework_mappings:
            raise ValueError(f"No mapping available for framework: {framework.value}")
        
        mapping = self.framework_mappings[framework]
        
        # Create adapted test suite
        adapted_suite = UniversalTestSuite(
            name=f"{universal_suite.name}_{framework.value}",
            metadata=TestMetadata(
                tags=universal_suite.metadata.tags + [framework.value],
                category=universal_suite.metadata.category,
                description=f"{universal_suite.metadata.description} (adapted for {framework.value})",
                framework=framework.value
            )
        )
        
        # Adapt each test case
        for test_case in universal_suite.test_cases:
            adapted_case = self._adapt_test_case(test_case, mapping, config)
            adapted_suite.test_cases.append(adapted_case)
        
        # Calculate metrics for adapted suite
        adapted_suite.calculate_metrics()
        
        return adapted_suite
    
    def _adapt_test_case(self, 
                        universal_case: UniversalTestCase,
                        mapping: TestFrameworkMapping,
                        config: FrameworkAdapterConfig) -> UniversalTestCase:
        """Adapt universal test case to framework."""
        
        # Create adapted test case
        adapted_case = UniversalTestCase(
            name=universal_case.name,
            description=universal_case.description,
            metadata=TestMetadata(
                tags=universal_case.metadata.tags + [mapping.framework.value],
                category=universal_case.metadata.category,
                framework=mapping.framework.value
            )
        )
        
        # Adapt each test
        for test in universal_case.tests:
            adapted_test = self._adapt_test(test, mapping, config)
            adapted_case.add_test(adapted_test)
        
        return adapted_case
    
    def _adapt_test(self, 
                   universal_test: UniversalTest,
                   mapping: TestFrameworkMapping,
                   config: FrameworkAdapterConfig) -> UniversalTest:
        """Adapt universal test to framework."""
        
        # Generate framework-specific test function
        adapted_function = self._generate_framework_test_function(
            universal_test, mapping, config
        )
        
        # Create adapted test
        adapted_test = UniversalTest(
            name=self._adapt_test_name(universal_test.name, config),
            test_function=adapted_function,
            description=universal_test.description,
            tags=universal_test.tags + [mapping.framework.value]
        )
        
        # Adapt assertions
        for assertion in universal_test.assertions:
            adapted_assertion = self._adapt_assertion(assertion, mapping)
            adapted_test.add_assertion(adapted_assertion)
        
        return adapted_test
    
    def _generate_framework_test_function(self, 
                                        universal_test: UniversalTest,
                                        mapping: TestFrameworkMapping,
                                        config: FrameworkAdapterConfig) -> str:
        """Generate framework-specific test function."""
        
        # Get test method template
        template = mapping.test_method_template
        if not template:
            template = self._get_default_test_template(mapping.framework)
        
        # Adapt test function content
        adapted_content = self._adapt_test_content(universal_test.test_function, mapping)
        
        # Generate assertions
        assertion_code = self._generate_assertion_code(universal_test.assertions, mapping)
        
        # Format template
        test_function = template.format(
            test_name=self._adapt_test_name(universal_test.name, config),
            test_content=adapted_content,
            assertions=assertion_code,
            description=universal_test.description
        )
        
        return test_function
    
    def _adapt_test_content(self, test_function: str, mapping: TestFrameworkMapping) -> str:
        """Adapt test function content to framework."""
        
        # Basic adaptations based on framework
        adapted_content = test_function
        
        # Framework-specific adaptations
        if mapping.framework == SupportedFramework.PYTEST:
            # pytest specific adaptations
            adapted_content = adapted_content.replace("self.", "")
        
        elif mapping.framework == SupportedFramework.UNITTEST:
            # unittest specific adaptations
            if not adapted_content.strip().startswith("self."):
                adapted_content = f"self.{adapted_content}"
        
        elif mapping.framework == SupportedFramework.JEST:
            # Jest specific adaptations
            adapted_content = self._adapt_to_javascript(adapted_content)
        
        elif mapping.framework == SupportedFramework.JUNIT:
            # JUnit specific adaptations
            adapted_content = self._adapt_to_java(adapted_content)
        
        return adapted_content
    
    def _adapt_assertion(self, assertion: TestAssertion, mapping: TestFrameworkMapping) -> TestAssertion:
        """Adapt assertion to framework."""
        
        # Get framework-specific assertion syntax
        assertion_syntax = mapping.assertion_mappings.get(assertion.assertion_type)
        
        if assertion_syntax:
            # Create adapted assertion with framework syntax
            adapted_assertion = TestAssertion(
                assertion_type=assertion.assertion_type,
                actual=assertion.actual,
                expected=assertion.expected,
                message=assertion.message,
                framework_syntax=assertion_syntax
            )
        else:
            # Use original assertion if no mapping available
            adapted_assertion = assertion
        
        return adapted_assertion
    
    def _generate_assertion_code(self, assertions: List[TestAssertion], mapping: TestFrameworkMapping) -> str:
        """Generate framework-specific assertion code."""
        
        assertion_lines = []
        
        for assertion in assertions:
            assertion_code = self._get_assertion_code(assertion, mapping)
            if assertion_code:
                assertion_lines.append(assertion_code)
        
        return "\\n".join(assertion_lines)
    
    def _get_assertion_code(self, assertion: TestAssertion, mapping: TestFrameworkMapping) -> str:
        """Get framework-specific assertion code."""
        
        # Use framework syntax if available
        if hasattr(assertion, 'framework_syntax') and assertion.framework_syntax:
            template = assertion.framework_syntax
        else:
            # Get default assertion syntax for framework
            template = mapping.assertion_mappings.get(assertion.assertion_type, "assert {actual}")
        
        # Format assertion
        assertion_code = template.format(
            actual=assertion.actual,
            expected=assertion.expected,
            message=assertion.message or ""
        )
        
        return assertion_code
    
    def _adapt_test_name(self, name: str, config: FrameworkAdapterConfig) -> str:
        """Adapt test name according to framework conventions."""
        
        # Apply naming convention
        if not name.startswith("test_"):
            name = f"test_{name}"
        
        # Remove invalid characters and apply naming format
        adapted_name = config.test_method_naming.format(method_name=name.replace("test_", ""))
        
        return adapted_name
    
    def _load_framework_mappings(self) -> Dict[SupportedFramework, TestFrameworkMapping]:
        """Load framework mappings."""
        
        mappings = {}
        
        # Python - pytest
        mappings[SupportedFramework.PYTEST] = TestFrameworkMapping(
            framework=SupportedFramework.PYTEST,
            language="python",
            file_extension=".py",
            test_file_template="# pytest test file\\n{imports}\\n\\n{test_classes}",
            test_class_template="class {class_name}:\\n{test_methods}",
            test_method_template="def {test_name}():\\n    \\\"\\\"\\\"{description}\\\"\\\"\\\"\\n    {test_content}\\n    {assertions}",
            assertion_mappings={
                AssertionType.EQUALS: "assert {actual} == {expected}",
                AssertionType.NOT_EQUALS: "assert {actual} != {expected}",
                AssertionType.TRUE: "assert {actual}",
                AssertionType.FALSE: "assert not {actual}",
                AssertionType.CONTAINS: "assert {expected} in {actual}",
                AssertionType.NOT_CONTAINS: "assert {expected} not in {actual}",
                AssertionType.THROWS: "with pytest.raises({expected}):\\n        {actual}",
                AssertionType.GREATER_THAN: "assert {actual} > {expected}",
                AssertionType.LESS_THAN: "assert {actual} < {expected}"
            },
            required_imports=["import pytest"]
        )
        
        # Python - unittest
        mappings[SupportedFramework.UNITTEST] = TestFrameworkMapping(
            framework=SupportedFramework.UNITTEST,
            language="python",
            file_extension=".py",
            test_file_template="import unittest\\n{imports}\\n\\n{test_classes}",
            test_class_template="class {class_name}(unittest.TestCase):\\n{test_methods}",
            test_method_template="    def {test_name}(self):\\n        \\\"\\\"\\\"{description}\\\"\\\"\\\"\\n        {test_content}\\n        {assertions}",
            assertion_mappings={
                AssertionType.EQUALS: "self.assertEqual({actual}, {expected})",
                AssertionType.NOT_EQUALS: "self.assertNotEqual({actual}, {expected})",
                AssertionType.TRUE: "self.assertTrue({actual})",
                AssertionType.FALSE: "self.assertFalse({actual})",
                AssertionType.CONTAINS: "self.assertIn({expected}, {actual})",
                AssertionType.NOT_CONTAINS: "self.assertNotIn({expected}, {actual})",
                AssertionType.THROWS: "self.assertRaises({expected}, {actual})",
                AssertionType.GREATER_THAN: "self.assertGreater({actual}, {expected})",
                AssertionType.LESS_THAN: "self.assertLess({actual}, {expected})"
            },
            required_imports=["import unittest"]
        )
        
        # JavaScript - Jest
        mappings[SupportedFramework.JEST] = TestFrameworkMapping(
            framework=SupportedFramework.JEST,
            language="javascript",
            file_extension=".test.js",
            test_file_template="{imports}\\n\\n{test_classes}",
            test_class_template="describe('{class_name}', () => {{\\n{test_methods}\\n}});",
            test_method_template="  test('{test_name}', () => {{\\n    // {description}\\n    {test_content}\\n    {assertions}\\n  }});",
            assertion_mappings={
                AssertionType.EQUALS: "expect({actual}).toBe({expected});",
                AssertionType.NOT_EQUALS: "expect({actual}).not.toBe({expected});",
                AssertionType.TRUE: "expect({actual}).toBeTruthy();",
                AssertionType.FALSE: "expect({actual}).toBeFalsy();",
                AssertionType.CONTAINS: "expect({actual}).toContain({expected});",
                AssertionType.NOT_CONTAINS: "expect({actual}).not.toContain({expected});",
                AssertionType.THROWS: "expect(() => {{ {actual} }}).toThrow({expected});",
                AssertionType.GREATER_THAN: "expect({actual}).toBeGreaterThan({expected});",
                AssertionType.LESS_THAN: "expect({actual}).toBeLessThan({expected});"
            }
        )
        
        # Java - JUnit
        mappings[SupportedFramework.JUNIT] = TestFrameworkMapping(
            framework=SupportedFramework.JUNIT,
            language="java",
            file_extension=".java",
            test_file_template="import org.junit.jupiter.api.Test;\\nimport static org.junit.jupiter.api.Assertions.*;\\n{imports}\\n\\npublic class {class_name} {{\\n{test_methods}\\n}}",
            test_class_template="public class {class_name} {{\\n{test_methods}\\n}}",
            test_method_template="    @Test\\n    public void {test_name}() {{\\n        // {description}\\n        {test_content}\\n        {assertions}\\n    }}",
            assertion_mappings={
                AssertionType.EQUALS: "assertEquals({expected}, {actual});",
                AssertionType.NOT_EQUALS: "assertNotEquals({expected}, {actual});",
                AssertionType.TRUE: "assertTrue({actual});",
                AssertionType.FALSE: "assertFalse({actual});",
                AssertionType.THROWS: "assertThrows({expected}.class, () -> {{ {actual} }});",
                AssertionType.NULL: "assertNull({actual});",
                AssertionType.NOT_NULL: "assertNotNull({actual});"
            },
            required_imports=["import org.junit.jupiter.api.Test;", "import static org.junit.jupiter.api.Assertions.*;"]
        )
        
        return mappings
    
    def _get_default_test_template(self, framework: SupportedFramework) -> str:
        """Get default test template for framework."""
        
        templates = {
            SupportedFramework.PYTEST: "def {test_name}():\\n    {test_content}\\n    {assertions}",
            SupportedFramework.UNITTEST: "def {test_name}(self):\\n    {test_content}\\n    {assertions}",
            SupportedFramework.JEST: "test('{test_name}', () => {{\\n    {test_content}\\n    {assertions}\\n}});",
            SupportedFramework.JUNIT: "@Test\\npublic void {test_name}() {{\\n    {test_content}\\n    {assertions}\\n}}"
        }
        
        return templates.get(framework, "def {test_name}():\\n    {test_content}\\n    {assertions}")
    
    def _adapt_to_javascript(self, content: str) -> str:
        """Adapt Python-like content to JavaScript."""
        
        # Basic Python to JavaScript adaptations
        adaptations = {
            "True": "true",
            "False": "false",
            "None": "null",
            "self.": "this.",
            "def ": "function ",
            "__init__": "constructor"
        }
        
        adapted = content
        for python_syntax, js_syntax in adaptations.items():
            adapted = adapted.replace(python_syntax, js_syntax)
        
        return adapted
    
    def _adapt_to_java(self, content: str) -> str:
        """Adapt Python-like content to Java."""
        
        # Basic Python to Java adaptations
        adaptations = {
            "True": "true",
            "False": "false",
            "None": "null",
            "self.": "this.",
            "def ": "public ",
            "__init__": "constructor"
        }
        
        adapted = content
        for python_syntax, java_syntax in adaptations.items():
            adapted = adapted.replace(python_syntax, java_syntax)
        
        return adapted
    
    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported frameworks."""
        return [framework.value for framework in SupportedFramework]
    
    def get_frameworks_for_language(self, language: str) -> List[str]:
        """Get supported frameworks for a specific language."""
        frameworks = []
        
        for framework, mapping in self.framework_mappings.items():
            if mapping.language.lower() == language.lower():
                frameworks.append(framework.value)
        
        return frameworks
    
    def detect_frameworks_in_directory(self, directory_path: str) -> List[str]:
        """Detect testing frameworks used in a directory."""
        detected_frameworks = []
        directory = Path(directory_path)
        
        # Framework detection patterns
        detection_patterns = {
            SupportedFramework.PYTEST: ["pytest.ini", "conftest.py", "import pytest"],
            SupportedFramework.UNITTEST: ["import unittest"],
            SupportedFramework.JEST: ["package.json", "jest.config.js", "describe(", "test("],
            SupportedFramework.JUNIT: ["@Test", "import org.junit"],
            SupportedFramework.NUNIT: ["[Test]", "using NUnit"],
            SupportedFramework.GO_TEST: ["_test.go", "func Test"]
        }
        
        # Check files for framework patterns
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    for framework, patterns in detection_patterns.items():
                        for pattern in patterns:
                            if pattern in content or pattern == file_path.name:
                                if framework.value not in detected_frameworks:
                                    detected_frameworks.append(framework.value)
                                break
                
                except Exception:
                    continue
        
        return detected_frameworks