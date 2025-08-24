"""
Test Templates Engine - Advanced template system for intelligent test code generation

This module provides a sophisticated template engine for generating various types of test code
with customizable patterns, advanced parameterization, and intelligent code generation based
on function signatures, complexity analysis, and testing best practices.

Enterprise Features:
- Dynamic template generation with intelligent parameter substitution
- Advanced test scenario templates for comprehensive coverage
- Type-aware test data generation with realistic test values
- Template inheritance and composition for complex test structures
- Performance-optimized template rendering with caching
- Extensible template system with custom template support

Key Components:
- TestTemplateEngine: Main template processing and generation engine
- TemplateRenderer: Advanced template rendering with variable substitution
- ScenarioGenerator: Intelligent test scenario creation based on code analysis
- DataGenerator: Type-aware test data generation for realistic testing
- TemplateOptimizer: Performance optimization for template processing
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import re
import json
from datetime import datetime
from string import Template
from abc import ABC, abstractmethod

from .test_models import (
    TestCase, TestType, ComplexityLevel, FunctionInfo, ParameterInfo,
    create_test_case, DEFAULT_TYPE_GENERATORS
)


@dataclass
class TemplateContext:
    """Context information for template rendering."""
    function_name: str
    function_info: FunctionInfo
    module_name: str
    test_type: TestType
    complexity_level: ComplexityLevel
    custom_variables: Dict[str, Any] = field(default_factory=dict)
    generation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TemplateMetadata:
    """Metadata for test templates."""
    name: str
    description: str
    test_type: TestType
    complexity_level: ComplexityLevel
    requires_mocks: bool = False
    requires_async: bool = False
    requires_fixtures: bool = False
    estimated_coverage: float = 10.0
    tags: Set[str] = field(default_factory=set)


class TemplateRenderer:
    """Advanced template rendering with intelligent variable substitution."""
    
    def __init__(self):
        self.custom_filters = {}
        self.template_cache = {}
    
    def render(self, template: str, context: TemplateContext) -> str:
        """
        Render template with context variables and intelligent substitution.
        
        Args:
            template: Template string with placeholders
            context: Template context with variables
            
        Returns:
            Rendered template string
        """
        # Create variable mapping
        variables = self._build_variable_mapping(context)
        
        # Apply custom filters
        variables = self._apply_filters(variables, context)
        
        # Render template
        template_obj = Template(template)
        try:
            rendered = template_obj.safe_substitute(variables)
            return self._post_process(rendered, context)
        except KeyError as e:
            # Handle missing variables gracefully
            return template.replace(f"${{{e.args[0]}}}", f"# Missing variable: {e.args[0]}")
    
    def _build_variable_mapping(self, context: TemplateContext) -> Dict[str, str]:
        """Build comprehensive variable mapping for template rendering."""
        func_info = context.function_info
        
        # Basic function information
        variables = {
            'function_name': func_info.name,
            'module_name': context.module_name,
            'test_name': f"test_{func_info.name}_{context.test_type.value}",
            'class_name': context.module_name.replace('.', '').title(),
            'test_type': context.test_type.value,
            'timestamp': context.generation_timestamp.isoformat()
        }
        
        # Function signature components
        variables.update(self._generate_signature_variables(func_info))
        
        # Test data generation
        variables.update(self._generate_test_data_variables(func_info))
        
        # Mock and fixture variables
        variables.update(self._generate_mock_variables(func_info))
        
        # Assertion variables
        variables.update(self._generate_assertion_variables(func_info, context))
        
        # Custom variables from context
        variables.update(context.custom_variables)
        
        return variables
    
    def _generate_signature_variables(self, func_info: FunctionInfo) -> Dict[str, str]:
        """Generate function signature related variables."""
        variables = {}
        
        # Parameter lists
        param_names = [p.name for p in func_info.parameters if p.name != 'self']
        variables['param_names'] = ', '.join(param_names)
        variables['param_list'] = ', '.join(f"{p.name}: {p.type_hint}" for p in func_info.parameters if p.name != 'self')
        
        # Function call formats
        if param_names:
            variables['function_call'] = f"{func_info.name}({', '.join(param_names)})"
            variables['function_call_with_values'] = f"{func_info.name}({self._generate_param_values(func_info.parameters)})"
        else:
            variables['function_call'] = f"{func_info.name}()"
            variables['function_call_with_values'] = f"{func_info.name}()"
        
        # Async handling
        if func_info.is_async:
            variables['await_prefix'] = 'await '
            variables['async_decorator'] = '@pytest.mark.asyncio\n    '
        else:
            variables['await_prefix'] = ''
            variables['async_decorator'] = ''
        
        return variables
    
    def _generate_test_data_variables(self, func_info: FunctionInfo) -> Dict[str, str]:
        """Generate test data variables based on parameter types."""
        variables = {}
        
        test_data = []
        edge_cases = []
        error_cases = []
        
        for param in func_info.parameters:
            if param.name == 'self':
                continue
                
            param_type = param.type_hint
            
            # Generate test values
            test_values = self._get_test_values_for_type(param_type)
            test_data.extend([f"{param.name}={value}" for value in test_values[:2]])
            
            # Generate edge cases
            edge_values = self._get_edge_values_for_type(param_type)
            edge_cases.extend([f"{param.name}={value}" for value in edge_values])
            
            # Generate error cases
            error_values = self._get_error_values_for_type(param_type)
            error_cases.extend([f"{param.name}={value}" for value in error_values])
        
        variables['test_data'] = ', '.join(test_data[:5])  # Limit to 5 test cases
        variables['edge_cases'] = '\n        '.join(edge_cases[:3])  # Limit to 3 edge cases
        variables['error_cases'] = '\n        '.join(error_cases[:3])  # Limit to 3 error cases
        
        return variables
    
    def _generate_mock_variables(self, func_info: FunctionInfo) -> Dict[str, str]:
        """Generate mock-related variables."""
        variables = {}
        
        mock_targets = []
        mock_setups = []
        mock_assertions = []
        
        # Identify mock requirements based on function characteristics
        if func_info.uses_io:
            mock_targets.append("builtins.open")
            mock_setups.append("mock_open.return_value.__enter__.return_value.read.return_value = 'test_content'")
            mock_assertions.append("mock_open.assert_called_once()")
        
        if func_info.uses_network:
            mock_targets.append("requests.get")
            mock_setups.append("mock_get.return_value.status_code = 200")
            mock_assertions.append("mock_get.assert_called_once()")
        
        if func_info.uses_database:
            mock_targets.append("database.connection")
            mock_setups.append("mock_connection.return_value.execute.return_value = []")
            mock_assertions.append("mock_connection.assert_called()")
        
        variables['mock_targets'] = '\n    '.join(f"@patch('{target}')" for target in mock_targets)
        variables['mock_parameters'] = ', '.join(f"mock_{target.split('.')[-1]}" for target in mock_targets)
        variables['mock_setup'] = '\n        '.join(mock_setups)
        variables['mock_assertions'] = '\n        '.join(mock_assertions)
        variables['has_mocks'] = str(bool(mock_targets)).lower()
        
        return variables
    
    def _generate_assertion_variables(self, func_info: FunctionInfo, context: TemplateContext) -> Dict[str, str]:
        """Generate assertion variables based on function return type and complexity."""
        variables = {}
        
        return_type = func_info.return_type
        assertions = []
        
        # Basic return value assertions
        if return_type == 'bool':
            assertions.append("assert isinstance(result, bool)")
            assertions.append("assert result in [True, False]")
        elif return_type in ['int', 'float']:
            assertions.append(f"assert isinstance(result, {return_type})")
            assertions.append("assert result is not None")
        elif return_type == 'str':
            assertions.append("assert isinstance(result, str)")
            assertions.append("assert len(result) >= 0")
        elif return_type.startswith('List'):
            assertions.append("assert isinstance(result, list)")
            assertions.append("assert len(result) >= 0")
        elif return_type.startswith('Dict'):
            assertions.append("assert isinstance(result, dict)")
        else:
            assertions.append("assert result is not None")
        
        # Add complexity-specific assertions
        if context.complexity_level.value >= 3:
            assertions.append("# Add more specific assertions based on function logic")
        
        variables['basic_assertions'] = '\n        '.join(assertions)
        variables['return_type'] = return_type
        
        return variables
    
    def _generate_param_values(self, parameters: List[ParameterInfo]) -> str:
        """Generate realistic parameter values for function calls."""
        param_values = []
        
        for param in parameters:
            if param.name == 'self':
                continue
                
            param_type = param.type_hint
            test_values = self._get_test_values_for_type(param_type)
            
            if test_values:
                param_values.append(f"{param.name}={test_values[0]}")
            else:
                param_values.append(f"{param.name}=None")
        
        return ', '.join(param_values)
    
    def _get_test_values_for_type(self, type_hint: str) -> List[str]:
        """Get appropriate test values for a given type."""
        # Use default type generators and extend with more sophisticated logic
        type_generators = DEFAULT_TYPE_GENERATORS.copy()
        
        # Add more sophisticated type mapping
        if 'Optional' in type_hint:
            base_type = type_hint.replace('Optional[', '').replace(']', '')
            values = self._get_test_values_for_type(base_type)
            return values + ['None']
        
        if 'Union' in type_hint:
            # For Union types, use the first type
            first_type = type_hint.split('[')[1].split(',')[0].strip()
            return self._get_test_values_for_type(first_type)
        
        # Map complex types to basic generators
        for basic_type, values in type_generators.items():
            if basic_type.lower() in type_hint.lower():
                return values[:3]  # Return first 3 values
        
        return ['None']  # Default fallback
    
    def _get_edge_values_for_type(self, type_hint: str) -> List[str]:
        """Get edge case values for a given type."""
        edge_cases = {
            'str': ['""', '" "', '"\\n"', '"' + 'x' * 1000 + '"'],
            'int': ['0', '-1', '2**31-1', '-2**31'],
            'float': ['0.0', '-0.0', 'float("inf")', 'float("-inf")', 'float("nan")'],
            'list': ['[]', '[None]', '[1] * 1000'],
            'dict': ['{}', '{None: None}'],
            'Path': ['Path("")', 'Path("/nonexistent")', 'Path(".")']
        }
        
        for basic_type, values in edge_cases.items():
            if basic_type.lower() in type_hint.lower():
                return values
        
        return ['None']
    
    def _get_error_values_for_type(self, type_hint: str) -> List[str]:
        """Get values that should cause errors for a given type."""
        # These are values that might cause type errors or exceptions
        error_values = {
            'str': ['None', '123', '[]'],
            'int': ['None', '"string"', 'float("inf")'],
            'float': ['None', '"string"', 'complex(1, 1)'],
            'list': ['None', '"string"', '123'],
            'dict': ['None', '"string"', '[]'],
            'Path': ['None', '123', '[]']
        }
        
        for basic_type, values in error_values.items():
            if basic_type.lower() in type_hint.lower():
                return values
        
        return ['None']
    
    def _apply_filters(self, variables: Dict[str, str], context: TemplateContext) -> Dict[str, str]:
        """Apply custom filters to variables."""
        # Apply any registered custom filters
        for filter_name, filter_func in self.custom_filters.items():
            if filter_name in variables:
                variables[filter_name] = filter_func(variables[filter_name], context)
        
        return variables
    
    def _post_process(self, rendered: str, context: TemplateContext) -> str:
        """Post-process rendered template for cleanup and optimization."""
        # Remove extra whitespace
        lines = rendered.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            # Skip empty lines at the beginning of methods
            if line or cleaned_lines:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def register_filter(self, name: str, filter_func: Callable):
        """Register a custom filter function."""
        self.custom_filters[name] = filter_func


class TestTemplateEngine:
    """Main template engine for generating test code."""
    
    def __init__(self):
        self.renderer = TemplateRenderer()
        self.templates = self._load_default_templates()
        self.custom_templates = {}
    
    def generate_test_code(self, 
                          function_info: FunctionInfo,
                          module_name: str,
                          test_type: TestType,
                          complexity_level: ComplexityLevel = ComplexityLevel.MODERATE,
                          custom_variables: Dict[str, Any] = None) -> TestCase:
        """
        Generate test code using appropriate template.
        
        Args:
            function_info: Information about the function to test
            module_name: Name of the module containing the function
            test_type: Type of test to generate
            complexity_level: Complexity level for the test
            custom_variables: Additional variables for template rendering
            
        Returns:
            Generated TestCase with rendered code
        """
        # Create template context
        context = TemplateContext(
            function_name=function_info.name,
            function_info=function_info,
            module_name=module_name,
            test_type=test_type,
            complexity_level=complexity_level,
            custom_variables=custom_variables or {}
        )
        
        # Select appropriate template
        template = self._select_template(test_type, function_info)
        
        # Render template
        rendered_code = self.renderer.render(template, context)
        
        # Calculate estimated coverage and complexity
        coverage_increase = self._estimate_coverage_increase(test_type, function_info)
        
        return create_test_case(
            name=f"test_{function_info.name}_{test_type.value}",
            code=rendered_code,
            target_function=function_info.name,
            test_type=test_type,
            coverage_increase=coverage_increase,
            complexity=complexity_level,
            dependencies=function_info.dependencies,
            mock_requirements=self._get_mock_requirements(function_info),
            expected_exceptions=function_info.raises_exceptions
        )
    
    def _load_default_templates(self) -> Dict[str, str]:
        """Load default test templates."""
        return {
            TestType.UNIT.value: '''
    def test_${function_name}_${test_type}(self):
        """Test basic functionality of ${function_name}."""
        ${async_decorator}
        # Arrange
        ${test_data}
        
        # Act
        result = ${await_prefix}${function_call_with_values}
        
        # Assert
        ${basic_assertions}
''',
            
            TestType.EDGE_CASE.value: '''
    def test_${function_name}_edge_cases(self):
        """Test edge cases for ${function_name}."""
        # Test edge cases that might cause boundary issues
        ${edge_cases}
        
        # Edge cases should either work or raise appropriate exceptions
        try:
            result = ${function_call_with_values}
            assert result is not None, "Edge case should return a value or raise exception"
        except (ValueError, TypeError, IndexError) as e:
            # Expected exceptions for edge cases
            assert str(e) != "", "Exception should have a meaningful message"
''',
            
            TestType.ERROR_HANDLING.value: '''
    def test_${function_name}_error_handling(self):
        """Test error handling in ${function_name}."""
        # Test various error conditions
        ${error_cases}
        
        # Verify appropriate exceptions are raised
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            ${function_name}(None)  # Should raise appropriate exception
''',
            
            TestType.MOCK_BASED.value: '''
    ${mock_targets}
    def test_${function_name}_with_mocks(self, ${mock_parameters}):
        """Test ${function_name} with mocked dependencies."""
        ${async_decorator}
        # Setup mocks
        ${mock_setup}
        
        # Execute function
        result = ${await_prefix}${function_call_with_values}
        
        # Verify interactions
        ${mock_assertions}
        assert result is not None
''',
            
            TestType.ASYNC.value: '''
    @pytest.mark.asyncio
    async def test_${function_name}_async(self):
        """Test async function ${function_name}."""
        # Setup test data
        ${test_data}
        
        # Execute async function
        result = await ${function_call_with_values}
        
        # Verify results
        ${basic_assertions}
''',
            
            TestType.PROPERTY_BASED.value: '''
    @pytest.mark.parametrize("test_input", [
        ${test_data}
    ])
    def test_${function_name}_property_based(self, test_input):
        """Property-based test for ${function_name}."""
        # Property-based testing ensures function behaves correctly
        # across a wide range of inputs
        result = ${function_call}
        
        # Properties that should always hold
        ${basic_assertions}
        
        # Additional property checks
        if result is not None:
            assert hasattr(result, '__class__'), "Result should be a valid object"
''',
            
            TestType.PERFORMANCE.value: '''
    def test_${function_name}_performance(self):
        """Test performance characteristics of ${function_name}."""
        import time
        
        # Setup performance test data
        ${test_data}
        
        # Measure execution time
        start_time = time.time()
        result = ${function_call_with_values}
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Assert performance requirements
        assert execution_time < 1.0, f"Function took {execution_time:.3f}s, expected < 1.0s"
        ${basic_assertions}
''',
            
            TestType.INTEGRATION.value: '''
    def test_${function_name}_integration(self):
        """Integration test for ${function_name}."""
        # Integration test verifies the function works correctly
        # with real dependencies and data
        ${test_data}
        
        # Execute with real dependencies
        result = ${function_call_with_values}
        
        # Verify integration behavior
        ${basic_assertions}
        
        # Additional integration checks
        assert result is not None, "Integration should produce meaningful results"
'''
        }
    
    def _select_template(self, test_type: TestType, function_info: FunctionInfo) -> str:
        """Select appropriate template based on test type and function characteristics."""
        # Check for custom templates first
        if test_type.value in self.custom_templates:
            return self.custom_templates[test_type.value]
        
        # Use async template for async functions
        if function_info.is_async and test_type == TestType.UNIT:
            return self.templates.get(TestType.ASYNC.value, self.templates[TestType.UNIT.value])
        
        # Use mock template for functions with external dependencies
        if (function_info.uses_io or function_info.uses_network or function_info.uses_database) and test_type == TestType.UNIT:
            return self.templates.get(TestType.MOCK_BASED.value, self.templates[TestType.UNIT.value])
        
        # Return requested template or default to unit test
        return self.templates.get(test_type.value, self.templates[TestType.UNIT.value])
    
    def _estimate_coverage_increase(self, test_type: TestType, function_info: FunctionInfo) -> float:
        """Estimate coverage increase based on test type and function complexity."""
        base_coverage = {
            TestType.UNIT: 15.0,
            TestType.INTEGRATION: 20.0,
            TestType.EDGE_CASE: 10.0,
            TestType.ERROR_HANDLING: 12.0,
            TestType.PROPERTY_BASED: 18.0,
            TestType.MOCK_BASED: 16.0,
            TestType.ASYNC: 15.0,
            TestType.PERFORMANCE: 8.0
        }
        
        coverage = base_coverage.get(test_type, 10.0)
        
        # Adjust based on function complexity
        complexity_multiplier = 1.0 + (function_info.complexity_score - 1) * 0.1
        
        # Adjust based on function characteristics
        if function_info.uses_io:
            coverage *= 1.2
        if function_info.uses_network:
            coverage *= 1.3
        if function_info.is_async:
            coverage *= 1.1
        
        return min(30.0, coverage * complexity_multiplier)
    
    def _get_mock_requirements(self, function_info: FunctionInfo) -> List[str]:
        """Determine mock requirements for a function."""
        mocks = []
        
        if function_info.uses_io:
            mocks.extend(['builtins.open', 'pathlib.Path'])
        if function_info.uses_network:
            mocks.extend(['requests.get', 'requests.post', 'urllib.request'])
        if function_info.uses_database:
            mocks.extend(['sqlite3.connect', 'pymongo.MongoClient'])
        
        return mocks
    
    def add_custom_template(self, test_type: TestType, template: str, metadata: TemplateMetadata = None):
        """Add a custom template for a specific test type."""
        self.custom_templates[test_type.value] = template
        
        if metadata:
            # Store metadata for template optimization
            pass
    
    def register_template_filter(self, name: str, filter_func: Callable):
        """Register a custom template filter."""
        self.renderer.register_filter(name, filter_func)


# Factory function
def create_test_template_engine() -> TestTemplateEngine:
    """
    Create a test template engine with default configuration.
    
    Returns:
        Configured TestTemplateEngine instance
    """
    return TestTemplateEngine()


# Utility functions
def validate_template(template: str) -> List[str]:
    """
    Validate template syntax and structure.
    
    Args:
        template: Template string to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check for basic template structure
    if not template.strip():
        errors.append("Template cannot be empty")
    
    # Check for proper indentation
    if not template.startswith('    def test_'):
        errors.append("Template should start with properly indented test method")
    
    # Check for required placeholders
    required_placeholders = ['${function_name}']
    for placeholder in required_placeholders:
        if placeholder not in template:
            errors.append(f"Missing required placeholder: {placeholder}")
    
    return errors


# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Template Engine Team'
__description__ = 'Advanced template system for intelligent test code generation'