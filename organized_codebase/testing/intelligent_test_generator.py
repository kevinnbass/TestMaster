"""
Intelligent Test Generator
AI-powered test generation using extracted patterns and intelligence synthesis.
"""

import os
import ast
import json
import time
import random
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
from enum import Enum
import re
import hashlib


class TestComplexity(Enum):
    """Test complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class TestCategory(Enum):
    """Categories of tests to generate"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    EDGE_CASE = "edge_case"
    REGRESSION = "regression"


@dataclass
class CodeAnalysis:
    """Analysis of code to be tested"""
    file_path: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    dependencies: List[str]
    complexity_score: float
    test_coverage_gaps: List[str]
    potential_edge_cases: List[str]
    security_concerns: List[str]
    performance_hotspots: List[str]


@dataclass
class TestTemplate:
    """Template for test generation"""
    template_id: str
    name: str
    description: str
    category: TestCategory
    complexity: TestComplexity
    template_code: str
    required_imports: List[str]
    setup_code: str
    teardown_code: str
    pattern_tags: List[str] = field(default_factory=list)
    applicability_score: float = 1.0
    
    def generate_test(self, context: Dict[str, Any]) -> str:
        """Generate test code from template"""
        code = self.template_code
        
        # Replace placeholders with context values
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            if isinstance(value, str):
                code = code.replace(placeholder, value)
            elif isinstance(value, list):
                code = code.replace(placeholder, ", ".join(map(str, value)))
            else:
                code = code.replace(placeholder, str(value))
        
        # Add imports
        imports_code = "\n".join(self.required_imports)
        
        # Combine all parts
        full_code = f"{imports_code}\n\n"
        if self.setup_code:
            full_code += f"{self.setup_code}\n\n"
        full_code += code
        if self.teardown_code:
            full_code += f"\n\n{self.teardown_code}"
        
        return full_code


@dataclass
class GeneratedTest:
    """A generated test case"""
    test_id: str
    name: str
    code: str
    category: TestCategory
    complexity: TestComplexity
    target_function: str
    confidence_score: float
    estimated_coverage: float
    generation_time: datetime
    template_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeAnalyzer:
    """Analyze code for intelligent test generation"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_file(self, file_path: str) -> CodeAnalysis:
        """Comprehensive analysis of a Python file"""
        if file_path in self.analysis_cache:
            return self.analysis_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = CodeAnalysis(
                file_path=file_path,
                functions=[],
                classes=[],
                imports=[],
                dependencies=[],
                complexity_score=0.0,
                test_coverage_gaps=[],
                potential_edge_cases=[],
                security_concerns=[],
                performance_hotspots=[]
            )
            
            # Analyze AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node, content)
                    analysis.functions.append(func_info)
                    analysis.complexity_score += func_info['complexity']
                
                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, content)
                    analysis.classes.append(class_info)
                    analysis.complexity_score += class_info['complexity']
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._extract_import(node)
                    analysis.imports.extend(import_info)
            
            # Identify gaps and concerns
            analysis.test_coverage_gaps = self._identify_coverage_gaps(analysis)
            analysis.potential_edge_cases = self._identify_edge_cases(analysis)
            analysis.security_concerns = self._identify_security_concerns(analysis)
            analysis.performance_hotspots = self._identify_performance_hotspots(analysis)
            
            self.analysis_cache[file_path] = analysis
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return CodeAnalysis(
                file_path=file_path, functions=[], classes=[], imports=[],
                dependencies=[], complexity_score=0.0, test_coverage_gaps=[],
                potential_edge_cases=[], security_concerns=[], performance_hotspots=[]
            )
    
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze a function node"""
        func_lines = content.split('\n')[node.lineno-1:getattr(node, 'end_lineno', node.lineno+10)]
        func_code = '\n'.join(func_lines)
        
        # Extract parameters
        params = []
        for arg in node.args.args:
            param_info = {'name': arg.arg, 'annotation': None}
            if arg.annotation:
                param_info['annotation'] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else 'annotated'
            params.append(param_info)
        
        # Calculate complexity
        complexity = len(node.body) * 0.5
        complexity += len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.While, ast.For))]) * 0.3
        
        # Identify patterns
        has_exception_handling = any(isinstance(n, ast.Try) for n in ast.walk(node))
        has_loops = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node))
        has_conditionals = any(isinstance(n, ast.If) for n in ast.walk(node))
        
        return {
            'name': node.name,
            'parameters': params,
            'returns': ast.unparse(node.returns) if node.returns and hasattr(ast, 'unparse') else None,
            'docstring': ast.get_docstring(node),
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [ast.unparse(d) if hasattr(ast, 'unparse') else 'decorated' for d in node.decorator_list],
            'complexity': complexity,
            'line_count': len(func_lines),
            'has_exception_handling': has_exception_handling,
            'has_loops': has_loops,
            'has_conditionals': has_conditionals,
            'code': func_code
        }
    
    def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze a class node"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, content)
                methods.append(method_info)
        
        # Calculate complexity
        complexity = len(methods) * 1.0 + len(node.body) * 0.2
        
        return {
            'name': node.name,
            'bases': [ast.unparse(base) if hasattr(ast, 'unparse') else 'base' for base in node.bases],
            'decorators': [ast.unparse(d) if hasattr(ast, 'unparse') else 'decorated' for d in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'methods': methods,
            'complexity': complexity,
            'is_exception': any('Exception' in base for base in [ast.unparse(base) if hasattr(ast, 'unparse') else '' for base in node.bases])
        }
    
    def _extract_import(self, node: Union[ast.Import, ast.ImportFrom]) -> List[str]:
        """Extract import information"""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return imports
    
    def _identify_coverage_gaps(self, analysis: CodeAnalysis) -> List[str]:
        """Identify potential test coverage gaps"""
        gaps = []
        
        for func in analysis.functions:
            if func['has_exception_handling']:
                gaps.append(f"Exception handling in {func['name']} needs error path testing")
            
            if func['has_loops']:
                gaps.append(f"Loop conditions in {func['name']} need boundary testing")
            
            if len(func['parameters']) > 3:
                gaps.append(f"Multiple parameters in {func['name']} need combination testing")
            
            if func['complexity'] > 10:
                gaps.append(f"High complexity in {func['name']} needs comprehensive path testing")
        
        return gaps
    
    def _identify_edge_cases(self, analysis: CodeAnalysis) -> List[str]:
        """Identify potential edge cases"""
        edge_cases = []
        
        for func in analysis.functions:
            # Parameter-based edge cases
            for param in func['parameters']:
                if 'str' in str(param.get('annotation', '')).lower():
                    edge_cases.append(f"Empty string, None, and special characters for {param['name']} in {func['name']}")
                elif 'int' in str(param.get('annotation', '')).lower():
                    edge_cases.append(f"Zero, negative, and boundary values for {param['name']} in {func['name']}")
                elif 'list' in str(param.get('annotation', '')).lower():
                    edge_cases.append(f"Empty list, single item, and large lists for {param['name']} in {func['name']}")
            
            # Function-specific edge cases
            if 'divide' in func['name'].lower() or '//' in func.get('code', ''):
                edge_cases.append(f"Division by zero in {func['name']}")
            
            if 'file' in func['name'].lower():
                edge_cases.append(f"File not found, permission errors in {func['name']}")
        
        return edge_cases
    
    def _identify_security_concerns(self, analysis: CodeAnalysis) -> List[str]:
        """Identify security concerns that need testing"""
        concerns = []
        
        # Check imports for security-sensitive modules
        security_modules = ['os', 'subprocess', 'pickle', 'eval', 'exec']
        for imp in analysis.imports:
            for sec_mod in security_modules:
                if sec_mod in imp:
                    concerns.append(f"Security testing needed for {sec_mod} usage")
        
        # Check functions for security patterns
        for func in analysis.functions:
            code = func.get('code', '').lower()
            if 'sql' in code or 'query' in code:
                concerns.append(f"SQL injection testing needed in {func['name']}")
            if 'password' in code or 'secret' in code:
                concerns.append(f"Credential handling testing needed in {func['name']}")
        
        return concerns
    
    def _identify_performance_hotspots(self, analysis: CodeAnalysis) -> List[str]:
        """Identify performance hotspots"""
        hotspots = []
        
        for func in analysis.functions:
            if func['has_loops'] and func['complexity'] > 5:
                hotspots.append(f"Performance testing needed for loops in {func['name']}")
            
            code = func.get('code', '').lower()
            if 'sort' in code or 'search' in code:
                hotspots.append(f"Algorithm performance testing needed in {func['name']}")
        
        return hotspots


class TestTemplateLibrary:
    """Library of test templates for different scenarios"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize built-in test templates"""
        
        # Basic unit test template
        self.templates['basic_unit'] = TestTemplate(
            template_id='basic_unit',
            name='Basic Unit Test',
            description='Basic unit test for function testing',
            category=TestCategory.UNIT,
            complexity=TestComplexity.SIMPLE,
            template_code='''def test_{{function_name}}():
    """Test {{function_name}} with basic inputs."""
    # Arrange
    {{setup_variables}}
    
    # Act
    result = {{function_call}}
    
    # Assert
    assert result is not None
    {{additional_assertions}}''',
            required_imports=['import pytest'],
            setup_code='',
            teardown_code='',
            pattern_tags=['unit', 'basic']
        )
        
        # Exception testing template
        self.templates['exception_test'] = TestTemplate(
            template_id='exception_test',
            name='Exception Testing',
            description='Test exception handling and error conditions',
            category=TestCategory.EDGE_CASE,
            complexity=TestComplexity.MODERATE,
            template_code='''def test_{{function_name}}_exceptions():
    """Test {{function_name}} exception handling."""
    with pytest.raises({{exception_type}}):
        {{function_call_with_invalid_input}}
    
    # Test multiple exception scenarios
    {{additional_exception_tests}}''',
            required_imports=['import pytest'],
            setup_code='',
            teardown_code='',
            pattern_tags=['exception', 'error_handling']
        )
        
        # Performance test template
        self.templates['performance_test'] = TestTemplate(
            template_id='performance_test',
            name='Performance Test',
            description='Test performance characteristics',
            category=TestCategory.PERFORMANCE,
            complexity=TestComplexity.ADVANCED,
            template_code='''def test_{{function_name}}_performance():
    """Test {{function_name}} performance."""
    import time
    
    start_time = time.time()
    for _ in range({{iterations}}):
        {{function_call}}
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < {{max_time_seconds}}, f"Function too slow: {execution_time}s"''',
            required_imports=['import pytest', 'import time'],
            setup_code='',
            teardown_code='',
            pattern_tags=['performance', 'timing']
        )
        
        # Mock-based test template
        self.templates['mock_test'] = TestTemplate(
            template_id='mock_test',
            name='Mock-based Test',
            description='Test with mocked dependencies',
            category=TestCategory.INTEGRATION,
            complexity=TestComplexity.COMPLEX,
            template_code='''@pytest.fixture
def mock_{{dependency_name}}():
    """Mock {{dependency_name}} for testing."""
    return Mock()

def test_{{function_name}}_with_mocks(mock_{{dependency_name}}):
    """Test {{function_name}} with mocked dependencies."""
    # Setup mock behavior
    mock_{{dependency_name}}.{{method_name}}.return_value = {{return_value}}
    
    # Execute test
    result = {{function_call}}
    
    # Verify mock interactions
    mock_{{dependency_name}}.{{method_name}}.assert_called_once()
    assert result == {{expected_result}}''',
            required_imports=['import pytest', 'from unittest.mock import Mock'],
            setup_code='',
            teardown_code='',
            pattern_tags=['mock', 'integration']
        )
        
        # Parametrized test template
        self.templates['parametrized_test'] = TestTemplate(
            template_id='parametrized_test',
            name='Parametrized Test',
            description='Test multiple input combinations',
            category=TestCategory.FUNCTIONAL,
            complexity=TestComplexity.MODERATE,
            template_code='''@pytest.mark.parametrize("{{parameter_names}}", [
    {{test_cases}}
])
def test_{{function_name}}_parametrized({{parameter_names}}):
    """Test {{function_name}} with various inputs."""
    result = {{function_call}}
    assert {{assertion_condition}}''',
            required_imports=['import pytest'],
            setup_code='',
            teardown_code='',
            pattern_tags=['parametrized', 'data_driven']
        )
        
        # Async test template
        self.templates['async_test'] = TestTemplate(
            template_id='async_test',
            name='Async Test',
            description='Test asynchronous functions',
            category=TestCategory.UNIT,
            complexity=TestComplexity.COMPLEX,
            template_code='''@pytest.mark.asyncio
async def test_{{function_name}}_async():
    """Test async {{function_name}}."""
    # Arrange
    {{setup_variables}}
    
    # Act
    result = await {{function_call}}
    
    # Assert
    assert result is not None
    {{additional_assertions}}''',
            required_imports=['import pytest', 'import asyncio'],
            setup_code='',
            teardown_code='',
            pattern_tags=['async', 'concurrent']
        )
    
    def get_applicable_templates(self, analysis: CodeAnalysis, 
                               function_info: Dict[str, Any]) -> List[TestTemplate]:
        """Get templates applicable to a specific function"""
        applicable = []
        
        # Always include basic unit test
        applicable.append(self.templates['basic_unit'])
        
        # Add specific templates based on function characteristics
        if function_info['has_exception_handling']:
            applicable.append(self.templates['exception_test'])
        
        if function_info['complexity'] > 5:
            applicable.append(self.templates['performance_test'])
        
        if function_info['is_async']:
            applicable.append(self.templates['async_test'])
        
        if len(function_info['parameters']) > 1:
            applicable.append(self.templates['parametrized_test'])
        
        # Check for dependencies that might need mocking
        external_deps = [imp for imp in analysis.imports 
                        if not imp.startswith(('builtins', 'sys', 'os'))]
        if external_deps:
            applicable.append(self.templates['mock_test'])
        
        return applicable


class IntelligentTestGenerator:
    """Main intelligent test generator"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.template_library = TestTemplateLibrary()
        self.generation_history = []
        self.generated_tests = []
        self.generation_metrics = {
            'files_processed': 0,
            'tests_generated': 0,
            'avg_confidence': 0.0,
            'coverage_improvement': 0.0
        }
    
    def generate_tests_for_file(self, file_path: str, 
                               max_tests_per_function: int = 3) -> List[GeneratedTest]:
        """Generate comprehensive tests for a Python file"""
        print(f"Generating tests for: {file_path}")
        
        # Analyze the file
        analysis = self.analyzer.analyze_file(file_path)
        generated_tests = []
        
        # Generate tests for each function
        for func_info in analysis.functions:
            func_tests = self._generate_function_tests(
                analysis, func_info, max_tests_per_function
            )
            generated_tests.extend(func_tests)
        
        # Generate class-level tests
        for class_info in analysis.classes:
            class_tests = self._generate_class_tests(analysis, class_info)
            generated_tests.extend(class_tests)
        
        # Generate integration tests
        integration_tests = self._generate_integration_tests(analysis)
        generated_tests.extend(integration_tests)
        
        # Update metrics
        self.generation_metrics['files_processed'] += 1
        self.generation_metrics['tests_generated'] += len(generated_tests)
        
        if generated_tests:
            avg_confidence = sum(test.confidence_score for test in generated_tests) / len(generated_tests)
            self.generation_metrics['avg_confidence'] = avg_confidence
        
        self.generated_tests.extend(generated_tests)
        
        print(f"Generated {len(generated_tests)} tests for {file_path}")
        return generated_tests
    
    def _generate_function_tests(self, analysis: CodeAnalysis, 
                               func_info: Dict[str, Any],
                               max_tests: int) -> List[GeneratedTest]:
        """Generate tests for a specific function"""
        applicable_templates = self.template_library.get_applicable_templates(analysis, func_info)
        generated_tests = []
        
        # Limit number of templates to use
        selected_templates = applicable_templates[:max_tests]
        
        for template in selected_templates:
            try:
                test = self._generate_test_from_template(template, analysis, func_info)
                if test:
                    generated_tests.append(test)
            except Exception as e:
                print(f"Error generating test with template {template.name}: {e}")
        
        return generated_tests
    
    def _generate_test_from_template(self, template: TestTemplate, 
                                   analysis: CodeAnalysis, 
                                   func_info: Dict[str, Any]) -> Optional[GeneratedTest]:
        """Generate a test from a template"""
        
        # Build context for template
        context = self._build_template_context(analysis, func_info, template)
        
        # Generate test code
        test_code = template.generate_test(context)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(template, func_info, context)
        
        # Estimate coverage
        coverage = self._estimate_coverage(template, func_info)
        
        # Create test object
        test_id = f"{func_info['name']}_{template.template_id}_{len(self.generated_tests)}"
        
        return GeneratedTest(
            test_id=test_id,
            name=f"test_{func_info['name']}_{template.template_id}",
            code=test_code,
            category=template.category,
            complexity=template.complexity,
            target_function=func_info['name'],
            confidence_score=confidence,
            estimated_coverage=coverage,
            generation_time=datetime.now(),
            template_used=template.template_id,
            metadata={
                'function_complexity': func_info['complexity'],
                'parameter_count': len(func_info['parameters']),
                'has_async': func_info['is_async']
            }
        )
    
    def _build_template_context(self, analysis: CodeAnalysis, 
                              func_info: Dict[str, Any], 
                              template: TestTemplate) -> Dict[str, Any]:
        """Build context dictionary for template substitution"""
        context = {
            'function_name': func_info['name'],
            'function_call': self._generate_function_call(func_info),
            'setup_variables': self._generate_setup_variables(func_info),
            'additional_assertions': self._generate_assertions(func_info),
        }
        
        # Template-specific context
        if template.template_id == 'exception_test':
            context.update({
                'exception_type': 'ValueError',  # Default, could be smarter
                'function_call_with_invalid_input': self._generate_invalid_call(func_info),
                'additional_exception_tests': '# Add more exception scenarios'
            })
        
        elif template.template_id == 'performance_test':
            context.update({
                'iterations': 1000,
                'max_time_seconds': 1.0
            })
        
        elif template.template_id == 'mock_test':
            deps = [imp for imp in analysis.imports if '.' in imp]
            if deps:
                dep = deps[0].split('.')[0]
                context.update({
                    'dependency_name': dep,
                    'method_name': 'some_method',
                    'return_value': 'mock_return',
                    'expected_result': 'expected_value'
                })
        
        elif template.template_id == 'parametrized_test':
            context.update({
                'parameter_names': ', '.join(p['name'] for p in func_info['parameters'][:2]),
                'test_cases': self._generate_test_cases(func_info),
                'assertion_condition': 'result is not None'
            })
        
        return context
    
    def _generate_function_call(self, func_info: Dict[str, Any]) -> str:
        """Generate a function call with appropriate parameters"""
        params = func_info['parameters']
        
        if not params:
            return f"{func_info['name']}()"
        
        # Generate simple parameter values
        param_values = []
        for param in params:
            annotation = param.get('annotation', '')
            
            if 'str' in annotation:
                param_values.append(f"'{param['name']}_test'")
            elif 'int' in annotation:
                param_values.append('42')
            elif 'bool' in annotation:
                param_values.append('True')
            elif 'list' in annotation:
                param_values.append('[1, 2, 3]')
            else:
                param_values.append(f"test_{param['name']}")
        
        return f"{func_info['name']}({', '.join(param_values)})"
    
    def _generate_setup_variables(self, func_info: Dict[str, Any]) -> str:
        """Generate setup variables for the test"""
        if not func_info['parameters']:
            return "# No setup needed"
        
        setup_lines = []
        for param in func_info['parameters']:
            setup_lines.append(f"test_{param['name']} = 'test_value'")
        
        return '\n    '.join(setup_lines)
    
    def _generate_assertions(self, func_info: Dict[str, Any]) -> str:
        """Generate additional assertions based on function characteristics"""
        assertions = []
        
        if func_info['returns']:
            assertions.append(f"assert isinstance(result, {func_info['returns']})")
        
        if 'list' in func_info.get('returns', ''):
            assertions.append("assert len(result) >= 0")
        
        if 'str' in func_info.get('returns', ''):
            assertions.append("assert len(result) >= 0")
        
        return '\n    '.join(assertions) if assertions else "# Additional assertions as needed"
    
    def _generate_invalid_call(self, func_info: Dict[str, Any]) -> str:
        """Generate function call with invalid parameters"""
        if not func_info['parameters']:
            return f"{func_info['name']}()"
        
        # Use None or invalid values
        invalid_params = ['None'] * len(func_info['parameters'])
        return f"{func_info['name']}({', '.join(invalid_params)})"
    
    def _generate_test_cases(self, func_info: Dict[str, Any]) -> str:
        """Generate test cases for parametrized tests"""
        if len(func_info['parameters']) < 2:
            return "(1, 2, 3),\n    (4, 5, 6)"
        
        # Generate combinations based on parameter types
        cases = []
        for i in range(3):  # Generate 3 test cases
            case_values = []
            for param in func_info['parameters'][:2]:  # Limit to first 2 params
                annotation = param.get('annotation', '')
                
                if 'str' in annotation:
                    case_values.append(f"'test{i}'")
                elif 'int' in annotation:
                    case_values.append(str(i + 1))
                else:
                    case_values.append(f"value{i}")
            
            cases.append(f"({', '.join(case_values)})")
        
        return ',\n    '.join(cases)
    
    def _calculate_confidence_score(self, template: TestTemplate, 
                                  func_info: Dict[str, Any], 
                                  context: Dict[str, Any]) -> float:
        """Calculate confidence score for generated test"""
        base_score = 0.7  # Base confidence
        
        # Adjust based on template applicability
        if template.template_id == 'basic_unit':
            base_score += 0.1
        
        # Adjust based on function complexity
        if func_info['complexity'] < 5:
            base_score += 0.1
        elif func_info['complexity'] > 10:
            base_score -= 0.1
        
        # Adjust based on available context
        if len(context) > 5:
            base_score += 0.05
        
        # Adjust based on function characteristics
        if func_info['docstring']:
            base_score += 0.05
        
        if func_info['has_exception_handling'] and template.template_id == 'exception_test':
            base_score += 0.15
        
        return min(1.0, max(0.0, base_score))
    
    def _estimate_coverage(self, template: TestTemplate, func_info: Dict[str, Any]) -> float:
        """Estimate test coverage for this test"""
        base_coverage = 0.3  # Base coverage estimate
        
        # Adjust based on template type
        coverage_multipliers = {
            'basic_unit': 0.4,
            'exception_test': 0.6,
            'performance_test': 0.3,
            'mock_test': 0.7,
            'parametrized_test': 0.8,
            'async_test': 0.5
        }
        
        multiplier = coverage_multipliers.get(template.template_id, 0.4)
        estimated_coverage = base_coverage + (multiplier * 0.5)
        
        # Adjust for function complexity
        if func_info['complexity'] > 10:
            estimated_coverage *= 0.8  # Harder to achieve full coverage
        
        return min(1.0, estimated_coverage)
    
    def _generate_class_tests(self, analysis: CodeAnalysis, 
                            class_info: Dict[str, Any]) -> List[GeneratedTest]:
        """Generate tests for class initialization and methods"""
        tests = []
        
        # Generate constructor test
        constructor_test = self._generate_constructor_test(class_info)
        if constructor_test:
            tests.append(constructor_test)
        
        # Generate method tests (sample a few methods)
        methods_to_test = class_info['methods'][:3]  # Limit to prevent overwhelming
        for method in methods_to_test:
            method_tests = self._generate_function_tests(analysis, method, 2)  # Max 2 tests per method
            tests.extend(method_tests)
        
        return tests
    
    def _generate_constructor_test(self, class_info: Dict[str, Any]) -> Optional[GeneratedTest]:
        """Generate constructor test for a class"""
        test_code = f'''def test_{class_info['name'].lower()}_initialization():
    """Test {class_info['name']} initialization."""
    instance = {class_info['name']}()
    assert instance is not None
    assert isinstance(instance, {class_info['name']})'''
        
        return GeneratedTest(
            test_id=f"{class_info['name']}_constructor_{len(self.generated_tests)}",
            name=f"test_{class_info['name'].lower()}_initialization",
            code=test_code,
            category=TestCategory.UNIT,
            complexity=TestComplexity.SIMPLE,
            target_function=f"{class_info['name']}.__init__",
            confidence_score=0.8,
            estimated_coverage=0.3,
            generation_time=datetime.now(),
            metadata={'class_name': class_info['name']}
        )
    
    def _generate_integration_tests(self, analysis: CodeAnalysis) -> List[GeneratedTest]:
        """Generate integration tests based on file analysis"""
        tests = []
        
        # If file has multiple functions, create integration test
        if len(analysis.functions) > 2:
            integration_test = self._generate_function_interaction_test(analysis)
            if integration_test:
                tests.append(integration_test)
        
        # If file has external dependencies, create dependency test
        external_deps = [imp for imp in analysis.imports if not imp.startswith(('builtins', 'sys'))]
        if external_deps:
            dependency_test = self._generate_dependency_test(analysis, external_deps[:2])
            if dependency_test:
                tests.append(dependency_test)
        
        return tests
    
    def _generate_function_interaction_test(self, analysis: CodeAnalysis) -> Optional[GeneratedTest]:
        """Generate test for function interactions"""
        functions = analysis.functions[:3]  # Use first 3 functions
        
        test_code = f'''def test_function_interactions():
    """Test interactions between multiple functions."""
    # Test function sequence
    {functions[0]['name']}()
    {functions[1]['name']}()
    
    # Verify interactions work correctly
    assert True  # Placeholder assertion'''
        
        return GeneratedTest(
            test_id=f"integration_interaction_{len(self.generated_tests)}",
            name="test_function_interactions",
            code=test_code,
            category=TestCategory.INTEGRATION,
            complexity=TestComplexity.MODERATE,
            target_function="multiple_functions",
            confidence_score=0.6,
            estimated_coverage=0.4,
            generation_time=datetime.now(),
            metadata={'integration_type': 'function_interaction'}
        )
    
    def _generate_dependency_test(self, analysis: CodeAnalysis, 
                                dependencies: List[str]) -> Optional[GeneratedTest]:
        """Generate test for external dependencies"""
        dep_names = [dep.split('.')[-1] for dep in dependencies]
        
        test_code = f'''def test_external_dependencies():
    """Test external dependency integration."""
    # Verify dependencies are available
    {chr(10).join([f"import {dep}" for dep in dependencies])}
    
    # Test basic dependency usage
    assert True  # Placeholder for dependency tests'''
        
        return GeneratedTest(
            test_id=f"integration_dependencies_{len(self.generated_tests)}",
            name="test_external_dependencies",
            code=test_code,
            category=TestCategory.INTEGRATION,
            complexity=TestComplexity.SIMPLE,
            target_function="dependencies",
            confidence_score=0.7,
            estimated_coverage=0.2,
            generation_time=datetime.now(),
            metadata={'dependencies': dependencies}
        )
    
    def export_tests_to_file(self, output_path: str, tests: List[GeneratedTest]) -> None:
        """Export generated tests to a Python file"""
        # Group tests by category for better organization
        categorized_tests = defaultdict(list)
        for test in tests:
            categorized_tests[test.category].append(test)
        
        with open(output_path, 'w') as f:
            # Write header
            f.write('"""\n')
            f.write(f'Generated Test Suite - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Total Tests: {len(tests)}\n')
            f.write(f'Generated by: Intelligent Test Generator\n')
            f.write('"""\n\n')
            
            # Write imports
            all_imports = set()
            for test in tests:
                # Extract imports from test code
                imports = re.findall(r'^import\s+\w+|^from\s+\w+\s+import', test.code, re.MULTILINE)
                all_imports.update(imports)
            
            all_imports.add('import pytest')
            for imp in sorted(all_imports):
                f.write(f'{imp}\n')
            
            f.write('\n\n')
            
            # Write tests by category
            for category, category_tests in categorized_tests.items():
                f.write(f'# {category.value.upper()} TESTS\n')
                f.write(f'# Generated {len(category_tests)} {category.value} tests\n\n')
                
                for test in category_tests:
                    f.write(f'# Test ID: {test.test_id}\n')
                    f.write(f'# Confidence: {test.confidence_score:.2f}\n')
                    f.write(f'# Coverage: {test.estimated_coverage:.2f}\n')
                    f.write(f'{test.code}\n\n\n')
        
        print(f'Generated test suite exported to: {output_path}')


# Testing framework
class IntelligentTestGeneratorFramework:
    """Testing framework for the intelligent test generator"""
    
    def test_code_analysis(self) -> bool:
        """Test code analysis functionality"""
        try:
            analyzer = CodeAnalyzer()
            
            # Test with sample code
            test_code = '''
def sample_function(x: int, y: str) -> bool:
    """Sample function for testing."""
    if x > 0:
        return True
    return False

class SampleClass:
    def method(self):
        pass
'''
            
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_path = f.name
            
            try:
                analysis = analyzer.analyze_file(temp_path)
                
                assert len(analysis.functions) > 0
                assert len(analysis.classes) > 0
                assert analysis.complexity_score > 0
                
                return True
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"Code analysis test failed: {e}")
            return False
    
    def test_template_library(self) -> bool:
        """Test template library functionality"""
        try:
            library = TestTemplateLibrary()
            
            # Test template retrieval
            assert len(library.templates) > 0
            assert 'basic_unit' in library.templates
            
            # Test template generation
            template = library.templates['basic_unit']
            context = {
                'function_name': 'test_func',
                'function_call': 'test_func()',
                'setup_variables': 'x = 1',
                'additional_assertions': 'assert True'
            }
            
            generated_code = template.generate_test(context)
            assert 'test_func' in generated_code
            assert 'pytest' in generated_code
            
            return True
        except Exception as e:
            print(f"Template library test failed: {e}")
            return False
    
    def test_test_generation(self) -> bool:
        """Test complete test generation process"""
        try:
            generator = IntelligentTestGenerator()
            
            # Create test file
            test_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_path = f.name
            
            try:
                # Generate tests
                tests = generator.generate_tests_for_file(temp_path)
                
                assert len(tests) > 0
                assert all(isinstance(test, GeneratedTest) for test in tests)
                assert all(test.confidence_score > 0 for test in tests)
                
                return True
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"Test generation test failed: {e}")
            return False
    
    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all test generator tests"""
        tests = [
            'test_code_analysis',
            'test_template_library',
            'test_test_generation'
        ]
        
        results = {}
        for test_name in tests:
            try:
                result = getattr(self, test_name)()
                results[test_name] = result
                print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: FAILED - {e}")
        
        return results


# Main execution
if __name__ == "__main__":
    import sys
    
    print("🤖 Intelligent Test Generator")
    
    # Run tests
    framework = IntelligentTestGeneratorFramework()
    results = framework.run_comprehensive_tests()
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All intelligent test generator tests passed!")
        
        # Demo test generation
        if len(sys.argv) > 1:
            target_file = sys.argv[1]
            print(f"\n🚀 Generating tests for: {target_file}")
            
            generator = IntelligentTestGenerator()
            generated_tests = generator.generate_tests_for_file(target_file)
            
            if generated_tests:
                output_file = f"generated_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                generator.export_tests_to_file(output_file, generated_tests)
                
                print(f"\n📈 Test Generation Complete:")
                print(f"  Tests generated: {len(generated_tests)}")
                print(f"  Average confidence: {generator.generation_metrics['avg_confidence']:.2f}")
                print(f"  Output file: {output_file}")
        else:
            print("\n💡 Usage: python intelligent_test_generator.py <target_file.py>")
    else:
        print("❌ Some tests failed. Check the output above.")