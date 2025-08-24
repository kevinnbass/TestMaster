"""
Multi-Modal Test Engine
Cross-language test generation and execution across Python, JavaScript, TypeScript, and more.
"""

import os
import json
import subprocess
import ast
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from enum import Enum
import tempfile
import shutil


class SupportedLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    RUST = "rust"
    GO = "go"


class TestFramework(Enum):
    """Supported test frameworks"""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    NUNIT = "nunit"
    CARGO_TEST = "cargo_test"
    GO_TEST = "go_test"


@dataclass
class LanguageConfig:
    """Configuration for a programming language"""
    language: SupportedLanguage
    file_extensions: List[str]
    test_frameworks: List[TestFramework]
    execution_command: str
    test_file_patterns: List[str]
    import_patterns: Dict[str, str]
    assertion_patterns: Dict[str, str]
    setup_teardown_patterns: Dict[str, str]


@dataclass
class CrossLanguageTestCase:
    """Test case that can be generated in multiple languages"""
    test_id: str
    name: str
    description: str
    test_logic: Dict[str, Any]  # Language-agnostic test logic
    expected_behavior: str
    input_parameters: Dict[str, Any]
    expected_output: Any
    test_category: str
    complexity_level: int
    
    def to_language_specific(self, language: SupportedLanguage, 
                           config: LanguageConfig) -> str:
        """Convert to language-specific test code"""
        generators = {
            SupportedLanguage.PYTHON: self._to_python,
            SupportedLanguage.JAVASCRIPT: self._to_javascript,
            SupportedLanguage.TYPESCRIPT: self._to_typescript,
            SupportedLanguage.JAVA: self._to_java,
            SupportedLanguage.CSHARP: self._to_csharp,
            SupportedLanguage.RUST: self._to_rust,
            SupportedLanguage.GO: self._to_go
        }
        
        if language in generators:
            return generators[language](config)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _to_python(self, config: LanguageConfig) -> str:
        """Generate Python test code"""
        imports = "import pytest\nimport unittest\nfrom unittest.mock import Mock, patch\n\n"
        
        test_code = f'''def test_{self.name.lower().replace(' ', '_')}():
    """
    {self.description}
    Expected: {self.expected_behavior}
    """
    # Arrange
    {self._format_python_setup()}
    
    # Act
    result = {self._format_python_action()}
    
    # Assert
    {self._format_python_assertions()}
'''
        
        return imports + test_code
    
    def _to_javascript(self, config: LanguageConfig) -> str:
        """Generate JavaScript test code"""
        framework = "jest"  # Default to Jest
        
        imports = "const { describe, test, expect, beforeEach, afterEach } = require('@jest/globals');\n\n"
        
        test_code = f'''describe('{self.name}', () => {{
    test('{self.description}', () => {{
        // Arrange
        {self._format_js_setup()}
        
        // Act
        const result = {self._format_js_action()};
        
        // Assert
        {self._format_js_assertions()}
    }});
}});
'''
        
        return imports + test_code
    
    def _to_typescript(self, config: LanguageConfig) -> str:
        """Generate TypeScript test code"""
        imports = '''import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';

'''
        
        test_code = f'''describe('{self.name}', () => {{
    test('{self.description}', () => {{
        // Arrange
        {self._format_ts_setup()}
        
        // Act
        const result: any = {self._format_ts_action()};
        
        // Assert
        {self._format_ts_assertions()}
    }});
}});
'''
        
        return imports + test_code
    
    def _to_java(self, config: LanguageConfig) -> str:
        """Generate Java test code"""
        imports = '''import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;

'''
        
        class_name = f"{self.name.replace(' ', '')}Test"
        
        test_code = f'''public class {class_name} {{
    
    @BeforeEach
    void setUp() {{
        {self._format_java_setup()}
    }}
    
    @Test
    void test{self.name.replace(' ', '')}() {{
        // Arrange
        {self._format_java_arrange()}
        
        // Act
        var result = {self._format_java_action()};
        
        // Assert
        {self._format_java_assertions()}
    }}
    
    @AfterEach
    void tearDown() {{
        // Cleanup if needed
    }}
}}
'''
        
        return imports + test_code
    
    def _to_csharp(self, config: LanguageConfig) -> str:
        """Generate C# test code"""
        imports = '''using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

'''
        
        class_name = f"{self.name.replace(' ', '')}Test"
        
        test_code = f'''[TestClass]
public class {class_name}
{{
    [TestInitialize]
    public void Setup()
    {{
        {self._format_csharp_setup()}
    }}
    
    [TestMethod]
    public void Test{self.name.replace(' ', '')}()
    {{
        // Arrange
        {self._format_csharp_arrange()}
        
        // Act
        var result = {self._format_csharp_action()};
        
        // Assert
        {self._format_csharp_assertions()}
    }}
    
    [TestCleanup]
    public void Cleanup()
    {{
        // Cleanup if needed
    }}
}}
'''
        
        return imports + test_code
    
    def _to_rust(self, config: LanguageConfig) -> str:
        """Generate Rust test code"""
        test_code = f'''#[cfg(test)]
mod tests {{
    use super::*;
    
    #[test]
    fn test_{self.name.lower().replace(' ', '_')}() {{
        // Arrange
        {self._format_rust_setup()}
        
        // Act
        let result = {self._format_rust_action()};
        
        // Assert
        {self._format_rust_assertions()}
    }}
}}
'''
        
        return test_code
    
    def _to_go(self, config: LanguageConfig) -> str:
        """Generate Go test code"""
        imports = '''package main

import (
    "testing"
)

'''
        
        test_code = f'''func Test{self.name.replace(' ', '')}(t *testing.T) {{
    // Arrange
    {self._format_go_setup()}
    
    // Act
    result := {self._format_go_action()}
    
    // Assert
    {self._format_go_assertions()}
}}
'''
        
        return imports + test_code
    
    # Language-specific formatting methods
    def _format_python_setup(self) -> str:
        setup_lines = []
        for param, value in self.input_parameters.items():
            if isinstance(value, str):
                setup_lines.append(f'{param} = "{value}"')
            else:
                setup_lines.append(f'{param} = {value}')
        return '\n    '.join(setup_lines) if setup_lines else '# No setup needed'
    
    def _format_python_action(self) -> str:
        # Simple function call based on test logic
        func_name = self.test_logic.get('function_name', 'target_function')
        params = ', '.join(self.input_parameters.keys())
        return f'{func_name}({params})' if params else f'{func_name}()'
    
    def _format_python_assertions(self) -> str:
        assertions = []
        assertions.append('assert result is not None')
        
        if self.expected_output is not None:
            if isinstance(self.expected_output, str):
                assertions.append(f'assert result == "{self.expected_output}"')
            else:
                assertions.append(f'assert result == {self.expected_output}')
        
        return '\n    '.join(assertions)
    
    def _format_js_setup(self) -> str:
        setup_lines = []
        for param, value in self.input_parameters.items():
            if isinstance(value, str):
                setup_lines.append(f"const {param} = '{value}';")
            else:
                setup_lines.append(f'const {param} = {json.dumps(value)};')
        return '\n        '.join(setup_lines) if setup_lines else '// No setup needed'
    
    def _format_js_action(self) -> str:
        func_name = self.test_logic.get('function_name', 'targetFunction')
        params = ', '.join(self.input_parameters.keys())
        return f'{func_name}({params})' if params else f'{func_name}()'
    
    def _format_js_assertions(self) -> str:
        assertions = []
        assertions.append('expect(result).toBeDefined();')
        
        if self.expected_output is not None:
            assertions.append(f'expect(result).toBe({json.dumps(self.expected_output)});')
        
        return '\n        '.join(assertions)
    
    def _format_ts_setup(self) -> str:
        # Similar to JS but with type annotations where beneficial
        return self._format_js_setup()
    
    def _format_ts_action(self) -> str:
        return self._format_js_action()
    
    def _format_ts_assertions(self) -> str:
        return self._format_js_assertions()
    
    def _format_java_setup(self) -> str:
        return '// Setup code here'
    
    def _format_java_arrange(self) -> str:
        arrange_lines = []
        for param, value in self.input_parameters.items():
            if isinstance(value, str):
                arrange_lines.append(f'String {param} = "{value}";')
            elif isinstance(value, int):
                arrange_lines.append(f'int {param} = {value};')
            elif isinstance(value, bool):
                arrange_lines.append(f'boolean {param} = {str(value).lower()};')
            else:
                arrange_lines.append(f'var {param} = {value};')
        return '\n        '.join(arrange_lines) if arrange_lines else '// No arrangement needed'
    
    def _format_java_action(self) -> str:
        func_name = self.test_logic.get('function_name', 'targetFunction')
        params = ', '.join(self.input_parameters.keys())
        return f'{func_name}({params})' if params else f'{func_name}()'
    
    def _format_java_assertions(self) -> str:
        assertions = []
        assertions.append('assertNotNull(result);')
        
        if self.expected_output is not None:
            if isinstance(self.expected_output, str):
                assertions.append(f'assertEquals("{self.expected_output}", result);')
            else:
                assertions.append(f'assertEquals({self.expected_output}, result);')
        
        return '\n        '.join(assertions)
    
    def _format_csharp_setup(self) -> str:
        return '// Setup code here'
    
    def _format_csharp_arrange(self) -> str:
        arrange_lines = []
        for param, value in self.input_parameters.items():
            if isinstance(value, str):
                arrange_lines.append(f'var {param} = "{value}";')
            else:
                arrange_lines.append(f'var {param} = {json.dumps(value)};')
        return '\n        '.join(arrange_lines) if arrange_lines else '// No arrangement needed'
    
    def _format_csharp_action(self) -> str:
        func_name = self.test_logic.get('function_name', 'TargetFunction')
        params = ', '.join(self.input_parameters.keys())
        return f'{func_name}({params})' if params else f'{func_name}()'
    
    def _format_csharp_assertions(self) -> str:
        assertions = []
        assertions.append('Assert.IsNotNull(result);')
        
        if self.expected_output is not None:
            assertions.append(f'Assert.AreEqual({json.dumps(self.expected_output)}, result);')
        
        return '\n        '.join(assertions)
    
    def _format_rust_setup(self) -> str:
        setup_lines = []
        for param, value in self.input_parameters.items():
            if isinstance(value, str):
                setup_lines.append(f'let {param} = "{value}";')
            else:
                setup_lines.append(f'let {param} = {value};')
        return '\n        '.join(setup_lines) if setup_lines else '// No setup needed'
    
    def _format_rust_action(self) -> str:
        func_name = self.test_logic.get('function_name', 'target_function')
        params = ', '.join(self.input_parameters.keys())
        return f'{func_name}({params})' if params else f'{func_name}()'
    
    def _format_rust_assertions(self) -> str:
        assertions = []
        
        if self.expected_output is not None:
            if isinstance(self.expected_output, str):
                assertions.append(f'assert_eq!(result, "{self.expected_output}");')
            else:
                assertions.append(f'assert_eq!(result, {self.expected_output});')
        else:
            assertions.append('assert!(result.is_some());')
        
        return '\n        '.join(assertions)
    
    def _format_go_setup(self) -> str:
        setup_lines = []
        for param, value in self.input_parameters.items():
            if isinstance(value, str):
                setup_lines.append(f'{param} := "{value}"')
            else:
                setup_lines.append(f'{param} := {value}')
        return '\n    '.join(setup_lines) if setup_lines else '// No setup needed'
    
    def _format_go_action(self) -> str:
        func_name = self.test_logic.get('function_name', 'TargetFunction')
        params = ', '.join(self.input_parameters.keys())
        return f'{func_name}({params})' if params else f'{func_name}()'
    
    def _format_go_assertions(self) -> str:
        assertions = []
        assertions.append('if result == nil {\n        t.Error("Expected non-nil result")\n    }')
        
        if self.expected_output is not None:
            if isinstance(self.expected_output, str):
                assertions.append(f'if result != "{self.expected_output}" {{\n        t.Errorf("Expected {self.expected_output}, got %v", result)\n    }}')
            else:
                assertions.append(f'if result != {self.expected_output} {{\n        t.Errorf("Expected {self.expected_output}, got %v", result)\n    }}')
        
        return '\n    '.join(assertions)


class LanguageConfigManager:
    """Manager for language-specific configurations"""
    
    def __init__(self):
        self.configs = self._initialize_language_configs()
    
    def _initialize_language_configs(self) -> Dict[SupportedLanguage, LanguageConfig]:
        """Initialize default language configurations"""
        configs = {}
        
        # Python Configuration
        configs[SupportedLanguage.PYTHON] = LanguageConfig(
            language=SupportedLanguage.PYTHON,
            file_extensions=['.py'],
            test_frameworks=[TestFramework.PYTEST, TestFramework.UNITTEST],
            execution_command='python -m pytest',
            test_file_patterns=['test_*.py', '*_test.py'],
            import_patterns={
                'pytest': 'import pytest',
                'unittest': 'import unittest',
                'mock': 'from unittest.mock import Mock, patch'
            },
            assertion_patterns={
                'equal': 'assert {actual} == {expected}',
                'not_none': 'assert {actual} is not None',
                'raises': 'with pytest.raises({exception}): {code}'
            },
            setup_teardown_patterns={
                'setup': '@pytest.fixture\ndef setup(): pass',
                'teardown': 'yield\n# teardown code'
            }
        )
        
        # JavaScript Configuration
        configs[SupportedLanguage.JAVASCRIPT] = LanguageConfig(
            language=SupportedLanguage.JAVASCRIPT,
            file_extensions=['.js'],
            test_frameworks=[TestFramework.JEST, TestFramework.MOCHA],
            execution_command='npm test',
            test_file_patterns=['*.test.js', '*.spec.js'],
            import_patterns={
                'jest': "const { describe, test, expect } = require('@jest/globals');",
                'mocha': "const { describe, it } = require('mocha');"
            },
            assertion_patterns={
                'equal': 'expect({actual}).toBe({expected});',
                'not_null': 'expect({actual}).toBeDefined();',
                'throws': 'expect(() => {code}).toThrow();'
            },
            setup_teardown_patterns={
                'setup': 'beforeEach(() => { /* setup */ });',
                'teardown': 'afterEach(() => { /* teardown */ });'
            }
        )
        
        # TypeScript Configuration
        configs[SupportedLanguage.TYPESCRIPT] = LanguageConfig(
            language=SupportedLanguage.TYPESCRIPT,
            file_extensions=['.ts'],
            test_frameworks=[TestFramework.JEST],
            execution_command='npm run test:ts',
            test_file_patterns=['*.test.ts', '*.spec.ts'],
            import_patterns={
                'jest': "import { describe, test, expect } from '@jest/globals';"
            },
            assertion_patterns={
                'equal': 'expect({actual}).toBe({expected});',
                'not_null': 'expect({actual}).toBeDefined();',
                'throws': 'expect(() => {code}).toThrow();'
            },
            setup_teardown_patterns={
                'setup': 'beforeEach(() => { /* setup */ });',
                'teardown': 'afterEach(() => { /* teardown */ });'
            }
        )
        
        return configs
    
    def get_config(self, language: SupportedLanguage) -> LanguageConfig:
        """Get configuration for a specific language"""
        if language not in self.configs:
            raise ValueError(f"Configuration not found for language: {language}")
        return self.configs[language]
    
    def add_custom_config(self, config: LanguageConfig) -> None:
        """Add custom language configuration"""
        self.configs[config.language] = config


class CrossLanguageTestGenerator:
    """Generator for cross-language test cases"""
    
    def __init__(self):
        self.config_manager = LanguageConfigManager()
        self.test_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> List[CrossLanguageTestCase]:
        """Initialize common test templates"""
        templates = []
        
        # Basic function test template
        templates.append(CrossLanguageTestCase(
            test_id='basic_function_test',
            name='Basic Function Test',
            description='Test basic function functionality',
            test_logic={'function_name': 'target_function', 'type': 'function_call'},
            expected_behavior='Function returns expected result',
            input_parameters={'input_value': 'test'},
            expected_output='processed_test',
            test_category='unit',
            complexity_level=1
        ))
        
        # Null/None handling test
        templates.append(CrossLanguageTestCase(
            test_id='null_handling_test',
            name='Null Handling Test', 
            description='Test handling of null/None values',
            test_logic={'function_name': 'target_function', 'type': 'null_test'},
            expected_behavior='Function handles null input gracefully',
            input_parameters={'input_value': None},
            expected_output=None,
            test_category='edge_case',
            complexity_level=2
        ))
        
        # Exception handling test
        templates.append(CrossLanguageTestCase(
            test_id='exception_test',
            name='Exception Handling Test',
            description='Test exception handling behavior',
            test_logic={'function_name': 'target_function', 'type': 'exception_test'},
            expected_behavior='Function throws expected exception',
            input_parameters={'input_value': 'invalid'},
            expected_output='ValueError',
            test_category='error_handling',
            complexity_level=3
        ))
        
        return templates
    
    def generate_cross_language_tests(self, 
                                    target_languages: List[SupportedLanguage],
                                    test_specifications: List[Dict[str, Any]]) -> Dict[SupportedLanguage, List[str]]:
        """Generate tests for multiple languages based on specifications"""
        
        generated_tests = {lang: [] for lang in target_languages}
        
        for spec in test_specifications:
            # Create cross-language test case from specification
            test_case = self._create_test_case_from_spec(spec)
            
            # Generate for each target language
            for language in target_languages:
                config = self.config_manager.get_config(language)
                test_code = test_case.to_language_specific(language, config)
                generated_tests[language].append(test_code)
        
        return generated_tests
    
    def _create_test_case_from_spec(self, spec: Dict[str, Any]) -> CrossLanguageTestCase:
        """Create test case from specification"""
        return CrossLanguageTestCase(
            test_id=spec.get('test_id', f'test_{hash(str(spec))}'),
            name=spec.get('name', 'Generated Test'),
            description=spec.get('description', 'Auto-generated test case'),
            test_logic=spec.get('test_logic', {'function_name': 'target_function'}),
            expected_behavior=spec.get('expected_behavior', 'Function behaves as expected'),
            input_parameters=spec.get('input_parameters', {}),
            expected_output=spec.get('expected_output', None),
            test_category=spec.get('test_category', 'unit'),
            complexity_level=spec.get('complexity_level', 1)
        )
    
    def generate_equivalent_tests(self, 
                                source_language: SupportedLanguage,
                                source_code: str,
                                target_languages: List[SupportedLanguage]) -> Dict[SupportedLanguage, str]:
        """Generate equivalent tests in target languages from source test"""
        
        # Parse source test to extract test specifications
        test_specs = self._parse_source_test(source_language, source_code)
        
        # Generate equivalent tests
        equivalent_tests = {}
        
        for target_lang in target_languages:
            if target_lang == source_language:
                equivalent_tests[target_lang] = source_code
                continue
            
            config = self.config_manager.get_config(target_lang)
            test_case = self._create_test_case_from_spec(test_specs)
            equivalent_tests[target_lang] = test_case.to_language_specific(target_lang, config)
        
        return equivalent_tests
    
    def _parse_source_test(self, language: SupportedLanguage, source_code: str) -> Dict[str, Any]:
        """Parse source test code to extract specifications"""
        # This is a simplified parser - in practice would need more sophisticated parsing
        
        spec = {
            'name': self._extract_test_name(source_code),
            'description': self._extract_test_description(source_code),
            'test_logic': {'function_name': self._extract_function_name(source_code)},
            'input_parameters': self._extract_input_parameters(source_code),
            'expected_output': self._extract_expected_output(source_code),
            'test_category': 'unit',
            'complexity_level': 1
        }
        
        return spec
    
    def _extract_test_name(self, source_code: str) -> str:
        """Extract test name from source code"""
        # Look for test function names
        patterns = [
            r'def\s+(test_\w+)',           # Python
            r'test\([\'"]([^\'"]+)[\'"]',  # JavaScript/TypeScript
            r'void\s+(test\w+)',          # Java/C#
            r'fn\s+(test_\w+)',           # Rust
            r'func\s+(Test\w+)'           # Go
        ]
        
        for pattern in patterns:
            match = re.search(pattern, source_code)
            if match:
                return match.group(1).replace('_', ' ').title()
        
        return 'Extracted Test'
    
    def _extract_test_description(self, source_code: str) -> str:
        """Extract test description from comments/docstrings"""
        # Look for docstrings, comments
        patterns = [
            r'"""([^"]+)"""',      # Python docstring
            r'//\s*(.+)',          # Single line comment
            r'/\*\s*([^*]+)\*/',   # Multi-line comment
        ]
        
        for pattern in patterns:
            match = re.search(pattern, source_code)
            if match:
                return match.group(1).strip()
        
        return 'Auto-extracted test description'
    
    def _extract_function_name(self, source_code: str) -> str:
        """Extract target function name being tested"""
        # Simple heuristic - look for function calls in test
        patterns = [
            r'(\w+)\(',  # Function call pattern
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, source_code)
            # Filter out common test keywords
            excluded = {'assert', 'expect', 'test', 'describe', 'it', 'should', 'assertEquals'}
            candidates = [match for match in matches if match not in excluded and not match.startswith('test')]
            
            if candidates:
                return candidates[0]
        
        return 'target_function'
    
    def _extract_input_parameters(self, source_code: str) -> Dict[str, Any]:
        """Extract input parameters from test code"""
        # Simple extraction - look for variable assignments
        params = {}
        
        # Look for variable assignments
        patterns = [
            r'(\w+)\s*=\s*["\']([^"\']+)["\']',  # String assignments
            r'(\w+)\s*=\s*(\d+)',               # Number assignments
            r'(\w+)\s*=\s*(True|False|true|false)',  # Boolean assignments
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, source_code)
            for var_name, value in matches:
                if var_name not in {'result', 'expected', 'actual'}:
                    # Try to convert value to appropriate type
                    if value.isdigit():
                        params[var_name] = int(value)
                    elif value.lower() in ('true', 'false'):
                        params[var_name] = value.lower() == 'true'
                    else:
                        params[var_name] = value
        
        return params
    
    def _extract_expected_output(self, source_code: str) -> Any:
        """Extract expected output from assertions"""
        # Look for assertion patterns
        patterns = [
            r'assert\s+\w+\s*==\s*["\']([^"\']+)["\']',  # String equality
            r'assert\s+\w+\s*==\s*(\d+)',               # Number equality
            r'expect\([^)]+\)\.toBe\(["\']([^"\']+)["\']', # Jest string
            r'expect\([^)]+\)\.toBe\((\d+)\)',          # Jest number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, source_code)
            if match:
                value = match.group(1)
                # Try to convert to appropriate type
                if value.isdigit():
                    return int(value)
                return value
        
        return None


class MultiLanguageTestExecutor:
    """Executor for running tests across multiple languages"""
    
    def __init__(self):
        self.config_manager = LanguageConfigManager()
        self.execution_results = {}
    
    def execute_tests(self, 
                     test_files: Dict[SupportedLanguage, List[str]], 
                     working_directory: Optional[str] = None) -> Dict[SupportedLanguage, Dict[str, Any]]:
        """Execute tests for multiple languages"""
        
        if working_directory is None:
            working_directory = tempfile.mkdtemp()
        
        results = {}
        
        for language, test_codes in test_files.items():
            print(f"Executing {language.value} tests...")
            
            language_results = []
            config = self.config_manager.get_config(language)
            
            for i, test_code in enumerate(test_codes):
                # Create temporary test file
                test_file_path = self._create_test_file(language, test_code, working_directory, i)
                
                # Execute test
                result = self._execute_single_test(language, test_file_path, config)
                language_results.append(result)
                
                # Cleanup
                if os.path.exists(test_file_path):
                    os.remove(test_file_path)
            
            results[language] = {
                'total_tests': len(test_codes),
                'results': language_results,
                'summary': self._summarize_results(language_results)
            }
        
        return results
    
    def _create_test_file(self, language: SupportedLanguage, test_code: str, 
                         working_dir: str, index: int) -> str:
        """Create temporary test file for execution"""
        config = self.config_manager.get_config(language)
        extension = config.file_extensions[0]
        
        filename = f"test_{language.value}_{index}{extension}"
        file_path = os.path.join(working_dir, filename)
        
        with open(file_path, 'w') as f:
            f.write(test_code)
        
        return file_path
    
    def _execute_single_test(self, language: SupportedLanguage, 
                           test_file_path: str, config: LanguageConfig) -> Dict[str, Any]:
        """Execute a single test file"""
        result = {
            'file_path': test_file_path,
            'language': language.value,
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0
        }
        
        try:
            start_time = datetime.now()
            
            # Prepare command based on language
            command = self._prepare_execution_command(language, test_file_path, config)
            
            # Execute command
            process = subprocess.run(
                command,
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=os.path.dirname(test_file_path)
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result.update({
                'success': process.returncode == 0,
                'output': process.stdout,
                'error': process.stderr,
                'execution_time': execution_time,
                'return_code': process.returncode
            })
            
        except subprocess.TimeoutExpired:
            result['error'] = 'Test execution timed out'
        except Exception as e:
            result['error'] = f'Execution failed: {str(e)}'
        
        return result
    
    def _prepare_execution_command(self, language: SupportedLanguage, 
                                 test_file_path: str, config: LanguageConfig) -> str:
        """Prepare execution command for specific language"""
        
        commands = {
            SupportedLanguage.PYTHON: f'python -m pytest "{test_file_path}" -v',
            SupportedLanguage.JAVASCRIPT: f'node "{test_file_path}"',  # Simplified
            SupportedLanguage.TYPESCRIPT: f'npx ts-node "{test_file_path}"',  # Simplified
            SupportedLanguage.JAVA: f'javac "{test_file_path}" && java -cp . {os.path.basename(test_file_path)[:-5]}Test',
            SupportedLanguage.CSHARP: f'dotnet test "{test_file_path}"',
            SupportedLanguage.RUST: f'rustc --test "{test_file_path}" && ./test',
            SupportedLanguage.GO: f'go test "{test_file_path}"'
        }
        
        return commands.get(language, config.execution_command.replace('{file}', test_file_path))
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize test execution results"""
        if not results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'success_rate': 0.0}
        
        total = len(results)
        passed = sum(1 for r in results if r['success'])
        failed = total - passed
        
        avg_execution_time = sum(r['execution_time'] for r in results) / total
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed / total if total > 0 else 0.0,
            'avg_execution_time': avg_execution_time,
            'errors': [r['error'] for r in results if r['error']]
        }


class MultiModalTestEngine:
    """Main engine for multi-modal cross-language testing"""
    
    def __init__(self):
        self.generator = CrossLanguageTestGenerator()
        self.executor = MultiLanguageTestExecutor()
        self.supported_languages = list(SupportedLanguage)
    
    def generate_multi_language_test_suite(self, 
                                         specifications: List[Dict[str, Any]],
                                         target_languages: Optional[List[SupportedLanguage]] = None) -> Dict[str, Any]:
        """Generate comprehensive multi-language test suite"""
        
        if target_languages is None:
            target_languages = [SupportedLanguage.PYTHON, SupportedLanguage.JAVASCRIPT, SupportedLanguage.TYPESCRIPT]
        
        # Generate tests
        generated_tests = self.generator.generate_cross_language_tests(target_languages, specifications)
        
        # Execute tests (optional - can be skipped for generation-only)
        execution_results = {}
        
        # Package results
        return {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'target_languages': [lang.value for lang in target_languages],
                'total_specifications': len(specifications),
                'total_generated_tests': sum(len(tests) for tests in generated_tests.values())
            },
            'generated_tests': {lang.value: tests for lang, tests in generated_tests.items()},
            'execution_results': execution_results,
            'language_coverage': self._analyze_language_coverage(generated_tests)
        }
    
    def port_tests_across_languages(self, 
                                   source_language: SupportedLanguage,
                                   source_tests: List[str],
                                   target_languages: List[SupportedLanguage]) -> Dict[str, List[str]]:
        """Port existing tests from one language to others"""
        
        ported_tests = {lang.value: [] for lang in target_languages}
        
        for source_test in source_tests:
            equivalent_tests = self.generator.generate_equivalent_tests(
                source_language, source_test, target_languages
            )
            
            for lang, test_code in equivalent_tests.items():
                ported_tests[lang.value].append(test_code)
        
        return ported_tests
    
    def _analyze_language_coverage(self, generated_tests: Dict[SupportedLanguage, List[str]]) -> Dict[str, Any]:
        """Analyze test coverage across languages"""
        
        coverage_analysis = {}
        
        for language, tests in generated_tests.items():
            coverage_analysis[language.value] = {
                'test_count': len(tests),
                'avg_test_length': sum(len(test) for test in tests) / len(tests) if tests else 0,
                'framework_compatibility': self._check_framework_compatibility(language, tests)
            }
        
        return coverage_analysis
    
    def _check_framework_compatibility(self, language: SupportedLanguage, 
                                     tests: List[str]) -> Dict[str, bool]:
        """Check compatibility with different test frameworks"""
        
        compatibility = {}
        config = self.generator.config_manager.get_config(language)
        
        for framework in config.test_frameworks:
            # Simple check based on framework patterns
            compatible = any(self._test_uses_framework(test, framework) for test in tests)
            compatibility[framework.value] = compatible
        
        return compatibility
    
    def _test_uses_framework(self, test_code: str, framework: TestFramework) -> bool:
        """Check if test code uses specific framework"""
        
        framework_indicators = {
            TestFramework.PYTEST: ['pytest', '@pytest', 'def test_'],
            TestFramework.UNITTEST: ['unittest', 'TestCase', 'setUp'],
            TestFramework.JEST: ['jest', 'describe(', 'test(', 'expect('],
            TestFramework.MOCHA: ['mocha', 'describe(', 'it('],
            TestFramework.JUNIT: ['@Test', 'junit', 'assertEquals'],
            TestFramework.NUNIT: ['[Test]', 'NUnit', 'Assert.'],
            TestFramework.CARGO_TEST: ['#[test]', 'cargo test'],
            TestFramework.GO_TEST: ['func Test', 'testing.T']
        }
        
        indicators = framework_indicators.get(framework, [])
        return any(indicator in test_code for indicator in indicators)
    
    def export_multi_language_suite(self, test_suite: Dict[str, Any], 
                                   output_directory: str) -> None:
        """Export multi-language test suite to files"""
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export tests for each language
        for language, tests in test_suite['generated_tests'].items():
            lang_dir = output_path / language
            lang_dir.mkdir(exist_ok=True)
            
            for i, test_code in enumerate(tests):
                # Determine file extension
                lang_enum = SupportedLanguage(language)
                config = self.generator.config_manager.get_config(lang_enum)
                extension = config.file_extensions[0]
                
                # Write test file
                test_file_path = lang_dir / f"test_{i:03d}{extension}"
                with open(test_file_path, 'w') as f:
                    f.write(test_code)
        
        # Export metadata
        metadata_path = output_path / "test_suite_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(test_suite, f, indent=2, default=str)
        
        print(f"Multi-language test suite exported to: {output_directory}")


# Testing framework
class MultiModalTestEngineFramework:
    """Testing framework for multi-modal test engine"""
    
    def test_cross_language_generation(self) -> bool:
        """Test cross-language test generation"""
        try:
            engine = MultiModalTestEngine()
            
            # Create test specifications
            specs = [
                {
                    'name': 'Basic Test',
                    'description': 'Test basic functionality',
                    'test_logic': {'function_name': 'add_numbers'},
                    'input_parameters': {'a': 5, 'b': 3},
                    'expected_output': 8,
                    'test_category': 'unit'
                }
            ]
            
            # Generate tests for multiple languages
            target_languages = [SupportedLanguage.PYTHON, SupportedLanguage.JAVASCRIPT]
            test_suite = engine.generate_multi_language_test_suite(specs, target_languages)
            
            assert 'generated_tests' in test_suite
            assert len(test_suite['generated_tests']) == 2
            assert 'python' in test_suite['generated_tests']
            assert 'javascript' in test_suite['generated_tests']
            
            return True
        except Exception as e:
            print(f"Cross-language generation test failed: {e}")
            return False
    
    def test_test_porting(self) -> bool:
        """Test porting tests between languages"""
        try:
            engine = MultiModalTestEngine()
            
            # Python source test
            python_test = '''
def test_add_function():
    """Test the add function."""
    result = add(2, 3)
    assert result == 5
'''
            
            # Port to other languages
            ported = engine.port_tests_across_languages(
                SupportedLanguage.PYTHON,
                [python_test],
                [SupportedLanguage.JAVASCRIPT, SupportedLanguage.TYPESCRIPT]
            )
            
            assert 'javascript' in ported
            assert 'typescript' in ported
            assert len(ported['javascript']) > 0
            assert len(ported['typescript']) > 0
            
            return True
        except Exception as e:
            print(f"Test porting test failed: {e}")
            return False
    
    def test_language_config(self) -> bool:
        """Test language configuration management"""
        try:
            config_manager = LanguageConfigManager()
            
            # Test getting configurations
            python_config = config_manager.get_config(SupportedLanguage.PYTHON)
            js_config = config_manager.get_config(SupportedLanguage.JAVASCRIPT)
            
            assert python_config.language == SupportedLanguage.PYTHON
            assert '.py' in python_config.file_extensions
            assert js_config.language == SupportedLanguage.JAVASCRIPT
            assert '.js' in js_config.file_extensions
            
            return True
        except Exception as e:
            print(f"Language config test failed: {e}")
            return False
    
    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all multi-modal test engine tests"""
        tests = [
            'test_cross_language_generation',
            'test_test_porting',
            'test_language_config'
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
    print("🌐 Multi-Modal Test Engine")
    
    # Run tests
    test_framework = MultiModalTestEngineFramework()
    results = test_framework.run_comprehensive_tests()
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All multi-modal test engine tests passed!")
        
        # Demonstrate multi-language test generation
        print("\n🚀 Running multi-language test generation demo...")
        
        engine = MultiModalTestEngine()
        
        # Create sample test specifications
        test_specs = [
            {
                'name': 'String Processing',
                'description': 'Test string processing functionality',
                'test_logic': {'function_name': 'process_string'},
                'input_parameters': {'input_str': 'hello world'},
                'expected_output': 'HELLO WORLD',
                'test_category': 'unit'
            },
            {
                'name': 'Number Calculation',
                'description': 'Test numeric calculation',
                'test_logic': {'function_name': 'calculate'},
                'input_parameters': {'x': 10, 'y': 5},
                'expected_output': 15,
                'test_category': 'unit'
            }
        ]
        
        # Generate tests for multiple languages
        target_languages = [
            SupportedLanguage.PYTHON,
            SupportedLanguage.JAVASCRIPT,
            SupportedLanguage.TYPESCRIPT
        ]
        
        test_suite = engine.generate_multi_language_test_suite(test_specs, target_languages)
        
        # Export test suite
        output_dir = f"multi_language_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        engine.export_multi_language_suite(test_suite, output_dir)
        
        print(f"\n📈 Multi-Language Test Generation Complete:")
        print(f"  Languages: {len(target_languages)}")
        print(f"  Test specifications: {len(test_specs)}")
        print(f"  Total generated tests: {test_suite['metadata']['total_generated_tests']}")
        print(f"  Output directory: {output_dir}")
        
        # Show sample generated test
        python_tests = test_suite['generated_tests'].get('python', [])
        if python_tests:
            print(f"\n📝 Sample Python Test:")
            print("```python")
            print(python_tests[0][:300] + "..." if len(python_tests[0]) > 300 else python_tests[0])
            print("```")
            
    else:
        print("❌ Some tests failed. Check the output above.")