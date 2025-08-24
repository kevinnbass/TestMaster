"""
Codebase-Agnostic Test Output System

Generates test outputs in multiple formats and structures, supporting any programming
language and testing framework. Provides unified output generation with comprehensive
documentation, metrics, and framework-specific adaptations.

Adapted from Agency Swarm's output generation patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Union
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
import yaml

from core.framework_abstraction import (
    UniversalTestSuite, UniversalTestCase, UniversalTest,
    TestAssertion, TestMetadata
)


class OutputFormat(Enum):
    """Supported output formats."""
    # Programming languages
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    
    # Universal formats
    UNIVERSAL = "universal"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    
    # Documentation formats
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    
    # Configuration formats
    CONFIG = "config"
    MAKEFILE = "makefile"
    DOCKERFILE = "dockerfile"


@dataclass
class TestOutputBundle:
    """Bundle of generated test outputs."""
    name: str
    format: OutputFormat
    
    # Generated files
    test_files: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    documentation_files: List[str] = field(default_factory=list)
    
    # Content
    main_content: str = ""
    additional_content: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    generation_timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'format': self.format.value,
            'files': {
                'test_files': self.test_files,
                'config_files': self.config_files,
                'documentation_files': self.documentation_files
            },
            'content': {
                'main_content_length': len(self.main_content),
                'additional_content_count': len(self.additional_content)
            },
            'metadata': {
                'generation_timestamp': self.generation_timestamp.isoformat(),
                'metrics': self.metrics
            }
        }


@dataclass
class OutputSystemConfig:
    """Configuration for output system."""
    # Output settings
    output_directory: str = "./generated_tests"
    output_formats: List[str] = field(default_factory=lambda: ["python", "universal"])
    
    # File organization
    organize_by_framework: bool = True
    organize_by_language: bool = True
    create_separate_directories: bool = True
    
    # Content generation
    include_documentation: bool = True
    include_metrics: bool = True
    include_configuration: bool = True
    include_examples: bool = True
    
    # Documentation settings
    documentation_format: str = "markdown"
    include_test_descriptions: bool = True
    include_coverage_reports: bool = True
    
    # File naming
    test_file_prefix: str = "test_"
    test_file_suffix: str = ""
    use_timestamp_suffix: bool = False
    
    # Quality settings
    validate_syntax: bool = True
    format_code: bool = True
    optimize_imports: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'output_settings': {
                'output_directory': self.output_directory,
                'output_formats': self.output_formats
            },
            'file_organization': {
                'organize_by_framework': self.organize_by_framework,
                'organize_by_language': self.organize_by_language,
                'create_separate_directories': self.create_separate_directories
            },
            'content_generation': {
                'include_documentation': self.include_documentation,
                'include_metrics': self.include_metrics,
                'include_configuration': self.include_configuration,
                'include_examples': self.include_examples
            },
            'documentation_settings': {
                'documentation_format': self.documentation_format,
                'include_test_descriptions': self.include_test_descriptions,
                'include_coverage_reports': self.include_coverage_reports
            },
            'file_naming': {
                'test_file_prefix': self.test_file_prefix,
                'test_file_suffix': self.test_file_suffix,
                'use_timestamp_suffix': self.use_timestamp_suffix
            },
            'quality_settings': {
                'validate_syntax': self.validate_syntax,
                'format_code': self.format_code,
                'optimize_imports': self.optimize_imports
            }
        }


class CodebaseAgnosticOutputSystem:
    """Codebase-agnostic test output generation system."""
    
    def __init__(self):
        # Load output templates
        self.output_templates = self._load_output_templates()
        self.language_configs = self._load_language_configs()
        
        print(f"Codebase-Agnostic Output System initialized")
        print(f"   Supported formats: {len(OutputFormat)}")
        print(f"   Template languages: {len(self.language_configs)}")
    
    def generate_outputs(self, 
                        test_suite: UniversalTestSuite,
                        config: OutputSystemConfig) -> List[str]:
        """Generate outputs for a test suite."""
        
        print(f"\\nGenerating outputs for: {test_suite.name}")
        print(f"   Target formats: {config.output_formats}")
        print(f"   Output directory: {config.output_directory}")
        
        generated_files = []
        
        # Create output directory structure
        base_output_dir = Path(config.output_directory)
        self._create_output_directories(base_output_dir, test_suite, config)
        
        # Generate outputs for each format
        for format_name in config.output_formats:
            try:
                format_enum = OutputFormat(format_name)
                bundle = self._generate_format_output(test_suite, format_enum, config)
                
                # Write bundle to files
                bundle_files = self._write_output_bundle(bundle, base_output_dir, config)
                generated_files.extend(bundle_files)
                
                print(f"   ✓ Generated {format_name}: {len(bundle_files)} files")
                
            except ValueError:
                print(f"   ❌ Unsupported format: {format_name}")
                continue
            except Exception as e:
                print(f"   ❌ Error generating {format_name}: {str(e)}")
                continue
        
        # Generate additional documentation if requested
        if config.include_documentation:
            doc_files = self._generate_documentation(test_suite, base_output_dir, config)
            generated_files.extend(doc_files)
            print(f"   ✓ Generated documentation: {len(doc_files)} files")
        
        # Generate configuration files if requested
        if config.include_configuration:
            config_files = self._generate_configuration_files(test_suite, base_output_dir, config)
            generated_files.extend(config_files)
            print(f"   ✓ Generated configuration: {len(config_files)} files")
        
        print(f"   Total files generated: {len(generated_files)}")
        
        return generated_files
    
    def _create_output_directories(self, 
                                  base_dir: Path,
                                  test_suite: UniversalTestSuite,
                                  config: OutputSystemConfig):
        """Create output directory structure."""
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        if config.create_separate_directories:
            # Create subdirectories for organization
            if config.organize_by_framework and test_suite.metadata.framework:
                framework_dir = base_dir / test_suite.metadata.framework
                framework_dir.mkdir(exist_ok=True)
            
            if config.organize_by_language:
                # Detect language from test suite or default to universal
                language = self._detect_suite_language(test_suite)
                language_dir = base_dir / language
                language_dir.mkdir(exist_ok=True)
    
    def _generate_format_output(self, 
                               test_suite: UniversalTestSuite,
                               format: OutputFormat,
                               config: OutputSystemConfig) -> TestOutputBundle:
        """Generate output for specific format."""
        
        bundle = TestOutputBundle(
            name=f"{test_suite.name}_{format.value}",
            format=format
        )
        
        # Generate main content based on format
        if format in [OutputFormat.PYTHON, OutputFormat.JAVASCRIPT, OutputFormat.JAVA, 
                     OutputFormat.CSHARP, OutputFormat.GO, OutputFormat.RUST]:
            # Programming language formats
            bundle.main_content = self._generate_programming_language_output(test_suite, format)
            
        elif format == OutputFormat.UNIVERSAL:
            # Universal format
            bundle.main_content = self._generate_universal_output(test_suite)
            
        elif format in [OutputFormat.JSON, OutputFormat.YAML]:
            # Structured data formats
            bundle.main_content = self._generate_structured_output(test_suite, format)
            
        elif format in [OutputFormat.MARKDOWN, OutputFormat.HTML]:
            # Documentation formats
            bundle.main_content = self._generate_documentation_output(test_suite, format)
        
        # Generate additional content
        bundle.additional_content = self._generate_additional_content(test_suite, format, config)
        
        # Calculate metrics
        bundle.metrics = self._calculate_output_metrics(test_suite, bundle)
        
        return bundle
    
    def _generate_programming_language_output(self, 
                                            test_suite: UniversalTestSuite,
                                            format: OutputFormat) -> str:
        """Generate programming language specific output."""
        
        language_name = format.value
        language_config = self.language_configs.get(language_name, {})
        
        # Get language template
        template = self.output_templates.get(language_name, {}).get('test_file', "")
        
        if not template:
            # Generate basic template
            template = self._generate_basic_language_template(format)
        
        # Generate imports
        imports = self._generate_imports(test_suite, language_config)
        
        # Generate test classes/functions
        test_content = self._generate_test_content(test_suite, language_config)
        
        # Format template
        output = template.format(
            suite_name=test_suite.name,
            imports=imports,
            test_content=test_content,
            framework=test_suite.metadata.framework or "universal",
            description=test_suite.metadata.description or ""
        )
        
        # Apply language-specific formatting
        output = self._apply_language_formatting(output, format)
        
        return output
    
    def _generate_universal_output(self, test_suite: UniversalTestSuite) -> str:
        """Generate universal format output."""
        
        # Convert to structured format that can be read by any system
        universal_data = {
            'test_suite': test_suite.to_dict(),
            'format': 'universal',
            'version': '1.0',
            'generated_at': datetime.now().isoformat()
        }
        
        return json.dumps(universal_data, indent=2)
    
    def _generate_structured_output(self, 
                                   test_suite: UniversalTestSuite,
                                   format: OutputFormat) -> str:
        """Generate structured data format output."""
        
        data = test_suite.to_dict()
        
        if format == OutputFormat.JSON:
            return json.dumps(data, indent=2)
        elif format == OutputFormat.YAML:
            return yaml.dump(data, default_flow_style=False, indent=2)
        
        return str(data)
    
    def _generate_documentation_output(self, 
                                     test_suite: UniversalTestSuite,
                                     format: OutputFormat) -> str:
        """Generate documentation format output."""
        
        if format == OutputFormat.MARKDOWN:
            return self._generate_markdown_documentation(test_suite)
        elif format == OutputFormat.HTML:
            return self._generate_html_documentation(test_suite)
        
        return ""
    
    def _generate_markdown_documentation(self, test_suite: UniversalTestSuite) -> str:
        """Generate Markdown documentation."""
        
        md_content = []
        
        # Header
        md_content.append(f"# {test_suite.name}")
        md_content.append("")
        md_content.append(f"{test_suite.metadata.description or 'Test suite documentation'}")
        md_content.append("")
        
        # Metadata
        md_content.append("## Test Suite Information")
        md_content.append("")
        md_content.append(f"- **Framework**: {test_suite.metadata.framework or 'Universal'}")
        md_content.append(f"- **Category**: {test_suite.metadata.category or 'General'}")
        md_content.append(f"- **Tags**: {', '.join(test_suite.metadata.tags) if test_suite.metadata.tags else 'None'}")
        md_content.append(f"- **Test Cases**: {len(test_suite.test_cases)}")
        md_content.append(f"- **Total Tests**: {test_suite.count_tests()}")
        md_content.append("")
        
        # Test cases
        md_content.append("## Test Cases")
        md_content.append("")
        
        for i, test_case in enumerate(test_suite.test_cases, 1):
            md_content.append(f"### {i}. {test_case.name}")
            md_content.append("")
            md_content.append(f"{test_case.description or 'No description available'}")
            md_content.append("")
            
            if test_case.tests:
                md_content.append("#### Tests:")
                md_content.append("")
                
                for j, test in enumerate(test_case.tests, 1):
                    md_content.append(f"{j}. **{test.name}**")
                    if test.description:
                        md_content.append(f"   - {test.description}")
                    md_content.append(f"   - Assertions: {len(test.assertions)}")
                    md_content.append("")
        
        return "\\n".join(md_content)
    
    def _generate_html_documentation(self, test_suite: UniversalTestSuite) -> str:
        """Generate HTML documentation."""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{test_suite.name} - Test Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #666; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .test-case {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .test {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>{test_suite.name}</h1>
    <p>{test_suite.metadata.description or 'Test suite documentation'}</p>
    
    <div class="metadata">
        <h2>Test Suite Information</h2>
        <p><strong>Framework:</strong> {test_suite.metadata.framework or 'Universal'}</p>
        <p><strong>Category:</strong> {test_suite.metadata.category or 'General'}</p>
        <p><strong>Tags:</strong> {', '.join(test_suite.metadata.tags) if test_suite.metadata.tags else 'None'}</p>
        <p><strong>Test Cases:</strong> {len(test_suite.test_cases)}</p>
        <p><strong>Total Tests:</strong> {test_suite.count_tests()}</p>
    </div>
    
    <h2>Test Cases</h2>
"""
        
        for i, test_case in enumerate(test_suite.test_cases, 1):
            html_content += f"""
    <div class="test-case">
        <h3>{i}. {test_case.name}</h3>
        <p>{test_case.description or 'No description available'}</p>
        
        <h4>Tests:</h4>
"""
            
            for j, test in enumerate(test_case.tests, 1):
                html_content += f"""
        <div class="test">
            <strong>{j}. {test.name}</strong>
            <p>{test.description or 'No description'}</p>
            <p><em>Assertions: {len(test.assertions)}</em></p>
        </div>
"""
            
            html_content += "</div>"
        
        html_content += """
</body>
</html>
"""
        
        return html_content
    
    def _generate_imports(self, test_suite: UniversalTestSuite, language_config: Dict) -> str:
        """Generate import statements for language."""
        
        imports = language_config.get('default_imports', [])
        
        # Add framework-specific imports
        framework = test_suite.metadata.framework
        if framework and framework in language_config.get('framework_imports', {}):
            framework_imports = language_config['framework_imports'][framework]
            imports.extend(framework_imports)
        
        # Format imports based on language
        import_format = language_config.get('import_format', 'import {module}')
        formatted_imports = []
        
        for import_item in imports:
            formatted_import = import_format.format(module=import_item)
            formatted_imports.append(formatted_import)
        
        return "\\n".join(formatted_imports)
    
    def _generate_test_content(self, test_suite: UniversalTestSuite, language_config: Dict) -> str:
        """Generate test content for language."""
        
        content_lines = []
        
        # Generate test classes/functions for each test case
        for test_case in test_suite.test_cases:
            case_content = self._generate_test_case_content(test_case, language_config)
            content_lines.append(case_content)
        
        return "\\n\\n".join(content_lines)
    
    def _generate_test_case_content(self, test_case: UniversalTestCase, language_config: Dict) -> str:
        """Generate content for a test case."""
        
        # Get templates
        class_template = language_config.get('class_template', 'class {class_name}:\\n{methods}')
        method_template = language_config.get('method_template', 'def {method_name}():\\n    {content}')
        
        # Generate methods for each test
        methods = []
        for test in test_case.tests:
            method_content = self._generate_test_method_content(test, language_config)
            method = method_template.format(
                method_name=test.name,
                content=method_content
            )
            methods.append(method)
        
        # Format class
        class_content = class_template.format(
            class_name=test_case.name,
            methods="\\n\\n".join(methods)
        )
        
        return class_content
    
    def _generate_test_method_content(self, test: UniversalTest, language_config: Dict) -> str:
        """Generate content for a test method."""
        
        content_lines = []
        
        # Add test description as comment
        if test.description:
            comment_format = language_config.get('comment_format', '# {comment}')
            comment = comment_format.format(comment=test.description)
            content_lines.append(comment)
        
        # Add test function content
        if test.test_function:
            content_lines.append(test.test_function)
        
        # Add assertions
        for assertion in test.assertions:
            assertion_code = self._generate_assertion_code(assertion, language_config)
            if assertion_code:
                content_lines.append(assertion_code)
        
        return "\\n    ".join(content_lines)
    
    def _generate_assertion_code(self, assertion: TestAssertion, language_config: Dict) -> str:
        """Generate assertion code for language."""
        
        # Get assertion templates
        assertion_templates = language_config.get('assertions', {})
        template = assertion_templates.get(assertion.assertion_type.value)
        
        if template:
            return template.format(
                actual=assertion.actual,
                expected=assertion.expected,
                message=assertion.message or ""
            )
        
        # Default assertion format
        return f"assert {assertion.actual}"
    
    def _write_output_bundle(self, 
                           bundle: TestOutputBundle,
                           base_dir: Path,
                           config: OutputSystemConfig) -> List[str]:
        """Write output bundle to files."""
        
        written_files = []
        
        # Determine output directory
        output_dir = base_dir
        if config.organize_by_framework and bundle.format != OutputFormat.UNIVERSAL:
            output_dir = output_dir / bundle.format.value
            output_dir.mkdir(exist_ok=True)
        
        # Generate main file name
        main_filename = self._generate_filename(bundle, config)
        main_file_path = output_dir / main_filename
        
        # Write main content
        if bundle.main_content:
            main_file_path.write_text(bundle.main_content, encoding='utf-8')
            written_files.append(str(main_file_path))
            bundle.test_files.append(str(main_file_path))
        
        # Write additional content
        for filename, content in bundle.additional_content.items():
            additional_file_path = output_dir / filename
            additional_file_path.write_text(content, encoding='utf-8')
            written_files.append(str(additional_file_path))
        
        return written_files
    
    def _generate_filename(self, bundle: TestOutputBundle, config: OutputSystemConfig) -> str:
        """Generate filename for bundle."""
        
        # Get file extension for format
        extensions = {
            OutputFormat.PYTHON: '.py',
            OutputFormat.JAVASCRIPT: '.js',
            OutputFormat.TYPESCRIPT: '.ts',
            OutputFormat.JAVA: '.java',
            OutputFormat.CSHARP: '.cs',
            OutputFormat.GO: '.go',
            OutputFormat.RUST: '.rs',
            OutputFormat.PHP: '.php',
            OutputFormat.RUBY: '.rb',
            OutputFormat.UNIVERSAL: '.json',
            OutputFormat.JSON: '.json',
            OutputFormat.YAML: '.yaml',
            OutputFormat.MARKDOWN: '.md',
            OutputFormat.HTML: '.html'
        }
        
        extension = extensions.get(bundle.format, '.txt')
        
        # Build filename
        filename = f"{config.test_file_prefix}{bundle.name}{config.test_file_suffix}"
        
        # Add timestamp if requested
        if config.use_timestamp_suffix:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename += f"_{timestamp}"
        
        filename += extension
        
        return filename
    
    def _load_output_templates(self) -> Dict[str, Dict[str, str]]:
        """Load output templates for different languages."""
        
        return {
            'python': {
                'test_file': '''"""
{description}

Test suite: {suite_name}
Framework: {framework}
Generated by TestMaster Universal Orchestrator
"""

{imports}


{test_content}


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__])
''',
                'class_template': '''class {class_name}:
    """Test case: {class_name}"""
    
{methods}''',
                'method_template': '''    def {method_name}(self):
        """{description}"""
        {content}'''
            },
            'javascript': {
                'test_file': '''/**
 * {description}
 * 
 * Test suite: {suite_name}
 * Framework: {framework}
 * Generated by TestMaster Universal Orchestrator
 */

{imports}

{test_content}
''',
                'describe_template': '''describe('{class_name}', () => {{
{methods}
}});''',
                'test_template': '''  test('{method_name}', () => {{
    // {description}
    {content}
  }});'''
            }
        }
    
    def _load_language_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load language-specific configurations."""
        
        return {
            'python': {
                'default_imports': ['import unittest', 'import pytest'],
                'import_format': 'import {module}',
                'comment_format': '# {comment}',
                'class_template': 'class {class_name}:\\n{methods}',
                'method_template': '    def {method_name}(self):\\n        {content}',
                'assertions': {
                    'equals': 'assert {actual} == {expected}',
                    'not_equals': 'assert {actual} != {expected}',
                    'true': 'assert {actual}',
                    'false': 'assert not {actual}'
                },
                'framework_imports': {
                    'pytest': ['import pytest'],
                    'unittest': ['import unittest']
                }
            },
            'javascript': {
                'default_imports': [],
                'import_format': "const {module} = require('{module}');",
                'comment_format': '// {comment}',
                'describe_template': "describe('{class_name}', () => {{\\n{methods}\\n}});",
                'test_template': "  test('{method_name}', () => {{\\n    {content}\\n  }});",
                'assertions': {
                    'equals': 'expect({actual}).toBe({expected});',
                    'not_equals': 'expect({actual}).not.toBe({expected});',
                    'true': 'expect({actual}).toBeTruthy();',
                    'false': 'expect({actual}).toBeFalsy();'
                },
                'framework_imports': {
                    'jest': [],
                    'mocha': ["const { expect } = require('chai');"]
                }
            }
        }
    
    def _detect_suite_language(self, test_suite: UniversalTestSuite) -> str:
        """Detect primary language of test suite."""
        
        # Check framework first
        framework = test_suite.metadata.framework
        if framework:
            framework_languages = {
                'pytest': 'python',
                'unittest': 'python',
                'jest': 'javascript',
                'mocha': 'javascript',
                'junit': 'java',
                'nunit': 'csharp'
            }
            
            if framework in framework_languages:
                return framework_languages[framework]
        
        # Default to universal
        return 'universal'
    
    def _generate_additional_content(self, 
                                   test_suite: UniversalTestSuite,
                                   format: OutputFormat,
                                   config: OutputSystemConfig) -> Dict[str, str]:
        """Generate additional content files."""
        
        additional_content = {}
        
        # Generate configuration files
        if config.include_configuration:
            if format == OutputFormat.PYTHON:
                additional_content['pytest.ini'] = self._generate_pytest_config()
                additional_content['setup.cfg'] = self._generate_python_setup_config()
            elif format == OutputFormat.JAVASCRIPT:
                additional_content['jest.config.js'] = self._generate_jest_config()
            elif format == OutputFormat.JAVA:
                additional_content['pom.xml'] = self._generate_maven_config()
        
        # Generate examples
        if config.include_examples:
            additional_content['examples.md'] = self._generate_examples(test_suite, format)
        
        return additional_content
    
    def _generate_pytest_config(self) -> str:
        """Generate pytest configuration."""
        return '''[tool:pytest]
testpaths = .
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --cov=.
    --cov-report=html
    --cov-report=term-missing
'''
    
    def _generate_jest_config(self) -> str:
        """Generate Jest configuration."""
        return '''module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.js', '**/?(*.)+(spec|test).js'],
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  verbose: true
};
'''
    
    def _calculate_output_metrics(self, test_suite: UniversalTestSuite, bundle: TestOutputBundle) -> Dict[str, Any]:
        """Calculate metrics for output bundle."""
        
        return {
            'content_size': len(bundle.main_content),
            'test_cases': len(test_suite.test_cases),
            'total_tests': test_suite.count_tests(),
            'format': bundle.format.value,
            'generation_time': bundle.generation_timestamp.isoformat()
        }
    
    def _apply_language_formatting(self, content: str, format: OutputFormat) -> str:
        """Apply language-specific formatting."""
        
        # Basic formatting improvements
        formatted_content = content
        
        # Remove excessive blank lines
        lines = formatted_content.split('\\n')
        cleaned_lines = []
        prev_blank = False
        
        for line in lines:
            is_blank = line.strip() == ''
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank
        
        return '\\n'.join(cleaned_lines)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return [format.value for format in OutputFormat]