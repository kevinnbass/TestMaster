"""
Automated Test Generation System - Enterprise-grade intelligent test generation

This package provides comprehensive automated test generation capabilities with
advanced code analysis, intelligent template systems, and multi-format export
support. Designed for enterprise applications requiring high test coverage,
quality assurance, and maintainable test suites.

Enterprise Features:
- Intelligent code analysis with AST parsing and complexity metrics
- Advanced template engine with type-aware test data generation
- Multi-strategy test generation with coverage optimization
- Comprehensive export system with multiple format support
- Quality assurance with validation and optimization
- Performance monitoring and analytics

Key Components:
- Code Analysis: Advanced AST-based code structure analysis
- Test Generation: Intelligent test case generation with multiple strategies
- Template System: Flexible template engine for various test types
- Export System: Multi-format export with comprehensive reporting
- Quality Assurance: Validation, optimization, and best practices

Integration Example:
    >>> from core.testing.automated_generation import create_test_generator, TestType
    >>> from pathlib import Path
    >>> 
    >>> # Create generator with custom configuration
    >>> generator = create_test_generator(target_coverage=85.0)
    >>> 
    >>> # Generate tests for a module
    >>> test_suite = generator.generate_test_suite(Path("src/my_module.py"))
    >>> 
    >>> # Export to pytest format
    >>> from core.testing.automated_generation import create_test_exporter
    >>> exporter = create_test_exporter()
    >>> result = exporter.export_test_suite(test_suite, Path("tests/test_my_module.py"))
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from .test_models import (
    # Core data models
    TestCase,
    TestSuite,
    ModuleAnalysis,
    FunctionInfo,
    ClassInfo,
    ParameterInfo,
    GenerationConfig,
    GenerationReport,
    
    # Enums
    TestType,
    ComplexityLevel,
    CoverageLevel,
    
    # Factory functions
    create_test_case,
    create_test_suite,
    create_function_info,
    create_module_analysis,
    create_generation_config,
    
    # Utility functions
    calculate_coverage_score,
    validate_test_case,
    merge_test_suites,
    
    # Constants
    DEFAULT_TYPE_GENERATORS,
    COVERAGE_TARGETS,
    COMPLEXITY_WEIGHTS
)

from .code_analyzer import (
    # Main analyzer
    CodeAnalyzer,
    create_code_analyzer,
    
    # Complexity calculation
    ComplexityCalculator,
    ComplexityMetrics,
    
    # Dependency tracking
    DependencyTracker,
    DependencyInfo,
    
    # Pattern detection
    PatternDetector,
    SecurityIssue,
    PerformanceIssue,
    SecurityPattern,
    PerformancePattern
)

from .test_templates import (
    # Template engine
    TestTemplateEngine,
    create_test_template_engine,
    
    # Template components
    TemplateRenderer,
    TemplateContext,
    TemplateMetadata,
    
    # Utility functions
    validate_template
)

from .test_generator import (
    # Main generator
    AutomatedTestGenerator,
    create_automated_test_generator,
    
    # Generation strategies
    TestGenerationStrategy,
    ComprehensiveStrategy,
    FocusedStrategy,
    create_comprehensive_strategy,
    create_focused_strategy,
    
    # Coverage optimization
    CoverageOptimizer,
    
    # Configuration and metrics
    GenerationStrategy,
    GenerationMetrics,
    
    # Utility functions
    generate_tests_async,
    validate_generation_config
)

from .test_exporter import (
    # Main exporter
    TestExporter,
    create_test_exporter,
    
    # Export configuration
    ExportConfig,
    ExportResult,
    create_export_config,
    
    # Formatters
    ExportFormatter,
    PytestFormatter,
    UnittestFormatter,
    
    # Report generation
    ReportGenerator,
    
    # Quality validation
    QualityValidator,
    
    # Utility functions
    export_test_suite_to_file,
    validate_exported_tests
)

__all__ = [
    # Main components
    'AutomatedTestGenerator',
    'CodeAnalyzer', 
    'TestTemplateEngine',
    'TestExporter',
    
    # Factory functions
    'create_automated_test_generator',
    'create_code_analyzer',
    'create_test_template_engine',
    'create_test_exporter',
    
    # Core data models
    'TestCase',
    'TestSuite',
    'ModuleAnalysis',
    'FunctionInfo',
    'ClassInfo',
    'ParameterInfo',
    'GenerationConfig',
    'GenerationReport',
    'ExportConfig',
    'ExportResult',
    
    # Enums
    'TestType',
    'ComplexityLevel',
    'CoverageLevel',
    
    # Generation strategies
    'TestGenerationStrategy',
    'ComprehensiveStrategy',
    'FocusedStrategy',
    'create_comprehensive_strategy',
    'create_focused_strategy',
    
    # Analysis components
    'ComplexityCalculator',
    'DependencyTracker',
    'PatternDetector',
    'ComplexityMetrics',
    'DependencyInfo',
    'SecurityIssue',
    'PerformanceIssue',
    
    # Template components
    'TemplateRenderer',
    'TemplateContext',
    'TemplateMetadata',
    
    # Export components
    'ExportFormatter',
    'PytestFormatter',
    'UnittestFormatter',
    'ReportGenerator',
    'QualityValidator',
    
    # Optimization
    'CoverageOptimizer',
    
    # Model factory functions
    'create_test_case',
    'create_test_suite',
    'create_function_info',
    'create_module_analysis',
    'create_generation_config',
    'create_export_config',
    
    # Utility functions
    'calculate_coverage_score',
    'validate_test_case',
    'merge_test_suites',
    'validate_template',
    'generate_tests_async',
    'validate_generation_config',
    'export_test_suite_to_file',
    'validate_exported_tests',
    
    # Constants
    'DEFAULT_TYPE_GENERATORS',
    'COVERAGE_TARGETS',
    'COMPLEXITY_WEIGHTS',
    
    # Convenience functions
    'generate_tests_for_module',
    'create_complete_test_suite',
    'analyze_and_generate_tests',
    'export_with_reports'
]

# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Automated Generation Team'
__description__ = 'Enterprise-grade intelligent automated test generation system'


def generate_tests_for_module(module_path, 
                             output_path=None,
                             target_coverage=85.0,
                             export_format="pytest") -> TestSuite:
    """
    Convenience function to generate tests for a single module.
    
    Args:
        module_path: Path to Python module (str or Path)
        output_path: Optional path for test output (str or Path)
        target_coverage: Target coverage percentage
        export_format: Export format (pytest, unittest)
        
    Returns:
        Generated TestSuite
        
    Example:
        >>> from core.testing.automated_generation import generate_tests_for_module
        >>> test_suite = generate_tests_for_module(
        ...     "src/my_module.py",
        ...     "tests/test_my_module.py",
        ...     target_coverage=90.0
        ... )
        >>> print(f"Generated {len(test_suite.test_cases)} tests")
    """
    from pathlib import Path
    
    module_path = Path(module_path)
    
    # Create generator with configuration
    config = create_generation_config(target_coverage=target_coverage)
    generator = create_automated_test_generator(config)
    
    # Generate test suite
    test_suite = generator.generate_test_suite(module_path)
    
    # Export if output path provided
    if output_path:
        output_path = Path(output_path)
        export_config = create_export_config(output_format=export_format)
        exporter = create_test_exporter(export_config)
        result = exporter.export_test_suite(test_suite, output_path)
        
        if not result.success:
            print(f"Export failed: {result.validation_errors}")
    
    return test_suite


def create_complete_test_suite(module_paths,
                              output_dir,
                              strategy="comprehensive",
                              target_coverage=85.0,
                              export_format="pytest",
                              generate_reports=True) -> Dict[str, TestSuite]:
    """
    Create complete test suite for multiple modules with comprehensive reporting.
    
    Args:
        module_paths: List of module paths to process
        output_dir: Output directory for test files and reports
        strategy: Generation strategy (comprehensive, focused)
        target_coverage: Target coverage percentage
        export_format: Export format (pytest, unittest)
        generate_reports: Whether to generate comprehensive reports
        
    Returns:
        Dictionary mapping module names to test suites
        
    Example:
        >>> from core.testing.automated_generation import create_complete_test_suite
        >>> from pathlib import Path
        >>> 
        >>> modules = [Path("src/module1.py"), Path("src/module2.py")]
        >>> test_suites = create_complete_test_suite(
        ...     modules,
        ...     "tests/",
        ...     strategy="comprehensive",
        ...     target_coverage=90.0
        ... )
        >>> print(f"Generated test suites for {len(test_suites)} modules")
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    module_paths = [Path(p) for p in module_paths]
    
    # Create generator with strategy
    config = create_generation_config(target_coverage=target_coverage)
    generator = create_automated_test_generator(config)
    generator.set_strategy(strategy)
    
    # Generate test suites
    test_suites = generator.generate_batch_test_suites(module_paths)
    
    # Export with reports
    export_config = create_export_config(
        output_format=export_format,
        generate_reports=generate_reports
    )
    exporter = create_test_exporter(export_config)
    
    export_results = exporter.export_multiple_suites(
        test_suites, 
        output_dir,
        generate_reports=generate_reports
    )
    
    # Print summary
    successful_exports = sum(1 for result in export_results.values() if result.success)
    total_tests = sum(len(suite.test_cases) for suite in test_suites.values())
    
    print(f"Test Generation Complete:")
    print(f"  Modules processed: {len(test_suites)}")
    print(f"  Successful exports: {successful_exports}")
    print(f"  Total tests generated: {total_tests}")
    print(f"  Output directory: {output_dir}")
    
    return test_suites


def analyze_and_generate_tests(source_dir, 
                              output_dir,
                              file_pattern="**/*.py",
                              **kwargs) -> Dict[str, TestSuite]:
    """
    Analyze source directory and generate tests for all matching files.
    
    Args:
        source_dir: Source directory to analyze
        output_dir: Output directory for tests
        file_pattern: Glob pattern for matching files
        **kwargs: Additional arguments for test generation
        
    Returns:
        Dictionary mapping module names to test suites
        
    Example:
        >>> from core.testing.automated_generation import analyze_and_generate_tests
        >>> test_suites = analyze_and_generate_tests(
        ...     "src/",
        ...     "tests/",
        ...     file_pattern="**/*.py",
        ...     target_coverage=85.0,
        ...     strategy="comprehensive"
        ... )
    """
    from pathlib import Path
    
    source_dir = Path(source_dir)
    
    # Find all Python files matching pattern
    python_files = list(source_dir.glob(file_pattern))
    
    # Filter out test files and other non-source files
    source_files = []
    for file_path in python_files:
        if not any(pattern in str(file_path) for pattern in ['test_', '__pycache__', '.pyc']):
            source_files.append(file_path)
    
    print(f"Found {len(source_files)} source files to analyze")
    
    # Generate tests for all source files
    return create_complete_test_suite(
        source_files,
        output_dir,
        **kwargs
    )


def export_with_reports(test_suites,
                       output_dir,
                       formats=None,
                       include_requirements=True,
                       include_config=True) -> Dict[str, Dict[str, ExportResult]]:
    """
    Export test suites in multiple formats with comprehensive reporting.
    
    Args:
        test_suites: Dictionary of test suites to export
        output_dir: Output directory for exports
        formats: List of export formats (default: ["pytest", "unittest"])
        include_requirements: Generate requirements files
        include_config: Generate configuration files
        
    Returns:
        Dictionary mapping formats to export results
        
    Example:
        >>> from core.testing.automated_generation import export_with_reports
        >>> results = export_with_reports(
        ...     test_suites,
        ...     "exports/",
        ...     formats=["pytest", "unittest"],
        ...     include_requirements=True
        ... )
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    formats = formats or ["pytest", "unittest"]
    
    all_results = {}
    
    for export_format in formats:
        format_dir = output_dir / export_format
        
        # Create export configuration
        export_config = create_export_config(
            output_format=export_format,
            generate_requirements=include_requirements,
            generate_config_files=include_config
        )
        
        # Create exporter and export
        exporter = create_test_exporter(export_config)
        results = exporter.export_multiple_suites(
            test_suites,
            format_dir,
            generate_reports=True
        )
        
        all_results[export_format] = results
        
        # Print format summary
        successful = sum(1 for r in results.values() if r.success)
        total_tests = sum(r.total_tests_exported for r in results.values())
        
        print(f"{export_format.title()} Export:")
        print(f"  Successful: {successful}/{len(results)}")
        print(f"  Total tests: {total_tests}")
        print(f"  Output: {format_dir}")
    
    return all_results


# Module initialization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())