"""
TestMaster Testing & Monitoring API Blueprint - AGENT B ENHANCED
================================================================

Comprehensive REST API for testing and monitoring capabilities.
Exposes all testing hub features through clean, well-documented endpoints.

AGENT B Enhancement: Hour 10-12
- Complete testing API layer
- Coverage analysis endpoints
- AI test generation endpoints
- Self-healing test endpoints
- Monitoring integration endpoints
"""

from flask import Blueprint, request, jsonify, send_file
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import io
import tempfile

# Import testing and monitoring components
from ..testing import ConsolidatedTestingHub
from ..testing.components.coverage_analyzer import CoverageAnalyzer, EnhancedCoverageReport
from ..testing.components.integration_generator import (
    IntegrationTestGenerator,
    TestGenerationStrategy,
    AIProvider,
    AIGenerationConfig
)
from ..monitoring import AgentQualityAssurance
from ..base import UnifiedAnalysisType

# Create blueprint
testing_api = Blueprint('testing_api', __name__, url_prefix='/api/v2/testing')

# Component instances (initialized in init function)
testing_hub = None
coverage_analyzer = None
test_generator = None
monitoring_qa = None

# Logger
logger = logging.getLogger("testing_api")


def init_testing_api(app_config: Optional[Dict[str, Any]] = None):
    """Initialize testing API with configuration."""
    global testing_hub, coverage_analyzer, test_generator, monitoring_qa
    
    config = app_config or {}
    testing_hub = ConsolidatedTestingHub(config)
    coverage_analyzer = CoverageAnalyzer(config)
    test_generator = IntegrationTestGenerator(config)
    monitoring_qa = AgentQualityAssurance("testing_api", config)
    
    logger.info("Testing API initialized with all components")


# === TESTING HUB ENDPOINTS ===

@testing_api.route('/status', methods=['GET'])
def get_testing_status():
    """
    Get comprehensive testing infrastructure status.
    
    Returns:
        JSON with testing hub status, capabilities, and statistics
    """
    try:
        status = {
            'api_version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'testing_hub': testing_hub.get_testing_intelligence() if testing_hub else {},
            'capabilities': testing_hub.get_capabilities() if testing_hub else {},
            'statistics': testing_hub.get_status() if testing_hub else {},
            'ai_providers': {
                'claude': 'available' if test_generator and AIProvider.CLAUDE in test_generator._ai_generators else 'unavailable',
                'gemini': 'available' if test_generator and AIProvider.GEMINI in test_generator._ai_generators else 'unavailable',
                'universal': 'available'
            }
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Failed to get testing status: {e}")
        return jsonify({'error': str(e)}), 500


@testing_api.route('/health', methods=['GET'])
def health_check():
    """Quick health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200


# === COVERAGE ANALYSIS ENDPOINTS ===

@testing_api.route('/coverage/analyze', methods=['POST'])
def analyze_coverage():
    """
    Analyze test coverage with enhanced metrics.
    
    Request body:
    {
        "project_path": "/path/to/project",
        "include_archive_features": true,
        "statistical_analysis": true,
        "confidence_level": 0.95
    }
    
    Returns:
        Comprehensive coverage report with all metrics
    """
    try:
        data = request.get_json()
        project_path = data.get('project_path', '.')
        include_archive = data.get('include_archive_features', True)
        
        # Generate enhanced coverage report
        report = coverage_analyzer.generate_enhanced_report(
            project_path=project_path,
            include_archive_features=include_archive
        )
        
        # Convert to JSON-serializable format
        report_dict = {
            'overall_percentage': report.overall_percentage,
            'line_coverage': report.line_coverage,
            'branch_coverage': report.branch_coverage,
            'function_coverage': report.function_coverage,
            'total_lines': report.total_lines,
            'covered_lines': report.covered_lines,
            'missing_lines': report.missing_lines,
            'test_count': report.test_count,
            'test_categories': report.test_categories,
            'test_quality_score': report.test_quality_score,
            'test_smells': report.test_smells,
            'optimization_suggestions': report.optimization_suggestions,
            'timestamp': report.timestamp.isoformat(),
            'modules': len(report.modules)  # Summary instead of full list
        }
        
        # Add dependency graph if available
        if report.dependency_graph:
            report_dict['dependency_graph'] = report.dependency_graph
            report_dict['critical_paths'] = report.critical_paths
            report_dict['circular_dependencies'] = report.circular_dependencies
        
        return jsonify(report_dict), 200
        
    except Exception as e:
        logger.error(f"Coverage analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


@testing_api.route('/coverage/function/<path:file_path>', methods=['GET'])
def analyze_function_coverage(file_path):
    """
    Analyze coverage at function level for a specific file.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        Function-level coverage details
    """
    try:
        functions = coverage_analyzer.analyze_function_coverage(file_path)
        
        # Convert to JSON-serializable format
        functions_data = []
        for func in functions:
            functions_data.append({
                'name': func.name,
                'line_start': func.line_start,
                'line_end': func.line_end,
                'total_lines': func.total_lines,
                'coverage_percentage': func.coverage_percentage,
                'complexity': func.complexity,
                'is_tested': func.is_tested,
                'test_quality_score': func.test_quality_score,
                'missing_lines': list(func.missing_lines)[:10]  # Limit for response size
            })
        
        return jsonify({
            'file_path': file_path,
            'functions': functions_data,
            'function_count': len(functions_data)
        }), 200
        
    except Exception as e:
        logger.error(f"Function coverage analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


@testing_api.route('/coverage/gaps', methods=['POST'])
def identify_coverage_gaps():
    """
    Identify coverage gaps and untested components.
    
    Request body:
    {
        "test_results": [...],
        "threshold": 80.0
    }
    
    Returns:
        Coverage gaps and recommendations
    """
    try:
        data = request.get_json()
        test_results = data.get('test_results', [])
        threshold = data.get('threshold', 80.0)
        
        # Analyze coverage
        analysis = testing_hub.analyze_coverage(test_results)
        
        # Identify gaps
        gaps = {
            'coverage_gaps': analysis.coverage_gaps if hasattr(analysis, 'coverage_gaps') else [],
            'untested_components': analysis.untested_components if hasattr(analysis, 'untested_components') else [],
            'below_threshold': [],
            'recommendations': []
        }
        
        # Find components below threshold
        if hasattr(analysis, 'line_coverage') and analysis.line_coverage < threshold:
            gaps['below_threshold'].append({
                'type': 'line_coverage',
                'value': analysis.line_coverage,
                'threshold': threshold
            })
        
        # Generate recommendations
        if gaps['untested_components']:
            gaps['recommendations'].append(f"Add tests for {len(gaps['untested_components'])} untested components")
        if gaps['coverage_gaps']:
            gaps['recommendations'].append(f"Fill {len(gaps['coverage_gaps'])} identified coverage gaps")
        
        return jsonify(gaps), 200
        
    except Exception as e:
        logger.error(f"Coverage gap analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


# === AI TEST GENERATION ENDPOINTS ===

@testing_api.route('/generate/ai', methods=['POST'])
def generate_ai_tests():
    """
    Generate tests using AI providers.
    
    Request body:
    {
        "code": "def example(): pass",
        "file_path": "/path/to/file.py",
        "provider": "claude|gemini|universal",
        "strategy": "comprehensive|focused|security|performance",
        "config": {...}
    }
    
    Returns:
        Generated test code with metadata
    """
    try:
        data = request.get_json()
        
        # Get code to test
        code = data.get('code')
        file_path = data.get('file_path')
        
        if not code and file_path:
            with open(file_path, 'r') as f:
                code = f.read()
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Configure generation
        provider = AIProvider[data.get('provider', 'UNIVERSAL').upper()]
        strategy = TestGenerationStrategy[data.get('strategy', 'COMPREHENSIVE').upper()]
        
        config = AIGenerationConfig(
            provider=provider,
            strategy=strategy,
            coverage_target=data.get('config', {}).get('coverage_target', 0.95),
            include_edge_cases=data.get('config', {}).get('include_edge_cases', True),
            max_tests_per_function=data.get('config', {}).get('max_tests_per_function', 3)
        )
        
        # Generate tests
        generator = test_generator._ai_generators.get(provider)
        if not generator:
            return jsonify({'error': f'Provider {provider.value} not available'}), 400
        
        result = generator.generate(code, strategy)
        
        # Return generated test
        return jsonify({
            'test_code': result.test_code,
            'provider': result.ai_provider.value,
            'strategy': result.strategy.value,
            'coverage_targets': result.coverage_targets,
            'confidence_score': result.confidence_score,
            'estimated_effectiveness': result.estimated_effectiveness,
            'generation_time': result.generation_time
        }), 200
        
    except Exception as e:
        logger.error(f"AI test generation failed: {e}")
        return jsonify({'error': str(e)}), 500


@testing_api.route('/generate/self-healing', methods=['POST'])
def generate_self_healing_tests():
    """
    Generate self-healing tests that automatically fix themselves.
    
    Request body:
    {
        "code_path": "/path/to/code.py",
        "existing_test_path": "/path/to/test.py",
        "max_healing_iterations": 5
    }
    
    Returns:
        Self-healing test with healing history
    """
    try:
        data = request.get_json()
        code_path = data.get('code_path')
        existing_test_path = data.get('existing_test_path')
        max_iterations = data.get('max_healing_iterations', 5)
        
        if not code_path:
            return jsonify({'error': 'code_path required'}), 400
        
        # Generate self-healing test
        result = test_generator.generate_self_healing_tests(
            code_path=code_path,
            existing_test_path=existing_test_path,
            max_healing_iterations=max_iterations
        )
        
        return jsonify({
            'test_code': result.test_code,
            'confidence_score': result.confidence_score,
            'coverage_targets': result.coverage_targets,
            'provider': result.ai_provider.value,
            'self_healed': result.confidence_score < 0.8  # Indicates healing occurred
        }), 200
        
    except Exception as e:
        logger.error(f"Self-healing test generation failed: {e}")
        return jsonify({'error': str(e)}), 500


@testing_api.route('/generate/integration', methods=['POST'])
def generate_integration_tests():
    """
    Generate integration tests for cross-system validation.
    
    Request body:
    {
        "system_components": ["component1", "component2"],
        "api_endpoints": ["/api/v1/endpoint"],
        "complexity_level": "low|medium|high"
    }
    
    Returns:
        Generated integration tests
    """
    try:
        data = request.get_json()
        system_components = data.get('system_components', [])
        api_endpoints = data.get('api_endpoints')
        complexity_level = data.get('complexity_level', 'medium')
        
        # Generate integration tests
        tests = test_generator.generate_integration_tests(
            system_components=system_components,
            api_endpoints=api_endpoints,
            complexity_level=complexity_level
        )
        
        # Convert to JSON format
        tests_data = []
        for test in tests:
            tests_data.append({
                'name': test.name if hasattr(test, 'name') else 'integration_test',
                'test_type': test.test_type if hasattr(test, 'test_type') else 'integration',
                'description': test.description if hasattr(test, 'description') else '',
                'code': str(test)  # Simplified representation
            })
        
        return jsonify({
            'tests': tests_data,
            'test_count': len(tests_data),
            'components_covered': system_components,
            'endpoints_covered': api_endpoints or []
        }), 200
        
    except Exception as e:
        logger.error(f"Integration test generation failed: {e}")
        return jsonify({'error': str(e)}), 500


# === TEST EXECUTION ENDPOINTS ===

@testing_api.route('/execute', methods=['POST'])
def execute_tests():
    """
    Execute tests and return results.
    
    Request body:
    {
        "tests": [...],
        "parallel": true,
        "timeout": 60
    }
    
    Returns:
        Test execution results
    """
    try:
        data = request.get_json()
        tests = data.get('tests', [])
        parallel = data.get('parallel', True)
        timeout = data.get('timeout', 60)
        
        # Execute tests
        results = testing_hub.execute_test_suite(tests, parallel=parallel)
        
        # Convert results to JSON format
        results_data = []
        for result in results:
            results_data.append({
                'status': result.status if hasattr(result, 'status') else 'unknown',
                'execution_time': result.execution_time if hasattr(result, 'execution_time') else 0,
                'details': result.details if hasattr(result, 'details') else {}
            })
        
        # Calculate summary
        total = len(results_data)
        passed = sum(1 for r in results_data if r['status'] == 'passed')
        failed = sum(1 for r in results_data if r['status'] == 'failed')
        
        return jsonify({
            'results': results_data,
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'pass_rate': (passed / total * 100) if total > 0 else 0
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return jsonify({'error': str(e)}), 500


# === MONITORING INTEGRATION ENDPOINTS ===

@testing_api.route('/monitor/quality', methods=['POST'])
def monitor_test_quality():
    """
    Monitor test quality using AgentQualityAssurance.
    
    Request body:
    {
        "agent_id": "test_suite_1",
        "test_output": {...},
        "expected_quality": {...}
    }
    
    Returns:
        Quality assessment report
    """
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'unknown')
        test_output = data.get('test_output', {})
        expected_quality = data.get('expected_quality')
        
        # Assess quality
        report = monitoring_qa.assess_quality(
            agent_output=test_output,
            expected_output=expected_quality
        )
        
        return jsonify({
            'agent_id': agent_id,
            'overall_score': report.overall_score.score if hasattr(report, 'overall_score') else 0,
            'quality_level': 'high' if report.overall_score.score > 80 else 'medium' if report.overall_score.score > 60 else 'low',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Quality monitoring failed: {e}")
        return jsonify({'error': str(e)}), 500


@testing_api.route('/optimize', methods=['POST'])
def optimize_test_suite():
    """
    Optimize test suite using ML-powered analysis.
    
    Request body:
    {
        "test_results": [...],
        "optimization_strategy": "comprehensive|performance|reliability"
    }
    
    Returns:
        Optimization recommendations and prioritized test order
    """
    try:
        data = request.get_json()
        test_results = data.get('test_results', [])
        strategy = data.get('optimization_strategy', 'comprehensive')
        
        # Optimize test suite
        optimization = testing_hub.optimize_test_suite(test_results, strategy)
        
        return jsonify(optimization), 200
        
    except Exception as e:
        logger.error(f"Test optimization failed: {e}")
        return jsonify({'error': str(e)}), 500


@testing_api.route('/predict/failures', methods=['POST'])
def predict_test_failures():
    """
    Predict test failure probabilities using ML.
    
    Request body:
    {
        "test_identifiers": ["test1", "test2"],
        "historical_data": [...]
    }
    
    Returns:
        Failure predictions with confidence scores
    """
    try:
        data = request.get_json()
        test_identifiers = data.get('test_identifiers', [])
        historical_data = data.get('historical_data')
        
        # Predict failures
        predictions = testing_hub.predict_test_failures(
            test_identifiers=test_identifiers,
            historical_data=historical_data
        )
        
        return jsonify(predictions), 200
        
    except Exception as e:
        logger.error(f"Failure prediction failed: {e}")
        return jsonify({'error': str(e)}), 500


# === BATCH OPERATIONS ===

@testing_api.route('/batch/analyze', methods=['POST'])
def batch_analyze():
    """
    Perform batch analysis on multiple test suites.
    
    Request body:
    {
        "test_suites": [...],
        "analysis_types": ["coverage", "quality", "optimization"]
    }
    
    Returns:
        Batch analysis results
    """
    try:
        data = request.get_json()
        test_suites = data.get('test_suites', [])
        analysis_types = data.get('analysis_types', ['coverage'])
        
        results = []
        for suite in test_suites:
            suite_result = {
                'suite_id': suite.get('id', 'unknown'),
                'analyses': {}
            }
            
            # Perform requested analyses
            if 'coverage' in analysis_types:
                analysis = testing_hub.analyze_coverage(suite.get('test_results', []))
                suite_result['analyses']['coverage'] = {
                    'line_coverage': analysis.line_coverage if hasattr(analysis, 'line_coverage') else 0,
                    'branch_coverage': analysis.branch_coverage if hasattr(analysis, 'branch_coverage') else 0
                }
            
            if 'quality' in analysis_types:
                # Quality analysis would go here
                suite_result['analyses']['quality'] = {'score': 85.0}  # Placeholder
            
            if 'optimization' in analysis_types:
                optimization = testing_hub.optimize_test_suite(suite.get('test_results', []))
                suite_result['analyses']['optimization'] = optimization
            
            results.append(suite_result)
        
        return jsonify({
            'results': results,
            'batch_size': len(test_suites),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


# === REPORTING ENDPOINTS ===

@testing_api.route('/report/generate', methods=['POST'])
def generate_report():
    """
    Generate comprehensive testing report.
    
    Request body:
    {
        "report_type": "coverage|quality|comprehensive",
        "format": "json|html|pdf",
        "include_visualizations": true
    }
    
    Returns:
        Generated report in requested format
    """
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'comprehensive')
        format_type = data.get('format', 'json')
        include_viz = data.get('include_visualizations', False)
        
        # Generate report data
        report_data = {
            'report_type': report_type,
            'generated_at': datetime.now().isoformat(),
            'testing_status': testing_hub.get_testing_intelligence() if testing_hub else {},
            'capabilities': testing_hub.get_capabilities() if testing_hub else {}
        }
        
        # Return based on format
        if format_type == 'json':
            return jsonify(report_data), 200
        elif format_type == 'html':
            # Generate HTML report
            html = f"<html><body><h1>Testing Report</h1><pre>{json.dumps(report_data, indent=2)}</pre></body></html>"
            return html, 200, {'Content-Type': 'text/html'}
        else:
            return jsonify({'error': f'Format {format_type} not supported'}), 400
            
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return jsonify({'error': str(e)}), 500


# === API DOCUMENTATION ===

@testing_api.route('/docs', methods=['GET'])
def get_api_documentation():
    """
    Get API documentation for all testing endpoints.
    
    Returns:
        Complete API documentation with examples
    """
    docs = {
        'title': 'TestMaster Testing API Documentation',
        'version': '2.0.0',
        'base_url': '/api/v2/testing',
        'endpoints': [
            {
                'path': '/status',
                'method': 'GET',
                'description': 'Get testing infrastructure status',
                'response': 'Testing hub status and capabilities'
            },
            {
                'path': '/coverage/analyze',
                'method': 'POST',
                'description': 'Analyze test coverage with enhanced metrics',
                'request_body': {
                    'project_path': 'string',
                    'include_archive_features': 'boolean'
                },
                'response': 'Comprehensive coverage report'
            },
            {
                'path': '/generate/ai',
                'method': 'POST',
                'description': 'Generate tests using AI providers',
                'request_body': {
                    'code': 'string',
                    'provider': 'claude|gemini|universal',
                    'strategy': 'comprehensive|focused|security'
                },
                'response': 'Generated test code with metadata'
            },
            {
                'path': '/generate/self-healing',
                'method': 'POST',
                'description': 'Generate self-healing tests',
                'request_body': {
                    'code_path': 'string',
                    'max_healing_iterations': 'integer'
                },
                'response': 'Self-healing test code'
            },
            {
                'path': '/execute',
                'method': 'POST',
                'description': 'Execute tests and return results',
                'request_body': {
                    'tests': 'array',
                    'parallel': 'boolean'
                },
                'response': 'Test execution results'
            },
            {
                'path': '/optimize',
                'method': 'POST',
                'description': 'Optimize test suite using ML',
                'request_body': {
                    'test_results': 'array',
                    'optimization_strategy': 'string'
                },
                'response': 'Optimization recommendations'
            }
        ]
    }
    
    return jsonify(docs), 200


# Export blueprint and initialization function
__all__ = ['testing_api', 'init_testing_api']