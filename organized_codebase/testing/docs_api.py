
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
Documentation REST API

REST endpoints for documentation generation and management.
"""

from flask import Flask, request, jsonify, Response
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import logging

from .doc_orchestrator import DocumentationOrchestrator
from .interactive_docs import InteractiveDocumentation
from .markdown_generator import MarkdownGenerator
from .api_spec_builder import APISpecBuilder
from .docstring_analyzer import DocstringAnalyzer
from .changelog_generator import ChangelogGenerator
from .metrics_reporter import MetricsReporter

logger = logging.getLogger(__name__)


class DocumentationAPI:
    """
    REST API for documentation services.
    Provides endpoints for all documentation functionality.
    """
    
    def __init__(self, app: Optional[Flask] = None):
        """
        Initialize documentation API.
        
        Args:
            app: Flask application instance
        """
        self.app = app or Flask(__name__)
        self.orchestrator = DocumentationOrchestrator()
        self.interactive_docs = InteractiveDocumentation()
        self.markdown_gen = MarkdownGenerator()
        self.api_builder = APISpecBuilder()
        self.docstring_analyzer = DocstringAnalyzer()
        self.changelog_gen = ChangelogGenerator()
        self.metrics_reporter = MetricsReporter()
        
        self._register_routes()
        logger.info("Documentation API initialized")
        
    def _register_routes(self) -> None:
        """Register API routes."""
        
        @self.app.route('/api/docs/generate', methods=['POST'])
        def generate_docs():
            """Generate complete documentation for a project."""
            try:
                data = request.get_json()
                project_path = data.get('project_path')
                
                if not project_path:
                    return jsonify({'error': 'project_path is required'}), 400
                    
                if not os.path.exists(project_path):
                    return jsonify({'error': 'Project path does not exist'}), 404
                    
                result = self.orchestrator.generate_complete_documentation(project_path)
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error generating documentation: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/quality', methods=['GET'])
        def check_quality():
            """Check documentation quality."""
            try:
                project_path = request.args.get('project_path')
                
                if not project_path:
                    return jsonify({'error': 'project_path parameter is required'}), 400
                    
                quality_report = self.orchestrator.monitor_documentation_quality(project_path)
                return jsonify({
                    'overall_score': quality_report.overall_score,
                    'coverage_percentage': quality_report.coverage_percentage,
                    'issues_found': quality_report.issues_found,
                    'recommendations': quality_report.recommendations
                })
                
            except Exception as e:
                logger.error(f"Error checking quality: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/api-spec', methods=['POST'])
        def generate_api_spec():
            """Generate OpenAPI specification."""
            try:
                data = request.get_json()
                project_path = data.get('project_path')
                title = data.get('title', 'API Documentation')
                version = data.get('version', '1.0.0')
                
                if not project_path:
                    return jsonify({'error': 'project_path is required'}), 400
                    
                endpoints = self.api_builder.scan_directory(project_path)
                spec = self.api_builder.build_openapi_spec(title, version)
                
                return jsonify({
                    'spec': spec,
                    'endpoints_discovered': len(endpoints)
                })
                
            except Exception as e:
                logger.error(f"Error generating API spec: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/interactive', methods=['POST'])
        def create_interactive_docs():
            """Create interactive documentation."""
            try:
                data = request.get_json()
                project_path = data.get('project_path')
                base_url = data.get('base_url', 'http://localhost:5000')
                
                if not project_path:
                    return jsonify({'error': 'project_path is required'}), 400
                    
                self.interactive_docs.base_url = base_url
                spec = self.interactive_docs.generate_interactive_spec(project_path)
                html_docs = self.interactive_docs.generate_interactive_html(spec)
                
                return jsonify({
                    'spec': spec,
                    'html_documentation': html_docs,
                    'endpoints_count': len(spec.get('x-endpoints', []))
                })
                
            except Exception as e:
                logger.error(f"Error creating interactive docs: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/test-endpoint', methods=['POST'])
        def test_endpoint():
            """Test an API endpoint."""
            try:
                data = request.get_json()
                endpoint = data.get('endpoint')
                method = data.get('method', 'GET')
                parameters = data.get('parameters', {})
                headers = data.get('headers', {})
                body = data.get('body')
                
                if not endpoint:
                    return jsonify({'error': 'endpoint is required'}), 400
                    
                test_result = self.interactive_docs.test_endpoint(
                    endpoint, method, parameters, headers, body
                )
                
                return jsonify({
                    'endpoint': test_result.endpoint,
                    'method': test_result.method,
                    'success': test_result.success,
                    'response_time': test_result.response_time,
                    'parameters': test_result.parameters
                })
                
            except Exception as e:
                logger.error(f"Error testing endpoint: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/markdown', methods=['POST'])
        def generate_markdown():
            """Generate markdown documentation."""
            try:
                data = request.get_json()
                title = data.get('title', 'Documentation')
                description = data.get('description', '')
                sections = data.get('sections', [])
                
                for section in sections:
                    self.markdown_gen.add_section(
                        section.get('title', ''),
                        section.get('content', ''),
                        section.get('level', 2)
                    )
                    
                markdown_content = self.markdown_gen.generate_document(title, description)
                
                return jsonify({
                    'markdown': markdown_content,
                    'sections_added': len(sections)
                })
                
            except Exception as e:
                logger.error(f"Error generating markdown: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/changelog', methods=['POST'])
        def generate_changelog():
            """Generate changelog from git history."""
            try:
                data = request.get_json()
                since = data.get('since')
                until = data.get('until', 'HEAD')
                format_type = data.get('format', 'markdown')
                
                commits = self.changelog_gen.parse_commits(since, until)
                changelog = self.changelog_gen.generate_changelog(format_type)
                
                return jsonify({
                    'changelog': changelog,
                    'commits_processed': len(commits),
                    'format': format_type
                })
                
            except Exception as e:
                logger.error(f"Error generating changelog: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/metrics', methods=['GET'])
        def get_metrics():
            """Get documentation metrics."""
            try:
                project_path = request.args.get('project_path')
                
                if not project_path:
                    return jsonify({'error': 'project_path parameter is required'}), 400
                    
                metrics = self.metrics_reporter.analyze_project(project_path)
                orchestrator_metrics = self.orchestrator.get_documentation_metrics()
                
                return jsonify({
                    'project_metrics': metrics,
                    'orchestrator_metrics': orchestrator_metrics,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/docstring-analysis', methods=['POST'])
        def analyze_docstrings():
            """Analyze docstring quality."""
            try:
                data = request.get_json()
                file_path = data.get('file_path')
                style = data.get('style', 'google')
                
                if not file_path:
                    return jsonify({'error': 'file_path is required'}), 400
                    
                self.docstring_analyzer.style = getattr(self.docstring_analyzer.style.__class__, style.upper())
                analysis = self.docstring_analyzer.analyze_file(file_path)
                
                return jsonify({
                    'analysis': {
                        name: {
                            'score': result.score,
                            'issues_count': len(result.issues),
                            'suggestions_count': len(result.suggestions)
                        }
                        for name, result in analysis.items()
                    },
                    'file_path': file_path,
                    'style': style
                })
                
            except Exception as e:
                logger.error(f"Error analyzing docstrings: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/auto-update', methods=['POST'])
        def auto_update_docs():
            """Auto-update documentation for changed files."""
            try:
                data = request.get_json()
                project_path = data.get('project_path')
                changed_files = data.get('changed_files', [])
                
                if not project_path:
                    return jsonify({'error': 'project_path is required'}), 400
                    
                update_result = self.orchestrator.auto_update_documentation(project_path, changed_files)
                
                return jsonify(update_result)
                
            except Exception as e:
                logger.error(f"Error auto-updating docs: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/search', methods=['GET'])
        def search_documentation():
            """Search documentation content."""
            try:
                query = request.args.get('query')
                project_path = request.args.get('project_path')
                
                if not query or not project_path:
                    return jsonify({'error': 'query and project_path parameters are required'}), 400
                    
                # Simple search implementation
                results = []
                # This would be expanded with actual search functionality
                
                return jsonify({
                    'query': query,
                    'results': results,
                    'total_results': len(results)
                })
                
            except Exception as e:
                logger.error(f"Error searching documentation: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/docs/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            try:
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'services': {
                        'orchestrator': 'operational',
                        'interactive_docs': 'operational',
                        'markdown_generator': 'operational',
                        'api_builder': 'operational',
                        'docstring_analyzer': 'operational',
                        'changelog_generator': 'operational',
                        'metrics_reporter': 'operational'
                    }
                })
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
                
    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False) -> None:
        """
        Run the documentation API server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Debug mode
        """
        self.app.run(host=host, port=port, debug=debug)
        
    def get_app(self) -> Flask:
        """Get Flask application instance."""
        return self.app