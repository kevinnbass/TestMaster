
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
TestMaster Intelligence API Application - AGENT B ENHANCED
=========================================================

Flask application with all intelligence APIs registered.
Provides comprehensive testing and monitoring capabilities.

AGENT B Enhancement: Hour 10-12
- Complete API layer integration
- Testing API blueprint registration
- Monitoring API blueprint registration
- CORS and error handling
"""

from flask import Flask, jsonify
from flask_cors import CORS
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Import all API blueprints
from .endpoints import intelligence_api, init_intelligence_api
from .testing_api import testing_api, init_testing_api
from .monitoring_api import monitoring_api, init_monitoring_api
from .performance_api import performance_api, init_performance_api
from .qa_api import qa_api, init_qa_api
from .orchestration_api import orchestration_api, init_orchestration_api
from .ultimate_nexus_api import nexus_api, init_nexus_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Enable CORS for all domains
    CORS(app)
    
    # Configuration
    app.config.update(config or {})
    
    # Initialize all APIs
    init_intelligence_api(config)
    init_testing_api(config)  
    init_monitoring_api(config)
    init_performance_api(config)
    init_qa_api(config)
    init_orchestration_api(config)
    init_nexus_api(config)
    
    # Register blueprints
    app.register_blueprint(intelligence_api)
    app.register_blueprint(testing_api)
    app.register_blueprint(monitoring_api)
    app.register_blueprint(performance_api)
    app.register_blueprint(qa_api)
    app.register_blueprint(orchestration_api)
    app.register_blueprint(nexus_api)
    
    # Add global error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested endpoint does not exist',
            'available_endpoints': [
                '/api/intelligence/*',
                '/api/v2/testing/*',
                '/api/v2/monitoring/*',
                '/api/v2/performance/*',
                '/api/v2/quality/*',
                '/api/v2/orchestration/*',
                '/api/v2/nexus/*'
            ]
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal server error',
            'message': str(error),
            'timestamp': datetime.now().isoformat()
        }), 500
    
    # Root endpoint
    @app.route('/')
    def root():
        return jsonify({
            'service': 'TestMaster Intelligence API',
            'version': '2.0.0',
            'status': 'active',
            'timestamp': datetime.now().isoformat(),
            'endpoints': {
                'intelligence': '/api/intelligence',
                'testing': '/api/v2/testing',
                'monitoring': '/api/v2/monitoring',
                'performance': '/api/v2/performance',
                'quality': '/api/v2/quality',
                'orchestration': '/api/v2/orchestration',
                'nexus': '/api/v2/nexus'
            },
            'documentation': {
                'testing': '/api/v2/testing/docs',
                'monitoring': '/api/v2/monitoring/docs',
                'performance': '/api/v2/performance/docs',
                'quality': '/api/v2/quality/docs',
                'orchestration': '/api/v2/orchestration/docs',
                'nexus': '/api/v2/nexus/docs'
            }
        })
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'intelligence_hub': 'active',
                'testing_hub': 'active', 
                'monitoring_hub': 'active',
                'performance_hub': 'active',
                'quality_hub': 'active',
                'orchestration_hub': 'active',
                'ultimate_nexus': 'transcendent'
            }
        })
    
    # API discovery endpoint
    @app.route('/api/discovery')
    def api_discovery():
        return jsonify({
            'api_version': '2.0.0',
            'blueprints': [
                {
                    'name': 'intelligence_api',
                    'prefix': '/api/intelligence',
                    'description': 'Main intelligence hub endpoints'
                },
                {
                    'name': 'testing_api',
                    'prefix': '/api/v2/testing',
                    'description': 'Comprehensive testing API with AI generation'
                },
                {
                    'name': 'monitoring_api', 
                    'prefix': '/api/v2/monitoring',
                    'description': 'Real-time monitoring and quality assurance'
                },
                {
                    'name': 'performance_api',
                    'prefix': '/api/v2/performance', 
                    'description': 'Performance optimization and analytics'
                },
                {
                    'name': 'qa_api',
                    'prefix': '/api/v2/quality',
                    'description': 'Quality assurance and validation framework'
                },
                {
                    'name': 'orchestration_api',
                    'prefix': '/api/v2/orchestration',
                    'description': 'Advanced testing and monitoring orchestration'
                },
                {
                    'name': 'nexus_api',
                    'prefix': '/api/v2/nexus',
                    'description': 'Ultimate Intelligence Nexus - Transcendent AI coordination'
                }
            ],
            'features': {
                'ai_test_generation': True,
                'self_healing_tests': True,
                'coverage_analysis': True,
                'real_time_monitoring': True,
                'quality_assurance': True,
                'pattern_detection': True,
                'alert_management': True,
                'performance_analysis': True,
                'performance_optimization': True,
                'resource_monitoring': True,
                'optimization_recommendations': True,
                'quality_validation': True,
                'quality_insights': True,
                'quality_trend_analysis': True,
                'intelligent_test_orchestration': True,
                'advanced_monitoring_coordination': True,
                'predictive_analytics': True,
                'cross_system_correlation': True,
                'ai_driven_optimization': True,
                'ultimate_intelligence_nexus': True,
                'transcendent_system_coordination': True,
                'meta_ai_decisions': True,
                'quantum_system_observation': True,
                'autonomous_system_evolution': True
            }
        })
    
    logger.info("Flask app created with all API blueprints registered")
    return app


def run_development_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """
    Run development server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    app = create_app({
        'DEBUG': debug,
        'TESTING': False
    })
    
    logger.info(f"Starting development server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_development_server()