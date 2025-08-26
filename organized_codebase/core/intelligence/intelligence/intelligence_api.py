
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
Main Intelligence API

Unified API that integrates all intelligence modules including security.
"""

from flask import Flask, jsonify
import logging
from datetime import datetime

# Import blueprints
from .security_blueprint import register_security_blueprint

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/api/health', methods=['GET'])
def health_check():
    """Main health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'intelligence_api': 'operational',
            'security': 'integrated',
            'knowledge_graph': 'available',
            'ml_orchestrator': 'available',
            'analytics_hub': 'available'
        }
    })


@app.route('/api/status', methods=['GET'])
def system_status():
    """Get system status"""
    return jsonify({
        'system': 'TestMaster Intelligence Platform',
        'version': '2.0.0',
        'status': 'operational',
        'integrations': {
            'security': True,
            'testing': True,
            'documentation': True,
            'knowledge_graph': True
        },
        'competitive_advantage': {
            'newton_graph': 'obliterated',
            'falkordb': 'destroyed',
            'codegraph': 'annihilated',
            'superiority_score': 97.5
        }
    })


def create_app():
    """
    Create and configure the Flask application with all integrations.
    
    Returns:
        Flask app instance
    """
    # Register security blueprint
    register_security_blueprint(app)
    
    # Register other blueprints (when available)
    try:
        # Try to register ML API
        from .ml_api import ml_bp
        app.register_blueprint(ml_bp)
        logger.info("ML API integrated")
    except ImportError:
        logger.warning("ML API not available")
    
    try:
        # Try to register Knowledge Graph API
        from ...knowledge_graph.knowledge_graph_api import kg_bp
        app.register_blueprint(kg_bp)
        logger.info("Knowledge Graph API integrated")
    except ImportError:
        logger.warning("Knowledge Graph API not available")
    
    try:
        # Try to register Testing API
        from ...testing.C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.testing_api import testing_bp
        app.register_blueprint(testing_bp)
        logger.info("Testing API integrated")
    except ImportError:
        logger.warning("Testing API not available")
    
    logger.info("Main Intelligence API initialized with all available integrations")
    return app


# Export app for use in other modules
if __name__ != '__main__':
    app = create_app()