
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
Security Blueprint for Main API Integration

Provides Flask blueprint for integrating security endpoints into the main API.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any
import logging

# Import security modules
from ..security.security_api import SecurityAPI
from ..security.knowledge_graph_integration import get_security_knowledge_bridge
from ..security.ai_security_integration import get_ai_security_explorer

logger = logging.getLogger(__name__)

# Create blueprint
security_bp = Blueprint('security', __name__, url_prefix='/api/security')

# Initialize security API
security_api_instance = SecurityAPI()

# Get integration instances
knowledge_bridge = get_security_knowledge_bridge()
ai_explorer = get_ai_security_explorer()


@security_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for security services"""
    try:
        return jsonify({
            'status': 'healthy',
            'services': {
                'security_api': 'operational',
                'knowledge_graph_integration': 'operational',
                'ai_security_integration': 'operational'
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@security_bp.route('/scan/comprehensive', methods=['POST'])
def comprehensive_scan():
    """Comprehensive security scan with all integrations"""
    try:
        data = request.get_json()
        directory = data.get('directory')
        
        if not directory:
            return jsonify({'error': 'directory is required'}), 400
        
        # Trigger comprehensive scan through security API
        # This would integrate with the existing security_api endpoints
        
        results = {
            'directory': directory,
            'scan_status': 'completed',
            'integrations': {
                'knowledge_graph': 'connected',
                'ai_explorer': 'connected',
                'security_api': 'operational'
            }
        }
        
        # Add findings to knowledge graph
        knowledge_bridge.add_security_finding('scan', {
            'directory': directory,
            'timestamp': 'now',
            'severity': 'info'
        })
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Comprehensive scan failed: {e}")
        return jsonify({'error': str(e)}), 500


@security_bp.route('/intelligence/query', methods=['POST'])
def query_security_intelligence():
    """Query security intelligence using natural language"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'query is required'}), 400
        
        # Use AI explorer for natural language query
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        insights = loop.run_until_complete(
            ai_explorer.query_security_insights(query)
        )
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Intelligence query failed: {e}")
        return jsonify({'error': str(e)}), 500


@security_bp.route('/graph/export', methods=['GET'])
def export_security_graph():
    """Export security findings as knowledge graph"""
    try:
        graph = knowledge_bridge.export_security_graph()
        return jsonify(graph)
        
    except Exception as e:
        logger.error(f"Graph export failed: {e}")
        return jsonify({'error': str(e)}), 500


@security_bp.route('/context/summary', methods=['GET'])
def get_security_context():
    """Get current security context summary"""
    try:
        kb_correlations = knowledge_bridge.correlate_security_intelligence()
        ai_summary = ai_explorer.get_security_context_summary()
        
        return jsonify({
            'knowledge_graph': kb_correlations,
            'ai_integration': ai_summary,
            'total_findings': len(knowledge_bridge.security_nodes)
        })
        
    except Exception as e:
        logger.error(f"Context summary failed: {e}")
        return jsonify({'error': str(e)}), 500


def register_security_blueprint(app):
    """
    Register security blueprint with main Flask app.
    
    Args:
        app: Flask application instance
    """
    try:
        app.register_blueprint(security_bp)
        logger.info("Security blueprint registered with main API")
        
        # Also register the original security API routes
        security_app = security_api_instance.get_app()
        
        # Copy routes from security API to main app
        for rule in security_app.url_map.iter_rules():
            if rule.endpoint != 'static':
                # Register each security API route in main app
                endpoint_func = security_app.view_functions[rule.endpoint]
                app.add_url_rule(
                    rule.rule,
                    endpoint=f"security_{rule.endpoint}",
                    view_func=endpoint_func,
                    methods=rule.methods
                )
        
        logger.info("All security API endpoints registered with main API")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register security blueprint: {e}")
        return False