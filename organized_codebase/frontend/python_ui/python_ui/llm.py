"""
LLM Integration API Module
==========================

Handles LLM status, toggle, and analysis endpoints.
Critical for the LLM toggle button functionality.

Author: TestMaster Team
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import logging
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from dashboard.dashboard_core.error_handler import (
        enhanced_api_endpoint, ValidationError, handle_api_error
    )
except ImportError:
    # Fallback decorators
    def enhanced_api_endpoint(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def handle_api_error(func):
        return func
    class ValidationError(Exception):
        pass

logger = logging.getLogger(__name__)

# Create blueprint  
llm_bp = Blueprint('llm', __name__)

# Global state (will be injected by server)
llm_monitor = None
llm_api_enabled = False


def init_llm_api(monitor=None, api_enabled=False):
    """
    Initialize the LLM API with required dependencies.
    
    Args:
        monitor: LLM monitor instance (optional)
        api_enabled (bool): Initial LLM API state
    """
    global llm_monitor, llm_api_enabled
    llm_monitor = monitor
    llm_api_enabled = api_enabled
    logger.info(f"LLM API initialized, enabled: {api_enabled}")


@llm_bp.route('/status')
@handle_api_error
def get_llm_status():
    """
    Get current LLM status.
    
    Returns LLM availability, API status, and usage metrics.
    Critical for the LLM toggle button state.
    
    Returns:
        JSON: LLM status information
        
    Example Response:
        {
            "status": "success",
            "llm_available": true,
            "api_enabled": false,
            "provider": "gemini",
            "usage": {
                "calls_today": 42,
                "tokens_used": 15420
            }
        }
    """
    # Basic status
    status_data = {
        'llm_available': llm_monitor is not None,
        'api_enabled': llm_api_enabled,
        'provider': 'gemini',
        'demo_mode': not llm_api_enabled,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add monitor data if available
    if llm_monitor:
        try:
            usage = llm_monitor.get_usage_stats()
            status_data['usage'] = {
                'calls_today': usage.get('calls_today', 0),
                'tokens_used': usage.get('tokens_used', 0),
                'last_call': usage.get('last_call'),
                'error_rate': usage.get('error_rate', 0.0)
            }
            
            # Model info
            status_data['model_info'] = {
                'name': usage.get('model_name', 'gemini-2.5-pro'),
                'version': usage.get('model_version', '2.5'),
                'max_tokens': usage.get('max_tokens', 32000)
            }
            
        except Exception as e:
            logger.warning(f"Could not get LLM usage stats: {e}")
            status_data['usage'] = {
                'calls_today': 0,
                'tokens_used': 0,
                'error_rate': 0.0
            }
    else:
        status_data['usage'] = {
            'calls_today': 0,
            'tokens_used': 0,
            'error_rate': 0.0
        }
        status_data['message'] = 'LLM monitor not available'
    
    return jsonify({
        'status': 'success',
        **status_data
    })


@llm_bp.route('/toggle-mode', methods=['POST'])
@enhanced_api_endpoint()
def toggle_llm_mode():
    """
    Toggle LLM API mode on/off.
    
    Critical for the LLM toggle button functionality.
    
    Request Body:
        enabled (bool): New LLM API state
        
    Returns:
        JSON: Toggle result
    """
    global llm_api_enabled
    
    try:
        data = request.get_json() or {}
        new_state = data.get('enabled')
        
        if new_state is None:
            return jsonify({
                'status': 'error',
                'error': 'enabled parameter is required',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Update global state
        old_state = llm_api_enabled
        llm_api_enabled = bool(new_state)
        
        # Update monitor if available
        if llm_monitor and hasattr(llm_monitor, 'set_api_enabled'):
            llm_monitor.set_api_enabled(llm_api_enabled)
        
        logger.info(f"LLM API toggled: {old_state} -> {llm_api_enabled}")
        
        return jsonify({
            'status': 'success',
            'enabled': llm_api_enabled,
            'previous_state': old_state,
            'message': f'LLM API {"enabled" if llm_api_enabled else "disabled"}',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error toggling LLM mode: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@llm_bp.route('/metrics')
def get_llm_metrics():
    """
    Get LLM usage metrics and statistics.
    
    Returns:
        JSON: LLM metrics
    """
    try:
        if llm_monitor is None:
            return jsonify({
                'status': 'success',
                'metrics': {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'average_response_time': 0,
                    'tokens_consumed': 0,
                    'cost_estimate': 0.0
                },
                'message': 'LLM monitor not available',
                'timestamp': datetime.now().isoformat()
            })
        
        # Get comprehensive metrics
        metrics = llm_monitor.get_comprehensive_metrics()
        
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'api_enabled': llm_api_enabled,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting LLM metrics: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@llm_bp.route('/analyze', methods=['POST'])
def analyze_with_llm():
    """
    Perform LLM-based analysis.
    
    Request Body:
        module_path (str): Path to module to analyze
        analysis_type (str): Type of analysis (code_review, test_generation, etc.)
        
    Returns:
        JSON: Analysis results
    """
    try:
        if not llm_api_enabled:
            return jsonify({
                'status': 'error',
                'error': 'LLM API is not enabled. Toggle LLM mode first.',
                'timestamp': datetime.now().isoformat()
            }), 403
        
        data = request.get_json() or {}
        module_path = data.get('module_path')
        analysis_type = data.get('analysis_type', 'code_review')
        
        if not module_path:
            return jsonify({
                'status': 'error',
                'error': 'module_path is required',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        if llm_monitor is None:
            return jsonify({
                'status': 'error',
                'error': 'LLM monitor not available',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Perform analysis
        result = llm_monitor.analyze_module(module_path, analysis_type)
        
        return jsonify({
            'status': 'success',
            'analysis': result,
            'module_path': module_path,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error performing LLM analysis: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@llm_bp.route('/estimate-cost', methods=['POST'])
def estimate_cost():
    """
    Estimate cost for LLM analysis.
    
    Request Body:
        operation (str): Type of operation
        input_size (int): Size of input in characters
        
    Returns:
        JSON: Cost estimate
    """
    try:
        data = request.get_json() or {}
        operation = data.get('operation', 'analysis')
        input_size = data.get('input_size', 0)
        
        # Basic cost estimation (can be made more sophisticated)
        tokens_estimate = input_size // 4  # Rough estimate: 4 chars per token
        cost_per_token = 0.00002  # Example cost
        estimated_cost = tokens_estimate * cost_per_token
        
        return jsonify({
            'status': 'success',
            'estimate': {
                'operation': operation,
                'input_size': input_size,
                'estimated_tokens': tokens_estimate,
                'estimated_cost': round(estimated_cost, 6),
                'currency': 'USD'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error estimating LLM cost: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@llm_bp.route('/toggle', methods=['POST'])
def toggle_llm_api():
    """
    Toggle LLM API on/off from frontend.
    
    Request Body:
        enabled (bool): Whether to enable or disable LLM API
        
    Returns:
        JSON: Toggle result and current status
    """
    global llm_api_enabled
    
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', False)
        
        # Validate input
        if not isinstance(enabled, bool):
            return jsonify({
                'status': 'error',
                'error': 'enabled field must be a boolean',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Update global state
        llm_api_enabled = enabled
        
        logger.info(f"LLM API {'enabled' if enabled else 'disabled'} via frontend toggle")
        
        return jsonify({
            'status': 'success',
            'llm_api_enabled': llm_api_enabled,
            'message': f'LLM API {"enabled" if enabled else "disabled"}',
            'timestamp': datetime.now().isoformat(),
            'safety_note': 'LLM calls are disabled by default for security. Toggle only when needed.'
        })
        
    except Exception as e:
        logger.error(f"Error toggling LLM API: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@llm_bp.route('/safety-status', methods=['GET'])
def get_llm_safety_status():
    """
    Get LLM safety and enablement status for frontend display.
    
    Returns:
        JSON: Safety status and toggle state
    """
    try:
        return jsonify({
            'status': 'success',
            'llm_api_enabled': llm_api_enabled,
            'safety_features': {
                'disabled_by_default': True,
                'frontend_toggle_required': True,
                'no_auto_calls': True,
                'user_controlled': True
            },
            'current_state': 'enabled' if llm_api_enabled else 'disabled',
            'toggle_available': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting LLM safety status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500