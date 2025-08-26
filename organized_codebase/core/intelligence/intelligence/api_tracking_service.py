"""
API Tracking Service - Flask Integration
Agent Alpha Implementation - Hours 0-2
"""

from flask import Flask, jsonify, request
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.monitoring.api_usage_tracker import APIUsageTracker, api_tracker
import json
from datetime import datetime

app = Flask(__name__)

# Initialize tracker
tracker = APIUsageTracker()

@app.route('/api/usage/status', methods=['GET'])
def get_budget_status():
    """Get current budget status and warnings."""
    try:
        status = tracker.check_budget_status()
        return jsonify({
            'success': True,
            'data': status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/usage/analytics', methods=['GET'])
def get_usage_analytics():
    """Get comprehensive usage analytics."""
    try:
        days = request.args.get('days', 7, type=int)
        analytics = tracker.get_usage_analytics(days)
        return jsonify({
            'success': True,
            'data': analytics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/usage/pre-call-check', methods=['POST'])
def pre_call_budget_check():
    """Check if API call is within budget before execution."""
    try:
        data = request.get_json()
        estimated_tokens = data.get('estimated_tokens', 1000)
        model = data.get('model', 'claude-sonnet-4')
        
        check_result = tracker.pre_call_budget_check(estimated_tokens, model)
        return jsonify({
            'success': True,
            'data': check_result,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/usage/log-call', methods=['POST'])
def log_api_call():
    """Log an API call manually."""
    try:
        data = request.get_json()
        
        call_id = tracker.log_api_call(
            model=data.get('model', 'unknown'),
            provider=data.get('provider', 'unknown'),
            purpose=data.get('purpose', 'manual_log'),
            input_tokens=data.get('input_tokens', 0),
            output_tokens=data.get('output_tokens', 0),
            request_data=data.get('request_data'),
            response_data=data.get('response_data'),
            execution_time=data.get('execution_time', 0.0),
            success=data.get('success', True),
            error_message=data.get('error_message')
        )
        
        return jsonify({
            'success': True,
            'data': {'call_id': call_id},
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/usage/budget', methods=['POST'])
def update_budget():
    """Update daily budget limit."""
    try:
        data = request.get_json()
        new_budget = data.get('budget', 50.0)
        
        tracker.daily_budget = new_budget
        
        return jsonify({
            'success': True,
            'data': {'new_budget': new_budget},
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/usage/export', methods=['GET'])
def export_usage_report():
    """Export comprehensive usage report."""
    try:
        report_path = tracker.export_usage_report()
        return jsonify({
            'success': True,
            'data': {'report_path': report_path},
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/usage/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get all data needed for usage dashboard."""
    try:
        # Get current status
        status = tracker.check_budget_status()
        
        # Get 7-day analytics
        analytics = tracker.get_usage_analytics(7)
        
        # Calculate additional metrics
        dashboard_data = {
            'budget_status': status,
            'analytics': analytics,
            'quick_stats': {
                'calls_today': status['call_count'],
                'cost_today': status['total_spent'],
                'budget_remaining': status['remaining_budget'],
                'most_used_model': analytics['model_breakdown'][0]['model'] if analytics['model_breakdown'] else 'None',
                'total_calls_week': analytics['total_stats']['total_calls'] or 0,
                'total_cost_week': analytics['total_stats']['total_cost'] or 0.0,
                'avg_cost_per_call': (analytics['total_stats']['total_cost'] / analytics['total_stats']['total_calls']) if analytics['total_stats']['total_calls'] > 0 else 0.0
            },
            'alerts': []
        }
        
        # Add alerts based on budget status
        if status['status'] == 'EXCEEDED':
            dashboard_data['alerts'].append({
                'level': 'danger',
                'message': 'Daily budget exceeded! API calls should be stopped.',
                'timestamp': datetime.now().isoformat()
            })
        elif status['status'] == 'CRITICAL':
            dashboard_data['alerts'].append({
                'level': 'warning',
                'message': f'Critical budget usage: {status["budget_used_percentage"]:.1f}% used',
                'timestamp': datetime.now().isoformat()
            })
        elif status['status'] == 'WARNING':
            dashboard_data['alerts'].append({
                'level': 'info',
                'message': f'High budget usage: {status["budget_used_percentage"]:.1f}% used',
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'success': True,
            'data': dashboard_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/usage/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'success': True,
        'service': 'api_tracking_service',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)