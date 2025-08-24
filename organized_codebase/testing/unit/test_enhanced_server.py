#!/usr/bin/env python3
"""Test server for enhanced dashboard"""

from flask import Flask, jsonify
import glob
import os

app = Flask(__name__)

@app.route('/')
def home():
    """Serve enhanced dashboard"""
    with open('enhanced_dashboard.html', 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

@app.route('/api/llm/list-modules')
def list_modules():
    """List Python modules"""
    modules = []
    for pattern in ['*.py', 'testmaster/**/*.py']:
        for file in glob.glob(pattern, recursive=True):
            if '__pycache__' not in file and '.pyc' not in file:
                modules.append(file.replace(os.sep, '/'))
    return jsonify(sorted(modules)[:100])

@app.route('/api/llm/estimate-cost', methods=['POST'])
def estimate_cost():
    """Estimate cost for analysis"""
    from flask import request
    data = request.get_json()
    module_path = data.get('module_path')
    
    if not module_path or not os.path.exists(module_path):
        return jsonify({'error': 'Module not found'}), 404
    
    with open(module_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    estimated_tokens = len(content) // 4 + 500
    estimated_cost = (estimated_tokens / 1000) * 0.00025 * 2
    
    return jsonify({
        'module_path': module_path,
        'file_size_bytes': len(content),
        'estimated_tokens': estimated_tokens,
        'estimated_cost_usd': round(estimated_cost, 4),
        'model': 'gemini-2.5-pro'
    })

@app.route('/api/llm/metrics')
def llm_metrics():
    """Return dummy LLM metrics"""
    return jsonify({
        'api_calls': {'total_calls': 0, 'success_rate': 0.0},
        'token_usage': {'total_tokens': 0},
        'cost_tracking': {'total_cost_estimate': 0.0},
        'analysis_status': {'gemini_available': True, 'completed_analyses': 0}
    })

@app.route('/api/metrics')
def metrics():
    """Return dummy metrics"""
    from datetime import datetime
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'system': {'cpu_usage': 0.0, 'memory_usage': 0.0},
        'components': {'active_agents': 0, 'active_bridges': 0},
        'workflow': {'consensus_decisions': 0},
        'security': {'security_alerts': 0}
    })

if __name__ == '__main__':
    print('Starting enhanced dashboard test server on port 8082...')
    print('Access at: http://localhost:8082')
    print('Module Analyzer tab should be available')
    app.run(port=8082, debug=False)