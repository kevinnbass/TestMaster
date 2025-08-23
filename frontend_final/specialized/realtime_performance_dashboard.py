#!/usr/bin/env python3
"""
Real-Time Performance Monitoring Dashboard - STEELCLAD Clean Version
====================================================================

Agent Y STEELCLAD Protocol: Modularized version of realtime_performance_dashboard.py
Reduced from 841 lines to <200 lines by extracting core functionality

This streamlined version coordinates modular components:
- PerformanceMonitorCore: Core metrics and monitoring logic
- performance_dashboard.html: Frontend template with real-time UI
- Flask routing and WebSocket coordination

Author: Agent Y - STEELCLAD Modularization Specialist
"""

import os
import sys
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# Import modularized performance monitoring core
try:
    from .performance_monitor_core import performance_core, PERFORMANCE_ENGINE_AVAILABLE
except ImportError:
    # Fallback to absolute import
    from performance_monitor_core import performance_core, PERFORMANCE_ENGINE_AVAILABLE

# Flask application setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'beta_performance_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Template path for modularized HTML
TEMPLATE_DIR = Path(__file__).parent / 'templates'

@app.route('/')
def index():
    """Serve the real-time performance dashboard"""
    try:
        template_path = TEMPLATE_DIR / 'performance_dashboard.html'
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return "<h1>Performance Dashboard Template Not Found</h1><p>Looking for: " + str(template_path) + "</p>"
    except Exception as e:
        return f"<h1>Template Error</h1><p>{str(e)}</p>"

@app.route('/performance-metrics')
def get_performance_metrics():
    """Get current performance metrics via modular core"""
    try:
        return jsonify(performance_core.get_performance_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trigger-optimization', methods=['POST'])
def trigger_optimization():
    """Trigger system optimization via modular core"""
    try:
        data = request.get_json() or {}
        optimization_type = data.get('type', 'general')
        
        result = performance_core.trigger_optimization(optimization_type)
        
        # Emit optimization started event via WebSocket
        if result['success']:
            socketio.emit('optimization_started', {
                'id': result['optimization_id'],
                'type': optimization_type,
                'message': result['message']
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket event handlers for real-time connectivity
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected to Performance Dashboard")
    emit('connection_established', {
        'message': 'Connected to real-time performance monitoring',
        'engine_status': 'Active' if PERFORMANCE_ENGINE_AVAILABLE else 'Simulated Mode'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected from Performance Dashboard")

def background_performance_monitor():
    """Background thread for continuous performance monitoring and WebSocket updates"""
    import time
    
    while True:
        try:
            # Update real-time performance data via modular core
            performance_core.update_realtime_data(socketio)
            
            # Sleep for update interval
            time.sleep(2)
            
        except Exception as e:
            print(f"Background monitoring error: {e}")
            time.sleep(5)  # Wait longer on error

def main():
    """Launch the real-time performance dashboard"""
    print("STARTING Real-Time Performance Monitoring Dashboard (STEELCLAD)")
    print("=" * 70)
    print("Agent Y STEELCLAD Protocol: Modularized Performance Monitor")
    print(f"Core Engine: {'Active' if PERFORMANCE_ENGINE_AVAILABLE else 'Simulated Mode'}")
    print(f"Template Location: {TEMPLATE_DIR}")
    print("=" * 70)
    print()
    print("Dashboard URL: http://localhost:5001")
    print()
    
    # Start background monitoring thread
    monitor_thread = threading.Thread(target=background_performance_monitor, daemon=True)
    monitor_thread.start()
    print("Background performance monitoring thread started")
    
    # Run the Flask application with WebSocket support
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=False)
    except Exception as e:
        print(f"Dashboard startup error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()