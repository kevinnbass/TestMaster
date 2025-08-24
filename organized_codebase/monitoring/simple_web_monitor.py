#!/usr/bin/env python3
"""
Simple Web Monitor - No LLM API Calls

A basic web monitoring server that serves the dashboard without any 
LLM analysis or API calls to external services.
"""

import sys
import os
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWebServer:
    """
    Simple web server that serves the dashboard without LLM monitoring.
    """
    
    def __init__(self, port: int = 5000, host: str = '0.0.0.0'):
        """Initialize the simple web server."""
        self.port = port
        self.host = host
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for API access
        
        # Configure routes
        self._setup_routes()
        
        logger.info(f"Simple web server initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            # Serve the grouped hybrid intelligence dashboard
            import os
            dashboard_path = 'hybrid_intelligence_dashboard_grouped.html'
            if os.path.exists(dashboard_path):
                with open(dashboard_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                return "Dashboard not found", 404
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get basic system metrics."""
            return jsonify({
                'timestamp': '2025-01-15T12:00:00',
                'system': {
                    'cpu_usage': 25.5,
                    'memory_usage': 45.2
                },
                'components': {
                    'active_agents': 16,
                    'active_bridges': 5,
                    'component_status': {
                        'orchestrator': 'active',
                        'state_manager': 'active',
                        'config_agent': 'active',
                        'planning_agent': 'active',
                        'consensus_engine': 'active',
                        'security_agent': 'active'
                    }
                },
                'workflow': {
                    'queue_size': 3,
                    'events_per_second': 12.5,
                    'consensus_decisions': 47
                },
                'security': {
                    'security_alerts': 0
                },
                'alerts': {
                    'total_alerts': 0,
                    'active_alerts': 0,
                    'recent_alerts': []
                }
            })
        
        @self.app.route('/api/tests/status')
        def get_tests_status():
            """Get test status - basic simulation."""
            codebase_path = request.args.get('codebase', '/testmaster')
            
            # Simple simulation based on codebase
            if 'regex' in codebase_path.lower():
                data = [
                    {'module': 'regex_gen.py', 'status': 'green', 'test_count': 15},
                    {'module': 'pattern_builder.py', 'status': 'yellow', 'test_count': 8},
                    {'module': 'regex_validator.py', 'status': 'red', 'test_count': 0},
                    {'module': 'test_regex.py', 'status': 'green', 'test_count': 23}
                ]
            else:
                data = [
                    {'module': 'main.py', 'status': 'green', 'test_count': 12},
                    {'module': 'utils.py', 'status': 'green', 'test_count': 8},
                    {'module': 'config.py', 'status': 'yellow', 'test_count': 3},
                    {'module': 'models.py', 'status': 'red', 'test_count': 0}
                ]
            
            return jsonify(data)
        
        @self.app.route('/api/dependencies/graph')
        def get_dependency_graph():
            """Get simplified dependency graph."""
            return jsonify({
                'nodes': [
                    {'id': 'main', 'label': 'main'},
                    {'id': 'utils', 'label': 'utils'},
                    {'id': 'config', 'label': 'config'}
                ],
                'edges': [
                    {'source': 'main', 'target': 'utils'},
                    {'source': 'main', 'target': 'config'}
                ],
                'metrics': {
                    'circular_dependencies': 0,
                    'isolated_modules': 0
                }
            })
        
        @self.app.route('/api/refactor/analysis')
        def get_refactor_analysis():
            """Get basic refactor analysis."""
            return jsonify({
                'refactor_opportunities': {
                    'code_duplication': [],
                    'long_methods': [
                        {'file': 'main.py', 'method': 'process_data', 'lines': 75}
                    ],
                    'complex_classes': [
                        {'file': 'models.py', 'class': 'DataProcessor', 'methods': 15}
                    ],
                    'unused_code': [],
                    'missing_tests': [
                        {'module': 'utils.py', 'status': 'no_tests'}
                    ]
                },
                'summary': {
                    'long_methods': 1,
                    'complex_classes': 1,
                    'missing_tests': 1
                }
            })
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': '2025-01-15T12:00:00',
                'monitoring_active': True
            })
    
    def run(self, debug: bool = False):
        """Run the simple web server."""
        logger.info(f"Starting Simple Dashboard at http://{self.host}:{self.port}")
        logger.info("No LLM monitoring - no API calls will be made")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Web Monitor - No LLM API Calls")
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and start simple web server
    server = SimpleWebServer(port=args.port, host=args.host)
    
    try:
        server.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\nSimple web server stopped")

if __name__ == "__main__":
    main()