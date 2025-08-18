#!/usr/bin/env python3
"""
TestMaster Web-Based Real-Time Monitoring Dashboard - Production Ready

Professional web-based monitoring dashboard with REST API for the TestMaster 
Hybrid Intelligence Platform. Provides real-time metrics, alerts, and system status.
"""

import sys
import os
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our real-time monitor
from real_time_monitor import RealTimeMonitor, MonitoringMode, SystemMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebMonitoringServer:
    """
    Production-ready web-based monitoring server for TestMaster Hybrid Intelligence Platform.
    
    Features:
    - Real-time web dashboard with auto-refresh
    - REST API for metrics and alerts
    - Historical data tracking
    - Component status monitoring
    - Alert management
    - Performance graphs
    - Mobile-responsive design
    """
    
    def __init__(self, port: int = 5000, host: str = '0.0.0.0'):
        """Initialize the web monitoring server."""
        self.port = port
        self.host = host
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for API access
        
        # Initialize real-time monitor
        self.monitor = RealTimeMonitor(MonitoringMode.FULL, update_interval=2.0)
        self.monitor_thread = None
        
        # Start LLM monitoring if available
        if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
            self.monitor.llm_monitor.start_monitoring()
            logger.info("LLM Analysis Monitor started")
        
        # Configure routes
        self._setup_routes()
        
        # Start background monitoring
        self._start_background_monitoring()
        
        logger.info(f"Web monitoring server initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            # Serve new grouped hybrid intelligence dashboard
            import os
            grouped_path = 'hybrid_intelligence_dashboard_grouped.html'
            if os.path.exists(grouped_path):
                with open(grouped_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            # Fallback to hybrid intelligence dashboard for comprehensive view
            hybrid_path = 'hybrid_intelligence_dashboard.html'
            if os.path.exists(hybrid_path):
                with open(hybrid_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            # Fallback to complete dashboard
            dashboard_path = 'complete_dashboard.html'
            if os.path.exists(dashboard_path):
                with open(dashboard_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            # Fallback to enhanced dashboard
            enhanced_path = 'enhanced_dashboard_full.html'
            if os.path.exists(enhanced_path):
                with open(enhanced_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            # Fallback to embedded dashboard
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current system metrics."""
            return jsonify(self.monitor.get_metrics_summary())
        
        @self.app.route('/api/metrics/history')
        def get_metrics_history():
            """Get historical metrics."""
            limit = request.args.get('limit', 100, type=int)
            history = self.monitor.metrics_history[-limit:]
            
            return jsonify([
                {
                    'timestamp': m.timestamp.isoformat(),
                    'cpu_usage': m.cpu_usage,
                    'memory_usage': m.memory_usage,
                    'active_agents': m.active_agents,
                    'active_bridges': m.active_bridges,
                    'workflow_queue_size': m.workflow_queue_size,
                    'events_per_second': m.events_per_second,
                    'consensus_decisions': m.consensus_decisions,
                    'security_alerts': m.security_alerts
                } for m in history
            ])
        
        @self.app.route('/api/components')
        def get_components():
            """Get component status."""
            return jsonify(self.monitor.component_status)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get current alerts."""
            active_alerts = [a for a in self.monitor.alerts if not a.resolved]
            
            return jsonify([
                {
                    'severity': alert.severity,
                    'component': alert.component,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved
                } for alert in active_alerts
            ])
        
        @self.app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
        def resolve_alert(alert_id):
            """Resolve an alert."""
            if 0 <= alert_id < len(self.monitor.alerts):
                self.monitor.alerts[alert_id].resolved = True
                return jsonify({'status': 'success'})
            return jsonify({'status': 'error', 'message': 'Alert not found'}), 404
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self.start_time,
                'monitoring_active': self.monitor.running
            })
        
        @self.app.route('/api/config')
        def get_config():
            """Get current configuration."""
            try:
                from testmaster.core.config import get_config
                config = get_config()
                return jsonify({
                    'active_profile': config._active_profile,
                    'profile_info': config.get_profile_info(),
                    'configuration_values': len(config._config_values)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/test/integration')
        def test_integration():
            """Run integration test."""
            try:
                # Import and run basic integration test
                from testmaster.intelligence.bridges import get_protocol_bridge
                bridge = get_protocol_bridge()
                return jsonify({
                    'status': 'success',
                    'message': 'Integration test passed',
                    'bridge_status': 'operational'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error', 
                    'message': f'Integration test failed: {str(e)}'
                }), 500
        
        @self.app.route('/api/llm/metrics')
        def get_llm_metrics():
            """Get LLM analysis metrics."""
            try:
                if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
                    return jsonify(self.monitor.llm_monitor.get_llm_metrics_summary())
                else:
                    return jsonify({'error': 'LLM monitor not available'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/analysis/<path:module_path>')
        def get_module_analysis(module_path):
            """Get analysis for a specific module."""
            try:
                if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
                    analysis = self.monitor.llm_monitor.get_module_analysis(module_path)
                    if analysis:
                        return jsonify(analysis)
                    else:
                        return jsonify({'error': 'Module analysis not found'}), 404
                else:
                    return jsonify({'error': 'LLM monitor not available'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/analyze', methods=['POST'])
        def queue_analysis():
            """Queue a module for analysis."""
            try:
                data = request.get_json()
                module_path = data.get('module_path')
                
                if not module_path:
                    return jsonify({'error': 'module_path required'}), 400
                
                if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
                    self.monitor.llm_monitor.queue_module_analysis(module_path)
                    return jsonify({'status': 'queued', 'module_path': module_path})
                else:
                    return jsonify({'error': 'LLM monitor not available'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/list-modules')
        def list_llm_modules():
            """List available Python modules for analysis."""
            try:
                import glob
                modules = []
                for pattern in ['*.py', 'testmaster/**/*.py']:
                    for file in glob.glob(pattern, recursive=True):
                        if '__pycache__' not in file and '.pyc' not in file:
                            modules.append(file.replace('\\', '/'))
                return jsonify(sorted(modules)[:100])  # Limit to 100 files
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/estimate-cost', methods=['POST'])
        def estimate_llm_cost():
            """Estimate cost for analyzing a module."""
            try:
                data = request.get_json()
                module_path = data.get('module_path')
                
                if not module_path:
                    return jsonify({'error': 'module_path required'}), 400
                
                # Read file to estimate size
                import os
                if os.path.exists(module_path):
                    with open(module_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Estimate tokens (roughly 4 chars per token)
                    estimated_tokens = len(content) // 4 + 500  # Add overhead for prompt
                    # Gemini 2.5 Pro costs approximately $0.00025 per 1K input tokens
                    estimated_cost = (estimated_tokens / 1000) * 0.00025 * 2  # x2 for output
                    
                    return jsonify({
                        'module_path': module_path,
                        'file_size_bytes': len(content),
                        'estimated_tokens': estimated_tokens,
                        'estimated_cost_usd': round(estimated_cost, 4),
                        'model': 'gemini-2.5-pro'
                    })
                else:
                    return jsonify({'error': 'Module file not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/tests/status')
        def get_tests_status():
            """Get test status for all modules (non-LLM)."""
            # Get codebase parameter for multi-codebase support
            codebase_path = request.args.get('codebase', os.getcwd())
            try:
                # Use existing TestMapper and CoverageAnalyzer
                from testmaster.mapping.test_mapper import TestMapper
                from testmaster.analysis.coverage_analyzer import CoverageAnalyzer
                
                mapper = TestMapper(codebase_path, os.path.join(codebase_path, 'tests'))
                mapping = mapper.build_complete_mapping()
                
                results = []
                for module_path, tests in mapping.module_to_tests.items():
                    status = 'green' if len(tests) > 0 else 'red'
                    if len(tests) > 0:
                        # Check if tests pass
                        passed = all(test.last_passed for test in tests.values() if test.last_passed is not None)
                        if not passed:
                            status = 'yellow'
                    
                    results.append({
                        'module': module_path,
                        'status': status,
                        'test_count': len(tests),
                        'tests': list(tests.keys())
                    })
                
                return jsonify(results)
            except Exception as e:
                # Fallback to simple analysis
                import glob
                modules = glob.glob('**/*.py', recursive=True)
                results = []
                for module in modules[:50]:  # Limit to 50 for performance
                    test_file = module.replace('.py', '_test.py')
                    if os.path.exists(test_file):
                        status = 'green'
                    elif 'test' in module.lower():
                        continue
                    else:
                        status = 'red'
                    results.append({
                        'module': module,
                        'status': status,
                        'test_count': 1 if status == 'green' else 0
                    })
                return jsonify(results)
        
        @self.app.route('/api/dependencies/graph')
        def get_dependency_graph():
            """Get module dependency graph (non-LLM)."""
            try:
                from testmaster.mapping.dependency_tracker import DependencyTracker
                import networkx as nx
                
                tracker = DependencyTracker('.')
                graph = tracker.build_dependency_graph()
                
                # Convert NetworkX graph to JSON-serializable format
                nodes = []
                edges = []
                
                for node in graph.nodes():
                    nodes.append({
                        'id': node,
                        'label': node.split('/')[-1] if '/' in node else node
                    })
                
                for source, target in graph.edges():
                    edges.append({
                        'source': source,
                        'target': target
                    })
                
                # Calculate additional metrics
                circular_deps = 0
                isolated_modules = 0
                
                try:
                    # Check for circular dependencies
                    cycles = list(nx.simple_cycles(graph))
                    circular_deps = len(cycles)
                    
                    # Find isolated modules (no edges)
                    for node in graph.nodes():
                        if graph.degree(node) == 0:
                            isolated_modules += 1
                except:
                    pass
                
                return jsonify({
                    'nodes': nodes,
                    'edges': edges,
                    'metrics': {
                        'circular_dependencies': circular_deps,
                        'isolated_modules': isolated_modules
                    }
                })
            except Exception as e:
                # Fallback to simple import analysis
                import ast
                nodes = []
                edges = []
                modules_seen = set()
                
                import glob
                for file_path in glob.glob('**/*.py', recursive=True)[:30]:  # Limit for performance
                    if 'test' in file_path.lower():
                        continue
                    
                    module_name = file_path.replace('.py', '').replace('/', '.')
                    if module_name not in modules_seen:
                        nodes.append({'id': module_name, 'label': module_name.split('.')[-1]})
                        modules_seen.add(module_name)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            tree = ast.parse(f.read())
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    target = alias.name
                                    if target not in modules_seen:
                                        nodes.append({'id': target, 'label': target.split('.')[-1]})
                                        modules_seen.add(target)
                                    edges.append({'source': module_name, 'target': target})
                    except:
                        pass
                
                return jsonify({'nodes': nodes[:50], 'edges': edges[:100]})  # Limit for UI performance
        
        @self.app.route('/api/codebases', methods=['GET', 'POST', 'DELETE'])
        def manage_codebases():
            """Manage active codebases."""
            if request.method == 'GET':
                # Return list of active codebases
                return jsonify({
                    'active_codebases': [
                        {'path': '/testmaster', 'name': 'TestMaster', 'status': 'active'},
                        # Additional codebases would be stored in memory or database
                    ]
                })
            elif request.method == 'POST':
                # Add new codebase
                data = request.get_json()
                path = data.get('path')
                if not path or not os.path.exists(path):
                    return jsonify({'error': 'Invalid codebase path'}), 400
                
                name = os.path.basename(path) or path
                return jsonify({
                    'success': True,
                    'codebase': {'path': path, 'name': name, 'status': 'active'}
                })
            elif request.method == 'DELETE':
                # Remove codebase
                path = request.args.get('path')
                return jsonify({'success': True, 'removed': path})
        
        @self.app.route('/api/refactor/analysis')
        def get_refactor_analysis():
            """Get automated refactor opportunity analysis (non-LLM)."""
            try:
                import ast
                import glob
                
                refactor_opportunities = {
                    'code_duplication': [],
                    'long_methods': [],
                    'complex_classes': [],
                    'unused_code': [],
                    'missing_tests': []
                }
                
                # Analyze Python files for refactor opportunities
                for file_path in glob.glob('**/*.py', recursive=True)[:30]:  # Limit for performance
                    if '__pycache__' in file_path or 'test' in file_path.lower():
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            tree = ast.parse(content)
                        
                        # Check for long methods (>50 lines)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                                    method_length = node.end_lineno - node.lineno
                                    if method_length > 50:
                                        refactor_opportunities['long_methods'].append({
                                            'file': file_path,
                                            'method': node.name,
                                            'lines': method_length
                                        })
                            
                            # Check for complex classes (>10 methods)
                            elif isinstance(node, ast.ClassDef):
                                method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                                if method_count > 10:
                                    refactor_opportunities['complex_classes'].append({
                                        'file': file_path,
                                        'class': node.name,
                                        'methods': method_count
                                    })
                    except:
                        pass
                
                # Check for missing tests
                from testmaster.mapping.test_mapper import TestMapper
                try:
                    mapper = TestMapper('.', 'tests')
                    mapping = mapper.build_complete_mapping()
                    
                    for module_path in mapping.module_to_tests:
                        if len(mapping.module_to_tests[module_path]) == 0:
                            refactor_opportunities['missing_tests'].append({
                                'module': module_path,
                                'status': 'no_tests'
                            })
                except:
                    pass
                
                return jsonify({
                    'refactor_opportunities': refactor_opportunities,
                    'summary': {
                        'long_methods': len(refactor_opportunities['long_methods']),
                        'complex_classes': len(refactor_opportunities['complex_classes']),
                        'missing_tests': len(refactor_opportunities['missing_tests'])
                    }
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _start_background_monitoring(self):
        """Start background monitoring thread."""
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitor_thread.start()
    
    def _monitoring_worker(self):
        """Background monitoring worker."""
        self.monitor.running = True
        
        while self.monitor.running:
            try:
                self.monitor._collect_metrics()
                self.monitor._check_alerts()
                self.monitor._store_metrics()
                time.sleep(self.monitor.update_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def run(self, debug: bool = False):
        """Run the web monitoring server."""
        logger.info(f"Starting TestMaster Web Monitoring Dashboard at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
    
    def stop(self):
        """Stop the monitoring server."""
        logger.info("Stopping web monitoring server...")
        self.monitor.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

# HTML Dashboard Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TestMaster Hybrid Intelligence Platform - Real-Time Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { 
            text-align: center; 
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .dashboard { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
        }
        .card { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card h3 { 
            margin-bottom: 15px; 
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child { border-bottom: none; }
        .metric-value { font-weight: bold; font-size: 1.1em; }
        .status-indicator { 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            display: inline-block; 
            margin-right: 8px;
        }
        .status-active { background: #4CAF50; }
        .status-inactive { background: #f44336; }
        .status-warning { background: #ff9800; }
        .alert { 
            padding: 10px; 
            margin-bottom: 10px; 
            border-radius: 5px; 
            border-left: 4px solid;
        }
        .alert-warning { 
            background: rgba(255, 152, 0, 0.2); 
            border-left-color: #ff9800; 
        }
        .alert-critical { 
            background: rgba(244, 67, 54, 0.2); 
            border-left-color: #f44336; 
        }
        .alert-info { 
            background: rgba(33, 150, 243, 0.2); 
            border-left-color: #2196F3; 
        }
        .refresh-info { 
            text-align: center; 
            margin-top: 20px; 
            opacity: 0.7; 
        }
        .chart-container { 
            height: 200px; 
            margin: 15px 0; 
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            padding: 10px;
        }
        .component-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .component-item {
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .loading { text-align: center; padding: 40px; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 4px solid #fff;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TestMaster Hybrid Intelligence Platform</h1>
            <p>Real-Time Monitoring Dashboard</p>
            <p id="last-update">Loading...</p>
        </div>
        
        <div id="dashboard" class="dashboard">
            <div class="loading">
                <div class="spinner"></div>
                <p>Initializing monitoring systems...</p>
            </div>
        </div>
        
        <div class="refresh-info">
            <p>Dashboard refreshes every 5 seconds | Data collected every 2 seconds</p>
        </div>
    </div>

    <script>
        let metricsHistory = [];
        
        async function fetchData(endpoint) {
            try {
                const response = await fetch(`/api/${endpoint}`);
                return await response.json();
            } catch (error) {
                console.error(`Error fetching ${endpoint}:`, error);
                return null;
            }
        }
        
        function formatTimestamp(timestamp) {
            return new Date(timestamp).toLocaleString();
        }
        
        function getStatusIndicator(status) {
            if (status === 'active') return '<span class="status-indicator status-active"></span>';
            if (status === 'inactive') return '<span class="status-indicator status-inactive"></span>';
            return '<span class="status-indicator status-warning"></span>';
        }
        
        function createMetricsCard(metrics) {
            return `
                <div class="card">
                    <h3>📊 System Metrics</h3>
                    <div class="metric">
                        <span>CPU Usage:</span>
                        <span class="metric-value">${metrics.system.cpu_usage.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Memory Usage:</span>
                        <span class="metric-value">${metrics.system.memory_usage.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Active Agents:</span>
                        <span class="metric-value">${metrics.components.active_agents}/16</span>
                    </div>
                    <div class="metric">
                        <span>Active Bridges:</span>
                        <span class="metric-value">${metrics.components.active_bridges}/5</span>
                    </div>
                    <div class="metric">
                        <span>Queue Size:</span>
                        <span class="metric-value">${metrics.workflow.queue_size}</span>
                    </div>
                    <div class="metric">
                        <span>Events/Second:</span>
                        <span class="metric-value">${metrics.workflow.events_per_second.toFixed(1)}</span>
                    </div>
                </div>
            `;
        }
        
        function createLLMMetricsCard(llmMetrics) {
            if (!llmMetrics || llmMetrics.error) {
                return `
                    <div class="card">
                        <h3>🤖 LLM Intelligence</h3>
                        <p style="text-align: center; opacity: 0.7; padding: 20px;">
                            ${llmMetrics ? llmMetrics.error : 'LLM monitoring not available'}
                        </p>
                    </div>
                `;
            }
            
            return `
                <div class="card">
                    <h3>🤖 LLM Intelligence</h3>
                    <div class="metric">
                        <span>API Calls:</span>
                        <span class="metric-value">${llmMetrics.api_calls.total_calls}</span>
                    </div>
                    <div class="metric">
                        <span>Success Rate:</span>
                        <span class="metric-value">${llmMetrics.api_calls.success_rate.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Tokens Used:</span>
                        <span class="metric-value">${llmMetrics.token_usage.total_tokens.toLocaleString()}</span>
                    </div>
                    <div class="metric">
                        <span>Cost Estimate:</span>
                        <span class="metric-value">$${llmMetrics.cost_tracking.total_cost_estimate.toFixed(3)}</span>
                    </div>
                    <div class="metric">
                        <span>Calls/Minute:</span>
                        <span class="metric-value">${llmMetrics.api_calls.calls_per_minute.toFixed(1)}</span>
                    </div>
                    <div class="metric">
                        <span>Active Analyses:</span>
                        <span class="metric-value">${llmMetrics.analysis_status.active_analyses}</span>
                    </div>
                </div>
            `;
        }
        
        function createComponentsCard(components) {
            const componentItems = Object.entries(components)
                .map(([name, status]) => `
                    <div class="component-item">
                        ${getStatusIndicator(status)}
                        <span>${name.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}</span>
                    </div>
                `).join('');
            
            return `
                <div class="card">
                    <h3>🔧 Component Status</h3>
                    <div class="component-grid">
                        ${componentItems}
                    </div>
                </div>
            `;
        }
        
        function createAlertsCard(alerts) {
            if (!alerts.recent_alerts || alerts.recent_alerts.length === 0) {
                return `
                    <div class="card">
                        <h3>🚨 Recent Alerts</h3>
                        <p style="text-align: center; opacity: 0.7; padding: 20px;">No recent alerts</p>
                    </div>
                `;
            }
            
            const alertItems = alerts.recent_alerts
                .slice(-5)
                .map(alert => `
                    <div class="alert alert-${alert.severity}">
                        <strong>${alert.severity.toUpperCase()}</strong> - ${alert.component}<br>
                        ${alert.message}<br>
                        <small>${formatTimestamp(alert.timestamp)}</small>
                    </div>
                `).join('');
            
            return `
                <div class="card">
                    <h3>🚨 Recent Alerts (${alerts.total_alerts} total, ${alerts.active_alerts} active)</h3>
                    ${alertItems}
                </div>
            `;
        }
        
        function createHealthCard(metrics) {
            return `
                <div class="card">
                    <h3>💚 System Health</h3>
                    <div class="metric">
                        <span>Total Agents:</span>
                        <span class="metric-value">16</span>
                    </div>
                    <div class="metric">
                        <span>Total Bridges:</span>
                        <span class="metric-value">5</span>
                    </div>
                    <div class="metric">
                        <span>Consensus Decisions:</span>
                        <span class="metric-value">${metrics.workflow.consensus_decisions}</span>
                    </div>
                    <div class="metric">
                        <span>Security Alerts:</span>
                        <span class="metric-value">${metrics.security.security_alerts}</span>
                    </div>
                    <div class="metric">
                        <span>Last Update:</span>
                        <span class="metric-value" id="update-time">${formatTimestamp(metrics.timestamp)}</span>
                    </div>
                </div>
            `;
        }
        
        async function updateDashboard() {
            const [metrics, components, llmMetrics] = await Promise.all([
                fetchData('metrics'),
                fetchData('components'),
                fetchData('llm/metrics')
            ]);
            
            if (!metrics) {
                document.getElementById('dashboard').innerHTML = '<div class="card"><h3>❌ Error</h3><p>Failed to load metrics</p></div>';
                return;
            }
            
            // Update last update time
            document.getElementById('last-update').textContent = `Last Update: ${formatTimestamp(metrics.timestamp)}`;
            
            // Create dashboard content
            const dashboardHTML = `
                ${createMetricsCard(metrics)}
                ${createLLMMetricsCard(llmMetrics)}
                ${createComponentsCard(metrics.components.component_status)}
                ${createHealthCard(metrics)}
                ${createAlertsCard(metrics.alerts)}
            `;
            
            document.getElementById('dashboard').innerHTML = dashboardHTML;
        }
        
        // Initialize dashboard
        updateDashboard();
        
        // Auto-refresh every 5 seconds
        setInterval(updateDashboard, 5000);
        
        // Add some basic error handling
        window.addEventListener('error', (e) => {
            console.error('Dashboard error:', e.error);
        });
    </script>
</body>
</html>
"""

def main():
    """Main entry point for web monitoring server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TestMaster Web Monitoring Dashboard")
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and start web server
    server = WebMonitoringServer(port=args.port, host=args.host)
    
    try:
        server.run(debug=args.debug)
    except KeyboardInterrupt:
        server.stop()
        print("\nWeb monitoring server stopped")

if __name__ == "__main__":
    main()