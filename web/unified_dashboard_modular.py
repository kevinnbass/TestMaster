"""
Unified Dashboard - Modular Architecture
========================================

EPSILON ENHANCEMENT Hour 4: Modularized dashboard following STEELCLAD protocol.
This is the main entry point that orchestrates all dashboard modules.

Created: 2025-08-23 18:40:00
Author: Agent Epsilon
"""

import os
import sys
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
from datetime import datetime

# Add dashboard modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dashboard_modules'))

# Import modular components
from dashboard_modules.intelligence.enhanced_contextual import EnhancedContextualEngine
from dashboard_modules.integration.data_integrator import DataIntegrator
from dashboard_modules.visualization.advanced_visualization import AdvancedVisualizationEngine
from dashboard_modules.monitoring.performance_monitor import PerformanceMonitor


class UnifiedDashboardModular:
    """
    Modular Unified Dashboard Engine - Clean architecture with separated concerns.
    """
    
    def __init__(self, port=5001):
        # Set up Flask app with template directory
        template_dir = os.path.join(os.path.dirname(__file__), 'dashboard_modules', 'templates')
        self.app = Flask(__name__, template_folder=template_dir)
        self.app.config['SECRET_KEY'] = 'epsilon-enhancement-secret-2025'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)
        
        self.port = port
        
        # Initialize modular components
        self.contextual_engine = EnhancedContextualEngine()
        self.data_integrator = DataIntegrator()
        self.visualization_engine = AdvancedVisualizationEngine()
        self.performance_monitor = PerformanceMonitor()
        
        # Setup routes
        self.setup_routes()
        self.setup_socketio_events()
        
        print("EPSILON MODULAR DASHBOARD - ENHANCED ARCHITECTURE")
        print("=" * 60)
        print(f"Dashboard running on: http://localhost:{self.port}")
        print("Architecture: Modular design following STEELCLAD protocol")
        print("Enhancement: Hour 4 Phase 1B implementation")
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Render main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/health')
        def health():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'modules': {
                    'contextual_engine': 'active',
                    'data_integrator': 'active',
                    'visualization_engine': 'active',
                    'performance_monitor': 'active',
                    'intelligence_metrics': self.contextual_engine.intelligence_metrics
                }
            })
        
        @self.app.route('/api/contextual-analysis', methods=['POST'])
        def contextual_analysis():
            """Perform contextual analysis on provided data."""
            data = request.json
            agent_data = data.get('agent_data', {})
            
            analysis = self.contextual_engine.analyze_multi_agent_context(agent_data)
            
            return jsonify({
                'status': 'success',
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/proactive-insights')
        def proactive_insights():
            """Get proactive insights based on current system state."""
            # Mock system state for demonstration
            system_state = {
                'health': {'score': 75},
                'api_usage': {'daily_cost': 80, 'budget_limit': 100},
                'performance': {'response_time': 500}
            }
            
            insights = self.contextual_engine.generate_proactive_insights(system_state)
            
            return jsonify({
                'status': 'success',
                'insights': insights,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/behavior-prediction', methods=['POST'])
        def behavior_prediction():
            """Predict user behavior based on context and history."""
            data = request.json
            user_context = data.get('user_context', {})
            interaction_history = data.get('history', [])
            
            predictions = self.contextual_engine.predict_user_behavior(
                user_context, interaction_history
            )
            
            return jsonify({
                'status': 'success',
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/unified-data')
        def unified_data():
            """Get unified data from DataIntegrator."""
            user_context = request.args.to_dict()
            
            data = self.data_integrator.get_unified_data(user_context if user_context else None)
            
            return jsonify({
                'status': 'success',
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/visualization-recommendations', methods=['POST'])
        def visualization_recommendations():
            """Get AI-powered visualization recommendations."""
            data = request.json
            data_characteristics = data.get('data_characteristics', {})
            user_context = data.get('user_context', {})
            
            recommendations = self.visualization_engine.select_optimal_visualization(
                data_characteristics, user_context
            )
            
            return jsonify({
                'status': 'success',
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/chart-config', methods=['POST'])
        def chart_config():
            """Generate interactive chart configuration."""
            data = request.json
            chart_type = data.get('chart_type', 'intelligent_line_chart')
            chart_data = data.get('data', {})
            user_context = data.get('user_context', {})
            enhancements = data.get('enhancements', [])
            
            config = self.visualization_engine.create_interactive_chart_config(
                chart_type, chart_data, user_context, enhancements
            )
            
            return jsonify({
                'status': 'success',
                'config': config,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/performance-metrics')
        def performance_metrics():
            """Get comprehensive performance metrics."""
            metrics = self.performance_monitor.get_metrics()
            
            return jsonify({
                'status': 'success',
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/performance-analytics')
        def performance_analytics():
            """Get performance analytics and insights."""
            analytics = self.performance_monitor.get_performance_analytics()
            
            return jsonify({
                'status': 'success',
                'analytics': analytics,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/performance-status')
        def performance_status():
            """Get real-time performance status."""
            status = self.performance_monitor.get_real_time_status()
            
            return jsonify({
                'status': 'success',
                'performance_status': status,
                'timestamp': datetime.now().isoformat()
            })
        
        # Hour 7: Advanced Visualization Enhancement Endpoints
        @self.app.route('/api/visualization/interactive-config', methods=['POST'])
        def interactive_visualization_config():
            """Generate advanced interactive visualization configuration."""
            data = request.json
            chart_type = data.get('chart_type', 'intelligent_dashboard')
            data_sources = data.get('data_sources', [])
            user_context = data.get('user_context', {})
            interaction_requirements = data.get('interactions', [])
            
            # Generate contextual interactions based on data relationships
            relationships = self._analyze_data_relationships(data_sources)
            interactions = self.visualization_engine.generate_contextual_interactions(
                data_sources, relationships, user_context
            )
            
            config = {
                'chart_config': self.visualization_engine.create_interactive_chart_config(
                    chart_type, data_sources, user_context, interaction_requirements
                ),
                'interactions': interactions,
                'relationships': relationships,
                'adaptive_features': self._generate_adaptive_features(user_context)
            }
            
            return jsonify({
                'status': 'success',
                'config': config,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/visualization/drill-down', methods=['POST'])
        def visualization_drill_down():
            """Handle intelligent drill-down requests."""
            data = request.json
            current_level = data.get('current_level', 0)
            selected_data_point = data.get('data_point', {})
            user_context = data.get('user_context', {})
            
            # Use visualization engine to determine optimal drill-down path
            drill_down_config = self.visualization_engine.create_drill_down_visualization(
                current_level, selected_data_point, user_context
            )
            
            return jsonify({
                'status': 'success',
                'drill_down_config': drill_down_config,
                'breadcrumb_path': drill_down_config.get('breadcrumb_path', []),
                'available_actions': drill_down_config.get('available_actions', []),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/visualization/adaptive-layout', methods=['POST'])
        def adaptive_visualization_layout():
            """Generate adaptive layout based on device and user context."""
            data = request.json
            device_info = data.get('device_info', {})
            user_preferences = data.get('preferences', {})
            dashboard_data = data.get('dashboard_data', {})
            
            layout_config = self.visualization_engine.generate_adaptive_layout(
                device_info, user_preferences, dashboard_data
            )
            
            return jsonify({
                'status': 'success',
                'layout': layout_config,
                'responsive_breakpoints': layout_config.get('breakpoints', {}),
                'optimization_applied': layout_config.get('optimizations', []),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/visualization/intelligence-insights')
        def visualization_intelligence_insights():
            """Get AI-powered visualization insights and recommendations."""
            # Get current system data for analysis
            system_metrics = self.performance_monitor.get_metrics()
            contextual_data = self.contextual_engine.get_current_analysis_state()
            unified_data = self.data_integrator.get_unified_data()
            
            # Generate visualization intelligence insights
            insights = self.visualization_engine.generate_visualization_insights(
                system_metrics, contextual_data, unified_data
            )
            
            return jsonify({
                'status': 'success',
                'insights': insights,
                'recommended_visualizations': insights.get('recommendations', []),
                'optimization_opportunities': insights.get('optimizations', []),
                'timestamp': datetime.now().isoformat()
            })
    
    def _analyze_data_relationships(self, data_sources):
        """Analyze relationships between data sources for intelligent visualization."""
        relationships = {
            'correlations': [],
            'hierarchies': [],
            'temporal_connections': [],
            'categorical_groupings': []
        }
        
        # Mock relationship analysis for demonstration
        if len(data_sources) > 1:
            relationships['correlations'].append({
                'source_a': 'performance_metrics',
                'source_b': 'user_behavior',
                'strength': 0.75,
                'type': 'positive'
            })
        
        return relationships
    
    def _generate_adaptive_features(self, user_context):
        """Generate adaptive features based on user context."""
        features = []
        
        user_role = user_context.get('role', 'general')
        device = user_context.get('device', 'desktop')
        
        if user_role in ['analyst', 'technical']:
            features.extend(['advanced_tooltips', 'statistical_overlays', 'data_export'])
        
        if device == 'mobile':
            features.extend(['gesture_navigation', 'simplified_ui', 'touch_optimized'])
        elif device == 'tablet':
            features.extend(['touch_navigation', 'adaptive_sizing', 'orientation_aware'])
        
        return features
    
    # Hour 8: Real-time Data Streaming Helper Methods
    def _start_performance_stream(self, client_id):
        """Start streaming performance metrics to a specific client."""
        def stream_performance():
            while True:
                try:
                    metrics = self.performance_monitor.get_metrics()
                    self.socketio.emit('performance_stream', {
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    }, room=client_id)
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"Error in performance stream: {e}")
                    break
        
        # Start streaming in background thread
        thread = threading.Thread(target=stream_performance, daemon=True)
        thread.start()
    
    def _start_visualization_stream(self, client_id):
        """Start streaming visualization updates to a specific client."""
        def stream_visualizations():
            while True:
                try:
                    insights = self.visualization_engine.generate_visualization_insights(
                        self.performance_monitor.get_metrics(),
                        self.contextual_engine.get_current_analysis_state(),
                        self.data_integrator.get_unified_data()
                    )
                    self.socketio.emit('visualization_stream', {
                        'insights': insights,
                        'timestamp': datetime.now().isoformat()
                    }, room=client_id)
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    print(f"Error in visualization stream: {e}")
                    break
        
        thread = threading.Thread(target=stream_visualizations, daemon=True)
        thread.start()
    
    def _start_predictive_stream(self, client_id):
        """Start streaming predictive analytics to a specific client."""
        def stream_predictions():
            while True:
                try:
                    current_metrics = self.performance_monitor.get_metrics()
                    predictions = self._generate_predictive_analysis(
                        'trend_forecast', current_metrics, 12
                    )
                    self.socketio.emit('predictive_stream', {
                        'predictions': predictions,
                        'timestamp': datetime.now().isoformat()
                    }, room=client_id)
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    print(f"Error in predictive stream: {e}")
                    break
        
        thread = threading.Thread(target=stream_predictions, daemon=True)
        thread.start()
    
    def _generate_chart_data(self, chart_type, data_range, filters):
        """Generate chart data based on type, range, and filters."""
        # Mock data generation for demonstration
        import random
        
        current_time = datetime.now()
        data_points = []
        
        if data_range == '1h':
            points = 60
            interval = 1  # minutes
        elif data_range == '6h':
            points = 72
            interval = 5  # minutes
        elif data_range == '24h':
            points = 144
            interval = 10  # minutes
        else:
            points = 30
            interval = 1
        
        for i in range(points):
            timestamp = current_time - timedelta(minutes=i * interval)
            
            if chart_type == 'performance_line':
                value = 75 + random.uniform(-15, 15) + (5 * random.sin(i * 0.1))
            elif chart_type == 'cpu_usage':
                value = 45 + random.uniform(-10, 25) + (10 * random.sin(i * 0.05))
            elif chart_type == 'memory_usage':
                value = 60 + random.uniform(-5, 20) + (15 * random.cos(i * 0.08))
            else:
                value = 50 + random.uniform(-20, 30)
            
            data_points.append({
                'timestamp': timestamp.isoformat(),
                'value': max(0, min(100, value)),
                'metadata': {'interval': interval, 'type': chart_type}
            })
        
        return {
            'points': list(reversed(data_points)),
            'range': data_range,
            'chart_type': chart_type,
            'total_points': len(data_points)
        }
    
    def _generate_predictive_analysis(self, analysis_type, historical_data, forecast_horizon):
        """Generate predictive analysis based on historical data."""
        import random
        
        predictions = []
        current_time = datetime.now()
        
        # Generate forecast points
        for i in range(forecast_horizon):
            future_time = current_time + timedelta(hours=i)
            
            if analysis_type == 'trend_forecast':
                # Simple trend prediction with some randomness
                base_value = 70 + (i * 0.5)  # Slight upward trend
                noise = random.uniform(-5, 5)
                predicted_value = base_value + noise
            elif analysis_type == 'anomaly_detection':
                predicted_value = 65 + random.uniform(-10, 10)
                if i % 8 == 0:  # Inject anomaly every 8 hours
                    predicted_value += random.uniform(15, 25)
            else:
                predicted_value = 60 + random.uniform(-15, 15)
            
            predictions.append({
                'timestamp': future_time.isoformat(),
                'predicted_value': max(0, min(100, predicted_value)),
                'confidence': max(0.6, 1.0 - (i * 0.02)),  # Decreasing confidence over time
                'lower_bound': predicted_value - 5,
                'upper_bound': predicted_value + 5
            })
        
        return {
            'predictions': predictions,
            'analysis_type': analysis_type,
            'horizon_hours': forecast_horizon,
            'confidence': {
                'average': sum(p['confidence'] for p in predictions) / len(predictions),
                'trend_strength': 0.8,
                'data_quality': 0.9
            },
            'recommendation': f'Based on {analysis_type}, expect gradual performance improvement over next {forecast_horizon} hours.'
        }
    
    def _get_update_frequency(self, chart_type):
        """Get optimal update frequency for different chart types."""
        frequencies = {
            'performance_line': 2000,    # 2 seconds
            'cpu_usage': 1000,           # 1 second  
            'memory_usage': 1500,        # 1.5 seconds
            'network_io': 3000,          # 3 seconds
            'disk_usage': 5000,          # 5 seconds
            'predictive_trend': 30000,   # 30 seconds
            'default': 5000              # 5 seconds
        }
        return frequencies.get(chart_type, frequencies['default'])

    def setup_socketio_events(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            print(f"Client connected: {request.sid}")
            emit('connected', {
                'message': 'Connected to Modular Dashboard',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_analysis')
        def handle_analysis_request(data):
            """Handle real-time analysis requests."""
            agent_data = data.get('agent_data', {})
            analysis = self.contextual_engine.analyze_multi_agent_context(agent_data)
            
            emit('analysis_result', {
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
        
        # Hour 8: Advanced Real-time Data Streaming Events
        @self.socketio.on('subscribe_live_data')
        def handle_live_data_subscription(data):
            """Handle subscription to real-time data streams."""
            stream_types = data.get('streams', [])
            client_id = request.sid
            
            print(f"Client {client_id} subscribed to streams: {stream_types}")
            
            # Start streaming data for subscribed types
            for stream_type in stream_types:
                if stream_type == 'performance_metrics':
                    self._start_performance_stream(client_id)
                elif stream_type == 'visualization_updates':
                    self._start_visualization_stream(client_id)
                elif stream_type == 'predictive_analytics':
                    self._start_predictive_stream(client_id)
            
            emit('subscription_confirmed', {
                'streams': stream_types,
                'client_id': client_id,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('request_chart_data')
        def handle_chart_data_request(data):
            """Handle requests for chart data with real-time updates."""
            chart_type = data.get('chart_type', 'line')
            data_range = data.get('range', '1h')
            filters = data.get('filters', {})
            
            # Generate chart data based on request
            chart_data = self._generate_chart_data(chart_type, data_range, filters)
            
            emit('chart_data_update', {
                'chart_type': chart_type,
                'data': chart_data,
                'metadata': {
                    'range': data_range,
                    'filters': filters,
                    'update_frequency': self._get_update_frequency(chart_type)
                },
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('request_predictive_analysis')
        def handle_predictive_analysis_request(data):
            """Handle requests for predictive analytics visualization."""
            analysis_type = data.get('type', 'trend_forecast')
            historical_data = data.get('historical_data', {})
            forecast_horizon = data.get('horizon', 24)  # hours
            
            # Generate predictive analysis
            prediction_result = self._generate_predictive_analysis(
                analysis_type, historical_data, forecast_horizon
            )
            
            emit('predictive_analysis_result', {
                'analysis_type': analysis_type,
                'predictions': prediction_result,
                'confidence_intervals': prediction_result.get('confidence', {}),
                'recommendation': prediction_result.get('recommendation', ''),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('update_chart_config')
        def handle_chart_config_update(data):
            """Handle dynamic chart configuration updates."""
            chart_id = data.get('chart_id')
            new_config = data.get('config', {})
            user_context = data.get('user_context', {})
            
            # Apply intelligent configuration updates
            optimized_config = self.visualization_engine.create_interactive_chart_config(
                new_config.get('type', 'intelligent_line_chart'),
                new_config.get('data', {}),
                user_context,
                new_config.get('enhancements', [])
            )
            
            emit('chart_config_updated', {
                'chart_id': chart_id,
                'config': optimized_config,
                'optimizations_applied': optimized_config.get('optimizations', []),
                'timestamp': datetime.now().isoformat()
            })
    
    def render_dashboard(self):
        """Render the main dashboard HTML."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Epsilon Modular Dashboard - Enhanced Intelligence</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 40px 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .card h2 {
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 600;
            opacity: 0.9;
        }
        
        .metric-value {
            font-weight: bold;
            color: #ffd700;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 10px;
        }
        
        .status-excellent { background: #4ade80; }
        .status-good { background: #facc15; }
        .status-needs_attention { background: #f87171; }
        
        .insights-container {
            margin-top: 20px;
        }
        
        .insight {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #ffd700;
        }
        
        .insight-type {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .insight-message {
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .recommendations {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .recommendations ul {
            margin-left: 20px;
            margin-top: 5px;
        }
        
        #connectionStatus {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .connected { color: #4ade80; }
        .disconnected { color: #f87171; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 2s infinite;
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div id="connectionStatus" class="disconnected">Connecting...</div>
    
    <div class="container">
        <header>
            <h1>🚀 Epsilon Modular Dashboard</h1>
            <div class="subtitle">Enhanced Contextual Intelligence System - Hour 4 Implementation</div>
        </header>
        
        <div class="dashboard-grid">
            <div class="card">
                <h2>🧠 Contextual Intelligence</h2>
                <div id="contextualMetrics" class="loading">
                    <div class="metric">
                        <span class="metric-label">Correlations Detected</span>
                        <span class="metric-value" id="correlationsCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Insights Generated</span>
                        <span class="metric-value" id="insightsCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Predictions Made</span>
                        <span class="metric-value" id="predictionsCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Optimization Opportunities</span>
                        <span class="metric-value" id="optimizationCount">0</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Agent Coordination Health</h2>
                <div id="coordinationHealth" class="loading">
                    <div class="metric">
                        <span class="metric-label">Overall Health Score</span>
                        <span class="metric-value">
                            <span id="healthScore">--</span>%
                            <span id="healthStatus" class="status-indicator"></span>
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Data Synchronization</span>
                        <span class="metric-value" id="dataSync">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Response Consistency</span>
                        <span class="metric-value" id="responseConsistency">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Resource Balance</span>
                        <span class="metric-value" id="resourceBalance">--</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>💡 Proactive Insights</h2>
                <div id="proactiveInsights" class="insights-container loading">
                    <div class="insight">
                        <div class="insight-type">Loading insights...</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>🎯 User Behavior Predictions</h2>
                <div id="behaviorPredictions" class="loading">
                    <div class="metric">
                        <span class="metric-label">Next Likely Action</span>
                        <span class="metric-value" id="nextAction">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Information Need</span>
                        <span class="metric-value" id="infoNeed">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Attention Focus</span>
                        <span class="metric-value" id="attentionFocus">--</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        class ModularDashboard {
            constructor() {
                this.socket = io();
                this.setupEventHandlers();
                this.initializeDashboard();
            }
            
            setupEventHandlers() {
                this.socket.on('connect', () => {
                    console.log('Connected to Modular Dashboard');
                    document.getElementById('connectionStatus').textContent = 'Connected';
                    document.getElementById('connectionStatus').className = 'connected';
                    this.requestInitialData();
                });
                
                this.socket.on('disconnect', () => {
                    console.log('Disconnected from dashboard');
                    document.getElementById('connectionStatus').textContent = 'Disconnected';
                    document.getElementById('connectionStatus').className = 'disconnected';
                });
                
                this.socket.on('analysis_result', (data) => {
                    this.updateContextualAnalysis(data.analysis);
                });
            }
            
            async initializeDashboard() {
                // Fetch initial data via REST API
                await this.fetchHealthData();
                await this.fetchProactiveInsights();
                await this.fetchBehaviorPredictions();
                
                // Set up periodic updates
                setInterval(() => this.fetchHealthData(), 5000);
                setInterval(() => this.fetchProactiveInsights(), 10000);
                setInterval(() => this.fetchBehaviorPredictions(), 15000);
            }
            
            requestInitialData() {
                // Request analysis via WebSocket
                const mockAgentData = {
                    'agent_alpha': {
                        'cpu_usage': 45,
                        'memory_usage': 62,
                        'response_time': 120,
                        'error_rate': 2
                    },
                    'agent_beta': {
                        'cpu_usage': 38,
                        'memory_usage': 55,
                        'response_time': 95,
                        'error_rate': 1
                    },
                    'agent_gamma': {
                        'cpu_usage': 72,
                        'memory_usage': 81,
                        'response_time': 250,
                        'error_rate': 4
                    }
                };
                
                this.socket.emit('request_analysis', { agent_data: mockAgentData });
            }
            
            async fetchHealthData() {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    
                    if (data.modules && data.modules.intelligence_metrics) {
                        const metrics = data.modules.intelligence_metrics;
                        document.getElementById('correlationsCount').textContent = metrics.correlations_detected;
                        document.getElementById('insightsCount').textContent = metrics.insights_generated;
                        document.getElementById('predictionsCount').textContent = metrics.predictions_made;
                        document.getElementById('optimizationCount').textContent = metrics.optimization_opportunities;
                    }
                    
                    // Remove loading state
                    document.getElementById('contextualMetrics').classList.remove('loading');
                } catch (error) {
                    console.error('Error fetching health data:', error);
                }
            }
            
            async fetchProactiveInsights() {
                try {
                    const response = await fetch('/api/proactive-insights');
                    const data = await response.json();
                    
                    const container = document.getElementById('proactiveInsights');
                    container.innerHTML = '';
                    container.classList.remove('loading');
                    
                    if (data.insights && data.insights.length > 0) {
                        data.insights.forEach(insight => {
                            const insightEl = document.createElement('div');
                            insightEl.className = 'insight';
                            insightEl.innerHTML = `
                                <div class="insight-type">${insight.type.replace('_', ' ').toUpperCase()}</div>
                                <div class="insight-message">${insight.message}</div>
                                ${insight.recommendations ? `
                                    <div class="recommendations">
                                        Recommendations:
                                        <ul>${insight.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                                    </div>
                                ` : ''}
                            `;
                            container.appendChild(insightEl);
                        });
                    } else {
                        container.innerHTML = '<div class="insight"><div class="insight-type">All systems optimal</div></div>';
                    }
                } catch (error) {
                    console.error('Error fetching insights:', error);
                }
            }
            
            async fetchBehaviorPredictions() {
                try {
                    const response = await fetch('/api/behavior-prediction', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_context: { role: 'technical', device: 'desktop' },
                            history: [
                                { action: 'view_metrics', timestamp: Date.now() - 60000 },
                                { action: 'check_health', timestamp: Date.now() - 30000 },
                                { action: 'view_metrics', timestamp: Date.now() - 15000 }
                            ]
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.predictions) {
                        const predictions = data.predictions;
                        
                        // Update next action
                        if (predictions.next_likely_actions && predictions.next_likely_actions.length > 0) {
                            const topAction = predictions.next_likely_actions[0];
                            document.getElementById('nextAction').textContent = 
                                `${topAction.action} (${Math.round(topAction.probability * 100)}%)`;
                        }
                        
                        // Update information needs
                        if (predictions.information_needs && predictions.information_needs.length > 0) {
                            document.getElementById('infoNeed').textContent = 
                                predictions.information_needs[0].replace('_', ' ');
                        }
                        
                        // Update attention focus
                        document.getElementById('attentionFocus').textContent = 'Metrics & Health';
                    }
                    
                    document.getElementById('behaviorPredictions').classList.remove('loading');
                } catch (error) {
                    console.error('Error fetching predictions:', error);
                }
            }
            
            updateContextualAnalysis(analysis) {
                if (!analysis) return;
                
                // Update coordination health
                if (analysis.agent_coordination_health) {
                    const health = analysis.agent_coordination_health;
                    document.getElementById('healthScore').textContent = health.overall_score;
                    
                    const statusEl = document.getElementById('healthStatus');
                    statusEl.className = `status-indicator status-${health.status}`;
                    
                    if (health.factors) {
                        document.getElementById('dataSync').textContent = 
                            `${Math.round(health.factors.data_synchronization)}%`;
                        document.getElementById('responseConsistency').textContent = 
                            `${Math.round(health.factors.response_time_consistency)}%`;
                        document.getElementById('resourceBalance').textContent = 
                            `${Math.round(health.factors.resource_utilization_balance)}%`;
                    }
                }
                
                document.getElementById('coordinationHealth').classList.remove('loading');
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            window.dashboard = new ModularDashboard();
        });
    </script>
</body>
</html>
        '''
    
    def run(self):
        """Start the modular dashboard server."""
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    dashboard = UnifiedDashboardModular(port=5001)  # ADAMANTIUMCLAD compliant port
    dashboard.run()