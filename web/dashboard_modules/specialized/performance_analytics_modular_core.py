#!/usr/bin/env python3
"""
Performance Analytics Dashboard - STEELCLAD Clean Version
=========================================================

Agent Y supporting Agent Z: STEELCLAD modularization of performance_analytics_dashboard.py
Reduced from 1,150 lines to <400 lines by extracting modular components

This streamlined version coordinates modular components:
- DashboardConfig: Configuration and thresholds management
- MetricsAggregator: Multi-system metrics collection  
- VisualizationEngine: Interactive Plotly chart generation
- Clean Flask application with real-time updates

Author: Agent Y - Supporting Agent Z STEELCLAD Protocol
"""

import asyncio
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Any, List
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit

# Import modular components
try:
    from .config.dashboard_config import DashboardConfig, get_dashboard_config
    from .metrics.metrics_aggregator import MetricsAggregator
    from .visualization.visualization_engine import VisualizationEngine
except ImportError:
    # Fallback to absolute imports
    from specialized.config.dashboard_config import DashboardConfig, get_dashboard_config
    from specialized.metrics.metrics_aggregator import MetricsAggregator
    from specialized.visualization.visualization_engine import VisualizationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class PerformanceAnalyticsDashboard:
    """
    Performance Analytics Dashboard - STEELCLAD Clean Version
    
    Coordinates modular components to provide comprehensive performance analytics
    with real-time visualizations and multi-system integration.
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or get_dashboard_config()
        self.logger = logging.getLogger('PerformanceAnalyticsDashboard')
        
        # Initialize modular components
        self.metrics_aggregator = MetricsAggregator(self.config)
        self.visualization_engine = VisualizationEngine(self.config)
        
        # Flask application setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'performance_analytics_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Background metrics collection
        self.metrics_thread = None
        self.running = False
        
        # Setup routes and WebSocket handlers
        self._setup_routes()
        self._setup_websocket_handlers()
        
        self.logger.info("Performance Analytics Dashboard initialized (STEELCLAD)")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            try:
                # Get latest metrics
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                current_metrics = loop.run_until_complete(
                    self.metrics_aggregator.collect_all_metrics()
                )
                loop.close()
                
                # Generate visualizations
                charts = self._generate_all_charts(current_metrics)
                
                # Return complete dashboard
                return self.visualization_engine.generate_dashboard_html(charts)
                
            except Exception as e:
                self.logger.error(f"Dashboard rendering error: {e}")
                return f"<h1>Dashboard Error</h1><p>{str(e)}</p>"
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """API endpoint for current metrics"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                metrics = loop.run_until_complete(
                    self.metrics_aggregator.collect_all_metrics()
                )
                loop.close()
                return jsonify(metrics)
            except Exception as e:
                self.logger.error(f"Metrics API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/history/<metric_name>')
        def get_metric_history(metric_name):
            """API endpoint for metric history"""
            try:
                limit = request.args.get('limit', 100, type=int)
                history = self.metrics_aggregator.get_metrics_history(metric_name, limit)
                return jsonify(history)
            except Exception as e:
                self.logger.error(f"History API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/correlations')
        def get_correlations():
            """API endpoint for metric correlations"""
            try:
                metrics_list = request.args.getlist('metrics')
                if not metrics_list:
                    metrics_list = [
                        'performance_monitoring.response_time',
                        'caching_system.hit_ratio',
                        'ml_optimizer.predictions_count'
                    ]
                
                correlations = self.metrics_aggregator.calculate_correlations(metrics_list)
                return jsonify(correlations)
            except Exception as e:
                self.logger.error(f"Correlations API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0-steelclad'
            })
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.logger.info("Client connected to performance dashboard")
            emit('connected', {'message': 'Connected to Performance Analytics Dashboard'})
        
        @self.socketio.on('disconnect')  
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info("Client disconnected from performance dashboard")
        
        @self.socketio.on('request_metrics')
        def handle_metrics_request():
            """Handle real-time metrics request"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                metrics = loop.run_until_complete(
                    self.metrics_aggregator.collect_all_metrics()
                )
                loop.close()
                emit('metrics_update', metrics)
            except Exception as e:
                self.logger.error(f"WebSocket metrics error: {e}")
                emit('error', {'message': str(e)})
    
    def _generate_all_charts(self, metrics_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate all dashboard charts"""
        try:
            charts = {}
            
            # System health gauge
            health_status = metrics_data.get('system_health', 'unknown')
            health_score = self._calculate_health_score(metrics_data)
            charts['health_gauge'] = self.visualization_engine.create_system_health_gauge(
                health_score, health_status
            )
            
            # Performance overview
            charts['overview'] = self.visualization_engine.create_performance_overview_chart(
                metrics_data
            )
            
            # Real-time metrics
            history_data = self.metrics_aggregator.get_metrics_history(limit=50)
            charts['realtime'] = self.visualization_engine.create_real_time_metrics_chart(
                history_data
            )
            
            # ML predictions
            ml_data = metrics_data.get('ml_optimizer', {})
            if 'predictions' in ml_data:
                charts['predictions'] = self.visualization_engine.create_predictions_chart(
                    ml_data['predictions']
                )
            else:
                charts['predictions'] = "<div>No predictions available</div>"
            
            # Correlation analysis
            metrics_list = [
                'performance_monitoring.response_time',
                'caching_system.hit_ratio',
                'distributed_scaling.avg_response_time_ms'
            ]
            correlations = self.metrics_aggregator.calculate_correlations(metrics_list)
            charts['correlation'] = self.visualization_engine.create_correlation_heatmap(
                correlations
            )
            
            # Performance trends
            trends_data = self._prepare_trends_data()
            charts['trends'] = self.visualization_engine.create_performance_trends_chart(
                trends_data
            )
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Chart generation error: {e}")
            return {}
    
    def _calculate_health_score(self, metrics_data: Dict[str, Any]) -> float:
        """Calculate normalized health score (0.0 to 1.0)"""
        try:
            scores = []
            
            # Performance monitoring score
            perf_data = metrics_data.get('performance_monitoring', {})
            if perf_data:
                scores.append(0.8)  # Base score if data available
            
            # Caching system score
            cache_data = metrics_data.get('caching_system', {})
            if cache_data:
                hit_ratio = cache_data.get('hit_ratio', 0.8)
                scores.append(min(hit_ratio * 1.1, 1.0))
            
            # Alpha systems score
            alpha_data = metrics_data.get('alpha_monitoring', {})
            if alpha_data:
                total_tests = alpha_data.get('total_tests', 1)
                passed_tests = alpha_data.get('passed_tests', 0)
                test_ratio = passed_tests / max(total_tests, 1)
                scores.append(test_ratio)
            
            # ML optimizer score  
            ml_data = metrics_data.get('ml_optimizer', {})
            if ml_data:
                scores.append(0.7)  # Base score for ML availability
            
            return sum(scores) / max(len(scores), 1) if scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Health score calculation error: {e}")
            return 0.5
    
    def _prepare_trends_data(self) -> Dict[str, Any]:
        """Prepare data for trends analysis"""
        try:
            history = self.metrics_aggregator.get_metrics_history('all_metrics', limit=50)
            all_metrics_history = history.get('all_metrics', [])
            
            if not all_metrics_history:
                return {}
            
            # Extract performance and resource trends
            performance_history = []
            resource_history = []
            
            for entry in all_metrics_history:
                if isinstance(entry, dict):
                    # Performance score
                    perf_score = self._calculate_health_score(entry) * 100
                    performance_history.append({'overall_score': perf_score})
                    
                    # Resource utilization
                    cache_data = entry.get('caching_system', {})
                    resource_history.append({
                        'cpu_usage': random.randint(30, 70),  # Simulated
                        'memory_usage': cache_data.get('memory_utilization', 50)
                    })
            
            return {
                'performance_history': performance_history,
                'resource_history': resource_history
            }
            
        except Exception as e:
            self.logger.error(f"Trends data preparation error: {e}")
            return {}
    
    def start_background_collection(self):
        """Start background metrics collection"""
        if self.metrics_thread and self.metrics_thread.is_alive():
            return
        
        self.running = True
        self.metrics_thread = threading.Thread(target=self._background_metrics_loop, daemon=True)
        self.metrics_thread.start()
        self.logger.info("Background metrics collection started")
    
    def _background_metrics_loop(self):
        """Background metrics collection loop"""
        while self.running:
            try:
                # Collect metrics
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                metrics = loop.run_until_complete(
                    self.metrics_aggregator.collect_all_metrics()
                )
                loop.close()
                
                # Broadcast via WebSocket if enabled
                if self.config.enable_real_time:
                    self.socketio.emit('metrics_broadcast', metrics)
                
                # Wait for next collection
                time.sleep(self.config.auto_refresh_seconds)
                
            except Exception as e:
                self.logger.error(f"Background collection error: {e}")
                time.sleep(self.config.auto_refresh_seconds * 2)  # Wait longer on error
    
    def stop_background_collection(self):
        """Stop background metrics collection"""
        self.running = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        self.logger.info("Background metrics collection stopped")
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the dashboard application"""
        try:
            # Use config values as defaults
            host = host or self.config.host
            port = port or self.config.port
            debug = debug if debug is not None else self.config.debug
            
            # Start background collection
            self.start_background_collection()
            
            # Run the application
            self.logger.info(f"Starting Performance Analytics Dashboard on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
            
        except KeyboardInterrupt:
            self.logger.info("Dashboard shutdown requested")
        finally:
            self.stop_background_collection()


def main():
    """Launch the performance analytics dashboard (STEELCLAD)"""
    try:
        print("🚀 Performance Analytics Dashboard - STEELCLAD Version")
        print("=" * 60)
        print("Agent Y supporting Agent Z: Modular Performance Analytics")
        print("=" * 60)
        
        # Load configuration
        config = get_dashboard_config()
        
        # Validate configuration
        if not config.validate_config():
            print("❌ Invalid configuration detected")
            return
        
        print(f"📊 Dashboard URL: http://{config.host}:{config.port}")
        print(f"⚡ Features: Real-time={config.enable_real_time}, Predictions={config.enable_predictions}")
        print(f"🔗 Integrations: Alpha={config.enable_alpha_integration}")
        print()
        
        # Create and run dashboard
        dashboard = PerformanceAnalyticsDashboard(config)
        dashboard.run()
        
    except Exception as e:
        logging.error(f"Dashboard startup error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()