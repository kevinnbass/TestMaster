#!/usr/bin/env python3
"""
ML Predictions Dashboard Integration - Agent A Hour 9
Real-time machine learning predictions integration for unified dashboard
Connects advanced predictive analytics engine to dashboard visualization
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from pathlib import Path
import threading
import time

from flask import Blueprint, jsonify, request
from flask_socketio import emit

# Import the advanced predictive analytics engine
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.dashboard.advanced_predictive_analytics import (
    AdvancedPredictiveAnalytics,
    MLPrediction,
    PredictionType,
    ConfidenceLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPredictionsIntegration:
    """
    Machine Learning Predictions Dashboard Integration
    
    Provides real-time ML predictions for dashboard visualization with:
    - Live prediction streaming via WebSocket
    - RESTful API endpoints for predictions
    - Automatic model training scheduler
    - Performance caching and optimization
    """
    
    def __init__(self, socketio=None):
        self.socketio = socketio
        self.analytics_engine = AdvancedPredictiveAnalytics()
        
        # Cache configuration
        self.prediction_cache = {}
        self.cache_ttl = 30  # seconds
        self.last_cache_time = {}
        
        # Performance metrics
        self.prediction_metrics = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_prediction_time': 0,
            'prediction_history': []
        }
        
        # Background tasks
        self.streaming_active = False
        self.training_scheduler_active = False
        
        logger.info("ML Predictions Integration initialized")
    
    def create_blueprint(self) -> Blueprint:
        """Create Flask blueprint with ML prediction endpoints"""
        ml_bp = Blueprint('ml_predictions', __name__)
        
        @ml_bp.route('/api/ml/predictions', methods=['GET'])
        def get_all_predictions():
            """Get all current ML predictions"""
            try:
                metrics = self._get_current_metrics()
                predictions = self._get_cached_predictions(metrics)
                
                return jsonify({
                    'status': 'success',
                    'predictions': predictions,
                    'cache_stats': {
                        'hits': self.prediction_metrics['cache_hits'],
                        'misses': self.prediction_metrics['cache_misses'],
                        'hit_rate': self._calculate_cache_hit_rate()
                    },
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to get predictions: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @ml_bp.route('/api/ml/predictions/<prediction_type>', methods=['GET'])
        def get_specific_prediction(prediction_type):
            """Get specific type of prediction"""
            try:
                metrics = self._get_current_metrics()
                
                if prediction_type == 'health':
                    prediction = self.analytics_engine.predict_health_trend(metrics)
                elif prediction_type == 'anomaly':
                    prediction = self.analytics_engine.detect_anomalies(metrics)
                elif prediction_type == 'performance':
                    prediction = self.analytics_engine.predict_performance(metrics)
                elif prediction_type == 'resource':
                    prediction = self.analytics_engine.predict_resource_utilization(metrics)
                else:
                    return jsonify({'status': 'error', 'message': 'Invalid prediction type'}), 400
                
                return jsonify({
                    'status': 'success',
                    'prediction': asdict(prediction),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to get {prediction_type} prediction: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @ml_bp.route('/api/ml/train', methods=['POST'])
        def trigger_training():
            """Trigger model training with current data"""
            try:
                # Get training data from request or use historical
                training_data = request.json.get('training_data') if request.json else None
                
                # Start training in background
                threading.Thread(
                    target=self._train_models_background,
                    args=(training_data,),
                    daemon=True
                ).start()
                
                return jsonify({
                    'status': 'success',
                    'message': 'Model training started in background'
                })
            except Exception as e:
                logger.error(f"Failed to start training: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @ml_bp.route('/api/ml/performance', methods=['GET'])
        def get_model_performance():
            """Get current model performance metrics"""
            try:
                performance = self.analytics_engine.model_performance
                
                return jsonify({
                    'status': 'success',
                    'model_performance': performance,
                    'prediction_metrics': self.prediction_metrics,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to get model performance: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @ml_bp.route('/api/ml/streaming/<action>', methods=['POST'])
        def control_streaming(action):
            """Control real-time prediction streaming"""
            try:
                if action == 'start':
                    self.start_prediction_streaming()
                    message = 'Prediction streaming started'
                elif action == 'stop':
                    self.stop_prediction_streaming()
                    message = 'Prediction streaming stopped'
                else:
                    return jsonify({'status': 'error', 'message': 'Invalid action'}), 400
                
                return jsonify({'status': 'success', 'message': message})
            except Exception as e:
                logger.error(f"Failed to {action} streaming: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        return ml_bp
    
    def start_prediction_streaming(self, interval: int = 5):
        """Start real-time prediction streaming via WebSocket"""
        if self.streaming_active:
            logger.warning("Prediction streaming already active")
            return
        
        self.streaming_active = True
        
        def stream_predictions():
            """Stream predictions in background"""
            while self.streaming_active:
                try:
                    metrics = self._get_current_metrics()
                    predictions = self._get_cached_predictions(metrics)
                    
                    # Emit via WebSocket if available
                    if self.socketio:
                        self.socketio.emit('ml_predictions_update', {
                            'predictions': predictions,
                            'timestamp': datetime.now().isoformat(),
                            'metrics': self.prediction_metrics
                        })
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    time.sleep(interval)
        
        # Start streaming thread
        threading.Thread(target=stream_predictions, daemon=True).start()
        logger.info(f"Started prediction streaming with {interval}s interval")
    
    def stop_prediction_streaming(self):
        """Stop real-time prediction streaming"""
        self.streaming_active = False
        logger.info("Stopped prediction streaming")
    
    def _get_cached_predictions(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get predictions with caching for performance"""
        cache_key = 'all_predictions'
        now = time.time()
        
        # Check cache
        if cache_key in self.prediction_cache:
            if cache_key in self.last_cache_time:
                if now - self.last_cache_time[cache_key] < self.cache_ttl:
                    self.prediction_metrics['cache_hits'] += 1
                    return self.prediction_cache[cache_key]
        
        # Cache miss - generate new predictions
        self.prediction_metrics['cache_misses'] += 1
        start_time = time.time()
        
        predictions = {}
        
        # Get all predictions
        try:
            health_pred = self.analytics_engine.predict_health_trend(metrics)
            predictions['health_trend'] = self._format_prediction(health_pred)
        except Exception as e:
            logger.error(f"Health prediction failed: {e}")
            predictions['health_trend'] = self._error_prediction('health_trend')
        
        try:
            anomaly_pred = self.analytics_engine.detect_anomalies(metrics)
            predictions['anomaly_detection'] = self._format_prediction(anomaly_pred)
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            predictions['anomaly_detection'] = self._error_prediction('anomaly_detection')
        
        try:
            perf_pred = self.analytics_engine.predict_performance(metrics)
            predictions['performance'] = self._format_prediction(perf_pred)
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            predictions['performance'] = self._error_prediction('performance')
        
        try:
            resource_pred = self.analytics_engine.predict_resource_utilization(metrics)
            predictions['resource_utilization'] = self._format_prediction(resource_pred)
        except Exception as e:
            logger.error(f"Resource prediction failed: {e}")
            predictions['resource_utilization'] = self._error_prediction('resource_utilization')
        
        # Update cache
        self.prediction_cache[cache_key] = predictions
        self.last_cache_time[cache_key] = now
        
        # Update metrics
        prediction_time = time.time() - start_time
        self._update_prediction_metrics(prediction_time)
        
        return predictions
    
    def _format_prediction(self, prediction: MLPrediction) -> Dict[str, Any]:
        """Format ML prediction for dashboard display"""
        pred_dict = asdict(prediction)
        
        # Add visualization-friendly fields
        pred_dict['confidence_percentage'] = prediction.confidence_score * 100
        pred_dict['confidence_color'] = self._get_confidence_color(prediction.confidence_level)
        pred_dict['trend_icon'] = self._get_trend_icon(prediction.predicted_value)
        
        # Format timestamp
        pred_dict['timestamp'] = prediction.timestamp.isoformat()
        
        # Add alert level
        pred_dict['alert_level'] = self._determine_alert_level(prediction)
        
        return pred_dict
    
    def _get_confidence_color(self, confidence: ConfidenceLevel) -> str:
        """Get color code for confidence level"""
        colors = {
            ConfidenceLevel.HIGH: '#28a745',      # Green
            ConfidenceLevel.MEDIUM: '#ffc107',   # Yellow
            ConfidenceLevel.LOW: '#fd7e14',       # Orange
            ConfidenceLevel.VERY_LOW: '#dc3545'   # Red
        }
        return colors.get(confidence, '#6c757d')  # Gray default
    
    def _get_trend_icon(self, value: float) -> str:
        """Get trend icon based on predicted value"""
        if value > 80:
            return '⬆️'  # Up arrow
        elif value > 60:
            return '➡️'  # Right arrow
        elif value > 40:
            return '⬇️'  # Down arrow
        else:
            return '⚠️'  # Warning
    
    def _determine_alert_level(self, prediction: MLPrediction) -> str:
        """Determine alert level based on prediction"""
        if prediction.prediction_type == PredictionType.ANOMALY_DETECTION:
            if prediction.predicted_value > 0.7:
                return 'critical'
            elif prediction.predicted_value > 0.5:
                return 'warning'
        elif prediction.prediction_type == PredictionType.PERFORMANCE_DEGRADATION:
            if prediction.predicted_value > 0.8:
                return 'critical'
            elif prediction.predicted_value > 0.6:
                return 'warning'
        elif prediction.prediction_type == PredictionType.HEALTH_TREND:
            if prediction.predicted_value < 40:
                return 'critical'
            elif prediction.predicted_value < 60:
                return 'warning'
        
        return 'normal'
    
    def _error_prediction(self, prediction_type: str) -> Dict[str, Any]:
        """Create error prediction when ML fails"""
        return {
            'prediction_type': prediction_type,
            'predicted_value': 0,
            'confidence_score': 0,
            'confidence_level': 'very_low',
            'confidence_percentage': 0,
            'confidence_color': '#dc3545',
            'trend_icon': '❌',
            'alert_level': 'error',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for predictions"""
        import psutil
        
        # Get CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Generate some historical data for variance calculations
        cpu_history = [cpu_percent + (i * 2 - 5) for i in range(10)]
        memory_history = [memory.percent + (i * 1.5 - 3) for i in range(10)]
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'avg_response_time': 100 + (cpu_percent * 2),  # Simulated
            'error_rate': max(0, cpu_percent / 50 - 1),    # Simulated
            'active_services': 15,
            'dependency_health': 95.0,
            'import_success_rate': 98.0,
            'cpu_history': cpu_history,
            'memory_history': memory_history,
            'response_time_spike': 1.0 if cpu_percent > 70 else 0,
            'error_rate_change': 0.1,
            'service_failures': 0,
            'dependency_changes': 0,
            'response_time_trend': 0.1,
            'throughput_change': -0.05,
            'error_rate_trend': 0.02,
            'cpu_trend': 0.05,
            'memory_pressure': memory.percent / 100,
            'queue_depth': 5,
            'cache_hit_rate': 85.0,
            'disk_usage': psutil.disk_usage('/').percent,
            'request_rate': 100,
            'active_connections': 10
        }
    
    def _train_models_background(self, training_data=None):
        """Train models in background thread"""
        try:
            logger.info("Starting background model training")
            
            # Save current metrics for training
            metrics = self._get_current_metrics()
            self.analytics_engine.save_metrics(metrics)
            
            # Train models
            self.analytics_engine.train_models(training_data)
            
            # Clear cache after training
            self.prediction_cache.clear()
            self.last_cache_time.clear()
            
            logger.info("Background model training complete")
            
            # Emit training complete event
            if self.socketio:
                self.socketio.emit('ml_training_complete', {
                    'status': 'success',
                    'performance': self.analytics_engine.model_performance,
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Background training failed: {e}")
            
            if self.socketio:
                self.socketio.emit('ml_training_complete', {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
    
    def _update_prediction_metrics(self, prediction_time: float):
        """Update prediction performance metrics"""
        self.prediction_metrics['total_predictions'] += 1
        
        # Update average prediction time
        current_avg = self.prediction_metrics['avg_prediction_time']
        total = self.prediction_metrics['total_predictions']
        self.prediction_metrics['avg_prediction_time'] = (
            (current_avg * (total - 1) + prediction_time) / total
        )
        
        # Keep last 100 prediction times
        self.prediction_metrics['prediction_history'].append({
            'time': prediction_time,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.prediction_metrics['prediction_history']) > 100:
            self.prediction_metrics['prediction_history'].pop(0)
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total = self.prediction_metrics['cache_hits'] + self.prediction_metrics['cache_misses']
        if total == 0:
            return 0.0
        return (self.prediction_metrics['cache_hits'] / total) * 100
    
    def start_training_scheduler(self, interval_hours: int = 6):
        """Start automatic model training scheduler"""
        if self.training_scheduler_active:
            logger.warning("Training scheduler already active")
            return
        
        self.training_scheduler_active = True
        
        def scheduled_training():
            """Run scheduled training"""
            while self.training_scheduler_active:
                time.sleep(interval_hours * 3600)  # Convert hours to seconds
                
                if self.training_scheduler_active:
                    logger.info("Starting scheduled model training")
                    self._train_models_background()
        
        threading.Thread(target=scheduled_training, daemon=True).start()
        logger.info(f"Started training scheduler with {interval_hours} hour interval")
    
    def stop_training_scheduler(self):
        """Stop automatic model training scheduler"""
        self.training_scheduler_active = False
        logger.info("Stopped training scheduler")


def create_ml_dashboard_app():
    """Create standalone ML predictions dashboard for testing"""
    from flask import Flask
    from flask_socketio import SocketIO
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'ml_predictions_secret'
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Initialize ML integration
    ml_integration = MLPredictionsIntegration(socketio)
    
    # Register blueprint
    app.register_blueprint(ml_integration.create_blueprint())
    
    # Add dashboard route
    @app.route('/')
    def ml_dashboard():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Predictions Dashboard</title>
            <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1400px; margin: 0 auto; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .predictions-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
                .prediction-card { background: white; padding: 20px; border-radius: 10px; 
                                 box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .prediction-value { font-size: 48px; font-weight: bold; margin: 20px 0; }
                .confidence-bar { height: 10px; background: #e0e0e0; border-radius: 5px; 
                                overflow: hidden; margin: 10px 0; }
                .confidence-fill { height: 100%; transition: width 0.5s, background 0.5s; }
                .alert-critical { border-left: 5px solid #dc3545; }
                .alert-warning { border-left: 5px solid #ffc107; }
                .alert-normal { border-left: 5px solid #28a745; }
                .metrics { display: flex; justify-content: space-around; margin-top: 20px; }
                .metric { text-align: center; }
                .metric-value { font-size: 24px; font-weight: bold; }
                .metric-label { color: #666; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 ML Predictions Dashboard</h1>
                    <p>Real-time machine learning predictions powered by Agent A</p>
                </div>
                
                <div class="predictions-grid">
                    <div class="prediction-card" id="health-card">
                        <h3>📊 Health Trend</h3>
                        <div class="prediction-value" id="health-value">--</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="health-confidence"></div>
                        </div>
                        <p id="health-features">Loading...</p>
                    </div>
                    
                    <div class="prediction-card" id="anomaly-card">
                        <h3>🔍 Anomaly Detection</h3>
                        <div class="prediction-value" id="anomaly-value">--</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="anomaly-confidence"></div>
                        </div>
                        <p id="anomaly-features">Loading...</p>
                    </div>
                    
                    <div class="prediction-card" id="performance-card">
                        <h3>⚡ Performance Prediction</h3>
                        <div class="prediction-value" id="performance-value">--</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="performance-confidence"></div>
                        </div>
                        <p id="performance-features">Loading...</p>
                    </div>
                    
                    <div class="prediction-card" id="resource-card">
                        <h3>💾 Resource Utilization</h3>
                        <div class="prediction-value" id="resource-value">--</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="resource-confidence"></div>
                        </div>
                        <p id="resource-features">Loading...</p>
                    </div>
                </div>
                
                <div class="prediction-card" style="margin-top: 20px;">
                    <h3>📈 Performance Metrics</h3>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value" id="total-predictions">0</div>
                            <div class="metric-label">Total Predictions</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="cache-hit-rate">0%</div>
                            <div class="metric-label">Cache Hit Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="avg-time">0ms</div>
                            <div class="metric-label">Avg Prediction Time</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="model-accuracy">0%</div>
                            <div class="metric-label">Model Accuracy</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                const socket = io();
                
                socket.on('connect', () => {
                    console.log('Connected to ML predictions stream');
                });
                
                socket.on('ml_predictions_update', (data) => {
                    updatePredictions(data.predictions);
                    updateMetrics(data.metrics);
                });
                
                function updatePredictions(predictions) {
                    // Update health trend
                    if (predictions.health_trend) {
                        updatePredictionCard('health', predictions.health_trend);
                    }
                    
                    // Update anomaly detection
                    if (predictions.anomaly_detection) {
                        updatePredictionCard('anomaly', predictions.anomaly_detection);
                    }
                    
                    // Update performance
                    if (predictions.performance) {
                        updatePredictionCard('performance', predictions.performance);
                    }
                    
                    // Update resource utilization
                    if (predictions.resource_utilization) {
                        updatePredictionCard('resource', predictions.resource_utilization);
                    }
                }
                
                function updatePredictionCard(type, prediction) {
                    const card = document.getElementById(`${type}-card`);
                    const value = document.getElementById(`${type}-value`);
                    const confidence = document.getElementById(`${type}-confidence`);
                    const features = document.getElementById(`${type}-features`);
                    
                    // Update value
                    if (type === 'anomaly') {
                        value.textContent = prediction.predicted_value > 0.5 ? '⚠️ ANOMALY' : '✅ NORMAL';
                    } else {
                        value.textContent = prediction.predicted_value.toFixed(1) + '%';
                    }
                    
                    // Update confidence bar
                    confidence.style.width = prediction.confidence_percentage + '%';
                    confidence.style.background = prediction.confidence_color;
                    
                    // Update alert level
                    card.className = 'prediction-card alert-' + prediction.alert_level;
                    
                    // Update features
                    if (prediction.feature_importance) {
                        const topFeatures = Object.entries(prediction.feature_importance)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3)
                            .map(([k, v]) => `${k}: ${v.toFixed(2)}`)
                            .join(' | ');
                        features.textContent = 'Key factors: ' + topFeatures;
                    }
                }
                
                function updateMetrics(metrics) {
                    if (metrics) {
                        document.getElementById('total-predictions').textContent = metrics.total_predictions || 0;
                        document.getElementById('cache-hit-rate').textContent = 
                            ((metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)) * 100).toFixed(1) + '%';
                        document.getElementById('avg-time').textContent = 
                            (metrics.avg_prediction_time * 1000).toFixed(0) + 'ms';
                    }
                }
                
                // Initial load
                fetch('/api/ml/predictions')
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            updatePredictions(data.predictions);
                        }
                    });
                
                // Start streaming
                fetch('/api/ml/streaming/start', { method: 'POST' });
            </script>
        </body>
        </html>
        '''
    
    return app, socketio, ml_integration


if __name__ == "__main__":
    # Test the ML predictions dashboard
    app, socketio, ml_integration = create_ml_dashboard_app()
    
    # Start prediction streaming
    ml_integration.start_prediction_streaming(interval=3)
    
    # Start training scheduler (every 6 hours)
    ml_integration.start_training_scheduler(interval_hours=6)
    
    print("\n=== ML Predictions Dashboard ===")
    print("Dashboard: http://localhost:5017")
    print("API Endpoints:")
    print("  - GET  /api/ml/predictions")
    print("  - GET  /api/ml/predictions/<type>")
    print("  - POST /api/ml/train")
    print("  - GET  /api/ml/performance")
    print("  - POST /api/ml/streaming/<start|stop>")
    print("\nWebSocket: Real-time predictions on 'ml_predictions_update' event")
    print("================================\n")
    
    socketio.run(app, host='0.0.0.0', port=5017, debug=False)