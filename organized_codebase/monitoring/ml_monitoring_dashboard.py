"""
Enterprise ML Monitoring Dashboard
Advanced real-time monitoring and visualization system for all 19 ML modules

This module provides comprehensive monitoring capabilities including:
- Real-time performance metrics visualization
- Predictive health scoring
- Resource utilization tracking
- Automated alert management
- Interactive dashboard interface
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import threading
import time
from collections import defaultdict, deque
import statistics
import numpy as np
from pathlib import Path

# Web framework for dashboard
try:
    from flask import Flask, render_template, jsonify, request, websocket
    from flask_socketio import SocketIO, emit
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

@dataclass
class MetricSnapshot:
    """Real-time metric snapshot for dashboard visualization"""
    timestamp: datetime
    module_name: str
    metric_type: str
    value: float
    threshold_status: str  # 'normal', 'warning', 'critical'
    trend: str  # 'increasing', 'decreasing', 'stable'
    prediction: Optional[float] = None
    confidence: Optional[float] = None

@dataclass
class SystemAlert:
    """System-wide alert for monitoring dashboard"""
    alert_id: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    module_name: str
    message: str
    timestamp: datetime
    auto_resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class ResourceMetrics:
    """Comprehensive resource utilization metrics"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_io: float
    network_io: float
    thread_count: int
    active_models: int
    queue_depth: int

class MLMonitoringDashboard:
    """
    Enterprise ML Monitoring Dashboard
    
    Provides real-time monitoring, visualization, and alerting for all 19 ML modules
    with advanced analytics and predictive capabilities.
    """
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        self.config_path = config_path
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = deque(maxlen=500)
        self.system_health = {}
        self.predictive_models = {}
        self.dashboard_clients = set()
        
        # Enterprise monitoring configuration
        self.monitoring_config = {
            "refresh_interval": 1.0,  # seconds
            "prediction_horizon": 300,  # 5 minutes
            "alert_cooldown": 60,  # seconds
            "health_score_weights": {
                "performance": 0.3,
                "resource_usage": 0.2,
                "error_rate": 0.25,
                "prediction_confidence": 0.25
            },
            "thresholds": {
                "cpu_critical": 90.0,
                "memory_critical": 85.0,
                "error_rate_warning": 5.0,
                "error_rate_critical": 15.0,
                "response_time_warning": 1000,  # ms
                "response_time_critical": 5000  # ms
            }
        }
        
        # Initialize ML modules registry
        self.ml_modules = {
            "anomaly_detector": {"status": "active", "last_update": datetime.now()},
            "smart_cache": {"status": "active", "last_update": datetime.now()},
            "correlation_engine": {"status": "active", "last_update": datetime.now()},
            "batch_processor": {"status": "active", "last_update": datetime.now()},
            "predictive_engine": {"status": "active", "last_update": datetime.now()},
            "performance_optimizer": {"status": "active", "last_update": datetime.now()},
            "circuit_breaker": {"status": "active", "last_update": datetime.now()},
            "delivery_optimizer": {"status": "active", "last_update": datetime.now()},
            "integrity_guardian": {"status": "active", "last_update": datetime.now()},
            "sla_optimizer": {"status": "active", "last_update": datetime.now()},
            "adaptive_load_balancer": {"status": "active", "last_update": datetime.now()},
            "intelligent_scheduler": {"status": "active", "last_update": datetime.now()},
            "resource_optimizer": {"status": "active", "last_update": datetime.now()},
            "failure_predictor": {"status": "active", "last_update": datetime.now()},
            "quality_monitor": {"status": "active", "last_update": datetime.now()},
            "scaling_coordinator": {"status": "active", "last_update": datetime.now()},
            "telemetry_analyzer": {"status": "active", "last_update": datetime.now()},
            "security_monitor": {"status": "active", "last_update": datetime.now()},
            "compliance_auditor": {"status": "active", "last_update": datetime.now()}
        }
        
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = True
        self.web_app = None
        self.socketio = None
        
        self._initialize_dashboard()
        self._start_monitoring_threads()
    
    def _initialize_dashboard(self):
        """Initialize the web dashboard interface"""
        if not WEB_AVAILABLE:
            self.logger.warning("Flask not available. Dashboard will run in console mode only.")
            return
        
        self.web_app = Flask(__name__)
        self.web_app.config['SECRET_KEY'] = 'ml_dashboard_secret'
        self.socketio = SocketIO(self.web_app, cors_allowed_origins="*")
        
        self._setup_routes()
        self._setup_websocket_handlers()
    
    def _setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.web_app.route('/')
        def dashboard():
            return self._render_dashboard_template()
        
        @self.web_app.route('/api/metrics')
        def get_metrics():
            return jsonify(self._get_current_metrics())
        
        @self.web_app.route('/api/alerts')
        def get_alerts():
            return jsonify([asdict(alert) for alert in list(self.alerts)])
        
        @self.web_app.route('/api/health')
        def get_health():
            return jsonify(self._calculate_system_health())
        
        @self.web_app.route('/api/modules')
        def get_modules():
            return jsonify(self.ml_modules)
        
        @self.web_app.route('/api/predictions')
        def get_predictions():
            return jsonify(self._get_system_predictions())
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.dashboard_clients.add(request.sid)
            emit('status', {'msg': 'Connected to ML monitoring dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.dashboard_clients.discard(request.sid)
        
        @self.socketio.on('request_update')
        def handle_update_request():
            self._broadcast_dashboard_update()
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads"""
        
        # Metrics collection thread
        metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        metrics_thread.start()
        
        # Health monitoring thread
        health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        health_thread.start()
        
        # Predictive analysis thread
        prediction_thread = threading.Thread(target=self._predictive_analysis_loop, daemon=True)
        prediction_thread.start()
        
        # Dashboard broadcast thread
        if self.socketio:
            broadcast_thread = threading.Thread(target=self._dashboard_broadcast_loop, daemon=True)
            broadcast_thread.start()
    
    def _metrics_collection_loop(self):
        """Continuous metrics collection from all ML modules"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                for module_name in self.ml_modules:
                    # Simulate real metrics collection (in production, this would connect to actual modules)
                    metrics = self._collect_module_metrics(module_name)
                    
                    for metric_type, value in metrics.items():
                        snapshot = MetricSnapshot(
                            timestamp=current_time,
                            module_name=module_name,
                            metric_type=metric_type,
                            value=value,
                            threshold_status=self._evaluate_threshold(metric_type, value),
                            trend=self._calculate_trend(module_name, metric_type, value)
                        )
                        
                        self.metrics_buffer[f"{module_name}_{metric_type}"].append(snapshot)
                
                time.sleep(self.monitoring_config["refresh_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)
    
    def _collect_module_metrics(self, module_name: str) -> Dict[str, float]:
        """Collect comprehensive metrics for a specific ML module"""
        
        # Simulate realistic ML module metrics
        base_time = time.time()
        noise = np.random.normal(0, 0.1)
        
        metrics = {
            "cpu_usage": max(0, min(100, 45 + 20 * np.sin(base_time / 60) + noise * 10)),
            "memory_usage": max(0, min(100, 60 + 15 * np.cos(base_time / 120) + noise * 8)),
            "gpu_usage": max(0, min(100, 70 + 25 * np.sin(base_time / 90) + noise * 12)),
            "response_time": max(0, 200 + 100 * np.sin(base_time / 30) + noise * 50),
            "throughput": max(0, 1000 + 200 * np.cos(base_time / 45) + noise * 100),
            "error_rate": max(0, min(20, 2 + 3 * np.sin(base_time / 180) + noise * 2)),
            "prediction_accuracy": max(0, min(100, 92 + 5 * np.cos(base_time / 300) + noise * 3)),
            "cache_hit_rate": max(0, min(100, 85 + 10 * np.sin(base_time / 150) + noise * 5))
        }
        
        return metrics
    
    def _evaluate_threshold(self, metric_type: str, value: float) -> str:
        """Evaluate metric against configured thresholds"""
        thresholds = self.monitoring_config["thresholds"]
        
        if metric_type == "cpu_usage" and value > thresholds["cpu_critical"]:
            return "critical"
        elif metric_type == "memory_usage" and value > thresholds["memory_critical"]:
            return "critical"
        elif metric_type == "error_rate":
            if value > thresholds["error_rate_critical"]:
                return "critical"
            elif value > thresholds["error_rate_warning"]:
                return "warning"
        elif metric_type == "response_time":
            if value > thresholds["response_time_critical"]:
                return "critical"
            elif value > thresholds["response_time_warning"]:
                return "warning"
        
        return "normal"
    
    def _calculate_trend(self, module_name: str, metric_type: str, current_value: float) -> str:
        """Calculate trend for metric based on recent history"""
        key = f"{module_name}_{metric_type}"
        recent_values = list(self.metrics_buffer[key])[-5:]
        
        if len(recent_values) < 3:
            return "stable"
        
        values = [snapshot.value for snapshot in recent_values]
        values.append(current_value)
        
        # Calculate trend using linear regression slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.5:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _health_monitoring_loop(self):
        """Continuous health monitoring and alerting"""
        while self.monitoring_active:
            try:
                self._update_system_health()
                self._check_and_generate_alerts()
                time.sleep(5)  # Health check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(10)
    
    def _update_system_health(self):
        """Update comprehensive system health scores"""
        for module_name in self.ml_modules:
            health_score = self._calculate_module_health(module_name)
            self.system_health[module_name] = health_score
            
            # Update module status based on health
            if health_score > 80:
                self.ml_modules[module_name]["status"] = "healthy"
            elif health_score > 60:
                self.ml_modules[module_name]["status"] = "degraded"
            else:
                self.ml_modules[module_name]["status"] = "critical"
            
            self.ml_modules[module_name]["last_update"] = datetime.now()
            self.ml_modules[module_name]["health_score"] = health_score
    
    def _calculate_module_health(self, module_name: str) -> float:
        """Calculate comprehensive health score for a module"""
        weights = self.monitoring_config["health_score_weights"]
        
        # Get recent metrics
        recent_metrics = {}
        for metric_type in ["cpu_usage", "memory_usage", "error_rate", "response_time"]:
            key = f"{module_name}_{metric_type}"
            if self.metrics_buffer[key]:
                recent_metrics[metric_type] = self.metrics_buffer[key][-1].value
        
        if not recent_metrics:
            return 50.0  # Default score when no metrics available
        
        # Performance score (inverted for response time and error rate)
        performance_score = 100.0
        if "response_time" in recent_metrics:
            performance_score -= min(50, recent_metrics["response_time"] / 100)
        if "error_rate" in recent_metrics:
            performance_score -= min(30, recent_metrics["error_rate"] * 3)
        
        # Resource usage score
        resource_score = 100.0
        if "cpu_usage" in recent_metrics:
            resource_score -= recent_metrics["cpu_usage"] * 0.8
        if "memory_usage" in recent_metrics:
            resource_score -= recent_metrics["memory_usage"] * 0.6
        
        # Error rate score
        error_score = 100.0
        if "error_rate" in recent_metrics:
            error_score = max(0, 100 - recent_metrics["error_rate"] * 5)
        
        # Prediction confidence score (simulated)
        prediction_score = 85.0 + np.random.normal(0, 5)
        
        # Weighted health score
        health_score = (
            performance_score * weights["performance"] +
            resource_score * weights["resource_usage"] +
            error_score * weights["error_rate"] +
            prediction_score * weights["prediction_confidence"]
        )
        
        return max(0, min(100, health_score))
    
    def _check_and_generate_alerts(self):
        """Check conditions and generate alerts"""
        current_time = datetime.now()
        
        for module_name in self.ml_modules:
            health_score = self.system_health.get(module_name, 100)
            
            # Generate alerts based on health score
            if health_score < 30:
                self._create_alert("critical", module_name, 
                                 f"Critical health score: {health_score:.1f}%", current_time)
            elif health_score < 50:
                self._create_alert("warning", module_name, 
                                 f"Low health score: {health_score:.1f}%", current_time)
            
            # Check specific metric alerts
            self._check_metric_alerts(module_name, current_time)
    
    def _create_alert(self, severity: str, module_name: str, message: str, timestamp: datetime):
        """Create and store a new alert"""
        alert_id = f"{module_name}_{severity}_{int(timestamp.timestamp())}"
        
        # Check if similar alert exists in cooldown period
        cooldown = timedelta(seconds=self.monitoring_config["alert_cooldown"])
        recent_alerts = [a for a in self.alerts if 
                        a.module_name == module_name and 
                        a.severity == severity and 
                        timestamp - a.timestamp < cooldown]
        
        if recent_alerts:
            return  # Skip duplicate alert
        
        alert = SystemAlert(
            alert_id=alert_id,
            severity=severity,
            module_name=module_name,
            message=message,
            timestamp=timestamp
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Alert generated: {severity} - {module_name} - {message}")
    
    def _predictive_analysis_loop(self):
        """Continuous predictive analysis for proactive monitoring"""
        while self.monitoring_active:
            try:
                self._update_predictions()
                time.sleep(30)  # Predictions every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in predictive analysis: {e}")
                time.sleep(60)
    
    def _update_predictions(self):
        """Update predictive models and forecasts"""
        for module_name in self.ml_modules:
            for metric_type in ["cpu_usage", "memory_usage", "response_time"]:
                key = f"{module_name}_{metric_type}"
                
                if len(self.metrics_buffer[key]) > 10:
                    prediction = self._generate_metric_prediction(key)
                    
                    # Update latest metric with prediction
                    if self.metrics_buffer[key]:
                        latest_metric = self.metrics_buffer[key][-1]
                        latest_metric.prediction = prediction["value"]
                        latest_metric.confidence = prediction["confidence"]
    
    def _generate_metric_prediction(self, metric_key: str) -> Dict[str, float]:
        """Generate prediction for a specific metric"""
        recent_metrics = list(self.metrics_buffer[metric_key])[-20:]
        values = [m.value for m in recent_metrics]
        
        if len(values) < 5:
            return {"value": values[-1] if values else 0, "confidence": 0.5}
        
        # Simple trend-based prediction
        x = np.array(range(len(values)))
        coeffs = np.polyfit(x, values, min(2, len(values) - 1))
        
        # Predict next value
        next_x = len(values)
        predicted_value = np.polyval(coeffs, next_x)
        
        # Calculate confidence based on recent variance
        variance = np.var(values[-5:])
        confidence = max(0.1, min(0.95, 1.0 - variance / 1000))
        
        return {"value": float(predicted_value), "confidence": float(confidence)}
    
    def _dashboard_broadcast_loop(self):
        """Continuous dashboard updates via WebSocket"""
        while self.monitoring_active and self.socketio:
            try:
                if self.dashboard_clients:
                    self._broadcast_dashboard_update()
                time.sleep(2)  # Broadcast every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error in dashboard broadcast: {e}")
                time.sleep(5)
    
    def _broadcast_dashboard_update(self):
        """Broadcast current system state to all dashboard clients"""
        if not self.socketio or not self.dashboard_clients:
            return
        
        update_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self._get_current_metrics(),
            "health": self._calculate_system_health(),
            "alerts": [asdict(alert) for alert in list(self.alerts)[-10:]],  # Last 10 alerts
            "modules": self.ml_modules,
            "predictions": self._get_system_predictions()
        }
        
        self.socketio.emit('dashboard_update', update_data)
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot for all modules"""
        current_metrics = {}
        
        for module_name in self.ml_modules:
            module_metrics = {}
            for metric_type in ["cpu_usage", "memory_usage", "response_time", "error_rate"]:
                key = f"{module_name}_{metric_type}"
                if self.metrics_buffer[key]:
                    latest = self.metrics_buffer[key][-1]
                    module_metrics[metric_type] = {
                        "value": latest.value,
                        "threshold_status": latest.threshold_status,
                        "trend": latest.trend,
                        "prediction": latest.prediction,
                        "confidence": latest.confidence
                    }
            
            current_metrics[module_name] = module_metrics
        
        return current_metrics
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics"""
        if not self.system_health:
            return {"overall_score": 100, "module_scores": {}, "status": "healthy"}
        
        overall_score = statistics.mean(self.system_health.values())
        
        status = "healthy"
        if overall_score < 60:
            status = "critical"
        elif overall_score < 80:
            status = "degraded"
        
        return {
            "overall_score": round(overall_score, 1),
            "module_scores": self.system_health,
            "status": status,
            "active_modules": len([m for m in self.ml_modules.values() if m["status"] == "active"]),
            "total_modules": len(self.ml_modules)
        }
    
    def _get_system_predictions(self) -> Dict[str, Any]:
        """Get system-wide predictions and forecasts"""
        predictions = {}
        
        for module_name in self.ml_modules:
            module_predictions = {}
            for metric_type in ["cpu_usage", "memory_usage", "response_time"]:
                key = f"{module_name}_{metric_type}"
                if self.metrics_buffer[key] and self.metrics_buffer[key][-1].prediction:
                    module_predictions[metric_type] = {
                        "predicted_value": self.metrics_buffer[key][-1].prediction,
                        "confidence": self.metrics_buffer[key][-1].confidence,
                        "horizon_minutes": 5
                    }
            
            if module_predictions:
                predictions[module_name] = module_predictions
        
        return predictions
    
    def run_dashboard(self, host: str = "localhost", port: int = 5000, debug: bool = False):
        """Run the web dashboard server"""
        if not self.web_app or not self.socketio:
            self.logger.error("Web dashboard not available. Install Flask and Flask-SocketIO.")
            return
        
        self.logger.info(f"Starting ML monitoring dashboard on http://{host}:{port}")
        self.socketio.run(self.web_app, host=host, port=port, debug=debug)
    
    def get_console_summary(self) -> str:
        """Get a console-friendly summary of system status"""
        health = self._calculate_system_health()
        recent_alerts = list(self.alerts)[-5:]
        
        summary = [
            "="*60,
            "ML MONITORING DASHBOARD - SYSTEM SUMMARY",
            "="*60,
            f"Overall Health: {health['overall_score']}% ({health['status'].upper()})",
            f"Active Modules: {health['active_modules']}/{health['total_modules']}",
            f"Recent Alerts: {len(recent_alerts)}",
            ""
        ]
        
        # Module status summary
        summary.append("MODULE STATUS:")
        for module_name, module_info in self.ml_modules.items():
            health_score = self.system_health.get(module_name, 0)
            status_indicator = "✓" if module_info["status"] == "active" else "⚠"
            summary.append(f"  {status_indicator} {module_name}: {health_score:.1f}%")
        
        summary.append("")
        
        # Recent alerts
        if recent_alerts:
            summary.append("RECENT ALERTS:")
            for alert in recent_alerts:
                severity_icon = {"info": "ℹ", "warning": "⚠", "error": "⚠", "critical": "🚨"}
                icon = severity_icon.get(alert.severity, "•")
                summary.append(f"  {icon} {alert.severity.upper()}: {alert.module_name} - {alert.message}")
        
        summary.append("="*60)
        
        return "\n".join(summary)
    
    def stop_monitoring(self):
        """Stop all monitoring processes"""
        self.monitoring_active = False
        self.logger.info("ML monitoring dashboard stopped")

# Dashboard template (simple HTML for basic visualization)
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>TestMaster ML Monitoring Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
        .alert-critical { background-color: #f8d7da; border-left: 4px solid #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TestMaster Enterprise ML Monitoring Dashboard</h1>
        <p>Real-time monitoring of 19 ML modules with predictive analytics</p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>System Health</h3>
            <div id="system-health">Loading...</div>
        </div>
        
        <div class="card">
            <h3>Module Status</h3>
            <div id="module-status">Loading...</div>
        </div>
        
        <div class="card">
            <h3>Recent Alerts</h3>
            <div id="alerts">Loading...</div>
        </div>
        
        <div class="card">
            <h3>Performance Metrics</h3>
            <canvas id="metricsChart" width="400" height="200"></canvas>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to monitoring dashboard');
        });
        
        socket.on('dashboard_update', function(data) {
            updateSystemHealth(data.health);
            updateModuleStatus(data.modules);
            updateAlerts(data.alerts);
        });
        
        function updateSystemHealth(health) {
            const element = document.getElementById('system-health');
            const statusClass = health.status === 'healthy' ? 'status-healthy' : 
                               health.status === 'degraded' ? 'status-warning' : 'status-critical';
            
            element.textContent = `
                <div class="metric">
                    <span>Overall Score:</span>
                    <span class="${statusClass}">${health.overall_score}%</span>
                </div>
                <div class="metric">
                    <span>Status:</span>
                    <span class="${statusClass}">${health.status.toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span>Active Modules:</span>
                    <span>${health.active_modules}/${health.total_modules}</span>
                </div>
            `;
        }
        
        function updateModuleStatus(modules) {
            const element = document.getElementById('module-status');
            let html = '';
            
            for (const [name, info] of Object.entries(modules)) {
                const statusClass = info.status === 'active' ? 'status-healthy' : 'status-critical';
                const healthScore = info.health_score || 0;
                html += `
                    <div class="metric">
                        <span>${name}:</span>
                        <span class="${statusClass}">${healthScore.toFixed(1)}%</span>
                    </div>
                `;
            }
            
            element.textContent = html;
        }
        
        function updateAlerts(alerts) {
            const element = document.getElementById('alerts');
            let html = '';
            
            if (alerts.length === 0) {
                html = '<p>No recent alerts</p>';
            } else {
                alerts.forEach(alert => {
                    const alertClass = alert.severity === 'critical' ? 'alert-critical' : 'alert-warning';
                    html += `
                        <div class="alert ${alertClass}">
                            <strong>${alert.severity.toUpperCase()}:</strong> ${alert.module_name} - ${alert.message}
                        </div>
                    `;
                });
            }
            
            element.textContent = html;
        }
        
        // Request updates every 5 seconds
        setInterval(() => {
            socket.emit('request_update');
        }, 5000);
    </script>
</body>
</html>
"""

def main():
    """Main function for standalone dashboard execution"""
    dashboard = MLMonitoringDashboard()
    
    try:
        # Print console summary every 30 seconds
        while True:
            print(dashboard.get_console_summary())
            time.sleep(30)
    except KeyboardInterrupt:
        dashboard.stop_monitoring()
        print("\nMonitoring dashboard stopped.")

if __name__ == "__main__":
    main()