"""
Performance Dashboard for TestMaster

Inspired by PraisonAI's comprehensive dashboard patterns, this provides
real-time performance monitoring, analytics, and visualization for all
TestMaster operations.

Features:
- Real-time performance metrics
- Component-specific analytics
- HTTP-based dashboard interface
- Interactive charts and graphs
- Performance trend analysis
- Toggleable via feature flags
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import logging
from enum import Enum

try:
    from flask import Flask, jsonify, render_template_string, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from core.tracking_manager import get_tracking_manager
from core.monitoring_decorators import monitor_performance


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: datetime
    component: str
    operation: str
    metric_type: MetricType
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentStats:
    """Statistics for a specific component."""
    component_name: str
    operation_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0
    last_activity: Optional[datetime] = None
    recent_operations: List[str] = field(default_factory=list)


@dataclass
class DashboardPanel:
    """Dashboard panel configuration and data."""
    panel_id: str
    title: str
    panel_type: str  # "chart", "table", "metric", "log"
    data: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    last_updated: Optional[datetime] = None


class PerformanceDashboard:
    """
    Comprehensive performance dashboard for TestMaster.
    
    Provides real-time monitoring, analytics, and visualization
    of all TestMaster operations with HTTP-based interface.
    """
    
    def __init__(self, port: int = 8080, auto_refresh: int = 5):
        """
        Initialize performance dashboard.
        
        Args:
            port: HTTP server port
            auto_refresh: Auto-refresh interval in seconds
        """
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'performance_dashboard')
        
        if not self.enabled:
            return
        
        config = FeatureFlags.get_config('layer3_orchestration', 'performance_dashboard')
        self.port = config.get('port', port)
        self.auto_refresh = config.get('auto_refresh', auto_refresh)
        
        # Data storage
        self.metrics: deque = deque(maxlen=10000)
        self.component_stats: Dict[str, ComponentStats] = {}
        self.panels: Dict[str, DashboardPanel] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # Threading and state
        self.lock = threading.RLock()
        self.flask_app: Optional[Flask] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Integration components
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
            
        if FeatureFlags.is_enabled('layer2_monitoring', 'tracking_manager'):
            self.tracking_manager = get_tracking_manager()
        else:
            self.tracking_manager = None
        
        # Initialize Flask app if available
        if FLASK_AVAILABLE:
            self._setup_flask_app()
        
        # Setup default panels
        self._setup_default_panels()
        
        print("Performance dashboard initialized")
        print(f"   Port: {self.port}")
        print(f"   Auto-refresh: {self.auto_refresh}s")
        print(f"   Flask available: {FLASK_AVAILABLE}")
    
    def _setup_flask_app(self):
        """Setup Flask application for dashboard interface."""
        if not FLASK_AVAILABLE:
            return
        
        self.flask_app = Flask(__name__)
        self.flask_app.config['SECRET_KEY'] = 'testmaster_dashboard'
        
        # Dashboard routes
        @self.flask_app.route('/')
        def dashboard_home():
            return self._render_dashboard()
        
        @self.flask_app.route('/api/metrics')
        def api_metrics():
            return jsonify(self._get_metrics_data())
        
        @self.flask_app.route('/api/components')
        def api_components():
            return jsonify(self._get_components_data())
        
        @self.flask_app.route('/api/panels/<panel_id>')
        def api_panel(panel_id):
            return jsonify(self._get_panel_data(panel_id))
        
        @self.flask_app.route('/api/health')
        def api_health():
            return jsonify(self._get_health_data())
        
        @self.flask_app.route('/api/alerts')
        def api_alerts():
            return jsonify({"alerts": self.alerts})
    
    def _setup_default_panels(self):
        """Setup default dashboard panels."""
        # System overview panel
        self.add_panel(DashboardPanel(
            panel_id="system_overview",
            title="System Overview",
            panel_type="metric",
            config={
                "metrics": ["total_operations", "success_rate", "avg_response_time"],
                "refresh_interval": 1
            }
        ))
        
        # Component performance panel
        self.add_panel(DashboardPanel(
            panel_id="component_performance",
            title="Component Performance",
            panel_type="chart",
            config={
                "chart_type": "bar",
                "x_axis": "component",
                "y_axis": "avg_duration_ms"
            }
        ))
        
        # Real-time activity panel
        self.add_panel(DashboardPanel(
            panel_id="realtime_activity",
            title="Real-time Activity",
            panel_type="log",
            config={
                "max_entries": 50,
                "auto_scroll": True
            }
        ))
        
        # Performance trends panel
        self.add_panel(DashboardPanel(
            panel_id="performance_trends",
            title="Performance Trends",
            panel_type="chart",
            config={
                "chart_type": "line",
                "time_range": "1h",
                "metrics": ["response_time", "throughput"]
            }
        ))
    
    def start_server(self):
        """Start the dashboard HTTP server."""
        if not self.enabled or not FLASK_AVAILABLE:
            print("Dashboard server not available (Flask not installed or dashboard disabled)")
            return
        
        if self.is_running:
            print("Dashboard server already running")
            return
        
        self.is_running = True
        
        def run_server():
            try:
                self.flask_app.run(
                    host='0.0.0.0',
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            except Exception as e:
                print(f"Dashboard server error: {e}")
                self.is_running = False
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        print(f"Dashboard server started at http://localhost:{self.port}")
    
    def stop_server(self):
        """Stop the dashboard HTTP server."""
        self.is_running = False
        print("Dashboard server stopped")
    
    @monitor_performance(name="record_metric")
    def record_metric(self, component: str, operation: str, 
                     metric_type: MetricType, value: float,
                     tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """
        Record a performance metric.
        
        Args:
            component: Component name (e.g., 'test_generator', 'file_watcher')
            operation: Operation name (e.g., 'generate_test', 'file_change')
            metric_type: Type of metric
            value: Metric value
            tags: Optional tags for filtering
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            metric_type=metric_type,
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics.append(metric)
            
            # Update component statistics
            self._update_component_stats(metric)
            
            # Update shared state if enabled
            if self.shared_state:
                self.shared_state.increment("dashboard_metrics_recorded")
                self.shared_state.set(f"latest_metric_{component}", {
                    "operation": operation,
                    "value": value,
                    "timestamp": metric.timestamp.isoformat()
                })
    
    def _update_component_stats(self, metric: PerformanceMetric):
        """Update component statistics with new metric."""
        component = metric.component
        
        if component not in self.component_stats:
            self.component_stats[component] = ComponentStats(component_name=component)
        
        stats = self.component_stats[component]
        
        # Update counters
        stats.operation_count += 1
        stats.last_activity = metric.timestamp
        
        # Add to recent operations
        if len(stats.recent_operations) >= 10:
            stats.recent_operations.pop(0)
        stats.recent_operations.append(f"{metric.operation}:{metric.value:.2f}")
        
        # Update duration statistics for timer metrics
        if metric.metric_type == MetricType.TIMER:
            stats.total_duration_ms += metric.value
            stats.avg_duration_ms = stats.total_duration_ms / stats.operation_count
            stats.min_duration_ms = min(stats.min_duration_ms, metric.value)
            stats.max_duration_ms = max(stats.max_duration_ms, metric.value)
        
        # Update error tracking
        if metric.metadata.get('success', True):
            pass  # Success - no action needed
        else:
            stats.error_count += 1
        
        # Calculate success rate
        if stats.operation_count > 0:
            stats.success_rate = ((stats.operation_count - stats.error_count) / 
                                stats.operation_count) * 100
    
    def add_panel(self, panel: DashboardPanel):
        """Add a panel to the dashboard."""
        if not self.enabled:
            return
        
        with self.lock:
            self.panels[panel.panel_id] = panel
            panel.last_updated = datetime.now()
    
    def update_panel(self, panel_id: str, data: Dict[str, Any]):
        """Update panel data."""
        if not self.enabled:
            return
        
        with self.lock:
            if panel_id in self.panels:
                self.panels[panel_id].data.update(data)
                self.panels[panel_id].last_updated = datetime.now()
    
    def add_alert(self, severity: str, message: str, component: str = None):
        """Add an alert to the dashboard."""
        if not self.enabled:
            return
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,  # "info", "warning", "error", "critical"
            "message": message,
            "component": component,
            "id": f"alert_{int(time.time() * 1000)}"
        }
        
        with self.lock:
            self.alerts.append(alert)
            
            # Keep only recent alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-50:]
    
    def _get_metrics_data(self) -> Dict[str, Any]:
        """Get metrics data for API."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            recent_metrics = list(self.metrics)[-1000:]  # Last 1000 metrics
            
            metrics_by_component = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_component[metric.component].append({
                    "timestamp": metric.timestamp.isoformat(),
                    "operation": metric.operation,
                    "type": metric.metric_type.value,
                    "value": metric.value,
                    "tags": metric.tags
                })
            
            return {
                "enabled": True,
                "total_metrics": len(self.metrics),
                "recent_metrics": len(recent_metrics),
                "components": dict(metrics_by_component),
                "collection_time": datetime.now().isoformat()
            }
    
    def _get_components_data(self) -> Dict[str, Any]:
        """Get component statistics for API."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            components_data = {}
            for name, stats in self.component_stats.items():
                components_data[name] = asdict(stats)
                # Convert datetime to string for JSON serialization
                if stats.last_activity:
                    components_data[name]['last_activity'] = stats.last_activity.isoformat()
            
            return {
                "enabled": True,
                "component_count": len(self.component_stats),
                "components": components_data,
                "collection_time": datetime.now().isoformat()
            }
    
    def _get_panel_data(self, panel_id: str) -> Dict[str, Any]:
        """Get specific panel data for API."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            if panel_id not in self.panels:
                return {"error": "Panel not found"}
            
            panel = self.panels[panel_id]
            panel_data = asdict(panel)
            
            # Convert datetime to string
            if panel.last_updated:
                panel_data['last_updated'] = panel.last_updated.isoformat()
            
            return panel_data
    
    def _get_health_data(self) -> Dict[str, Any]:
        """Get system health data for API."""
        if not self.enabled:
            return {"enabled": False}
        
        current_time = datetime.now()
        
        # Calculate health metrics
        total_operations = sum(stats.operation_count for stats in self.component_stats.values())
        total_errors = sum(stats.error_count for stats in self.component_stats.values())
        overall_success_rate = ((total_operations - total_errors) / max(total_operations, 1)) * 100
        
        # Component health
        component_health = {}
        for name, stats in self.component_stats.items():
            health_score = stats.success_rate
            if stats.last_activity:
                time_since_activity = (current_time - stats.last_activity).total_seconds()
                if time_since_activity > 300:  # 5 minutes
                    health_score *= 0.8  # Reduce health if inactive
            
            component_health[name] = {
                "health_score": health_score,
                "status": "healthy" if health_score > 95 else "warning" if health_score > 80 else "critical",
                "last_activity": stats.last_activity.isoformat() if stats.last_activity else None
            }
        
        return {
            "enabled": True,
            "overall_health": {
                "score": overall_success_rate,
                "status": "healthy" if overall_success_rate > 95 else "warning" if overall_success_rate > 80 else "critical",
                "total_operations": total_operations,
                "total_errors": total_errors
            },
            "components": component_health,
            "uptime_seconds": (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds(),
            "check_time": current_time.isoformat()
        }
    
    def _render_dashboard(self) -> str:
        """Render the main dashboard HTML."""
        # Simple HTML template for the dashboard
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TestMaster Performance Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .header h1 { margin: 0; }
                .header p { margin: 5px 0 0 0; opacity: 0.8; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .panel { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .panel h3 { margin-top: 0; color: #2c3e50; }
                .metric { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee; }
                .metric:last-child { border-bottom: none; }
                .metric-value { font-weight: bold; font-size: 1.2em; color: #27ae60; }
                .component-list { list-style: none; padding: 0; margin: 0; }
                .component-item { padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 4px; }
                .status-healthy { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-critical { color: #e74c3c; }
                .auto-refresh { position: fixed; top: 10px; right: 10px; background: #3498db; color: white; padding: 5px 10px; border-radius: 4px; font-size: 12px; }
            </style>
            <script>
                // Auto-refresh every {{ auto_refresh }} seconds
                setTimeout(() => location.reload(), {{ auto_refresh }} * 1000);
                
                // Fetch and update data
                async function updateDashboard() {
                    try {
                        const [metrics, components, health] = await Promise.all([
                            fetch('/api/metrics').then(r => r.json()),
                            fetch('/api/components').then(r => r.json()),
                            fetch('/api/health').then(r => r.json())
                        ]);
                        
                        // Update displays would go here
                        console.log('Dashboard data updated', { metrics, components, health });
                    } catch (error) {
                        console.error('Failed to update dashboard:', error);
                    }
                }
                
                // Update immediately and then every 5 seconds
                document.addEventListener('DOMContentLoaded', () => {
                    updateDashboard();
                    setInterval(updateDashboard, 5000);
                });
            </script>
        </head>
        <body>
            <div class="auto-refresh">Auto-refresh: {{ auto_refresh }}s</div>
            
            <div class="header">
                <h1>TestMaster Performance Dashboard</h1>
                <p>Real-time monitoring and analytics • Last updated: {{ current_time }}</p>
            </div>
            
            <div class="dashboard">
                <div class="panel">
                    <h3>System Overview</h3>
                    <div class="metric">
                        <span>Total Components</span>
                        <span class="metric-value">{{ component_count }}</span>
                    </div>
                    <div class="metric">
                        <span>Total Metrics</span>
                        <span class="metric-value">{{ total_metrics }}</span>
                    </div>
                    <div class="metric">
                        <span>Dashboard Status</span>
                        <span class="metric-value status-healthy">Running</span>
                    </div>
                    <div class="metric">
                        <span>Auto Refresh</span>
                        <span class="metric-value">{{ auto_refresh }}s</span>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>Component Status</h3>
                    <ul class="component-list">
                        {% for component, stats in components.items() %}
                        <li class="component-item">
                            <strong>{{ component }}</strong><br>
                            Operations: {{ stats.operation_count }}<br>
                            Success Rate: <span class="status-{% if stats.success_rate > 95 %}healthy{% elif stats.success_rate > 80 %}warning{% else %}critical{% endif %}">{{ "%.1f"|format(stats.success_rate) }}%</span>
                        </li>
                        {% endfor %}
                    </ul>
                </div>
                
                <div class="panel">
                    <h3>Quick Actions</h3>
                    <p><a href="/api/metrics">View Raw Metrics</a></p>
                    <p><a href="/api/components">View Component Data</a></p>
                    <p><a href="/api/health">View Health Status</a></p>
                    <p><a href="/api/alerts">View Alerts</a></p>
                </div>
                
                <div class="panel">
                    <h3>Integration Status</h3>
                    <div class="metric">
                        <span>Shared State</span>
                        <span class="metric-value">{{ "Enabled" if shared_state else "Disabled" }}</span>
                    </div>
                    <div class="metric">
                        <span>Tracking Manager</span>
                        <span class="metric-value">{{ "Enabled" if tracking_manager else "Disabled" }}</span>
                    </div>
                    <div class="metric">
                        <span>Flask Available</span>
                        <span class="metric-value">{{ "Yes" if flask_available else "No" }}</span>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Template data
        template_data = {
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'auto_refresh': self.auto_refresh,
            'component_count': len(self.component_stats),
            'total_metrics': len(self.metrics),
            'components': {name: stats for name, stats in self.component_stats.items()},
            'shared_state': self.shared_state is not None,
            'tracking_manager': self.tracking_manager is not None,
            'flask_available': FLASK_AVAILABLE
        }
        
        # Simple template rendering (basic replacement)
        html = html_template
        for key, value in template_data.items():
            html = html.replace(f'{{{{ {key} }}}}', str(value))
        
        return html
    
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            stats = {
                "enabled": True,
                "server_running": self.is_running,
                "port": self.port,
                "auto_refresh": self.auto_refresh,
                "total_metrics": len(self.metrics),
                "total_components": len(self.component_stats),
                "total_panels": len(self.panels),
                "total_alerts": len(self.alerts),
                "flask_available": FLASK_AVAILABLE
            }
            
            # Add integration status
            if self.shared_state:
                shared_stats = self.shared_state.get_stats()
                stats["shared_state"] = shared_stats
            else:
                stats["shared_state"] = {"enabled": False}
            
            if self.tracking_manager:
                tracking_stats = self.tracking_manager.get_tracking_statistics()
                stats["tracking"] = tracking_stats
            else:
                stats["tracking"] = {"enabled": False}
            
            return stats


# Global dashboard instance
_dashboard: Optional[PerformanceDashboard] = None


def get_performance_dashboard() -> PerformanceDashboard:
    """Get the global performance dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = PerformanceDashboard()
    return _dashboard


def record_dashboard_metric(component: str, operation: str, value: float, 
                          metric_type: MetricType = MetricType.TIMER):
    """Convenience function to record a dashboard metric."""
    dashboard = get_performance_dashboard()
    if dashboard.enabled:
        dashboard.record_metric(component, operation, metric_type, value)


def dashboard_alert(severity: str, message: str, component: str = None):
    """Convenience function to add a dashboard alert."""
    dashboard = get_performance_dashboard()
    if dashboard.enabled:
        dashboard.add_alert(severity, message, component)