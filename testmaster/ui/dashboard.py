"""
TestMaster Real-time Web Dashboard

Inspired by Agency-Swarm's Gradio patterns for web-based UI
with real-time updates and queue-based state management.

Features:
- WebSocket-based live updates
- Module health visualization  
- Test execution status display
- Interactive controls and monitoring
"""

import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import webbrowser

try:
    import tornado.web
    import tornado.websocket
    import tornado.ioloop
    TORNADO_AVAILABLE = True
except ImportError:
    TORNADO_AVAILABLE = False
    print("⚠️ Tornado not available. Install with: pip install tornado")

from ..core.layer_manager import requires_layer


class DashboardTheme(Enum):
    """Dashboard visual themes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "localhost"
    port: int = 8080
    theme: DashboardTheme = DashboardTheme.DARK
    auto_refresh_seconds: int = 5
    max_history_items: int = 100
    show_debug_info: bool = False
    enable_notifications: bool = True


class DashboardData:
    """Central data store for dashboard."""
    
    def __init__(self):
        self.system_status = {}
        self.test_results = []
        self.coverage_data = {}
        self.idle_modules = []
        self.breaking_tests = []
        self.queue_stats = {}
        self.alerts = []
        self.last_updated = datetime.now()
        
        # Real-time metrics
        self.metrics_history = []
        self.performance_data = {}
        
        self._lock = threading.Lock()
    
    def update_system_status(self, status: Dict[str, Any]):
        """Update system status."""
        with self._lock:
            self.system_status = status
            self.last_updated = datetime.now()
    
    def add_test_result(self, result: Dict[str, Any]):
        """Add test result."""
        with self._lock:
            self.test_results.append({
                **result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only recent results
            if len(self.test_results) > 100:
                self.test_results = self.test_results[-100:]
    
    def update_coverage_data(self, coverage: Dict[str, Any]):
        """Update coverage data."""
        with self._lock:
            self.coverage_data = coverage
            self.last_updated = datetime.now()
    
    def update_idle_modules(self, modules: List[Dict[str, Any]]):
        """Update idle modules list."""
        with self._lock:
            self.idle_modules = modules
            self.last_updated = datetime.now()
    
    def update_breaking_tests(self, tests: List[Dict[str, Any]]):
        """Update breaking tests list."""
        with self._lock:
            self.breaking_tests = tests
            self.last_updated = datetime.now()
    
    def add_alert(self, alert: Dict[str, Any]):
        """Add alert."""
        with self._lock:
            self.alerts.append({
                **alert,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only recent alerts
            if len(self.alerts) > 50:
                self.alerts = self.alerts[-50:]
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get complete dashboard state."""
        with self._lock:
            return {
                'system_status': self.system_status,
                'test_results': self.test_results[-20:],  # Last 20 results
                'coverage_data': self.coverage_data,
                'idle_modules': self.idle_modules,
                'breaking_tests': self.breaking_tests,
                'queue_stats': self.queue_stats,
                'alerts': self.alerts[-10:],  # Last 10 alerts
                'last_updated': self.last_updated.isoformat(),
                'metrics_history': self.metrics_history[-50:],  # Last 50 metrics
                'performance_data': self.performance_data
            }


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    """WebSocket handler for real-time updates."""
    
    clients = set()
    
    def open(self):
        """Handle WebSocket connection open."""
        self.clients.add(self)
        print(f"📱 Dashboard client connected (total: {len(self.clients)})")
        
        # Send initial data
        initial_data = self.application.dashboard_data.get_dashboard_state()
        self.write_message({
            'type': 'initial_data',
            'data': initial_data
        })
    
    def on_close(self):
        """Handle WebSocket connection close."""
        self.clients.discard(self)
        print(f"📱 Dashboard client disconnected (total: {len(self.clients)})")
    
    def on_message(self, message):
        """Handle WebSocket message from client."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'request_update':
                # Send current state
                current_data = self.application.dashboard_data.get_dashboard_state()
                self.write_message({
                    'type': 'data_update',
                    'data': current_data
                })
            
            elif message_type == 'clear_alerts':
                # Clear alerts
                self.application.dashboard_data.alerts = []
                self.broadcast_update('alerts_cleared', {})
            
        except Exception as e:
            print(f"⚠️ Error handling WebSocket message: {e}")
    
    @classmethod
    def broadcast_update(cls, update_type: str, data: Any):
        """Broadcast update to all connected clients."""
        message = {
            'type': update_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to all connected clients
        disconnected = set()
        for client in cls.clients:
            try:
                client.write_message(message)
            except Exception as e:
                print(f"⚠️ Error sending to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        cls.clients -= disconnected


class DashboardHandler(tornado.web.RequestHandler):
    """Main dashboard page handler."""
    
    def get(self):
        """Serve dashboard HTML."""
        html_content = self._generate_dashboard_html()
        self.write(html_content)
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        config = self.application.config
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TestMaster Dashboard</title>
    <style>
        {self._get_css_styles(config.theme)}
    </style>
</head>
<body>
    <div id="app">
        <header class="header">
            <h1>🧪 TestMaster Dashboard</h1>
            <div class="status-indicator" id="connection-status">Connecting...</div>
        </header>
        
        <div class="dashboard-grid">
            <!-- System Status -->
            <div class="card">
                <h2>🖥️ System Status</h2>
                <div id="system-status">Loading...</div>
            </div>
            
            <!-- Test Results -->
            <div class="card">
                <h2>✅ Recent Test Results</h2>
                <div id="test-results">Loading...</div>
            </div>
            
            <!-- Coverage Data -->
            <div class="card">
                <h2>📊 Coverage Overview</h2>
                <div id="coverage-data">Loading...</div>
            </div>
            
            <!-- Idle Modules -->
            <div class="card">
                <h2>😴 Idle Modules</h2>
                <div id="idle-modules">Loading...</div>
            </div>
            
            <!-- Breaking Tests -->
            <div class="card alert-card">
                <h2>🚨 Breaking Tests</h2>
                <div id="breaking-tests">Loading...</div>
            </div>
            
            <!-- Alerts -->
            <div class="card">
                <h2>🔔 Recent Alerts</h2>
                <div id="alerts-list">Loading...</div>
                <button onclick="clearAlerts()" class="btn btn-secondary">Clear Alerts</button>
            </div>
        </div>
        
        <footer class="footer">
            <div>Last Updated: <span id="last-updated">Never</span></div>
            <div>Auto-refresh: {config.auto_refresh_seconds}s</div>
        </footer>
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
        """
    
    def _get_css_styles(self, theme: DashboardTheme) -> str:
        """Get CSS styles for the dashboard."""
        if theme == DashboardTheme.DARK:
            return """
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: #1a1a1a; color: #e0e0e0; 
                    line-height: 1.6;
                }
                .header { 
                    background: #2d2d2d; padding: 1rem; 
                    display: flex; justify-content: space-between; align-items: center;
                    border-bottom: 2px solid #444;
                }
                .status-indicator { 
                    padding: 0.5rem 1rem; border-radius: 20px; 
                    font-weight: bold; background: #333;
                }
                .status-indicator.connected { background: #0d7377; }
                .status-indicator.disconnected { background: #d63031; }
                .dashboard-grid { 
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                    gap: 1rem; padding: 1rem; 
                }
                .card { 
                    background: #2d2d2d; border-radius: 8px; padding: 1.5rem; 
                    border: 1px solid #444; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                }
                .card h2 { color: #74b9ff; margin-bottom: 1rem; }
                .alert-card { border-left: 4px solid #e17055; }
                .btn { 
                    padding: 0.5rem 1rem; border: none; border-radius: 4px; 
                    cursor: pointer; margin: 0.25rem;
                }
                .btn-secondary { background: #636e72; color: white; }
                .btn-secondary:hover { background: #2d3436; }
                .footer { 
                    background: #2d2d2d; padding: 1rem; text-align: center; 
                    border-top: 1px solid #444; margin-top: 2rem;
                }
                .metric { 
                    display: flex; justify-content: space-between; 
                    padding: 0.5rem; margin: 0.25rem 0; 
                    background: #3d3d3d; border-radius: 4px;
                }
                .metric-value { font-weight: bold; color: #00b894; }
                .alert-item { 
                    padding: 0.75rem; margin: 0.5rem 0; 
                    background: #e17055; color: white; border-radius: 4px;
                }
                .test-result { 
                    display: flex; justify-content: space-between; 
                    padding: 0.5rem; margin: 0.25rem 0; border-radius: 4px;
                }
                .test-passed { background: #00b894; color: white; }
                .test-failed { background: #d63031; color: white; }
                .idle-module { 
                    padding: 0.5rem; margin: 0.25rem 0; 
                    background: #fdcb6e; color: #2d3436; border-radius: 4px;
                }
            """
        else:  # Light theme
            return """
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: #f8f9fa; color: #343a40; 
                    line-height: 1.6;
                }
                .header { 
                    background: #ffffff; padding: 1rem; 
                    display: flex; justify-content: space-between; align-items: center;
                    border-bottom: 2px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .status-indicator { 
                    padding: 0.5rem 1rem; border-radius: 20px; 
                    font-weight: bold; background: #e9ecef;
                }
                .status-indicator.connected { background: #d4edda; color: #155724; }
                .status-indicator.disconnected { background: #f8d7da; color: #721c24; }
                .dashboard-grid { 
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                    gap: 1rem; padding: 1rem; 
                }
                .card { 
                    background: #ffffff; border-radius: 8px; padding: 1.5rem; 
                    border: 1px solid #dee2e6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .card h2 { color: #0056b3; margin-bottom: 1rem; }
                .alert-card { border-left: 4px solid #dc3545; }
                .btn { 
                    padding: 0.5rem 1rem; border: none; border-radius: 4px; 
                    cursor: pointer; margin: 0.25rem;
                }
                .btn-secondary { background: #6c757d; color: white; }
                .btn-secondary:hover { background: #5a6268; }
                .footer { 
                    background: #ffffff; padding: 1rem; text-align: center; 
                    border-top: 1px solid #dee2e6; margin-top: 2rem;
                }
            """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for dashboard functionality."""
        return """
            let ws = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log('Connected to TestMaster');
                    document.getElementById('connection-status').textContent = 'Connected';
                    document.getElementById('connection-status').className = 'status-indicator connected';
                    reconnectAttempts = 0;
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    handleMessage(message);
                };
                
                ws.onclose = function() {
                    console.log('Disconnected from TestMaster');
                    document.getElementById('connection-status').textContent = 'Disconnected';
                    document.getElementById('connection-status').className = 'status-indicator disconnected';
                    
                    // Attempt to reconnect
                    if (reconnectAttempts < maxReconnectAttempts) {
                        setTimeout(() => {
                            reconnectAttempts++;
                            connectWebSocket();
                        }, 2000 * reconnectAttempts);
                    }
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function handleMessage(message) {
                switch (message.type) {
                    case 'initial_data':
                    case 'data_update':
                        updateDashboard(message.data);
                        break;
                    case 'test_result':
                        updateTestResults([message.data]);
                        break;
                    case 'alert':
                        addAlert(message.data);
                        break;
                    case 'alerts_cleared':
                        document.getElementById('alerts-list').innerHTML = 'No alerts';
                        break;
                }
            }
            
            function updateDashboard(data) {
                updateSystemStatus(data.system_status);
                updateTestResults(data.test_results);
                updateCoverageData(data.coverage_data);
                updateIdleModules(data.idle_modules);
                updateBreakingTests(data.breaking_tests);
                updateAlerts(data.alerts);
                
                document.getElementById('last-updated').textContent = 
                    new Date(data.last_updated).toLocaleString();
            }
            
            function updateSystemStatus(status) {
                const container = document.getElementById('system-status');
                if (!status || Object.keys(status).length === 0) {
                    container.innerHTML = 'No status data available';
                    return;
                }
                
                let html = '';
                for (const [key, value] of Object.entries(status)) {
                    html += `
                        <div class="metric">
                            <span>${key.replace(/_/g, ' ').toUpperCase()}</span>
                            <span class="metric-value">${value}</span>
                        </div>
                    `;
                }
                container.innerHTML = html;
            }
            
            function updateTestResults(results) {
                const container = document.getElementById('test-results');
                if (!results || results.length === 0) {
                    container.innerHTML = 'No recent test results';
                    return;
                }
                
                let html = '';
                results.slice(-10).forEach(result => {
                    const status = result.status || result.result_code === 0 ? 'passed' : 'failed';
                    const statusClass = status === 'passed' ? 'test-passed' : 'test-failed';
                    html += `
                        <div class="test-result ${statusClass}">
                            <span>${result.test_name || result.test_path || 'Unknown Test'}</span>
                            <span>${status.toUpperCase()}</span>
                        </div>
                    `;
                });
                container.innerHTML = html;
            }
            
            function updateCoverageData(coverage) {
                const container = document.getElementById('coverage-data');
                if (!coverage || Object.keys(coverage).length === 0) {
                    container.innerHTML = 'No coverage data available';
                    return;
                }
                
                let html = '';
                if (coverage.overall_coverage !== undefined) {
                    html += `
                        <div class="metric">
                            <span>Overall Coverage</span>
                            <span class="metric-value">${coverage.overall_coverage.toFixed(1)}%</span>
                        </div>
                    `;
                }
                if (coverage.files_covered !== undefined) {
                    html += `
                        <div class="metric">
                            <span>Files Covered</span>
                            <span class="metric-value">${coverage.files_covered}</span>
                        </div>
                    `;
                }
                container.innerHTML = html || 'Coverage data available';
            }
            
            function updateIdleModules(modules) {
                const container = document.getElementById('idle-modules');
                if (!modules || modules.length === 0) {
                    container.innerHTML = 'No idle modules detected';
                    return;
                }
                
                let html = '';
                modules.slice(0, 10).forEach(module => {
                    const path = module.module_path || module.path || 'Unknown Module';
                    const fileName = path.split('/').pop() || path;
                    const idleHours = module.idle_duration_hours || module.idle_time || 0;
                    html += `
                        <div class="idle-module">
                            <strong>${fileName}</strong><br>
                            Idle for ${idleHours.toFixed(1)} hours
                        </div>
                    `;
                });
                container.innerHTML = html;
            }
            
            function updateBreakingTests(tests) {
                const container = document.getElementById('breaking-tests');
                if (!tests || tests.length === 0) {
                    container.innerHTML = '✅ No breaking tests';
                    return;
                }
                
                let html = '';
                tests.forEach(test => {
                    const testName = test.test || test.test_name || test.test_path || 'Unknown Test';
                    const error = test.failure || test.error_message || 'Unknown error';
                    html += `
                        <div class="alert-item">
                            <strong>${testName}</strong><br>
                            ${error}
                        </div>
                    `;
                });
                container.innerHTML = html;
            }
            
            function updateAlerts(alerts) {
                const container = document.getElementById('alerts-list');
                if (!alerts || alerts.length === 0) {
                    container.innerHTML = 'No alerts';
                    return;
                }
                
                let html = '';
                alerts.forEach(alert => {
                    const time = new Date(alert.timestamp).toLocaleTimeString();
                    const message = alert.message || alert.text || 'Alert';
                    html += `
                        <div class="alert-item">
                            <strong>${time}</strong><br>
                            ${message}
                        </div>
                    `;
                });
                container.innerHTML = html;
            }
            
            function addAlert(alert) {
                const container = document.getElementById('alerts-list');
                const time = new Date().toLocaleTimeString();
                const message = alert.message || alert.text || 'Alert';
                
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert-item';
                alertDiv.innerHTML = `
                    <strong>${time}</strong><br>
                    ${message}
                `;
                
                container.insertBefore(alertDiv, container.firstChild);
                
                // Keep only last 10 alerts
                const alerts = container.children;
                while (alerts.length > 10) {
                    container.removeChild(alerts[alerts.length - 1]);
                }
            }
            
            function clearAlerts() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'clear_alerts' }));
                }
            }
            
            // Auto-refresh request
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'request_update' }));
                }
            }, 5000);
            
            // Initialize connection
            connectWebSocket();
        """


class APIHandler(tornado.web.RequestHandler):
    """API handler for dashboard data."""
    
    def get(self):
        """Get dashboard data via REST API."""
        dashboard_data = self.application.dashboard_data.get_dashboard_state()
        self.write(dashboard_data)


class TestMasterDashboard:
    """
    Real-time web dashboard for TestMaster.
    
    Provides live monitoring of test execution, coverage,
    idle detection, and system health.
    """
    
    @requires_layer("layer2_monitoring", "dashboard_ui")
    def __init__(self, config: DashboardConfig = None):
        """
        Initialize TestMaster dashboard.
        
        Args:
            config: Dashboard configuration
        """
        if not TORNADO_AVAILABLE:
            raise ImportError("Tornado library required for web dashboard")
        
        self.config = config or DashboardConfig()
        self.dashboard_data = DashboardData()
        
        # Tornado application
        self.app = None
        self.ioloop = None
        
        # Server control
        self._is_running = False
        self._server_thread: Optional[threading.Thread] = None
        
        print(f"📊 TestMaster dashboard initialized")
        print(f"   🌐 Host: {self.config.host}:{self.config.port}")
        print(f"   🎨 Theme: {self.config.theme.value}")
    
    def start(self, open_browser: bool = True):
        """
        Start the dashboard web server.
        
        Args:
            open_browser: Whether to open browser automatically
        """
        if self._is_running:
            print("⚠️ Dashboard is already running")
            return
        
        print("🚀 Starting TestMaster dashboard...")
        
        # Create Tornado application
        self.app = tornado.web.Application([
            (r"/", DashboardHandler),
            (r"/ws", WebSocketHandler),
            (r"/api/data", APIHandler),
        ])
        
        # Store references in application
        self.app.config = self.config
        self.app.dashboard_data = self.dashboard_data
        
        # Start server in thread
        self._is_running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(1)
        
        dashboard_url = f"http://{self.config.host}:{self.config.port}"
        print(f"✅ Dashboard running at {dashboard_url}")
        
        if open_browser:
            try:
                webbrowser.open(dashboard_url)
                print("🌐 Opened dashboard in browser")
            except Exception as e:
                print(f"⚠️ Could not open browser: {e}")
    
    def stop(self):
        """Stop the dashboard web server."""
        if not self._is_running:
            return
        
        print("🛑 Stopping TestMaster dashboard...")
        
        self._is_running = False
        
        if self.ioloop:
            self.ioloop.add_callback(self.ioloop.stop)
        
        if self._server_thread:
            self._server_thread.join(timeout=10)
        
        print("✅ Dashboard stopped")
    
    def _run_server(self):
        """Run the Tornado server."""
        try:
            self.app.listen(self.config.port, self.config.host)
            self.ioloop = tornado.ioloop.IOLoop.current()
            self.ioloop.start()
        except Exception as e:
            print(f"⚠️ Error running dashboard server: {e}")
            self._is_running = False
    
    def update_system_status(self, status: Dict[str, Any]):
        """Update system status display."""
        self.dashboard_data.update_system_status(status)
        self._broadcast_update('system_status_update', status)
    
    def add_test_result(self, result: Dict[str, Any]):
        """Add test result to dashboard."""
        self.dashboard_data.add_test_result(result)
        self._broadcast_update('test_result', result)
    
    def update_coverage_data(self, coverage: Dict[str, Any]):
        """Update coverage data display."""
        self.dashboard_data.update_coverage_data(coverage)
        self._broadcast_update('coverage_update', coverage)
    
    def update_idle_modules(self, modules: List[Dict[str, Any]]):
        """Update idle modules display."""
        self.dashboard_data.update_idle_modules(modules)
        self._broadcast_update('idle_modules_update', modules)
    
    def update_breaking_tests(self, tests: List[Dict[str, Any]]):
        """Update breaking tests display."""
        self.dashboard_data.update_breaking_tests(tests)
        self._broadcast_update('breaking_tests_update', tests)
    
    def add_alert(self, message: str, level: str = "info", metadata: Dict[str, Any] = None):
        """Add alert to dashboard."""
        alert = {
            'message': message,
            'level': level,
            'metadata': metadata or {}
        }
        
        self.dashboard_data.add_alert(alert)
        self._broadcast_update('alert', alert)
    
    def _broadcast_update(self, update_type: str, data: Any):
        """Broadcast update to connected clients."""
        if self._is_running and hasattr(WebSocketHandler, 'broadcast_update'):
            WebSocketHandler.broadcast_update(update_type, data)
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return f"http://{self.config.host}:{self.config.port}"
    
    def get_connected_clients_count(self) -> int:
        """Get number of connected dashboard clients."""
        if hasattr(WebSocketHandler, 'clients'):
            return len(WebSocketHandler.clients)
        return 0


# Convenience function for quick dashboard setup
def start_dashboard(host: str = "localhost", port: int = 8080,
                   theme: DashboardTheme = DashboardTheme.DARK,
                   open_browser: bool = True) -> TestMasterDashboard:
    """
    Start TestMaster dashboard with simple configuration.
    
    Args:
        host: Server host
        port: Server port
        theme: Dashboard theme
        open_browser: Whether to open browser
        
    Returns:
        TestMasterDashboard instance
    """
    config = DashboardConfig(host=host, port=port, theme=theme)
    dashboard = TestMasterDashboard(config)
    dashboard.start(open_browser=open_browser)
    return dashboard