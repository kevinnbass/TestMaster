#!/usr/bin/env python3
"""
Simple Database Dashboard
Agent B Hours 90-100: Personal Database Monitoring

Simple web dashboard to view your database metrics.
No enterprise fluff - just the info you need.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Simple web server
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse
    WEB_SERVER_AVAILABLE = True
except ImportError:
    WEB_SERVER_AVAILABLE = False

from .personal_database_monitor import PersonalDatabaseMonitor

class DashboardHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for dashboard"""
    
    def __init__(self, *args, monitor=None, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/":
            self.serve_dashboard()
        elif self.path == "/api/status":
            self.serve_status()
        elif self.path == "/api/metrics":
            self.serve_metrics()
        elif self.path == "/api/alerts":
            self.serve_alerts()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Serve main dashboard HTML"""
        html = self.generate_dashboard_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_status(self):
        """Serve current status as JSON"""
        if self.monitor:
            status = self.monitor.get_current_status()
        else:
            status = {"error": "Monitor not available"}
        
        self.send_json_response(status)
    
    def serve_metrics(self):
        """Serve metrics summary as JSON"""
        if self.monitor:
            metrics = self.monitor.get_metrics_summary(24)
        else:
            metrics = {"error": "Monitor not available"}
        
        self.send_json_response(metrics)
    
    def serve_alerts(self):
        """Serve active alerts as JSON"""
        if self.monitor:
            alerts = self.monitor.get_active_alerts()
        else:
            alerts = []
        
        self.send_json_response(alerts)
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        json_data = json.dumps(data, indent=2, default=str)
        self.wfile.write(json_data.encode())
    
    def generate_dashboard_html(self) -> str:
        """Generate simple HTML dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Personal Database Monitor</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            color: #333;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: #e8f5e8;
            border-radius: 8px;
        }
        .metric.warning {
            background: #fff3cd;
        }
        .metric.critical {
            background: #f8d7da;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .alert {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #dc3545;
            background: #f8d7da;
        }
        .alert.warning {
            border-left-color: #ffc107;
            background: #fff3cd;
        }
        .refresh-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 10px 0;
        }
        .refresh-btn:hover {
            background: #0056b3;
        }
        .timestamp {
            color: #666;
            font-size: 12px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <h1>📊 Personal Database Monitor</h1>
                <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>
                <div id="last-update" class="timestamp"></div>
            </div>
        </div>

        <div class="card">
            <h2>📈 Current Status</h2>
            <div class="status-grid" id="status-grid">
                <div class="metric">
                    <div class="metric-value" id="monitoring-status">-</div>
                    <div class="metric-label">Monitoring</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="db-count">-</div>
                    <div class="metric-label">Databases</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metrics-count">-</div>
                    <div class="metric-label">Metrics Collected</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="alert-count">-</div>
                    <div class="metric-label">Active Alerts</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📊 24-Hour Summary</h2>
            <div class="status-grid" id="metrics-grid">
                <div class="metric">
                    <div class="metric-value" id="avg-cpu">-</div>
                    <div class="metric-label">Avg CPU %</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="avg-memory">-</div>
                    <div class="metric-label">Avg Memory (MB)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="avg-connections">-</div>
                    <div class="metric-label">Avg Connections</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="db-size">-</div>
                    <div class="metric-label">DB Size (MB)</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🚨 Active Alerts</h2>
            <div id="alerts-container">
                <p>Loading alerts...</p>
            </div>
        </div>

        <div class="card">
            <h2>📋 Recent Metrics</h2>
            <div id="recent-metrics">
                <p>Loading recent metrics...</p>
            </div>
        </div>
    </div>

    <script>
        let lastUpdateTime = new Date();

        async function fetchData(endpoint) {
            try {
                const response = await fetch(endpoint);
                return await response.json();
            } catch (error) {
                console.error('Error fetching data:', error);
                return null;
            }
        }

        async function updateStatus() {
            const status = await fetchData('/api/status');
            if (!status) return;

            document.getElementById('monitoring-status').textContent = 
                status.monitoring_active ? 'Active' : 'Inactive';
            document.getElementById('db-count').textContent = 
                status.databases_configured || 0;
            document.getElementById('metrics-count').textContent = 
                status.metrics_collected || 0;
            document.getElementById('alert-count').textContent = 
                status.total_alerts || 0;

            // Color code alert count
            const alertElement = document.getElementById('alert-count').parentElement;
            if (status.total_alerts > 0) {
                alertElement.className = 'metric warning';
            } else {
                alertElement.className = 'metric';
            }
        }

        async function updateMetrics() {
            const metrics = await fetchData('/api/metrics');
            if (!metrics || !metrics.averages) return;

            document.getElementById('avg-cpu').textContent = 
                metrics.averages.cpu_usage + '%';
            document.getElementById('avg-memory').textContent = 
                metrics.averages.memory_usage_mb;
            document.getElementById('avg-connections').textContent = 
                metrics.averages.connection_count;
            document.getElementById('db-size').textContent = 
                metrics.current?.database_size_mb || 0;

            // Color code CPU usage
            const cpuElement = document.getElementById('avg-cpu').parentElement;
            if (metrics.averages.cpu_usage > 80) {
                cpuElement.className = 'metric critical';
            } else if (metrics.averages.cpu_usage > 60) {
                cpuElement.className = 'metric warning';
            } else {
                cpuElement.className = 'metric';
            }
        }

        async function updateAlerts() {
            const alerts = await fetchData('/api/alerts');
            const container = document.getElementById('alerts-container');
            
            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<p>✅ No active alerts</p>';
                return;
            }

            let html = '';
            alerts.forEach(alert => {
                const alertClass = alert.severity === 'critical' ? 'alert' : 'alert warning';
                const timestamp = new Date(alert.timestamp).toLocaleString();
                html += `
                    <div class="${alertClass}">
                        <strong>${alert.severity.toUpperCase()}</strong>: ${alert.message}
                        <div class="timestamp">🕒 ${timestamp}</div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        async function refreshDashboard() {
            lastUpdateTime = new Date();
            document.getElementById('last-update').textContent = 
                'Last updated: ' + lastUpdateTime.toLocaleTimeString();

            await Promise.all([
                updateStatus(),
                updateMetrics(),
                updateAlerts()
            ]);
        }

        // Auto-refresh every 30 seconds
        setInterval(refreshDashboard, 30000);

        // Initial load
        refreshDashboard();
    </script>
</body>
</html>
        """

class SimpleDashboard:
    """Simple web dashboard for database monitoring"""
    
    def __init__(self, monitor: PersonalDatabaseMonitor, port: int = 8080):
        self.monitor = monitor
        self.port = port
        self.server = None
        self.server_thread = None
        
        if not WEB_SERVER_AVAILABLE:
            print("Warning: Web server not available. Install required packages.")
    
    def start(self):
        """Start the dashboard web server"""
        if not WEB_SERVER_AVAILABLE:
            print("Cannot start web server - missing dependencies")
            return False
        
        try:
            # Create handler with monitor
            def handler(*args, **kwargs):
                return DashboardHandler(*args, monitor=self.monitor, **kwargs)
            
            self.server = HTTPServer(('localhost', self.port), handler)
            
            print(f"🌐 Dashboard starting at http://localhost:{self.port}")
            print("📊 Open your browser to view database metrics")
            print("Press Ctrl+C to stop")
            
            # Run server
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"Dashboard error: {e}")
            return False
    
    def stop(self):
        """Stop the dashboard web server"""
        if self.server:
            print("\n🛑 Stopping dashboard...")
            self.server.shutdown()
            self.server = None

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "databases": {
            "sqlite_example": {
                "type": "sqlite",
                "path": "example.db",
                "enabled": True,
                "description": "Example SQLite database"
            }
        },
        "monitoring_interval": 30,
        "enable_alerts": True,
        "log_slow_queries": True,
        "dashboard_port": 8080,
        "alert_thresholds": {
            "slow_query_ms": 1000,
            "connection_count": 50,
            "cpu_usage": 80,
            "memory_usage_mb": 1000,
            "error_rate": 5
        }
    }
    
    config_file = Path("database_monitor_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Sample config created: {config_file}")
    print("Edit this file to add your databases:")
    print("- SQLite: Just provide the path")
    print("- MySQL: Add host, user, password, database")
    print("- PostgreSQL: Add host, user, password, database")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-config":
        create_sample_config()
        sys.exit(0)
    
    # Create monitor and dashboard
    monitor = PersonalDatabaseMonitor()
    dashboard = SimpleDashboard(monitor, port=8080)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Start dashboard (this will block)
        dashboard.start()
    finally:
        # Stop monitoring when dashboard stops
        monitor.stop_monitoring()