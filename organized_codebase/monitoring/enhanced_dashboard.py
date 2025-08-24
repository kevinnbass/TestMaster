#!/usr/bin/env python3
"""
Enhanced Database Dashboard
Agent B Hours 100-110: Advanced User Experience & Enhancement

Enhanced dashboard with improved visualization and user experience.
"""

import json
import sqlite3
import time
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socket

class EnhancedDashboardServer:
    """Enhanced database monitoring dashboard with improved UX"""
    
    def __init__(self, config_file: str = "db_monitor_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.metrics_history = []
        self.alerts = []
        self.running = False
        
        # Enhanced features
        self.performance_baselines = {}
        self.user_preferences = self.load_user_preferences()
        
    def load_config(self):
        """Load database configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"databases": {}}
    
    def load_user_preferences(self):
        """Load user dashboard preferences"""
        prefs_file = Path("dashboard_preferences.json")
        default_prefs = {
            "refresh_interval": 30,
            "alert_threshold_cpu": 80,
            "alert_threshold_memory_gb": 28,
            "dashboard_theme": "light",
            "show_detailed_metrics": True,
            "auto_backup_alerts": True,
            "growth_prediction_days": 30
        }
        
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r') as f:
                    loaded_prefs = json.load(f)
                    default_prefs.update(loaded_prefs)
            except:
                pass
        
        return default_prefs
    
    def save_user_preferences(self):
        """Save user dashboard preferences"""
        try:
            with open("dashboard_preferences.json", 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save preferences: {e}")
    
    def collect_enhanced_metrics(self):
        """Collect enhanced system metrics"""
        timestamp = datetime.now()
        
        # Get system metrics
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            system_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100
            }
        except ImportError:
            # Fallback without psutil
            system_metrics = {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_used_gb": 0,
                "memory_total_gb": 0,
                "disk_used_gb": 0,
                "disk_free_gb": 0,
                "disk_percent": 0
            }
        
        # Database metrics
        db_metrics = {}
        total_db_size = 0
        total_queries = 0
        
        for db_name, db_config in self.config["databases"].items():
            if db_config.get("enabled", False):
                db_path = Path(db_config["path"])
                if db_path.exists():
                    size_mb = db_path.stat().st_size / (1024 * 1024)
                    total_db_size += size_mb
                    
                    # Simulate query analysis
                    query_count = self.simulate_query_metrics(db_path)
                    total_queries += query_count
                    
                    db_metrics[db_name] = {
                        "size_mb": size_mb,
                        "query_count": query_count,
                        "status": "healthy"
                    }
        
        # Combined metrics
        metrics = {
            "timestamp": timestamp.isoformat(),
            "system": system_metrics,
            "databases": db_metrics,
            "totals": {
                "database_size_mb": total_db_size,
                "query_count": total_queries,
                "database_count": len([db for db in self.config["databases"].values() if db.get("enabled", False)])
            }
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # Check for alerts
        self.check_alerts(metrics)
        
        return metrics
    
    def simulate_query_metrics(self, db_path):
        """Simulate query count for demo purposes"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            return table_count * 2  # Simulate queries
        except:
            return 0
    
    def check_alerts(self, metrics):
        """Check for alert conditions"""
        alerts = []
        
        # CPU alerts
        if metrics["system"]["cpu_percent"] > self.user_preferences["alert_threshold_cpu"]:
            alerts.append({
                "type": "cpu",
                "severity": "warning",
                "message": f"High CPU usage: {metrics['system']['cpu_percent']:.1f}%",
                "timestamp": metrics["timestamp"]
            })
        
        # Memory alerts  
        if metrics["system"]["memory_used_gb"] > self.user_preferences["alert_threshold_memory_gb"]:
            alerts.append({
                "type": "memory", 
                "severity": "warning",
                "message": f"High memory usage: {metrics['system']['memory_used_gb']:.1f}GB",
                "timestamp": metrics["timestamp"]
            })
        
        # Database size alerts
        for db_name, db_metrics in metrics["databases"].items():
            if db_metrics["size_mb"] > 100:  # Alert for DBs over 100MB
                alerts.append({
                    "type": "database_size",
                    "severity": "info",
                    "message": f"Database {db_name} is {db_metrics['size_mb']:.1f}MB",
                    "timestamp": metrics["timestamp"]
                })
        
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_dashboard_data(self):
        """Get comprehensive dashboard data"""
        if not self.metrics_history:
            return {"message": "No metrics available yet"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate trends
        trends = self.calculate_trends()
        
        # Get recent alerts
        recent_alerts = [alert for alert in self.alerts 
                        if datetime.fromisoformat(alert["timestamp"]) > datetime.now() - timedelta(hours=1)]
        
        return {
            "current_metrics": latest_metrics,
            "trends": trends,
            "alerts": {
                "recent": recent_alerts,
                "total": len(self.alerts)
            },
            "system_health": self.calculate_system_health(),
            "recommendations": self.generate_recommendations(),
            "user_preferences": self.user_preferences
        }
    
    def calculate_trends(self):
        """Calculate performance trends"""
        if len(self.metrics_history) < 2:
            return {}
        
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2] if len(self.metrics_history) >= 2 else current
        
        # Calculate changes
        cpu_trend = current["system"]["cpu_percent"] - previous["system"]["cpu_percent"]
        memory_trend = current["system"]["memory_used_gb"] - previous["system"]["memory_used_gb"]
        
        return {
            "cpu_change": cpu_trend,
            "memory_change_gb": memory_trend,
            "database_growth_mb": current["totals"]["database_size_mb"] - previous["totals"]["database_size_mb"]
        }
    
    def calculate_system_health(self):
        """Calculate overall system health score"""
        if not self.metrics_history:
            return 50
        
        latest = self.metrics_history[-1]
        score = 100
        
        # Deduct for high resource usage
        if latest["system"]["cpu_percent"] > 80:
            score -= 20
        elif latest["system"]["cpu_percent"] > 60:
            score -= 10
        
        if latest["system"]["memory_percent"] > 90:
            score -= 20
        elif latest["system"]["memory_percent"] > 80:
            score -= 10
        
        # Check recent alerts
        recent_alerts = [a for a in self.alerts 
                        if datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(hours=1)]
        score -= len(recent_alerts) * 5
        
        return max(0, min(100, score))
    
    def generate_recommendations(self):
        """Generate system recommendations"""
        if not self.metrics_history:
            return []
        
        recommendations = []
        latest = self.metrics_history[-1]
        
        if latest["system"]["cpu_percent"] > 80:
            recommendations.append("Consider investigating high CPU usage processes")
        
        if latest["system"]["memory_percent"] > 90:
            recommendations.append("Memory usage is high - consider restarting applications")
        
        if latest["totals"]["database_size_mb"] > 500:
            recommendations.append("Large database detected - consider archiving old data")
        
        if len(self.alerts) > 50:
            recommendations.append("Many alerts generated - review system configuration")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def generate_enhanced_html(self):
        """Generate enhanced HTML dashboard"""
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Database Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 1.8rem;
        }}
        
        .subtitle {{
            opacity: 0.9;
            margin-top: 0.5rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1rem;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        
        .card h3 {{
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            font-weight: 500;
        }}
        
        .metric-value {{
            font-weight: 700;
            color: #667eea;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-good {{ background: #4caf50; }}
        .status-warning {{ background: #ff9800; }}
        .status-critical {{ background: #f44336; }}
        
        .alerts-section {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }}
        
        .alert {{
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            border-left: 4px solid;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border-color: #ff9800;
            color: #856404;
        }}
        
        .alert-info {{
            background: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }}
        
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            height: 8px;
            margin-top: 0.5rem;
        }}
        
        .progress-fill {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            transition: width 0.3s ease;
        }}
        
        .settings-panel {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
        }}
        
        .settings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        
        .setting {{
            display: flex;
            flex-direction: column;
        }}
        
        .setting label {{
            margin-bottom: 0.5rem;
            font-weight: 500;
        }}
        
        .setting input, .setting select {{
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        
        .btn {{
            background: #667eea;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            margin-top: 1rem;
        }}
        
        .btn:hover {{
            background: #5a67d8;
        }}
        
        .refresh-indicator {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #667eea;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            display: none;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 0 0.5rem;
            }}
            
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Enhanced Database Monitor</h1>
        <div class="subtitle">
            Agent B Hours 100-110: Advanced User Experience Dashboard
            <span id="lastUpdate"></span>
        </div>
    </div>
    
    <div class="refresh-indicator" id="refreshIndicator">Updating...</div>
    
    <div class="container">
        <div class="dashboard-grid">
            <div class="card">
                <h3>System Overview</h3>
                <div id="systemMetrics"></div>
            </div>
            
            <div class="card">
                <h3>Database Status</h3>
                <div id="databaseMetrics"></div>
            </div>
            
            <div class="card">
                <h3>Performance Trends</h3>
                <div id="performanceTrends"></div>
            </div>
            
            <div class="card">
                <h3>System Health Score</h3>
                <div id="healthScore"></div>
            </div>
        </div>
        
        <div class="alerts-section">
            <h3>Active Alerts</h3>
            <div id="alertsContainer"></div>
        </div>
        
        <div class="alerts-section">
            <h3>Recommendations</h3>
            <div id="recommendationsContainer"></div>
        </div>
        
        <div class="settings-panel">
            <h3>Dashboard Settings</h3>
            <div class="settings-grid">
                <div class="setting">
                    <label for="refreshInterval">Refresh Interval (seconds)</label>
                    <input type="number" id="refreshInterval" min="10" max="300" value="30">
                </div>
                <div class="setting">
                    <label for="cpuThreshold">CPU Alert Threshold (%)</label>
                    <input type="number" id="cpuThreshold" min="50" max="100" value="80">
                </div>
                <div class="setting">
                    <label for="memoryThreshold">Memory Alert Threshold (GB)</label>
                    <input type="number" id="memoryThreshold" min="1" max="64" value="28">
                </div>
                <div class="setting">
                    <label for="theme">Dashboard Theme</label>
                    <select id="theme">
                        <option value="light">Light</option>
                        <option value="dark">Dark</option>
                    </select>
                </div>
            </div>
            <button class="btn" onclick="saveSettings()">Save Settings</button>
        </div>
    </div>
    
    <script>
        let refreshInterval = 30000; // Default 30 seconds
        let refreshTimer;
        
        function updateDashboard() {{
            const indicator = document.getElementById('refreshIndicator');
            indicator.style.display = 'block';
            
            fetch('/api/enhanced')
                .then(response => response.json())
                .then(data => {{
                    updateSystemMetrics(data.current_metrics);
                    updateDatabaseMetrics(data.current_metrics);
                    updateTrends(data.trends);
                    updateHealthScore(data.system_health);
                    updateAlerts(data.alerts);
                    updateRecommendations(data.recommendations);
                    updateLastUpdate();
                    
                    // Update settings
                    const prefs = data.user_preferences;
                    if (prefs) {{
                        document.getElementById('refreshInterval').value = prefs.refresh_interval;
                        document.getElementById('cpuThreshold').value = prefs.alert_threshold_cpu;
                        document.getElementById('memoryThreshold').value = prefs.alert_threshold_memory_gb;
                        document.getElementById('theme').value = prefs.dashboard_theme;
                    }}
                }})
                .catch(error => {{
                    console.error('Error updating dashboard:', error);
                }})
                .finally(() => {{
                    indicator.style.display = 'none';
                }});
        }}
        
        function updateSystemMetrics(metrics) {{
            const container = document.getElementById('systemMetrics');
            if (!metrics || !metrics.system) return;
            
            const system = metrics.system;
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value">${{system.cpu_percent.toFixed(1)}}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${{system.cpu_percent}}%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value">${{system.memory_used_gb.toFixed(1)}} GB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${{system.memory_percent}}%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Disk Usage</span>
                    <span class="metric-value">${{system.disk_percent.toFixed(1)}}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${{system.disk_percent}}%"></div>
                </div>
            `;
        }}
        
        function updateDatabaseMetrics(metrics) {{
            const container = document.getElementById('databaseMetrics');
            if (!metrics || !metrics.databases) return;
            
            let html = '';
            for (const [name, db] of Object.entries(metrics.databases)) {{
                const statusClass = db.status === 'healthy' ? 'status-good' : 'status-warning';
                html += `
                    <div class="metric">
                        <span class="metric-label">
                            <span class="status-indicator ${{statusClass}}"></span>
                            ${{name}}
                        </span>
                        <span class="metric-value">${{db.size_mb.toFixed(2)}} MB</span>
                    </div>
                `;
            }}
            
            container.innerHTML = html;
        }}
        
        function updateTrends(trends) {{
            const container = document.getElementById('performanceTrends');
            if (!trends) return;
            
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">CPU Change</span>
                    <span class="metric-value">${{trends.cpu_change > 0 ? '+' : ''}}${{trends.cpu_change.toFixed(1)}}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Change</span>
                    <span class="metric-value">${{trends.memory_change_gb > 0 ? '+' : ''}}${{trends.memory_change_gb.toFixed(2)}} GB</span>
                </div>
                <div class="metric">
                    <span class="metric-label">DB Growth</span>
                    <span class="metric-value">${{trends.database_growth_mb > 0 ? '+' : ''}}${{trends.database_growth_mb.toFixed(2)}} MB</span>
                </div>
            `;
        }}
        
        function updateHealthScore(score) {{
            const container = document.getElementById('healthScore');
            const color = score > 80 ? '#4caf50' : score > 60 ? '#ff9800' : '#f44336';
            
            container.innerHTML = `
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: bold; color: ${{color}};">${{score}}</div>
                    <div style="margin-top: 0.5rem;">out of 100</div>
                    <div class="progress-bar" style="margin-top: 1rem;">
                        <div class="progress-fill" style="width: ${{score}}%; background: ${{color}};"></div>
                    </div>
                </div>
            `;
        }}
        
        function updateAlerts(alerts) {{
            const container = document.getElementById('alertsContainer');
            if (!alerts || !alerts.recent) {{
                container.innerHTML = '<p>No recent alerts</p>';
                return;
            }}
            
            if (alerts.recent.length === 0) {{
                container.innerHTML = '<p>No recent alerts</p>';
                return;
            }}
            
            let html = '';
            alerts.recent.forEach(alert => {{
                const alertClass = alert.severity === 'warning' ? 'alert-warning' : 'alert-info';
                html += `
                    <div class="alert ${{alertClass}}">
                        <strong>${{alert.type.toUpperCase()}}</strong>: ${{alert.message}}
                        <br><small>Time: ${{new Date(alert.timestamp).toLocaleString()}}</small>
                    </div>
                `;
            }});
            
            container.innerHTML = html;
        }}
        
        function updateRecommendations(recommendations) {{
            const container = document.getElementById('recommendationsContainer');
            if (!recommendations || recommendations.length === 0) {{
                container.innerHTML = '<p>No recommendations at this time</p>';
                return;
            }}
            
            let html = '<ul>';
            recommendations.forEach(rec => {{
                html += `<li>${{rec}}</li>`;
            }});
            html += '</ul>';
            
            container.innerHTML = html;
        }}
        
        function updateLastUpdate() {{
            document.getElementById('lastUpdate').textContent = ' - Last updated: ' + new Date().toLocaleTimeString();
        }}
        
        function saveSettings() {{
            const settings = {{
                refresh_interval: parseInt(document.getElementById('refreshInterval').value),
                alert_threshold_cpu: parseInt(document.getElementById('cpuThreshold').value),
                alert_threshold_memory_gb: parseInt(document.getElementById('memoryThreshold').value),
                dashboard_theme: document.getElementById('theme').value
            }};
            
            fetch('/api/settings', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify(settings)
            }})
            .then(() => {{
                alert('Settings saved successfully!');
                
                // Update refresh interval
                refreshInterval = settings.refresh_interval * 1000;
                clearInterval(refreshTimer);
                refreshTimer = setInterval(updateDashboard, refreshInterval);
            }})
            .catch(error => {{
                console.error('Error saving settings:', error);
                alert('Failed to save settings');
            }});
        }}
        
        // Initialize dashboard
        updateDashboard();
        refreshTimer = setInterval(updateDashboard, refreshInterval);
        
        // Handle tab visibility for performance
        document.addEventListener('visibilitychange', function() {{
            if (document.hidden) {{
                clearInterval(refreshTimer);
            }} else {{
                updateDashboard();
                refreshTimer = setInterval(updateDashboard, refreshInterval);
            }}
        }});
    </script>
</body>
</html>'''

class EnhancedRequestHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP request handler"""
    
    def __init__(self, dashboard_server, *args, **kwargs):
        self.dashboard_server = dashboard_server
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        try:
            if path == '/':
                # Main dashboard
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                html_content = self.dashboard_server.generate_enhanced_html()
                self.wfile.write(html_content.encode('utf-8'))
                
            elif path == '/api/enhanced':
                # Enhanced dashboard data
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                data = self.dashboard_server.get_dashboard_data()
                self.wfile.write(json.dumps(data, default=str).encode('utf-8'))
                
            elif path == '/api/metrics':
                # Current metrics only
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                metrics = self.dashboard_server.collect_enhanced_metrics()
                self.wfile.write(json.dumps(metrics, default=str).encode('utf-8'))
                
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Not Found')
                
        except Exception as e:
            print(f"[ERROR] Request handler error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'Server Error: {e}'.encode('utf-8'))
    
    def do_POST(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        try:
            if path == '/api/settings':
                # Save settings
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                settings = json.loads(post_data.decode('utf-8'))
                
                # Update user preferences
                self.dashboard_server.user_preferences.update(settings)
                self.dashboard_server.save_user_preferences()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "success"}')
                
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Not Found')
                
        except Exception as e:
            print(f"[ERROR] POST handler error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'Server Error: {e}'.encode('utf-8'))
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def create_enhanced_handler(dashboard_server):
    """Create request handler with dashboard server reference"""
    def handler(*args, **kwargs):
        return EnhancedRequestHandler(dashboard_server, *args, **kwargs)
    return handler

def main():
    """Main function to run enhanced dashboard"""
    dashboard = EnhancedDashboardServer()
    
    # Start metrics collection
    def metrics_loop():
        while dashboard.running:
            try:
                dashboard.collect_enhanced_metrics()
                time.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                print(f"[ERROR] Metrics collection error: {e}")
                time.sleep(60)  # Wait longer on error
    
    dashboard.running = True
    metrics_thread = threading.Thread(target=metrics_loop, daemon=True)
    metrics_thread.start()
    
    # Start web server
    port = 8080
    handler = create_enhanced_handler(dashboard)
    
    try:
        server = HTTPServer(('localhost', port), handler)
        print(f"[OK] Enhanced Dashboard running at http://localhost:{port}")
        print("[OK] Features: Real-time metrics, alerts, trends, settings")
        print("[OK] Press Ctrl+C to stop")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[OK] Shutting down enhanced dashboard...")
        dashboard.running = False
        server.shutdown()
    except Exception as e:
        print(f"[ERROR] Server error: {e}")

if __name__ == "__main__":
    main()