#!/usr/bin/env python3
"""
Standalone Database Monitor
Simple, practical database monitoring without TestMaster dependencies
"""

import asyncio
import logging
import time
import json
import sqlite3
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

@dataclass
class DatabaseMetrics:
    """Simple database metrics"""
    timestamp: datetime
    connection_count: int
    query_count: int
    slow_query_count: int
    database_size_mb: float
    cpu_usage: float
    memory_usage_mb: float
    avg_query_time_ms: float
    errors: int

@dataclass
class DatabaseAlert:
    """Database alert"""
    alert_type: str
    message: str
    severity: str
    timestamp: datetime
    resolved: bool = False

class DatabaseMonitor:
    """Simple database monitor"""
    
    def __init__(self):
        self.logger = logging.getLogger("DatabaseMonitor")
        self.monitoring_active = False
        self.monitoring_thread = None
        self.config_file = Path("db_monitor_config.json")
        
        # Data storage
        self.metrics_history = deque(maxlen=1000)
        self.alerts = []
        
        # Load configuration
        self.config = self._load_config()
        
        self._setup_logging()
    
    def _load_config(self):
        """Load configuration from file"""
        default_config = {
            "databases": {},
            "monitoring_interval": 30,
            "alert_thresholds": {
                'cpu_usage': 80,
                'memory_usage_mb': 1000,
                'connection_count': 50
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load config: {e}")
        
        return default_config
    
    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _setup_logging(self):
        """Setup basic logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('db_monitor.log'),
                logging.StreamHandler()
            ]
        )
    
    def add_database(self, name: str, db_type: str, path: str = None, **kwargs):
        """Add database to monitor"""
        if db_type == "sqlite":
            if not path or not Path(path).exists():
                print(f"[ERROR] SQLite database not found: {path}")
                return False
            
            self.config["databases"][name] = {
                "type": "sqlite",
                "path": path,
                "enabled": True
            }
            self._save_config()
            print(f"[OK] Added SQLite database: {name} ({path})")
            return True
        
        print(f"[ERROR] Database type '{db_type}' not supported in standalone version")
        return False
    
    def start_monitoring(self):
        """Start monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("[STARTED] Database monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("[STOPPED] Database monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                for db_name, db_config in self.config["databases"].items():
                    if db_config.get("enabled", False):
                        metrics = self._collect_metrics(db_name, db_config)
                        if metrics:
                            self.metrics_history.append(metrics)
                            self._check_alerts(metrics)
                
                time.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def _collect_metrics(self, db_name: str, db_config: Dict[str, Any]) -> Optional[DatabaseMetrics]:
        """Collect database metrics"""
        try:
            if db_config["type"] == "sqlite":
                db_path = Path(db_config["path"])
                
                # Database size
                db_size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
                
                # System metrics
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
                
                # SQLite connection test
                connection_count = 1
                query_count = 0
                errors = 0
                
                try:
                    conn = sqlite3.connect(db_config["path"])
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    query_count = len(tables)
                    conn.close()
                except Exception as e:
                    self.logger.error(f"SQLite query failed: {e}")
                    errors = 1
                
                return DatabaseMetrics(
                    timestamp=datetime.now(),
                    connection_count=connection_count,
                    query_count=query_count,
                    slow_query_count=0,
                    database_size_mb=db_size_mb,
                    cpu_usage=cpu_usage,
                    memory_usage_mb=memory_usage_mb,
                    avg_query_time_ms=0,
                    errors=errors
                )
                
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for {db_name}: {e}")
            return None
    
    def _check_alerts(self, metrics: DatabaseMetrics):
        """Check for alerts"""
        thresholds = self.config["alert_thresholds"]
        
        if metrics.cpu_usage > thresholds['cpu_usage']:
            alert = DatabaseAlert(
                alert_type="high_cpu",
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                severity="warning",
                timestamp=datetime.now()
            )
            self.alerts.append(alert)
            print(f"[ALERT] {alert.message}")
        
        if metrics.memory_usage_mb > thresholds['memory_usage_mb']:
            alert = DatabaseAlert(
                alert_type="high_memory",
                message=f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                severity="warning",
                timestamp=datetime.now()
            )
            self.alerts.append(alert)
            print(f"[ALERT] {alert.message}")
    
    def get_status(self):
        """Get current status"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "monitoring_active": self.monitoring_active,
            "databases_configured": len(self.config["databases"]),
            "metrics_collected": len(self.metrics_history),
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "latest_metrics": asdict(latest_metrics) if latest_metrics else None
        }
    
    def get_summary(self, hours: int = 24):
        """Get metrics summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        latest_size = recent_metrics[-1].database_size_mb if recent_metrics else 0
        
        return {
            "time_period_hours": hours,
            "metrics_count": len(recent_metrics),
            "avg_cpu_usage": round(avg_cpu, 1),
            "avg_memory_usage_mb": round(avg_memory, 1),
            "current_db_size_mb": round(latest_size, 1),
            "alerts_in_period": len([a for a in self.alerts if a.timestamp > cutoff_time])
        }
    
    def generate_report(self):
        """Generate simple text report"""
        status = self.get_status()
        summary = self.get_summary(24)
        active_alerts = [a for a in self.alerts if not a.resolved][-5:]
        
        report = f"""
DATABASE MONITORING REPORT
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STATUS:
- Monitoring Active: {'Yes' if status['monitoring_active'] else 'No'}
- Databases Configured: {status['databases_configured']}
- Metrics Collected: {status['metrics_collected']}
- Active Alerts: {len(active_alerts)}

LAST 24 HOURS:
- Average CPU Usage: {summary.get('avg_cpu_usage', 'N/A')}%
- Average Memory Usage: {summary.get('avg_memory_usage_mb', 'N/A')} MB
- Current Database Size: {summary.get('current_db_size_mb', 'N/A')} MB
- Alerts: {summary.get('alerts_in_period', 0)}

RECENT ALERTS:
"""
        
        if active_alerts:
            for alert in active_alerts:
                report += f"- {alert.severity.upper()}: {alert.message} ({alert.timestamp.strftime('%H:%M:%S')})\n"
        else:
            report += "- No recent alerts\n"
        
        return report

class DashboardHandler(BaseHTTPRequestHandler):
    """Simple dashboard handler"""
    
    def __init__(self, *args, monitor=None, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == "/":
            self.serve_dashboard()
        elif self.path == "/api/status":
            self.serve_json(self.monitor.get_status())
        elif self.path == "/api/summary":
            self.serve_json(self.monitor.get_summary())
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Database Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #e8f5e8; border-radius: 8px; text-align: center; min-width: 150px; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 14px; color: #666; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Database Monitor</h1>
        <button class="refresh-btn" onclick="location.reload()">Refresh</button>
    </div>
    
    <div class="card">
        <h2>Current Status</h2>
        <div id="status">Loading...</div>
    </div>
    
    <div class="card">
        <h2>24-Hour Summary</h2>
        <div id="summary">Loading...</div>
    </div>
    
    <script>
        async function loadData() {
            try {
                const [statusResp, summaryResp] = await Promise.all([
                    fetch('/api/status'),
                    fetch('/api/summary')
                ]);
                
                const status = await statusResp.json();
                const summary = await summaryResp.json();
                
                document.getElementById('status').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${status.monitoring_active ? 'Active' : 'Inactive'}</div>
                        <div class="metric-label">Monitoring</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${status.databases_configured}</div>
                        <div class="metric-label">Databases</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${status.metrics_collected}</div>
                        <div class="metric-label">Metrics</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${status.active_alerts}</div>
                        <div class="metric-label">Alerts</div>
                    </div>
                `;
                
                document.getElementById('summary').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${summary.avg_cpu_usage || 'N/A'}%</div>
                        <div class="metric-label">Avg CPU</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${summary.avg_memory_usage_mb || 'N/A'} MB</div>
                        <div class="metric-label">Avg Memory</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${summary.current_db_size_mb || 'N/A'} MB</div>
                        <div class="metric-label">DB Size</div>
                    </div>
                `;
                
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        loadData();
        setInterval(loadData, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

def main():
    """Main CLI function"""
    monitor = DatabaseMonitor()
    
    if len(sys.argv) < 2:
        print("Database Monitor Commands:")
        print("  setup                    - Create configuration")
        print("  add <name> sqlite <path> - Add SQLite database")
        print("  start                    - Start monitoring")
        print("  status                   - Show status")
        print("  report                   - Generate report")
        print("  dashboard                - Start web dashboard")
        return
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        config_file = Path("db_monitor_config.json")
        config = {
            "databases": {},
            "monitoring_interval": 30,
            "alert_thresholds": {
                "cpu_usage": 80,
                "memory_usage_mb": 1000,
                "connection_count": 50
            }
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[OK] Configuration created: {config_file}")
        print("Next: python db_monitor_standalone.py add mydb sqlite /path/to/database.db")
        
    elif command == "add":
        if len(sys.argv) < 5:
            print("Usage: python db_monitor_standalone.py add <name> sqlite <path>")
            return
        name, db_type, path = sys.argv[2], sys.argv[3], sys.argv[4]
        monitor.add_database(name, db_type, path)
        
    elif command == "start":
        print("[STARTING] Database monitoring...")
        print("Press Ctrl+C to stop")
        monitor.start_monitoring()
        try:
            while True:
                time.sleep(5)
                status = monitor.get_status()
                if status['latest_metrics']:
                    m = status['latest_metrics']
                    print(f"[{m['timestamp'][:19]}] CPU: {m['cpu_usage']:.1f}% | Memory: {m['memory_usage_mb']:.1f}MB | DB Size: {m['database_size_mb']:.1f}MB")
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            
    elif command == "status":
        status = monitor.get_status()
        summary = monitor.get_summary()
        print(f"Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")
        print(f"Databases: {status['databases_configured']}")
        print(f"Metrics: {status['metrics_collected']}")
        print(f"Alerts: {status['active_alerts']}")
        if summary.get('avg_cpu_usage'):
            print(f"Avg CPU: {summary['avg_cpu_usage']}%")
            print(f"Avg Memory: {summary['avg_memory_usage_mb']}MB")
            print(f"DB Size: {summary['current_db_size_mb']}MB")
            
    elif command == "report":
        print(monitor.generate_report())
        
    elif command == "dashboard":
        print("[DASHBOARD] Starting at http://localhost:8080")
        monitor.start_monitoring()
        
        def handler(*args, **kwargs):
            return DashboardHandler(*args, monitor=monitor, **kwargs)
        
        server = HTTPServer(('localhost', 8080), handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[STOPPING] Dashboard stopping...")
            monitor.stop_monitoring()
            server.shutdown()
            
    else:
        print(f"[ERROR] Unknown command: {command}")

if __name__ == "__main__":
    main()