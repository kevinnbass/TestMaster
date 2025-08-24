#!/usr/bin/env python3
"""
Personal Database Monitor
Agent B Hours 90-100: Practical Database Monitoring

Simple, focused database monitoring system for personal use.
Tracks performance, queries, connections, and storage without enterprise overhead.
"""

import asyncio
import logging
import time
import json
import sqlite3
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import deque, defaultdict

# Database connectors (install as needed)
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

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
class SlowQuery:
    """Slow query information"""
    query: str
    duration_ms: float
    timestamp: datetime
    database: str

@dataclass
class DatabaseAlert:
    """Database alert"""
    alert_type: str
    message: str
    severity: str
    timestamp: datetime
    resolved: bool = False

class PersonalDatabaseMonitor:
    """
    Personal Database Monitor
    
    Simple, practical database monitoring for personal use.
    Focuses on essential metrics without enterprise complexity.
    """
    
    def __init__(self, config_file: str = "database_monitor_config.json"):
        self.logger = logging.getLogger("PersonalDatabaseMonitor")
        
        # Configuration
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.slow_queries: deque = deque(maxlen=100)      # Keep last 100 slow queries
        self.alerts: List[DatabaseAlert] = []
        
        # Database connections
        self.db_connections: Dict[str, Any] = {}
        
        # Thresholds for alerts
        self.alert_thresholds = {
            'slow_query_ms': 1000,      # Alert if query takes > 1 second
            'connection_count': 50,      # Alert if > 50 connections
            'cpu_usage': 80,            # Alert if CPU > 80%
            'memory_usage_mb': 1000,    # Alert if memory > 1GB
            'error_rate': 5             # Alert if > 5 errors per minute
        }
        
        self._setup_logging()
        self.logger.info("Personal database monitor initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "databases": {
                "sqlite_example": {
                    "type": "sqlite",
                    "path": "example.db",
                    "enabled": True
                }
            },
            "monitoring_interval": 30,
            "enable_alerts": True,
            "log_slow_queries": True,
            "dashboard_port": 8080
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
        
        # Save default config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging"""
        log_file = Path("database_monitor.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def add_database(self, name: str, db_type: str, **connection_params):
        """Add a database to monitor"""
        db_config = {
            "type": db_type,
            "enabled": True,
            **connection_params
        }
        
        self.config["databases"][name] = db_config
        self._save_config()
        
        # Test connection
        if self._test_connection(name, db_config):
            self.logger.info(f"Added database: {name}")
            return True
        else:
            self.logger.error(f"Failed to connect to database: {name}")
            return False
    
    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _test_connection(self, name: str, db_config: Dict[str, Any]) -> bool:
        """Test database connection"""
        try:
            if db_config["type"] == "sqlite":
                conn = sqlite3.connect(db_config["path"])
                conn.close()
                return True
            
            elif db_config["type"] == "mysql" and MYSQL_AVAILABLE:
                conn = mysql.connector.connect(
                    host=db_config.get("host", "localhost"),
                    user=db_config["user"],
                    password=db_config["password"],
                    database=db_config.get("database", "")
                )
                conn.close()
                return True
            
            elif db_config["type"] == "postgresql" and POSTGRESQL_AVAILABLE:
                conn = psycopg2.connect(
                    host=db_config.get("host", "localhost"),
                    user=db_config["user"],
                    password=db_config["password"],
                    database=db_config.get("database", "postgres")
                )
                conn.close()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Connection test failed for {name}: {e}")
            return False
    
    def start_monitoring(self):
        """Start database monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Database monitoring started")
    
    def stop_monitoring(self):
        """Stop database monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Database monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics from all configured databases
                for db_name, db_config in self.config["databases"].items():
                    if db_config.get("enabled", False):
                        metrics = self._collect_database_metrics(db_name, db_config)
                        if metrics:
                            self.metrics_history.append(metrics)
                            self._check_alerts(metrics)
                
                # Sleep for monitoring interval
                time.sleep(self.config.get("monitoring_interval", 30))
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_database_metrics(self, db_name: str, db_config: Dict[str, Any]) -> Optional[DatabaseMetrics]:
        """Collect metrics from a database"""
        try:
            if db_config["type"] == "sqlite":
                return self._collect_sqlite_metrics(db_name, db_config)
            elif db_config["type"] == "mysql":
                return self._collect_mysql_metrics(db_name, db_config)
            elif db_config["type"] == "postgresql":
                return self._collect_postgresql_metrics(db_name, db_config)
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for {db_name}: {e}")
            return None
    
    def _collect_sqlite_metrics(self, db_name: str, db_config: Dict[str, Any]) -> DatabaseMetrics:
        """Collect SQLite metrics"""
        db_path = Path(db_config["path"])
        
        # Database size
        db_size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
        
        # SQLite specific metrics (simplified)
        connection_count = 1  # SQLite is typically single connection
        query_count = 0       # Would need query logging to track
        slow_query_count = 0
        avg_query_time_ms = 0
        errors = 0
        
        # Try to get some basic info from SQLite
        try:
            conn = sqlite3.connect(db_config["path"])
            cursor = conn.cursor()
            
            # Get table count as a proxy for activity
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_count = len(cursor.fetchall())
            query_count = table_count  # Rough approximation
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"SQLite query failed: {e}")
            errors = 1
        
        return DatabaseMetrics(
            timestamp=datetime.now(),
            connection_count=connection_count,
            query_count=query_count,
            slow_query_count=slow_query_count,
            database_size_mb=db_size_mb,
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            avg_query_time_ms=avg_query_time_ms,
            errors=errors
        )
    
    def _collect_mysql_metrics(self, db_name: str, db_config: Dict[str, Any]) -> Optional[DatabaseMetrics]:
        """Collect MySQL metrics"""
        if not MYSQL_AVAILABLE:
            return None
        
        try:
            conn = mysql.connector.connect(
                host=db_config.get("host", "localhost"),
                user=db_config["user"],
                password=db_config["password"],
                database=db_config.get("database", "")
            )
            cursor = conn.cursor()
            
            # Connection count
            cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
            connection_count = int(cursor.fetchone()[1])
            
            # Query count
            cursor.execute("SHOW STATUS LIKE 'Questions'")
            query_count = int(cursor.fetchone()[1])
            
            # Database size
            cursor.execute(f"""
                SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 1) AS 'DB Size in MB' 
                FROM information_schema.tables 
                WHERE table_schema='{db_config.get("database", "")}'
            """)
            result = cursor.fetchone()
            db_size_mb = float(result[0]) if result[0] else 0
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            
            conn.close()
            
            return DatabaseMetrics(
                timestamp=datetime.now(),
                connection_count=connection_count,
                query_count=query_count,
                slow_query_count=0,  # Would need slow query log analysis
                database_size_mb=db_size_mb,
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                avg_query_time_ms=0,  # Would need performance schema
                errors=0
            )
            
        except Exception as e:
            self.logger.error(f"MySQL metrics collection failed: {e}")
            return None
    
    def _collect_postgresql_metrics(self, db_name: str, db_config: Dict[str, Any]) -> Optional[DatabaseMetrics]:
        """Collect PostgreSQL metrics"""
        if not POSTGRESQL_AVAILABLE:
            return None
        
        try:
            conn = psycopg2.connect(
                host=db_config.get("host", "localhost"),
                user=db_config["user"],
                password=db_config["password"],
                database=db_config.get("database", "postgres")
            )
            cursor = conn.cursor()
            
            # Connection count
            cursor.execute("SELECT count(*) FROM pg_stat_activity")
            connection_count = cursor.fetchone()[0]
            
            # Database size
            cursor.execute(f"SELECT pg_size_pretty(pg_database_size('{db_config.get('database', 'postgres')}'))")
            size_str = cursor.fetchone()[0]
            # Parse size string (e.g., "8023 MB" -> 8023)
            db_size_mb = float(size_str.split()[0]) if "MB" in size_str else 0
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            
            conn.close()
            
            return DatabaseMetrics(
                timestamp=datetime.now(),
                connection_count=connection_count,
                query_count=0,  # Would need pg_stat_statements
                slow_query_count=0,
                database_size_mb=db_size_mb,
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                avg_query_time_ms=0,
                errors=0
            )
            
        except Exception as e:
            self.logger.error(f"PostgreSQL metrics collection failed: {e}")
            return None
    
    def _check_alerts(self, metrics: DatabaseMetrics):
        """Check if metrics exceed alert thresholds"""
        alerts = []
        
        # Slow query alert
        if metrics.slow_query_count > 0:
            alerts.append(DatabaseAlert(
                alert_type="slow_queries",
                message=f"Found {metrics.slow_query_count} slow queries",
                severity="warning",
                timestamp=datetime.now()
            ))
        
        # Connection count alert
        if metrics.connection_count > self.alert_thresholds['connection_count']:
            alerts.append(DatabaseAlert(
                alert_type="high_connections",
                message=f"High connection count: {metrics.connection_count}",
                severity="warning",
                timestamp=datetime.now()
            ))
        
        # CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(DatabaseAlert(
                alert_type="high_cpu",
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                severity="critical",
                timestamp=datetime.now()
            ))
        
        # Memory usage alert
        if metrics.memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
            alerts.append(DatabaseAlert(
                alert_type="high_memory",
                message=f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                severity="warning",
                timestamp=datetime.now()
            ))
        
        # Add alerts and log them
        for alert in alerts:
            self.alerts.append(alert)
            self.logger.warning(f"ALERT: {alert.message}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current database status"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "monitoring_active": self.monitoring_active,
            "databases_configured": len(self.config["databases"]),
            "latest_metrics": asdict(latest_metrics) if latest_metrics else None,
            "total_alerts": len([a for a in self.alerts if not a.resolved]),
            "metrics_collected": len(self.metrics_history),
            "slow_queries_logged": len(self.slow_queries)
        }
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_connections = sum(m.connection_count for m in recent_metrics) / len(recent_metrics)
        
        # Get current database size
        latest_size = recent_metrics[-1].database_size_mb if recent_metrics else 0
        
        return {
            "time_period_hours": hours,
            "metrics_count": len(recent_metrics),
            "averages": {
                "cpu_usage": round(avg_cpu, 1),
                "memory_usage_mb": round(avg_memory, 1),
                "connection_count": round(avg_connections, 1)
            },
            "current": {
                "database_size_mb": round(latest_size, 1),
                "timestamp": recent_metrics[-1].timestamp.isoformat() if recent_metrics else None
            },
            "alerts_in_period": len([a for a in self.alerts if a.timestamp > cutoff_time])
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        active_alerts = [a for a in self.alerts if not a.resolved]
        return [asdict(alert) for alert in active_alerts[-10:]]  # Last 10 alerts
    
    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True
            self.logger.info(f"Alert resolved: {self.alerts[alert_index].message}")
    
    def generate_simple_report(self) -> str:
        """Generate a simple text report"""
        status = self.get_current_status()
        summary = self.get_metrics_summary(24)
        alerts = self.get_active_alerts()
        
        report = f"""
DATABASE MONITORING REPORT
==========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STATUS:
- Monitoring Active: {'Yes' if status['monitoring_active'] else 'No'}
- Databases Configured: {status['databases_configured']}
- Metrics Collected: {status['metrics_collected']}
- Active Alerts: {len(alerts)}

LAST 24 HOURS SUMMARY:
- Average CPU Usage: {summary.get('averages', {}).get('cpu_usage', 'N/A')}%
- Average Memory Usage: {summary.get('averages', {}).get('memory_usage_mb', 'N/A')} MB
- Average Connections: {summary.get('averages', {}).get('connection_count', 'N/A')}
- Current Database Size: {summary.get('current', {}).get('database_size_mb', 'N/A')} MB

ACTIVE ALERTS:
"""
        
        if alerts:
            for i, alert in enumerate(alerts):
                report += f"- {alert['severity'].upper()}: {alert['message']} ({alert['timestamp'][:19]})\n"
        else:
            report += "- No active alerts\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = PersonalDatabaseMonitor()
    
    # Example: Add SQLite database
    monitor.add_database("my_app", "sqlite", path="my_app.db")
    
    # Example: Add MySQL database (uncomment if you have MySQL)
    # monitor.add_database("mysql_db", "mysql", 
    #                     host="localhost", 
    #                     user="username", 
    #                     password="password", 
    #                     database="my_database")
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Let it run for a bit
        time.sleep(60)
        
        # Print report
        print(monitor.generate_simple_report())
        
    except KeyboardInterrupt:
        print("Stopping monitor...")
    finally:
        monitor.stop_monitoring()