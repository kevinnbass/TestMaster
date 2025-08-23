#!/usr/bin/env python3
"""
AGENT BETA - PERFORMANCE MONITORING INFRASTRUCTURE
Phase 1, Hours 5-10: Performance Monitoring System
================================================

Real-time performance monitoring system with Prometheus integration,
custom metrics collection, alerting with configurable thresholds, 
and performance dashboard with historical trends.

Created: 2025-08-23 02:20:00 UTC
Agent: Beta (Performance Optimization Specialist)
Phase: 1 (Hours 5-10)
"""

import os
import sys
import time
import json
import threading
import logging
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
from contextlib import contextmanager
import asyncio
import aiohttp
import platform

# HTTP server for metrics endpoint
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Performance monitoring configuration
@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring system"""
    metrics_port: int = 9090
    metrics_endpoint: str = "/metrics"
    alert_thresholds: Dict[str, float] = None
    data_retention_days: int = 30
    collection_interval: float = 1.0
    alert_channels: List[str] = None
    dashboard_port: int = 3000
    enable_prometheus: bool = True
    enable_grafana: bool = True
    enable_alerting: bool = True
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'cpu_usage_percent': 80.0,
                'memory_usage_percent': 85.0,
                'response_time_ms': 100.0,
                'error_rate_percent': 5.0,
                'disk_usage_percent': 90.0,
                'network_latency_ms': 500.0
            }
        
        if self.alert_channels is None:
            self.alert_channels = ['console', 'file', 'webhook']

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    unit: str = ""
    help_text: str = ""
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

class MetricsCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self, config: MonitoringConfig, db_path: str = "performance_metrics.db"):
        self.config = config
        self.db_path = Path(db_path)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.custom_metrics: Dict[str, Callable] = {}
        self.running = False
        self.collection_thread = None
        
        # Initialize database
        self._init_database()
        
        # Set up logging
        self.logger = logging.getLogger('MetricsCollector')
        
        # Register default metrics
        self._register_default_metrics()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    labels TEXT,
                    unit TEXT,
                    help_text TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
    
    def _register_default_metrics(self):
        """Register default system metrics"""
        self.custom_metrics.update({
            'cpu_usage_percent': self._collect_cpu_usage,
            'memory_usage_percent': self._collect_memory_usage,
            'disk_usage_percent': self._collect_disk_usage,
            'network_bytes_sent': self._collect_network_sent,
            'network_bytes_recv': self._collect_network_recv,
            'process_memory_mb': self._collect_process_memory,
            'process_cpu_percent': self._collect_process_cpu,
            'thread_count': self._collect_thread_count,
            'open_file_descriptors': self._collect_open_files,
            'uptime_seconds': self._collect_uptime_seconds
        })
    
    def _collect_cpu_usage(self) -> float:
        """Collect CPU usage percentage"""
        return psutil.cpu_percent(interval=None)
    
    def _collect_memory_usage(self) -> float:
        """Collect memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def _collect_disk_usage(self) -> float:
        """Collect disk usage percentage"""
        return psutil.disk_usage('C:' if platform.system() == 'Windows' else '/').percent
    
    def _collect_network_sent(self) -> float:
        """Collect network bytes sent"""
        return psutil.net_io_counters().bytes_sent
    
    def _collect_network_recv(self) -> float:
        """Collect network bytes received"""
        return psutil.net_io_counters().bytes_recv
    
    def _collect_process_memory(self) -> float:
        """Collect process memory usage in MB"""
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def _collect_process_cpu(self) -> float:
        """Collect process CPU usage percentage"""
        return psutil.Process().cpu_percent()
    
    def _collect_thread_count(self) -> float:
        """Collect thread count"""
        return psutil.Process().num_threads()
    
    def _collect_open_files(self) -> float:
        """Collect open file descriptor count"""
        try:
            return len(psutil.Process().open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return 0
    
    def _collect_uptime_seconds(self) -> float:
        """Collect system uptime in seconds"""
        return time.time() - psutil.boot_time()
    
    def register_metric(self, name: str, collector: Callable[[], float], 
                       unit: str = "", help_text: str = ""):
        """Register a custom metric collector"""
        self.custom_metrics[name] = collector
        self.logger.info(f"Registered custom metric: {name}")
    
    def collect_metric(self, name: str, value: float, labels: Dict[str, str] = None, 
                      unit: str = "", help_text: str = ""):
        """Collect a single metric value"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            unit=unit,
            help_text=help_text
        )
        
        # Store in memory
        self.metrics[name].append(metric)
        
        # Store in database
        self._store_metric_db(metric)
    
    def _store_metric_db(self, metric: PerformanceMetric):
        """Store metric in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (name, value, timestamp, labels, unit, help_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.labels),
                    metric.unit,
                    metric.help_text
                ))
        except Exception as e:
            self.logger.error(f"Failed to store metric {metric.name}: {e}")
    
    def start_collection(self):
        """Start metric collection in background thread"""
        if self.running:
            self.logger.warning("Metric collection already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info(f"Started metric collection (interval: {self.config.collection_interval}s)")
    
    def stop_collection(self):
        """Stop metric collection"""
        if not self.running:
            return
        
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        self.logger.info("Stopped metric collection")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Create a copy of metrics to avoid concurrent modification
                metrics_to_collect = dict(self.custom_metrics)
                
                # Collect all registered metrics
                for name, collector in metrics_to_collect.items():
                    try:
                        value = collector()
                        self.collect_metric(name, value)
                    except Exception as e:
                        self.logger.error(f"Failed to collect metric {name}: {e}")
                
                time.sleep(self.config.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(self.config.collection_interval)
    
    def get_metrics(self, name: Optional[str] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> Dict[str, List[PerformanceMetric]]:
        """Retrieve metrics from memory or database"""
        if name:
            if name in self.metrics and not start_time and not end_time:
                return {name: list(self.metrics[name])}
            else:
                return {name: self._query_metrics_db(name, start_time, end_time)}
        else:
            # Return all metrics from memory if no time range specified
            if not start_time and not end_time:
                return {k: list(v) for k, v in self.metrics.items()}
            else:
                # Query database for time range
                return self._query_all_metrics_db(start_time, end_time)
    
    def _query_metrics_db(self, name: str, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Query specific metric from database"""
        query = "SELECT * FROM metrics WHERE name = ?"
        params = [name]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp"
        
        metrics = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                for row in cursor.fetchall():
                    metrics.append(PerformanceMetric(
                        name=row[1],
                        value=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        labels=json.loads(row[4]) if row[4] else {},
                        unit=row[5] or "",
                        help_text=row[6] or ""
                    ))
        except Exception as e:
            self.logger.error(f"Failed to query metrics for {name}: {e}")
        
        return metrics
    
    def _query_all_metrics_db(self, start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict[str, List[PerformanceMetric]]:
        """Query all metrics from database"""
        query = "SELECT * FROM metrics"
        params = []
        
        if start_time or end_time:
            query += " WHERE"
            conditions = []
            
            if start_time:
                conditions.append(" timestamp >= ?")
                params.append(start_time.isoformat())
            
            if end_time:
                conditions.append(" timestamp <= ?")
                params.append(end_time.isoformat())
            
            query += " AND".join(conditions)
        
        query += " ORDER BY name, timestamp"
        
        metrics = defaultdict(list)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                for row in cursor.fetchall():
                    metric = PerformanceMetric(
                        name=row[1],
                        value=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        labels=json.loads(row[4]) if row[4] else {},
                        unit=row[5] or "",
                        help_text=row[6] or ""
                    )
                    metrics[metric.name].append(metric)
        except Exception as e:
            self.logger.error(f"Failed to query all metrics: {e}")
        
        return dict(metrics)

class PerformanceAlerting:
    """Performance alerting system with configurable thresholds"""
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.collector = metrics_collector
        self.alert_history: List[Dict] = []
        self.active_alerts: Dict[str, Dict] = {}
        self.running = False
        self.alert_thread = None
        
        # Set up logging
        self.logger = logging.getLogger('PerformanceAlerting')
        
        # Set up alert channels
        self.alert_channels = {
            'console': self._alert_console,
            'file': self._alert_file,
            'webhook': self._alert_webhook
        }
    
    def start_alerting(self):
        """Start alerting system"""
        if not self.config.enable_alerting or self.running:
            return
        
        self.running = True
        self.alert_thread = threading.Thread(target=self._alerting_loop, daemon=True)
        self.alert_thread.start()
        self.logger.info("Started performance alerting system")
    
    def stop_alerting(self):
        """Stop alerting system"""
        if not self.running:
            return
        
        self.running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        
        self.logger.info("Stopped performance alerting system")
    
    def _alerting_loop(self):
        """Main alerting loop"""
        while self.running:
            try:
                self._check_thresholds()
                time.sleep(self.config.collection_interval * 2)  # Check less frequently than collection
            except Exception as e:
                self.logger.error(f"Error in alerting loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _check_thresholds(self):
        """Check all thresholds and trigger alerts"""
        current_metrics = self.collector.get_metrics()
        
        for threshold_name, threshold_value in self.config.alert_thresholds.items():
            # Map threshold names to metric names
            metric_name = self._map_threshold_to_metric(threshold_name)
            
            if metric_name in current_metrics and current_metrics[metric_name]:
                latest_metric = current_metrics[metric_name][-1]
                current_value = latest_metric.value
                
                # Check if threshold is exceeded
                if current_value > threshold_value:
                    self._trigger_alert(threshold_name, current_value, threshold_value, latest_metric)
                else:
                    # Resolve alert if it was active
                    self._resolve_alert(threshold_name, current_value, latest_metric)
    
    def _map_threshold_to_metric(self, threshold_name: str) -> str:
        """Map threshold names to actual metric names"""
        mapping = {
            'cpu_usage_percent': 'cpu_usage_percent',
            'memory_usage_percent': 'memory_usage_percent',
            'disk_usage_percent': 'disk_usage_percent',
            'response_time_ms': 'response_time_ms',
            'error_rate_percent': 'error_rate_percent',
            'network_latency_ms': 'network_latency_ms'
        }
        return mapping.get(threshold_name, threshold_name)
    
    def _trigger_alert(self, threshold_name: str, current_value: float, 
                      threshold_value: float, metric: PerformanceMetric):
        """Trigger an alert"""
        alert_key = f"{threshold_name}_{metric.name}"
        
        # Don't spam alerts - only trigger if not already active or value significantly changed
        if alert_key in self.active_alerts:
            last_value = self.active_alerts[alert_key]['current_value']
            if abs(current_value - last_value) < threshold_value * 0.1:  # Less than 10% change
                return
        
        alert_data = {
            'alert_id': alert_key,
            'threshold_name': threshold_name,
            'metric_name': metric.name,
            'current_value': current_value,
            'threshold_value': threshold_value,
            'severity': self._calculate_severity(current_value, threshold_value),
            'timestamp': datetime.now(timezone.utc),
            'status': 'active',
            'metric': metric
        }
        
        self.active_alerts[alert_key] = alert_data
        self.alert_history.append(alert_data.copy())
        
        # Send alert through configured channels
        self._send_alert(alert_data)
        
        self.logger.warning(f"ALERT TRIGGERED: {threshold_name} = {current_value:.2f} "
                           f"(threshold: {threshold_value})")
    
    def _resolve_alert(self, threshold_name: str, current_value: float, metric: PerformanceMetric):
        """Resolve an active alert"""
        alert_key = f"{threshold_name}_{metric.name}"
        
        if alert_key in self.active_alerts:
            alert_data = self.active_alerts[alert_key].copy()
            alert_data['status'] = 'resolved'
            alert_data['resolved_timestamp'] = datetime.now(timezone.utc)
            alert_data['resolved_value'] = current_value
            
            # Remove from active alerts
            del self.active_alerts[alert_key]
            
            # Add to history
            self.alert_history.append(alert_data)
            
            # Send resolution notification
            self._send_alert_resolution(alert_data)
            
            self.logger.info(f"ALERT RESOLVED: {threshold_name} = {current_value:.2f}")
    
    def _calculate_severity(self, current_value: float, threshold_value: float) -> str:
        """Calculate alert severity based on how much threshold is exceeded"""
        ratio = current_value / threshold_value
        
        if ratio >= 2.0:
            return 'critical'
        elif ratio >= 1.5:
            return 'high'
        elif ratio >= 1.2:
            return 'medium'
        else:
            return 'low'
    
    def _send_alert(self, alert_data: Dict):
        """Send alert through configured channels"""
        for channel in self.config.alert_channels:
            if channel in self.alert_channels:
                try:
                    self.alert_channels[channel](alert_data)
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _send_alert_resolution(self, alert_data: Dict):
        """Send alert resolution notification"""
        alert_data['notification_type'] = 'resolution'
        self._send_alert(alert_data)
    
    def _alert_console(self, alert_data: Dict):
        """Send alert to console"""
        if alert_data.get('notification_type') == 'resolution':
            print(f"[ALERT RESOLVED] {alert_data['threshold_name']}: "
                  f"{alert_data['resolved_value']:.2f} (back under threshold)")
        else:
            print(f"[ALERT {alert_data['severity'].upper()}] {alert_data['threshold_name']}: "
                  f"{alert_data['current_value']:.2f} exceeds threshold {alert_data['threshold_value']}")
    
    def _alert_file(self, alert_data: Dict):
        """Send alert to file"""
        alert_file = Path("performance_alerts.log")
        
        timestamp = alert_data['timestamp'].isoformat()
        if alert_data.get('notification_type') == 'resolution':
            message = (f"{timestamp} [RESOLVED] {alert_data['threshold_name']}: "
                      f"{alert_data['resolved_value']:.2f}\n")
        else:
            message = (f"{timestamp} [{alert_data['severity'].upper()}] "
                      f"{alert_data['threshold_name']}: {alert_data['current_value']:.2f} "
                      f"exceeds {alert_data['threshold_value']}\n")
        
        try:
            with open(alert_file, 'a') as f:
                f.write(message)
        except Exception as e:
            self.logger.error(f"Failed to write alert to file: {e}")
    
    def _alert_webhook(self, alert_data: Dict):
        """Send alert via webhook (placeholder implementation)"""
        # In a real implementation, this would send HTTP POST to webhook URL
        webhook_payload = {
            'alert_id': alert_data['alert_id'],
            'severity': alert_data.get('severity', 'unknown'),
            'message': f"{alert_data['threshold_name']}: {alert_data['current_value']:.2f}",
            'timestamp': alert_data['timestamp'].isoformat(),
            'status': alert_data['status']
        }
        
        self.logger.info(f"Webhook alert: {json.dumps(webhook_payload)}")

class PrometheusExporter:
    """Prometheus metrics exporter"""
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.collector = metrics_collector
        self.server = None
        self.logger = logging.getLogger('PrometheusExporter')
    
    def start_server(self):
        """Start Prometheus metrics server"""
        if not self.config.enable_prometheus:
            return
        
        handler = self._create_handler()
        self.server = HTTPServer(('localhost', self.config.metrics_port), handler)
        
        server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        server_thread.start()
        
        self.logger.info(f"Prometheus metrics server started on port {self.config.metrics_port}")
    
    def stop_server(self):
        """Stop Prometheus metrics server"""
        if self.server:
            self.server.shutdown()
            self.logger.info("Prometheus metrics server stopped")
    
    def _create_handler(self):
        """Create HTTP request handler for metrics endpoint"""
        collector = self.collector
        config = self.config
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == config.metrics_endpoint:
                    try:
                        metrics_text = self._format_prometheus_metrics()
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(metrics_text.encode('utf-8'))
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(f"Error: {str(e)}".encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def _format_prometheus_metrics(self) -> str:
                """Format metrics in Prometheus format"""
                metrics = collector.get_metrics()
                lines = []
                
                for metric_name, metric_list in metrics.items():
                    if not metric_list:
                        continue
                    
                    latest_metric = metric_list[-1]
                    
                    # Add help text
                    if latest_metric.help_text:
                        lines.append(f"# HELP {metric_name} {latest_metric.help_text}")
                    
                    # Add type (assume gauge for now)
                    lines.append(f"# TYPE {metric_name} gauge")
                    
                    # Add metric value
                    labels = ""
                    if latest_metric.labels:
                        label_pairs = [f'{k}="{v}"' for k, v in latest_metric.labels.items()]
                        labels = "{" + ",".join(label_pairs) + "}"
                    
                    lines.append(f"{metric_name}{labels} {latest_metric.value}")
                
                return "\n".join(lines) + "\n"
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        return MetricsHandler

class PerformanceMonitoringSystem:
    """Main performance monitoring system orchestrator"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.metrics_collector = MetricsCollector(self.config)
        self.alerting_system = PerformanceAlerting(self.config, self.metrics_collector)
        self.prometheus_exporter = PrometheusExporter(self.config, self.metrics_collector)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('performance_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PerformanceMonitoringSystem')
        
        # System state
        self.running = False
    
    def start(self):
        """Start the complete monitoring system"""
        if self.running:
            self.logger.warning("Monitoring system already running")
            return
        
        self.running = True
        
        # Start components
        self.metrics_collector.start_collection()
        self.alerting_system.start_alerting()
        self.prometheus_exporter.start_server()
        
        self.logger.info("Performance monitoring system started successfully")
        
        # Display system information
        self._display_startup_info()
    
    def stop(self):
        """Stop the complete monitoring system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        self.metrics_collector.stop_collection()
        self.alerting_system.stop_alerting()
        self.prometheus_exporter.stop_server()
        
        self.logger.info("Performance monitoring system stopped")
    
    def _display_startup_info(self):
        """Display system startup information"""
        info = [
            "=" * 60,
            "PERFORMANCE MONITORING SYSTEM ACTIVE",
            "=" * 60,
            f"Metrics Collection: {self.config.collection_interval}s intervals",
            f"Prometheus Endpoint: http://localhost:{self.config.metrics_port}{self.config.metrics_endpoint}",
            f"Alert Channels: {', '.join(self.config.alert_channels)}",
            f"Data Retention: {self.config.data_retention_days} days",
            "=" * 60,
            "ACTIVE THRESHOLDS:",
        ]
        
        for threshold, value in self.config.alert_thresholds.items():
            info.append(f"  {threshold}: {value}")
        
        info.append("=" * 60)
        
        for line in info:
            print(line)
    
    def add_custom_metric(self, name: str, collector: Callable[[], float], 
                         unit: str = "", help_text: str = ""):
        """Add a custom metric to the monitoring system"""
        self.metrics_collector.register_metric(name, collector, unit, help_text)
        self.logger.info(f"Added custom metric: {name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        recent_metrics = self.metrics_collector.get_metrics()
        active_alerts = list(self.alerting_system.active_alerts.values())
        
        # Calculate summary statistics
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'running': self.running,
            'metrics_count': len(recent_metrics),
            'active_alerts': len(active_alerts),
            'alert_history_count': len(self.alerting_system.alert_history),
            'system_health': 'healthy'  # Will be calculated based on alerts
        }
        
        # Determine system health
        if active_alerts:
            severities = [alert['severity'] for alert in active_alerts]
            if 'critical' in severities:
                status['system_health'] = 'critical'
            elif 'high' in severities:
                status['system_health'] = 'degraded'
            else:
                status['system_health'] = 'warning'
        
        # Add recent metric values
        status['current_metrics'] = {}
        for name, metrics_list in recent_metrics.items():
            if metrics_list:
                status['current_metrics'][name] = {
                    'value': metrics_list[-1].value,
                    'timestamp': metrics_list[-1].timestamp.isoformat(),
                    'unit': metrics_list[-1].unit
                }
        
        return status
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor a specific operation"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            # Record operation metrics
            duration_ms = (end_time - start_time) * 1000
            memory_delta_mb = (end_memory - start_memory) / (1024 * 1024)
            
            self.metrics_collector.collect_metric(
                f"operation_duration_ms",
                duration_ms,
                labels={'operation': operation_name},
                unit="milliseconds",
                help_text="Operation execution time"
            )
            
            self.metrics_collector.collect_metric(
                f"operation_memory_delta_mb",
                memory_delta_mb,
                labels={'operation': operation_name},
                unit="megabytes",
                help_text="Memory usage change during operation"
            )

def main():
    """Main function to demonstrate monitoring system"""
    print("AGENT BETA - Performance Monitoring Infrastructure")
    print("=" * 55)
    
    # Create custom configuration
    config = MonitoringConfig(
        metrics_port=9090,
        collection_interval=2.0,  # Collect every 2 seconds for demo
        alert_thresholds={
            'cpu_usage_percent': 50.0,  # Lower threshold for demo
            'memory_usage_percent': 60.0,
            'process_memory_mb': 200.0
        },
        alert_channels=['console', 'file'],
        enable_prometheus=True,
        enable_alerting=True
    )
    
    # Initialize monitoring system
    monitoring = PerformanceMonitoringSystem(config)
    
    try:
        # Start monitoring
        monitoring.start()
        
        # Add custom metrics
        monitoring.add_custom_metric(
            "demo_operations_total",
            lambda: time.time() % 100,  # Demo metric
            unit="count",
            help_text="Total demo operations performed"
        )
        
        # Simulate some operations to generate metrics
        print("\nSimulating operations to generate metrics...")
        
        for i in range(10):
            with monitoring.monitor_operation(f"demo_operation_{i}"):
                # Simulate work
                time.sleep(0.1)
                result = sum(j * j for j in range(1000))
                print(f"  Operation {i}: result = {result}")
        
        # Wait for metrics collection and potential alerts
        print("\nCollecting metrics for 30 seconds...")
        print("Check http://localhost:9090/metrics for Prometheus format")
        print("Press Ctrl+C to stop...")
        
        time.sleep(30)
        
        # Display final status
        status = monitoring.get_system_status()
        print(f"\nFinal System Status:")
        print(f"  System Health: {status['system_health']}")
        print(f"  Active Metrics: {status['metrics_count']}")
        print(f"  Active Alerts: {status['active_alerts']}")
        
    except KeyboardInterrupt:
        print("\nShutting down monitoring system...")
    
    finally:
        monitoring.stop()
        print("Monitoring system stopped successfully")

if __name__ == "__main__":
    main()