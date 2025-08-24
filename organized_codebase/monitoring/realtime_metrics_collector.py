#!/usr/bin/env python3
"""
Real-Time Metrics Collection System
===================================
High-frequency metrics collection with 100ms intervals for enterprise monitoring.
Provides comprehensive system and application metrics with minimal performance impact.
"""

import sys
import time
import json
import asyncio
import threading
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import statistics
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric measurement point."""
    timestamp: float
    value: float
    tags: Dict[str, str]
    metric_type: str  # 'gauge', 'counter', 'histogram', 'summary'

@dataclass
class SystemMetrics:
    """System-level metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_read_bytes: int
    disk_write_bytes: int
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    open_files: int
    load_average: List[float]

@dataclass
class ApplicationMetrics:
    """Application-level metrics snapshot."""
    timestamp: float
    request_count: int
    response_time_ms: float
    error_rate: float
    active_connections: int
    queue_size: int
    cache_hit_rate: float
    throughput_rps: float
    custom_metrics: Dict[str, float]

class MetricsBuffer:
    """Thread-safe circular buffer for metrics storage."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, metric: MetricPoint):
        """Add metric to buffer."""
        with self.lock:
            self.buffer.append(metric)
    
    def get_recent(self, seconds: int = 60) -> List[MetricPoint]:
        """Get metrics from last N seconds."""
        cutoff = time.time() - seconds
        with self.lock:
            return [m for m in self.buffer if m.timestamp >= cutoff]
    
    def get_all(self) -> List[MetricPoint]:
        """Get all metrics in buffer."""
        with self.lock:
            return list(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()

class MetricsAggregator:
    """Aggregates and analyzes metrics data."""
    
    def __init__(self):
        self.metrics_by_name = defaultdict(list)
        self.last_aggregation = time.time()
    
    def add_metric(self, name: str, metric: MetricPoint):
        """Add metric for aggregation."""
        self.metrics_by_name[name].append(metric)
        
        # Keep only recent metrics to prevent memory growth
        cutoff = time.time() - 300  # 5 minutes
        self.metrics_by_name[name] = [
            m for m in self.metrics_by_name[name] 
            if m.timestamp >= cutoff
        ]
    
    def get_statistics(self, metric_name: str, seconds: int = 60) -> Dict[str, float]:
        """Get statistical summary for a metric."""
        cutoff = time.time() - seconds
        values = [
            m.value for m in self.metrics_by_name[metric_name]
            if m.timestamp >= cutoff
        ]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

class AlertManager:
    """Manages metric-based alerting."""
    
    def __init__(self):
        self.thresholds = {}
        self.alert_callbacks = []
        self.alert_history = deque(maxlen=1000)
        self.alert_cooldown = {}  # Prevent alert spam
    
    def set_threshold(self, metric_name: str, operator: str, value: float, 
                     duration_seconds: int = 30):
        """Set alert threshold for a metric."""
        self.thresholds[metric_name] = {
            'operator': operator,  # 'gt', 'lt', 'eq'
            'value': value,
            'duration': duration_seconds,
            'violations': deque(maxlen=100)
        }
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def check_metric(self, metric_name: str, metric: MetricPoint):
        """Check if metric violates thresholds."""
        if metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_name]
        violated = self._check_threshold(metric.value, threshold)
        
        if violated:
            threshold['violations'].append(metric.timestamp)
        
        # Check if we have sustained violations
        recent_violations = [
            v for v in threshold['violations']
            if metric.timestamp - v <= threshold['duration']
        ]
        
        if len(recent_violations) >= 3:  # Sustained violation
            self._trigger_alert(metric_name, metric, threshold)
    
    def _check_threshold(self, value: float, threshold: Dict) -> bool:
        """Check if value violates threshold."""
        operator = threshold['operator']
        threshold_value = threshold['value']
        
        if operator == 'gt':
            return value > threshold_value
        elif operator == 'lt':
            return value < threshold_value
        elif operator == 'eq':
            return abs(value - threshold_value) < 0.001
        
        return False
    
    def _trigger_alert(self, metric_name: str, metric: MetricPoint, threshold: Dict):
        """Trigger alert for threshold violation."""
        # Check cooldown to prevent spam
        cooldown_key = f"{metric_name}_{threshold['operator']}_{threshold['value']}"
        if cooldown_key in self.alert_cooldown:
            if time.time() - self.alert_cooldown[cooldown_key] < 60:  # 1 minute cooldown
                return
        
        alert = {
            'timestamp': metric.timestamp,
            'metric_name': metric_name,
            'value': metric.value,
            'threshold': threshold['value'],
            'operator': threshold['operator'],
            'severity': self._calculate_severity(metric.value, threshold)
        }
        
        self.alert_history.append(alert)
        self.alert_cooldown[cooldown_key] = time.time()
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _calculate_severity(self, value: float, threshold: Dict) -> str:
        """Calculate alert severity based on threshold violation magnitude."""
        threshold_value = threshold['value']
        if threshold['operator'] == 'gt':
            ratio = value / threshold_value
        elif threshold['operator'] == 'lt':
            ratio = threshold_value / value
        else:
            return 'medium'
        
        if ratio >= 2.0:
            return 'critical'
        elif ratio >= 1.5:
            return 'high'
        else:
            return 'medium'

class RealtimeMetricsCollector:
    """High-frequency real-time metrics collection system."""
    
    def __init__(self, collection_interval_ms: int = 100):
        self.collection_interval = collection_interval_ms / 1000.0  # Convert to seconds
        self.running = False
        self.collector_thread = None
        
        # Storage
        self.system_buffer = MetricsBuffer(max_size=2000)
        self.application_buffer = MetricsBuffer(max_size=2000)
        self.custom_buffer = MetricsBuffer(max_size=2000)
        
        # Analytics
        self.aggregator = MetricsAggregator()
        self.alert_manager = AlertManager()
        
        # State tracking
        self.last_system_metrics = None
        self.collection_stats = {
            'collections': 0,
            'errors': 0,
            'start_time': None,
            'avg_collection_time': 0
        }
        
        # Performance tracking
        self.collection_times = deque(maxlen=100)
        
        # Custom metric callbacks
        self.custom_collectors = {}
        
        # Setup default alerts
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default system alert thresholds."""
        self.alert_manager.set_threshold('cpu_percent', 'gt', 80.0, duration_seconds=30)
        self.alert_manager.set_threshold('memory_percent', 'gt', 85.0, duration_seconds=30)
        self.alert_manager.set_threshold('disk_usage_percent', 'gt', 90.0, duration_seconds=60)
        self.alert_manager.set_threshold('response_time_ms', 'gt', 1000.0, duration_seconds=20)
        self.alert_manager.set_threshold('error_rate', 'gt', 5.0, duration_seconds=30)
        
        # Add default alert logger
        self.alert_manager.add_alert_callback(self._log_alert)
    
    def _log_alert(self, alert: Dict):
        """Default alert logging callback."""
        severity = alert['severity'].upper()
        logger.warning(f"[{severity}] ALERT: {alert['metric_name']} = {alert['value']:.2f} "
                      f"(threshold: {alert['threshold']:.2f})")
    
    def register_custom_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register custom metrics collector function."""
        self.custom_collectors[name] = collector_func
    
    def start_collection(self):
        """Start real-time metrics collection."""
        if self.running:
            logger.warning("Metrics collection already running")
            return
        
        self.running = True
        self.collection_stats['start_time'] = time.time()
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        
        logger.info(f"Started real-time metrics collection (interval: {self.collection_interval*1000:.0f}ms)")
    
    def stop_collection(self):
        """Stop real-time metrics collection."""
        if not self.running:
            return
        
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=1.0)
        
        logger.info("Stopped real-time metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running at specified interval."""
        while self.running:
            start_time = time.time()
            
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                self._collect_custom_metrics()
                
                self.collection_stats['collections'] += 1
                
            except Exception as e:
                self.collection_stats['errors'] += 1
                logger.error(f"Metrics collection error: {e}")
            
            # Track collection performance
            collection_time = time.time() - start_time
            self.collection_times.append(collection_time)
            
            # Calculate sleep time to maintain interval
            sleep_time = max(0, self.collection_interval - collection_time)
            time.sleep(sleep_time)
            
            # Update average collection time
            if self.collection_times:
                self.collection_stats['avg_collection_time'] = statistics.mean(self.collection_times)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        current_time = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]  # Windows fallback
            
            # Open files count
            try:
                open_files = len(psutil.Process().open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            # Create system metrics snapshot
            metrics = SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_read_bytes=disk_io.read_bytes if disk_io else 0,
                disk_write_bytes=disk_io.write_bytes if disk_io else 0,
                network_bytes_sent=network.bytes_sent if network else 0,
                network_bytes_recv=network.bytes_recv if network else 0,
                process_count=process_count,
                open_files=open_files,
                load_average=load_avg
            )
            
            # Store as individual metric points for analysis
            for field_name, value in asdict(metrics).items():
                if field_name == 'timestamp':
                    continue
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        metric_point = MetricPoint(
                            timestamp=current_time,
                            value=v,
                            tags={'component': 'system', 'index': str(i)},
                            metric_type='gauge'
                        )
                        self.system_buffer.add(metric_point)
                        self.aggregator.add_metric(f"{field_name}_{i}", metric_point)
                        self.alert_manager.check_metric(f"{field_name}_{i}", metric_point)
                else:
                    metric_point = MetricPoint(
                        timestamp=current_time,
                        value=float(value),
                        tags={'component': 'system'},
                        metric_type='gauge'
                    )
                    self.system_buffer.add(metric_point)
                    self.aggregator.add_metric(field_name, metric_point)
                    self.alert_manager.check_metric(field_name, metric_point)
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-level metrics."""
        current_time = time.time()
        
        try:
            # These would typically come from your application
            # For demonstration, we'll simulate some metrics
            metrics = ApplicationMetrics(
                timestamp=current_time,
                request_count=self._get_simulated_metric('request_count', 100, 1000),
                response_time_ms=self._get_simulated_metric('response_time', 50, 500),
                error_rate=self._get_simulated_metric('error_rate', 0, 5),
                active_connections=self._get_simulated_metric('connections', 10, 100),
                queue_size=self._get_simulated_metric('queue_size', 0, 50),
                cache_hit_rate=self._get_simulated_metric('cache_hit_rate', 80, 95),
                throughput_rps=self._get_simulated_metric('throughput', 100, 1000),
                custom_metrics={}
            )
            
            # Store as individual metric points
            for field_name, value in asdict(metrics).items():
                if field_name in ['timestamp', 'custom_metrics']:
                    continue
                
                metric_point = MetricPoint(
                    timestamp=current_time,
                    value=float(value),
                    tags={'component': 'application'},
                    metric_type='gauge'
                )
                self.application_buffer.add(metric_point)
                self.aggregator.add_metric(field_name, metric_point)
                self.alert_manager.check_metric(field_name, metric_point)
            
        except Exception as e:
            logger.error(f"Application metrics collection error: {e}")
    
    def _collect_custom_metrics(self):
        """Collect custom metrics from registered collectors."""
        current_time = time.time()
        
        for collector_name, collector_func in self.custom_collectors.items():
            try:
                custom_metrics = collector_func()
                
                for metric_name, value in custom_metrics.items():
                    metric_point = MetricPoint(
                        timestamp=current_time,
                        value=float(value),
                        tags={'component': 'custom', 'collector': collector_name},
                        metric_type='gauge'
                    )
                    self.custom_buffer.add(metric_point)
                    self.aggregator.add_metric(f"custom_{metric_name}", metric_point)
                    self.alert_manager.check_metric(f"custom_{metric_name}", metric_point)
                    
            except Exception as e:
                logger.error(f"Custom collector '{collector_name}' error: {e}")
    
    def _get_simulated_metric(self, metric_name: str, min_val: float, max_val: float) -> float:
        """Generate simulated metric values for demonstration."""
        import random
        
        # Add some noise and trends to make it more realistic
        base_time = time.time() % 3600  # Hour cycle
        trend = 0.1 * (base_time / 3600)  # Slight upward trend
        noise = random.uniform(-0.1, 0.1)
        
        normalized = (trend + noise + 1) / 2  # Normalize to 0-1
        return min_val + (max_val - min_val) * normalized
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        current_time = time.time()
        
        return {
            'timestamp': current_time,
            'collection_stats': self.collection_stats.copy(),
            'system_metrics': {
                'recent_count': len(self.system_buffer.get_recent(60)),
                'total_count': len(self.system_buffer.get_all())
            },
            'application_metrics': {
                'recent_count': len(self.application_buffer.get_recent(60)),
                'total_count': len(self.application_buffer.get_all())
            },
            'custom_metrics': {
                'recent_count': len(self.custom_buffer.get_recent(60)),
                'total_count': len(self.custom_buffer.get_all())
            },
            'performance': {
                'avg_collection_time_ms': self.collection_stats['avg_collection_time'] * 1000,
                'collections_per_second': self.collection_stats['collections'] / max(1, current_time - (self.collection_stats['start_time'] or current_time)),
                'error_rate': self.collection_stats['errors'] / max(1, self.collection_stats['collections']) * 100
            }
        }
    
    def get_metric_statistics(self, metric_name: str, seconds: int = 60) -> Dict[str, float]:
        """Get statistical analysis for a specific metric."""
        return self.aggregator.get_statistics(metric_name, seconds)
    
    def get_alert_history(self, count: int = 50) -> List[Dict]:
        """Get recent alert history."""
        alerts = list(self.alert_manager.alert_history)
        return alerts[-count:] if count < len(alerts) else alerts
    
    def export_metrics(self, format: str = 'json', filename: Optional[str] = None) -> str:
        """Export collected metrics to file."""
        current_time = time.time()
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metrics_export_{timestamp}.{format}"
        
        export_data = {
            'export_timestamp': current_time,
            'collection_stats': self.collection_stats,
            'system_metrics': [asdict(m) for m in self.system_buffer.get_all()],
            'application_metrics': [asdict(m) for m in self.application_buffer.get_all()],
            'custom_metrics': [asdict(m) for m in self.custom_buffer.get_all()],
            'alert_history': list(self.alert_manager.alert_history)
        }
        
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to: {filename}")
        return filename


def main():
    """Demo and testing of real-time metrics collection."""
    # Create collector with 100ms interval
    collector = RealtimeMetricsCollector(collection_interval_ms=100)
    
    # Register a custom collector
    def sample_custom_collector():
        return {
            'test_metric_1': time.time() % 100,
            'test_metric_2': (time.time() * 2) % 50
        }
    
    collector.register_custom_collector('test_collector', sample_custom_collector)
    
    # Add custom alert callback
    def custom_alert_handler(alert):
        print(f"🚨 CUSTOM ALERT: {alert['metric_name']} = {alert['value']}")
    
    collector.alert_manager.add_alert_callback(custom_alert_handler)
    
    try:
        # Start collection
        collector.start_collection()
        
        # Let it run for a bit
        logger.info("Collecting metrics for 10 seconds...")
        time.sleep(10)
        
        # Show current status
        current_metrics = collector.get_current_metrics()
        logger.info(f"Current metrics: {json.dumps(current_metrics, indent=2)}")
        
        # Show some statistics
        cpu_stats = collector.get_metric_statistics('cpu_percent', seconds=10)
        logger.info(f"CPU statistics: {json.dumps(cpu_stats, indent=2)}")
        
        # Show alerts
        alerts = collector.get_alert_history(10)
        if alerts:
            logger.info(f"Recent alerts: {json.dumps(alerts, indent=2)}")
        
        # Export metrics
        export_file = collector.export_metrics()
        logger.info(f"Metrics exported to: {export_file}")
        
    finally:
        # Stop collection
        collector.stop_collection()
        logger.info("Collection stopped")


if __name__ == "__main__":
    main()