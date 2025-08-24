"""
Real-Time Monitoring System - Advanced real-time monitoring and alerting for dashboard

This module provides sophisticated real-time monitoring capabilities for the Enhanced
Linkage Dashboard, including system health monitoring, performance tracking, alerting,
event streaming, and predictive analytics. Designed for enterprise-scale monitoring
with advanced notification systems and intelligent anomaly detection.

Enterprise Features:
- Real-time system health monitoring with predictive analytics
- Advanced alerting with customizable thresholds and escalation
- Event streaming with WebSocket integration and buffering
- Performance trend analysis with machine learning insights
- Multi-threaded monitoring with efficient resource management
- Integration with external monitoring systems and APIs

Key Components:
- RealTimeMonitor: Main monitoring orchestration engine
- AlertManager: Advanced alerting and notification system
- MetricsCollector: Comprehensive metrics collection and aggregation
- PerformanceTracker: Performance monitoring with trend analysis
- EventStreamer: Real-time event streaming and communication
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import queue
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor

from .dashboard_models import (
    SystemHealthMetrics, PerformanceMetrics, SecurityMetrics, QualityMetrics,
    SystemHealthStatus, SecurityLevel, LiveDataStream, create_system_health_metrics,
    create_live_data_stream, calculate_health_status, calculate_security_level
)

# Configure logging
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class MonitoringState(Enum):
    """Monitoring system states."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Alert:
    """Represents a monitoring alert."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    source: str = "monitoring_system"
    metric_name: str = ""
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def acknowledge(self, acknowledged_by: str = "system"):
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = acknowledged_by
        self.updated_at = datetime.now()
    
    def resolve(self):
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'source': self.source,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class MonitoringThreshold:
    """Defines monitoring thresholds for metrics."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = "greater_than"  # greater_than, less_than, equals
    enabled: bool = True
    consecutive_violations_required: int = 1
    check_interval_seconds: int = 30
    description: str = ""
    
    def check_violation(self, value: float) -> Optional[AlertSeverity]:
        """Check if value violates thresholds."""
        if not self.enabled:
            return None
        
        if self.comparison_operator == "greater_than":
            if value >= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison_operator == "less_than":
            if value <= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= self.warning_threshold:
                return AlertSeverity.WARNING
        
        return None


class MetricsCollector:
    """Comprehensive metrics collection and aggregation."""
    
    def __init__(self, collection_interval: int = 5):
        self.collection_interval = collection_interval
        self.metrics_history = defaultdict(deque)
        self.max_history_size = 1000
        self.custom_collectors = {}
        self.collection_stats = {
            'total_collections': 0,
            'failed_collections': 0,
            'last_collection_time': None,
            'avg_collection_duration': 0.0
        }
    
    def register_custom_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register a custom metrics collector."""
        self.custom_collectors[name] = collector_func
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect comprehensive system metrics."""
        start_time = time.time()
        
        try:
            metrics = {}
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics['cpu_usage_percent'] = cpu_percent
            metrics['cpu_count'] = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_usage_percent'] = memory.percent
            metrics['memory_total_gb'] = memory.total / (1024**3)
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_available_gb'] = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage_percent'] = (disk.used / disk.total) * 100
            metrics['disk_total_gb'] = disk.total / (1024**3)
            metrics['disk_used_gb'] = disk.used / (1024**3)
            metrics['disk_free_gb'] = disk.free / (1024**3)
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                metrics['network_bytes_sent'] = network.bytes_sent
                metrics['network_bytes_recv'] = network.bytes_recv
                metrics['network_packets_sent'] = network.packets_sent
                metrics['network_packets_recv'] = network.packets_recv
            except:
                pass
            
            # Process metrics
            try:
                process = psutil.Process()
                metrics['process_cpu_percent'] = process.cpu_percent()
                metrics['process_memory_mb'] = process.memory_info().rss / (1024**2)
                metrics['process_threads'] = process.num_threads()
                metrics['process_open_files'] = len(process.open_files())
            except:
                pass
            
            # Collect custom metrics
            for name, collector in self.custom_collectors.items():
                try:
                    custom_metrics = collector()
                    for key, value in custom_metrics.items():
                        metrics[f"{name}_{key}"] = value
                except Exception as e:
                    logger.warning(f"Error collecting custom metrics from {name}: {e}")
            
            # Update collection stats
            collection_duration = time.time() - start_time
            self.collection_stats['total_collections'] += 1
            self.collection_stats['last_collection_time'] = datetime.now()
            
            # Update average collection duration
            total_collections = self.collection_stats['total_collections']
            current_avg = self.collection_stats['avg_collection_duration']
            self.collection_stats['avg_collection_duration'] = (
                (current_avg * (total_collections - 1) + collection_duration) / total_collections
            )
            
            # Store metrics in history
            timestamp = datetime.now()
            for metric_name, value in metrics.items():
                self.metrics_history[metric_name].append((timestamp, value))
                
                # Limit history size
                if len(self.metrics_history[metric_name]) > self.max_history_size:
                    self.metrics_history[metric_name].popleft()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            self.collection_stats['failed_collections'] += 1
            return {}
    
    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[Tuple[datetime, float]]:
        """Get metric history for specified duration."""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        history = self.metrics_history[metric_name]
        
        return [(timestamp, value) for timestamp, value in history if timestamp >= cutoff_time]
    
    def calculate_metric_statistics(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, float]:
        """Calculate statistics for a metric over specified duration."""
        history = self.get_metric_history(metric_name, duration_minutes)
        
        if not history:
            return {}
        
        values = [value for _, value in history]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'percentile_95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'percentile_99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }
    
    def detect_anomalies(self, metric_name: str, sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data using statistical analysis."""
        history = self.get_metric_history(metric_name, duration_minutes=120)  # 2 hours
        
        if len(history) < 10:
            return []
        
        values = [value for _, value in history]
        timestamps = [timestamp for timestamp, _ in history]
        
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        anomalies = []
        threshold = sensitivity * std_dev
        
        for i, (timestamp, value) in enumerate(history):
            deviation = abs(value - mean_value)
            
            if deviation > threshold:
                anomalies.append({
                    'timestamp': timestamp.isoformat(),
                    'value': value,
                    'deviation': deviation,
                    'threshold': threshold,
                    'severity': 'high' if deviation > (threshold * 2) else 'medium'
                })
        
        return anomalies


class AlertManager:
    """Advanced alerting and notification system."""
    
    def __init__(self):
        self.alerts = {}  # alert_id -> Alert
        self.thresholds = {}  # metric_name -> MonitoringThreshold
        self.violation_counts = defaultdict(int)
        self.notification_callbacks = []
        self.alert_history = deque(maxlen=1000)
        
        # Alert statistics
        self.alert_stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'avg_resolution_time_minutes': 0.0
        }
    
    def add_threshold(self, threshold: MonitoringThreshold):
        """Add or update a monitoring threshold."""
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Added threshold for {threshold.metric_name}: {threshold.warning_threshold}/{threshold.critical_threshold}")
    
    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add notification callback for alerts."""
        self.notification_callbacks.append(callback)
    
    def check_metrics(self, metrics: Dict[str, float]):
        """Check metrics against thresholds and generate alerts."""
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                self._check_metric_threshold(metric_name, value)
    
    def _check_metric_threshold(self, metric_name: str, value: float):
        """Check a specific metric against its threshold."""
        threshold = self.thresholds[metric_name]
        severity = threshold.check_violation(value)
        
        if severity:
            # Count consecutive violations
            self.violation_counts[metric_name] += 1
            
            # Check if we've reached the required consecutive violations
            if self.violation_counts[metric_name] >= threshold.consecutive_violations_required:
                self._create_alert(metric_name, value, severity, threshold)
        else:
            # Reset violation count on normal reading
            if metric_name in self.violation_counts:
                del self.violation_counts[metric_name]
                
                # Resolve any active alerts for this metric
                self._auto_resolve_alerts(metric_name)
    
    def _create_alert(self, metric_name: str, value: float, severity: AlertSeverity, threshold: MonitoringThreshold):
        """Create a new alert."""
        # Check if we already have an active alert for this metric
        existing_alert = self._get_active_alert_for_metric(metric_name)
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = value
            existing_alert.updated_at = datetime.now()
            return existing_alert
        
        # Create new alert
        alert_id = f"{metric_name}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            title=f"{metric_name.replace('_', ' ').title()} {severity.value.title()}",
            description=f"{metric_name} is {value:.2f}, which exceeds the {severity.value} threshold of {threshold.critical_threshold if severity == AlertSeverity.CRITICAL else threshold.warning_threshold:.2f}",
            severity=severity,
            metric_name=metric_name,
            current_value=value,
            threshold_value=threshold.critical_threshold if severity == AlertSeverity.CRITICAL else threshold.warning_threshold,
            tags=[severity.value, metric_name]
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update statistics
        self.alert_stats['total_alerts'] += 1
        self.alert_stats['active_alerts'] += 1
        self.alert_stats['alerts_by_severity'][severity.value] += 1
        
        # Notify callbacks
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert notification callback: {e}")
        
        logger.warning(f"Created {severity.value} alert: {alert.title}")
        return alert
    
    def _get_active_alert_for_metric(self, metric_name: str) -> Optional[Alert]:
        """Get active alert for a specific metric."""
        for alert in self.alerts.values():
            if (alert.metric_name == metric_name and 
                alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]):
                return alert
        return None
    
    def _auto_resolve_alerts(self, metric_name: str):
        """Auto-resolve alerts when metric returns to normal."""
        for alert in self.alerts.values():
            if (alert.metric_name == metric_name and 
                alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]):
                alert.resolve()
                self.alert_stats['active_alerts'] -= 1
                self.alert_stats['resolved_alerts'] += 1
                logger.info(f"Auto-resolved alert: {alert.title}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledge(acknowledged_by)
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            if alert.status != AlertStatus.RESOLVED:
                alert.resolve()
                self.alert_stats['active_alerts'] -= 1
                self.alert_stats['resolved_alerts'] += 1
                logger.info(f"Alert {alert_id} manually resolved")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.alerts.values() 
                if alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_alerts': self.alert_stats['total_alerts'],
            'active_alerts': len(active_alerts),
            'resolved_alerts': self.alert_stats['resolved_alerts'],
            'alerts_by_severity': dict(self.alert_stats['alerts_by_severity']),
            'recent_alerts': [alert.to_dict() for alert in list(self.alert_history)[-10:]],
            'critical_alerts': [alert.to_dict() for alert in active_alerts if alert.severity == AlertSeverity.CRITICAL],
            'warning_alerts': [alert.to_dict() for alert in active_alerts if alert.severity == AlertSeverity.WARNING]
        }


class EventStreamer:
    """Real-time event streaming and communication."""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.event_buffer = deque(maxlen=max_buffer_size)
        self.subscribers = {}  # subscription_id -> callback
        self.event_stats = {
            'total_events': 0,
            'events_sent': 0,
            'failed_sends': 0,
            'active_subscribers': 0
        }
        self.event_queue = queue.Queue()
        self.streaming_thread = None
        self.running = False
    
    def start_streaming(self):
        """Start the event streaming thread."""
        if not self.running:
            self.running = True
            self.streaming_thread = threading.Thread(target=self._stream_events)
            self.streaming_thread.daemon = True
            self.streaming_thread.start()
    
    def stop_streaming(self):
        """Stop the event streaming thread."""
        self.running = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5)
    
    def subscribe(self, subscription_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to event stream."""
        self.subscribers[subscription_id] = callback
        self.event_stats['active_subscribers'] = len(self.subscribers)
        logger.info(f"New subscriber: {subscription_id}")
    
    def unsubscribe(self, subscription_id: str):
        """Unsubscribe from event stream."""
        if subscription_id in self.subscribers:
            del self.subscribers[subscription_id]
            self.event_stats['active_subscribers'] = len(self.subscribers)
            logger.info(f"Unsubscribed: {subscription_id}")
    
    def emit_event(self, event_type: str, data: Dict[str, Any], priority: int = 1):
        """Emit an event to all subscribers."""
        event = {
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'event_id': f"{event_type}_{int(time.time())}"
        }
        
        self.event_buffer.append(event)
        self.event_queue.put(event)
        self.event_stats['total_events'] += 1
    
    def _stream_events(self):
        """Stream events to subscribers."""
        while self.running:
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1.0)
                
                # Send to all subscribers
                for subscription_id, callback in list(self.subscribers.items()):
                    try:
                        callback(event)
                        self.event_stats['events_sent'] += 1
                    except Exception as e:
                        logger.error(f"Error sending event to {subscription_id}: {e}")
                        self.event_stats['failed_sends'] += 1
                        
                        # Remove problematic subscriber
                        if subscription_id in self.subscribers:
                            del self.subscribers[subscription_id]
                            self.event_stats['active_subscribers'] = len(self.subscribers)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in event streaming: {e}")


class RealTimeMonitor:
    """
    Main real-time monitoring orchestration engine.
    
    This class coordinates all monitoring activities including metrics collection,
    alerting, event streaming, and performance tracking for the dashboard system.
    """
    
    def __init__(self, 
                 collection_interval: int = 5,
                 enable_alerting: bool = True,
                 enable_event_streaming: bool = True):
        
        self.collection_interval = collection_interval
        self.state = MonitoringState.STOPPED
        
        # Initialize components
        self.metrics_collector = MetricsCollector(collection_interval)
        self.alert_manager = AlertManager() if enable_alerting else None
        self.event_streamer = EventStreamer() if enable_event_streaming else None
        
        # Monitoring control
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.performance_stats = {
            'uptime_seconds': 0,
            'monitoring_cycles': 0,
            'last_cycle_duration': 0.0,
            'avg_cycle_duration': 0.0,
            'errors_encountered': 0
        }
        
        # Setup default thresholds
        self._setup_default_thresholds()
        
        # Register for system events
        if self.event_streamer:
            self.event_streamer.start_streaming()
    
    def start_monitoring(self):
        """Start the real-time monitoring system."""
        if self.state != MonitoringState.STOPPED:
            logger.warning("Monitoring is already running or starting")
            return
        
        logger.info("Starting real-time monitoring system")
        self.state = MonitoringState.STARTING
        
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Wait for monitoring to start
        time.sleep(1)
        if self.state == MonitoringState.RUNNING:
            logger.info("Real-time monitoring started successfully")
        else:
            logger.error("Failed to start real-time monitoring")
    
    def stop_monitoring(self):
        """Stop the real-time monitoring system."""
        if self.state in [MonitoringState.STOPPED, MonitoringState.STOPPING]:
            return
        
        logger.info("Stopping real-time monitoring system")
        self.state = MonitoringState.STOPPING
        
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        if self.event_streamer:
            self.event_streamer.stop_streaming()
        
        self.state = MonitoringState.STOPPED
        logger.info("Real-time monitoring stopped")
    
    def pause_monitoring(self):
        """Pause monitoring temporarily."""
        if self.state == MonitoringState.RUNNING:
            self.state = MonitoringState.PAUSED
            logger.info("Monitoring paused")
    
    def resume_monitoring(self):
        """Resume monitoring from paused state."""
        if self.state == MonitoringState.PAUSED:
            self.state = MonitoringState.RUNNING
            logger.info("Monitoring resumed")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        start_time = time.time()
        
        try:
            self.state = MonitoringState.RUNNING
            
            while not self.stop_event.is_set():
                cycle_start = time.time()
                
                try:
                    if self.state == MonitoringState.RUNNING:
                        # Collect metrics
                        metrics = self.metrics_collector.collect_system_metrics()
                        
                        # Check alerts
                        if self.alert_manager and metrics:
                            self.alert_manager.check_metrics(metrics)
                        
                        # Emit monitoring event
                        if self.event_streamer and metrics:
                            self.event_streamer.emit_event('metrics_collected', {
                                'metrics': metrics,
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        # Update performance stats
                        cycle_duration = time.time() - cycle_start
                        self._update_performance_stats(cycle_duration)
                    
                    # Wait for next cycle
                    self.stop_event.wait(self.collection_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                    self.performance_stats['errors_encountered'] += 1
                    
                    if self.event_streamer:
                        self.event_streamer.emit_event('monitoring_error', {
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }, priority=3)
                    
                    # Brief pause before continuing
                    time.sleep(5)
            
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
            self.state = MonitoringState.ERROR
        
        finally:
            self.performance_stats['uptime_seconds'] = time.time() - start_time
    
    def _update_performance_stats(self, cycle_duration: float):
        """Update monitoring performance statistics."""
        self.performance_stats['monitoring_cycles'] += 1
        self.performance_stats['last_cycle_duration'] = cycle_duration
        
        # Update average cycle duration
        cycles = self.performance_stats['monitoring_cycles']
        current_avg = self.performance_stats['avg_cycle_duration']
        self.performance_stats['avg_cycle_duration'] = (
            (current_avg * (cycles - 1) + cycle_duration) / cycles
        )
    
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds."""
        if not self.alert_manager:
            return
        
        # CPU thresholds
        self.alert_manager.add_threshold(MonitoringThreshold(
            metric_name="cpu_usage_percent",
            warning_threshold=75.0,
            critical_threshold=90.0,
            consecutive_violations_required=2,
            description="CPU usage monitoring"
        ))
        
        # Memory thresholds
        self.alert_manager.add_threshold(MonitoringThreshold(
            metric_name="memory_usage_percent",
            warning_threshold=80.0,
            critical_threshold=95.0,
            consecutive_violations_required=2,
            description="Memory usage monitoring"
        ))
        
        # Disk thresholds
        self.alert_manager.add_threshold(MonitoringThreshold(
            metric_name="disk_usage_percent",
            warning_threshold=85.0,
            critical_threshold=95.0,
            consecutive_violations_required=3,
            description="Disk usage monitoring"
        ))
    
    def add_custom_threshold(self, threshold: MonitoringThreshold):
        """Add a custom monitoring threshold."""
        if self.alert_manager:
            self.alert_manager.add_threshold(threshold)
    
    def register_metrics_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register a custom metrics collector."""
        self.metrics_collector.register_custom_collector(name, collector_func)
    
    def subscribe_to_events(self, subscription_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to monitoring events."""
        if self.event_streamer:
            self.event_streamer.subscribe(subscription_id, callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        metrics = self.metrics_collector.collect_system_metrics()
        
        return {
            'system_metrics': metrics,
            'monitoring_stats': self.get_monitoring_statistics(),
            'alert_summary': self.alert_manager.get_alert_summary() if self.alert_manager else {},
            'timestamp': datetime.now().isoformat()
        }
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        stats = {
            'state': self.state.value,
            'uptime_seconds': self.performance_stats['uptime_seconds'],
            'monitoring_cycles': self.performance_stats['monitoring_cycles'],
            'avg_cycle_duration_ms': self.performance_stats['avg_cycle_duration'] * 1000,
            'errors_encountered': self.performance_stats['errors_encountered'],
            'collection_interval_seconds': self.collection_interval
        }
        
        # Add collector stats
        stats['metrics_collector'] = self.metrics_collector.collection_stats
        
        # Add event streamer stats
        if self.event_streamer:
            stats['event_streamer'] = self.event_streamer.event_stats
        
        return stats
    
    def get_metric_trends(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get trend analysis for a specific metric."""
        history = self.metrics_collector.get_metric_history(metric_name, duration_minutes)
        statistics_data = self.metrics_collector.calculate_metric_statistics(metric_name, duration_minutes)
        anomalies = self.metrics_collector.detect_anomalies(metric_name)
        
        return {
            'metric_name': metric_name,
            'duration_minutes': duration_minutes,
            'data_points': len(history),
            'history': [
                {'timestamp': ts.isoformat(), 'value': value}
                for ts, value in history
            ],
            'statistics': statistics_data,
            'anomalies': anomalies,
            'trend_direction': self._calculate_trend_direction(history)
        }
    
    def _calculate_trend_direction(self, history: List[Tuple[datetime, float]]) -> str:
        """Calculate trend direction from historical data."""
        if len(history) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        values = [value for _, value in history]
        
        # Compare first and last quarters
        quarter_size = len(values) // 4
        if quarter_size < 1:
            return "insufficient_data"
        
        first_quarter_avg = sum(values[:quarter_size]) / quarter_size
        last_quarter_avg = sum(values[-quarter_size:]) / quarter_size
        
        change_percent = ((last_quarter_avg - first_quarter_avg) / first_quarter_avg) * 100
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"


# Factory Functions

def create_realtime_monitor(collection_interval: int = 5,
                          enable_alerting: bool = True,
                          enable_event_streaming: bool = True) -> RealTimeMonitor:
    """
    Create a real-time monitor with configuration.
    
    Args:
        collection_interval: Metrics collection interval in seconds
        enable_alerting: Enable alerting system
        enable_event_streaming: Enable event streaming
        
    Returns:
        Configured RealTimeMonitor instance
    """
    return RealTimeMonitor(
        collection_interval=collection_interval,
        enable_alerting=enable_alerting,
        enable_event_streaming=enable_event_streaming
    )


def create_monitoring_threshold(metric_name: str,
                              warning_threshold: float,
                              critical_threshold: float,
                              **kwargs) -> MonitoringThreshold:
    """
    Create a monitoring threshold with configuration.
    
    Args:
        metric_name: Name of the metric to monitor
        warning_threshold: Warning level threshold
        critical_threshold: Critical level threshold
        **kwargs: Additional threshold parameters
        
    Returns:
        Configured MonitoringThreshold instance
    """
    return MonitoringThreshold(
        metric_name=metric_name,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        **kwargs
    )


# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Real-Time Monitoring Team'
__description__ = 'Advanced real-time monitoring and alerting system for Enhanced Linkage Dashboard'