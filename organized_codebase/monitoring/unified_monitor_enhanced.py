"""
Enhanced Unified Monitor
=======================
Comprehensive monitoring system with full implementation.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class Alert:
    """Monitoring alert."""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    threshold: float
    actual_value: float
    tags: Dict[str, str] = field(default_factory=dict)

class Metric:
    """Enhanced metric with full implementation."""
    
    def __init__(self, name: str, metric_type: MetricType, max_points: int = 1000):
        self.name = name
        self.type = metric_type
        self.points = deque(maxlen=max_points)
        self.tags = {}
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # Statistics
        self._count = 0
        self._sum = 0.0
        self._min = float('inf')
        self._max = float('-inf')
    
    def record(self, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        
        self.points.append(point)
        self.last_updated = datetime.now()
        
        # Update statistics
        self._count += 1
        self._sum += value
        self._min = min(self._min, value)
        self._max = max(self._max, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metric statistics."""
        if self._count == 0:
            return {
                "count": 0,
                "sum": 0,
                "min": None,
                "max": None,
                "avg": None
            }
        
        return {
            "count": self._count,
            "sum": self._sum,
            "min": self._min,
            "max": self._max,
            "avg": self._sum / self._count,
            "latest": self.points[-1].value if self.points else None,
            "last_updated": self.last_updated.isoformat()
        }
    
    def get_recent_values(self, duration_minutes: int = 10) -> List[float]:
        """Get values from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        return [p.value for p in self.points if p.timestamp >= cutoff]

class EnhancedUnifiedMonitor:
    """Enhanced unified monitoring system with full implementation."""
    
    def __init__(self):
        self.enabled = True
        self.metrics = {}  # name -> Metric
        self.alerts = deque(maxlen=1000)
        self.alert_rules = {}  # name -> {threshold, comparator, level}
        self.subscribers = []  # Callback functions
        
        # Background monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.operation_count = 0
        
        logger.info("Enhanced Unified Monitor initialized")
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if not self.enabled:
            return
        
        # Create metric if it doesn't exist
        if name not in self.metrics:
            self.metrics[name] = Metric(name, metric_type)
        
        # Record the value
        self.metrics[name].record(value, tags)
        self.operation_count += 1
        
        # Check alert rules
        self._check_alerts(name, value)
    
    def increment_counter(self, name: str, amount: float = 1.0, 
                         tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        current_value = 0
        if name in self.metrics and self.metrics[name].points:
            current_value = self.metrics[name].points[-1].value
        
        self.record_metric(name, current_value + amount, MetricType.COUNTER, tags)
    
    def time_operation(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        class TimerContext:
            def __init__(self, monitor, metric_name, metric_tags):
                self.monitor = monitor
                self.metric_name = metric_name
                self.metric_tags = metric_tags
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.monitor.record_metric(
                    self.metric_name, 
                    duration * 1000,  # Convert to milliseconds
                    MetricType.TIMER,
                    self.metric_tags
                )
        
        return TimerContext(self, name, tags)
    
    def add_alert_rule(self, metric_name: str, threshold: float, 
                      comparator: str = "greater_than", level: AlertLevel = AlertLevel.WARNING):
        """Add an alert rule for a metric."""
        self.alert_rules[metric_name] = {
            "threshold": threshold,
            "comparator": comparator,
            "level": level
        }
        
        logger.info(f"Added alert rule for {metric_name}: {comparator} {threshold}")
    
    def _check_alerts(self, metric_name: str, value: float):
        """Check if value triggers any alerts."""
        if metric_name not in self.alert_rules:
            return
        
        rule = self.alert_rules[metric_name]
        threshold = rule["threshold"]
        comparator = rule["comparator"]
        
        triggered = False
        
        if comparator == "greater_than" and value > threshold:
            triggered = True
        elif comparator == "less_than" and value < threshold:
            triggered = True
        elif comparator == "equals" and value == threshold:
            triggered = True
        
        if triggered:
            alert = Alert(
                id=str(uuid.uuid4()),
                level=rule["level"],
                message=f"Metric {metric_name} triggered alert: {value} {comparator} {threshold}",
                timestamp=datetime.now(),
                metric_name=metric_name,
                threshold=threshold,
                actual_value=value
            )
            
            self.alerts.append(alert)
            self._notify_subscribers(alert)
            
            logger.warning(f"Alert triggered: {alert.message}")
    
    def subscribe_to_alerts(self, callback: Callable[[Alert], None]):
        """Subscribe to alert notifications."""
        self.subscribers.append(callback)
    
    def _notify_subscribers(self, alert: Alert):
        """Notify all subscribers of an alert."""
        for callback in self.subscribers:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metric data."""
        if name not in self.metrics:
            return None
        
        metric = self.metrics[name]
        return {
            "name": name,
            "type": metric.type.value,
            "stats": metric.get_stats(),
            "recent_values": metric.get_recent_values(),
            "created_at": metric.created_at.isoformat(),
            "point_count": len(metric.points)
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics data."""
        return {name: self.get_metric(name) for name in self.metrics.keys()}
    
    def get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        recent_alerts = [
            {
                "id": alert.id,
                "level": alert.level.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "threshold": alert.threshold,
                "actual_value": alert.actual_value
            }
            for alert in self.alerts
            if alert.timestamp >= cutoff
        ]
        
        return sorted(recent_alerts, key=lambda x: x["timestamp"], reverse=True)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        total_metrics = len(self.metrics)
        total_alerts = len(self.alerts)
        recent_alerts = len(self.get_recent_alerts(60))
        
        # Determine health status
        if recent_alerts == 0:
            health_status = "healthy"
        elif recent_alerts < 5:
            health_status = "warning"
        else:
            health_status = "critical"
        
        uptime = datetime.now() - self.start_time
        
        return {
            "status": health_status,
            "uptime_seconds": uptime.total_seconds(),
            "total_metrics": total_metrics,
            "total_alerts": total_alerts,
            "recent_alerts": recent_alerts,
            "operations_count": self.operation_count,
            "operations_per_second": self.operation_count / max(uptime.total_seconds(), 1),
            "timestamp": datetime.now().isoformat()
        }
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Record system metrics
                self.record_metric("system.monitor.metrics_count", len(self.metrics))
                self.record_metric("system.monitor.alerts_count", len(self.alerts))
                self.record_metric("system.monitor.operations_count", self.operation_count)
                
                # Cleanup old data
                self._cleanup_old_data()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(30)
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory growth."""
        cutoff = datetime.now() - timedelta(hours=24)
        
        # Clean old alerts
        self.alerts = deque(
            (alert for alert in self.alerts if alert.timestamp >= cutoff),
            maxlen=1000
        )
    
    def shutdown(self):
        """Shutdown the monitor."""
        self.running = False
        logger.info("Enhanced Unified Monitor shutdown")
    
    # Legacy compatibility methods
    def start_monitoring(self, component: str = "system"):
        """Start monitoring (compatibility method)."""
        self.record_metric(f"{component}.monitoring.started", 1)
    
    def record_event(self, event_type: str, data: Dict[str, Any] = None):
        """Record an event (compatibility method)."""
        self.increment_counter(f"events.{event_type}")
        if data:
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    self.record_metric(f"events.{event_type}.{key}", value)

# Global monitor instance
monitor = EnhancedUnifiedMonitor()

# Legacy compatibility functions
def get_monitor():
    """Get the global monitor instance."""
    return monitor

def record_metric(name: str, value: float, tags: Dict[str, str] = None):
    """Record a metric."""
    monitor.record_metric(name, value, tags=tags)

def get_health_status():
    """Get system health."""
    return monitor.get_system_health()
