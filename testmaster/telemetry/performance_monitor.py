"""
Advanced Performance Monitor for TestMaster

Comprehensive performance monitoring system inspired by PraisonAI
with detailed execution tracking, bottleneck detection, and analysis.
"""

import time
import threading
import functools
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from contextlib import contextmanager
import json
import statistics

from ..core.feature_flags import FeatureFlags
from ..core.shared_state import get_shared_state
from .telemetry_collector import get_telemetry_collector

@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    metric_id: str
    component: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    thread_id: str = ""

@dataclass
class ComponentStats:
    """Performance statistics for a component."""
    component_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    median_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    recent_durations: List[float] = field(default_factory=list)
    error_rate: float = 0.0
    throughput_ops_per_sec: float = 0.0
    last_operation: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0

class AdvancedPerformanceMonitor:
    """
    Advanced performance monitoring system for TestMaster.
    
    Features:
    - Detailed execution timing and resource usage
    - Component-level performance statistics
    - Bottleneck detection and analysis
    - Memory and CPU tracking
    - Performance trend analysis
    - Integration with telemetry system
    """
    
    def __init__(self, max_metrics: int = 50000, stats_window: int = 1000):
        """
        Initialize advanced performance monitor.
        
        Args:
            max_metrics: Maximum metrics to keep in memory
            stats_window: Window size for recent statistics
        """
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system')
        
        if not self.enabled:
            return
        
        self.max_metrics = max_metrics
        self.stats_window = stats_window
        
        # Data storage
        self.metrics: deque = deque(maxlen=max_metrics)
        self.component_stats: Dict[str, ComponentStats] = {}
        self.active_operations: Dict[str, PerformanceMetric] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Integrations
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        self.telemetry = get_telemetry_collector()
        
        # Statistics
        self.operations_tracked = 0
        self.bottlenecks_detected = 0
        self.performance_alerts = []
        
        # System monitoring
        self.system_monitor_enabled = self._check_system_monitoring()
        
        # Start background monitoring
        self._start_monitoring_thread()
        
        print("Advanced performance monitor initialized")
        print(f"   Metrics buffer: {self.max_metrics}")
        print(f"   System monitoring: {self.system_monitor_enabled}")
    
    def _check_system_monitoring(self) -> bool:
        """Check if system monitoring is available."""
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if not self.system_monitor_enabled:
            return None
        
        try:
            import psutil
            process = psutil.Process()
            return round(process.memory_info().rss / 1024 / 1024, 2)
        except:
            return None
    
    def _get_cpu_percent(self) -> Optional[float]:
        """Get current CPU usage percentage."""
        if not self.system_monitor_enabled:
            return None
        
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except:
            return None
    
    @contextmanager
    def track_operation(self, component: str, operation: str, 
                       metadata: Dict[str, Any] = None):
        """
        Context manager to track operation performance.
        
        Args:
            component: Component name
            operation: Operation name
            metadata: Additional metadata
        """
        if not self.enabled:
            yield
            return
        
        metric_id = f"{component}_{operation}_{int(time.time() * 1000)}"
        thread_id = str(threading.get_ident())
        
        # Create initial metric
        metric = PerformanceMetric(
            metric_id=metric_id,
            component=component,
            operation=operation,
            start_time=datetime.now(),
            metadata=metadata or {},
            memory_before_mb=self._get_memory_usage(),
            cpu_percent=self._get_cpu_percent(),
            thread_id=thread_id
        )
        
        with self.lock:
            self.active_operations[metric_id] = metric
        
        try:
            yield metric
            metric.success = True
        except Exception as e:
            metric.success = False
            metric.error_message = str(e)
            raise
        finally:
            # Complete the metric
            metric.end_time = datetime.now()
            metric.duration_ms = (metric.end_time - metric.start_time).total_seconds() * 1000
            metric.memory_after_mb = self._get_memory_usage()
            
            with self.lock:
                # Remove from active operations
                self.active_operations.pop(metric_id, None)
                
                # Add to metrics
                self.metrics.append(metric)
                self.operations_tracked += 1
                
                # Update component statistics
                self._update_component_stats(metric)
                
                # Send to telemetry
                self.telemetry.record_event(
                    event_type="performance_measurement",
                    component=component,
                    operation=operation,
                    metadata=metadata,
                    duration_ms=metric.duration_ms,
                    success=metric.success,
                    error_message=metric.error_message
                )
                
                # Update shared state
                if self.shared_state:
                    self.shared_state.increment("performance_operations_tracked")
                    if not metric.success:
                        self.shared_state.increment("performance_operations_failed")
    
    def _update_component_stats(self, metric: PerformanceMetric):
        """Update component statistics with new metric."""
        component = metric.component
        
        if component not in self.component_stats:
            self.component_stats[component] = ComponentStats(component_name=component)
        
        stats = self.component_stats[component]
        
        # Update counters
        stats.total_operations += 1
        if metric.success:
            stats.successful_operations += 1
        else:
            stats.failed_operations += 1
        
        # Update duration statistics
        if metric.duration_ms is not None:
            stats.total_duration_ms += metric.duration_ms
            stats.min_duration_ms = min(stats.min_duration_ms, metric.duration_ms)
            stats.max_duration_ms = max(stats.max_duration_ms, metric.duration_ms)
            
            # Add to recent durations (keep last N measurements)
            stats.recent_durations.append(metric.duration_ms)
            if len(stats.recent_durations) > self.stats_window:
                stats.recent_durations.pop(0)
            
            # Calculate statistics
            if stats.total_operations > 0:
                stats.avg_duration_ms = stats.total_duration_ms / stats.total_operations
                stats.error_rate = (stats.failed_operations / stats.total_operations) * 100
            
            if stats.recent_durations:
                stats.median_duration_ms = statistics.median(stats.recent_durations)
                if len(stats.recent_durations) >= 20:  # Need sufficient data for percentiles
                    sorted_durations = sorted(stats.recent_durations)
                    stats.p95_duration_ms = sorted_durations[int(len(sorted_durations) * 0.95)]
                    stats.p99_duration_ms = sorted_durations[int(len(sorted_durations) * 0.99)]
        
        # Update resource usage
        if metric.memory_after_mb:
            stats.memory_usage_mb = metric.memory_after_mb
        
        if metric.cpu_percent:
            stats.cpu_utilization = metric.cpu_percent
        
        # Update timestamps
        stats.last_operation = metric.end_time
        
        # Calculate throughput (operations per second over last minute)
        recent_ops = [m for m in self.metrics 
                     if m.component == component and 
                     m.end_time and 
                     (datetime.now() - m.end_time) < timedelta(minutes=1)]
        stats.throughput_ops_per_sec = len(recent_ops) / 60.0
    
    def get_component_stats(self, component: str = None) -> Union[ComponentStats, Dict[str, ComponentStats]]:
        """
        Get performance statistics for components.
        
        Args:
            component: Specific component name, or None for all
            
        Returns:
            ComponentStats for specific component or dict of all stats
        """
        if not self.enabled:
            return {} if component is None else ComponentStats(component_name=component or "unknown")
        
        with self.lock:
            if component:
                return self.component_stats.get(component, ComponentStats(component_name=component))
            else:
                return dict(self.component_stats)
    
    def get_bottlenecks(self, threshold_ms: float = 1000, min_operations: int = 10) -> List[Dict[str, Any]]:
        """
        Detect performance bottlenecks.
        
        Args:
            threshold_ms: Duration threshold for bottleneck detection
            min_operations: Minimum operations required for analysis
            
        Returns:
            List of detected bottlenecks
        """
        if not self.enabled:
            return []
        
        bottlenecks = []
        
        with self.lock:
            for component, stats in self.component_stats.items():
                if stats.total_operations < min_operations:
                    continue
                
                issues = []
                
                # Check average duration
                if stats.avg_duration_ms > threshold_ms:
                    issues.append(f"High average duration: {stats.avg_duration_ms:.1f}ms")
                
                # Check P95 duration
                if stats.p95_duration_ms > threshold_ms * 2:
                    issues.append(f"High P95 duration: {stats.p95_duration_ms:.1f}ms")
                
                # Check error rate
                if stats.error_rate > 5.0:  # 5% error rate threshold
                    issues.append(f"High error rate: {stats.error_rate:.1f}%")
                
                # Check memory usage (if available)
                if stats.memory_usage_mb > 1000:  # 1GB threshold
                    issues.append(f"High memory usage: {stats.memory_usage_mb:.1f}MB")
                
                if issues:
                    bottlenecks.append({
                        "component": component,
                        "severity": len(issues),
                        "issues": issues,
                        "stats": asdict(stats)
                    })
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: x["severity"], reverse=True)
        
        self.bottlenecks_detected = len(bottlenecks)
        return bottlenecks
    
    def get_performance_trends(self, component: str = None, 
                             timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            component: Specific component or None for all
            timeframe_hours: Analysis timeframe in hours
            
        Returns:
            Performance trends analysis
        """
        if not self.enabled:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        with self.lock:
            # Filter metrics by timeframe and component
            filtered_metrics = [
                m for m in self.metrics
                if m.end_time and m.end_time >= cutoff_time and
                (not component or m.component == component)
            ]
        
        if not filtered_metrics:
            return {"error": "No metrics found for the specified criteria"}
        
        # Group by hour for trend analysis
        hourly_stats = defaultdict(lambda: {"count": 0, "durations": [], "errors": 0})
        
        for metric in filtered_metrics:
            hour_key = metric.end_time.replace(minute=0, second=0, microsecond=0)
            hourly_stats[hour_key]["count"] += 1
            if metric.duration_ms:
                hourly_stats[hour_key]["durations"].append(metric.duration_ms)
            if not metric.success:
                hourly_stats[hour_key]["errors"] += 1
        
        # Calculate trends
        trends = []
        for hour, stats in sorted(hourly_stats.items()):
            avg_duration = statistics.mean(stats["durations"]) if stats["durations"] else 0
            error_rate = (stats["errors"] / stats["count"]) * 100 if stats["count"] > 0 else 0
            
            trends.append({
                "timestamp": hour.isoformat(),
                "operations": stats["count"],
                "avg_duration_ms": round(avg_duration, 2),
                "error_rate": round(error_rate, 2)
            })
        
        return {
            "component": component or "all",
            "timeframe_hours": timeframe_hours,
            "total_operations": len(filtered_metrics),
            "trends": trends
        }
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get currently active operations."""
        if not self.enabled:
            return []
        
        with self.lock:
            active = []
            current_time = datetime.now()
            
            for metric in self.active_operations.values():
                duration = (current_time - metric.start_time).total_seconds() * 1000
                active.append({
                    "metric_id": metric.metric_id,
                    "component": metric.component,
                    "operation": metric.operation,
                    "duration_ms": round(duration, 2),
                    "start_time": metric.start_time.isoformat(),
                    "thread_id": metric.thread_id
                })
        
        return active
    
    def export_metrics(self, format: str = "json", component: str = None) -> str:
        """Export performance metrics."""
        if not self.enabled:
            return "{}" if format == "json" else ""
        
        with self.lock:
            # Filter metrics
            metrics_to_export = list(self.metrics)
            if component:
                metrics_to_export = [m for m in metrics_to_export if m.component == component]
            
            # Convert to serializable format
            export_data = []
            for metric in metrics_to_export:
                metric_dict = asdict(metric)
                metric_dict['start_time'] = metric.start_time.isoformat()
                if metric.end_time:
                    metric_dict['end_time'] = metric.end_time.isoformat()
                export_data.append(metric_dict)
        
        if format == "json":
            return json.dumps({
                "export_timestamp": datetime.now().isoformat(),
                "total_metrics": len(export_data),
                "component_filter": component,
                "metrics": export_data
            }, indent=2)
        
        return str(export_data)
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        if not self.enabled:
            return
        
        with self.lock:
            self.metrics.clear()
            self.component_stats.clear()
            self.operations_tracked = 0
            self.bottlenecks_detected = 0
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread."""
        if not self.enabled:
            return
        
        def monitoring_worker():
            while not self.shutdown_event.is_set():
                try:
                    if self.shutdown_event.wait(timeout=30):  # Check every 30 seconds
                        break
                    
                    self._periodic_analysis()
                    
                except Exception:
                    # Handle errors silently
                    pass
        
        self.monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitor_thread.start()
    
    def _periodic_analysis(self):
        """Perform periodic performance analysis."""
        # Detect bottlenecks
        bottlenecks = self.get_bottlenecks()
        
        # Update shared state
        if self.shared_state:
            self.shared_state.set("performance_bottlenecks_detected", len(bottlenecks))
            self.shared_state.set("performance_operations_tracked", self.operations_tracked)
    
    def shutdown(self):
        """Shutdown performance monitor."""
        if not self.enabled:
            return
        
        self.shutdown_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        print(f"Performance monitor shutdown - tracked {self.operations_tracked} operations")

# Global instance
_performance_monitor: Optional[AdvancedPerformanceMonitor] = None

def get_performance_monitor() -> AdvancedPerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = AdvancedPerformanceMonitor()
    return _performance_monitor

# Convenience functions and decorators
def monitor_execution(component: str, operation: str = None):
    """Decorator to monitor function execution performance."""
    def decorator(func):
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            if not monitor.enabled:
                return func(*args, **kwargs)
            
            with monitor.track_operation(component, op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def track_operation(component: str, operation: str, metadata: Dict[str, Any] = None):
    """Context manager for tracking operations."""
    monitor = get_performance_monitor()
    return monitor.track_operation(component, operation, metadata)

# Alias for compatibility
monitor_performance = monitor_execution