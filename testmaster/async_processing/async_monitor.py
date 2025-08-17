"""
Async Monitor for TestMaster

Comprehensive monitoring system for asynchronous operations
with real-time tracking, performance analysis, and alerting.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
import uuid

from ..core.feature_flags import FeatureFlags
from ..core.shared_state import get_shared_state
from ..telemetry import get_telemetry_collector, get_performance_monitor

class TaskState(Enum):
    """Async task states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class AsyncTaskInfo:
    """Information about an async task."""
    task_id: str
    task_name: str
    component: str
    state: TaskState
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    thread_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    retry_count: int = 0

@dataclass
class ExecutionStats:
    """Execution statistics for async operations."""
    component: str
    total_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    timeout_tasks: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    success_rate: float = 100.0
    last_activity: Optional[datetime] = None

@dataclass
class PerformanceAlert:
    """Performance alert for async operations."""
    alert_id: str
    component: str
    alert_type: str  # "high_latency", "high_failure_rate", "resource_exhaustion"
    severity: str    # "info", "warning", "critical"
    message: str
    threshold: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None

class AsyncMonitor:
    """
    Comprehensive monitoring system for async operations.
    
    Features:
    - Real-time task tracking and state management
    - Performance metrics and statistics
    - Alert generation for performance issues
    - Integration with telemetry and monitoring systems
    """
    
    def __init__(self, max_tasks: int = 10000, monitoring_interval: float = 30.0):
        """
        Initialize async monitor.
        
        Args:
            max_tasks: Maximum tasks to keep in memory
            monitoring_interval: Monitoring cycle interval in seconds
        """
        self.enabled = FeatureFlags.is_enabled('layer2_monitoring', 'async_processing')
        
        if not self.enabled:
            return
        
        self.max_tasks = max_tasks
        self.monitoring_interval = monitoring_interval
        
        # Task tracking
        self.active_tasks: Dict[str, AsyncTaskInfo] = {}
        self.completed_tasks: deque = deque(maxlen=max_tasks)
        self.component_stats: Dict[str, ExecutionStats] = {}
        
        # Alerts
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_thresholds = {
            "high_latency_ms": 5000.0,     # 5 seconds
            "high_failure_rate": 10.0,     # 10%
            "high_pending_count": 100      # 100 pending tasks
        }
        
        # Monitoring
        self.lock = threading.RLock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.is_monitoring = False
        
        # Integrations
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        if FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system'):
            self.telemetry = get_telemetry_collector()
            self.performance_monitor = get_performance_monitor()
        else:
            self.telemetry = None
            self.performance_monitor = None
        
        # Statistics
        self.total_tasks_tracked = 0
        self.alerts_generated = 0
        
        print("Async monitor initialized")
        print(f"   Max tasks: {self.max_tasks}")
        print(f"   Monitoring interval: {self.monitoring_interval}s")
    
    def start_monitoring(self):
        """Start background monitoring."""
        if not self.enabled or self.is_monitoring:
            return
        
        def monitor_worker():
            self.is_monitoring = True
            
            while not self.shutdown_event.is_set():
                try:
                    # Update statistics
                    self._update_component_stats()
                    
                    # Check for performance issues
                    self._check_performance_alerts()
                    
                    # Cleanup old tasks
                    self._cleanup_old_tasks()
                    
                    # Send telemetry
                    self._send_monitoring_telemetry()
                    
                    # Wait for next cycle
                    if self.shutdown_event.wait(timeout=self.monitoring_interval):
                        break
                        
                except Exception as e:
                    print(f"Async monitoring error: {e}")
            
            self.is_monitoring = False
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        print("Async monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.enabled or not self.is_monitoring:
            return
        
        self.shutdown_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        print("Async monitoring stopped")
    
    def track_task_start(self, task_id: str, task_name: str, component: str,
                        metadata: Dict[str, Any] = None, parent_task_id: str = None) -> AsyncTaskInfo:
        """
        Start tracking an async task.
        
        Args:
            task_id: Unique task identifier
            task_name: Human-readable task name
            component: Component name
            metadata: Additional metadata
            parent_task_id: Parent task ID for hierarchical tracking
            
        Returns:
            AsyncTaskInfo object for the task
        """
        if not self.enabled:
            return None
        
        task_info = AsyncTaskInfo(
            task_id=task_id,
            task_name=task_name,
            component=component,
            state=TaskState.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {},
            thread_id=str(threading.get_ident()),
            parent_task_id=parent_task_id
        )
        
        with self.lock:
            self.active_tasks[task_id] = task_info
            self.total_tasks_tracked += 1
            
            # Initialize component stats if needed
            if component not in self.component_stats:
                self.component_stats[component] = ExecutionStats(component=component)
            
            # Update component stats
            stats = self.component_stats[component]
            stats.total_tasks += 1
            stats.pending_tasks += 1
            stats.last_activity = datetime.now()
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="async_task_started",
                component="async_monitor",
                operation="track_task_start",
                metadata={
                    "task_id": task_id,
                    "task_name": task_name,
                    "component": component,
                    "parent_task_id": parent_task_id
                }
            )
        
        return task_info
    
    def track_task_running(self, task_id: str):
        """Mark a task as running."""
        if not self.enabled:
            return
        
        with self.lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                task_info.state = TaskState.RUNNING
                task_info.started_at = datetime.now()
                
                # Update component stats
                stats = self.component_stats[task_info.component]
                stats.pending_tasks -= 1
                stats.running_tasks += 1
    
    def track_task_completion(self, task_id: str, success: bool = True, 
                            error_message: str = None, metadata: Dict[str, Any] = None):
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier
            success: Whether task completed successfully
            error_message: Error message if failed
            metadata: Additional completion metadata
        """
        if not self.enabled:
            return
        
        with self.lock:
            if task_id not in self.active_tasks:
                return
            
            task_info = self.active_tasks[task_id]
            task_info.completed_at = datetime.now()
            
            # Calculate duration
            if task_info.started_at:
                duration = (task_info.completed_at - task_info.started_at).total_seconds() * 1000
                task_info.duration_ms = duration
            
            # Set state
            if success:
                task_info.state = TaskState.COMPLETED
            else:
                task_info.state = TaskState.FAILED
                task_info.error_message = error_message
            
            # Update metadata
            if metadata:
                task_info.metadata.update(metadata)
            
            # Update component stats
            stats = self.component_stats[task_info.component]
            stats.running_tasks -= 1
            stats.last_activity = datetime.now()
            
            if success:
                stats.completed_tasks += 1
            else:
                stats.failed_tasks += 1
            
            # Update duration statistics
            if task_info.duration_ms is not None:
                stats.total_duration_ms += task_info.duration_ms
                stats.avg_duration_ms = stats.total_duration_ms / (stats.completed_tasks + stats.failed_tasks)
                stats.min_duration_ms = min(stats.min_duration_ms, task_info.duration_ms)
                stats.max_duration_ms = max(stats.max_duration_ms, task_info.duration_ms)
            
            # Update success rate
            total_completed = stats.completed_tasks + stats.failed_tasks
            if total_completed > 0:
                stats.success_rate = (stats.completed_tasks / total_completed) * 100
            
            # Move to completed tasks
            self.completed_tasks.append(task_info)
            del self.active_tasks[task_id]
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="async_task_completed",
                component="async_monitor",
                operation="track_task_completion",
                metadata={
                    "task_id": task_id,
                    "task_name": task_info.task_name,
                    "component": task_info.component,
                    "success": success
                },
                duration_ms=task_info.duration_ms,
                success=success,
                error_message=error_message
            )
        
        # Update shared state
        if self.shared_state:
            if success:
                self.shared_state.increment("async_tasks_completed")
            else:
                self.shared_state.increment("async_tasks_failed")
    
    def track_task_cancellation(self, task_id: str, reason: str = None):
        """Mark a task as cancelled."""
        if not self.enabled:
            return
        
        with self.lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                task_info.state = TaskState.CANCELLED
                task_info.completed_at = datetime.now()
                task_info.error_message = reason or "Task cancelled"
                
                # Update component stats
                stats = self.component_stats[task_info.component]
                if task_info.state == TaskState.PENDING:
                    stats.pending_tasks -= 1
                else:
                    stats.running_tasks -= 1
                stats.cancelled_tasks += 1
                
                # Move to completed tasks
                self.completed_tasks.append(task_info)
                del self.active_tasks[task_id]
    
    def track_task_timeout(self, task_id: str):
        """Mark a task as timed out."""
        if not self.enabled:
            return
        
        with self.lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                task_info.state = TaskState.TIMEOUT
                task_info.completed_at = datetime.now()
                task_info.error_message = "Task timed out"
                
                # Update component stats
                stats = self.component_stats[task_info.component]
                stats.running_tasks -= 1
                stats.timeout_tasks += 1
                
                # Move to completed tasks
                self.completed_tasks.append(task_info)
                del self.active_tasks[task_id]
    
    def get_task_info(self, task_id: str) -> Optional[AsyncTaskInfo]:
        """Get information about a specific task."""
        if not self.enabled:
            return None
        
        with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            
            # Check completed tasks
            for task_info in self.completed_tasks:
                if task_info.task_id == task_id:
                    return task_info
        
        return None
    
    def get_active_tasks(self, component: str = None) -> List[AsyncTaskInfo]:
        """Get list of active tasks."""
        if not self.enabled:
            return []
        
        with self.lock:
            tasks = list(self.active_tasks.values())
            
            if component:
                tasks = [task for task in tasks if task.component == component]
            
            return tasks
    
    def get_component_stats(self, component: str = None) -> Union[ExecutionStats, Dict[str, ExecutionStats]]:
        """Get execution statistics for components."""
        if not self.enabled:
            return {}
        
        with self.lock:
            if component:
                return self.component_stats.get(component, ExecutionStats(component=component))
            else:
                return dict(self.component_stats)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            total_active = len(self.active_tasks)
            total_completed = len(self.completed_tasks)
            
            # Aggregate stats
            total_pending = sum(stats.pending_tasks for stats in self.component_stats.values())
            total_running = sum(stats.running_tasks for stats in self.component_stats.values())
            total_failed = sum(stats.failed_tasks for stats in self.component_stats.values())
            total_success = sum(stats.completed_tasks for stats in self.component_stats.values())
            
            overall_success_rate = 0.0
            if total_success + total_failed > 0:
                overall_success_rate = (total_success / (total_success + total_failed)) * 100
            
            avg_duration = 0.0
            if self.component_stats:
                total_avg_duration = sum(stats.avg_duration_ms for stats in self.component_stats.values())
                avg_duration = total_avg_duration / len(self.component_stats)
            
            return {
                "enabled": True,
                "is_monitoring": self.is_monitoring,
                "total_tasks_tracked": self.total_tasks_tracked,
                "active_tasks": total_active,
                "completed_tasks": total_completed,
                "pending_tasks": total_pending,
                "running_tasks": total_running,
                "success_rate": round(overall_success_rate, 2),
                "avg_duration_ms": round(avg_duration, 2),
                "components_monitored": len(self.component_stats),
                "active_alerts": len(self.active_alerts),
                "alerts_generated": self.alerts_generated
            }
    
    def _update_component_stats(self):
        """Update component statistics."""
        current_time = datetime.now()
        
        with self.lock:
            for stats in self.component_stats.values():
                # Update active task counts
                component_active_tasks = [
                    task for task in self.active_tasks.values()
                    if task.component == stats.component
                ]
                
                stats.pending_tasks = sum(1 for task in component_active_tasks 
                                        if task.state == TaskState.PENDING)
                stats.running_tasks = sum(1 for task in component_active_tasks 
                                        if task.state == TaskState.RUNNING)
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts."""
        current_time = datetime.now()
        
        with self.lock:
            for component, stats in self.component_stats.items():
                # Check high latency
                if (stats.avg_duration_ms > self.alert_thresholds["high_latency_ms"] and
                    stats.completed_tasks > 10):  # Need sufficient data
                    
                    self._generate_alert(
                        component=component,
                        alert_type="high_latency",
                        severity="warning",
                        message=f"High average latency: {stats.avg_duration_ms:.1f}ms",
                        threshold=self.alert_thresholds["high_latency_ms"],
                        current_value=stats.avg_duration_ms
                    )
                
                # Check high failure rate
                if (stats.success_rate < (100 - self.alert_thresholds["high_failure_rate"]) and
                    stats.total_tasks > 10):
                    
                    self._generate_alert(
                        component=component,
                        alert_type="high_failure_rate",
                        severity="critical",
                        message=f"High failure rate: {100 - stats.success_rate:.1f}%",
                        threshold=self.alert_thresholds["high_failure_rate"],
                        current_value=100 - stats.success_rate
                    )
                
                # Check high pending count
                if stats.pending_tasks > self.alert_thresholds["high_pending_count"]:
                    self._generate_alert(
                        component=component,
                        alert_type="high_pending_count",
                        severity="warning",
                        message=f"High pending task count: {stats.pending_tasks}",
                        threshold=self.alert_thresholds["high_pending_count"],
                        current_value=stats.pending_tasks
                    )
    
    def _generate_alert(self, component: str, alert_type: str, severity: str,
                       message: str, threshold: float, current_value: float):
        """Generate a performance alert."""
        # Check if similar alert already exists
        existing_alert = next(
            (alert for alert in self.active_alerts
             if (alert.component == component and 
                 alert.alert_type == alert_type and 
                 alert.resolved_at is None)),
            None
        )
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.triggered_at = datetime.now()
            return
        
        # Create new alert
        alert = PerformanceAlert(
            alert_id=str(uuid.uuid4()),
            component=component,
            alert_type=alert_type,
            severity=severity,
            message=message,
            threshold=threshold,
            current_value=current_value,
            triggered_at=datetime.now()
        )
        
        self.active_alerts.append(alert)
        self.alerts_generated += 1
        
        print(f"Async performance alert: {message}")
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="async_performance_alert",
                component="async_monitor",
                operation="generate_alert",
                metadata={
                    "alert_id": alert.alert_id,
                    "component": component,
                    "alert_type": alert_type,
                    "severity": severity,
                    "threshold": threshold,
                    "current_value": current_value
                }
            )
    
    def _cleanup_old_tasks(self):
        """Clean up old completed tasks and resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self.lock:
            # Clean up old alerts
            self.active_alerts = [
                alert for alert in self.active_alerts
                if (alert.resolved_at is None or alert.resolved_at > cutoff_time)
            ]
    
    def _send_monitoring_telemetry(self):
        """Send monitoring telemetry."""
        if not self.telemetry:
            return
        
        summary = self.get_performance_summary()
        
        self.telemetry.record_event(
            event_type="async_monitoring_summary",
            component="async_monitor",
            operation="monitoring_cycle",
            metadata={
                "active_tasks": summary.get("active_tasks", 0),
                "success_rate": summary.get("success_rate", 0),
                "avg_duration_ms": summary.get("avg_duration_ms", 0),
                "components_monitored": summary.get("components_monitored", 0),
                "active_alerts": summary.get("active_alerts", 0)
            }
        )
    
    def shutdown(self):
        """Shutdown async monitor."""
        if not self.enabled:
            return
        
        self.stop_monitoring()
        
        with self.lock:
            total_tracked = self.total_tasks_tracked
            total_alerts = self.alerts_generated
            
            self.active_tasks.clear()
            self.completed_tasks.clear()
            self.component_stats.clear()
            self.active_alerts.clear()
        
        print(f"Async monitor shutdown - tracked {total_tracked} tasks, generated {total_alerts} alerts")

# Global instance
_async_monitor: Optional[AsyncMonitor] = None

def get_async_monitor() -> AsyncMonitor:
    """Get the global async monitor instance."""
    global _async_monitor
    if _async_monitor is None:
        _async_monitor = AsyncMonitor()
    return _async_monitor

# Convenience function
def track_async_execution(task_name: str, component: str, metadata: Dict[str, Any] = None):
    """
    Context manager for tracking async execution.
    
    Args:
        task_name: Task name
        component: Component name
        metadata: Additional metadata
    """
    from contextlib import contextmanager
    
    @contextmanager
    def tracking_context():
        monitor = get_async_monitor()
        if not monitor.enabled:
            yield None
            return
        
        task_id = str(uuid.uuid4())
        
        try:
            # Start tracking
            task_info = monitor.track_task_start(task_id, task_name, component, metadata)
            monitor.track_task_running(task_id)
            
            yield task_info
            
            # Task completed successfully
            monitor.track_task_completion(task_id, success=True)
            
        except Exception as e:
            # Task failed
            monitor.track_task_completion(task_id, success=False, error_message=str(e))
            raise
    
    return tracking_context()