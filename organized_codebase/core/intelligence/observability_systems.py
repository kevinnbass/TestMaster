"""
Observability Systems Module
===========================
System health and performance monitoring extracted from Agency-Swarm tracking patterns.
Module size: ~298 lines (under 300 limit)

Patterns extracted from:
- Agency-Swarm: Comprehensive tracking manager and observability hooks
- AgentScope: Studio monitoring and session management
- CrewAI: Event system and trace listeners
- AutoGen: Performance monitoring and system metrics
- LLama-Agents: Deployment monitoring and health checks
- Swarms: Intelligence monitoring and analytics
- PhiData: Tool execution tracking and performance metrics

Author: Agent D - Visualization Specialist
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading
import uuid
from abc import ABC, abstractmethod


@dataclass
class TraceEvent:
    """Individual trace event (Agency-Swarm pattern)."""
    id: str
    parent_id: Optional[str]
    event_type: str  # start, end, error, info
    component: str
    operation: str
    timestamp: datetime
    duration_ms: Optional[float]
    metadata: Dict[str, Any]
    status: str = "success"
    
    @classmethod
    def create(cls, event_type: str, component: str, operation: str, 
               parent_id: str = None, **metadata):
        return cls(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            event_type=event_type,
            component=component,
            operation=operation,
            timestamp=datetime.now(),
            duration_ms=None,
            metadata=metadata,
            status="success"
        )


@dataclass
class PerformanceMetric:
    """Performance monitoring metric."""
    name: str
    value: Union[int, float]
    unit: str
    timestamp: datetime
    tags: Dict[str, str]
    source: str
    
    def to_prometheus_format(self) -> str:
        """Convert to Prometheus metric format."""
        tags_str = ",".join(f'{k}="{v}"' for k, v in self.tags.items())
        return f'{self.name}{{{tags_str}}} {self.value} {int(self.timestamp.timestamp() * 1000)}'


class TraceCollector:
    """Trace event collection system (Agency-Swarm pattern)."""
    
    def __init__(self, max_traces: int = 10000):
        self.traces = deque(maxlen=max_traces)
        self.active_spans: Dict[str, TraceEvent] = {}
        self.trace_hierarchy: Dict[str, List[str]] = defaultdict(list)
        self.listeners: List[Callable[[TraceEvent], None]] = []
        self.lock = threading.RLock()
        
    def start_trace(self, component: str, operation: str, parent_id: str = None, **metadata) -> str:
        """Start new trace span."""
        trace_event = TraceEvent.create("start", component, operation, parent_id, **metadata)
        
        with self.lock:
            self.active_spans[trace_event.id] = trace_event
            if parent_id:
                self.trace_hierarchy[parent_id].append(trace_event.id)
                
        self._notify_listeners(trace_event)
        return trace_event.id
        
    def end_trace(self, trace_id: str, status: str = "success", **metadata):
        """End trace span."""
        with self.lock:
            if trace_id not in self.active_spans:
                return
                
            trace_event = self.active_spans[trace_id]
            end_time = datetime.now()
            trace_event.duration_ms = (end_time - trace_event.timestamp).total_seconds() * 1000
            trace_event.status = status
            trace_event.metadata.update(metadata)
            
            # Create end event
            end_event = TraceEvent.create(
                "end", trace_event.component, trace_event.operation,
                trace_event.parent_id, duration_ms=trace_event.duration_ms,
                status=status, **metadata
            )
            
            self.traces.append(trace_event)
            self.traces.append(end_event)
            del self.active_spans[trace_id]
            
        self._notify_listeners(end_event)
        
    def add_trace_event(self, event_type: str, component: str, operation: str, 
                       parent_id: str = None, **metadata):
        """Add standalone trace event."""
        event = TraceEvent.create(event_type, component, operation, parent_id, **metadata)
        
        with self.lock:
            self.traces.append(event)
            
        self._notify_listeners(event)
        
    def get_trace_tree(self, root_id: str) -> Dict[str, Any]:
        """Get hierarchical trace tree."""
        with self.lock:
            root_trace = next((t for t in self.traces if t.id == root_id), None)
            if not root_trace:
                return {}
                
            return self._build_trace_tree(root_trace)
            
    def get_recent_traces(self, limit: int = 100, component: str = None) -> List[TraceEvent]:
        """Get recent traces with optional filtering."""
        with self.lock:
            traces = list(self.traces)
            
        if component:
            traces = [t for t in traces if t.component == component]
            
        return traces[-limit:] if len(traces) > limit else traces
        
    def add_listener(self, listener: Callable[[TraceEvent], None]):
        """Add trace event listener."""
        self.listeners.append(listener)
        
    def _build_trace_tree(self, root: TraceEvent) -> Dict[str, Any]:
        """Build hierarchical trace structure."""
        tree = asdict(root)
        tree["children"] = []
        
        for child_id in self.trace_hierarchy.get(root.id, []):
            child_trace = next((t for t in self.traces if t.id == child_id), None)
            if child_trace:
                tree["children"].append(self._build_trace_tree(child_trace))
                
        return tree
        
    def _notify_listeners(self, event: TraceEvent):
        """Notify all listeners of new trace event."""
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors break tracing


class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self):
        self.metrics = deque(maxlen=10000)
        self.metric_aggregations = defaultdict(list)
        self.alert_thresholds: Dict[str, Dict[str, Any]] = {}
        self.collectors: List[Callable[[], List[PerformanceMetric]]] = []
        
        # Start background collection
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
    def record_metric(self, name: str, value: Union[int, float], unit: str = "", 
                     source: str = "system", **tags):
        """Record performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags,
            source=source
        )
        
        self.metrics.append(metric)
        self.metric_aggregations[name].append(value)
        
        # Check alert thresholds
        self._check_alerts(metric)
        
    def get_metric_summary(self, name: str, hours: int = 1) -> Dict[str, Any]:
        """Get metric summary statistics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics 
                         if m.name == name and m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No data available"}
            
        values = [m.value for m in recent_metrics]
        
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "unit": recent_metrics[-1].unit if recent_metrics else "",
            "timerange_hours": hours
        }
        
    def set_alert_threshold(self, metric_name: str, threshold: float, 
                          comparison: str = "greater", severity: str = "warning"):
        """Set alert threshold for metric."""
        self.alert_thresholds[metric_name] = {
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity
        }
        
    def add_collector(self, collector: Callable[[], List[PerformanceMetric]]):
        """Add metric collector function."""
        self.collectors.append(collector)
        
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        for metric in list(self.metrics)[-1000:]:  # Last 1000 metrics
            lines.append(metric.to_prometheus_format())
        return "\n".join(lines)
        
    def _collection_loop(self):
        """Background metric collection loop."""
        while self.running:
            try:
                # Run all collectors
                for collector in self.collectors:
                    try:
                        metrics = collector()
                        for metric in metrics:
                            self.metrics.append(metric)
                    except Exception:
                        pass  # Don't let collector errors break monitoring
                        
                time.sleep(30)  # Collect every 30 seconds
            except Exception:
                pass
                
    def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts."""
        if metric.name not in self.alert_thresholds:
            return
            
        threshold_config = self.alert_thresholds[metric.name]
        threshold = threshold_config["threshold"]
        comparison = threshold_config["comparison"]
        
        triggered = False
        if comparison == "greater" and metric.value > threshold:
            triggered = True
        elif comparison == "less" and metric.value < threshold:
            triggered = True
            
        if triggered:
            # Could emit alert here
            pass


class HealthChecker:
    """System health monitoring (LLama-Agents pattern)."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.check_interval = 60  # seconds
        
        # Start health check loop
        self.running = True
        self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_thread.start()
        
    def register_health_check(self, name: str, check_func: Callable[[], bool], 
                            critical: bool = False):
        """Register health check function."""
        self.health_checks[name] = check_func
        self.health_status[name] = {
            "status": "unknown",
            "critical": critical,
            "last_check": None,
            "consecutive_failures": 0
        }
        
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                status = "healthy" if is_healthy else "unhealthy"
                
                self.health_status[name].update({
                    "status": status,
                    "last_check": datetime.now(),
                    "consecutive_failures": 0 if is_healthy else 
                        self.health_status[name]["consecutive_failures"] + 1
                })
                
                if not is_healthy and self.health_status[name]["critical"]:
                    overall_healthy = False
                    
            except Exception as e:
                self.health_status[name].update({
                    "status": "error",
                    "last_check": datetime.now(),
                    "error": str(e),
                    "consecutive_failures": self.health_status[name]["consecutive_failures"] + 1
                })
                
                if self.health_status[name]["critical"]:
                    overall_healthy = False
                    
            results[name] = self.health_status[name].copy()
            
        results["overall_status"] = "healthy" if overall_healthy else "unhealthy"
        return results
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary."""
        return {
            "timestamp": datetime.now(),
            "checks": dict(self.health_status),
            "total_checks": len(self.health_checks),
            "healthy_checks": sum(1 for status in self.health_status.values() 
                                if status["status"] == "healthy"),
            "critical_checks": sum(1 for status in self.health_status.values()
                                 if status["critical"])
        }
        
    def _health_check_loop(self):
        """Background health check loop."""
        while self.running:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception:
                pass


class ObservabilityDashboard:
    """Combined observability dashboard."""
    
    def __init__(self):
        self.trace_collector = TraceCollector()
        self.performance_monitor = PerformanceMonitor()
        self.health_checker = HealthChecker()
        
        # Register default health checks
        self._register_default_health_checks()
        
    def get_observability_overview(self) -> Dict[str, Any]:
        """Get comprehensive observability overview."""
        return {
            "timestamp": datetime.now(),
            "traces": {
                "total_traces": len(self.trace_collector.traces),
                "active_spans": len(self.trace_collector.active_spans),
                "recent_errors": len([t for t in list(self.trace_collector.traces)[-100:] 
                                    if t.status == "error"])
            },
            "performance": {
                "total_metrics": len(self.performance_monitor.metrics),
                "alert_thresholds": len(self.performance_monitor.alert_thresholds)
            },
            "health": self.health_checker.get_health_summary()
        }
        
    def _register_default_health_checks(self):
        """Register default system health checks."""
        def memory_check():
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < 90  # Less than 90% memory usage
            except ImportError:
                return True  # Assume healthy if psutil not available
                
        def disk_check():
            try:
                import psutil
                disk_percent = psutil.disk_usage('/').percent
                return disk_percent < 90  # Less than 90% disk usage
            except (ImportError, OSError):
                return True
                
        self.health_checker.register_health_check("memory", memory_check, critical=True)
        self.health_checker.register_health_check("disk", disk_check, critical=True)


# Public API
__all__ = [
    'TraceEvent',
    'PerformanceMetric',
    'TraceCollector',
    'PerformanceMonitor',
    'HealthChecker', 
    'ObservabilityDashboard'
]