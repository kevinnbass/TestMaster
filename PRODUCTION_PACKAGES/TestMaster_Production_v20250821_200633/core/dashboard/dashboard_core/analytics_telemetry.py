"""
Analytics Telemetry and Observability System
===========================================

Comprehensive telemetry collection, distributed tracing, and observability
for the analytics system with OpenTelemetry integration and metrics export.

Author: TestMaster Team
"""

import logging
import time
import threading
import queue
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import socket
import psutil

logger = logging.getLogger(__name__)

class TelemetryLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class TraceType(Enum):
    REQUEST = "request"
    OPERATION = "operation"
    COMPONENT = "component"
    SYSTEM = "system"
    USER = "user"

@dataclass
class TelemetryEvent:
    """Structured telemetry event."""
    event_id: str
    timestamp: datetime
    level: TelemetryLevel
    component: str
    operation: str
    message: str
    duration_ms: Optional[float] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class Span:
    """Distributed tracing span."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    component: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "active"
    attributes: Dict[str, Any] = None
    events: List[Dict[str, Any]] = None
    links: List[Dict[str, Any]] = None

@dataclass
class MetricPoint:
    """Time-series metric point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    unit: str = ""
    metric_type: str = "gauge"  # gauge, counter, histogram, summary

class AnalyticsTelemetryCollector:
    """
    Comprehensive telemetry and observability system for analytics.
    """
    
    def __init__(self, service_name: str = "analytics_system", 
                 export_interval: int = 30,
                 max_events: int = 10000,
                 enable_tracing: bool = True,
                 enable_metrics: bool = True):
        """
        Initialize analytics telemetry collector.
        
        Args:
            service_name: Name of the service
            export_interval: Interval for exporting telemetry data
            max_events: Maximum events to keep in memory
            enable_tracing: Enable distributed tracing
            enable_metrics: Enable metrics collection
        """
        self.service_name = service_name
        self.export_interval = export_interval
        self.max_events = max_events
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        
        # Service metadata
        self.service_metadata = {
            'service_name': service_name,
            'service_version': '1.0.0',
            'instance_id': str(uuid.uuid4()),
            'hostname': socket.gethostname(),
            'start_time': datetime.now().isoformat()
        }
        
        # Telemetry storage
        self.events = deque(maxlen=max_events)
        self.active_spans = {}
        self.completed_spans = deque(maxlen=max_events)
        self.metrics = defaultdict(list)
        
        # Export destinations
        self.event_exporters = []
        self.span_exporters = []
        self.metric_exporters = []
        
        # Threading
        self.telemetry_active = False
        self.export_thread = None
        self.collection_thread = None
        
        # Statistics
        self.telemetry_stats = {
            'events_collected': 0,
            'spans_created': 0,
            'spans_completed': 0,
            'metrics_collected': 0,
            'export_operations': 0,
            'export_failures': 0,
            'start_time': datetime.now()
        }
        
        # Sampling configuration
        self.sampling_rate = 1.0  # Sample all by default
        self.error_sampling_rate = 1.0  # Always sample errors
        
        # Context tracking
        self.current_context = threading.local()
        
        # Built-in exporters
        self._setup_default_exporters()
        
        logger.info(f"Analytics Telemetry Collector initialized: {service_name}")
    
    def start_collection(self):
        """Start telemetry collection and export."""
        if self.telemetry_active:
            return
        
        self.telemetry_active = True
        
        # Start export thread
        self.export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        
        self.export_thread.start()
        self.collection_thread.start()
        
        logger.info("Analytics telemetry collection started")
    
    def stop_collection(self):
        """Stop telemetry collection."""
        self.telemetry_active = False
        
        # Wait for threads to finish
        for thread in [self.export_thread, self.collection_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        # Final export
        self._export_telemetry_data()
        
        logger.info("Analytics telemetry collection stopped")
    
    def record_event(self, level: TelemetryLevel, component: str, operation: str,
                    message: str, duration_ms: Optional[float] = None,
                    attributes: Dict[str, Any] = None, metrics: Dict[str, float] = None,
                    error: Optional[str] = None) -> str:
        """
        Record a telemetry event.
        
        Args:
            level: Severity level
            component: Component name
            operation: Operation name
            message: Event message
            duration_ms: Operation duration in milliseconds
            attributes: Additional attributes
            metrics: Associated metrics
            error: Error information
        
        Returns:
            Event ID
        """
        # Apply sampling
        if not self._should_sample(level):
            return None
        
        event_id = str(uuid.uuid4())
        
        # Get current trace context
        trace_id = getattr(self.current_context, 'trace_id', None)
        span_id = getattr(self.current_context, 'span_id', None)
        
        event = TelemetryEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            level=level,
            component=component,
            operation=operation,
            message=message,
            duration_ms=duration_ms,
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {},
            metrics=metrics or {},
            error=error
        )
        
        self.events.append(event)
        self.telemetry_stats['events_collected'] += 1
        
        # Add to current span if active
        if span_id and span_id in self.active_spans:
            span = self.active_spans[span_id]
            if span.events is None:
                span.events = []
            span.events.append({
                'timestamp': event.timestamp.isoformat(),
                'level': level.value,
                'message': message,
                'attributes': attributes or {}
            })
        
        return event_id
    
    def start_span(self, operation_name: str, component: str,
                   parent_span_id: Optional[str] = None,
                   attributes: Dict[str, Any] = None) -> str:
        """
        Start a new distributed tracing span.
        
        Args:
            operation_name: Name of the operation
            component: Component performing the operation
            parent_span_id: ID of parent span
            attributes: Initial span attributes
        
        Returns:
            Span ID
        """
        if not self.enable_tracing:
            return None
        
        span_id = str(uuid.uuid4())
        
        # Get or create trace ID
        trace_id = getattr(self.current_context, 'trace_id', None)
        if not trace_id:
            trace_id = str(uuid.uuid4())
            self.current_context.trace_id = trace_id
        
        # Use current span as parent if not specified
        if not parent_span_id:
            parent_span_id = getattr(self.current_context, 'span_id', None)
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            component=component,
            start_time=datetime.now(),
            attributes=attributes or {},
            events=[],
            links=[]
        )
        
        self.active_spans[span_id] = span
        self.current_context.span_id = span_id
        self.telemetry_stats['spans_created'] += 1
        
        return span_id
    
    def end_span(self, span_id: str, status: str = "ok",
                attributes: Dict[str, Any] = None):
        """
        End a distributed tracing span.
        
        Args:
            span_id: ID of span to end
            status: Final status (ok, error, timeout, cancelled)
            attributes: Final attributes to add
        """
        if not self.enable_tracing or span_id not in self.active_spans:
            return
        
        span = self.active_spans.pop(span_id)
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        if attributes:
            span.attributes.update(attributes)
        
        self.completed_spans.append(span)
        self.telemetry_stats['spans_completed'] += 1
        
        # Update context to parent span
        if span.parent_span_id and span.parent_span_id in self.active_spans:
            self.current_context.span_id = span.parent_span_id
        else:
            if hasattr(self.current_context, 'span_id'):
                delattr(self.current_context, 'span_id')
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None,
                     unit: str = "", metric_type: str = "gauge"):
        """
        Record a metric point.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Metric labels
            unit: Unit of measurement
            metric_type: Type of metric
        """
        if not self.enable_metrics:
            return
        
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit,
            metric_type=metric_type
        )
        
        self.metrics[name].append(metric_point)
        self.telemetry_stats['metrics_collected'] += 1
        
        # Keep only recent metrics to prevent memory issues
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-500:]
    
    def add_span_link(self, span_id: str, linked_trace_id: str, linked_span_id: str,
                     attributes: Dict[str, Any] = None):
        """Add a link to another span."""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.links.append({
                'trace_id': linked_trace_id,
                'span_id': linked_span_id,
                'attributes': attributes or {}
            })
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        return getattr(self.current_context, 'trace_id', None)
    
    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID."""
        return getattr(self.current_context, 'span_id', None)
    
    def add_event_exporter(self, exporter: Callable[[List[TelemetryEvent]], None]):
        """Add an event exporter function."""
        self.event_exporters.append(exporter)
    
    def add_span_exporter(self, exporter: Callable[[List[Span]], None]):
        """Add a span exporter function."""
        self.span_exporters.append(exporter)
    
    def add_metric_exporter(self, exporter: Callable[[Dict[str, List[MetricPoint]]], None]):
        """Add a metric exporter function."""
        self.metric_exporters.append(exporter)
    
    def _should_sample(self, level: TelemetryLevel) -> bool:
        """Determine if event should be sampled."""
        if level in [TelemetryLevel.ERROR, TelemetryLevel.CRITICAL]:
            return True  # Always sample errors
        
        import random
        return random.random() < self.sampling_rate
    
    def _collection_loop(self):
        """Background collection loop for system metrics."""
        while self.telemetry_active:
            try:
                time.sleep(10)  # Collect system metrics every 10 seconds
                
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                self.record_metric('system.cpu.usage_percent', cpu_percent, 
                                 {'service': self.service_name})
                self.record_metric('system.memory.usage_percent', memory.percent,
                                 {'service': self.service_name})
                self.record_metric('system.memory.used_bytes', memory.used,
                                 {'service': self.service_name}, unit="bytes")
                
                # Collect telemetry system metrics
                self.record_metric('telemetry.events.collected', 
                                 self.telemetry_stats['events_collected'],
                                 {'service': self.service_name}, metric_type="counter")
                self.record_metric('telemetry.spans.active', len(self.active_spans),
                                 {'service': self.service_name})
                self.record_metric('telemetry.events.buffer_size', len(self.events),
                                 {'service': self.service_name})
                
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
    
    def _export_loop(self):
        """Background export loop."""
        while self.telemetry_active:
            try:
                time.sleep(self.export_interval)
                self._export_telemetry_data()
            except Exception as e:
                logger.error(f"Export loop error: {e}")
                self.telemetry_stats['export_failures'] += 1
    
    def _export_telemetry_data(self):
        """Export collected telemetry data."""
        try:
            # Export events
            if self.events and self.event_exporters:
                events_to_export = list(self.events)
                for exporter in self.event_exporters:
                    try:
                        exporter(events_to_export)
                    except Exception as e:
                        logger.error(f"Event export failed: {e}")
            
            # Export completed spans
            if self.completed_spans and self.span_exporters:
                spans_to_export = list(self.completed_spans)
                for exporter in self.span_exporters:
                    try:
                        exporter(spans_to_export)
                    except Exception as e:
                        logger.error(f"Span export failed: {e}")
            
            # Export metrics
            if self.metrics and self.metric_exporters:
                metrics_to_export = dict(self.metrics)
                for exporter in self.metric_exporters:
                    try:
                        exporter(metrics_to_export)
                    except Exception as e:
                        logger.error(f"Metric export failed: {e}")
            
            self.telemetry_stats['export_operations'] += 1
            
        except Exception as e:
            logger.error(f"Telemetry export failed: {e}")
            self.telemetry_stats['export_failures'] += 1
    
    def _setup_default_exporters(self):
        """Setup default file-based exporters."""
        export_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'telemetry')
        os.makedirs(export_dir, exist_ok=True)
        
        def json_event_exporter(events: List[TelemetryEvent]):
            """Export events to JSON file."""
            if not events:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"events_{timestamp}.json"
            filepath = os.path.join(export_dir, filename)
            
            event_data = []
            for event in events:
                event_dict = asdict(event)
                event_dict['timestamp'] = event.timestamp.isoformat()
                # Convert enum to string for JSON serialization
                if 'level' in event_dict and hasattr(event_dict['level'], 'value'):
                    event_dict['level'] = event_dict['level'].value
                event_data.append(event_dict)
            
            with open(filepath, 'w') as f:
                json.dump({
                    'service_metadata': self.service_metadata,
                    'events': event_data
                }, f, indent=2)
        
        def json_span_exporter(spans: List[Span]):
            """Export spans to JSON file."""
            if not spans:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spans_{timestamp}.json"
            filepath = os.path.join(export_dir, filename)
            
            span_data = []
            for span in spans:
                span_dict = asdict(span)
                span_dict['start_time'] = span.start_time.isoformat()
                if span.end_time:
                    span_dict['end_time'] = span.end_time.isoformat()
                span_data.append(span_dict)
            
            with open(filepath, 'w') as f:
                json.dump({
                    'service_metadata': self.service_metadata,
                    'spans': span_data
                }, f, indent=2)
        
        def prometheus_metric_exporter(metrics: Dict[str, List[MetricPoint]]):
            """Export metrics in Prometheus format."""
            if not metrics:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.prom"
            filepath = os.path.join(export_dir, filename)
            
            with open(filepath, 'w') as f:
                for metric_name, points in metrics.items():
                    if not points:
                        continue
                    
                    # Get latest point for each label combination
                    latest_points = {}
                    for point in points:
                        label_key = str(sorted(point.labels.items()) if point.labels else [])
                        if label_key not in latest_points or point.timestamp > latest_points[label_key].timestamp:
                            latest_points[label_key] = point
                    
                    # Write metric
                    for point in latest_points.values():
                        labels_str = ""
                        if point.labels:
                            label_pairs = [f'{k}="{v}"' for k, v in point.labels.items()]
                            labels_str = "{" + ",".join(label_pairs) + "}"
                        
                        metric_line = f"{metric_name.replace('.', '_')}{labels_str} {point.value}"
                        f.write(metric_line + "\n")
        
        # Add default exporters
        self.add_event_exporter(json_event_exporter)
        self.add_span_exporter(json_span_exporter)
        self.add_metric_exporter(prometheus_metric_exporter)
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get telemetry system summary."""
        uptime = (datetime.now() - self.telemetry_stats['start_time']).total_seconds()
        
        # Calculate recent activity
        recent_events = [e for e in self.events 
                        if (datetime.now() - e.timestamp).total_seconds() < 300]  # Last 5 minutes
        recent_spans = [s for s in self.completed_spans 
                       if s.end_time and (datetime.now() - s.end_time).total_seconds() < 300]
        
        # Error rate calculation
        error_events = [e for e in recent_events 
                       if e.level in [TelemetryLevel.ERROR, TelemetryLevel.CRITICAL]]
        error_rate = len(error_events) / len(recent_events) if recent_events else 0
        
        return {
            'service_metadata': self.service_metadata,
            'collection_status': {
                'active': self.telemetry_active,
                'uptime_seconds': uptime,
                'sampling_rate': self.sampling_rate
            },
            'statistics': self.telemetry_stats.copy(),
            'current_state': {
                'events_in_buffer': len(self.events),
                'active_spans': len(self.active_spans),
                'completed_spans': len(self.completed_spans),
                'metrics_tracked': len(self.metrics)
            },
            'recent_activity': {
                'events_last_5min': len(recent_events),
                'spans_completed_last_5min': len(recent_spans),
                'error_rate': error_rate,
                'avg_span_duration_ms': sum(s.duration_ms for s in recent_spans if s.duration_ms) / len(recent_spans) if recent_spans else 0
            },
            'exporters': {
                'event_exporters': len(self.event_exporters),
                'span_exporters': len(self.span_exporters),
                'metric_exporters': len(self.metric_exporters)
            }
        }
    
    def get_recent_events(self, limit: int = 100, level: Optional[TelemetryLevel] = None) -> List[TelemetryEvent]:
        """Get recent events, optionally filtered by level."""
        events = list(self.events)
        
        if level:
            events = [e for e in events if e.level == level]
        
        # Sort by timestamp and return most recent
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_trace_spans(self, trace_id: str) -> List[Span]:
        """Get all spans for a specific trace."""
        spans = []
        
        # Check active spans
        for span in self.active_spans.values():
            if span.trace_id == trace_id:
                spans.append(span)
        
        # Check completed spans
        for span in self.completed_spans:
            if span.trace_id == trace_id:
                spans.append(span)
        
        # Sort by start time
        spans.sort(key=lambda s: s.start_time)
        return spans
    
    def get_metric_values(self, metric_name: str, hours: int = 24) -> List[MetricPoint]:
        """Get metric values for the specified time period."""
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [point for point in self.metrics[metric_name] 
                if point.timestamp >= cutoff_time]
    
    def shutdown(self):
        """Shutdown telemetry system."""
        self.stop_collection()
        
        # Close any active spans
        for span_id in list(self.active_spans.keys()):
            self.end_span(span_id, status="cancelled")
        
        logger.info("Analytics Telemetry Collector shutdown")

# Decorator for automatic span creation
def traced_operation(operation_name: str, component: str = "unknown"):
    """Decorator to automatically trace an operation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to get telemetry collector from first argument if it's an object with one
            telemetry = None
            if args and hasattr(args[0], 'telemetry_collector'):
                telemetry = args[0].telemetry_collector
            
            if telemetry and telemetry.enable_tracing:
                span_id = telemetry.start_span(operation_name, component)
                try:
                    result = func(*args, **kwargs)
                    telemetry.end_span(span_id, "ok")
                    return result
                except Exception as e:
                    telemetry.end_span(span_id, "error", {"error": str(e)})
                    telemetry.record_event(
                        TelemetryLevel.ERROR, component, operation_name,
                        f"Operation failed: {str(e)}", error=str(e)
                    )
                    raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator