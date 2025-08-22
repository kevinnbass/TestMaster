
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
TestMaster Enhanced Telemetry Integration
========================================

Advanced telemetry and monitoring system inspired by PraisonAI patterns.
Provides comprehensive tracking, OpenTelemetry integration, and custom metrics collection.

Author: TestMaster Team
"""

import os
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from flask import Blueprint, jsonify, request
from functools import wraps

# Enhanced telemetry components
@dataclass
class TelemetrySpan:
    """Represents a telemetry span for tracking operations."""
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    status: str = "active"
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def duration(self) -> Optional[float]:
        """Get span duration if completed."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
        
    def finish(self, status: str = "success", error: Optional[str] = None):
        """Finish the span."""
        self.end_time = time.time()
        self.status = status
        if error:
            self.error = error

@dataclass
class TelemetryMetric:
    """Represents a custom metric."""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

class EnhancedTelemetryCollector:
    """
    Enhanced telemetry collector with OpenTelemetry-inspired patterns.
    Provides comprehensive tracking for TestMaster operations.
    """
    
    def __init__(self, service_name: str = "testmaster", 
                 backend: str = "custom",
                 exporter: str = "console"):
        self.service_name = service_name
        self.backend = backend
        self.exporter = exporter
        self.enabled = True
        
        # Storage
        self.spans: Dict[str, TelemetrySpan] = {}
        self.metrics: List[TelemetryMetric] = []
        self.events: List[Dict[str, Any]] = []
        
        # Counters
        self.counters = {
            'agent_executions': 0,
            'task_completions': 0,
            'tool_calls': 0,
            'llm_calls': 0,
            'total_tokens': 0,
            'errors': 0,
            'crew_executions': 0,
            'swarm_orchestrations': 0,
            'api_requests': 0
        }
        
        # Performance tracking
        self.performance_data = {
            'average_response_time': 0,
            'total_response_time': 0,
            'min_response_time': float('inf'),
            'max_response_time': 0,
            'active_operations': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        self.logger = logging.getLogger('EnhancedTelemetry')
        
        self.logger.info(f"Enhanced Telemetry initialized: {service_name} ({backend}/{exporter})")
        
    def create_span(self, name: str, parent_id: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new telemetry span."""
        if not self.enabled:
            return "disabled"
            
        span_id = f"span_{int(time.time() * 1000000)}_{name}"
        
        span = TelemetrySpan(
            span_id=span_id,
            name=name,
            start_time=time.time(),
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.spans[span_id] = span
            self.performance_data['active_operations'] += 1
            
        self.logger.debug(f"Created span: {span_id} ({name})")
        return span_id
        
    def finish_span(self, span_id: str, status: str = "success", 
                   error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Finish a telemetry span."""
        if not self.enabled or span_id == "disabled":
            return
            
        with self.lock:
            if span_id in self.spans:
                span = self.spans[span_id]
                span.finish(status, error)
                
                if metadata:
                    span.metadata.update(metadata)
                
                # Update performance metrics
                if span.duration:
                    self.performance_data['total_response_time'] += span.duration
                    self.performance_data['min_response_time'] = min(
                        self.performance_data['min_response_time'], span.duration
                    )
                    self.performance_data['max_response_time'] = max(
                        self.performance_data['max_response_time'], span.duration
                    )
                    
                    # Calculate average
                    completed_spans = len([s for s in self.spans.values() if s.end_time is not None])
                    if completed_spans > 0:
                        self.performance_data['average_response_time'] = (
                            self.performance_data['total_response_time'] / completed_spans
                        )
                
                self.performance_data['active_operations'] -= 1
                
                if status == "error":
                    self.counters['errors'] += 1
                    
                self.logger.debug(f"Finished span: {span_id} ({status}, {span.duration:.3f}s)")
        
    @contextmanager
    def trace(self, name: str, **metadata) -> ContextManager[str]:
        """Context manager for tracing operations."""
        span_id = self.create_span(name, metadata=metadata)
        try:
            yield span_id
            self.finish_span(span_id, "success")
        except Exception as e:
            self.finish_span(span_id, "error", str(e))
            raise
            
    def trace_agent_execution(self, agent_name: str, **metadata) -> ContextManager[str]:
        """Context manager for tracing agent executions."""
        with self.lock:
            self.counters['agent_executions'] += 1
            
        return self.trace(f"agent_execution_{agent_name}", agent=agent_name, **metadata)
        
    def trace_crew_execution(self, crew_id: str, **metadata) -> ContextManager[str]:
        """Context manager for tracing crew executions."""
        with self.lock:
            self.counters['crew_executions'] += 1
            
        return self.trace(f"crew_execution_{crew_id}", crew_id=crew_id, **metadata)
        
    def trace_swarm_orchestration(self, swarm_id: str, **metadata) -> ContextManager[str]:
        """Context manager for tracing swarm orchestrations."""
        with self.lock:
            self.counters['swarm_orchestrations'] += 1
            
        return self.trace(f"swarm_orchestration_{swarm_id}", swarm_id=swarm_id, **metadata)
        
    def record_metric(self, name: str, value: Union[int, float], 
                     tags: Optional[Dict[str, str]] = None):
        """Record a custom metric."""
        if not self.enabled:
            return
            
        metric = TelemetryMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics.append(metric)
            
        self.logger.debug(f"Recorded metric: {name}={value}")
        
    def record_cost(self, cost: float, model: str = "unknown", 
                   tokens: int = 0, operation: str = "llm_call"):
        """Record cost and token usage."""
        if not self.enabled:
            return
            
        with self.lock:
            self.counters['llm_calls'] += 1
            self.counters['total_tokens'] += tokens
            
        self.record_metric("cost", cost, {"model": model, "operation": operation})
        if tokens > 0:
            self.record_metric("tokens", tokens, {"model": model})
            
    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record a custom event."""
        if not self.enabled:
            return
            
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        
        with self.lock:
            self.events.append(event)
            
        self.logger.debug(f"Recorded event: {event_type}")
        
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter."""
        if not self.enabled:
            return
            
        with self.lock:
            if counter_name in self.counters:
                self.counters[counter_name] += value
            else:
                self.counters[counter_name] = value
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get current telemetry metrics."""
        with self.lock:
            return {
                **self.counters,
                **self.performance_data,
                'total_spans': len(self.spans),
                'active_spans': len([s for s in self.spans.values() if s.end_time is None]),
                'custom_metrics_count': len(self.metrics),
                'events_count': len(self.events)
            }
            
    def get_spans(self, include_active: bool = True) -> List[Dict[str, Any]]:
        """Get all spans."""
        with self.lock:
            spans = []
            for span in self.spans.values():
                if not include_active and span.end_time is None:
                    continue
                spans.append(asdict(span))
            return spans
            
    def get_custom_metrics(self, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get custom metrics."""
        with self.lock:
            if metric_name:
                return [asdict(m) for m in self.metrics if m.name == metric_name]
            return [asdict(m) for m in self.metrics]
            
    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recorded events."""
        with self.lock:
            if event_type:
                return [e for e in self.events if e['type'] == event_type]
            return self.events.copy()
            
    def export_data(self, format: str = "json") -> str:
        """Export telemetry data."""
        data = {
            'service_name': self.service_name,
            'export_time': datetime.now().isoformat(),
            'metrics': self.get_metrics(),
            'spans': self.get_spans(),
            'custom_metrics': self.get_custom_metrics(),
            'events': self.get_events()
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "summary":
            return self._generate_summary(data)
        else:
            return str(data)
            
    def _generate_summary(self, data: Dict[str, Any]) -> str:
        """Generate a human-readable summary."""
        metrics = data['metrics']
        return f"""
TestMaster Telemetry Summary
===========================
Service: {self.service_name}
Export Time: {data['export_time']}

Operations:
- Agent Executions: {metrics['agent_executions']}
- Crew Executions: {metrics['crew_executions']}
- Swarm Orchestrations: {metrics['swarm_orchestrations']}
- Task Completions: {metrics['task_completions']}
- Tool Calls: {metrics['tool_calls']}
- LLM Calls: {metrics['llm_calls']}
- API Requests: {metrics['api_requests']}

Performance:
- Total Spans: {metrics['total_spans']}
- Active Operations: {metrics['active_operations']}
- Average Response Time: {metrics['average_response_time']:.3f}s
- Min Response Time: {metrics['min_response_time']:.3f}s
- Max Response Time: {metrics['max_response_time']:.3f}s
- Error Count: {metrics['errors']}

Usage:
- Total Tokens: {metrics['total_tokens']}
- Custom Metrics: {metrics['custom_metrics_count']}
- Events Recorded: {metrics['events_count']}
        """.strip()
        
    def reset(self):
        """Reset all telemetry data."""
        with self.lock:
            self.spans.clear()
            self.metrics.clear()
            self.events.clear()
            
            # Reset counters
            for key in self.counters:
                self.counters[key] = 0
                
            # Reset performance data
            self.performance_data = {
                'average_response_time': 0,
                'total_response_time': 0,
                'min_response_time': float('inf'),
                'max_response_time': 0,
                'active_operations': 0
            }
            
        self.logger.info("Telemetry data reset")
        
    def enable(self):
        """Enable telemetry collection."""
        self.enabled = True
        self.logger.info("Telemetry enabled")
        
    def disable(self):
        """Disable telemetry collection."""
        self.enabled = False
        self.logger.info("Telemetry disabled")


# Decorator for automatic telemetry
def trace_operation(operation_name: str = None, telemetry_collector: Optional[EnhancedTelemetryCollector] = None):
    """
    Decorator to automatically trace function executions.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        telemetry_collector: Telemetry collector to use (defaults to global)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get telemetry collector
            collector = telemetry_collector or enhanced_telemetry
            
            op_name = operation_name or func.__name__
            
            with collector.trace(op_name, function=func.__name__):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


# Global enhanced telemetry instance
enhanced_telemetry = EnhancedTelemetryCollector()

# Flask Blueprint for enhanced telemetry API
enhanced_telemetry_bp = Blueprint('enhanced_telemetry', __name__)

@enhanced_telemetry_bp.route('/metrics', methods=['GET'])
def get_enhanced_metrics():
    """Get comprehensive telemetry metrics."""
    try:
        metrics = enhanced_telemetry.get_metrics()
        
        return jsonify({
            'status': 'success',
            'service_name': enhanced_telemetry.service_name,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_telemetry_bp.route('/spans', methods=['GET'])
def get_telemetry_spans():
    """Get telemetry spans."""
    try:
        include_active = request.args.get('include_active', 'true').lower() == 'true'
        spans = enhanced_telemetry.get_spans(include_active=include_active)
        
        return jsonify({
            'status': 'success',
            'spans': spans,
            'total': len(spans)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_telemetry_bp.route('/custom-metrics', methods=['GET'])
def get_custom_telemetry_metrics():
    """Get custom metrics."""
    try:
        metric_name = request.args.get('metric_name')
        metrics = enhanced_telemetry.get_custom_metrics(metric_name)
        
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'total': len(metrics)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_telemetry_bp.route('/events', methods=['GET'])
def get_telemetry_events():
    """Get recorded events."""
    try:
        event_type = request.args.get('event_type')
        events = enhanced_telemetry.get_events(event_type)
        
        return jsonify({
            'status': 'success',
            'events': events,
            'total': len(events)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_telemetry_bp.route('/export', methods=['GET'])
def export_telemetry_data():
    """Export telemetry data."""
    try:
        format_type = request.args.get('format', 'json')
        data = enhanced_telemetry.export_data(format_type)
        
        if format_type == 'json':
            return jsonify({
                'status': 'success',
                'data': json.loads(data)
            })
        else:
            return data, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_telemetry_bp.route('/record-metric', methods=['POST'])
def record_custom_metric():
    """Record a custom metric."""
    try:
        data = request.get_json() or {}
        name = data.get('name')
        value = data.get('value')
        tags = data.get('tags', {})
        
        if not name or value is None:
            return jsonify({
                'status': 'error',
                'error': 'Missing required fields: name, value'
            }), 400
            
        enhanced_telemetry.record_metric(name, value, tags)
        
        return jsonify({
            'status': 'success',
            'message': f'Metric {name} recorded'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_telemetry_bp.route('/record-event', methods=['POST'])
def record_custom_event():
    """Record a custom event."""
    try:
        data = request.get_json() or {}
        event_type = data.get('type')
        event_data = data.get('data', {})
        
        if not event_type:
            return jsonify({
                'status': 'error',
                'error': 'Missing required field: type'
            }), 400
            
        enhanced_telemetry.record_event(event_type, event_data)
        
        return jsonify({
            'status': 'success',
            'message': f'Event {event_type} recorded'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_telemetry_bp.route('/control', methods=['POST'])
def control_telemetry():
    """Enable/disable telemetry or reset data."""
    try:
        data = request.get_json() or {}
        action = data.get('action')
        
        if action == 'enable':
            enhanced_telemetry.enable()
            return jsonify({
                'status': 'success',
                'message': 'Telemetry enabled'
            })
        elif action == 'disable':
            enhanced_telemetry.disable()
            return jsonify({
                'status': 'success',
                'message': 'Telemetry disabled'
            })
        elif action == 'reset':
            enhanced_telemetry.reset()
            return jsonify({
                'status': 'success',
                'message': 'Telemetry data reset'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Invalid action. Use: enable, disable, or reset'
            }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_telemetry_bp.route('/health', methods=['GET'])
def enhanced_telemetry_health():
    """Health check for enhanced telemetry."""
    try:
        health_data = {
            'status': 'healthy',
            'enabled': enhanced_telemetry.enabled,
            'service_name': enhanced_telemetry.service_name,
            'backend': enhanced_telemetry.backend,
            'timestamp': datetime.now().isoformat(),
            'active_operations': enhanced_telemetry.performance_data['active_operations'],
            'total_spans': len(enhanced_telemetry.spans)
        }
        
        return jsonify(health_data)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Export key components
__all__ = [
    'EnhancedTelemetryCollector',
    'TelemetrySpan',
    'TelemetryMetric',
    'enhanced_telemetry',
    'enhanced_telemetry_bp',
    'trace_operation'
]