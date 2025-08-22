
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
Enhanced Analytics & Real-time Dashboard Data
=============================================

Comprehensive analytics system providing real-time insights, performance metrics,
and dashboard-ready data visualization support.

Author: TestMaster Team
"""

import time
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from flask import Blueprint, jsonify, request
from dataclasses import dataclass, asdict
import threading
import logging

@dataclass
class TimeSeriesPoint:
    """Single point in time series data."""
    timestamp: str
    value: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AnalyticsMetric:
    """Enhanced analytics metric with trend analysis."""
    name: str
    current_value: float
    previous_value: Optional[float]
    change_percentage: Optional[float]
    trend: str  # 'up', 'down', 'stable'
    time_series: List[TimeSeriesPoint]
    unit: Optional[str] = None
    category: str = 'general'
    
    def __post_init__(self):
        if self.previous_value is not None and self.previous_value != 0:
            self.change_percentage = ((self.current_value - self.previous_value) / self.previous_value) * 100
        else:
            self.change_percentage = 0
            
        # Determine trend
        if self.change_percentage is None:
            self.trend = 'stable'
        elif self.change_percentage > 5:
            self.trend = 'up'
        elif self.change_percentage < -5:
            self.trend = 'down'
        else:
            self.trend = 'stable'

class EnhancedAnalyticsEngine:
    """
    Enhanced analytics engine providing comprehensive insights and dashboard data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('EnhancedAnalytics')
        
        # Data storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.events_log: deque = deque(maxlen=5000)
        self.performance_data: Dict[str, Any] = {}
        
        # Real-time tracking
        self.active_sessions = {}
        self.system_metrics = {
            'requests_per_minute': deque(maxlen=60),
            'error_rate': deque(maxlen=100),
            'response_times': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100)
        }
        
        # Analytics categories
        self.metric_categories = {
            'system': ['cpu_usage', 'memory_usage', 'response_time', 'error_rate'],
            'agents': ['agent_executions', 'crew_executions', 'swarm_orchestrations'],
            'performance': ['avg_response_time', 'success_rate', 'throughput'],
            'business': ['tasks_completed', 'user_sessions', 'api_calls']
        }
        
        # Aggregation windows
        self.aggregation_windows = {
            '1m': 60,      # 1 minute
            '5m': 300,     # 5 minutes
            '15m': 900,    # 15 minutes
            '1h': 3600,    # 1 hour
            '6h': 21600,   # 6 hours
            '24h': 86400   # 24 hours
        }
        
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        self.logger.info("Enhanced Analytics Engine initialized")
        
    def record_metric(self, name: str, value: float, 
                     category: str = 'general',
                     metadata: Dict[str, Any] = None):
        """Record a metric with timestamp and metadata."""
        timestamp = datetime.now().isoformat()
        
        point = TimeSeriesPoint(
            timestamp=timestamp,
            value=value,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics_history[name].append(point)
            
        # Update real-time tracking for system metrics
        if name in ['response_time', 'error_rate', 'cpu_usage', 'memory_usage']:
            self.system_metrics[name].append(value)
            
        self.logger.debug(f"Recorded metric: {name}={value}")
        
    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record an analytics event."""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with self.lock:
            self.events_log.append(event)
            
        self.logger.debug(f"Recorded event: {event_type}")
        
    def get_metric_trend(self, name: str, window: str = '1h') -> AnalyticsMetric:
        """Get metric with trend analysis."""
        if name not in self.metrics_history:
            return None
            
        history = list(self.metrics_history[name])
        if not history:
            return None
            
        # Get current value
        current_value = history[-1].value
        
        # Get window duration
        window_seconds = self.aggregation_windows.get(window, 3600)
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        
        # Filter points within window
        window_points = [
            p for p in history 
            if datetime.fromisoformat(p.timestamp) >= cutoff_time
        ]
        
        if len(window_points) < 2:
            previous_value = None
        else:
            previous_value = window_points[0].value
            
        # Determine category
        category = 'general'
        for cat, metrics in self.metric_categories.items():
            if name in metrics:
                category = cat
                break
                
        return AnalyticsMetric(
            name=name,
            current_value=current_value,
            previous_value=previous_value,
            change_percentage=None,  # Will be calculated in __post_init__
            trend='stable',  # Will be calculated in __post_init__
            time_series=window_points,
            category=category
        )
        
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        uptime = time.time() - self.start_time
        
        # Calculate key metrics
        overview = {
            'uptime_seconds': uptime,
            'uptime_formatted': self._format_duration(uptime),
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'healthy'
        }
        
        # Get key system metrics
        key_metrics = ['response_time', 'error_rate', 'cpu_usage', 'memory_usage']
        
        for metric_name in key_metrics:
            if metric_name in self.system_metrics and self.system_metrics[metric_name]:
                values = list(self.system_metrics[metric_name])
                overview['metrics'][metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': statistics.mean(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'count': len(values)
                }
                
        # Determine overall system status
        if overview['metrics'].get('error_rate', {}).get('current', 0) > 10:
            overview['status'] = 'degraded'
        elif overview['metrics'].get('response_time', {}).get('current', 0) > 5:
            overview['status'] = 'slow'
            
        return overview
        
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get data optimized for performance dashboard."""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_overview(),
            'key_metrics': {},
            'charts_data': {},
            'alerts': []
        }
        
        # Get trending metrics
        trending_metrics = [
            'agent_executions', 'crew_executions', 'swarm_orchestrations',
            'response_time', 'success_rate', 'throughput'
        ]
        
        for metric_name in trending_metrics:
            trend_data = self.get_metric_trend(metric_name, '1h')
            if trend_data:
                dashboard_data['key_metrics'][metric_name] = asdict(trend_data)
                
        # Prepare chart data (last 60 points for real-time charts)
        for metric_name in ['response_time', 'error_rate', 'throughput']:
            if metric_name in self.metrics_history:
                recent_points = list(self.metrics_history[metric_name])[-60:]
                dashboard_data['charts_data'][metric_name] = [
                    {
                        'x': p.timestamp,
                        'y': p.value
                    } for p in recent_points
                ]
                
        # Generate alerts
        dashboard_data['alerts'] = self._generate_alerts()
        
        return dashboard_data
        
    def get_agent_analytics(self) -> Dict[str, Any]:
        """Get analytics specific to agent operations."""
        agent_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_agents': 0,
                'active_agents': 0,
                'total_executions': 0,
                'avg_execution_time': 0
            },
            'by_type': {},
            'performance_trends': {}
        }
        
        # Calculate summary from metrics
        for metric_name in ['agent_executions', 'crew_executions', 'swarm_orchestrations']:
            if metric_name in self.metrics_history:
                history = list(self.metrics_history[metric_name])
                if history:
                    total_value = sum(p.value for p in history)
                    agent_data['summary']['total_executions'] += total_value
                    
        # Get performance trends
        agent_metrics = ['agent_executions', 'crew_executions', 'swarm_orchestrations']
        for metric in agent_metrics:
            trend = self.get_metric_trend(metric, '6h')
            if trend:
                agent_data['performance_trends'][metric] = asdict(trend)
                
        return agent_data
        
    def get_real_time_metrics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get real-time metrics for live dashboard updates."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        real_time_data = {
            'timestamp': datetime.now().isoformat(),
            'window_minutes': window_minutes,
            'metrics': {},
            'events_count': 0,
            'active_operations': len(self.active_sessions)
        }
        
        # Get recent metrics
        for metric_name, history in self.metrics_history.items():
            recent_points = [
                p for p in history 
                if datetime.fromisoformat(p.timestamp) >= cutoff_time
            ]
            
            if recent_points:
                values = [p.value for p in recent_points]
                real_time_data['metrics'][metric_name] = {
                    'count': len(values),
                    'sum': sum(values),
                    'avg': statistics.mean(values),
                    'latest': values[-1],
                    'trend': 'up' if len(values) > 1 and values[-1] > values[0] else 'stable'
                }
                
        # Count recent events
        recent_events = [
            e for e in self.events_log
            if datetime.fromisoformat(e['timestamp']) >= cutoff_time
        ]
        real_time_data['events_count'] = len(recent_events)
        
        return real_time_data
        
    def get_historical_analysis(self, metric_name: str, 
                              hours: int = 24) -> Dict[str, Any]:
        """Get detailed historical analysis for a specific metric."""
        if metric_name not in self.metrics_history:
            return {'error': 'Metric not found'}
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = [
            p for p in self.metrics_history[metric_name]
            if datetime.fromisoformat(p.timestamp) >= cutoff_time
        ]
        
        if not history:
            return {'error': 'No data in time range'}
            
        values = [p.value for p in history]
        
        analysis = {
            'metric_name': metric_name,
            'time_range_hours': hours,
            'data_points': len(values),
            'statistics': {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values)
            },
            'percentiles': {
                'p50': self._percentile(values, 50),
                'p75': self._percentile(values, 75),
                'p90': self._percentile(values, 90),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99)
            },
            'time_series': [asdict(p) for p in history]
        }
        
        return analysis
        
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate system alerts based on metrics."""
        alerts = []
        
        # Check error rate
        if 'error_rate' in self.system_metrics and self.system_metrics['error_rate']:
            current_error_rate = self.system_metrics['error_rate'][-1]
            if current_error_rate > 5:
                alerts.append({
                    'type': 'warning',
                    'metric': 'error_rate',
                    'message': f'High error rate: {current_error_rate:.2f}%',
                    'threshold': 5,
                    'current_value': current_error_rate
                })
                
        # Check response time
        if 'response_time' in self.system_metrics and self.system_metrics['response_time']:
            current_response_time = self.system_metrics['response_time'][-1]
            if current_response_time > 2:
                alerts.append({
                    'type': 'warning',
                    'metric': 'response_time',
                    'message': f'Slow response time: {current_response_time:.2f}s',
                    'threshold': 2,
                    'current_value': current_response_time
                })
                
        return alerts
        
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"
            
    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = (p / 100) * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = math.floor(index)
            upper = math.ceil(index)
            weight = index - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

# Global analytics engine
analytics_engine = EnhancedAnalyticsEngine()

# Flask Blueprint for enhanced analytics API
enhanced_analytics_bp = Blueprint('enhanced_analytics', __name__)

@enhanced_analytics_bp.route('/overview', methods=['GET'])
def get_analytics_overview():
    """Get comprehensive analytics overview."""
    try:
        overview = analytics_engine.get_system_overview()
        return jsonify({
            'status': 'success',
            'data': overview
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_analytics_bp.route('/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard-optimized analytics data."""
    try:
        dashboard_data = analytics_engine.get_performance_dashboard()
        return jsonify({
            'status': 'success',
            'data': dashboard_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_analytics_bp.route('/real-time', methods=['GET'])
def get_realtime_metrics():
    """Get real-time metrics for live updates."""
    try:
        window_minutes = request.args.get('window', 5, type=int)
        real_time_data = analytics_engine.get_real_time_metrics(window_minutes)
        
        return jsonify({
            'status': 'success',
            'data': real_time_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_analytics_bp.route('/agents', methods=['GET'])
def get_agent_analytics():
    """Get agent-specific analytics."""
    try:
        agent_data = analytics_engine.get_agent_analytics()
        return jsonify({
            'status': 'success',
            'data': agent_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_analytics_bp.route('/metrics/<metric_name>/trend', methods=['GET'])
def get_metric_trend(metric_name):
    """Get trend analysis for a specific metric."""
    try:
        window = request.args.get('window', '1h')
        trend_data = analytics_engine.get_metric_trend(metric_name, window)
        
        if trend_data is None:
            return jsonify({
                'status': 'error',
                'error': 'Metric not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'data': asdict(trend_data)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_analytics_bp.route('/metrics/<metric_name>/history', methods=['GET'])
def get_metric_history(metric_name):
    """Get historical analysis for a specific metric."""
    try:
        hours = request.args.get('hours', 24, type=int)
        analysis = analytics_engine.get_historical_analysis(metric_name, hours)
        
        return jsonify({
            'status': 'success',
            'data': analysis
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_analytics_bp.route('/record', methods=['POST'])
def record_analytics_data():
    """Record custom analytics data."""
    try:
        data = request.get_json() or {}
        
        if 'metric' in data:
            # Record metric
            metric_name = data['metric']['name']
            value = data['metric']['value']
            category = data['metric'].get('category', 'general')
            metadata = data['metric'].get('metadata', {})
            
            analytics_engine.record_metric(metric_name, value, category, metadata)
            
        if 'event' in data:
            # Record event
            event_type = data['event']['type']
            event_data = data['event'].get('data', {})
            
            analytics_engine.record_event(event_type, event_data)
            
        return jsonify({
            'status': 'success',
            'message': 'Analytics data recorded'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@enhanced_analytics_bp.route('/health', methods=['GET'])
def enhanced_analytics_health():
    """Health check for enhanced analytics."""
    try:
        total_metrics = len(analytics_engine.metrics_history)
        total_events = len(analytics_engine.events_log)
        
        return jsonify({
            'status': 'healthy',
            'service': 'Enhanced Analytics',
            'metrics_tracked': total_metrics,
            'events_logged': total_events,
            'uptime': analytics_engine._format_duration(time.time() - analytics_engine.start_time),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Export key components
__all__ = [
    'EnhancedAnalyticsEngine',
    'AnalyticsMetric',
    'TimeSeriesPoint',
    'analytics_engine',
    'enhanced_analytics_bp'
]