"""
Performance Profiler for Personal Analytics Dashboard
Real-time performance monitoring and optimization for Agent E

Author: Agent E - Latin Swarm
Created: 2025-08-23 22:10:00
Purpose: Monitor dashboard performance, API response times, and system health
"""

import time
import psutil
import threading
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    timestamp: datetime
    metric_type: str
    value: float
    component: str
    additional_data: Dict[str, Any] = None


class PerformanceProfiler:
    """
    Real-time performance monitoring for personal analytics dashboard.
    
    Monitors:
    - API response times
    - Dashboard render performance
    - System resource usage
    - Cache hit rates
    - Data processing times
    """
    
    def __init__(self, max_metrics_history: int = 1000):
        """
        Initialize performance profiler.
        
        Args:
            max_metrics_history: Maximum number of metrics to keep in memory
        """
        self.max_metrics_history = max_metrics_history
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.current_metrics: Dict[str, Any] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds (aligned with roadmap requirements)
        self.thresholds = {
            'api_response_time_ms': 200,  # p95 target
            'api_response_time_p99_ms': 600,  # p99 target
            'first_contentful_paint_ms': 2500,  # FCP target
            'cpu_usage_percent': 80,
            'memory_usage_percent': 80,
            'cache_hit_rate_percent': 70
        }
        
        # Component performance tracking
        self.component_timers: Dict[str, float] = {}
        
        logger.info("Performance Profiler initialized")
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """
        Start continuous performance monitoring.
        
        Args:
            interval_seconds: Monitoring interval
        """
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        timestamp = datetime.now()
        
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Add metrics
            self._add_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_type='system_cpu_percent',
                value=cpu_percent,
                component='system'
            ))
            
            self._add_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_type='system_memory_percent',
                value=memory.percent,
                component='system',
                additional_data={
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2)
                }
            ))
            
            # Update current metrics
            self.current_metrics.update({
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory.percent,
                'timestamp': timestamp.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def time_component(self, component_name: str):
        """
        Context manager for timing component operations.
        
        Usage:
            with profiler.time_component('dashboard_render'):
                # Code to time
                pass
        """
        return ComponentTimer(self, component_name)
    
    def record_api_response(self, endpoint: str, response_time_ms: float, 
                          status_code: int = 200, data_size_bytes: int = 0):
        """
        Record API response performance.
        
        Args:
            endpoint: API endpoint path
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            data_size_bytes: Response data size in bytes
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type='api_response_time_ms',
            value=response_time_ms,
            component=endpoint,
            additional_data={
                'status_code': status_code,
                'data_size_bytes': data_size_bytes,
                'data_size_kb': round(data_size_bytes / 1024, 2)
            }
        )
        
        self._add_metric(metric)
        
        # Update current metrics
        self.current_metrics[f'api_{endpoint.replace("/", "_")}_response_ms'] = response_time_ms
    
    def record_cache_performance(self, cache_name: str, hit_rate: float, 
                               total_requests: int, hits: int):
        """
        Record cache performance metrics.
        
        Args:
            cache_name: Name of the cache
            hit_rate: Cache hit rate percentage
            total_requests: Total cache requests
            hits: Cache hits
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type='cache_hit_rate_percent',
            value=hit_rate,
            component=cache_name,
            additional_data={
                'total_requests': total_requests,
                'hits': hits,
                'misses': total_requests - hits
            }
        )
        
        self._add_metric(metric)
        self.current_metrics[f'cache_{cache_name}_hit_rate'] = hit_rate
    
    def record_dashboard_render(self, render_time_ms: float, component_count: int = 0):
        """
        Record dashboard rendering performance.
        
        Args:
            render_time_ms: Total render time in milliseconds
            component_count: Number of components rendered
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type='dashboard_render_ms',
            value=render_time_ms,
            component='dashboard',
            additional_data={
                'component_count': component_count,
                'is_fast': render_time_ms < self.thresholds['first_contentful_paint_ms']
            }
        )
        
        self._add_metric(metric)
        self.current_metrics['dashboard_render_ms'] = render_time_ms
    
    def _add_metric(self, metric: PerformanceMetric):
        """Add metric to history and check thresholds."""
        self.metrics_history.append(metric)
        
        # Check if metric exceeds thresholds
        threshold_key = metric.metric_type
        if threshold_key in self.thresholds:
            threshold = self.thresholds[threshold_key]
            if metric.value > threshold:
                logger.warning(
                    f"Performance threshold exceeded: {metric.component} "
                    f"{metric.metric_type}={metric.value} > {threshold}"
                )
    
    def get_dashboard_performance_data(self) -> Dict[str, Any]:
        """
        Get performance data formatted for dashboard display.
        
        Returns:
            Formatted performance data for Gamma dashboard integration
        """
        now = datetime.now()
        last_5_min = now - timedelta(minutes=5)
        
        # Filter recent metrics
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= last_5_min
        ]
        
        # Calculate performance statistics
        api_times = [m.value for m in recent_metrics if m.metric_type == 'api_response_time_ms']
        cpu_values = [m.value for m in recent_metrics if m.metric_type == 'system_cpu_percent']
        memory_values = [m.value for m in recent_metrics if m.metric_type == 'system_memory_percent']
        
        # Performance summary
        performance_summary = {
            'api_performance': {
                'avg_response_ms': round(sum(api_times) / len(api_times), 2) if api_times else 0,
                'max_response_ms': round(max(api_times), 2) if api_times else 0,
                'min_response_ms': round(min(api_times), 2) if api_times else 0,
                'total_requests': len(api_times),
                'fast_requests': len([t for t in api_times if t < 200]),
                'slow_requests': len([t for t in api_times if t > 600])
            },
            'system_performance': {
                'avg_cpu_percent': round(sum(cpu_values) / len(cpu_values), 1) if cpu_values else 0,
                'max_cpu_percent': round(max(cpu_values), 1) if cpu_values else 0,
                'avg_memory_percent': round(sum(memory_values) / len(memory_values), 1) if memory_values else 0,
                'max_memory_percent': round(max(memory_values), 1) if memory_values else 0
            },
            'health_status': self._calculate_health_status(),
            'recommendations': self._generate_performance_recommendations()
        }
        
        return {
            'summary': performance_summary,
            'current_metrics': self.current_metrics.copy(),
            'charts': self._generate_performance_charts(recent_metrics),
            'alerts': self._check_performance_alerts(),
            'timestamp': now.isoformat()
        }
    
    def _calculate_health_status(self) -> str:
        """Calculate overall system health status."""
        current = self.current_metrics
        
        # Check critical metrics
        cpu = current.get('system_cpu_percent', 0)
        memory = current.get('system_memory_percent', 0)
        
        if cpu > 90 or memory > 90:
            return 'critical'
        elif cpu > 70 or memory > 70:
            return 'warning'
        else:
            return 'healthy'
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        current = self.current_metrics
        
        # CPU recommendations
        cpu = current.get('system_cpu_percent', 0)
        if cpu > 80:
            recommendations.append("High CPU usage detected - consider optimizing background processes")
        
        # Memory recommendations
        memory = current.get('system_memory_percent', 0)
        if memory > 80:
            recommendations.append("High memory usage - consider clearing caches or restarting services")
        
        # API performance recommendations
        api_times = [m.value for m in list(self.metrics_history)[-50:] if m.metric_type == 'api_response_time_ms']
        if api_times and sum(api_times) / len(api_times) > 200:
            recommendations.append("API response times above target - consider cache optimization")
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _generate_performance_charts(self, recent_metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Generate chart data for performance visualization."""
        
        # Time series data for the last 5 minutes
        timestamps = []
        cpu_data = []
        memory_data = []
        api_times = []
        
        # Group metrics by minute for cleaner visualization
        minute_groups = {}
        for metric in recent_metrics:
            minute_key = metric.timestamp.replace(second=0, microsecond=0)
            if minute_key not in minute_groups:
                minute_groups[minute_key] = {'cpu': [], 'memory': [], 'api': []}
            
            if metric.metric_type == 'system_cpu_percent':
                minute_groups[minute_key]['cpu'].append(metric.value)
            elif metric.metric_type == 'system_memory_percent':
                minute_groups[minute_key]['memory'].append(metric.value)
            elif metric.metric_type == 'api_response_time_ms':
                minute_groups[minute_key]['api'].append(metric.value)
        
        # Prepare chart data
        sorted_minutes = sorted(minute_groups.keys())
        for minute in sorted_minutes:
            timestamps.append(minute.strftime('%H:%M'))
            
            # Average values for each minute
            group = minute_groups[minute]
            cpu_data.append(round(sum(group['cpu']) / len(group['cpu']), 1) if group['cpu'] else 0)
            memory_data.append(round(sum(group['memory']) / len(group['memory']), 1) if group['memory'] else 0)
            api_times.append(round(sum(group['api']) / len(group['api']), 1) if group['api'] else 0)
        
        return {
            'system_performance_chart': {
                'type': 'line',
                'data': {
                    'labels': timestamps,
                    'datasets': [
                        {
                            'label': 'CPU %',
                            'data': cpu_data,
                            'borderColor': '#ff6b6b',
                            'backgroundColor': 'rgba(255, 107, 107, 0.1)'
                        },
                        {
                            'label': 'Memory %',
                            'data': memory_data,
                            'borderColor': '#4ecdc4',
                            'backgroundColor': 'rgba(78, 205, 196, 0.1)'
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'max': 100
                        }
                    }
                }
            },
            'api_performance_chart': {
                'type': 'line',
                'data': {
                    'labels': timestamps,
                    'datasets': [{
                        'label': 'Response Time (ms)',
                        'data': api_times,
                        'borderColor': '#45b7d1',
                        'backgroundColor': 'rgba(69, 183, 209, 0.1)',
                        'borderDash': []
                    }]
                },
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'plugins': {
                        'legend': {
                            'display': True
                        }
                    }
                }
            }
        }
    
    def _check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts based on thresholds."""
        alerts = []
        current = self.current_metrics
        
        # Check each threshold
        for metric_key, threshold in self.thresholds.items():
            current_value = current.get(metric_key.replace('_ms', '').replace('_percent', ''), 0)
            
            if current_value > threshold:
                severity = 'critical' if current_value > threshold * 1.2 else 'warning'
                alerts.append({
                    'metric': metric_key,
                    'current_value': current_value,
                    'threshold': threshold,
                    'severity': severity,
                    'message': f"{metric_key.replace('_', ' ').title()} is {current_value} (threshold: {threshold})"
                })
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a concise performance summary for logging/debugging."""
        return {
            'metrics_collected': len(self.metrics_history),
            'monitoring_active': self.monitoring_active,
            'current_health': self._calculate_health_status(),
            'last_update': self.current_metrics.get('timestamp', 'never')
        }


class ComponentTimer:
    """Context manager for timing component operations."""
    
    def __init__(self, profiler: PerformanceProfiler, component_name: str):
        self.profiler = profiler
        self.component_name = component_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type='component_duration_ms',
                value=duration_ms,
                component=self.component_name,
                additional_data={
                    'success': exc_type is None,
                    'error_type': exc_type.__name__ if exc_type else None
                }
            )
            
            self.profiler._add_metric(metric)
            self.profiler.current_metrics[f'component_{self.component_name}_ms'] = duration_ms


def create_performance_profiler() -> PerformanceProfiler:
    """Factory function to create a configured performance profiler."""
    profiler = PerformanceProfiler()
    
    # Start monitoring automatically
    profiler.start_monitoring()
    
    logger.info("Performance profiler created and monitoring started")
    return profiler


if __name__ == '__main__':
    # Demo usage
    profiler = create_performance_profiler()
    
    # Simulate some metrics
    profiler.record_api_response('/api/personal-analytics/overview', 45.3, 200, 1024)
    profiler.record_cache_performance('panel_data', 85.5, 100, 85)
    profiler.record_dashboard_render(1250.0, 8)
    
    time.sleep(2)
    
    # Get dashboard data
    dashboard_data = profiler.get_dashboard_performance_data()
    print("Performance Dashboard Data:")
    print(json.dumps(dashboard_data, indent=2, default=str))
    
    profiler.stop_monitoring()