"""
Analytics Performance Monitor
=============================

Monitors the performance of the analytics system itself.
Tracks processing times, resource usage, and system efficiency.

Author: TestMaster Team
"""

import logging
import time
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)

class AnalyticsPerformanceMonitor:
    """
    Monitors the performance and efficiency of the analytics system.
    """
    
    def __init__(self, monitor_interval: int = 60, history_size: int = 1000):
        """
        Initialize the performance monitor.
        
        Args:
            monitor_interval: Monitoring interval in seconds
            history_size: Number of performance samples to keep
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        
        # Performance metrics
        self.processing_times = deque(maxlen=history_size)
        self.resource_usage = deque(maxlen=history_size)
        self.operation_metrics = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = datetime.now()
        
        # Performance thresholds
        self.thresholds = {
            'processing_time_warning': 5.0,  # seconds
            'processing_time_critical': 10.0,  # seconds
            'memory_usage_warning': 80.0,  # percent
            'memory_usage_critical': 90.0,  # percent
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.10   # 10%
        }
        
        logger.info("Analytics Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Analytics performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Analytics performance monitoring stopped")
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True, metadata: Dict = None):
        """
        Record a performance metric for an operation.
        
        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
            success: Whether the operation succeeded
            metadata: Additional metadata about the operation
        """
        timestamp = datetime.now()
        
        metric = {
            'operation': operation_name,
            'duration': duration,
            'success': success,
            'timestamp': timestamp.isoformat(),
            'metadata': metadata or {}
        }
        
        self.operation_metrics[operation_name].append(metric)
        
        # Track errors
        if not success:
            self.error_counts[operation_name] += 1
        
        # Detect performance issues
        self._check_performance_thresholds(operation_name, duration)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Returns:
            Performance summary with key metrics
        """
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        # Calculate operation statistics
        operation_stats = {}
        total_operations = 0
        total_errors = 0
        
        for operation_name, metrics in self.operation_metrics.items():
            if not metrics:
                continue
            
            durations = [m['duration'] for m in metrics]
            successes = [m['success'] for m in metrics]
            
            operation_stats[operation_name] = {
                'total_calls': len(metrics),
                'success_rate': sum(successes) / len(successes) * 100 if successes else 0,
                'avg_duration': statistics.mean(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'error_count': self.error_counts.get(operation_name, 0)
            }
            
            total_operations += len(metrics)
            total_errors += self.error_counts.get(operation_name, 0)
        
        # Calculate overall system health
        system_health = self._calculate_system_health()
        
        return {
            'uptime_seconds': uptime,
            'monitoring_active': self.monitoring_active,
            'total_operations': total_operations,
            'total_errors': total_errors,
            'overall_error_rate': (total_errors / total_operations * 100) if total_operations > 0 else 0,
            'system_health': system_health,
            'operation_statistics': operation_stats,
            'resource_usage': self._get_current_resource_usage(),
            'performance_alerts': self._get_active_alerts(),
            'timestamp': current_time.isoformat()
        }
    
    def get_operation_trends(self, operation_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trends for a specific operation.
        
        Args:
            operation_name: Name of the operation
            hours: Number of hours to analyze
            
        Returns:
            Trend analysis for the operation
        """
        if operation_name not in self.operation_metrics:
            return {'error': f'No data for operation: {operation_name}'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.operation_metrics[operation_name]
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': f'No recent data for operation: {operation_name}'}
        
        # Calculate trends
        durations = [m['duration'] for m in recent_metrics]
        successes = [m['success'] for m in recent_metrics]
        
        # Time-based analysis
        hourly_buckets = defaultdict(list)
        for metric in recent_metrics:
            hour = datetime.fromisoformat(metric['timestamp']).replace(minute=0, second=0, microsecond=0)
            hourly_buckets[hour].append(metric)
        
        hourly_stats = {}
        for hour, metrics in hourly_buckets.items():
            hour_durations = [m['duration'] for m in metrics]
            hour_successes = [m['success'] for m in metrics]
            
            hourly_stats[hour.isoformat()] = {
                'call_count': len(metrics),
                'avg_duration': statistics.mean(hour_durations),
                'success_rate': sum(hour_successes) / len(hour_successes) * 100
            }
        
        return {
            'operation_name': operation_name,
            'time_period_hours': hours,
            'total_calls': len(recent_metrics),
            'avg_duration': statistics.mean(durations),
            'duration_trend': self._calculate_trend([m['duration'] for m in recent_metrics[-20:]]),
            'success_rate': sum(successes) / len(successes) * 100,
            'hourly_breakdown': hourly_stats,
            'performance_percentiles': {
                '50th': statistics.median(durations),
                '90th': self._percentile(durations, 90),
                '95th': self._percentile(durations, 95),
                '99th': self._percentile(durations, 99)
            }
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """
        Analyze performance and suggest optimizations.
        
        Returns:
            Performance optimization recommendations
        """
        recommendations = []
        
        # Analyze slow operations
        slow_operations = []
        for operation_name, metrics in self.operation_metrics.items():
            if not metrics:
                continue
            
            recent_metrics = metrics[-50:]  # Last 50 operations
            avg_duration = statistics.mean([m['duration'] for m in recent_metrics])
            
            if avg_duration > self.thresholds['processing_time_warning']:
                slow_operations.append({
                    'operation': operation_name,
                    'avg_duration': avg_duration,
                    'call_count': len(recent_metrics)
                })
        
        if slow_operations:
            slow_operations.sort(key=lambda x: x['avg_duration'], reverse=True)
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'message': f"Slow operations detected: {', '.join([op['operation'] for op in slow_operations[:3]])}",
                'details': slow_operations[:5]
            })
        
        # Analyze error-prone operations
        error_prone = []
        for operation_name, error_count in self.error_counts.items():
            if operation_name in self.operation_metrics:
                total_calls = len(self.operation_metrics[operation_name])
                error_rate = error_count / total_calls if total_calls > 0 else 0
                
                if error_rate > self.thresholds['error_rate_warning']:
                    error_prone.append({
                        'operation': operation_name,
                        'error_rate': error_rate * 100,
                        'error_count': error_count,
                        'total_calls': total_calls
                    })
        
        if error_prone:
            error_prone.sort(key=lambda x: x['error_rate'], reverse=True)
            recommendations.append({
                'type': 'reliability',
                'priority': 'high',
                'message': f"High error rates detected in: {', '.join([op['operation'] for op in error_prone[:3]])}",
                'details': error_prone[:5]
            })
        
        # Resource usage recommendations
        resource_usage = self._get_current_resource_usage()
        if resource_usage['memory_percent'] > self.thresholds['memory_usage_warning']:
            recommendations.append({
                'type': 'resource',
                'priority': 'medium',
                'message': f"High memory usage: {resource_usage['memory_percent']:.1f}%",
                'details': {'memory_usage': resource_usage}
            })
        
        return {
            'recommendations': recommendations,
            'optimization_score': self._calculate_optimization_score(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system resource metrics
                resource_metrics = self._get_current_resource_usage()
                resource_metrics['timestamp'] = datetime.now().isoformat()
                self.resource_usage.append(resource_metrics)
                
                # Clean up old operation metrics
                self._cleanup_old_metrics()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                time.sleep(5)  # Back off on error
    
    def _get_current_resource_usage(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            # Get current process
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            # CPU usage (for this process)
            cpu_percent = process.cpu_percent()
            
            return {
                'memory_mb': memory_info.rss / (1024 * 1024),
                'memory_percent': (memory_info.rss / system_memory.total) * 100,
                'cpu_percent': cpu_percent,
                'thread_count': process.num_threads(),
                'file_descriptors': process.num_fds() if hasattr(process, 'num_fds') else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {}
    
    def _check_performance_thresholds(self, operation_name: str, duration: float):
        """Check if operation duration exceeds thresholds."""
        if duration > self.thresholds['processing_time_critical']:
            logger.warning(f"Critical performance issue: {operation_name} took {duration:.2f}s")
        elif duration > self.thresholds['processing_time_warning']:
            logger.info(f"Performance warning: {operation_name} took {duration:.2f}s")
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health score."""
        if not self.operation_metrics:
            return 'unknown'
        
        # Calculate error rates
        total_operations = sum(len(metrics) for metrics in self.operation_metrics.values())
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / total_operations if total_operations > 0 else 0
        
        # Calculate average response time
        all_durations = []
        for metrics in self.operation_metrics.values():
            all_durations.extend([m['duration'] for m in metrics[-20:]])  # Recent operations
        
        avg_duration = statistics.mean(all_durations) if all_durations else 0
        
        # Get resource usage
        resource_usage = self._get_current_resource_usage()
        memory_percent = resource_usage.get('memory_percent', 0)
        
        # Determine health
        if (error_rate > self.thresholds['error_rate_critical'] or 
            avg_duration > self.thresholds['processing_time_critical'] or
            memory_percent > self.thresholds['memory_usage_critical']):
            return 'critical'
        elif (error_rate > self.thresholds['error_rate_warning'] or 
              avg_duration > self.thresholds['processing_time_warning'] or
              memory_percent > self.thresholds['memory_usage_warning']):
            return 'warning'
        else:
            return 'healthy'
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active performance alerts."""
        alerts = []
        
        # Check recent operations for issues
        for operation_name, metrics in self.operation_metrics.items():
            if not metrics:
                continue
            
            recent_metrics = metrics[-10:]  # Last 10 operations
            
            # Check for slow operations
            slow_ops = [m for m in recent_metrics if m['duration'] > self.thresholds['processing_time_warning']]
            if slow_ops:
                alerts.append({
                    'type': 'performance',
                    'severity': 'warning',
                    'operation': operation_name,
                    'message': f"{len(slow_ops)} slow operations in last 10 calls",
                    'threshold': self.thresholds['processing_time_warning']
                })
            
            # Check for errors
            failed_ops = [m for m in recent_metrics if not m['success']]
            if failed_ops:
                alerts.append({
                    'type': 'reliability',
                    'severity': 'warning',
                    'operation': operation_name,
                    'message': f"{len(failed_ops)} failed operations in last 10 calls",
                    'error_rate': len(failed_ops) / len(recent_metrics) * 100
                })
        
        return alerts
    
    def _calculate_optimization_score(self) -> float:
        """Calculate overall optimization score (0-100)."""
        if not self.operation_metrics:
            return 100.0
        
        # Factor 1: Error rate (40% weight)
        total_operations = sum(len(metrics) for metrics in self.operation_metrics.values())
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / total_operations if total_operations > 0 else 0
        error_score = max(0, 100 - (error_rate * 1000))  # Penalize errors heavily
        
        # Factor 2: Performance (40% weight)
        all_durations = []
        for metrics in self.operation_metrics.values():
            all_durations.extend([m['duration'] for m in metrics[-20:]])
        
        avg_duration = statistics.mean(all_durations) if all_durations else 0
        performance_score = max(0, 100 - (avg_duration * 10))  # Penalize slow operations
        
        # Factor 3: Resource efficiency (20% weight)
        resource_usage = self._get_current_resource_usage()
        memory_score = max(0, 100 - resource_usage.get('memory_percent', 0))
        
        # Weighted average
        optimization_score = (error_score * 0.4 + performance_score * 0.4 + memory_score * 0.2)
        
        return min(100, max(0, optimization_score))
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 5:
            return 'stable'
        
        # Compare first half with second half
        mid = len(values) // 2
        first_half_avg = statistics.mean(values[:mid])
        second_half_avg = statistics.mean(values[mid:])
        
        change_percent = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
        
        if change_percent > 10:
            return 'increasing'
        elif change_percent < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _cleanup_old_metrics(self):
        """Clean up old operation metrics to prevent memory leaks."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for operation_name in list(self.operation_metrics.keys()):
            metrics = self.operation_metrics[operation_name]
            
            # Keep only recent metrics
            recent_metrics = [
                m for m in metrics
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            if recent_metrics:
                # Keep only the most recent 100 metrics per operation
                self.operation_metrics[operation_name] = recent_metrics[-100:]
            else:
                # Remove operation if no recent metrics
                del self.operation_metrics[operation_name]
                if operation_name in self.error_counts:
                    del self.error_counts[operation_name]