"""
Enterprise Performance Monitor
=============================

Comprehensive performance monitoring system providing real-time metrics collection,
trend analysis, optimization recommendations, and system health tracking across
all intelligence components.

Features:
- Real-time performance metrics collection
- Operation-specific timing and success tracking
- Resource usage monitoring (CPU, memory, threads)
- Performance trend analysis and percentile calculations
- Intelligent optimization recommendations
- Alert generation for performance issues
- Historical data management with cleanup

Author: TestMaster Intelligence Team
"""

import logging
import time
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
import statistics
import json

logger = logging.getLogger(__name__)

class PerformanceAlert:
    """Performance alert definition"""
    
    def __init__(self, alert_type: str, severity: str, operation: str, 
                 message: str, threshold: float, current_value: float, metadata: Dict = None):
        self.alert_type = alert_type
        self.severity = severity
        self.operation = operation
        self.message = message
        self.threshold = threshold
        self.current_value = current_value
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.alert_id = f"alert_{int(time.time() * 1000000)}"

class EnterprisePerformanceMonitor:
    """
    Enterprise-grade performance monitoring system providing comprehensive
    metrics collection, analysis, and optimization recommendations.
    """
    
    def __init__(self, monitor_interval: int = 60, history_size: int = 1000,
                 enable_detailed_tracking: bool = True):
        """
        Initialize the enterprise performance monitor.
        
        Args:
            monitor_interval: Monitoring interval in seconds
            history_size: Number of performance samples to keep
            enable_detailed_tracking: Enable detailed operation tracking
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        self.enable_detailed_tracking = enable_detailed_tracking
        
        # Performance metrics storage
        self.processing_times = deque(maxlen=history_size)
        self.resource_usage = deque(maxlen=history_size)
        self.operation_metrics = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        
        # Active alerts
        self.active_alerts = []
        self.alert_history = deque(maxlen=history_size)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = datetime.now()
        
        # Performance thresholds (configurable)
        self.thresholds = {
            'processing_time_warning': 5.0,      # seconds
            'processing_time_critical': 10.0,    # seconds
            'memory_usage_warning': 80.0,        # percent
            'memory_usage_critical': 90.0,       # percent
            'cpu_usage_warning': 80.0,           # percent
            'cpu_usage_critical': 95.0,          # percent
            'error_rate_warning': 0.05,          # 5%
            'error_rate_critical': 0.10,         # 10%
            'thread_count_warning': 50,          # threads
            'thread_count_critical': 100,        # threads
            'response_time_p95_warning': 2.0,    # seconds
            'response_time_p95_critical': 5.0,   # seconds
        }
        
        # Performance optimization recommendations
        self.optimization_rules = {
            'slow_operations': {
                'threshold': 3.0,
                'message': "Operations taking longer than {threshold}s detected",
                'recommendation': "Consider optimizing algorithm or adding caching"
            },
            'high_error_rate': {
                'threshold': 0.05,
                'message': "Error rate above {threshold}% detected",
                'recommendation': "Review error patterns and implement better error handling"
            },
            'memory_leak': {
                'threshold': 0.1,  # 10% increase per hour
                'message': "Potential memory leak detected",
                'recommendation': "Review memory allocation and implement proper cleanup"
            },
            'cpu_saturation': {
                'threshold': 0.9,
                'message': "CPU usage consistently above {threshold}%",
                'recommendation': "Consider load balancing or algorithmic optimization"
            }
        }
        
        # System baseline metrics (established during initial monitoring)
        self.baseline_metrics = {
            'avg_processing_time': 0.0,
            'avg_memory_usage': 0.0,
            'avg_cpu_usage': 0.0,
            'baseline_established': False
        }
        
        logger.info("Enterprise Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start comprehensive performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Enterprise performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Enterprise performance monitoring stopped")
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True, 
                        metadata: Dict = None, component: str = "unknown"):
        """
        Record a performance metric for an operation with enhanced tracking.
        
        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
            success: Whether the operation succeeded
            metadata: Additional metadata about the operation
            component: Component that performed the operation
        """
        timestamp = datetime.now()
        
        metric = {
            'operation': operation_name,
            'component': component,
            'duration': duration,
            'success': success,
            'timestamp': timestamp.isoformat(),
            'metadata': metadata or {},
            'thread_id': threading.get_ident(),
            'process_memory_mb': self._get_current_memory_usage()
        }
        
        # Store in operation-specific metrics
        self.operation_metrics[operation_name].append(metric)
        
        # Update success/error counts
        if success:
            self.success_counts[operation_name] += 1
        else:
            self.error_counts[operation_name] += 1
        
        # Check for performance issues immediately
        self._check_operation_performance(operation_name, duration, success)
        
        # Detailed tracking for debugging
        if self.enable_detailed_tracking:
            self._track_detailed_metrics(operation_name, metric)
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary with enterprise-level detail.
        
        Returns:
            Detailed performance summary with all key metrics
        """
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        # Calculate operation statistics
        operation_stats = {}
        total_operations = 0
        total_errors = 0
        total_successes = 0
        
        for operation_name, metrics in self.operation_metrics.items():
            if not metrics:
                continue
            
            durations = [m['duration'] for m in metrics]
            successes = [m['success'] for m in metrics]
            components = [m.get('component', 'unknown') for m in metrics]
            
            operation_stats[operation_name] = {
                'total_calls': len(metrics),
                'success_count': sum(successes),
                'error_count': len(successes) - sum(successes),
                'success_rate': sum(successes) / len(successes) * 100 if successes else 0,
                'avg_duration': statistics.mean(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'median_duration': statistics.median(durations) if durations else 0,
                'p95_duration': self._percentile(durations, 95) if durations else 0,
                'p99_duration': self._percentile(durations, 99) if durations else 0,
                'components_involved': list(set(components)),
                'trend': self._calculate_trend([m['duration'] for m in metrics[-20:]]),
                'last_execution': max(m['timestamp'] for m in metrics) if metrics else None
            }
            
            total_operations += len(metrics)
            total_errors += len(successes) - sum(successes)
            total_successes += sum(successes)
        
        # Calculate system health score
        system_health = self._calculate_comprehensive_health()
        
        # Get current resource usage
        current_resources = self._get_comprehensive_resource_usage()
        
        # Calculate performance trends
        performance_trends = self._calculate_performance_trends()
        
        return {
            'monitoring_status': {
                'active': self.monitoring_active,
                'uptime_seconds': uptime,
                'uptime_formatted': self._format_duration(uptime),
                'monitoring_interval': self.monitor_interval,
                'detailed_tracking_enabled': self.enable_detailed_tracking
            },
            'system_overview': {
                'total_operations': total_operations,
                'total_successes': total_successes,
                'total_errors': total_errors,
                'overall_success_rate': (total_successes / total_operations * 100) if total_operations > 0 else 100,
                'overall_error_rate': (total_errors / total_operations * 100) if total_operations > 0 else 0,
                'operations_per_minute': (total_operations / (uptime / 60)) if uptime > 0 else 0
            },
            'system_health': system_health,
            'operation_statistics': operation_stats,
            'resource_usage': current_resources,
            'performance_trends': performance_trends,
            'active_alerts': [self._alert_to_dict(alert) for alert in self.active_alerts],
            'performance_score': self._calculate_performance_score(),
            'optimization_opportunities': self.identify_optimization_opportunities(),
            'baseline_metrics': self.baseline_metrics,
            'timestamp': current_time.isoformat()
        }
    
    def get_operation_deep_analysis(self, operation_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive deep analysis for a specific operation.
        
        Args:
            operation_name: Name of the operation to analyze
            hours: Number of hours to analyze
            
        Returns:
            Deep analysis of the operation performance
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
        
        # Calculate comprehensive statistics
        durations = [m['duration'] for m in recent_metrics]
        successes = [m['success'] for m in recent_metrics]
        components = [m.get('component', 'unknown') for m in recent_metrics]
        
        # Time-based analysis with hourly granularity
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
                'min_duration': min(hour_durations),
                'max_duration': max(hour_durations),
                'success_rate': sum(hour_successes) / len(hour_successes) * 100,
                'error_count': len(hour_successes) - sum(hour_successes)
            }
        
        # Performance pattern analysis
        performance_patterns = self._analyze_performance_patterns(recent_metrics)
        
        # Error analysis
        error_analysis = self._analyze_operation_errors(recent_metrics)
        
        return {
            'operation_name': operation_name,
            'analysis_period': {
                'hours': hours,
                'start_time': cutoff_time.isoformat(),
                'end_time': datetime.now().isoformat()
            },
            'execution_summary': {
                'total_calls': len(recent_metrics),
                'successful_calls': sum(successes),
                'failed_calls': len(successes) - sum(successes),
                'success_rate': sum(successes) / len(successes) * 100,
                'components_involved': list(set(components))
            },
            'performance_metrics': {
                'avg_duration': statistics.mean(durations),
                'median_duration': statistics.median(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'duration_std_dev': statistics.stdev(durations) if len(durations) > 1 else 0,
                'percentiles': {
                    '50th': statistics.median(durations),
                    '75th': self._percentile(durations, 75),
                    '90th': self._percentile(durations, 90),
                    '95th': self._percentile(durations, 95),
                    '99th': self._percentile(durations, 99),
                    '99.9th': self._percentile(durations, 99.9)
                }
            },
            'temporal_analysis': {
                'hourly_breakdown': hourly_stats,
                'performance_trend': self._calculate_trend(durations),
                'peak_hours': self._identify_peak_hours(hourly_stats),
                'performance_stability': self._calculate_stability_score(durations)
            },
            'performance_patterns': performance_patterns,
            'error_analysis': error_analysis,
            'optimization_recommendations': self._get_operation_optimization_recommendations(operation_name, recent_metrics)
        }
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify and prioritize optimization opportunities across the system.
        
        Returns:
            List of optimization opportunities with priorities and recommendations
        """
        opportunities = []
        
        # Analyze slow operations
        slow_operations = self._identify_slow_operations()
        if slow_operations:
            opportunities.append({
                'type': 'performance',
                'priority': 'high',
                'category': 'slow_operations',
                'title': 'Slow Operations Detected',
                'description': f"{len(slow_operations)} operations are performing slower than optimal",
                'affected_operations': slow_operations,
                'potential_impact': 'High - Significant user experience improvement',
                'estimated_effort': 'Medium - Algorithm optimization required',
                'recommendations': [
                    'Profile slow operations to identify bottlenecks',
                    'Implement caching for frequently accessed data',
                    'Consider algorithm optimization or parallelization',
                    'Add performance monitoring to track improvements'
                ]
            })
        
        # Analyze error-prone operations
        error_prone = self._identify_error_prone_operations()
        if error_prone:
            opportunities.append({
                'type': 'reliability',
                'priority': 'high',
                'category': 'error_prone_operations',
                'title': 'High Error Rate Operations',
                'description': f"{len(error_prone)} operations have elevated error rates",
                'affected_operations': error_prone,
                'potential_impact': 'High - Improved system reliability',
                'estimated_effort': 'Medium - Error handling improvements',
                'recommendations': [
                    'Implement better error handling and retry logic',
                    'Add input validation and sanitization',
                    'Review operation dependencies and failure modes',
                    'Consider circuit breaker patterns for external dependencies'
                ]
            })
        
        # Resource usage optimization
        resource_issues = self._identify_resource_optimization_opportunities()
        if resource_issues:
            opportunities.extend(resource_issues)
        
        # Performance trend analysis
        trend_issues = self._identify_trend_based_opportunities()
        if trend_issues:
            opportunities.extend(trend_issues)
        
        # Sort by priority and potential impact
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        opportunities.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return opportunities
    
    def generate_performance_report(self, include_recommendations: bool = True,
                                  include_trends: bool = True,
                                  format_type: str = 'detailed') -> Dict[str, Any]:
        """
        Generate comprehensive performance report for management and technical teams.
        
        Args:
            include_recommendations: Include optimization recommendations
            include_trends: Include trend analysis
            format_type: 'summary', 'detailed', or 'executive'
            
        Returns:
            Formatted performance report
        """
        base_summary = self.get_comprehensive_summary()
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': format_type,
                'monitoring_period': {
                    'start': self.start_time.isoformat(),
                    'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600
                },
                'data_points_analyzed': sum(len(metrics) for metrics in self.operation_metrics.values())
            },
            'executive_summary': self._generate_executive_summary(base_summary),
            'system_health_overview': base_summary['system_health'],
            'key_performance_indicators': self._extract_kpis(base_summary)
        }
        
        if format_type in ['detailed', 'technical']:
            report.update({
                'detailed_operation_analysis': base_summary['operation_statistics'],
                'resource_utilization': base_summary['resource_usage'],
                'performance_alerts': base_summary['active_alerts']
            })
        
        if include_trends and format_type != 'summary':
            report['performance_trends'] = base_summary['performance_trends']
        
        if include_recommendations:
            report['optimization_recommendations'] = self.identify_optimization_opportunities()
        
        return report
    
    def _monitoring_loop(self):
        """Enhanced monitoring loop with comprehensive metrics collection."""
        baseline_samples = 0
        baseline_target = 10  # Collect 10 samples to establish baseline
        
        while self.monitoring_active:
            try:
                # Collect comprehensive system resource metrics
                resource_metrics = self._get_comprehensive_resource_usage()
                resource_metrics['timestamp'] = datetime.now().isoformat()
                self.resource_usage.append(resource_metrics)
                
                # Establish baseline metrics
                if not self.baseline_metrics['baseline_established'] and baseline_samples < baseline_target:
                    self._update_baseline_metrics(resource_metrics)
                    baseline_samples += 1
                    if baseline_samples >= baseline_target:
                        self.baseline_metrics['baseline_established'] = True
                        logger.info("Performance baseline established")
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Check for system-wide performance issues
                self._check_system_performance()
                
                # Update performance trends
                self._update_performance_trends()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                time.sleep(5)  # Back off on error
    
    def _get_comprehensive_resource_usage(self) -> Dict[str, Any]:
        """Get comprehensive system resource usage with enhanced metrics."""
        try:
            # Get current process
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            system_cpu = psutil.cpu_percent()
            
            # Disk I/O
            io_counters = process.io_counters() if hasattr(process, 'io_counters') else None
            
            # Network connections
            connections = len(process.connections()) if hasattr(process, 'connections') else 0
            
            return {
                'process_memory_mb': memory_info.rss / (1024 * 1024),
                'process_memory_percent': (memory_info.rss / system_memory.total) * 100,
                'system_memory_percent': system_memory.percent,
                'process_cpu_percent': cpu_percent,
                'system_cpu_percent': system_cpu,
                'thread_count': process.num_threads(),
                'file_descriptors': process.num_fds() if hasattr(process, 'num_fds') else 0,
                'network_connections': connections,
                'disk_io_read_mb': io_counters.read_bytes / (1024 * 1024) if io_counters else 0,
                'disk_io_write_mb': io_counters.write_bytes / (1024 * 1024) if io_counters else 0,
                'process_status': process.status(),
                'cpu_times': process.cpu_times()._asdict() if hasattr(process, 'cpu_times') else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive resource usage: {e}")
            return {}
    
    def _get_current_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _check_operation_performance(self, operation_name: str, duration: float, success: bool):
        """Check individual operation performance and generate alerts."""
        # Check duration thresholds
        if duration > self.thresholds['processing_time_critical']:
            self._create_alert('performance', 'critical', operation_name,
                             f"Critical performance issue: {operation_name} took {duration:.2f}s",
                             self.thresholds['processing_time_critical'], duration)
        elif duration > self.thresholds['processing_time_warning']:
            self._create_alert('performance', 'warning', operation_name,
                             f"Performance warning: {operation_name} took {duration:.2f}s",
                             self.thresholds['processing_time_warning'], duration)
        
        # Check error rates
        if not success:
            total_calls = len(self.operation_metrics[operation_name])
            error_rate = self.error_counts[operation_name] / total_calls if total_calls > 0 else 0
            
            if error_rate > self.thresholds['error_rate_critical']:
                self._create_alert('reliability', 'critical', operation_name,
                                 f"Critical error rate: {error_rate*100:.1f}% for {operation_name}",
                                 self.thresholds['error_rate_critical'], error_rate)
            elif error_rate > self.thresholds['error_rate_warning']:
                self._create_alert('reliability', 'warning', operation_name,
                                 f"High error rate: {error_rate*100:.1f}% for {operation_name}",
                                 self.thresholds['error_rate_warning'], error_rate)
    
    def _create_alert(self, alert_type: str, severity: str, operation: str, 
                     message: str, threshold: float, current_value: float, metadata: Dict = None):
        """Create and manage performance alerts."""
        alert = PerformanceAlert(alert_type, severity, operation, message, 
                               threshold, current_value, metadata)
        
        # Check if similar alert already exists
        existing_alert = next((a for a in self.active_alerts 
                             if a.alert_type == alert_type and a.operation == operation), None)
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.timestamp = datetime.now()
        else:
            # Add new alert
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
        
        # Log the alert
        log_level = logging.CRITICAL if severity == 'critical' else logging.WARNING
        logger.log(log_level, f"Performance Alert: {message}")
    
    def _calculate_comprehensive_health(self) -> Dict[str, Any]:
        """Calculate comprehensive system health with multiple factors."""
        if not self.operation_metrics:
            return {'status': 'unknown', 'score': 0, 'factors': {}}
        
        # Calculate various health factors
        factors = {}
        
        # Error rate factor
        total_operations = sum(len(metrics) for metrics in self.operation_metrics.values())
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / total_operations if total_operations > 0 else 0
        factors['error_rate'] = {
            'value': error_rate,
            'score': max(0, 100 - (error_rate * 1000)),  # Heavily penalize errors
            'weight': 0.3
        }
        
        # Performance factor
        all_durations = []
        for metrics in self.operation_metrics.values():
            all_durations.extend([m['duration'] for m in metrics[-20:]])  # Recent operations
        
        avg_duration = statistics.mean(all_durations) if all_durations else 0
        performance_score = max(0, 100 - (avg_duration * 10))  # 10s -> 0 score
        factors['performance'] = {
            'value': avg_duration,
            'score': performance_score,
            'weight': 0.25
        }
        
        # Resource usage factor
        resource_usage = self._get_comprehensive_resource_usage()
        memory_percent = resource_usage.get('process_memory_percent', 0)
        cpu_percent = resource_usage.get('process_cpu_percent', 0)
        resource_score = max(0, 100 - max(memory_percent, cpu_percent))
        factors['resource_usage'] = {
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent,
            'score': resource_score,
            'weight': 0.2
        }
        
        # Alert factor
        critical_alerts = len([a for a in self.active_alerts if a.severity == 'critical'])
        warning_alerts = len([a for a in self.active_alerts if a.severity == 'warning'])
        alert_score = max(0, 100 - (critical_alerts * 30) - (warning_alerts * 10))
        factors['alerts'] = {
            'critical_count': critical_alerts,
            'warning_count': warning_alerts,
            'score': alert_score,
            'weight': 0.15
        }
        
        # Trend factor
        trend_score = self._calculate_trend_health_score()
        factors['trends'] = {
            'score': trend_score,
            'weight': 0.1
        }
        
        # Calculate weighted overall score
        total_weight = sum(factor['weight'] for factor in factors.values())
        overall_score = sum(factor['score'] * factor['weight'] for factor in factors.values()) / total_weight
        
        # Determine status
        if overall_score >= 90:
            status = 'excellent'
        elif overall_score >= 75:
            status = 'good'
        elif overall_score >= 60:
            status = 'fair'
        elif overall_score >= 40:
            status = 'poor'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': round(overall_score, 2),
            'factors': factors,
            'recommendations': self._get_health_recommendations(factors)
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value with improved accuracy."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[min(int(index) + 1, len(sorted_values) - 1)]
            return lower + (upper - lower) * (index - int(index))
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values with improved analysis."""
        if len(values) < 5:
            return 'stable'
        
        # Use linear regression for better trend detection
        try:
            x = list(range(len(values)))
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Determine trend based on slope and significance
            avg_value = statistics.mean(values)
            relative_slope = slope / avg_value if avg_value != 0 else 0
            
            if relative_slope > 0.1:
                return 'increasing'
            elif relative_slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except (ZeroDivisionError, ValueError):
            return 'stable'
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        else:
            return f"{seconds/86400:.1f} days"
    
    def _alert_to_dict(self, alert: PerformanceAlert) -> Dict[str, Any]:
        """Convert alert object to dictionary."""
        return {
            'alert_id': alert.alert_id,
            'type': alert.alert_type,
            'severity': alert.severity,
            'operation': alert.operation,
            'message': alert.message,
            'threshold': alert.threshold,
            'current_value': alert.current_value,
            'timestamp': alert.timestamp.isoformat(),
            'metadata': alert.metadata
        }
    
    def _cleanup_old_metrics(self):
        """Enhanced cleanup with intelligent retention."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for operation_name in list(self.operation_metrics.keys()):
            metrics = self.operation_metrics[operation_name]
            
            # Keep recent metrics and maintain minimum samples
            recent_metrics = [
                m for m in metrics
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            if recent_metrics:
                # Keep only the most recent metrics per operation with intelligent sampling
                if len(recent_metrics) > 200:
                    # Keep all recent (last hour) + sample older ones
                    very_recent_cutoff = datetime.now() - timedelta(hours=1)
                    very_recent = [m for m in recent_metrics 
                                 if datetime.fromisoformat(m['timestamp']) > very_recent_cutoff]
                    older = [m for m in recent_metrics 
                           if datetime.fromisoformat(m['timestamp']) <= very_recent_cutoff]
                    
                    # Sample older metrics (keep every nth metric)
                    sample_rate = max(1, len(older) // 100)
                    sampled_older = older[::sample_rate]
                    
                    self.operation_metrics[operation_name] = very_recent + sampled_older
                else:
                    self.operation_metrics[operation_name] = recent_metrics
            else:
                # Remove operation if no recent metrics
                del self.operation_metrics[operation_name]
                if operation_name in self.error_counts:
                    del self.error_counts[operation_name]
                if operation_name in self.success_counts:
                    del self.success_counts[operation_name]
        
        # Clean up old alerts
        self.active_alerts = [
            alert for alert in self.active_alerts
            if (datetime.now() - alert.timestamp).total_seconds() < 3600  # Keep for 1 hour
        ]
    
    def shutdown(self):
        """Shutdown performance monitoring system."""
        self.stop_monitoring()
        logger.info("Enterprise Performance Monitor shutdown")

# Global instance
enterprise_performance_monitor = EnterprisePerformanceMonitor()