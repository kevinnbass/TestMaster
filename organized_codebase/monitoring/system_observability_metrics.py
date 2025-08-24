"""
System-Wide Observability Metrics
==================================

Comprehensive observability system providing deep insights into all aspects
of the analytics pipeline with advanced metrics collection and analysis.

Author: TestMaster Team
"""

import logging
import time
import threading
import json
import os
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple, Set
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric type classifications."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"
    PERCENTAGE = "percentage"

class MetricSeverity(Enum):
    """Metric severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    severity: MetricSeverity = MetricSeverity.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'labels': self.labels,
            'severity': self.severity.value
        }

@dataclass
class MetricSeries:
    """Time series of metric points."""
    name: str
    metric_type: MetricType
    description: str
    points: deque
    max_points: int = 1000
    
    def __post_init__(self):
        if not isinstance(self.points, deque):
            self.points = deque(maxlen=self.max_points)
    
    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None, severity: MetricSeverity = MetricSeverity.INFO):
        """Add a metric point."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {},
            severity=severity
        )
        self.points.append(point)
    
    def get_latest(self) -> Optional[MetricPoint]:
        """Get latest metric point."""
        return self.points[-1] if self.points else None
    
    def get_average(self, time_window: Optional[timedelta] = None) -> float:
        """Get average value over time window."""
        if not self.points:
            return 0.0
        
        if time_window:
            cutoff = datetime.now() - time_window
            values = [p.value for p in self.points if p.timestamp >= cutoff]
        else:
            values = [p.value for p in self.points]
        
        return statistics.mean(values) if values else 0.0
    
    def get_percentile(self, percentile: float, time_window: Optional[timedelta] = None) -> float:
        """Get percentile value over time window."""
        if not self.points:
            return 0.0
        
        if time_window:
            cutoff = datetime.now() - time_window
            values = [p.value for p in self.points if p.timestamp >= cutoff]
        else:
            values = [p.value for p in self.points]
        
        if not values:
            return 0.0
        
        values.sort()
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]

class SystemObservabilityMetrics:
    """
    Comprehensive system observability and metrics collection.
    """
    
    def __init__(self, 
                 aggregator=None,
                 collection_interval: float = 5.0,
                 retention_hours: int = 24):
        """
        Initialize observability metrics.
        
        Args:
            aggregator: Analytics aggregator instance
            collection_interval: Seconds between metric collection
            retention_hours: Hours to retain metrics
        """
        self.aggregator = aggregator
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        
        # Metric storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.metric_groups = {
            'system': [],
            'analytics': [],
            'robustness': [],
            'performance': [],
            'errors': [],
            'business': []
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'disk_usage': {'warning': 85.0, 'critical': 95.0},
            'error_rate': {'warning': 5.0, 'critical': 10.0},
            'response_time': {'warning': 2000.0, 'critical': 5000.0},
            'delivery_failure_rate': {'warning': 5.0, 'critical': 10.0},
            'queue_depth': {'warning': 1000, 'critical': 5000}
        }
        
        # Statistics
        self.stats = {
            'collection_cycles': 0,
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'last_collection': None,
            'collection_errors': 0
        }
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Collection thread
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize core metrics
        self._initialize_metrics()
        
        # Start threads
        self.collection_thread.start()
        self.cleanup_thread.start()
        
        logger.info("System Observability Metrics initialized")
    
    def _initialize_metrics(self):
        """Initialize core metric series."""
        
        # System metrics
        system_metrics = [
            ('cpu_usage_percent', MetricType.GAUGE, 'CPU usage percentage'),
            ('memory_usage_percent', MetricType.GAUGE, 'Memory usage percentage'),
            ('disk_usage_percent', MetricType.GAUGE, 'Disk usage percentage'),
            ('network_bytes_sent', MetricType.COUNTER, 'Network bytes sent'),
            ('network_bytes_recv', MetricType.COUNTER, 'Network bytes received'),
            ('open_file_descriptors', MetricType.GAUGE, 'Open file descriptors'),
            ('thread_count', MetricType.GAUGE, 'Active thread count'),
            ('process_count', MetricType.GAUGE, 'Active process count')
        ]
        
        for name, metric_type, desc in system_metrics:
            self.metrics[name] = MetricSeries(name, metric_type, desc, deque(maxlen=1000))
            self.metric_groups['system'].append(name)
        
        # Analytics metrics
        analytics_metrics = [
            ('analytics_requests_total', MetricType.COUNTER, 'Total analytics requests'),
            ('analytics_requests_rate', MetricType.RATE, 'Analytics requests per second'),
            ('analytics_response_time', MetricType.HISTOGRAM, 'Analytics response time (ms)'),
            ('analytics_success_rate', MetricType.PERCENTAGE, 'Analytics success rate'),
            ('analytics_error_rate', MetricType.PERCENTAGE, 'Analytics error rate'),
            ('analytics_throughput', MetricType.GAUGE, 'Analytics throughput (req/s)'),
            ('analytics_queue_depth', MetricType.GAUGE, 'Analytics queue depth'),
            ('analytics_cache_hit_rate', MetricType.PERCENTAGE, 'Cache hit rate')
        ]
        
        for name, metric_type, desc in analytics_metrics:
            self.metrics[name] = MetricSeries(name, metric_type, desc, deque(maxlen=1000))
            self.metric_groups['analytics'].append(name)
        
        # Robustness metrics
        robustness_metrics = [
            ('heartbeat_health_score', MetricType.GAUGE, 'Heartbeat health score'),
            ('fallback_activations', MetricType.COUNTER, 'Fallback activations'),
            ('dead_letter_queue_size', MetricType.GAUGE, 'Dead letter queue size'),
            ('batch_processing_efficiency', MetricType.PERCENTAGE, 'Batch processing efficiency'),
            ('compression_ratio', MetricType.GAUGE, 'Data compression ratio'),
            ('retry_attempts', MetricType.COUNTER, 'Total retry attempts'),
            ('circuit_breaker_trips', MetricType.COUNTER, 'Circuit breaker trips'),
            ('recovery_actions', MetricType.COUNTER, 'Recovery actions triggered')
        ]
        
        for name, metric_type, desc in robustness_metrics:
            self.metrics[name] = MetricSeries(name, metric_type, desc, deque(maxlen=1000))
            self.metric_groups['robustness'].append(name)
        
        # Performance metrics
        performance_metrics = [
            ('request_latency_p50', MetricType.GAUGE, 'Request latency 50th percentile'),
            ('request_latency_p95', MetricType.GAUGE, 'Request latency 95th percentile'),
            ('request_latency_p99', MetricType.GAUGE, 'Request latency 99th percentile'),
            ('database_query_time', MetricType.HISTOGRAM, 'Database query time'),
            ('cache_operation_time', MetricType.HISTOGRAM, 'Cache operation time'),
            ('gc_pause_time', MetricType.HISTOGRAM, 'Garbage collection pause time'),
            ('connection_pool_usage', MetricType.GAUGE, 'Connection pool usage'),
            ('memory_allocation_rate', MetricType.RATE, 'Memory allocation rate')
        ]
        
        for name, metric_type, desc in performance_metrics:
            self.metrics[name] = MetricSeries(name, metric_type, desc, deque(maxlen=1000))
            self.metric_groups['performance'].append(name)
        
        # Error metrics
        error_metrics = [
            ('error_count_total', MetricType.COUNTER, 'Total error count'),
            ('error_rate', MetricType.RATE, 'Error rate per second'),
            ('http_errors_4xx', MetricType.COUNTER, 'HTTP 4xx errors'),
            ('http_errors_5xx', MetricType.COUNTER, 'HTTP 5xx errors'),
            ('timeout_errors', MetricType.COUNTER, 'Timeout errors'),
            ('connection_errors', MetricType.COUNTER, 'Connection errors'),
            ('validation_errors', MetricType.COUNTER, 'Validation errors'),
            ('critical_errors', MetricType.COUNTER, 'Critical errors')
        ]
        
        for name, metric_type, desc in error_metrics:
            self.metrics[name] = MetricSeries(name, metric_type, desc, deque(maxlen=1000))
            self.metric_groups['errors'].append(name)
        
        # Business metrics
        business_metrics = [
            ('active_users', MetricType.GAUGE, 'Active users'),
            ('data_volume_processed', MetricType.COUNTER, 'Data volume processed (bytes)'),
            ('transactions_completed', MetricType.COUNTER, 'Transactions completed'),
            ('revenue_impact', MetricType.GAUGE, 'Revenue impact (estimated)'),
            ('sla_compliance', MetricType.PERCENTAGE, 'SLA compliance percentage'),
            ('customer_satisfaction', MetricType.GAUGE, 'Customer satisfaction score'),
            ('feature_usage', MetricType.COUNTER, 'Feature usage count'),
            ('conversion_rate', MetricType.PERCENTAGE, 'Conversion rate')
        ]
        
        for name, metric_type, desc in business_metrics:
            self.metrics[name] = MetricSeries(name, metric_type, desc, deque(maxlen=1000))
            self.metric_groups['business'].append(name)
    
    def add_alert_handler(self, handler: Callable[[str, str, str, Dict], None]):
        """
        Add alert handler.
        
        Args:
            handler: Function(metric_name, message, severity, context)
        """
        self.alert_handlers.append(handler)
    
    def record_metric(self,
                     name: str,
                     value: float,
                     labels: Optional[Dict[str, str]] = None,
                     severity: MetricSeverity = MetricSeverity.INFO):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            severity: Severity level
        """
        with self.lock:
            if name in self.metrics:
                self.metrics[name].add_point(value, labels, severity)
                self.stats['metrics_collected'] += 1
                
                # Check thresholds
                self._check_threshold(name, value)
            else:
                logger.warning(f"Unknown metric: {name}")
    
    def _check_threshold(self, metric_name: str, value: float):
        """Check if metric value exceeds thresholds."""
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        
        if value >= thresholds.get('critical', float('inf')):
            self._trigger_alert(metric_name, value, MetricSeverity.CRITICAL)
        elif value >= thresholds.get('warning', float('inf')):
            self._trigger_alert(metric_name, value, MetricSeverity.WARNING)
    
    def _trigger_alert(self, metric_name: str, value: float, severity: MetricSeverity):
        """Trigger alert for metric threshold breach."""
        self.stats['alerts_triggered'] += 1
        
        message = f"Metric {metric_name} = {value} exceeded {severity.value} threshold"
        context = {
            'metric_name': metric_name,
            'value': value,
            'severity': severity.value,
            'timestamp': datetime.now().isoformat()
        }
        
        for handler in self.alert_handlers:
            try:
                handler(metric_name, message, severity.value, context)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"OBSERVABILITY ALERT [{severity.value.upper()}]: {message}")
    
    def _collection_loop(self):
        """Background metric collection loop."""
        while self.collection_active:
            try:
                start_time = time.time()
                
                with self.lock:
                    self.stats['collection_cycles'] += 1
                    
                    # Collect system metrics
                    self._collect_system_metrics()
                    
                    # Collect analytics metrics
                    self._collect_analytics_metrics()
                    
                    # Collect robustness metrics
                    self._collect_robustness_metrics()
                    
                    # Collect performance metrics
                    self._collect_performance_metrics()
                    
                    # Collect error metrics
                    self._collect_error_metrics()
                    
                    # Collect business metrics
                    self._collect_business_metrics()
                    
                    self.stats['last_collection'] = datetime.now().isoformat()
                
                # Calculate collection time
                collection_time = (time.time() - start_time) * 1000
                self.record_metric('collection_time_ms', collection_time)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                self.stats['collection_errors'] += 1
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric('cpu_usage_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric('memory_usage_percent', memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric('disk_usage_percent', disk_percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.record_metric('network_bytes_sent', net_io.bytes_sent)
            self.record_metric('network_bytes_recv', net_io.bytes_recv)
            
            # Process info
            process = psutil.Process()
            self.record_metric('open_file_descriptors', process.num_fds())
            self.record_metric('thread_count', process.num_threads())
            
            # System-wide process count
            self.record_metric('process_count', len(psutil.pids()))
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def _collect_analytics_metrics(self):
        """Collect analytics-specific metrics."""
        if not self.aggregator:
            return
        
        try:
            # Get flow monitoring data
            if hasattr(self.aggregator, 'flow_monitor'):
                flow_summary = self.aggregator.flow_monitor.get_flow_summary()
                
                self.record_metric('analytics_queue_depth', 
                                 flow_summary.get('active_transactions', 0))
                
                # Calculate success rate
                completed = flow_summary.get('completed_transactions', 0)
                failed = flow_summary.get('failed_transactions', 0)
                total = completed + failed
                
                if total > 0:
                    success_rate = (completed / total) * 100
                    error_rate = (failed / total) * 100
                    self.record_metric('analytics_success_rate', success_rate)
                    self.record_metric('analytics_error_rate', error_rate)
            
            # Get performance booster metrics
            if hasattr(self.aggregator, 'performance_booster'):
                perf_stats = self.aggregator.performance_booster.get_performance_stats()
                
                avg_response_time = perf_stats.get('avg_response_time', 0) * 1000  # Convert to ms
                self.record_metric('analytics_response_time', avg_response_time)
                
                cache_hit_rate = perf_stats.get('cache_hit_rate', 0) * 100
                self.record_metric('analytics_cache_hit_rate', cache_hit_rate)
            
            # Get batch processor metrics
            if hasattr(self.aggregator, 'batch_processor'):
                batch_status = self.aggregator.batch_processor.get_status()
                
                pending = batch_status.get('pending_items', 0)
                queued = batch_status.get('queued_batches', 0)
                
                self.record_metric('analytics_queue_depth', pending + queued)
                
                # Calculate throughput
                stats = batch_status.get('statistics', {})
                processed = stats.get('items_processed', 0)
                self.record_metric('analytics_throughput', processed)
            
        except Exception as e:
            logger.error(f"Analytics metrics collection failed: {e}")
    
    def _collect_robustness_metrics(self):
        """Collect robustness-specific metrics."""
        if not self.aggregator:
            return
        
        try:
            # Heartbeat health score
            if hasattr(self.aggregator, 'heartbeat_monitor'):
                status = self.aggregator.heartbeat_monitor.get_connection_status()
                
                # Convert health status to numeric score
                health_map = {'healthy': 100, 'degraded': 75, 'failing': 25, 'critical': 0}
                overall_health = status.get('overall_health', 'unknown')
                score = health_map.get(overall_health, 50)
                
                self.record_metric('heartbeat_health_score', score)
            
            # Dead letter queue
            if hasattr(self.aggregator, 'dead_letter_queue'):
                dlq_stats = self.aggregator.dead_letter_queue.get_statistics()
                queue_size = dlq_stats.get('queue_size', 0)
                self.record_metric('dead_letter_queue_size', queue_size)
            
            # Fallback system
            if hasattr(self.aggregator, 'fallback_system'):
                fallback_status = self.aggregator.fallback_system.get_fallback_status()
                
                # Count fallback activations
                current_level = fallback_status.get('current_level', 'primary')
                if current_level != 'primary':
                    self.record_metric('fallback_activations', 1)
            
            # Compression metrics
            if hasattr(self.aggregator, 'compressor'):
                compression_stats = self.aggregator.compressor.get_compression_stats()
                
                ratio = compression_stats.get('avg_compression_ratio', 1.0)
                self.record_metric('compression_ratio', ratio)
            
            # Retry manager
            if hasattr(self.aggregator, 'retry_manager'):
                retry_stats = self.aggregator.retry_manager.get_retry_statistics()
                
                total_retries = retry_stats.get('total_retries', 0)
                self.record_metric('retry_attempts', total_retries)
                
                circuit_trips = retry_stats.get('circuit_breaker_trips', 0)
                self.record_metric('circuit_breaker_trips', circuit_trips)
            
            # Recovery orchestrator
            if hasattr(self.aggregator, 'recovery_orchestrator'):
                orchestrator_status = self.aggregator.recovery_orchestrator.get_orchestrator_status()
                
                stats = orchestrator_status.get('statistics', {})
                recoveries = stats.get('recoveries_initiated', 0)
                self.record_metric('recovery_actions', recoveries)
            
        except Exception as e:
            logger.error(f"Robustness metrics collection failed: {e}")
    
    def _collect_performance_metrics(self):
        """Collect performance-specific metrics."""
        try:
            # Calculate latency percentiles from analytics response times
            if 'analytics_response_time' in self.metrics:
                response_time_series = self.metrics['analytics_response_time']
                time_window = timedelta(minutes=5)
                
                p50 = response_time_series.get_percentile(50, time_window)
                p95 = response_time_series.get_percentile(95, time_window)
                p99 = response_time_series.get_percentile(99, time_window)
                
                self.record_metric('request_latency_p50', p50)
                self.record_metric('request_latency_p95', p95)
                self.record_metric('request_latency_p99', p99)
            
            # Memory allocation rate (approximate)
            memory = psutil.virtual_memory()
            if hasattr(self, '_last_memory_used'):
                allocation_rate = abs(memory.used - self._last_memory_used)
                self.record_metric('memory_allocation_rate', allocation_rate)
            self._last_memory_used = memory.used
            
        except Exception as e:
            logger.error(f"Performance metrics collection failed: {e}")
    
    def _collect_error_metrics(self):
        """Collect error-specific metrics."""
        try:
            # Aggregate error counts from various sources
            total_errors = 0
            
            # From dead letter queue
            if (self.aggregator and 
                hasattr(self.aggregator, 'dead_letter_queue')):
                dlq_stats = self.aggregator.dead_letter_queue.get_statistics()
                dlq_errors = dlq_stats.get('total_failures', 0)
                total_errors += dlq_errors
            
            # From flow monitor failures
            if (self.aggregator and 
                hasattr(self.aggregator, 'flow_monitor')):
                flow_summary = self.aggregator.flow_monitor.get_flow_summary()
                flow_errors = flow_summary.get('failed_transactions', 0)
                total_errors += flow_errors
            
            self.record_metric('error_count_total', total_errors)
            
            # Calculate error rate
            if 'analytics_requests_total' in self.metrics:
                total_requests = self.metrics['analytics_requests_total'].get_latest()
                if total_requests and total_requests.value > 0:
                    error_rate = (total_errors / total_requests.value) * 100
                    self.record_metric('error_rate', error_rate)
            
        except Exception as e:
            logger.error(f"Error metrics collection failed: {e}")
    
    def _collect_business_metrics(self):
        """Collect business-specific metrics."""
        try:
            # Calculate SLA compliance based on success rate and response time
            success_rate = 100.0
            if 'analytics_success_rate' in self.metrics:
                latest = self.metrics['analytics_success_rate'].get_latest()
                if latest:
                    success_rate = latest.value
            
            response_time = 0.0
            if 'analytics_response_time' in self.metrics:
                latest = self.metrics['analytics_response_time'].get_latest()
                if latest:
                    response_time = latest.value
            
            # SLA: 99% success rate and <2000ms response time
            sla_success = success_rate >= 99.0
            sla_performance = response_time <= 2000.0
            sla_compliance = 100.0 if (sla_success and sla_performance) else 0.0
            
            self.record_metric('sla_compliance', sla_compliance)
            
            # Data volume processed (from batch processor)
            if (self.aggregator and 
                hasattr(self.aggregator, 'batch_processor')):
                batch_status = self.aggregator.batch_processor.get_status()
                stats = batch_status.get('statistics', {})
                processed_items = stats.get('items_processed', 0)
                
                # Estimate data volume (assuming 1KB per item)
                data_volume = processed_items * 1024
                self.record_metric('data_volume_processed', data_volume)
            
            # Transaction completion rate
            if (self.aggregator and 
                hasattr(self.aggregator, 'flow_monitor')):
                flow_summary = self.aggregator.flow_monitor.get_flow_summary()
                completed = flow_summary.get('completed_transactions', 0)
                self.record_metric('transactions_completed', completed)
            
        except Exception as e:
            logger.error(f"Business metrics collection failed: {e}")
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.collection_active:
            try:
                time.sleep(3600)  # Cleanup every hour
                
                with self.lock:
                    cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
                    
                    # Clean old metric points
                    for metric_series in self.metrics.values():
                        original_count = len(metric_series.points)
                        
                        # Remove old points
                        while (metric_series.points and 
                               metric_series.points[0].timestamp < cutoff_time):
                            metric_series.points.popleft()
                        
                        cleaned_count = original_count - len(metric_series.points)
                        if cleaned_count > 0:
                            logger.debug(f"Cleaned {cleaned_count} old points from {metric_series.name}")
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
    
    def get_metric_summary(self, group: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metric summary.
        
        Args:
            group: Metric group to filter by
            
        Returns:
            Metric summary
        """
        with self.lock:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'collection_stats': dict(self.stats),
                'metrics': {}
            }
            
            # Filter metrics by group
            if group and group in self.metric_groups:
                metric_names = self.metric_groups[group]
            else:
                metric_names = list(self.metrics.keys())
            
            for name in metric_names:
                if name in self.metrics:
                    series = self.metrics[name]
                    latest = series.get_latest()
                    
                    summary['metrics'][name] = {
                        'type': series.metric_type.value,
                        'description': series.description,
                        'latest_value': latest.value if latest else None,
                        'latest_timestamp': latest.timestamp.isoformat() if latest else None,
                        'point_count': len(series.points),
                        'avg_5min': series.get_average(timedelta(minutes=5)),
                        'avg_1hour': series.get_average(timedelta(hours=1)),
                        'p95_5min': series.get_percentile(95, timedelta(minutes=5)),
                        'severity': latest.severity.value if latest else 'info'
                    }
            
            return summary
    
    def get_metric_data(self,
                       metric_name: str,
                       time_window: Optional[timedelta] = None,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get raw metric data points.
        
        Args:
            metric_name: Name of metric
            time_window: Time window to filter
            limit: Maximum points to return
            
        Returns:
            List of metric points
        """
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            series = self.metrics[metric_name]
            points = list(series.points)
            
            # Apply time window filter
            if time_window:
                cutoff = datetime.now() - time_window
                points = [p for p in points if p.timestamp >= cutoff]
            
            # Apply limit
            if limit:
                points = points[-limit:]
            
            return [point.to_dict() for point in points]
    
    def get_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        with self.lock:
            scores = {}
            weights = {}
            
            # System health (30% weight)
            cpu_latest = self.metrics.get('cpu_usage_percent', MetricSeries('', MetricType.GAUGE, '', deque())).get_latest()
            memory_latest = self.metrics.get('memory_usage_percent', MetricSeries('', MetricType.GAUGE, '', deque())).get_latest()
            
            if cpu_latest and memory_latest:
                system_score = 100 - max(cpu_latest.value, memory_latest.value)
                scores['system'] = max(0, system_score)
                weights['system'] = 30
            
            # Analytics health (40% weight)
            success_rate = self.metrics.get('analytics_success_rate', MetricSeries('', MetricType.GAUGE, '', deque())).get_latest()
            if success_rate:
                scores['analytics'] = success_rate.value
                weights['analytics'] = 40
            
            # Robustness health (30% weight)
            heartbeat_score = self.metrics.get('heartbeat_health_score', MetricSeries('', MetricType.GAUGE, '', deque())).get_latest()
            dlq_size = self.metrics.get('dead_letter_queue_size', MetricSeries('', MetricType.GAUGE, '', deque())).get_latest()
            
            if heartbeat_score:
                robustness_score = heartbeat_score.value
                if dlq_size and dlq_size.value > 0:
                    robustness_score = max(0, robustness_score - dlq_size.value * 2)  # Penalty for queue size
                
                scores['robustness'] = robustness_score
                weights['robustness'] = 30
            
            # Calculate weighted average
            if scores and weights:
                total_weight = sum(weights.values())
                weighted_sum = sum(score * weights[component] for component, score in scores.items())
                overall_score = weighted_sum / total_weight
            else:
                overall_score = 0
            
            return {
                'overall_score': round(overall_score, 1),
                'component_scores': scores,
                'weights': weights,
                'timestamp': datetime.now().isoformat()
            }
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """
        Export metrics in specified format.
        
        Args:
            format_type: Export format ('json', 'prometheus', 'csv')
            
        Returns:
            Formatted metrics string
        """
        summary = self.get_metric_summary()
        
        if format_type == 'json':
            return json.dumps(summary, indent=2)
        
        elif format_type == 'prometheus':
            lines = []
            for name, data in summary['metrics'].items():
                if data['latest_value'] is not None:
                    lines.append(f"# HELP {name} {data['description']}")
                    lines.append(f"# TYPE {name} {data['type']}")
                    lines.append(f"{name} {data['latest_value']}")
            return '\n'.join(lines)
        
        elif format_type == 'csv':
            lines = ['metric_name,latest_value,avg_5min,p95_5min,point_count']
            for name, data in summary['metrics'].items():
                lines.append(f"{name},{data['latest_value']},{data['avg_5min']},{data['p95_5min']},{data['point_count']}")
            return '\n'.join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def shutdown(self):
        """Shutdown observability system."""
        self.collection_active = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=10)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        logger.info(f"System Observability Metrics shutdown - Stats: {self.stats}")

# Global observability instance
observability_metrics = None