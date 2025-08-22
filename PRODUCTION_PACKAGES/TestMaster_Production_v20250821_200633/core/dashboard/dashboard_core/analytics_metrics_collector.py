"""
Analytics Metrics Collector
===========================

Comprehensive metrics collection and exposure system for monitoring
all analytics components and system performance.

Author: TestMaster Team
"""

import logging
import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"

@dataclass
class Metric:
    """Represents a single metric."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    description: Optional[str] = None
    unit: Optional[str] = None

@dataclass
class HistogramBucket:
    """Histogram bucket for distribution metrics."""
    upper_bound: float
    count: int

class Timer:
    """Timer context manager for measuring durations."""
    
    def __init__(self, collector: 'AnalyticsMetricsCollector', name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.labels)

class AnalyticsMetricsCollector:
    """
    Comprehensive metrics collection system for analytics components.
    """
    
    def __init__(self, collection_interval: float = 10.0):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Interval between automatic collections in seconds
        """
        self.collection_interval = collection_interval
        
        # Metric storage
        self.metrics = defaultdict(list)
        self.metric_metadata = {}
        self.labels_index = defaultdict(set)
        
        # Counters
        self.counters = defaultdict(float)
        self.counter_metadata = {}
        
        # Gauges
        self.gauges = defaultdict(float)
        self.gauge_metadata = {}
        
        # Histograms
        self.histograms = defaultdict(lambda: {
            'buckets': defaultdict(int),
            'sum': 0.0,
            'count': 0
        })
        self.histogram_metadata = {}
        
        # Timers (special case of histograms)
        self.timers = defaultdict(lambda: {
            'total_time': 0.0,
            'count': 0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'recent_times': deque(maxlen=1000)
        })
        
        # System metrics
        self.system_metrics_enabled = True
        self.last_system_collection = None
        
        # Collection control
        self.collecting_active = False
        self.collection_thread = None
        
        # Export formats
        self.exporters = {}
        
        # Callbacks
        self.metric_callbacks = []
        
        # Statistics
        self.collector_stats = {
            'total_metrics_collected': 0,
            'collections_performed': 0,
            'start_time': datetime.now(),
            'last_collection_time': None,
            'collection_errors': 0
        }
        
        # Setup default histogram buckets
        self.default_histogram_buckets = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')
        ]
        
        logger.info("Analytics Metrics Collector initialized")
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if self.collecting_active:
            return
        
        self.collecting_active = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self.collecting_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def increment_counter(self, name: str, value: float = 1.0, 
                         labels: Dict[str, str] = None,
                         description: str = None, unit: str = None):
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            labels: Metric labels
            description: Metric description
            unit: Metric unit
        """
        labels = labels or {}
        metric_key = self._create_metric_key(name, labels)
        
        self.counters[metric_key] += value
        self.counter_metadata[name] = {
            'description': description,
            'unit': unit,
            'type': MetricType.COUNTER
        }
        
        self._record_metric(name, MetricType.COUNTER, self.counters[metric_key], labels, description, unit)
        self._update_labels_index(name, labels)
    
    def set_gauge(self, name: str, value: float,
                  labels: Dict[str, str] = None,
                  description: str = None, unit: str = None):
        """
        Set a gauge metric value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
            description: Metric description
            unit: Metric unit
        """
        labels = labels or {}
        metric_key = self._create_metric_key(name, labels)
        
        self.gauges[metric_key] = value
        self.gauge_metadata[name] = {
            'description': description,
            'unit': unit,
            'type': MetricType.GAUGE
        }
        
        self._record_metric(name, MetricType.GAUGE, value, labels, description, unit)
        self._update_labels_index(name, labels)
    
    def record_histogram(self, name: str, value: float,
                        labels: Dict[str, str] = None,
                        description: str = None, unit: str = None,
                        buckets: List[float] = None):
        """
        Record a histogram metric.
        
        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
            description: Metric description
            unit: Metric unit
            buckets: Histogram buckets
        """
        labels = labels or {}
        buckets = buckets or self.default_histogram_buckets
        metric_key = self._create_metric_key(name, labels)
        
        histogram = self.histograms[metric_key]
        histogram['sum'] += value
        histogram['count'] += 1
        
        # Update buckets
        for bucket in buckets:
            if value <= bucket:
                histogram['buckets'][bucket] += 1
        
        self.histogram_metadata[name] = {
            'description': description,
            'unit': unit,
            'type': MetricType.HISTOGRAM,
            'buckets': buckets
        }
        
        self._record_metric(name, MetricType.HISTOGRAM, value, labels, description, unit)
        self._update_labels_index(name, labels)
    
    def record_timer(self, name: str, duration: float,
                    labels: Dict[str, str] = None,
                    description: str = None):
        """
        Record a timer metric.
        
        Args:
            name: Timer name
            duration: Duration in seconds
            labels: Metric labels
            description: Timer description
        """
        labels = labels or {}
        metric_key = self._create_metric_key(name, labels)
        
        timer = self.timers[metric_key]
        timer['total_time'] += duration
        timer['count'] += 1
        timer['min_time'] = min(timer['min_time'], duration)
        timer['max_time'] = max(timer['max_time'], duration)
        timer['recent_times'].append(duration)
        
        # Also record as histogram for distribution analysis
        self.record_histogram(f"{name}_histogram", duration, labels, 
                            description, "seconds")
        
        self._record_metric(name, MetricType.TIMER, duration, labels, description, "seconds")
        self._update_labels_index(name, labels)
    
    def timer(self, name: str, labels: Dict[str, str] = None) -> Timer:
        """
        Create a timer context manager.
        
        Args:
            name: Timer name
            labels: Metric labels
        
        Returns:
            Timer context manager
        """
        return Timer(self, name, labels)
    
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        if not self.system_metrics_enabled:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.set_gauge("system_cpu_usage_percent", cpu_percent,
                          description="CPU usage percentage")
            
            cpu_count = psutil.cpu_count()
            self.set_gauge("system_cpu_count", cpu_count,
                          description="Number of CPU cores")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_total_bytes", memory.total,
                          description="Total memory in bytes", unit="bytes")
            self.set_gauge("system_memory_used_bytes", memory.used,
                          description="Used memory in bytes", unit="bytes")
            self.set_gauge("system_memory_available_bytes", memory.available,
                          description="Available memory in bytes", unit="bytes")
            self.set_gauge("system_memory_usage_percent", memory.percent,
                          description="Memory usage percentage")
            
            # Disk metrics
            try:
                disk = psutil.disk_usage('/')
                self.set_gauge("system_disk_total_bytes", disk.total,
                              description="Total disk space in bytes", unit="bytes")
                self.set_gauge("system_disk_used_bytes", disk.used,
                              description="Used disk space in bytes", unit="bytes")
                self.set_gauge("system_disk_free_bytes", disk.free,
                              description="Free disk space in bytes", unit="bytes")
            except Exception:
                pass  # Disk metrics may not be available on all systems
            
            # Network metrics
            try:
                network = psutil.net_io_counters()
                self.increment_counter("system_network_bytes_sent_total", network.bytes_sent,
                                     description="Total bytes sent over network", unit="bytes")
                self.increment_counter("system_network_bytes_received_total", network.bytes_recv,
                                     description="Total bytes received over network", unit="bytes")
                self.increment_counter("system_network_packets_sent_total", network.packets_sent,
                                     description="Total packets sent over network")
                self.increment_counter("system_network_packets_received_total", network.packets_recv,
                                     description="Total packets received over network")
            except Exception:
                pass  # Network metrics may not be available
            
            # Process metrics
            process = psutil.Process()
            self.set_gauge("process_memory_rss_bytes", process.memory_info().rss,
                          description="Process RSS memory in bytes", unit="bytes")
            self.set_gauge("process_memory_vms_bytes", process.memory_info().vms,
                          description="Process VMS memory in bytes", unit="bytes")
            self.set_gauge("process_cpu_percent", process.cpu_percent(),
                          description="Process CPU usage percentage")
            
            # Python GC metrics
            gc_stats = gc.get_stats()
            for i, gen_stats in enumerate(gc_stats):
                self.increment_counter(f"python_gc_collections_total", gen_stats['collections'],
                                     labels={'generation': str(i)},
                                     description="Total garbage collections by generation")
                self.increment_counter(f"python_gc_objects_collected_total", gen_stats['collected'],
                                     labels={'generation': str(i)},
                                     description="Total objects collected by generation")
            
            self.last_system_collection = datetime.now()
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            self.collector_stats['collection_errors'] += 1
    
    def collect_analytics_component_metrics(self, component_name: str, 
                                          component_stats: Dict[str, Any]):
        """
        Collect metrics from analytics components.
        
        Args:
            component_name: Name of the component
            component_stats: Component statistics
        """
        try:
            labels = {'component': component_name}
            
            # Convert stats to metrics
            for key, value in component_stats.items():
                if isinstance(value, (int, float)):
                    metric_name = f"analytics_{component_name}_{key}"
                    self.set_gauge(metric_name, float(value), labels,
                                  description=f"{component_name} {key}")
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            metric_name = f"analytics_{component_name}_{key}_{sub_key}"
                            self.set_gauge(metric_name, float(sub_value), labels,
                                          description=f"{component_name} {key} {sub_key}")
        
        except Exception as e:
            logger.error(f"Component metrics collection failed for {component_name}: {e}")
    
    def get_metrics(self, name_filter: str = None, 
                   label_filter: Dict[str, str] = None,
                   max_age_seconds: int = None) -> List[Metric]:
        """
        Get collected metrics with optional filtering.
        
        Args:
            name_filter: Filter by metric name (substring match)
            label_filter: Filter by labels (exact match)
            max_age_seconds: Maximum age of metrics to return
        
        Returns:
            List of matching metrics
        """
        filtered_metrics = []
        cutoff_time = None
        
        if max_age_seconds:
            cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)
        
        for metric_name, metric_list in self.metrics.items():
            if name_filter and name_filter not in metric_name:
                continue
            
            for metric in metric_list:
                # Age filter
                if cutoff_time and metric.timestamp < cutoff_time:
                    continue
                
                # Label filter
                if label_filter:
                    if not all(metric.labels.get(k) == v for k, v in label_filter.items()):
                        continue
                
                filtered_metrics.append(metric)
        
        return sorted(filtered_metrics, key=lambda m: m.timestamp, reverse=True)
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current values of all metrics."""
        current_values = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {},
            'timers': {},
            'timestamp': datetime.now().isoformat(),
            'collector_stats': self.collector_stats.copy()
        }
        
        # Process histograms
        for key, histogram in self.histograms.items():
            current_values['histograms'][key] = {
                'buckets': dict(histogram['buckets']),
                'sum': histogram['sum'],
                'count': histogram['count'],
                'average': histogram['sum'] / histogram['count'] if histogram['count'] > 0 else 0
            }
        
        # Process timers
        for key, timer in self.timers.items():
            recent_times = list(timer['recent_times'])
            current_values['timers'][key] = {
                'total_time': timer['total_time'],
                'count': timer['count'],
                'min_time': timer['min_time'] if timer['min_time'] != float('inf') else 0,
                'max_time': timer['max_time'],
                'average_time': timer['total_time'] / timer['count'] if timer['count'] > 0 else 0,
                'recent_average': sum(recent_times) / len(recent_times) if recent_times else 0
            }
        
        return current_values
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Export counters
        for name, metadata in self.counter_metadata.items():
            if metadata.get('description'):
                lines.append(f"# HELP {name} {metadata['description']}")
            lines.append(f"# TYPE {name} counter")
            
            for key, value in self.counters.items():
                if key.startswith(f"{name}|"):
                    labels_str = self._extract_labels_from_key(key)
                    lines.append(f"{name}{labels_str} {value}")
        
        # Export gauges
        for name, metadata in self.gauge_metadata.items():
            if metadata.get('description'):
                lines.append(f"# HELP {name} {metadata['description']}")
            lines.append(f"# TYPE {name} gauge")
            
            for key, value in self.gauges.items():
                if key.startswith(f"{name}|"):
                    labels_str = self._extract_labels_from_key(key)
                    lines.append(f"{name}{labels_str} {value}")
        
        # Export histograms
        for name, metadata in self.histogram_metadata.items():
            if metadata.get('description'):
                lines.append(f"# HELP {name} {metadata['description']}")
            lines.append(f"# TYPE {name} histogram")
            
            for key, histogram in self.histograms.items():
                if key.startswith(f"{name}|"):
                    labels_str = self._extract_labels_from_key(key)
                    base_labels = labels_str.rstrip('}')
                    
                    # Export buckets
                    for bucket, count in histogram['buckets'].items():
                        bucket_labels = f"{base_labels},le=\"{bucket}\"}}" if base_labels else f"{{le=\"{bucket}\"}}"
                        lines.append(f"{name}_bucket{bucket_labels} {count}")
                    
                    # Export sum and count
                    lines.append(f"{name}_sum{labels_str} {histogram['sum']}")
                    lines.append(f"{name}_count{labels_str} {histogram['count']}")
        
        return '\n'.join(lines)
    
    def export_json(self) -> str:
        """Export metrics in JSON format."""
        return json.dumps(self.get_current_values(), indent=2, default=str)
    
    def _create_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        
        label_pairs = sorted(labels.items())
        label_str = ','.join(f"{k}={v}" for k, v in label_pairs)
        return f"{name}|{label_str}"
    
    def _extract_labels_from_key(self, key: str) -> str:
        """Extract labels from metric key for Prometheus format."""
        if '|' not in key:
            return ""
        
        _, label_str = key.split('|', 1)
        label_pairs = label_str.split(',')
        formatted_labels = []
        
        for pair in label_pairs:
            k, v = pair.split('=', 1)
            formatted_labels.append(f'{k}="{v}"')
        
        return '{' + ','.join(formatted_labels) + '}'
    
    def _record_metric(self, name: str, metric_type: MetricType, value: float,
                      labels: Dict[str, str], description: str = None, unit: str = None):
        """Record a metric to the metrics list."""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels or {},
            timestamp=datetime.now(),
            description=description,
            unit=unit
        )
        
        self.metrics[name].append(metric)
        
        # Keep only recent metrics to prevent memory growth
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-500:]
        
        self.collector_stats['total_metrics_collected'] += 1
        
        # Trigger callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Metric callback error: {e}")
    
    def _update_labels_index(self, name: str, labels: Dict[str, str]):
        """Update labels index for efficient querying."""
        for key, value in labels.items():
            self.labels_index[f"{name}.{key}"].add(value)
    
    def _collection_loop(self):
        """Background collection loop."""
        while self.collecting_active:
            try:
                start_time = time.time()
                
                # Collect system metrics
                self.collect_system_metrics()
                
                # Update collector stats
                self.collector_stats['collections_performed'] += 1
                self.collector_stats['last_collection_time'] = datetime.now()
                
                collection_duration = time.time() - start_time
                self.record_timer("metrics_collection_duration", collection_duration,
                                description="Time taken to collect all metrics")
                
                # Sleep until next collection
                time.sleep(max(0, self.collection_interval - collection_duration))
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                self.collector_stats['collection_errors'] += 1
                time.sleep(self.collection_interval)
    
    def add_metric_callback(self, callback: Callable[[Metric], None]):
        """Add a callback for when metrics are recorded."""
        self.metric_callbacks.append(callback)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get collector summary."""
        uptime = (datetime.now() - self.collector_stats['start_time']).total_seconds()
        
        return {
            'total_metrics': len(self.metrics),
            'total_counters': len(self.counters),
            'total_gauges': len(self.gauges),
            'total_histograms': len(self.histograms),
            'total_timers': len(self.timers),
            'collection_interval': self.collection_interval,
            'collecting_active': self.collecting_active,
            'system_metrics_enabled': self.system_metrics_enabled,
            'uptime_seconds': uptime,
            'stats': self.collector_stats.copy()
        }
    
    def shutdown(self):
        """Shutdown metrics collector."""
        self.stop_collection()
        logger.info("Analytics Metrics Collector shutdown")