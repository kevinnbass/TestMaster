"""
Metric Collector Module
======================

Handles collection and processing of performance metrics.
Extracted from realtime_performance_monitoring.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
import logging
import time
import psutil
import random
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from .monitoring_models import (
    PerformanceMetric, SystemType, MetricCategory,
    AlertSeverity, MonitoringMode
)


class MetricCollector:
    """Handles collection and processing of performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("metric_collector")
        
        # Metrics storage
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.custom_collectors: Dict[str, callable] = {}
        
        # Collection configuration
        self.collection_config = {
            "interval_seconds": 15,
            "batch_size": 50,
            "parallel_collection": True,
            "timeout_seconds": 10
        }
        
        # Collection statistics
        self.collection_stats = {
            "total_collections": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "avg_collection_time_ms": 0.0,
            "last_collection_time": None
        }
        
        # Thread pool for parallel collection
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default system metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize default system metrics."""
        default_metrics = [
            # System resource metrics
            ("cpu_usage", "CPU Usage", MetricCategory.SYSTEM_RESOURCE, "%", 70, 90),
            ("memory_usage", "Memory Usage", MetricCategory.SYSTEM_RESOURCE, "%", 80, 95),
            ("disk_usage", "Disk Usage", MetricCategory.SYSTEM_RESOURCE, "%", 85, 95),
            ("network_latency", "Network Latency", MetricCategory.NETWORK_PERFORMANCE, "ms", 100, 500),
            
            # Application performance metrics
            ("response_time", "Response Time", MetricCategory.APPLICATION_PERFORMANCE, "ms", 1000, 5000),
            ("throughput", "Throughput", MetricCategory.APPLICATION_PERFORMANCE, "req/s", None, None),
            ("error_rate", "Error Rate", MetricCategory.APPLICATION_PERFORMANCE, "%", 1, 5),
            ("queue_depth", "Queue Depth", MetricCategory.APPLICATION_PERFORMANCE, "items", 100, 500),
        ]
        
        for system in SystemType:
            for metric_name, display_name, category, unit, warning, critical in default_metrics:
                metric_id = f"{system.value}.{metric_name}"
                
                metric = PerformanceMetric(
                    metric_id=metric_id,
                    name=display_name,
                    system=system,
                    category=category,
                    unit=unit,
                    warning_threshold=warning,
                    critical_threshold=critical,
                    description=f"{display_name} for {system.value} system"
                )
                
                self.metrics[metric_id] = metric
    
    def register_metric(self, metric: PerformanceMetric):
        """Register a new metric for collection."""
        self.metrics[metric.metric_id] = metric
        self.logger.info(f"Registered metric: {metric.metric_id}")
    
    def register_custom_collector(self, metric_id: str, collector_func: callable):
        """Register custom collector function for a metric."""
        self.custom_collectors[metric_id] = collector_func
        self.logger.info(f"Registered custom collector for: {metric_id}")
    
    async def collect_metrics(self, metric_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Collect metrics asynchronously."""
        start_time = time.time()
        
        # Determine which metrics to collect
        if metric_ids:
            metrics_to_collect = {mid: self.metrics[mid] for mid in metric_ids if mid in self.metrics}
        else:
            metrics_to_collect = {mid: m for mid, m in self.metrics.items() if m.enabled}
        
        results = {}
        
        if self.collection_config["parallel_collection"]:
            # Parallel collection
            tasks = []
            for metric_id, metric in metrics_to_collect.items():
                task = asyncio.create_task(self._collect_single_metric(metric_id, metric))
                tasks.append(task)
            
            collected = await asyncio.gather(*tasks, return_exceptions=True)
            
            for metric_id, value in zip(metrics_to_collect.keys(), collected):
                if not isinstance(value, Exception):
                    results[metric_id] = value
                else:
                    self.logger.error(f"Failed to collect {metric_id}: {value}")
        else:
            # Sequential collection
            for metric_id, metric in metrics_to_collect.items():
                try:
                    value = await self._collect_single_metric(metric_id, metric)
                    results[metric_id] = value
                except Exception as e:
                    self.logger.error(f"Failed to collect {metric_id}: {e}")
        
        # Update statistics
        collection_time = (time.time() - start_time) * 1000
        self._update_collection_stats(len(results), len(metrics_to_collect), collection_time)
        
        return results
    
    async def _collect_single_metric(self, metric_id: str, metric: PerformanceMetric) -> float:
        """Collect a single metric value."""
        # Check for custom collector
        if metric_id in self.custom_collectors:
            value = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.custom_collectors[metric_id]
            )
        else:
            # Use default collection based on metric type
            value = await self._default_collection(metric)
        
        # Add value to metric history
        metric.add_value(value)
        
        return value
    
    async def _default_collection(self, metric: PerformanceMetric) -> float:
        """Default metric collection based on type."""
        # System resource metrics
        if "cpu_usage" in metric.metric_id:
            return psutil.cpu_percent(interval=0.1)
        elif "memory_usage" in metric.metric_id:
            return psutil.virtual_memory().percent
        elif "disk_usage" in metric.metric_id:
            return psutil.disk_usage('/').percent
        elif "network_latency" in metric.metric_id:
            # Simulate network latency
            return random.uniform(10, 150)
        
        # Application performance metrics (simulated)
        elif "response_time" in metric.metric_id:
            return random.uniform(50, 2000)
        elif "throughput" in metric.metric_id:
            return random.uniform(100, 1000)
        elif "error_rate" in metric.metric_id:
            return random.uniform(0, 2)
        elif "queue_depth" in metric.metric_id:
            return random.randint(0, 200)
        
        # Default random value
        return random.uniform(0, 100)
    
    def _update_collection_stats(self, successful: int, total: int, time_ms: float):
        """Update collection statistics."""
        self.collection_stats["total_collections"] += 1
        self.collection_stats["successful_collections"] += successful
        self.collection_stats["failed_collections"] += (total - successful)
        
        # Update average collection time
        prev_avg = self.collection_stats["avg_collection_time_ms"]
        total_count = self.collection_stats["total_collections"]
        self.collection_stats["avg_collection_time_ms"] = \
            (prev_avg * (total_count - 1) + time_ms) / total_count
        
        self.collection_stats["last_collection_time"] = datetime.now()
    
    def get_metric_summary(self, metric_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a metric."""
        if metric_id not in self.metrics:
            return None
        
        metric = self.metrics[metric_id]
        return metric.to_dict()
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics."""
        return {metric_id: metric.to_dict() for metric_id, metric in self.metrics.items()}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.collection_stats.copy()


__all__ = ['MetricCollector']