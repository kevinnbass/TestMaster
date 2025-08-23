#!/usr/bin/env python3
"""
🏗️ MODULE: Profiling Metrics - Performance Measurement & Collection
==================================================================

📋 PURPOSE:
    Performance metric collection and monitoring functionality extracted
    from performance_profiler.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • PerformanceMetric dataclass for individual measurements
    • Core profiling methods (system metrics, API responses, cache performance)
    • Component timing functionality
    • Metric collection and threshold checking

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract profiling metrics from performance_profiler.py
   └─ Source: Lines 23-248 (225 lines)
   └─ Purpose: Separate metric collection logic into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: psutil, datetime, threading, dataclasses, logging
📤 Provides: Performance metric collection and monitoring
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    timestamp: datetime
    metric_type: str
    value: float
    component: str
    additional_data: Dict[str, Any] = None


class MetricsCollector:
    """Handles performance metric collection and monitoring."""
    
    def __init__(self, max_metrics_history: int = 1000):
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
                self.collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def collect_system_metrics(self):
        """Collect system-level performance metrics."""
        timestamp = datetime.now()
        
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Add metrics
            self.add_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_type='system_cpu_percent',
                value=cpu_percent,
                component='system'
            ))
            
            self.add_metric(PerformanceMetric(
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
        
        self.add_metric(metric)
        
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
        
        self.add_metric(metric)
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
        
        self.add_metric(metric)
        self.current_metrics['dashboard_render_ms'] = render_time_ms
    
    def record_component_timing(self, component_name: str, duration_ms: float, 
                              success: bool = True, error_type: str = None):
        """Record component timing metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type='component_duration_ms',
            value=duration_ms,
            component=component_name,
            additional_data={
                'success': success,
                'error_type': error_type
            }
        )
        
        self.add_metric(metric)
        self.current_metrics[f'component_{component_name}_ms'] = duration_ms
    
    def add_metric(self, metric: PerformanceMetric):
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
    
    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff]
    
    def get_metrics_by_type(self, metric_type: str, minutes: int = 5) -> List[float]:
        """Get values for a specific metric type from recent history."""
        recent = self.get_recent_metrics(minutes)
        return [m.value for m in recent if m.metric_type == metric_type]