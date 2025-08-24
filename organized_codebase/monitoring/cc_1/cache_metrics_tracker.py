#!/usr/bin/env python3
"""
🏗️ MODULE: Cache Metrics Tracker - Performance Tracking and Analytics
=====================================================================

📋 PURPOSE:
    Cache performance metrics tracking and analysis system.
    Provides comprehensive monitoring and reporting for cache operations.

🎯 CORE FUNCTIONALITY:
    • Cache hit/miss ratio tracking and analysis
    • Layer-specific performance metrics collection
    • Integration with performance monitoring infrastructure
    • Response time analysis and eviction tracking

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 10:05:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract metrics tracking from advanced_caching_architecture.py via STEELCLAD
   └─ Changes: Created dedicated module for cache performance metrics
   └─ Impact: Clean separation of metrics tracking from core caching operations

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: collections, logging, typing, caching_models
🎯 Integration Points: Performance monitoring infrastructure, all cache layers
⚡ Performance Notes: Efficient metrics collection with minimal cache operation overhead
🔒 Security Notes: Safe metrics aggregation without exposing cache data

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via caching system validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: caching_models, performance monitoring infrastructure (optional)
📤 Provides: Cache performance metrics for all cache operations
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

# Import models
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.caching_models import CacheLayer

# Performance monitoring integration
try:
    from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.monitoring.cc_1.performance_monitoring_infrastructure import PerformanceMonitoringSystem
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class CacheMetrics:
    """Cache performance metrics tracking"""
    
    def __init__(self, monitoring_system: Optional['PerformanceMonitoringSystem'] = None):
        self.monitoring = monitoring_system
        
        # Metrics tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.error_count = 0
        
        # Performance metrics
        self.avg_response_time = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.redis_memory_usage = deque(maxlen=1000)
        
        # Layer-specific metrics
        self.layer_metrics = defaultdict(lambda: {
            'hits': 0, 'misses': 0, 'errors': 0, 'response_times': deque(maxlen=100)
        })
        
        self.logger = logging.getLogger('CacheMetrics')
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    @property
    def miss_ratio(self) -> float:
        """Calculate cache miss ratio"""
        return 1.0 - self.hit_ratio
    
    def record_hit(self, layer: CacheLayer, response_time: float):
        """Record cache hit"""
        self.hit_count += 1
        self.layer_metrics[layer]['hits'] += 1
        self.layer_metrics[layer]['response_times'].append(response_time)
        self.avg_response_time.append(response_time)
        
        if self.monitoring:
            self.monitoring.metrics_collector.collect_metric(
                "cache_hit_total",
                self.hit_count,
                labels={'layer': layer.value},
                unit="count",
                help_text="Total cache hits by layer"
            )
    
    def record_miss(self, layer: CacheLayer, response_time: float):
        """Record cache miss"""
        self.miss_count += 1
        self.layer_metrics[layer]['misses'] += 1
        self.layer_metrics[layer]['response_times'].append(response_time)
        self.avg_response_time.append(response_time)
        
        if self.monitoring:
            self.monitoring.metrics_collector.collect_metric(
                "cache_miss_total",
                self.miss_count,
                labels={'layer': layer.value},
                unit="count",
                help_text="Total cache misses by layer"
            )
    
    def record_eviction(self, layer: CacheLayer, count: int = 1):
        """Record cache evictions"""
        self.eviction_count += count
        
        if self.monitoring:
            self.monitoring.metrics_collector.collect_metric(
                "cache_eviction_total",
                self.eviction_count,
                labels={'layer': layer.value},
                unit="count",
                help_text="Total cache evictions by layer"
            )
    
    def record_error(self, layer: CacheLayer, error_type: str):
        """Record cache error"""
        self.error_count += 1
        self.layer_metrics[layer]['errors'] += 1
        
        if self.monitoring:
            self.monitoring.metrics_collector.collect_metric(
                "cache_error_total",
                self.error_count,
                labels={'layer': layer.value, 'error_type': error_type},
                unit="count",
                help_text="Total cache errors by layer and type"
            )
    
    def record_memory_usage(self, memory_bytes: int, layer: CacheLayer = CacheLayer.MEMORY):
        """Record memory usage metrics"""
        memory_mb = memory_bytes / (1024 * 1024)
        
        if layer == CacheLayer.MEMORY:
            self.memory_usage.append(memory_mb)
        elif layer == CacheLayer.REDIS:
            self.redis_memory_usage.append(memory_mb)
        
        if self.monitoring:
            self.monitoring.metrics_collector.collect_metric(
                "cache_memory_usage_mb",
                memory_mb,
                labels={'layer': layer.value},
                unit="megabytes",
                help_text="Cache memory usage by layer"
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            'hit_ratio': self.hit_ratio,
            'miss_ratio': self.miss_ratio,
            'total_operations': self.hit_count + self.miss_count,
            'error_rate': self.error_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0,
            'avg_response_time': sum(self.avg_response_time) / len(self.avg_response_time) if self.avg_response_time else 0,
            'layer_breakdown': dict(self.layer_metrics),
            'total_evictions': self.eviction_count,
            'memory_stats': {
                'current_memory_mb': self.memory_usage[-1] if self.memory_usage else 0,
                'current_redis_memory_mb': self.redis_memory_usage[-1] if self.redis_memory_usage else 0,
                'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                'avg_redis_memory_mb': sum(self.redis_memory_usage) / len(self.redis_memory_usage) if self.redis_memory_usage else 0
            }
        }
    
    def get_layer_performance(self, layer: CacheLayer) -> Dict[str, Any]:
        """Get performance metrics for specific cache layer"""
        layer_data = self.layer_metrics[layer]
        total_ops = layer_data['hits'] + layer_data['misses']
        
        response_times = list(layer_data['response_times'])
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'layer': layer.value,
            'hits': layer_data['hits'],
            'misses': layer_data['misses'],
            'errors': layer_data['errors'],
            'total_operations': total_ops,
            'hit_ratio': layer_data['hits'] / total_ops if total_ops > 0 else 0,
            'error_rate': layer_data['errors'] / total_ops if total_ops > 0 else 0,
            'avg_response_time': avg_response_time,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0
        }
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        recent_response_times = list(self.avg_response_time)[-100:] if len(self.avg_response_time) > 100 else list(self.avg_response_time)
        
        # Calculate trend (simple linear regression)
        if len(recent_response_times) < 2:
            trend_slope = 0
        else:
            n = len(recent_response_times)
            x_sum = sum(range(n))
            y_sum = sum(recent_response_times)
            xy_sum = sum(i * recent_response_times[i] for i in range(n))
            x2_sum = sum(i * i for i in range(n))
            
            denominator = n * x2_sum - x_sum * x_sum
            trend_slope = (n * xy_sum - x_sum * y_sum) / denominator if denominator != 0 else 0
        
        return {
            'response_time_trend': 'improving' if trend_slope < -0.1 else 'degrading' if trend_slope > 0.1 else 'stable',
            'trend_slope': trend_slope,
            'recent_avg_response_time': sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0,
            'performance_score': min(100, max(0, (self.hit_ratio * 100) - (self.error_count * 10)))
        }
    
    def reset_metrics(self):
        """Reset all metrics counters"""
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.error_count = 0
        self.avg_response_time.clear()
        self.memory_usage.clear()
        self.redis_memory_usage.clear()
        self.layer_metrics.clear()
        
        self.logger.info("Cache metrics reset")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external analysis"""
        return {
            'summary': self.get_metrics_summary(),
            'layer_performance': {
                layer.value: self.get_layer_performance(layer) 
                for layer in [CacheLayer.MEMORY, CacheLayer.REDIS, CacheLayer.HYBRID]
            },
            'trends': self.get_performance_trends(),
            'raw_data': {
                'response_times': list(self.avg_response_time),
                'memory_usage': list(self.memory_usage),
                'redis_memory_usage': list(self.redis_memory_usage)
            }
        }