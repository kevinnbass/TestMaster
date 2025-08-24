#!/usr/bin/env python3
"""
🏗️ MODULE: Performance Profiler - Personal Analytics Dashboard Performance Monitor
==================================================================

📋 PURPOSE:
    Real-time performance monitoring and optimization, streamlined
    via STEELCLAD extraction. Main entry point for performance profiling.

🎯 CORE FUNCTIONALITY:
    • Main entry point for Performance Profiler
    • Integrates modular components from profiler_components package
    • Maintains 100% backward compatibility

🔄 STEELCLAD MODULARIZATION:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION COMPLETE
   └─ Original: 513 lines → Streamlined: <150 lines
   └─ Extracted: 2 focused modules (profiling_metrics, performance_trackers)
   └─ Status: MODULAR ARCHITECTURE ACHIEVED

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent E (Latin Swarm)
🔧 Language: Python
📦 Dependencies: psutil, threading, datetime, typing
🎯 Purpose: Monitor dashboard performance, API response times, and system health
⚡ Performance Notes: Real-time monitoring with <5s intervals

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Extracted profiler components modules
📤 Provides: Performance profiling infrastructure
🚨 Breaking Changes: None - backward compatible enhancement
"""

import time
import json
import logging
from typing import Dict, Any, Optional

# Import extracted modular components
from .profiler_components import PerformanceMetric, MetricsCollector, PerformanceTracker

logger = logging.getLogger(__name__)


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
        self.metrics_collector = MetricsCollector(max_metrics_history)
        self.performance_tracker = PerformanceTracker(self.metrics_collector.thresholds)
        
        logger.info("Performance Profiler initialized")
    
    # Delegation methods to metrics collector
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start continuous performance monitoring."""
        self.metrics_collector.start_monitoring(interval_seconds)
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        self.metrics_collector.stop_monitoring()
    
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
        """Record API response performance."""
        self.metrics_collector.record_api_response(endpoint, response_time_ms, status_code, data_size_bytes)
    
    def record_cache_performance(self, cache_name: str, hit_rate: float, 
                               total_requests: int, hits: int):
        """Record cache performance metrics."""
        self.metrics_collector.record_cache_performance(cache_name, hit_rate, total_requests, hits)
    
    def record_dashboard_render(self, render_time_ms: float, component_count: int = 0):
        """Record dashboard rendering performance."""
        self.metrics_collector.record_dashboard_render(render_time_ms, component_count)
    
    # Delegation methods to performance tracker
    def get_dashboard_performance_data(self) -> Dict[str, Any]:
        """Get performance data formatted for dashboard display."""
        return self.performance_tracker.get_dashboard_performance_data(
            list(self.metrics_collector.metrics_history),
            self.metrics_collector.current_metrics
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a concise performance summary for logging/debugging."""
        return self.performance_tracker.get_performance_summary(
            list(self.metrics_collector.metrics_history),
            self.metrics_collector.current_metrics,
            self.metrics_collector.monitoring_active
        )


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
            
            self.profiler.metrics_collector.record_component_timing(
                self.component_name,
                duration_ms,
                success=(exc_type is None),
                error_type=exc_type.__name__ if exc_type else None
            )


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