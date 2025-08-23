#!/usr/bin/env python3
"""
🏗️ MODULE: Memory Management Core - Main Orchestration Engine
============================================================

📋 PURPOSE:
    Main orchestration layer for memory management system coordination.
    Integrates all memory management components and provides unified interface.

🎯 CORE FUNCTIONALITY:
    • Memory management system initialization and lifecycle management
    • Component orchestration between leak detection, GC optimization, and pooling
    • Comprehensive memory monitoring with performance integration
    • Memory cleanup operations and system health reporting

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 09:20:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract main orchestration from memory_management_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for memory management orchestration
   └─ Impact: Clean separation of orchestration from individual optimization components

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: All memory management modules, psutil, logging
🎯 Integration Points: All memory management child modules
⚡ Performance Notes: Orchestration layer with minimal processing overhead
🔒 Security Notes: Safe memory operations with resource cleanup

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via memory management validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: All memory management child modules
📤 Provides: Complete memory management framework capabilities
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import gc
import sys
import time
import psutil
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
from contextlib import contextmanager

# Import all child modules
from memory_management_models import DEFAULT_MEMORY_LIMIT_MB, DEFAULT_CLEANUP_THRESHOLD
from memory_leak_detector import MemoryLeakDetector
from garbage_collection_optimizer import GarbageCollectionOptimizer
from memory_pool_manager import MemoryPool

# Performance monitoring integration
try:
    from performance_monitoring_infrastructure import PerformanceMonitoringSystem, MonitoringConfig
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class MemoryManager:
    """Main memory management orchestrator"""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.leak_detector = MemoryLeakDetector()
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.memory_pools: Dict[int, MemoryPool] = {}
        
        # Memory management settings
        self.max_memory_mb = DEFAULT_MEMORY_LIMIT_MB  # 1GB default limit
        self.cleanup_threshold = DEFAULT_CLEANUP_THRESHOLD  # Trigger cleanup at 80% of limit
        
        # Performance monitoring integration
        self.monitoring_system = None
        if enable_monitoring and MONITORING_AVAILABLE:
            config = MonitoringConfig(
                collection_interval=30.0,  # Less frequent for memory monitoring
                alert_channels=['console'],
                enable_prometheus=False,
                enable_alerting=True,
                alert_thresholds={
                    'memory_usage_mb': self.max_memory_mb * self.cleanup_threshold,
                    'memory_leak_count': 1,
                    'gc_collection_time_ms': 100
                }
            )
            self.monitoring_system = PerformanceMonitoringSystem(config)
            self._setup_memory_metrics()
        
        # Set up logging
        self.logger = logging.getLogger('MemoryManager')
        
        # Start components
        self.running = False
    
    def _setup_memory_metrics(self):
        """Set up memory-specific metrics"""
        if not self.monitoring_system:
            return
        
        # Memory usage metrics
        self.monitoring_system.add_custom_metric(
            "memory_usage_mb",
            lambda: psutil.Process().memory_info().rss / (1024 * 1024),
            unit="megabytes",
            help_text="Process memory usage"
        )
        
        self.monitoring_system.add_custom_metric(
            "memory_leak_count",
            lambda: len([l for l in self.leak_detector.detected_leaks.values() if not l.resolved]),
            unit="count",
            help_text="Number of active memory leaks"
        )
        
        self.monitoring_system.add_custom_metric(
            "gc_objects_total",
            lambda: len(gc.get_objects()),
            unit="count",
            help_text="Total objects tracked by garbage collector"
        )
        
        self.monitoring_system.add_custom_metric(
            "memory_pool_utilization",
            lambda: sum(len(pool.allocated_objects) for pool in self.memory_pools.values()),
            unit="count",
            help_text="Total objects allocated from memory pools"
        )
    
    def start(self):
        """Start memory management system"""
        if self.running:
            return
        
        self.running = True
        
        # Start monitoring system
        if self.monitoring_system:
            self.monitoring_system.start()
        
        # Start leak detection
        self.leak_detector.start_monitoring()
        
        # Optimize GC settings
        self.gc_optimizer.optimize_gc_settings('balanced')
        
        self.logger.info("Memory management system started")
    
    def stop(self):
        """Stop memory management system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        self.leak_detector.stop_monitoring()
        
        if self.monitoring_system:
            self.monitoring_system.stop()
        
        # Reset GC settings
        self.gc_optimizer.reset_gc_settings()
        
        self.logger.info("Memory management system stopped")
    
    def create_memory_pool(self, object_size: int, pool_size: int = 1000) -> MemoryPool:
        """Create a memory pool for frequent allocations"""
        if object_size in self.memory_pools:
            return self.memory_pools[object_size]
        
        pool = MemoryPool(object_size, pool_size)
        self.memory_pools[object_size] = pool
        
        self.logger.info(f"Created memory pool: {object_size} bytes, {pool_size} objects")
        return pool
    
    @contextmanager
    def monitor_memory_usage(self, operation_name: str):
        """Context manager to monitor memory usage of an operation"""
        start_memory = psutil.Process().memory_info().rss
        start_objects = len(gc.get_objects())
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            end_objects = len(gc.get_objects())
            
            memory_delta = (end_memory - start_memory) / (1024 * 1024)  # MB
            object_delta = end_objects - start_objects
            duration = end_time - start_time
            
            # Log significant memory usage
            if memory_delta > 10:  # More than 10MB
                self.logger.warning(f"High memory usage in {operation_name}: {memory_delta:.2f}MB, "
                                  f"{object_delta} objects, {duration:.3f}s")
            
            # Record in monitoring system
            if self.monitoring_system:
                self.monitoring_system.metrics_collector.collect_metric(
                    f"operation_memory_delta_mb",
                    memory_delta,
                    labels={'operation': operation_name}
                )
                
                self.monitoring_system.metrics_collector.collect_metric(
                    f"operation_object_delta",
                    object_delta,
                    labels={'operation': operation_name}
                )
    
    def cleanup_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory cleanup"""
        cleanup_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'before_cleanup': {},
            'after_cleanup': {},
            'actions_performed': []
        }
        
        # Collect before stats
        before_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        before_objects = len(gc.get_objects())
        cleanup_results['before_cleanup'] = {
            'memory_mb': before_memory,
            'objects': before_objects
        }
        
        # Perform garbage collection
        collected_objects = gc.collect()
        cleanup_results['actions_performed'].append(f"GC collected {collected_objects} objects")
        
        # Clear memory pools
        for size, pool in self.memory_pools.items():
            cleared = pool.clear_pool()
            if cleared > 0:
                cleanup_results['actions_performed'].append(f"Cleared pool {size}: {cleared} objects")
        
        # Clear caches
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
            cleanup_results['actions_performed'].append("Cleared type cache")
        
        # Collect after stats
        after_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        after_objects = len(gc.get_objects())
        cleanup_results['after_cleanup'] = {
            'memory_mb': after_memory,
            'objects': after_objects
        }
        
        # Calculate savings
        memory_saved = before_memory - after_memory
        objects_freed = before_objects - after_objects
        cleanup_results['savings'] = {
            'memory_mb': memory_saved,
            'objects': objects_freed
        }
        
        self.logger.info(f"Memory cleanup completed: {memory_saved:.2f}MB freed, {objects_freed} objects")
        return cleanup_results
    
    def generate_memory_report(self) -> str:
        """Generate comprehensive memory management report"""
        # Get memory usage report from leak detector
        memory_report = self.leak_detector.get_memory_usage_report()
        
        # Get GC analysis
        gc_analysis = self.gc_optimizer.analyze_gc_performance()
        
        # Get GC health assessment
        gc_health = self.gc_optimizer.get_gc_health_assessment()
        
        # Get pool statistics
        pool_stats = {}
        for size, pool in self.memory_pools.items():
            pool_stats[f"{size}_bytes"] = pool.get_stats()
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        process_memory = psutil.Process().memory_info()
        
        # Generate report
        report_lines = [
            "MEMORY MANAGEMENT COMPREHENSIVE REPORT",
            "=" * 50,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"System Memory: {system_memory.total / (1024**3):.1f}GB total, "
            f"{system_memory.available / (1024**3):.1f}GB available ({system_memory.percent:.1f}% used)",
            "",
            "PROCESS MEMORY USAGE:",
            f"  Current Memory: {process_memory.rss / (1024**2):.1f}MB",
            f"  Virtual Memory: {process_memory.vms / (1024**2):.1f}MB",
        ]
        
        if 'error' not in memory_report:
            report_lines.append(f"  Peak Memory: {memory_report['current_memory_usage']['peak_memory_mb']:.1f}MB")
            
            if memory_report['current_memory_usage']['traced_memory_mb'] > 0:
                report_lines.append(f"  Traced Memory: {memory_report['current_memory_usage']['traced_memory_mb']:.1f}MB")
        
        report_lines.extend([
            "",
            "GARBAGE COLLECTION STATUS:",
            f"  GC Enabled: {gc_analysis['gc_enabled']}",
            f"  Total Objects: {gc_analysis['total_objects']:,}",
            f"  GC Thresholds: {gc_analysis['current_thresholds']}",
            f"  GC Health Score: {gc_health['health_score']:.1f}/100 ({gc_health['current_status']})",
        ])
        
        # Add GC statistics
        for stat in gc_analysis['gc_statistics']:
            report_lines.append(f"  Gen {stat['generation']}: {stat['collections']} collections, "
                              f"{stat['collected']} collected")
        
        # Memory leak information
        if 'error' not in memory_report:
            report_lines.extend([
                "",
                "MEMORY LEAK DETECTION:",
                f"  Total Leaks Detected: {memory_report['detected_leaks']}",
                f"  Active Leaks: {memory_report['active_leaks']}",
                f"  Monitoring Duration: {memory_report['monitoring_duration_minutes']:.1f} minutes",
                f"  Tracemalloc Enabled: {memory_report['tracemalloc_enabled']}",
            ])
        
        # Memory pools
        if pool_stats:
            report_lines.extend([
                "",
                "MEMORY POOLS:",
            ])
            for pool_name, stats in pool_stats.items():
                pool_health = self.memory_pools[int(pool_name.split('_')[0])].get_pool_health()
                report_lines.extend([
                    f"  {pool_name}:",
                    f"    Utilization: {stats['utilization_percent']:.1f}%",
                    f"    Health Score: {pool_health['health_score']:.1f}/100 ({pool_health['status']})",
                    f"    Pool Hits: {stats['stats']['pool_hits']}",
                    f"    Pool Misses: {stats['stats']['pool_misses']}",
                    f"    Peak Usage: {stats['stats']['peak_usage']}"
                ])
        
        # Top memory allocations
        if 'error' not in memory_report and memory_report.get('top_memory_allocations'):
            report_lines.extend([
                "",
                "TOP MEMORY ALLOCATIONS:",
            ])
            for i, allocation in enumerate(memory_report['top_memory_allocations'][:5], 1):
                report_lines.append(f"  {i}. {allocation['size_mb']:.2f}MB - {allocation['filename'][:80]}...")
        
        # Recommendations
        if gc_health.get('recommendations'):
            report_lines.extend([
                "",
                "RECOMMENDATIONS:",
            ])
            for rec in gc_health['recommendations']:
                report_lines.append(f"  • {rec}")
        
        return "\n".join(report_lines)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'running': self.running,
            'max_memory_mb': self.max_memory_mb,
            'cleanup_threshold': self.cleanup_threshold,
            'monitoring_enabled': self.monitoring_system is not None,
            'components': {
                'leak_detector': {
                    'running': self.leak_detector.running,
                    'snapshots_collected': len(self.leak_detector.memory_snapshots),
                    'leaks_detected': len(self.leak_detector.detected_leaks)
                },
                'gc_optimizer': {
                    'original_thresholds': self.gc_optimizer.original_thresholds,
                    'current_thresholds': gc.get_threshold(),
                    'optimizations_performed': len(self.gc_optimizer.optimization_history)
                },
                'memory_pools': {
                    'active_pools': len(self.memory_pools),
                    'total_allocated': sum(len(pool.allocated_objects) for pool in self.memory_pools.values())
                }
            },
            'current_memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024),
            'total_objects': len(gc.get_objects())
        }
        
        return status
    
    def configure_limits(self, max_memory_mb: int = None, cleanup_threshold: float = None):
        """Configure memory management limits"""
        if max_memory_mb is not None:
            self.max_memory_mb = max_memory_mb
            self.logger.info(f"Set memory limit to {max_memory_mb}MB")
        
        if cleanup_threshold is not None:
            self.cleanup_threshold = cleanup_threshold
            self.logger.info(f"Set cleanup threshold to {cleanup_threshold * 100}%")
        
        # Update monitoring thresholds if available
        if self.monitoring_system and max_memory_mb is not None:
            self.monitoring_system.update_alert_threshold(
                'memory_usage_mb', 
                self.max_memory_mb * self.cleanup_threshold
            )
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """Perform emergency memory cleanup when limits are exceeded"""
        self.logger.warning("Performing emergency memory cleanup")
        
        # Force aggressive garbage collection
        collected = 0
        for _ in range(3):  # Multiple passes
            collected += gc.collect()
        
        # Clear all memory pools
        pools_cleared = 0
        for pool in self.memory_pools.values():
            pools_cleared += pool.clear_pool()
        
        # Clear system caches
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
        
        # Get final memory state
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        cleanup_result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'objects_collected': collected,
            'pools_cleared': pools_cleared,
            'final_memory_mb': final_memory,
            'emergency_cleanup': True
        }
        
        self.logger.warning(f"Emergency cleanup completed: {collected} objects collected, "
                           f"{pools_cleared} pool objects cleared, {final_memory:.1f}MB final memory")
        
        return cleanup_result