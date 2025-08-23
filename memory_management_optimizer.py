#!/usr/bin/env python3
"""
AGENT BETA - MEMORY MANAGEMENT OPTIMIZER
Phase 1, Hours 15-20: Memory Management & Garbage Collection
===========================================================

Advanced memory management system with leak detection, garbage collection tuning,
memory pool implementation, and comprehensive memory usage monitoring.

Created: 2025-08-23 02:50:00 UTC
Agent: Beta (Performance Optimization Specialist)
Phase: 1 (Hours 15-20)
"""

import os
import sys
import gc
import time
import json
import psutil
import threading
import weakref
import tracemalloc
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
import logging
import mmap
import pickle
import platform

# Unix-specific resource module
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False  # Windows doesn't have resource module

# Memory profiling
try:
    import pympler
    from pympler import tracker, muppy, summary
    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False

# Memory monitoring integration
try:
    from performance_monitoring_infrastructure import PerformanceMonitoringSystem, MonitoringConfig
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time"""
    timestamp: datetime
    process_memory_mb: float
    virtual_memory_mb: float
    peak_memory_mb: float
    gc_collections: Dict[int, int]
    object_counts: Dict[str, int]
    memory_blocks: int = 0
    memory_size: int = 0
    tracemalloc_top: List[Dict] = None
    
    def __post_init__(self):
        if self.tracemalloc_top is None:
            self.tracemalloc_top = []

@dataclass
class MemoryLeak:
    """Detected memory leak information"""
    leak_id: str
    detection_timestamp: datetime
    object_type: str
    object_count: int
    memory_size_mb: float
    growth_rate_mb_per_minute: float
    stack_trace: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class GCStats:
    """Garbage collection statistics"""
    generation: int
    collections: int
    collected: int
    uncollectable: int
    threshold: Tuple[int, int, int]

class MemoryPool:
    """Memory pool for frequent allocations"""
    
    def __init__(self, object_size: int, pool_size: int = 1000):
        self.object_size = object_size
        self.pool_size = pool_size
        self.available_objects: deque = deque()
        self.allocated_objects: Set = set()
        self.pool_stats = {
            'allocations': 0,
            'deallocations': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'peak_usage': 0
        }
        
        # Pre-allocate objects
        self._initialize_pool()
        
        # Thread safety
        self.lock = threading.Lock()
        
    def _initialize_pool(self):
        """Initialize the memory pool with pre-allocated objects"""
        for _ in range(self.pool_size):
            # Create a byte array of specified size
            obj = bytearray(self.object_size)
            self.available_objects.append(obj)
    
    def allocate(self) -> Optional[bytearray]:
        """Allocate an object from the pool"""
        with self.lock:
            if self.available_objects:
                obj = self.available_objects.popleft()
                self.allocated_objects.add(id(obj))
                self.pool_stats['allocations'] += 1
                self.pool_stats['pool_hits'] += 1
                self.pool_stats['peak_usage'] = max(
                    self.pool_stats['peak_usage'], 
                    len(self.allocated_objects)
                )
                return obj
            else:
                # Pool exhausted, create new object
                obj = bytearray(self.object_size)
                self.allocated_objects.add(id(obj))
                self.pool_stats['allocations'] += 1
                self.pool_stats['pool_misses'] += 1
                return obj
    
    def deallocate(self, obj: bytearray):
        """Return an object to the pool"""
        with self.lock:
            obj_id = id(obj)
            if obj_id in self.allocated_objects:
                self.allocated_objects.remove(obj_id)
                
                # Clear the object and return to pool if there's space
                if len(self.available_objects) < self.pool_size:
                    # Clear the memory
                    for i in range(len(obj)):
                        obj[i] = 0
                    self.available_objects.append(obj)
                
                self.pool_stats['deallocations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            return {
                'object_size': self.object_size,
                'pool_size': self.pool_size,
                'available_objects': len(self.available_objects),
                'allocated_objects': len(self.allocated_objects),
                'utilization_percent': (len(self.allocated_objects) / self.pool_size) * 100,
                'stats': self.pool_stats.copy()
            }

class MemoryLeakDetector:
    """Detects memory leaks using various techniques"""
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.memory_snapshots: deque = deque(maxlen=100)
        self.detected_leaks: Dict[str, MemoryLeak] = {}
        self.object_trackers: Dict[type, int] = defaultdict(int)
        self.running = False
        self.detector_thread = None
        
        # Configure tracemalloc if available
        self.tracemalloc_enabled = False
        if hasattr(tracemalloc, 'start'):
            try:
                tracemalloc.start(10)  # Keep 10 frames
                self.tracemalloc_enabled = True
            except Exception:
                pass
        
        # Pympler tracker if available
        self.pympler_tracker = None
        if PYMPLER_AVAILABLE:
            self.pympler_tracker = tracker.SummaryTracker()
        
        self.logger = logging.getLogger('MemoryLeakDetector')
    
    def _get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB (cross-platform)"""
        if RESOURCE_AVAILABLE:
            try:
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert from KB
            except Exception:
                pass
        
        # Fallback for Windows - use current memory as approximation
        return psutil.Process().memory_info().rss / (1024 * 1024)
        
    def start_monitoring(self):
        """Start memory leak monitoring"""
        if self.running:
            return
        
        self.running = True
        self.detector_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.detector_thread.start()
        self.logger.info(f"Started memory leak monitoring (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop memory leak monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.detector_thread:
            self.detector_thread.join(timeout=10)
        
        self.logger.info("Stopped memory leak monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._take_memory_snapshot()
                self._analyze_memory_trends()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _take_memory_snapshot(self):
        """Take a memory usage snapshot"""
        try:
            # Process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # GC statistics
            gc_stats = {}
            for i in range(3):
                stats = gc.get_stats()[i] if i < len(gc.get_stats()) else {}
                gc_stats[i] = stats.get('collections', 0)
            
            # Object counts using gc
            object_counts = {}
            for obj_type in [dict, list, tuple, set, str]:
                object_counts[obj_type.__name__] = len([obj for obj in gc.get_objects() 
                                                      if type(obj) is obj_type])
            
            # Tracemalloc information
            tracemalloc_top = []
            if self.tracemalloc_enabled:
                try:
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')[:10]
                    
                    for stat in top_stats:
                        tracemalloc_top.append({
                            'filename': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                            'size_mb': stat.size / (1024 * 1024),
                            'count': stat.count
                        })
                except Exception as e:
                    self.logger.debug(f"Tracemalloc error: {e}")
            
            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=datetime.now(timezone.utc),
                process_memory_mb=memory_info.rss / (1024 * 1024),
                virtual_memory_mb=memory_info.vms / (1024 * 1024),
                peak_memory_mb=self._get_peak_memory_mb(),
                gc_collections=gc_stats,
                object_counts=object_counts,
                tracemalloc_top=tracemalloc_top
            )
            
            if self.tracemalloc_enabled:
                try:
                    current_size, peak_size = tracemalloc.get_traced_memory()
                    snapshot.memory_blocks = len(tracemalloc.take_snapshot().statistics('filename'))
                    snapshot.memory_size = current_size
                except Exception:
                    pass
            
            self.memory_snapshots.append(snapshot)
            
        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {e}")
    
    def _analyze_memory_trends(self):
        """Analyze memory trends for leak detection"""
        if len(self.memory_snapshots) < 5:
            return  # Need more data points
        
        recent_snapshots = list(self.memory_snapshots)[-10:]  # Last 10 snapshots
        
        # Analyze memory growth
        memory_values = [s.process_memory_mb for s in recent_snapshots]
        if len(memory_values) >= 3:
            # Simple linear regression to detect upward trend
            growth_rate = self._calculate_growth_rate(memory_values)
            
            if growth_rate > 1.0:  # Growing by more than 1MB per interval
                self._detect_memory_leak('process_memory', growth_rate, recent_snapshots)
        
        # Analyze object count growth
        for obj_type in ['dict', 'list', 'tuple']:
            if obj_type in recent_snapshots[0].object_counts:
                obj_counts = [s.object_counts.get(obj_type, 0) for s in recent_snapshots]
                growth_rate = self._calculate_growth_rate(obj_counts)
                
                if growth_rate > 100:  # Growing by more than 100 objects per interval
                    estimated_size_mb = growth_rate * 0.001  # Rough estimate
                    self._detect_memory_leak(f'{obj_type}_objects', estimated_size_mb, recent_snapshots)
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate growth rate using simple linear regression"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        # Calculate slope (growth rate)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _detect_memory_leak(self, leak_type: str, growth_rate: float, snapshots: List[MemorySnapshot]):
        """Detect and record a potential memory leak"""
        leak_id = f"{leak_type}_{int(time.time())}"
        
        # Calculate severity
        if growth_rate > 10:
            severity = 'critical'
        elif growth_rate > 5:
            severity = 'high'
        elif growth_rate > 2:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Get stack trace
        stack_trace = traceback.format_stack()
        
        # Create memory leak record
        leak = MemoryLeak(
            leak_id=leak_id,
            detection_timestamp=datetime.now(timezone.utc),
            object_type=leak_type,
            object_count=len(snapshots),
            memory_size_mb=snapshots[-1].process_memory_mb - snapshots[0].process_memory_mb,
            growth_rate_mb_per_minute=growth_rate * (60 / self.check_interval),
            stack_trace=[line.strip() for line in stack_trace[-5:]],  # Last 5 frames
            severity=severity
        )
        
        self.detected_leaks[leak_id] = leak
        
        # Log the leak
        self.logger.warning(f"Memory leak detected: {leak_type} growing at {growth_rate:.2f} MB/interval")
        
        return leak
    
    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report"""
        if not self.memory_snapshots:
            return {'error': 'No memory snapshots available'}
        
        latest_snapshot = self.memory_snapshots[-1]
        
        # Calculate trends if we have enough data
        trends = {}
        if len(self.memory_snapshots) >= 5:
            memory_values = [s.process_memory_mb for s in list(self.memory_snapshots)[-10:]]
            trends['memory_growth_rate'] = self._calculate_growth_rate(memory_values)
            
            # Object growth trends
            for obj_type in ['dict', 'list', 'tuple']:
                obj_values = [s.object_counts.get(obj_type, 0) for s in list(self.memory_snapshots)[-10:]]
                trends[f'{obj_type}_growth_rate'] = self._calculate_growth_rate(obj_values)
        
        return {
            'current_memory_usage': {
                'process_memory_mb': latest_snapshot.process_memory_mb,
                'virtual_memory_mb': latest_snapshot.virtual_memory_mb,
                'peak_memory_mb': latest_snapshot.peak_memory_mb,
                'memory_blocks': latest_snapshot.memory_blocks,
                'traced_memory_mb': latest_snapshot.memory_size / (1024 * 1024) if latest_snapshot.memory_size else 0
            },
            'garbage_collection': {
                'collections': latest_snapshot.gc_collections,
                'total_objects': len(gc.get_objects()),
                'gc_thresholds': gc.get_threshold()
            },
            'object_counts': latest_snapshot.object_counts,
            'memory_trends': trends,
            'detected_leaks': len(self.detected_leaks),
            'active_leaks': len([l for l in self.detected_leaks.values() if not l.resolved]),
            'tracemalloc_enabled': self.tracemalloc_enabled,
            'top_memory_allocations': latest_snapshot.tracemalloc_top,
            'snapshot_count': len(self.memory_snapshots),
            'monitoring_duration_minutes': (
                (latest_snapshot.timestamp - self.memory_snapshots[0].timestamp).total_seconds() / 60
                if len(self.memory_snapshots) > 1 else 0
            )
        }

class GarbageCollectionOptimizer:
    """Optimizes Python garbage collection settings"""
    
    def __init__(self):
        self.original_thresholds = gc.get_threshold()
        self.gc_stats_history: List[Dict] = []
        self.optimization_history: List[Dict] = []
        
        self.logger = logging.getLogger('GarbageCollectionOptimizer')
    
    def analyze_gc_performance(self) -> Dict[str, Any]:
        """Analyze current garbage collection performance"""
        # Get current GC statistics
        gc_stats = []
        for i in range(3):
            if i < len(gc.get_stats()):
                stats = gc.get_stats()[i]
                gc_stats.append(GCStats(
                    generation=i,
                    collections=stats.get('collections', 0),
                    collected=stats.get('collected', 0),
                    uncollectable=stats.get('uncollectable', 0),
                    threshold=gc.get_threshold()
                ))
        
        # Count objects by generation
        object_counts = [0, 0, 0]
        try:
            for obj in gc.get_objects():
                gen = gc.get_referents(obj)
                if len(gen) < 3:
                    object_counts[0] += 1
                elif len(gen) < 10:
                    object_counts[1] += 1
                else:
                    object_counts[2] += 1
        except Exception:
            # Fallback to simple counting
            object_counts = [len(gc.get_objects()), 0, 0]
        
        analysis = {
            'gc_statistics': [asdict(stat) for stat in gc_stats],
            'object_counts_by_generation': object_counts,
            'total_objects': len(gc.get_objects()),
            'gc_enabled': gc.isenabled(),
            'current_thresholds': gc.get_threshold(),
            'original_thresholds': self.original_thresholds
        }
        
        return analysis
    
    def optimize_gc_settings(self, target_performance: str = 'balanced') -> Dict[str, Any]:
        """Optimize garbage collection settings"""
        current_analysis = self.analyze_gc_performance()
        
        # Record baseline
        baseline = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis': current_analysis,
            'target': target_performance
        }
        
        # Determine optimal thresholds based on target
        if target_performance == 'memory':
            # Optimize for low memory usage (more frequent GC)
            new_thresholds = (500, 8, 8)
        elif target_performance == 'speed':
            # Optimize for speed (less frequent GC)
            new_thresholds = (2000, 15, 15)
        else:  # balanced
            # Balanced approach
            total_objects = current_analysis['total_objects']
            if total_objects < 10000:
                new_thresholds = (700, 10, 10)
            elif total_objects < 50000:
                new_thresholds = (1000, 12, 12)
            else:
                new_thresholds = (1500, 15, 15)
        
        # Apply new thresholds
        gc.set_threshold(*new_thresholds)
        
        # Force garbage collection to clear current state
        collected = gc.collect()
        
        optimization_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'baseline': baseline,
            'applied_thresholds': new_thresholds,
            'objects_collected': collected,
            'target_performance': target_performance
        }
        
        self.optimization_history.append(optimization_record)
        
        self.logger.info(f"Optimized GC settings for {target_performance}: {new_thresholds}")
        self.logger.info(f"Collected {collected} objects during optimization")
        
        return optimization_record
    
    def benchmark_gc_performance(self) -> Dict[str, Any]:
        """Benchmark garbage collection performance"""
        results = {}
        
        # Test different threshold settings
        test_configurations = [
            ('conservative', (500, 8, 8)),
            ('default', (700, 10, 10)),
            ('aggressive', (2000, 15, 15))
        ]
        
        original_thresholds = gc.get_threshold()
        
        for config_name, thresholds in test_configurations:
            # Set test thresholds
            gc.set_threshold(*thresholds)
            gc.collect()  # Clear state
            
            # Create test objects and measure GC performance
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss
            
            # Create objects that will need garbage collection
            test_objects = []
            for i in range(1000):
                obj = {'data': [j for j in range(100)], 'refs': []}
                # Create circular references
                obj['refs'].append(obj)
                test_objects.append(obj)
            
            # Force garbage collection and measure
            gc_start = time.perf_counter()
            collected = gc.collect()
            gc_time = time.perf_counter() - gc_start
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            results[config_name] = {
                'thresholds': thresholds,
                'total_time': end_time - start_time,
                'gc_time': gc_time,
                'objects_collected': collected,
                'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
                'gc_overhead_percent': (gc_time / (end_time - start_time)) * 100
            }
            
            # Cleanup
            del test_objects
            gc.collect()
        
        # Restore original thresholds
        gc.set_threshold(*original_thresholds)
        
        self.logger.info("GC performance benchmark completed")
        return results
    
    def reset_gc_settings(self):
        """Reset GC settings to original values"""
        gc.set_threshold(*self.original_thresholds)
        self.logger.info(f"Reset GC settings to original: {self.original_thresholds}")

class MemoryManager:
    """Main memory management orchestrator"""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.leak_detector = MemoryLeakDetector()
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.memory_pools: Dict[int, MemoryPool] = {}
        
        # Memory management settings
        self.max_memory_mb = 1024  # 1GB default limit
        self.cleanup_threshold = 0.8  # Trigger cleanup at 80% of limit
        
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
            if pool.available_objects:
                cleared = len(pool.available_objects)
                pool.available_objects.clear()
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
            f"  Peak Memory: {memory_report['current_memory_usage']['peak_memory_mb']:.1f}MB",
        ]
        
        if memory_report['current_memory_usage']['traced_memory_mb'] > 0:
            report_lines.append(f"  Traced Memory: {memory_report['current_memory_usage']['traced_memory_mb']:.1f}MB")
        
        report_lines.extend([
            "",
            "GARBAGE COLLECTION STATUS:",
            f"  GC Enabled: {gc_analysis['gc_enabled']}",
            f"  Total Objects: {gc_analysis['total_objects']:,}",
            f"  GC Thresholds: {gc_analysis['current_thresholds']}",
        ])
        
        # Add GC statistics
        for stat in gc_analysis['gc_statistics']:
            report_lines.append(f"  Gen {stat['generation']}: {stat['collections']} collections, "
                              f"{stat['collected']} collected")
        
        # Memory leak information
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
                report_lines.extend([
                    f"  {pool_name}:",
                    f"    Utilization: {stats['utilization_percent']:.1f}%",
                    f"    Pool Hits: {stats['stats']['pool_hits']}",
                    f"    Pool Misses: {stats['stats']['pool_misses']}",
                    f"    Peak Usage: {stats['stats']['peak_usage']}"
                ])
        
        # Top memory allocations
        if memory_report.get('top_memory_allocations'):
            report_lines.extend([
                "",
                "TOP MEMORY ALLOCATIONS:",
            ])
            for i, allocation in enumerate(memory_report['top_memory_allocations'][:5], 1):
                report_lines.append(f"  {i}. {allocation['size_mb']:.2f}MB - {allocation['filename'][:80]}...")
        
        return "\n".join(report_lines)

def main():
    """Main function to demonstrate memory management"""
    print("AGENT BETA - Memory Management Optimizer")
    print("=" * 50)
    
    # Initialize memory manager
    memory_manager = MemoryManager(enable_monitoring=True)
    
    try:
        # Start memory management
        memory_manager.start()
        
        # Demonstrate memory pool usage
        print("\nCreating memory pools...")
        small_pool = memory_manager.create_memory_pool(1024, 500)   # 1KB objects
        medium_pool = memory_manager.create_memory_pool(8192, 100)  # 8KB objects
        
        # Test memory usage monitoring
        print("\nTesting memory usage monitoring...")
        
        with memory_manager.monitor_memory_usage("memory_intensive_operation"):
            # Simulate memory-intensive operation
            data = []
            for i in range(1000):
                # Use memory pool for some allocations
                if i % 2 == 0:
                    obj = small_pool.allocate()
                    if obj:
                        data.append(obj)
                else:
                    # Regular allocation
                    data.append([j for j in range(100)])
            
            # Create some circular references to test GC
            circular_refs = []
            for i in range(100):
                obj = {'id': i, 'refs': []}
                obj['refs'].append(obj)  # Circular reference
                circular_refs.append(obj)
        
        # Wait for leak detection to collect some data
        print("\nCollecting memory usage data...")
        time.sleep(65)  # Wait for at least one leak detection cycle
        
        # Demonstrate GC optimization
        print("\nOptimizing garbage collection...")
        gc_results = memory_manager.gc_optimizer.optimize_gc_settings('balanced')
        print(f"Objects collected during optimization: {gc_results['objects_collected']}")
        
        # Benchmark GC performance
        print("\nBenchmarking GC performance...")
        benchmark_results = memory_manager.gc_optimizer.benchmark_gc_performance()
        
        print("\nGC Benchmark Results:")
        for config, results in benchmark_results.items():
            print(f"  {config.upper()}: {results['gc_time']:.4f}s GC time, "
                  f"{results['objects_collected']} objects collected, "
                  f"{results['gc_overhead_percent']:.2f}% overhead")
        
        # Perform memory cleanup
        print("\nPerforming memory cleanup...")
        cleanup_results = memory_manager.cleanup_memory()
        print(f"Memory saved: {cleanup_results['savings']['memory_mb']:.2f}MB")
        print(f"Objects freed: {cleanup_results['savings']['objects']}")
        
        # Generate comprehensive report
        print("\nGenerating memory management report...")
        report = memory_manager.generate_memory_report()
        print("\n" + report)
        
        print("\nMemory management optimization completed successfully!")
        
    except Exception as e:
        print(f"Memory management failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        memory_manager.stop()

if __name__ == "__main__":
    main()