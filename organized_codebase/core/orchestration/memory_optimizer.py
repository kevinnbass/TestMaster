#!/usr/bin/env python3
"""
Advanced Memory Optimizer
Agent B Hours 80-90: Memory Usage Optimization & Garbage Collection Tuning

Comprehensive memory optimization system with intelligent garbage collection,
object pooling, memory leak detection, and advanced memory management strategies.
"""

import gc
import sys
import weakref
import tracemalloc
import psutil
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Type, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import objgraph
import resource

T = TypeVar('T')

class MemoryOptimizationStrategy(Enum):
    """Memory optimization strategies"""
    OBJECT_POOLING = "object_pooling"
    LAZY_LOADING = "lazy_loading"
    MEMORY_MAPPING = "memory_mapping"
    COMPRESSION = "compression"
    WEAK_REFERENCES = "weak_references"
    GENERATIONAL_GC = "generational_gc"
    INCREMENTAL_GC = "incremental_gc"
    SLOTS_OPTIMIZATION = "slots_optimization"

class GCStrategy(Enum):
    """Garbage collection strategies"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    GENERATIONAL = "generational"
    INCREMENTAL = "incremental"

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    process_memory: int
    available_memory: int
    percent_used: float
    gc_stats: Dict[int, int]
    object_counts: Dict[str, int]
    largest_objects: List[Tuple[str, int]]
    memory_growth_rate: float
    fragmentation_level: float

@dataclass
class MemoryLeak:
    """Detected memory leak information"""
    object_type: str
    growth_rate: float
    instances: int
    total_size: int
    traceback: List[str]
    severity: str
    detected_at: datetime

class ObjectPool(Generic[T]):
    """Generic object pool for memory optimization"""
    
    def __init__(self, factory: Type[T], max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool: deque = deque(maxlen=max_size)
        self.borrowed: Set[T] = set()
        self.stats = {"created": 0, "reused": 0, "returned": 0}
        self.lock = threading.RLock()
    
    def acquire(self, *args, **kwargs) -> T:
        """Acquire object from pool or create new one"""
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
                self.stats["reused"] += 1
            else:
                obj = self.factory(*args, **kwargs)
                self.stats["created"] += 1
            
            self.borrowed.add(obj)
            return obj
    
    def release(self, obj: T):
        """Return object to pool"""
        with self.lock:
            if obj in self.borrowed:
                self.borrowed.remove(obj)
                if len(self.pool) < self.max_size:
                    # Reset object state if it has a reset method
                    if hasattr(obj, 'reset'):
                        obj.reset()
                    self.pool.append(obj)
                    self.stats["returned"] += 1
    
    def clear(self):
        """Clear the pool"""
        with self.lock:
            self.pool.clear()
            self.borrowed.clear()

class MemoryOptimizer:
    """
    Advanced Memory Optimizer
    
    Provides comprehensive memory optimization with intelligent garbage collection,
    object pooling, memory leak detection, and advanced memory management strategies
    for optimal memory utilization in orchestration and processing components.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MemoryOptimizer")
        
        # Memory monitoring configuration
        self.monitoring_enabled = False
        self.monitoring_interval = 5.0  # seconds
        self.leak_detection_interval = 60.0  # seconds
        
        # Memory snapshots and history
        self.memory_snapshots: deque = deque(maxlen=1000)
        self.detected_leaks: List[MemoryLeak] = []
        self.memory_baseline: Optional[MemorySnapshot] = None
        
        # Object pools
        self.object_pools: Dict[str, ObjectPool] = {}
        
        # Weak reference registry
        self.weak_refs: Dict[str, weakref.WeakValueDictionary] = defaultdict(weakref.WeakValueDictionary)
        
        # GC configuration
        self.gc_strategy = GCStrategy.ADAPTIVE
        self.gc_thresholds = {
            0: (700, 10, 10),    # Generation 0
            1: (10, 10, 10),     # Generation 1
            2: (10, 10, 10)      # Generation 2
        }
        
        # Memory optimization settings
        self.optimization_strategies: Set[MemoryOptimizationStrategy] = {
            MemoryOptimizationStrategy.OBJECT_POOLING,
            MemoryOptimizationStrategy.WEAK_REFERENCES,
            MemoryOptimizationStrategy.GENERATIONAL_GC
        }
        
        # Memory limits and thresholds
        self.memory_limit = psutil.virtual_memory().total * 0.8  # 80% of total memory
        self.memory_warning_threshold = 0.7  # 70% of limit
        self.memory_critical_threshold = 0.9  # 90% of limit
        
        # Monitoring threads
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        self.logger.info("Memory optimizer initialized")
    
    async def start_optimization(self):
        """Start memory optimization and monitoring"""
        try:
            self.monitoring_enabled = True
            
            # Start memory tracking
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            
            # Configure garbage collection
            self._configure_gc()
            
            # Take baseline snapshot
            self.memory_baseline = await self._take_memory_snapshot()
            
            # Start monitoring threads
            self.monitoring_thread = threading.Thread(target=self._monitor_memory)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            # Start async monitoring tasks
            asyncio.create_task(self._leak_detection_task())
            asyncio.create_task(self._optimization_task())
            asyncio.create_task(self._gc_tuning_task())
            
            self.logger.info("Memory optimization started")
            
        except Exception as e:
            self.logger.error(f"Failed to start memory optimization: {e}")
    
    async def stop_optimization(self) -> Dict[str, Any]:
        """Stop memory optimization and generate report"""
        try:
            self.monitoring_enabled = False
            self.stop_monitoring.set()
            
            # Wait for monitoring thread
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            # Generate optimization report
            report = await self._generate_optimization_report()
            
            # Stop memory tracking
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            
            self.logger.info("Memory optimization stopped")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to stop memory optimization: {e}")
            return {"error": str(e)}
    
    def _configure_gc(self):
        """Configure garbage collection based on strategy"""
        try:
            if self.gc_strategy == GCStrategy.AGGRESSIVE:
                # Aggressive GC - collect frequently
                gc.set_threshold(500, 5, 5)
                gc.collect(2)  # Full collection
                self.logger.info("Configured aggressive GC strategy")
            
            elif self.gc_strategy == GCStrategy.CONSERVATIVE:
                # Conservative GC - collect less frequently
                gc.set_threshold(1000, 20, 20)
                self.logger.info("Configured conservative GC strategy")
            
            elif self.gc_strategy == GCStrategy.ADAPTIVE:
                # Adaptive GC - adjust based on memory pressure
                self._adaptive_gc_tuning()
                self.logger.info("Configured adaptive GC strategy")
            
            elif self.gc_strategy == GCStrategy.GENERATIONAL:
                # Generational GC - optimize for generation-specific collection
                gc.set_threshold(700, 10, 10)
                self.logger.info("Configured generational GC strategy")
            
            elif self.gc_strategy == GCStrategy.INCREMENTAL:
                # Incremental GC - spread collection over time
                gc.set_threshold(600, 8, 8)
                self.logger.info("Configured incremental GC strategy")
            
            # Enable GC debugging if needed
            gc.set_debug(gc.DEBUG_STATS)
            
        except Exception as e:
            self.logger.error(f"GC configuration failed: {e}")
    
    def _adaptive_gc_tuning(self):
        """Adaptively tune GC based on memory pressure"""
        try:
            memory_info = psutil.virtual_memory()
            memory_pressure = memory_info.percent / 100.0
            
            if memory_pressure < 0.5:
                # Low memory pressure - conservative GC
                gc.set_threshold(1000, 20, 20)
            elif memory_pressure < 0.7:
                # Medium memory pressure - balanced GC
                gc.set_threshold(700, 10, 10)
            else:
                # High memory pressure - aggressive GC
                gc.set_threshold(500, 5, 5)
                gc.collect(2)  # Immediate full collection
            
        except Exception as e:
            self.logger.error(f"Adaptive GC tuning failed: {e}")
    
    def _monitor_memory(self):
        """Background thread for memory monitoring"""
        while not self.stop_monitoring.is_set():
            try:
                # Take memory snapshot
                snapshot = asyncio.run(self._take_memory_snapshot())
                self.memory_snapshots.append(snapshot)
                
                # Check memory thresholds
                self._check_memory_thresholds(snapshot)
                
                # Perform incremental optimization
                if snapshot.percent_used > self.memory_warning_threshold * 100:
                    self._perform_memory_cleanup()
                
                self.stop_monitoring.wait(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                self.stop_monitoring.wait(self.monitoring_interval)
    
    async def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot"""
        try:
            # Get process memory info
            process = psutil.Process()
            process_memory = process.memory_info().rss
            
            # Get system memory info
            memory_info = psutil.virtual_memory()
            
            # Get GC stats
            gc_stats = {}
            for i in range(gc.get_count().__len__()):
                gc_stats[i] = gc.get_count()[i]
            
            # Get object counts by type
            object_counts = defaultdict(int)
            for obj in gc.get_objects():
                object_counts[type(obj).__name__] += 1
            
            # Get largest objects
            largest_objects = []
            if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                for stat in top_stats[:10]:
                    largest_objects.append((str(stat.traceback), stat.size))
            
            # Calculate memory growth rate
            growth_rate = 0.0
            if self.memory_snapshots:
                prev_memory = self.memory_snapshots[-1].process_memory
                time_diff = (datetime.now() - self.memory_snapshots[-1].timestamp).total_seconds()
                if time_diff > 0:
                    growth_rate = (process_memory - prev_memory) / time_diff
            
            # Estimate fragmentation
            fragmentation = self._estimate_fragmentation()
            
            return MemorySnapshot(
                timestamp=datetime.now(),
                process_memory=process_memory,
                available_memory=memory_info.available,
                percent_used=memory_info.percent,
                gc_stats=gc_stats,
                object_counts=dict(object_counts),
                largest_objects=largest_objects,
                memory_growth_rate=growth_rate,
                fragmentation_level=fragmentation
            )
            
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
            raise
    
    def _estimate_fragmentation(self) -> float:
        """Estimate memory fragmentation level"""
        try:
            # Simple fragmentation estimation based on GC stats
            gc_stats = gc.get_stats()
            if gc_stats:
                # Higher uncollectable objects indicate fragmentation
                uncollectable = gc_stats[0].get('uncollectable', 0)
                collected = gc_stats[0].get('collected', 1)
                fragmentation = min(1.0, uncollectable / max(1, collected))
                return fragmentation
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Fragmentation estimation failed: {e}")
            return 0.0
    
    async def _leak_detection_task(self):
        """Async task for memory leak detection"""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(self.leak_detection_interval)
                
                if len(self.memory_snapshots) >= 10:
                    leaks = await self._detect_memory_leaks()
                    self.detected_leaks.extend(leaks)
                    
                    # Clean old leak reports
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.detected_leaks = [l for l in self.detected_leaks if l.detected_at > cutoff_time]
                
            except Exception as e:
                self.logger.error(f"Leak detection error: {e}")
    
    async def _detect_memory_leaks(self) -> List[MemoryLeak]:
        """Detect potential memory leaks"""
        leaks = []
        
        try:
            # Analyze object growth over time
            if len(self.memory_snapshots) < 10:
                return leaks
            
            recent_snapshots = list(self.memory_snapshots)[-10:]
            first_snapshot = recent_snapshots[0]
            last_snapshot = recent_snapshots[-1]
            
            # Check for consistent object growth
            for obj_type, final_count in last_snapshot.object_counts.items():
                initial_count = first_snapshot.object_counts.get(obj_type, 0)
                growth = final_count - initial_count
                
                if growth > 1000:  # Significant growth
                    # Calculate growth rate
                    time_diff = (last_snapshot.timestamp - first_snapshot.timestamp).total_seconds()
                    growth_rate = growth / max(1, time_diff)
                    
                    # Get traceback if available
                    traceback_info = []
                    if tracemalloc.is_tracing():
                        snapshot = tracemalloc.take_snapshot()
                        for stat in snapshot.statistics('traceback'):
                            if obj_type in str(stat):
                                traceback_info = stat.traceback.format()[:5]
                                break
                    
                    leak = MemoryLeak(
                        object_type=obj_type,
                        growth_rate=growth_rate,
                        instances=final_count,
                        total_size=0,  # Would need objgraph for accurate size
                        traceback=traceback_info,
                        severity="high" if growth > 10000 else "medium",
                        detected_at=datetime.now()
                    )
                    leaks.append(leak)
                    
                    self.logger.warning(f"Potential memory leak detected: {obj_type} ({growth} instances)")
            
            # Check for overall memory growth
            memory_growth = last_snapshot.process_memory - first_snapshot.process_memory
            if memory_growth > 100 * 1024 * 1024:  # More than 100MB growth
                self.logger.warning(f"Significant memory growth detected: {memory_growth / 1024 / 1024:.2f}MB")
            
        except Exception as e:
            self.logger.error(f"Memory leak detection failed: {e}")
        
        return leaks
    
    async def _optimization_task(self):
        """Async task for continuous memory optimization"""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(self.monitoring_interval * 6)  # Every 30 seconds
                
                # Apply optimization strategies
                for strategy in self.optimization_strategies:
                    await self._apply_optimization_strategy(strategy)
                
            except Exception as e:
                self.logger.error(f"Optimization task error: {e}")
    
    async def _apply_optimization_strategy(self, strategy: MemoryOptimizationStrategy):
        """Apply specific memory optimization strategy"""
        try:
            if strategy == MemoryOptimizationStrategy.OBJECT_POOLING:
                # Clean up unused pools
                for pool_name, pool in self.object_pools.items():
                    if pool.stats["reused"] < pool.stats["created"] * 0.1:
                        # Pool is underutilized
                        pool.clear()
                        self.logger.info(f"Cleared underutilized pool: {pool_name}")
            
            elif strategy == MemoryOptimizationStrategy.WEAK_REFERENCES:
                # Clean up dead weak references
                for ref_dict in self.weak_refs.values():
                    # WeakValueDictionary automatically removes dead references
                    pass
            
            elif strategy == MemoryOptimizationStrategy.GENERATIONAL_GC:
                # Trigger generational collection
                gc.collect(0)  # Collect youngest generation
            
            elif strategy == MemoryOptimizationStrategy.INCREMENTAL_GC:
                # Perform incremental collection
                gc.collect(1)  # Collect middle generation
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization strategy {strategy}: {e}")
    
    async def _gc_tuning_task(self):
        """Async task for adaptive GC tuning"""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(self.monitoring_interval * 12)  # Every minute
                
                if self.gc_strategy == GCStrategy.ADAPTIVE:
                    self._adaptive_gc_tuning()
                
                # Log GC statistics
                gc_stats = gc.get_stats()
                if gc_stats:
                    self.logger.debug(f"GC stats: {gc_stats[0]}")
                
            except Exception as e:
                self.logger.error(f"GC tuning task error: {e}")
    
    def _check_memory_thresholds(self, snapshot: MemorySnapshot):
        """Check if memory usage exceeds thresholds"""
        memory_usage_ratio = snapshot.process_memory / self.memory_limit
        
        if memory_usage_ratio > self.memory_critical_threshold:
            self.logger.critical(f"Critical memory usage: {memory_usage_ratio:.1%}")
            self._perform_emergency_cleanup()
        elif memory_usage_ratio > self.memory_warning_threshold:
            self.logger.warning(f"High memory usage: {memory_usage_ratio:.1%}")
            self._perform_memory_cleanup()
    
    def _perform_memory_cleanup(self):
        """Perform standard memory cleanup"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear caches
            for pool in self.object_pools.values():
                pool.clear()
            
            self.logger.info("Memory cleanup performed")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
    
    def _perform_emergency_cleanup(self):
        """Perform emergency memory cleanup"""
        try:
            # Aggressive garbage collection
            gc.collect(2)  # Full collection
            
            # Clear all object pools
            self.object_pools.clear()
            
            # Clear weak references
            self.weak_refs.clear()
            
            # Force memory compaction if possible
            if hasattr(gc, 'compact'):
                gc.compact()
            
            self.logger.warning("Emergency memory cleanup performed")
            
        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")
    
    def create_object_pool(self, name: str, factory: Type[T], max_size: int = 100) -> ObjectPool[T]:
        """Create and register an object pool"""
        pool = ObjectPool(factory, max_size)
        self.object_pools[name] = pool
        return pool
    
    def register_weak_reference(self, category: str, key: str, obj: Any):
        """Register object with weak reference"""
        self.weak_refs[category][key] = obj
    
    def optimize_slots(self, cls: Type) -> Type:
        """Optimize class memory usage with __slots__"""
        if not hasattr(cls, '__slots__'):
            # Get all attributes from class
            attrs = [attr for attr in dir(cls) if not attr.startswith('_')]
            
            # Create new class with slots
            class OptimizedClass(cls):
                __slots__ = attrs
            
            OptimizedClass.__name__ = f"Optimized{cls.__name__}"
            return OptimizedClass
        return cls
    
    async def _generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory optimization report"""
        try:
            current_snapshot = await self._take_memory_snapshot()
            
            # Calculate memory savings
            memory_saved = 0
            if self.memory_baseline:
                memory_saved = self.memory_baseline.process_memory - current_snapshot.process_memory
            
            # Calculate pool efficiency
            pool_stats = {}
            for name, pool in self.object_pools.items():
                efficiency = 0.0
                if pool.stats["created"] > 0:
                    efficiency = pool.stats["reused"] / pool.stats["created"] * 100
                pool_stats[name] = {
                    "created": pool.stats["created"],
                    "reused": pool.stats["reused"],
                    "efficiency": efficiency
                }
            
            report = {
                "optimization_summary": {
                    "duration": (datetime.now() - self.memory_baseline.timestamp).total_seconds() if self.memory_baseline else 0,
                    "memory_saved_mb": memory_saved / 1024 / 1024,
                    "current_memory_mb": current_snapshot.process_memory / 1024 / 1024,
                    "peak_memory_mb": max(s.process_memory for s in self.memory_snapshots) / 1024 / 1024 if self.memory_snapshots else 0
                },
                "gc_statistics": current_snapshot.gc_stats,
                "object_pools": pool_stats,
                "memory_leaks": [
                    {
                        "type": leak.object_type,
                        "growth_rate": leak.growth_rate,
                        "severity": leak.severity
                    }
                    for leak in self.detected_leaks[:5]
                ],
                "optimization_strategies": [s.value for s in self.optimization_strategies],
                "fragmentation_level": current_snapshot.fragmentation_level,
                "recommendations": self._generate_recommendations(current_snapshot)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization report: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, snapshot: MemorySnapshot) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        # Check for high fragmentation
        if snapshot.fragmentation_level > 0.3:
            recommendations.append("High memory fragmentation detected - consider memory compaction")
        
        # Check for memory leaks
        if self.detected_leaks:
            recommendations.append(f"Memory leaks detected in {len(self.detected_leaks)} object types")
        
        # Check object pool efficiency
        for name, pool in self.object_pools.items():
            if pool.stats["created"] > 0:
                efficiency = pool.stats["reused"] / pool.stats["created"]
                if efficiency < 0.3:
                    recommendations.append(f"Low efficiency in object pool '{name}' - consider removing")
        
        # Check memory growth rate
        if snapshot.memory_growth_rate > 1024 * 1024:  # More than 1MB/s
            recommendations.append("High memory growth rate - investigate allocations")
        
        return recommendations
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        if not self.memory_snapshots:
            return {}
        
        current = self.memory_snapshots[-1]
        return {
            "current_memory_mb": current.process_memory / 1024 / 1024,
            "available_memory_mb": current.available_memory / 1024 / 1024,
            "percent_used": current.percent_used,
            "gc_collections": current.gc_stats,
            "object_count": sum(current.object_counts.values()),
            "fragmentation": current.fragmentation_level
        }