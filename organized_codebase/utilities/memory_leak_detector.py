#!/usr/bin/env python3
"""
🏗️ MODULE: Memory Leak Detector - Advanced Leak Detection and Analysis
=====================================================================

📋 PURPOSE:
    Memory leak detection using various techniques including tracemalloc,
    object counting, and trend analysis to identify potential memory leaks.

🎯 CORE FUNCTIONALITY:
    • Continuous memory usage monitoring and snapshot collection
    • Statistical trend analysis for memory leak detection
    • Integration with tracemalloc and pympler for detailed analysis
    • Memory usage reporting and leak severity classification

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 09:10:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract leak detection functionality from memory_management_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for memory leak detection and analysis
   └─ Impact: Clean separation of leak detection logic from memory management orchestration

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: gc, time, psutil, tracemalloc, threading, logging
🎯 Integration Points: MemoryManager orchestration and monitoring systems
⚡ Performance Notes: Efficient background monitoring with configurable intervals
🔒 Security Notes: Safe memory inspection without data exposure

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via memory management validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: psutil, tracemalloc, pympler (optional), memory_management_models
📤 Provides: Memory leak detection and analysis capabilities
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import gc
import time
import psutil
import tracemalloc
import threading
import traceback
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

# Import models
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.memory_management_models import MemorySnapshot, MemoryLeak

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
    
    def resolve_leak(self, leak_id: str) -> bool:
        """Mark a detected leak as resolved"""
        if leak_id in self.detected_leaks:
            self.detected_leaks[leak_id].resolved = True
            self.detected_leaks[leak_id].resolution_timestamp = datetime.now(timezone.utc)
            self.logger.info(f"Memory leak {leak_id} marked as resolved")
            return True
        return False
    
    def get_leak_summary(self) -> Dict[str, Any]:
        """Get summary of all detected leaks"""
        active_leaks = [l for l in self.detected_leaks.values() if not l.resolved]
        resolved_leaks = [l for l in self.detected_leaks.values() if l.resolved]
        
        severity_counts = defaultdict(int)
        for leak in self.detected_leaks.values():
            severity_counts[leak.severity] += 1
        
        return {
            'total_leaks_detected': len(self.detected_leaks),
            'active_leaks': len(active_leaks),
            'resolved_leaks': len(resolved_leaks),
            'severity_breakdown': dict(severity_counts),
            'most_severe_active': max(active_leaks, key=lambda x: x.growth_rate_mb_per_minute) if active_leaks else None,
            'monitoring_enabled': self.running,
            'tracemalloc_enabled': self.tracemalloc_enabled,
            'pympler_available': PYMPLER_AVAILABLE
        }