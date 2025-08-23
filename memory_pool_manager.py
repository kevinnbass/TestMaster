#!/usr/bin/env python3
"""
🏗️ MODULE: Memory Pool Manager - Object Allocation Pool Management
================================================================

📋 PURPOSE:
    Memory pool implementation for efficient object allocation and reuse.
    Manages pools of pre-allocated objects to reduce garbage collection overhead.

🎯 CORE FUNCTIONALITY:
    • Memory pool creation and management for frequent allocations
    • Thread-safe object allocation and deallocation
    • Pool statistics tracking and utilization monitoring
    • Automatic pool cleanup and memory reclamation

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 09:05:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract memory pool functionality from memory_management_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for memory pool management
   └─ Impact: Clean separation of pooling logic from orchestration layer

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: threading, collections, typing
🎯 Integration Points: MemoryManager orchestration class
⚡ Performance Notes: Thread-safe implementation with minimal locking overhead
🔒 Security Notes: Safe memory clearing and object lifecycle management

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via memory management validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Standard library threading and collections
📤 Provides: Memory pooling capabilities for efficient object reuse
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import threading
from collections import deque
from typing import Dict, Any, Optional, Set

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

    def clear_pool(self) -> int:
        """Clear all available objects from the pool and return count cleared"""
        with self.lock:
            cleared_count = len(self.available_objects)
            self.available_objects.clear()
            return cleared_count
    
    def resize_pool(self, new_pool_size: int):
        """Resize the pool to a new size"""
        with self.lock:
            if new_pool_size > self.pool_size:
                # Expand pool - add more objects
                additional_objects = new_pool_size - self.pool_size
                for _ in range(additional_objects):
                    obj = bytearray(self.object_size)
                    self.available_objects.append(obj)
            elif new_pool_size < self.pool_size:
                # Shrink pool - remove excess objects
                objects_to_remove = self.pool_size - new_pool_size
                for _ in range(min(objects_to_remove, len(self.available_objects))):
                    self.available_objects.pop()
            
            self.pool_size = new_pool_size
    
    def get_pool_health(self) -> Dict[str, Any]:
        """Get pool health metrics"""
        with self.lock:
            hit_rate = 0.0
            if self.pool_stats['allocations'] > 0:
                hit_rate = (self.pool_stats['pool_hits'] / self.pool_stats['allocations']) * 100
            
            utilization = (len(self.allocated_objects) / self.pool_size) * 100
            
            # Health score based on hit rate and utilization balance
            health_score = (hit_rate * 0.7) + ((100 - abs(utilization - 70)) * 0.3)
            health_score = max(0, min(100, health_score))
            
            return {
                'hit_rate_percent': hit_rate,
                'utilization_percent': utilization,
                'health_score': health_score,
                'status': 'excellent' if health_score > 80 else 
                         'good' if health_score > 60 else 
                         'fair' if health_score > 40 else 'poor',
                'recommendations': self._get_health_recommendations(hit_rate, utilization)
            }
    
    def _get_health_recommendations(self, hit_rate: float, utilization: float) -> List[str]:
        """Get recommendations based on pool health metrics"""
        recommendations = []
        
        if hit_rate < 50:
            recommendations.append("Consider increasing pool size to improve hit rate")
        
        if utilization > 90:
            recommendations.append("Pool is heavily utilized - consider expanding pool size")
        elif utilization < 20:
            recommendations.append("Pool is underutilized - consider reducing pool size")
        
        if self.pool_stats['pool_misses'] > self.pool_stats['pool_hits']:
            recommendations.append("Pool misses exceed hits - pool may be too small for workload")
        
        return recommendations