#!/usr/bin/env python3
"""
🏗️ MODULE: Memory Management Models - Data Structures and Configuration
====================================================================

📋 PURPOSE:
    Core data structures and configuration classes for memory management system.
    Contains all dataclasses, type definitions, and model objects.

🎯 CORE FUNCTIONALITY:
    • Memory snapshot data structures for tracking system state
    • Memory leak detection data models with severity classification
    • Garbage collection statistics structures and threshold configuration
    • Type definitions and constants for memory management operations

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 09:00:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract data models from memory_management_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for memory management data structures
   └─ Impact: Clean separation of data models from business logic components

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: datetime, typing, dataclasses
🎯 Integration Points: All memory management child modules
⚡ Performance Notes: Lightweight data structures with minimal overhead
🔒 Security Notes: Immutable data structures where appropriate

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via memory management validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Standard library only
📤 Provides: Core data structures for memory management system
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

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

# Constants for memory management
MEMORY_LEAK_SEVERITY_LEVELS = ['low', 'medium', 'high', 'critical']
GC_TARGET_PERFORMANCE_MODES = ['memory', 'speed', 'balanced']
DEFAULT_POOL_SIZES = {
    'small': (1024, 500),    # 1KB objects, 500 pool size
    'medium': (8192, 100),   # 8KB objects, 100 pool size  
    'large': (65536, 20)     # 64KB objects, 20 pool size
}

# Memory thresholds and limits
DEFAULT_MEMORY_LIMIT_MB = 1024
DEFAULT_CLEANUP_THRESHOLD = 0.8
DEFAULT_LEAK_CHECK_INTERVAL = 60.0
DEFAULT_MONITORING_INTERVAL = 30.0

# GC optimization presets
GC_THRESHOLDS_PRESETS = {
    'memory': (500, 8, 8),      # Optimize for low memory usage (more frequent GC)
    'speed': (2000, 15, 15),    # Optimize for speed (less frequent GC)
    'balanced_small': (700, 10, 10),     # Balanced approach for small object counts
    'balanced_medium': (1000, 12, 12),   # Balanced approach for medium object counts
    'balanced_large': (1500, 15, 15)     # Balanced approach for large object counts
}