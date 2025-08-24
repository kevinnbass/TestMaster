#!/usr/bin/env python3
"""
🏗️ MODULE: Caching Models - Data Structures and Configuration
=============================================================

📋 PURPOSE:
    Core data structures, enums, and configuration classes for advanced caching system.
    Contains all dataclasses, cache strategies, and model objects.

🎯 CORE FUNCTIONALITY:
    • Cache strategy and layer enumeration definitions
    • Cache configuration structures with Redis and performance settings
    • Cache entry data models with metadata tracking
    • Type definitions and constants for caching operations

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 10:00:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract data models from advanced_caching_architecture.py via STEELCLAD
   └─ Changes: Created dedicated module for caching system data structures
   └─ Impact: Clean separation of data models from caching business logic

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: datetime, typing, dataclasses, enum
🎯 Integration Points: All caching system child modules
⚡ Performance Notes: Lightweight data structures with minimal overhead
🔒 Security Notes: Immutable data structures where appropriate for cache integrity

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via caching system validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Standard library only
📤 Provides: Core data structures for advanced caching system
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    FIFO = "fifo"         # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns

class CacheLayer(Enum):
    """Cache layer types"""
    MEMORY = "memory"     # In-memory L1 cache
    REDIS = "redis"       # Redis L2 cache
    DISK = "disk"         # Disk-based L3 cache
    HYBRID = "hybrid"     # Intelligent multi-layer

@dataclass
class CacheConfig:
    """Configuration for advanced caching system"""
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_pool_size: int = 20
    
    # Cache strategy configuration
    default_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    default_ttl: int = 3600  # 1 hour default TTL
    max_memory_cache_size: int = 1000  # Max items in memory
    max_redis_memory: str = "500mb"
    
    # Performance configuration
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress objects larger than 1KB
    enable_async: bool = True
    batch_size: int = 100
    
    # Monitoring integration
    enable_performance_tracking: bool = True
    metrics_collection_interval: float = 30.0
    enable_cache_warming: bool = True
    cache_warming_threshold: float = 0.8  # Start warming when hit ratio drops below 80%
    
    # Advanced features
    enable_predictive_caching: bool = True
    enable_adaptive_ttl: bool = True
    enable_distributed_invalidation: bool = True
    enable_cache_partitioning: bool = True

@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[int]
    size_bytes: int
    compression_used: bool = False
    layer: CacheLayer = CacheLayer.MEMORY
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)
    
    @property
    def age_seconds(self) -> float:
        """Age of cache entry in seconds"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl is None:
            return False
        return self.age_seconds > self.ttl
    
    def update_access(self):
        """Update access metadata"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

# Cache configuration presets
CACHE_CONFIG_PRESETS = {
    'development': CacheConfig(
        redis_host="localhost",
        max_memory_cache_size=100,
        max_redis_memory="50mb",
        default_ttl=1800,  # 30 minutes
        enable_compression=False,
        enable_performance_tracking=False
    ),
    'production': CacheConfig(
        redis_host="redis-cluster",
        redis_ssl=True,
        max_memory_cache_size=10000,
        max_redis_memory="2gb",
        default_ttl=3600,  # 1 hour
        enable_compression=True,
        enable_performance_tracking=True,
        enable_predictive_caching=True
    ),
    'high_performance': CacheConfig(
        redis_host="redis-cluster",
        redis_pool_size=50,
        max_memory_cache_size=5000,
        max_redis_memory="1gb",
        default_ttl=7200,  # 2 hours
        enable_compression=True,
        compression_threshold=2048,
        batch_size=200,
        enable_cache_warming=True,
        enable_adaptive_ttl=True
    )
}

# Constants for cache operations
DEFAULT_CACHE_TTL = 3600
MAX_CACHE_KEY_LENGTH = 250
DEFAULT_COMPRESSION_THRESHOLD = 1024
DEFAULT_BATCH_SIZE = 100
DEFAULT_POOL_SIZE = 20

# Cache performance thresholds
CACHE_HIT_RATIO_WARNING_THRESHOLD = 0.7
CACHE_HIT_RATIO_CRITICAL_THRESHOLD = 0.5
CACHE_MEMORY_WARNING_THRESHOLD = 0.8
CACHE_MEMORY_CRITICAL_THRESHOLD = 0.9

# Redis connection defaults
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_REDIS_SSL = False