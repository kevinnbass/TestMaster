#!/usr/bin/env python3
"""
Perfected Cache Manager
======================

Optimized high-performance caching layer replacing the 636-line intelligent_cache.py.
Enterprise-grade caching with thread-safety, persistence, and advanced features.

Key improvements over intelligent_cache.py:
- Modular design with focused classes under 300 lines each
- Enhanced thread safety with minimal locking
- Background cleanup and maintenance tasks
- Advanced analytics and monitoring
- Event-driven cache updates
- Multi-tier caching (memory + disk)
- Intelligent compression algorithms
- Performance monitoring and optimization

Author: Agent E - Infrastructure Consolidation
"""

import os
import json
import time
import hashlib
import sqlite3
import pickle
import zlib
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class CacheEvent(Enum):
    """Cache events for listeners."""
    SET = "set"
    GET = "get"
    DELETE = "delete"
    EVICT = "evict"
    EXPIRE = "expire"
    CLEAR = "clear"


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compressions: int = 0
    decompressions: int = 0
    disk_reads: int = 0
    disk_writes: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    def miss_rate(self) -> float:
        """Calculate miss rate percentage."""
        return 100.0 - self.hit_rate()


@dataclass
class CacheEntry(Generic[T]):
    """Thread-safe cache entry with metadata."""
    key: str
    value: T
    created_at: datetime
    accessed_at: datetime
    access_count: int = 1
    ttl_seconds: int = 3600
    size_bytes: int = 0
    compressed: bool = False
    tier: str = "memory"  # memory, disk, hybrid
    metadata: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        with self._lock:
            if self.ttl_seconds <= 0:
                return False
            expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
            return datetime.now() > expiry_time
    
    def touch(self) -> None:
        """Update access statistics atomically."""
        with self._lock:
            self.accessed_at = datetime.now()
            self.access_count += 1
    
    def estimate_size(self) -> int:
        """Estimate entry memory footprint."""
        try:
            return len(pickle.dumps(self.value))
        except:
            return 1024  # Default estimate


class MemoryCache:
    """High-performance in-memory cache tier."""
    
    def __init__(self, max_size_mb: int = 256, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from memory cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                self.current_size -= entry.size_bytes
                return None
            
            entry.touch()
            if self.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
            
            return entry
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set entry in memory cache."""
        with self.lock:
            # Remove old entry if exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size -= old_entry.size_bytes
            
            # Evict entries if needed
            while (self.current_size + entry.size_bytes > self.max_size_bytes 
                   and self.cache):
                self._evict_one()
            
            self.cache[key] = entry
            self.current_size += entry.size_bytes
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from memory cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_size -= entry.size_bytes
                del self.cache[key]
                return True
            return False
    
    def _evict_one(self) -> Optional[str]:
        """Evict one entry based on strategy."""
        if not self.cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            key = next(iter(self.cache))
        elif self.strategy == CacheStrategy.LFU:
            key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].access_count)
        elif self.strategy == CacheStrategy.FIFO:
            key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].created_at)
        else:  # TTL or ADAPTIVE
            expired = [k for k, v in self.cache.items() if v.is_expired()]
            if expired:
                key = expired[0]
            else:
                key = next(iter(self.cache))
        
        self.delete(key)
        return key


class DiskCache:
    """Persistent disk-based cache tier."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "cache.db"
        self.lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TIMESTAMP,
                    accessed_at TIMESTAMP,
                    access_count INTEGER,
                    ttl_seconds INTEGER,
                    size_bytes INTEGER,
                    compressed INTEGER,
                    metadata TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON cache_entries(created_at, ttl_seconds)")
            
            conn.commit()
            conn.close()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from disk cache."""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT value, created_at, accessed_at, access_count, 
                           ttl_seconds, size_bytes, compressed, metadata
                    FROM cache_entries WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    return None
                
                value_blob, created_at, accessed_at, access_count, ttl_seconds, size_bytes, compressed, metadata_json = row
                
                # Deserialize value
                if compressed:
                    value_blob = zlib.decompress(value_blob)
                value = SafePickleHandler.safe_load(value_blob)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.fromisoformat(created_at),
                    accessed_at=datetime.fromisoformat(accessed_at),
                    access_count=access_count,
                    ttl_seconds=ttl_seconds,
                    size_bytes=size_bytes,
                    compressed=bool(compressed),
                    tier="disk",
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                
                if entry.is_expired():
                    self.delete(key)
                    return None
                
                return entry
                
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            return None
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set entry in disk cache."""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Serialize value
                value_blob = pickle.dumps(entry.value)
                if entry.compressed:
                    value_blob = zlib.compress(value_blob)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, accessed_at, access_count, 
                     ttl_seconds, size_bytes, compressed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key,
                    value_blob,
                    entry.created_at.isoformat(),
                    entry.accessed_at.isoformat(),
                    entry.access_count,
                    entry.ttl_seconds,
                    entry.size_bytes,
                    int(entry.compressed),
                    json.dumps(entry.metadata)
                ))
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Error deleting from disk cache: {e}")
            return False


class PerfectedCacheManager:
    """
    Enterprise-grade multi-tier cache manager.
    
    Replaces intelligent_cache.py with enhanced features:
    - Multi-tier caching (memory + disk)
    - Thread-safe operations with minimal locking
    - Background cleanup and analytics
    - Event-driven architecture
    - Performance monitoring
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        memory_size_mb: int = 256,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_compression: bool = True,
        enable_persistence: bool = True,
        cleanup_interval: int = 300  # 5 minutes
    ):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.enable_persistence = enable_persistence
        
        # Initialize cache tiers
        self.memory_cache = MemoryCache(memory_size_mb, strategy)
        self.disk_cache = DiskCache(self.cache_dir) if enable_persistence else None
        
        # Metrics and monitoring
        self.metrics = CacheMetrics()
        self.listeners: Dict[CacheEvent, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache")
        self.cleanup_interval = cleanup_interval
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        def cleanup_task():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")
        
        self.executor.submit(cleanup_task)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with multi-tier lookup."""
        start_time = time.time()
        
        try:
            # Try memory cache first
            entry = self.memory_cache.get(key)
            if entry:
                self.metrics.hits += 1
                self._emit_event(CacheEvent.GET, key, entry.value)
                return entry.value
            
            # Try disk cache
            if self.disk_cache:
                entry = self.disk_cache.get(key)
                if entry:
                    self.metrics.hits += 1
                    self.metrics.disk_reads += 1
                    
                    # Promote to memory cache
                    self.memory_cache.set(key, entry)
                    
                    self._emit_event(CacheEvent.GET, key, entry.value)
                    return entry.value
            
            # Cache miss
            self.metrics.misses += 1
            return default
            
        finally:
            self.metrics.total_requests += 1
            response_time = time.time() - start_time
            self.metrics.avg_response_time = (
                (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + response_time) 
                / self.metrics.total_requests
            )
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set value in cache with intelligent compression."""
        try:
            # Estimate size and determine compression
            size_bytes = len(pickle.dumps(value))
            compressed = False
            
            if self.enable_compression and size_bytes > 1024:
                compressed_data = zlib.compress(pickle.dumps(value))
                if len(compressed_data) < size_bytes * 0.9:
                    compressed = True
                    self.metrics.compressions += 1
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes,
                compressed=compressed,
                metadata=metadata or {}
            )
            
            # Store in memory cache
            self.memory_cache.set(key, entry)
            
            # Store in disk cache (async)
            if self.disk_cache:
                self.executor.submit(self._async_disk_write, key, entry)
            
            self._emit_event(CacheEvent.SET, key, value)
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache entry: {e}")
            return False
    
    def _async_disk_write(self, key: str, entry: CacheEntry):
        """Asynchronously write to disk cache."""
        try:
            self.disk_cache.set(key, entry)
            self.metrics.disk_writes += 1
        except Exception as e:
            logger.error(f"Async disk write error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete entry from all cache tiers."""
        deleted = False
        
        if self.memory_cache.delete(key):
            deleted = True
        
        if self.disk_cache and self.disk_cache.delete(key):
            deleted = True
        
        if deleted:
            self._emit_event(CacheEvent.DELETE, key, None)
        
        return deleted
    
    def clear(self):
        """Clear all cache tiers."""
        self.memory_cache = MemoryCache(
            self.memory_cache.max_size_bytes // (1024 * 1024),
            self.memory_cache.strategy
        )
        
        if self.disk_cache:
            try:
                conn = sqlite3.connect(str(self.disk_cache.db_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries")
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error clearing disk cache: {e}")
        
        self._emit_event(CacheEvent.CLEAR, None, None)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from all tiers."""
        removed = 0
        
        # Cleanup memory cache
        with self.memory_cache.lock:
            expired_keys = [
                key for key, entry in self.memory_cache.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self.memory_cache.delete(key)
                self._emit_event(CacheEvent.EXPIRE, key, None)
                removed += 1
        
        # Cleanup disk cache (async)
        if self.disk_cache:
            self.executor.submit(self._async_disk_cleanup)
        
        return removed
    
    def _async_disk_cleanup(self):
        """Asynchronously cleanup disk cache."""
        try:
            with self.disk_cache.lock:
                conn = sqlite3.connect(str(self.disk_cache.db_path))
                cursor = conn.cursor()
                
                # Find expired entries
                cursor.execute("""
                    DELETE FROM cache_entries
                    WHERE datetime(created_at, '+' || ttl_seconds || ' seconds') < datetime('now')
                    AND ttl_seconds > 0
                """)
                
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"Async disk cleanup error: {e}")
    
    def add_listener(self, event: CacheEvent, callback: Callable):
        """Add event listener."""
        self.listeners[event].append(callback)
    
    def _emit_event(self, event: CacheEvent, key: Optional[str], value: Any):
        """Emit cache event to listeners."""
        for callback in self.listeners[event]:
            try:
                callback(event, key, value)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        return {
            "memory_entries": len(self.memory_cache.cache),
            "memory_size_mb": self.memory_cache.current_size / (1024 * 1024),
            "memory_utilization": (
                self.memory_cache.current_size / self.memory_cache.max_size_bytes * 100
                if self.memory_cache.max_size_bytes > 0 else 0
            ),
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "hit_rate": self.metrics.hit_rate(),
            "total_requests": self.metrics.total_requests,
            "evictions": self.metrics.evictions,
            "compressions": self.metrics.compressions,
            "disk_reads": self.metrics.disk_reads,
            "disk_writes": self.metrics.disk_writes,
            "avg_response_time_ms": self.metrics.avg_response_time * 1000,
            "strategy": self.memory_cache.strategy.value
        }
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate consistent cache key from arguments."""
        key_parts = []
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(json.dumps(arg, sort_keys=True, default=str))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()


# Global cache instance
_global_cache: Optional[PerfectedCacheManager] = None


def get_cache() -> PerfectedCacheManager:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PerfectedCacheManager()
    return _global_cache


def cached(ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_cache()
            cache_key = cache.generate_key(func.__name__, *args, **kwargs)
            
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo and testing
    cache = PerfectedCacheManager()
    
    # Test basic operations
    cache.set("test", "value", ttl=60)
    print(f"Retrieved: {cache.get('test')}")
    
    # Test metrics
    metrics = cache.get_metrics()
    print(f"Cache metrics: {json.dumps(metrics, indent=2)}")