#!/usr/bin/env python3
"""
Smart Caching Layer for TestMaster
Intelligent caching system for API responses and test results.

Features:
- LRU cache with configurable TTL
- Content-based deduplication
- Persistent cache with SQLite backend
- Compression and encryption support
- Cache warming and preloading strategies
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import pickle
import zlib
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based


@dataclass
class CacheEntry:
    """Represents a single cache entry."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 1
    ttl_seconds: int = 3600
    size_bytes: int = 0
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class IntelligentCache:
    """Main intelligent caching system."""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_size_mb: int = 500,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
        enable_compression: bool = True,
        enable_persistence: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_compression = enable_compression
        self.enable_persistence = enable_persistence
        
        # In-memory cache
        self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0
        }
        
        # Persistent storage
        if self.enable_persistence:
            self.db_path = self.cache_dir / "cache.db"
            self._init_database()
            self._load_from_disk()
    
    def _init_database(self):
        """Initialize SQLite database for persistent cache."""
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
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _load_from_disk(self):
        """Load cache entries from persistent storage."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT key, value, created_at, accessed_at, access_count, 
                       ttl_seconds, size_bytes, compressed, metadata
                FROM cache_entries
                ORDER BY accessed_at DESC
            """)
            
            for row in cursor.fetchall():
                key, value_blob, created_at, accessed_at, access_count, ttl_seconds, size_bytes, compressed, metadata_json = row
                
                # Deserialize value
                if compressed:
                    value_blob = zlib.decompress(value_blob)
                value = pickle.loads(value_blob)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.fromisoformat(created_at),
                    accessed_at=datetime.fromisoformat(accessed_at),
                    access_count=access_count,
                    ttl_seconds=ttl_seconds,
                    size_bytes=size_bytes,
                    compressed=bool(compressed),
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                
                # Add to memory cache if not expired
                if not entry.is_expired():
                    self.memory_cache[key] = entry
                    self.current_size += size_bytes
            
            conn.close()
            logger.info(f"Loaded {len(self.memory_cache)} entries from persistent cache")
            
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
    
    def _save_to_disk(self, entry: CacheEntry):
        """Save cache entry to persistent storage."""
        if not self.enable_persistence:
            return
        
        try:
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
                entry.key,
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
            
        except Exception as e:
            logger.error(f"Failed to save cache entry to disk: {e}")
    
    def _remove_from_disk(self, key: str):
        """Remove cache entry from persistent storage."""
        if not self.enable_persistence:
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to remove cache entry from disk: {e}")
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a string representation of all arguments
        key_parts = []
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(json.dumps(arg, sort_keys=True, default=str))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        
        # Hash the key for consistent length
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self.lock:
            # Check if key exists
            if key not in self.memory_cache:
                self.stats["misses"] += 1
                return default
            
            entry = self.memory_cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._evict(key)
                self.stats["misses"] += 1
                return default
            
            # Update access info
            entry.touch()
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self.memory_cache.move_to_end(key)
            
            self.stats["hits"] += 1
            
            # Decompress if needed
            value = entry.value
            if entry.compressed and isinstance(value, bytes):
                value = pickle.loads(zlib.decompress(value))
                self.stats["decompressions"] += 1
            
            return value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set value in cache."""
        with self.lock:
            # Calculate size
            value_bytes = pickle.dumps(value)
            size_bytes = len(value_bytes)
            
            # Compress if beneficial
            compressed = False
            if self.enable_compression and size_bytes > 1024:  # Compress if > 1KB
                compressed_bytes = zlib.compress(value_bytes)
                if len(compressed_bytes) < size_bytes * 0.9:  # At least 10% reduction
                    value = compressed_bytes
                    size_bytes = len(compressed_bytes)
                    compressed = True
                    self.stats["compressions"] += 1
            
            # Check if we need to evict entries
            while self.current_size + size_bytes > self.max_size_bytes and self.memory_cache:
                self._evict_one()
            
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
            
            # Remove old entry if exists
            if key in self.memory_cache:
                old_entry = self.memory_cache[key]
                self.current_size -= old_entry.size_bytes
            
            # Add new entry
            self.memory_cache[key] = entry
            self.current_size += size_bytes
            
            # Save to disk
            self._save_to_disk(entry)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key in self.memory_cache:
                self._evict(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.memory_cache.clear()
            self.current_size = 0
            
            if self.enable_persistence:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries")
                conn.commit()
                conn.close()
    
    def _evict_one(self):
        """Evict one entry based on strategy."""
        if not self.memory_cache:
            return
        
        key_to_evict = None
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            key_to_evict = next(iter(self.memory_cache))
            
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            key_to_evict = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].access_count
            )
            
        elif self.strategy == CacheStrategy.FIFO:
            # Evict oldest
            key_to_evict = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].created_at
            )
            
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired or closest to expiry
            expired = [k for k, v in self.memory_cache.items() if v.is_expired()]
            if expired:
                key_to_evict = expired[0]
            else:
                key_to_evict = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].created_at + 
                                 timedelta(seconds=self.memory_cache[k].ttl_seconds)
                )
        
        if key_to_evict:
            self._evict(key_to_evict)
    
    def _evict(self, key: str):
        """Evict entry from cache."""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.current_size -= entry.size_bytes
            del self.memory_cache[key]
            self._remove_from_disk(key)
            self.stats["evictions"] += 1
    
    def warm_cache(self, keys_and_values: List[Tuple[str, Any, Optional[Dict]]]):
        """Pre-populate cache with known values."""
        success_count = 0
        for item in keys_and_values:
            if len(item) == 2:
                key, value = item
                metadata = None
            else:
                key, value, metadata = item
            
            if self.set(key, value, metadata=metadata):
                success_count += 1
        
        logger.info(f"Warmed cache with {success_count}/{len(keys_and_values)} entries")
        return success_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "entries": len(self.memory_cache),
                "size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": (self.current_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self.stats["evictions"],
                "compressions": self.stats["compressions"],
                "decompressions": self.stats["decompressions"],
                "strategy": self.strategy.value
            }
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        with self.lock:
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._evict(key)
            
            return len(expired_keys)
    
    def export_cache(self, file_path: Path) -> bool:
        """Export cache to file."""
        try:
            with self.lock:
                cache_data = {
                    "metadata": {
                        "exported_at": datetime.now().isoformat(),
                        "entries": len(self.memory_cache),
                        "size_mb": self.current_size / (1024 * 1024)
                    },
                    "entries": []
                }
                
                for key, entry in self.memory_cache.items():
                    if not entry.is_expired():
                        cache_data["entries"].append({
                            "key": key,
                            "created_at": entry.created_at.isoformat(),
                            "ttl_seconds": entry.ttl_seconds,
                            "metadata": entry.metadata
                        })
                
                with open(file_path, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                
                logger.info(f"Exported {len(cache_data['entries'])} cache entries to {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export cache: {e}")
            return False


class CachedFunction:
    """Decorator for caching function results."""
    
    def __init__(self, cache: IntelligentCache, ttl: Optional[int] = None):
        self.cache = cache
        self.ttl = ttl
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self.cache.generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = self.cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            self.cache.set(cache_key, result, ttl=self.ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


# Global cache instance
_global_cache: Optional[IntelligentCache] = None


def get_cache() -> IntelligentCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        from config.testmaster_config import config
        _global_cache = IntelligentCache(
            cache_dir=config.caching.cache_directory,
            max_size_mb=config.caching.max_cache_size_mb,
            default_ttl=config.caching.cache_ttl_seconds,
            strategy=CacheStrategy(config.caching.cache_strategy),
            enable_compression=config.caching.compression_enabled,
            enable_persistence=config.caching.persistent_cache
        )
    return _global_cache


def cached(ttl: Optional[int] = None):
    """Decorator for caching function results using global cache."""
    cache = get_cache()
    return CachedFunction(cache, ttl)


def main():
    """CLI for cache management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Cache Manager")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--clear", action="store_true", help="Clear all cache entries")
    parser.add_argument("--cleanup", action="store_true", help="Remove expired entries")
    parser.add_argument("--export", help="Export cache metadata to file")
    parser.add_argument("--warm", help="Warm cache from file")
    parser.add_argument("--test", action="store_true", help="Run cache tests")
    
    args = parser.parse_args()
    
    cache = get_cache()
    
    if args.stats:
        stats = cache.get_stats()
        print("\nCache Statistics:")
        print("="*40)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key:20}: {value:.2f}")
            else:
                print(f"{key:20}: {value}")
    
    if args.clear:
        cache.clear()
        print("Cache cleared successfully")
    
    if args.cleanup:
        removed = cache.cleanup_expired()
        print(f"Removed {removed} expired entries")
    
    if args.export:
        if cache.export_cache(Path(args.export)):
            print(f"Cache exported to {args.export}")
        else:
            print("Failed to export cache")
    
    if args.test:
        print("\nRunning cache tests...")
        
        # Test basic operations
        cache.set("test_key", {"data": "test_value"}, ttl=5)
        assert cache.get("test_key") == {"data": "test_value"}
        print("✓ Basic set/get works")
        
        # Test expiration
        time.sleep(6)
        assert cache.get("test_key") is None
        print("✓ TTL expiration works")
        
        # Test compression
        large_data = "x" * 10000
        cache.set("large_key", large_data)
        assert cache.get("large_key") == large_data
        print("✓ Compression works")
        
        # Test decorator
        @cached(ttl=10)
        def expensive_function(x):
            time.sleep(1)
            return x * 2
        
        start = time.time()
        result1 = expensive_function(5)
        duration1 = time.time() - start
        
        start = time.time()
        result2 = expensive_function(5)
        duration2 = time.time() - start
        
        assert result1 == result2 == 10
        assert duration2 < duration1 / 10  # Second call should be much faster
        print("✓ Function caching works")
        
        print("\nAll tests passed!")


if __name__ == "__main__":
    main()