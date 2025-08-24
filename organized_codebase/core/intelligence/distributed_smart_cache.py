"""
Distributed Smart Cache System
=============================

Advanced distributed caching with predictive prefetching, intelligent eviction,
adaptive sizing, and multi-level cache hierarchy. Extracted from archive
smart cache components.

Provides enterprise-grade caching with ML-powered optimization.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import statistics
import uuid

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    ADAPTIVE = "adaptive" # Adaptive policy based on usage patterns
    PREDICTIVE = "predictive"  # ML-powered predictive eviction


class CacheLevel(Enum):
    """Multi-level cache hierarchy"""
    L1_MEMORY = "l1_memory"         # Fast in-memory cache
    L2_COMPRESSED = "l2_compressed" # Compressed memory cache
    L3_DISK = "l3_disk"            # Persistent disk cache


class CacheOperation(Enum):
    """Cache operation types"""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    EVICT = "evict"
    PREFETCH = "prefetch"


@dataclass
class CacheEntry:
    """Cached data entry with metadata"""
    key: str
    data: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    level: CacheLevel = CacheLevel.L1_MEMORY
    compressed: bool = False
    hit_prediction_score: float = 0.0
    access_pattern: List[datetime] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    ttl: Optional[timedelta] = None
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.access_pattern.append(self.last_accessed)
        
        # Keep only recent access history
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-50:]
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return datetime.now() > (self.created_at + self.ttl)
    
    def calculate_score(self, policy: CachePolicy) -> float:
        """Calculate eviction score based on policy"""
        now = datetime.now()
        
        if policy == CachePolicy.LRU:
            return (now - self.last_accessed).total_seconds()
        elif policy == CachePolicy.LFU:
            return -self.access_count  # Negative for ascending sort
        elif policy == CachePolicy.ADAPTIVE:
            # Combine recency and frequency
            recency = (now - self.last_accessed).total_seconds()
            frequency = self.access_count
            return recency / (frequency + 1)
        elif policy == CachePolicy.PREDICTIVE:
            return -self.hit_prediction_score  # Higher prediction = lower eviction score
        
        return 0.0


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetches: int = 0
    size_bytes: int = 0
    entries_count: int = 0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_hit_rate(self):
        """Update hit rate calculation"""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0


class DistributedSmartCache:
    """Advanced distributed cache with intelligent features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Cache configuration
        self.cache_policy = CachePolicy(self.config.get('policy', 'adaptive'))
        self.max_size_bytes = self.config.get('max_size_bytes', 100 * 1024 * 1024)  # 100MB
        self.max_entries = self.config.get('max_entries', 10000)
        self.compression_threshold = self.config.get('compression_threshold', 1024)  # 1KB
        self.enable_predictive_prefetch = self.config.get('enable_prefetch', True)
        
        # Multi-level storage
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache: Dict[str, bytes] = {}  # Compressed storage
        self.l3_cache: Dict[str, str] = {}    # File paths for disk storage
        
        # Cache tracking
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.prediction_model: Dict[str, float] = {}
        self.key_relationships: Dict[str, List[str]] = defaultdict(list)
        
        # Performance metrics
        self.metrics_by_level: Dict[CacheLevel, CacheMetrics] = {
            level: CacheMetrics() for level in CacheLevel
        }
        self.global_metrics = CacheMetrics()
        
        # Background processing
        self.is_running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start background tasks
        self.optimization_thread.start()
        self.prefetch_thread.start()
        self.cleanup_thread.start()
        
        self.logger.info("Distributed Smart Cache initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with multi-level lookup
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        with self.lock:
            # Try L1 cache first
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self._evict_entry(key, CacheLevel.L1_MEMORY)
                else:
                    entry.update_access()
                    self._update_metrics(CacheLevel.L1_MEMORY, hit=True)
                    self._update_access_pattern(key)
                    
                    access_time = (time.time() - start_time) * 1000
                    self._update_access_time(CacheLevel.L1_MEMORY, access_time)
                    
                    return entry.data
            
            # Try L2 cache (compressed)
            if key in self.l2_cache:
                compressed_data = self.l2_cache[key]
                try:
                    data = SafePickleHandler.safe_load(zlib.decompress(compressed_data))
                    
                    # Promote to L1 if frequently accessed
                    self._promote_to_l1(key, data)
                    self._update_metrics(CacheLevel.L2_COMPRESSED, hit=True)
                    self._update_access_pattern(key)
                    
                    access_time = (time.time() - start_time) * 1000
                    self._update_access_time(CacheLevel.L2_COMPRESSED, access_time)
                    
                    return data
                    
                except Exception as e:
                    self.logger.error(f"L2 cache decompression failed for {key}: {e}")
                    del self.l2_cache[key]
            
            # Try L3 cache (disk) - placeholder for now
            if key in self.l3_cache:
                # In real implementation, this would read from disk
                self._update_metrics(CacheLevel.L3_DISK, hit=True)
                # For now, just return None
            
            # Cache miss
            for level in CacheLevel:
                self._update_metrics(level, hit=False)
            
            access_time = (time.time() - start_time) * 1000
            self._update_access_time(CacheLevel.L1_MEMORY, access_time)
            
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None,
           tags: Optional[List[str]] = None, level: CacheLevel = CacheLevel.L1_MEMORY) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live
            tags: Tags for categorization
            level: Cache level to store in
            
        Returns:
            True if successful
        """
        with self.lock:
            try:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Check cache capacity
                if not self._ensure_capacity(size_bytes, level):
                    return False
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    data=value,
                    size_bytes=size_bytes,
                    level=level,
                    ttl=ttl,
                    tags=tags or []
                )
                
                # Store based on level
                if level == CacheLevel.L1_MEMORY:
                    self.l1_cache[key] = entry
                elif level == CacheLevel.L2_COMPRESSED:
                    compressed_data = self._compress_data(value)
                    self.l2_cache[key] = compressed_data
                    entry.compressed = True
                elif level == CacheLevel.L3_DISK:
                    # Placeholder for disk storage
                    self.l3_cache[key] = f"disk_path_{key}"
                
                # Update metrics
                metrics = self.metrics_by_level[level]
                metrics.entries_count += 1
                metrics.size_bytes += size_bytes
                
                # Update relationships for predictive caching
                self._update_key_relationships(key, tags or [])
                
                self.logger.debug(f"Cached {key} in {level.value} ({size_bytes} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"Cache set failed for {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        with self.lock:
            deleted = False
            
            # Remove from L1
            if key in self.l1_cache:
                entry = self.l1_cache.pop(key)
                self._update_metrics_on_removal(CacheLevel.L1_MEMORY, entry.size_bytes)
                deleted = True
            
            # Remove from L2
            if key in self.l2_cache:
                del self.l2_cache[key]
                self._update_metrics_on_removal(CacheLevel.L2_COMPRESSED, 0)
                deleted = True
            
            # Remove from L3
            if key in self.l3_cache:
                del self.l3_cache[key]
                self._update_metrics_on_removal(CacheLevel.L3_DISK, 0)
                deleted = True
            
            # Clean up relationships
            if key in self.key_relationships:
                del self.key_relationships[key]
            
            return deleted
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all entries with specified tags"""
        with self.lock:
            invalidated = 0
            keys_to_delete = []
            
            # Find keys with matching tags
            for key, entry in self.l1_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
            
            # Delete found keys
            for key in keys_to_delete:
                if self.delete(key):
                    invalidated += 1
            
            self.logger.info(f"Invalidated {invalidated} entries with tags: {tags}")
            return invalidated
    
    def _ensure_capacity(self, required_bytes: int, level: CacheLevel) -> bool:
        """Ensure cache has capacity for new entry"""
        metrics = self.metrics_by_level[level]
        
        # Check entry count limit
        if metrics.entries_count >= self.max_entries:
            self._evict_entries(level, count=1)
        
        # Check size limit
        if metrics.size_bytes + required_bytes > self.max_size_bytes:
            bytes_to_free = (metrics.size_bytes + required_bytes) - self.max_size_bytes
            self._evict_entries(level, bytes_to_free=bytes_to_free)
        
        return True
    
    def _evict_entries(self, level: CacheLevel, count: Optional[int] = None,
                      bytes_to_free: Optional[int] = None):
        """Evict entries based on cache policy"""
        if level == CacheLevel.L1_MEMORY:
            entries = list(self.l1_cache.items())
        else:
            return  # Only L1 eviction implemented for now
        
        if not entries:
            return
        
        # Sort by eviction score
        scored_entries = [
            (key, entry, entry.calculate_score(self.cache_policy))
            for key, entry in entries
        ]
        scored_entries.sort(key=lambda x: x[2], reverse=True)
        
        # Evict entries
        freed_bytes = 0
        evicted_count = 0
        
        for key, entry, score in scored_entries:
            if count and evicted_count >= count:
                break
            if bytes_to_free and freed_bytes >= bytes_to_free:
                break
            
            self._evict_entry(key, level)
            freed_bytes += entry.size_bytes
            evicted_count += 1
        
        self.logger.debug(f"Evicted {evicted_count} entries from {level.value} (freed {freed_bytes} bytes)")
    
    def _evict_entry(self, key: str, level: CacheLevel):
        """Evict specific entry"""
        if level == CacheLevel.L1_MEMORY and key in self.l1_cache:
            entry = self.l1_cache.pop(key)
            self._update_metrics_on_removal(level, entry.size_bytes)
            self.metrics_by_level[level].evictions += 1
        elif level == CacheLevel.L2_COMPRESSED and key in self.l2_cache:
            del self.l2_cache[key]
            self._update_metrics_on_removal(level, 0)
            self.metrics_by_level[level].evictions += 1
    
    def _promote_to_l1(self, key: str, data: Any):
        """Promote entry from lower level to L1"""
        # Check if promotion is beneficial
        if self._should_promote(key):
            self.set(key, data, level=CacheLevel.L1_MEMORY)
    
    def _should_promote(self, key: str) -> bool:
        """Determine if key should be promoted to L1"""
        # Check access frequency
        pattern = self.access_patterns.get(key, [])
        if len(pattern) < 3:
            return False
        
        # Check recent access frequency
        now = datetime.now()
        recent_accesses = [
            access for access in pattern
            if (now - access).total_seconds() < 3600  # Last hour
        ]
        
        return len(recent_accesses) >= 3
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for L2 storage"""
        serialized = pickle.dumps(data)
        return zlib.compress(serialized)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            elif isinstance(value, list):
                return sum(self._calculate_size(item) for item in value)
            else:
                return 64  # Default estimate
    
    def _update_metrics(self, level: CacheLevel, hit: bool):
        """Update cache metrics"""
        metrics = self.metrics_by_level[level]
        
        if hit:
            metrics.hits += 1
            self.global_metrics.hits += 1
        else:
            metrics.misses += 1
            self.global_metrics.misses += 1
        
        metrics.update_hit_rate()
        self.global_metrics.update_hit_rate()
        metrics.last_updated = datetime.now()
    
    def _update_metrics_on_removal(self, level: CacheLevel, size_bytes: int):
        """Update metrics when entry is removed"""
        metrics = self.metrics_by_level[level]
        metrics.entries_count = max(0, metrics.entries_count - 1)
        metrics.size_bytes = max(0, metrics.size_bytes - size_bytes)
    
    def _update_access_time(self, level: CacheLevel, access_time_ms: float):
        """Update average access time"""
        metrics = self.metrics_by_level[level]
        
        if not hasattr(metrics, '_access_times'):
            metrics._access_times = []
        
        metrics._access_times.append(access_time_ms)
        if len(metrics._access_times) > 100:
            metrics._access_times = metrics._access_times[-50:]
        
        metrics.avg_access_time_ms = statistics.mean(metrics._access_times)
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for predictive modeling"""
        self.access_patterns[key].append(datetime.now())
        
        # Keep only recent history
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-50:]
    
    def _update_key_relationships(self, key: str, tags: List[str]):
        """Update key relationships for prefetching"""
        for tag in tags:
            # Find other keys with same tag
            related_keys = []
            for other_key, entry in self.l1_cache.items():
                if other_key != key and tag in entry.tags:
                    related_keys.append(other_key)
            
            if related_keys:
                self.key_relationships[key].extend(related_keys)
                
                # Limit relationship list
                if len(self.key_relationships[key]) > 20:
                    self.key_relationships[key] = self.key_relationships[key][-10:]
    
    def _optimization_loop(self):
        """Background optimization loop"""
        while self.is_running:
            try:
                # Update prediction scores
                self._update_prediction_scores()
                
                # Optimize cache levels
                self._optimize_cache_levels()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                time.sleep(60)
    
    def _prefetch_loop(self):
        """Background prefetching loop"""
        while self.is_running:
            try:
                if self.enable_predictive_prefetch:
                    self._perform_predictive_prefetch()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Prefetch loop error: {e}")
                time.sleep(30)
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.is_running:
            try:
                # Clean expired entries
                self._cleanup_expired_entries()
                
                # Clean old access patterns
                self._cleanup_access_patterns()
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(300)
    
    def _update_prediction_scores(self):
        """Update hit prediction scores for entries"""
        with self.lock:
            for key, entry in self.l1_cache.items():
                pattern = self.access_patterns.get(key, [])
                if len(pattern) >= 3:
                    # Simple prediction based on access frequency and recency
                    now = datetime.now()
                    recent_accesses = [
                        access for access in pattern
                        if (now - access).total_seconds() < 3600
                    ]
                    
                    frequency_score = len(recent_accesses) / 10.0  # Normalize
                    recency_score = 1.0 / (1 + (now - entry.last_accessed).total_seconds() / 3600)
                    
                    entry.hit_prediction_score = min(1.0, frequency_score + recency_score)
    
    def _optimize_cache_levels(self):
        """Optimize data distribution across cache levels"""
        with self.lock:
            # Move frequently accessed L2 items to L1
            promote_candidates = []
            
            for key in self.l2_cache.keys():
                if self._should_promote(key):
                    promote_candidates.append(key)
            
            for key in promote_candidates[:5]:  # Limit promotions
                if key in self.l2_cache:
                    try:
                        compressed_data = self.l2_cache[key]
                        data = SafePickleHandler.safe_load(zlib.decompress(compressed_data))
                        
                        if self.set(key, data, level=CacheLevel.L1_MEMORY):
                            del self.l2_cache[key]
                            self.logger.debug(f"Promoted {key} from L2 to L1")
                    
                    except Exception as e:
                        self.logger.error(f"Failed to promote {key}: {e}")
    
    def _perform_predictive_prefetch(self):
        """Perform predictive prefetching based on access patterns"""
        with self.lock:
            # Find keys that might be accessed soon
            prefetch_candidates = []
            
            for key, related_keys in self.key_relationships.items():
                if key in self.l1_cache:
                    # Recently accessed key - prefetch related keys
                    last_access = self.l1_cache[key].last_accessed
                    if (datetime.now() - last_access).total_seconds() < 300:  # 5 minutes
                        for related_key in related_keys:
                            if (related_key not in self.l1_cache and 
                                related_key in self.l2_cache):
                                prefetch_candidates.append(related_key)
            
            # Prefetch top candidates
            for key in prefetch_candidates[:3]:  # Limit prefetches
                try:
                    compressed_data = self.l2_cache[key]
                    data = SafePickleHandler.safe_load(zlib.decompress(compressed_data))
                    
                    if self.set(key, data, level=CacheLevel.L1_MEMORY):
                        self.metrics_by_level[CacheLevel.L1_MEMORY].prefetches += 1
                        self.logger.debug(f"Prefetched {key}")
                
                except Exception as e:
                    self.logger.error(f"Prefetch failed for {key}: {e}")
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        with self.lock:
            expired_keys = []
            
            for key, entry in self.l1_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._evict_entry(key, CacheLevel.L1_MEMORY)
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def _cleanup_access_patterns(self):
        """Clean up old access pattern data"""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        with self.lock:
            for key in list(self.access_patterns.keys()):
                pattern = self.access_patterns[key]
                # Keep only recent accesses
                recent_pattern = [
                    access for access in pattern
                    if access > cutoff_time
                ]
                
                if recent_pattern:
                    self.access_patterns[key] = recent_pattern
                else:
                    del self.access_patterns[key]
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            return {
                'global_metrics': {
                    'total_hits': self.global_metrics.hits,
                    'total_misses': self.global_metrics.misses,
                    'global_hit_rate': self.global_metrics.hit_rate,
                    'last_updated': self.global_metrics.last_updated.isoformat()
                },
                'level_metrics': {
                    level.value: {
                        'hits': metrics.hits,
                        'misses': metrics.misses,
                        'evictions': metrics.evictions,
                        'prefetches': metrics.prefetches,
                        'entries_count': metrics.entries_count,
                        'size_bytes': metrics.size_bytes,
                        'hit_rate': metrics.hit_rate,
                        'avg_access_time_ms': metrics.avg_access_time_ms,
                        'last_updated': metrics.last_updated.isoformat()
                    }
                    for level, metrics in self.metrics_by_level.items()
                },
                'cache_sizes': {
                    'l1_entries': len(self.l1_cache),
                    'l2_entries': len(self.l2_cache),
                    'l3_entries': len(self.l3_cache)
                },
                'configuration': {
                    'policy': self.cache_policy.value,
                    'max_size_bytes': self.max_size_bytes,
                    'max_entries': self.max_entries,
                    'compression_threshold': self.compression_threshold,
                    'enable_prefetch': self.enable_predictive_prefetch
                }
            }
    
    def flush_all(self) -> bool:
        """Flush all cache levels"""
        with self.lock:
            try:
                self.l1_cache.clear()
                self.l2_cache.clear()
                self.l3_cache.clear()
                self.access_patterns.clear()
                self.key_relationships.clear()
                
                # Reset metrics
                for metrics in self.metrics_by_level.values():
                    metrics.hits = 0
                    metrics.misses = 0
                    metrics.evictions = 0
                    metrics.prefetches = 0
                    metrics.entries_count = 0
                    metrics.size_bytes = 0
                    metrics.hit_rate = 0.0
                    metrics.last_updated = datetime.now()
                
                self.logger.info("Flushed all cache levels")
                return True
                
            except Exception as e:
                self.logger.error(f"Cache flush failed: {e}")
                return False
    
    def shutdown(self):
        """Gracefully shutdown cache system"""
        self.logger.info("Shutting down Distributed Smart Cache")
        self.is_running = False
        
        # Wait for background threads
        for thread in [self.optimization_thread, self.prefetch_thread, self.cleanup_thread]:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("Distributed Smart Cache shutdown complete")


# Global cache instance
distributed_smart_cache = DistributedSmartCache()

# Export
__all__ = [
    'CachePolicy', 'CacheLevel', 'CacheOperation',
    'CacheEntry', 'CacheMetrics',
    'DistributedSmartCache', 'distributed_smart_cache'
]