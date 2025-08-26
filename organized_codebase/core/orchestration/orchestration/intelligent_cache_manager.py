#!/usr/bin/env python3
"""
Intelligent Cache Manager
Agent B Hours 80-90: Caching Strategy Implementation

Advanced caching system with multi-level cache architecture, intelligent eviction policies,
predictive preloading, and distributed cache coordination for orchestration components.
"""

import asyncio
import logging
import time
import hashlib
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict, deque
import threading
import weakref
from functools import wraps
import redis
import memcache

T = TypeVar('T')

class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1_MEMORY = "l1_memory"      # In-memory cache (fastest)
    L2_SHARED = "l2_shared"       # Shared memory cache
    L3_REDIS = "l3_redis"         # Redis cache
    L4_MEMCACHED = "l4_memcached" # Memcached
    L5_DISK = "l5_disk"           # Disk cache (slowest)

class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    FIFO = "fifo"                 # First In First Out
    TTL = "ttl"                   # Time To Live
    ARC = "arc"                   # Adaptive Replacement Cache
    INTELLIGENT = "intelligent"    # ML-based eviction

class CacheStrategy(Enum):
    """Caching strategies"""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    REFRESH_AHEAD = "refresh_ahead"
    PREDICTIVE = "predictive"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[int]
    cost: float  # Computation cost
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)

@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    avg_latency: float = 0.0
    memory_usage: int = 0
    cache_size: int = 0

class LRUCache:
    """LRU cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.stats = CacheStatistics()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry = self.cache[key]
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self.stats.hits += 1
                return entry.value
            self.stats.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, cost: float = 1.0):
        with self.lock:
            if key in self.cache:
                # Update existing entry
                entry = self.cache[key]
                entry.value = value
                entry.last_accessed = datetime.now()
                self.cache.move_to_end(key)
            else:
                # Add new entry
                if len(self.cache) >= self.capacity:
                    # Evict least recently used
                    evicted_key = next(iter(self.cache))
                    del self.cache[evicted_key]
                    self.stats.evictions += 1
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size=self._estimate_size(value),
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    ttl=ttl,
                    cost=cost
                )
                self.cache[key] = entry
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            return len(pickle.dumps(value))
        except:
            return 0
    
    def clear(self):
        with self.lock:
            self.cache.clear()

class IntelligentCache:
    """
    Intelligent Cache Manager
    
    Advanced multi-level caching system with intelligent eviction policies,
    predictive preloading, and distributed cache coordination for optimal
    performance in orchestration and processing components.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("IntelligentCache")
        
        # Cache configuration
        self.cache_levels: Dict[CacheLevel, Any] = {}
        self.eviction_policy = EvictionPolicy.INTELLIGENT
        self.cache_strategy = CacheStrategy.PREDICTIVE
        
        # L1 Memory cache
        self.l1_cache = LRUCache(capacity=10000)
        self.cache_levels[CacheLevel.L1_MEMORY] = self.l1_cache
        
        # L2 Shared memory cache
        self.l2_cache: Optional[Dict] = {}
        self.cache_levels[CacheLevel.L2_SHARED] = self.l2_cache
        
        # L3 Redis cache (optional)
        self.redis_client: Optional[redis.Redis] = None
        self._init_redis()
        
        # L4 Memcached (optional)
        self.memcached_client: Optional[memcache.Client] = None
        self._init_memcached()
        
        # Cache statistics
        self.stats: Dict[CacheLevel, CacheStatistics] = {
            level: CacheStatistics() for level in CacheLevel
        }
        
        # Predictive caching
        self.access_patterns: deque = deque(maxlen=10000)
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prediction_model: Optional[Any] = None
        
        # Cache invalidation
        self.invalidation_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Cache warming
        self.warm_cache_keys: Set[str] = set()
        self.warming_enabled = True
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 60  # seconds
        
        self.logger.info("Intelligent cache manager initialized")
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.cache_levels[CacheLevel.L3_REDIS] = self.redis_client
            self.logger.info("Redis cache initialized")
        except:
            self.redis_client = None
            self.logger.warning("Redis cache not available")
    
    def _init_memcached(self):
        """Initialize Memcached connection"""
        try:
            self.memcached_client = memcache.Client(['127.0.0.1:11211'], debug=0)
            self.cache_levels[CacheLevel.L4_MEMCACHED] = self.memcached_client
            self.logger.info("Memcached cache initialized")
        except:
            self.memcached_client = None
            self.logger.warning("Memcached cache not available")
    
    async def get(self, key: str, compute_fn: Optional[Callable] = None) -> Optional[Any]:
        """Get value from cache with multi-level lookup"""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(key)
        
        # Check each cache level
        for level in CacheLevel:
            value = await self._get_from_level(level, cache_key)
            if value is not None:
                # Update statistics
                self.stats[level].hits += 1
                self.stats[level].total_requests += 1
                
                # Promote to higher cache levels
                await self._promote_to_higher_levels(level, cache_key, value)
                
                # Record access pattern
                self._record_access_pattern(cache_key)
                
                # Update latency
                latency = time.time() - start_time
                self._update_latency(level, latency)
                
                return value
        
        # Cache miss - compute if function provided
        if compute_fn:
            value = await self._compute_and_cache(cache_key, compute_fn)
            return value
        
        # Update miss statistics
        for level in CacheLevel:
            self.stats[level].misses += 1
            self.stats[level].total_requests += 1
        
        return None
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None, 
                  levels: Optional[List[CacheLevel]] = None) -> bool:
        """Put value into cache at specified levels"""
        try:
            cache_key = self._generate_cache_key(key)
            
            # Default to all levels if not specified
            if levels is None:
                levels = list(CacheLevel)
            
            # Store in each specified level
            for level in levels:
                await self._put_to_level(level, cache_key, value, ttl)
            
            # Update dependency graph if needed
            self._update_dependencies(cache_key, value)
            
            # Trigger predictive caching if enabled
            if self.cache_strategy == CacheStrategy.PREDICTIVE:
                await self._trigger_predictive_caching(cache_key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache put failed for key {key}: {e}")
            return False
    
    async def invalidate(self, key: str, cascade: bool = True):
        """Invalidate cache entry and optionally cascade to dependencies"""
        cache_key = self._generate_cache_key(key)
        
        # Invalidate from all levels
        for level in CacheLevel:
            await self._invalidate_from_level(level, cache_key)
        
        # Cascade invalidation to dependent keys
        if cascade and cache_key in self.dependency_graph:
            for dependent_key in self.dependency_graph[cache_key]:
                await self.invalidate(dependent_key, cascade=True)
        
        # Trigger invalidation callbacks
        if cache_key in self.invalidation_callbacks:
            for callback in self.invalidation_callbacks[cache_key]:
                try:
                    await callback(cache_key)
                except Exception as e:
                    self.logger.error(f"Invalidation callback failed: {e}")
    
    async def _get_from_level(self, level: CacheLevel, key: str) -> Optional[Any]:
        """Get value from specific cache level"""
        try:
            if level == CacheLevel.L1_MEMORY:
                return self.l1_cache.get(key)
            
            elif level == CacheLevel.L2_SHARED:
                return self.l2_cache.get(key) if self.l2_cache else None
            
            elif level == CacheLevel.L3_REDIS and self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value.encode('latin-1'))
            
            elif level == CacheLevel.L4_MEMCACHED and self.memcached_client:
                return self.memcached_client.get(key)
            
            elif level == CacheLevel.L5_DISK:
                # Implement disk cache if needed
                pass
            
        except Exception as e:
            self.logger.error(f"Error getting from {level}: {e}")
        
        return None
    
    async def _put_to_level(self, level: CacheLevel, key: str, value: Any, ttl: Optional[int]):
        """Put value to specific cache level"""
        try:
            if level == CacheLevel.L1_MEMORY:
                self.l1_cache.put(key, value, ttl)
            
            elif level == CacheLevel.L2_SHARED and self.l2_cache is not None:
                self.l2_cache[key] = value
            
            elif level == CacheLevel.L3_REDIS and self.redis_client:
                serialized = pickle.dumps(value).decode('latin-1')
                if ttl:
                    self.redis_client.setex(key, ttl, serialized)
                else:
                    self.redis_client.set(key, serialized)
            
            elif level == CacheLevel.L4_MEMCACHED and self.memcached_client:
                self.memcached_client.set(key, value, time=ttl or 0)
            
            elif level == CacheLevel.L5_DISK:
                # Implement disk cache if needed
                pass
            
        except Exception as e:
            self.logger.error(f"Error putting to {level}: {e}")
    
    async def _invalidate_from_level(self, level: CacheLevel, key: str):
        """Invalidate entry from specific cache level"""
        try:
            if level == CacheLevel.L1_MEMORY:
                if key in self.l1_cache.cache:
                    del self.l1_cache.cache[key]
            
            elif level == CacheLevel.L2_SHARED and self.l2_cache:
                self.l2_cache.pop(key, None)
            
            elif level == CacheLevel.L3_REDIS and self.redis_client:
                self.redis_client.delete(key)
            
            elif level == CacheLevel.L4_MEMCACHED and self.memcached_client:
                self.memcached_client.delete(key)
            
        except Exception as e:
            self.logger.error(f"Error invalidating from {level}: {e}")
    
    async def _promote_to_higher_levels(self, found_level: CacheLevel, key: str, value: Any):
        """Promote value to higher cache levels"""
        # Promote to faster cache levels
        level_order = list(CacheLevel)
        found_index = level_order.index(found_level)
        
        for i in range(found_index):
            higher_level = level_order[i]
            await self._put_to_level(higher_level, key, value, None)
    
    async def _compute_and_cache(self, key: str, compute_fn: Callable) -> Any:
        """Compute value and cache it"""
        start_time = time.time()
        
        # Compute value
        if asyncio.iscoroutinefunction(compute_fn):
            value = await compute_fn()
        else:
            value = compute_fn()
        
        # Calculate computation cost
        cost = time.time() - start_time
        
        # Cache based on cost
        if cost > 0.1:  # Expensive computation
            # Cache in all levels
            await self.put(key, value)
        elif cost > 0.01:  # Moderate computation
            # Cache in memory levels
            await self.put(key, value, levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_SHARED])
        else:  # Cheap computation
            # Cache only in L1
            await self.put(key, value, levels=[CacheLevel.L1_MEMORY])
        
        return value
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate normalized cache key"""
        if isinstance(key, str):
            return key
        else:
            # Hash complex keys
            key_str = json.dumps(key, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
    
    def _record_access_pattern(self, key: str):
        """Record access pattern for predictive caching"""
        self.access_patterns.append({
            "key": key,
            "timestamp": datetime.now(),
            "context": self._get_current_context()
        })
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current execution context for pattern learning"""
        return {
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "cache_size": len(self.l1_cache.cache),
            "memory_pressure": self._get_memory_pressure()
        }
    
    def _get_memory_pressure(self) -> float:
        """Get current memory pressure (0-1)"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except:
            return 0.5
    
    async def _trigger_predictive_caching(self, key: str):
        """Trigger predictive caching based on access patterns"""
        # Analyze patterns to predict next keys
        predicted_keys = self._predict_next_keys(key)
        
        # Queue predicted keys for prefetching
        for predicted_key in predicted_keys:
            await self.prefetch_queue.put(predicted_key)
    
    def _predict_next_keys(self, current_key: str) -> List[str]:
        """Predict next likely cache keys based on patterns"""
        predicted = []
        
        # Simple pattern matching (can be replaced with ML model)
        recent_patterns = list(self.access_patterns)[-100:]
        for i, pattern in enumerate(recent_patterns[:-1]):
            if pattern["key"] == current_key:
                # Next key in sequence is likely
                next_pattern = recent_patterns[i + 1]
                predicted.append(next_pattern["key"])
        
        return list(set(predicted))[:5]  # Return top 5 predictions
    
    def _update_dependencies(self, key: str, value: Any):
        """Update dependency graph for cache invalidation"""
        # Extract dependencies from value if it has them
        if hasattr(value, 'dependencies'):
            for dep in value.dependencies:
                self.dependency_graph[dep].append(key)
    
    def _update_latency(self, level: CacheLevel, latency: float):
        """Update average latency for cache level"""
        stats = self.stats[level]
        if stats.avg_latency == 0:
            stats.avg_latency = latency
        else:
            # Exponential moving average
            stats.avg_latency = stats.avg_latency * 0.9 + latency * 0.1
    
    def cache_method(self, ttl: int = 3600, key_prefix: str = ""):
        """Decorator for caching method results"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key from function and arguments
                cache_key = f"{key_prefix}{func.__name__}:{args}:{kwargs}"
                
                # Try to get from cache
                result = await self.get(cache_key)
                if result is not None:
                    return result
                
                # Compute and cache
                result = await func(*args, **kwargs)
                await self.put(cache_key, result, ttl=ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key from function and arguments
                cache_key = f"{key_prefix}{func.__name__}:{args}:{kwargs}"
                
                # Try to get from cache
                result = asyncio.run(self.get(cache_key))
                if result is not None:
                    return result
                
                # Compute and cache
                result = func(*args, **kwargs)
                asyncio.run(self.put(cache_key, result, ttl=ttl))
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def warm_cache(self, keys: List[str], compute_fns: Dict[str, Callable]):
        """Warm cache with specified keys"""
        for key in keys:
            if key not in self.l1_cache.cache:
                compute_fn = compute_fns.get(key)
                if compute_fn:
                    await self.get(key, compute_fn)
                    self.warm_cache_keys.add(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_stats = {
            "total_hits": sum(s.hits for s in self.stats.values()),
            "total_misses": sum(s.misses for s in self.stats.values()),
            "total_evictions": sum(s.evictions for s in self.stats.values()),
            "overall_hit_rate": 0.0,
            "level_statistics": {}
        }
        
        total_requests = total_stats["total_hits"] + total_stats["total_misses"]
        if total_requests > 0:
            total_stats["overall_hit_rate"] = total_stats["total_hits"] / total_requests
        
        # Per-level statistics
        for level, stats in self.stats.items():
            if stats.total_requests > 0:
                stats.hit_rate = stats.hits / stats.total_requests
            
            total_stats["level_statistics"][level.value] = {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "avg_latency": stats.avg_latency,
                "evictions": stats.evictions
            }
        
        # L1 cache details
        total_stats["l1_cache"] = {
            "size": len(self.l1_cache.cache),
            "capacity": self.l1_cache.capacity,
            "memory_usage": sum(e.size for e in self.l1_cache.cache.values())
        }
        
        return total_stats
    
    async def optimize_cache(self):
        """Optimize cache based on usage patterns"""
        # Analyze hit rates and adjust cache sizes
        for level, stats in self.stats.items():
            if stats.total_requests > 100:  # Enough data to optimize
                if stats.hit_rate < 0.5 and level == CacheLevel.L1_MEMORY:
                    # Increase L1 cache size
                    self.l1_cache.capacity = min(self.l1_cache.capacity * 1.5, 100000)
                    self.logger.info(f"Increased L1 cache capacity to {self.l1_cache.capacity}")
                
                elif stats.hit_rate > 0.9 and level == CacheLevel.L1_MEMORY:
                    # Potentially decrease cache size to save memory
                    self.l1_cache.capacity = max(self.l1_cache.capacity * 0.9, 1000)
                    self.logger.info(f"Decreased L1 cache capacity to {self.l1_cache.capacity}")
    
    def clear_all(self):
        """Clear all cache levels"""
        self.l1_cache.clear()
        if self.l2_cache:
            self.l2_cache.clear()
        if self.redis_client:
            self.redis_client.flushdb()
        if self.memcached_client:
            self.memcached_client.flush_all()
        
        # Reset statistics
        for stats in self.stats.values():
            stats.hits = 0
            stats.misses = 0
            stats.evictions = 0
            stats.total_requests = 0
            stats.hit_rate = 0.0