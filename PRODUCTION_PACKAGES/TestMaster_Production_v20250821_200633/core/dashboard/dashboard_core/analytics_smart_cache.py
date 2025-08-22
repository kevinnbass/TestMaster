"""
Smart Analytics Cache System
============================

Advanced caching system with predictive prefetching, intelligent eviction,
and adaptive cache sizing based on usage patterns.

Author: TestMaster Team
"""

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass
from enum import Enum
import statistics
import zlib

logger = logging.getLogger(__name__)

class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"

class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"
    L2_COMPRESSED = "l2_compressed"
    L3_DISK = "l3_disk"

@dataclass
class CacheEntry:
    """Represents a cached data entry."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    level: CacheLevel
    compressed: bool = False
    hit_prediction_score: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)

@dataclass
class AccessPattern:
    """Tracks access patterns for predictive caching."""
    key: str
    access_times: deque
    frequency: float
    last_access: datetime
    prediction_score: float = 0.0
    
    def __post_init__(self):
        if not isinstance(self.access_times, deque):
            self.access_times = deque(self.access_times, maxlen=100)

class SmartAnalyticsCache:
    """
    Advanced caching system with multiple levels and predictive capabilities.
    """
    
    def __init__(self, 
                 max_memory_size: int = 100 * 1024 * 1024,  # 100MB
                 max_compressed_size: int = 500 * 1024 * 1024,  # 500MB
                 cache_policy: CachePolicy = CachePolicy.ADAPTIVE,
                 compression_threshold: int = 1024,  # 1KB
                 prediction_enabled: bool = True):
        """
        Initialize the smart cache system.
        
        Args:
            max_memory_size: Maximum L1 cache size in bytes
            max_compressed_size: Maximum L2 cache size in bytes
            cache_policy: Cache eviction policy
            compression_threshold: Minimum size for compression
            prediction_enabled: Enable predictive prefetching
        """
        self.max_memory_size = max_memory_size
        self.max_compressed_size = max_compressed_size
        self.cache_policy = cache_policy
        self.compression_threshold = compression_threshold
        self.prediction_enabled = prediction_enabled
        
        # Multi-level cache storage
        self.l1_cache = OrderedDict()  # Memory cache
        self.l2_cache = OrderedDict()  # Compressed cache
        self.cache_metadata = {}  # Metadata for all entries
        
        # Size tracking
        self.l1_size = 0
        self.l2_size = 0
        
        # Access pattern tracking
        self.access_patterns = {}
        self.access_history = deque(maxlen=10000)
        
        # Performance tracking
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'decompressions': 0,
            'predictions': 0,
            'prediction_hits': 0,
            'start_time': datetime.now()
        }
        
        # Predictive caching
        self.prediction_scores = {}
        self.prefetch_queue = deque(maxlen=1000)
        self.prediction_thread = None
        self.prediction_active = False
        
        # Adaptive sizing
        self.adaptive_config = {
            'hit_rate_threshold': 0.8,
            'resize_factor': 0.1,
            'min_size_mb': 10,
            'max_size_mb': 1000
        }
        
        # Thread safety
        self.cache_lock = threading.RLock()
        
        if self.prediction_enabled:
            self.start_prediction_engine()
        
        logger.info("Smart Analytics Cache initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get data from cache."""
        with self.cache_lock:
            # Record access
            access_time = datetime.now()
            self._record_access(key, access_time)
            
            # Check L1 cache first
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                entry.last_accessed = access_time
                entry.access_count += 1
                
                # Move to end (LRU)
                try:
                    self.l1_cache.move_to_end(key)
                except KeyError:
                    logger.warning(f"Key {key} not found in L1 cache during move_to_end")
                
                self.cache_stats['hits'] += 1
                logger.debug(f"L1 cache hit: {key}")
                return entry.data
            
            # Check L2 cache
            if key in self.l2_cache:
                compressed_entry = self.l2_cache[key]
                
                # Decompress data
                decompressed_data = self._decompress_data(compressed_entry.data)
                
                # Promote to L1 if frequently accessed
                if compressed_entry.access_count > 3:
                    self._promote_to_l1(key, decompressed_data, compressed_entry)
                
                compressed_entry.last_accessed = access_time
                compressed_entry.access_count += 1
                try:
                    self.l2_cache.move_to_end(key)
                except KeyError:
                    logger.warning(f"Key {key} not found in L2 cache during move_to_end")
                
                self.cache_stats['hits'] += 1
                self.cache_stats['decompressions'] += 1
                logger.debug(f"L2 cache hit: {key}")
                return decompressed_data
            
            # Cache miss
            self.cache_stats['misses'] += 1
            logger.debug(f"Cache miss: {key}")
            
            # Update prediction scores
            if self.prediction_enabled:
                self._update_prediction_scores(key, False)
            
            return default
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Put data into cache."""
        with self.cache_lock:
            # Calculate data size
            data_size = self._calculate_size(data)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=data_size,
                level=CacheLevel.L1_MEMORY
            )
            
            # Determine cache level based on size and policy
            if data_size > self.compression_threshold and data_size < self.max_memory_size * 0.1:
                # Put in L2 with compression
                compressed_data = self._compress_data(data)
                entry.data = compressed_data
                entry.compressed = True
                entry.level = CacheLevel.L2_COMPRESSED
                entry.size_bytes = len(compressed_data)
                
                self._put_l2(key, entry)
                self.cache_stats['compressions'] += 1
            else:
                # Put in L1
                self._put_l1(key, entry)
            
            # Update metadata
            self.cache_metadata[key] = {
                'created_at': entry.created_at.isoformat(),
                'level': entry.level.value,
                'size_bytes': entry.size_bytes,
                'ttl': ttl
            }
            
            # Trigger prefetching if enabled
            if self.prediction_enabled:
                self._trigger_prefetch_analysis(key)
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Remove entry from cache."""
        with self.cache_lock:
            removed = False
            
            if key in self.l1_cache:
                entry = self.l1_cache.pop(key)
                self.l1_size -= entry.size_bytes
                removed = True
            
            if key in self.l2_cache:
                entry = self.l2_cache.pop(key)
                self.l2_size -= entry.size_bytes
                removed = True
            
            if key in self.cache_metadata:
                del self.cache_metadata[key]
            
            if key in self.access_patterns:
                del self.access_patterns[key]
            
            return removed
    
    def clear(self):
        """Clear all cache levels."""
        with self.cache_lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.cache_metadata.clear()
            self.access_patterns.clear()
            self.l1_size = 0
            self.l2_size = 0
            
            logger.info("Cache cleared")
    
    def prefetch(self, keys: List[str], data_loader: Callable[[str], Any]):
        """Prefetch data for specified keys."""
        if not self.prediction_enabled:
            return
        
        for key in keys:
            if key not in self.l1_cache and key not in self.l2_cache:
                try:
                    data = data_loader(key)
                    if data is not None:
                        self.put(key, data)
                        self.cache_stats['predictions'] += 1
                        logger.debug(f"Prefetched: {key}")
                except Exception as e:
                    logger.error(f"Prefetch failed for {key}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.cache_lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = (self.cache_stats['hits'] / max(total_requests, 1)) * 100
            
            uptime = (datetime.now() - self.cache_stats['start_time']).total_seconds()
            
            return {
                'hit_rate': hit_rate,
                'miss_rate': 100 - hit_rate,
                'total_requests': total_requests,
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'evictions': self.cache_stats['evictions'],
                'compressions': self.cache_stats['compressions'],
                'decompressions': self.cache_stats['decompressions'],
                'predictions': self.cache_stats['predictions'],
                'prediction_hits': self.cache_stats['prediction_hits'],
                'prediction_accuracy': (self.cache_stats['prediction_hits'] / 
                                      max(self.cache_stats['predictions'], 1)) * 100,
                'l1_cache': {
                    'size_mb': self.l1_size / (1024 * 1024),
                    'max_size_mb': self.max_memory_size / (1024 * 1024),
                    'utilization': (self.l1_size / self.max_memory_size) * 100,
                    'entry_count': len(self.l1_cache)
                },
                'l2_cache': {
                    'size_mb': self.l2_size / (1024 * 1024),
                    'max_size_mb': self.max_compressed_size / (1024 * 1024),
                    'utilization': (self.l2_size / self.max_compressed_size) * 100,
                    'entry_count': len(self.l2_cache)
                },
                'access_patterns': len(self.access_patterns),
                'uptime_seconds': uptime,
                'requests_per_second': total_requests / max(uptime, 1)
            }
    
    def get_cache_efficiency(self) -> Dict[str, Any]:
        """Analyze cache efficiency and provide recommendations."""
        stats = self.get_cache_stats()
        
        recommendations = []
        efficiency_score = 0
        
        # Analyze hit rate
        hit_rate = stats['hit_rate']
        if hit_rate >= 90:
            efficiency_score += 40
        elif hit_rate >= 80:
            efficiency_score += 30
            recommendations.append("Good hit rate, but could be improved")
        elif hit_rate >= 70:
            efficiency_score += 20
            recommendations.append("Consider increasing cache size or improving prefetch algorithms")
        else:
            efficiency_score += 10
            recommendations.append("Low hit rate - cache configuration needs optimization")
        
        # Analyze utilization
        l1_util = stats['l1_cache']['utilization']
        l2_util = stats['l2_cache']['utilization']
        
        if 70 <= l1_util <= 90:
            efficiency_score += 20
        elif l1_util > 95:
            recommendations.append("L1 cache is overutilized - consider increasing size")
        elif l1_util < 50:
            recommendations.append("L1 cache is underutilized - consider decreasing size")
        
        if 60 <= l2_util <= 85:
            efficiency_score += 15
        elif l2_util > 90:
            recommendations.append("L2 cache is nearly full - consider increasing size")
        
        # Analyze prediction accuracy
        pred_accuracy = stats['prediction_accuracy']
        if pred_accuracy >= 80:
            efficiency_score += 15
        elif pred_accuracy >= 60:
            efficiency_score += 10
            recommendations.append("Prediction accuracy is moderate - tune prediction algorithms")
        else:
            efficiency_score += 5
            recommendations.append("Low prediction accuracy - review access patterns")
        
        # Analyze eviction rate
        eviction_rate = (stats['evictions'] / max(stats['total_requests'], 1)) * 100
        if eviction_rate < 5:
            efficiency_score += 10
        elif eviction_rate > 15:
            recommendations.append("High eviction rate - consider increasing cache size")
        
        return {
            'efficiency_score': min(100, efficiency_score),
            'performance_grade': self._calculate_performance_grade(efficiency_score),
            'recommendations': recommendations,
            'bottlenecks': self._identify_cache_bottlenecks(stats),
            'optimization_potential': 100 - efficiency_score
        }
    
    def start_prediction_engine(self):
        """Start the predictive caching engine."""
        if self.prediction_active:
            return
        
        self.prediction_active = True
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.prediction_thread.start()
        
        logger.info("Prediction engine started")
    
    def stop_prediction_engine(self):
        """Stop the predictive caching engine."""
        self.prediction_active = False
        if self.prediction_thread:
            self.prediction_thread.join(timeout=5)
        
        logger.info("Prediction engine stopped")
    
    def _put_l1(self, key: str, entry: CacheEntry):
        """Put entry in L1 cache."""
        # Check if we need to evict
        while self.l1_size + entry.size_bytes > self.max_memory_size and self.l1_cache:
            self._evict_l1()
        
        self.l1_cache[key] = entry
        self.l1_size += entry.size_bytes
    
    def _put_l2(self, key: str, entry: CacheEntry):
        """Put entry in L2 cache."""
        # Check if we need to evict
        while self.l2_size + entry.size_bytes > self.max_compressed_size and self.l2_cache:
            self._evict_l2()
        
        self.l2_cache[key] = entry
        self.l2_size += entry.size_bytes
    
    def _evict_l1(self):
        """Evict entry from L1 cache."""
        if not self.l1_cache:
            return
        
        if self.cache_policy == CachePolicy.LRU:
            # Remove least recently used
            key, entry = self.l1_cache.popitem(last=False)
        elif self.cache_policy == CachePolicy.LFU:
            # Remove least frequently used
            min_key = min(self.l1_cache.keys(), 
                         key=lambda k: self.l1_cache[k].access_count)
            entry = self.l1_cache.pop(min_key)
            key = min_key
        else:
            # Adaptive policy - consider multiple factors
            key = self._select_adaptive_eviction_candidate(self.l1_cache)
            entry = self.l1_cache.pop(key)
        
        self.l1_size -= entry.size_bytes
        self.cache_stats['evictions'] += 1
        
        # Consider moving to L2 if valuable
        if entry.access_count > 2:
            # Compress and move to L2
            compressed_data = self._compress_data(entry.data)
            entry.data = compressed_data
            entry.compressed = True
            entry.level = CacheLevel.L2_COMPRESSED
            entry.size_bytes = len(compressed_data)
            
            self._put_l2(key, entry)
            self.cache_stats['compressions'] += 1
        
        logger.debug(f"Evicted from L1: {key}")
    
    def _evict_l2(self):
        """Evict entry from L2 cache."""
        if not self.l2_cache:
            return
        
        # Always use LRU for L2
        key, entry = self.l2_cache.popitem(last=False)
        self.l2_size -= entry.size_bytes
        self.cache_stats['evictions'] += 1
        
        logger.debug(f"Evicted from L2: {key}")
    
    def _promote_to_l1(self, key: str, data: Any, l2_entry: CacheEntry):
        """Promote an entry from L2 to L1."""
        # Remove from L2
        self.l2_cache.pop(key, None)
        self.l2_size -= l2_entry.size_bytes
        
        # Create L1 entry
        l1_entry = CacheEntry(
            key=key,
            data=data,
            created_at=l2_entry.created_at,
            last_accessed=l2_entry.last_accessed,
            access_count=l2_entry.access_count,
            size_bytes=self._calculate_size(data),
            level=CacheLevel.L1_MEMORY
        )
        
        self._put_l1(key, l1_entry)
        logger.debug(f"Promoted to L1: {key}")
    
    def _select_adaptive_eviction_candidate(self, cache: OrderedDict) -> str:
        """Select eviction candidate using adaptive policy."""
        scores = {}
        current_time = datetime.now()
        
        for key, entry in cache.items():
            # Calculate composite score
            age_penalty = (current_time - entry.last_accessed).total_seconds() / 3600  # Hours
            frequency_bonus = entry.access_count
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            # Lower score = better candidate for eviction
            scores[key] = age_penalty + size_penalty - (frequency_bonus * 0.1)
        
        # Return key with highest score (worst candidate)
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _record_access(self, key: str, access_time: datetime):
        """Record access pattern for prediction."""
        self.access_history.append((key, access_time))
        
        if key not in self.access_patterns:
            self.access_patterns[key] = AccessPattern(
                key=key,
                access_times=deque(maxlen=100),
                frequency=0.0,
                last_access=access_time
            )
        
        pattern = self.access_patterns[key]
        pattern.access_times.append(access_time)
        pattern.last_access = access_time
        
        # Calculate frequency (accesses per hour)
        if len(pattern.access_times) > 1:
            time_span = (pattern.access_times[-1] - pattern.access_times[0]).total_seconds()
            pattern.frequency = len(pattern.access_times) / max(time_span / 3600, 0.1)
    
    def _update_prediction_scores(self, key: str, was_hit: bool):
        """Update prediction scores based on access outcomes."""
        if key in self.prediction_scores:
            if was_hit:
                self.cache_stats['prediction_hits'] += 1
                # Increase score for successful prediction
                self.prediction_scores[key] = min(1.0, self.prediction_scores[key] + 0.1)
            else:
                # Decrease score for missed prediction
                self.prediction_scores[key] = max(0.0, self.prediction_scores[key] - 0.05)
    
    def _trigger_prefetch_analysis(self, key: str):
        """Analyze if we should prefetch related data."""
        if not self.prediction_enabled:
            return
        
        # Simple prediction: prefetch keys that often follow this key
        related_keys = self._find_related_keys(key)
        for related_key in related_keys:
            if (related_key not in self.l1_cache and 
                related_key not in self.l2_cache and
                related_key not in self.prefetch_queue):
                self.prefetch_queue.append(related_key)
    
    def _find_related_keys(self, key: str) -> List[str]:
        """Find keys that are often accessed together."""
        related = []
        
        # Look at recent access history
        recent_accesses = list(self.access_history)[-100:]
        
        # Find keys accessed within 5 minutes of this key
        for i, (access_key, access_time) in enumerate(recent_accesses):
            if access_key == key:
                # Look at surrounding accesses
                for j in range(max(0, i-5), min(len(recent_accesses), i+6)):
                    if j != i:
                        related_key, related_time = recent_accesses[j]
                        time_diff = abs((access_time - related_time).total_seconds())
                        if time_diff <= 300 and related_key not in related:  # 5 minutes
                            related.append(related_key)
        
        return related[:3]  # Return top 3 related keys
    
    def _prediction_loop(self):
        """Main prediction loop."""
        while self.prediction_active:
            try:
                # Run prediction analysis every 30 seconds
                time.sleep(30)
                
                if not self.prediction_active:
                    break
                
                self._analyze_prediction_opportunities()
                self._adaptive_cache_sizing()
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
    
    def _analyze_prediction_opportunities(self):
        """Analyze access patterns to identify prediction opportunities."""
        current_time = datetime.now()
        
        for key, pattern in self.access_patterns.items():
            if key in self.l1_cache or key in self.l2_cache:
                continue  # Already cached
            
            # Predict based on frequency and recency
            time_since_last = (current_time - pattern.last_access).total_seconds() / 3600
            
            # High frequency + recent access = good prediction candidate
            if pattern.frequency > 1.0 and time_since_last < 1.0:
                prediction_score = pattern.frequency * (1.0 / max(time_since_last, 0.1))
                self.prediction_scores[key] = min(1.0, prediction_score / 10.0)
            else:
                # Decay prediction score
                if key in self.prediction_scores:
                    self.prediction_scores[key] *= 0.9
    
    def _adaptive_cache_sizing(self):
        """Adaptively adjust cache sizes based on performance."""
        stats = self.get_cache_stats()
        hit_rate = stats['hit_rate']
        
        if hit_rate < self.adaptive_config['hit_rate_threshold']:
            # Increase cache size if hit rate is low
            increase_factor = 1 + self.adaptive_config['resize_factor']
            new_memory_size = min(
                self.max_memory_size * increase_factor,
                self.adaptive_config['max_size_mb'] * 1024 * 1024
            )
            
            if new_memory_size != self.max_memory_size:
                self.max_memory_size = int(new_memory_size)
                logger.info(f"Increased L1 cache size to {self.max_memory_size / (1024*1024):.1f}MB")
        
        elif hit_rate > 95 and stats['l1_cache']['utilization'] < 70:
            # Decrease cache size if hit rate is very high and utilization is low
            decrease_factor = 1 - self.adaptive_config['resize_factor']
            new_memory_size = max(
                self.max_memory_size * decrease_factor,
                self.adaptive_config['min_size_mb'] * 1024 * 1024
            )
            
            if new_memory_size != self.max_memory_size:
                self.max_memory_size = int(new_memory_size)
                logger.info(f"Decreased L1 cache size to {self.max_memory_size / (1024*1024):.1f}MB")
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for L2 storage."""
        serialized = pickle.dumps(data)
        return zlib.compress(serialized, level=6)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from L2 storage."""
        decompressed = zlib.decompress(compressed_data)
        return SafePickleHandler.safe_load(decompressed)
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            return len(pickle.dumps(data))
        except Exception:
            # Fallback size estimation
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (int, float)):
                return 8
            elif isinstance(data, dict):
                return sum(len(str(k)) + len(str(v)) for k, v in data.items())
            elif isinstance(data, list):
                return sum(len(str(item)) for item in data)
            else:
                return 1024  # Default estimate
    
    def _calculate_performance_grade(self, efficiency_score: float) -> str:
        """Calculate performance grade based on efficiency score."""
        if efficiency_score >= 90:
            return 'A+'
        elif efficiency_score >= 85:
            return 'A'
        elif efficiency_score >= 80:
            return 'B+'
        elif efficiency_score >= 75:
            return 'B'
        elif efficiency_score >= 70:
            return 'C+'
        elif efficiency_score >= 65:
            return 'C'
        elif efficiency_score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _identify_cache_bottlenecks(self, stats: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks in the cache system."""
        bottlenecks = []
        
        if stats['hit_rate'] < 70:
            bottlenecks.append("Low hit rate - cache size or policy needs optimization")
        
        if stats['l1_cache']['utilization'] > 95:
            bottlenecks.append("L1 cache is constantly full - increase size")
        
        if stats['l2_cache']['utilization'] > 90:
            bottlenecks.append("L2 cache is nearly full - increase size or improve eviction")
        
        if stats['prediction_accuracy'] < 60:
            bottlenecks.append("Poor prediction accuracy - review access patterns")
        
        eviction_rate = (stats['evictions'] / max(stats['total_requests'], 1)) * 100
        if eviction_rate > 20:
            bottlenecks.append("High eviction rate - cache thrashing detected")
        
        return bottlenecks
    
    def shutdown(self):
        """Shutdown the cache system."""
        self.stop_prediction_engine()
        self.clear()
        logger.info("Smart Analytics Cache shutdown")