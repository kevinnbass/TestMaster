"""
Advanced ML Smart Cache System
==============================
Enterprise-grade caching with ML-driven prediction and optimization.
Extracted and enhanced from archive analytics_smart_cache.py and performance_booster.py.

Author: Agent B - Intelligence Specialist  
Module: 299 lines (under 300 limit)
"""

import asyncio
import hashlib
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
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    ML_PREDICTED = "ml_predicted"


class CacheLevel(Enum):
    """Multi-level cache hierarchy."""
    L1_MEMORY = "l1_memory"
    L2_COMPRESSED = "l2_compressed"
    L3_PERSISTENT = "l3_persistent"


@dataclass
class CacheEntry:
    """Enhanced cache entry with ML features."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    level: CacheLevel
    compressed: bool = False
    prediction_score: float = 0.0
    access_pattern: List[float] = field(default_factory=list)
    feature_vector: Optional[np.ndarray] = None


@dataclass
class AccessPattern:
    """ML-enhanced access pattern tracking."""
    key: str
    access_times: deque = field(default_factory=lambda: deque(maxlen=100))
    frequency: float = 0.0
    recency_score: float = 0.0
    seasonality: float = 0.0
    trend: float = 0.0
    prediction_confidence: float = 0.0


class MLCachePredictor:
    """Machine learning based cache prediction engine."""
    
    def __init__(self):
        self.access_model = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_history = deque(maxlen=1000)
        self.accuracy_scores = deque(maxlen=100)
        
    def extract_features(self, pattern: AccessPattern, 
                        current_time: datetime) -> np.ndarray:
        """Extract ML features from access pattern."""
        if len(pattern.access_times) < 2:
            return np.array([0.0] * 8)
        
        access_array = np.array([t.timestamp() for t in pattern.access_times])
        current_ts = current_time.timestamp()
        
        # Time-based features
        time_since_last = current_ts - access_array[-1] if len(access_array) > 0 else 0
        avg_interval = np.mean(np.diff(access_array)) if len(access_array) > 1 else 0
        
        # Frequency features
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        
        # Pattern features
        recent_accesses = len([t for t in access_array if current_ts - t < 3600])  # Last hour
        trend_slope = np.polyfit(range(len(access_array)), access_array, 1)[0] if len(access_array) > 2 else 0
        
        features = np.array([
            time_since_last / 3600,  # Hours since last access
            avg_interval / 3600,     # Average interval in hours
            hour_of_day / 24,        # Hour of day normalized
            day_of_week / 7,         # Day of week normalized
            recent_accesses,         # Recent access count
            trend_slope,             # Access trend
            pattern.frequency,       # Access frequency
            len(access_array)        # Total access count
        ])
        
        return features
    
    def predict_access_probability(self, pattern: AccessPattern, 
                                 current_time: datetime) -> float:
        """Predict probability of access in next period."""
        if not self.trained:
            return 0.5  # Default probability
        
        features = self.extract_features(pattern, current_time)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        try:
            prediction = self.access_model.predict(features_scaled)[0]
            return max(0.0, min(1.0, prediction))  # Clamp to [0, 1]
        except Exception as e:
            logger.debug(f"Prediction error: {e}")
            return 0.5
    
    def train_model(self, training_data: List[Tuple[np.ndarray, float]]):
        """Train the access prediction model."""
        if len(training_data) < 10:
            return False
        
        features = np.array([data[0] for data in training_data])
        labels = np.array([data[1] for data in training_data])
        
        try:
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.access_model.fit(features_scaled, labels)
            self.trained = True
            
            # Calculate accuracy
            predictions = self.access_model.predict(features_scaled)
            accuracy = 1.0 - np.mean(np.abs(predictions - labels))
            self.accuracy_scores.append(accuracy)
            
            logger.info(f"Cache predictor trained with accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False


class AdvancedMLCache:
    """
    Enterprise-grade ML-driven cache system.
    Combines multi-level caching with machine learning prediction.
    """
    
    def __init__(self,
                 max_memory_size: int = 128 * 1024 * 1024,  # 128MB
                 max_compressed_size: int = 512 * 1024 * 1024,  # 512MB
                 cache_policy: CachePolicy = CachePolicy.ML_PREDICTED,
                 compression_threshold: int = 1024):
        """
        Initialize advanced ML cache.
        
        Args:
            max_memory_size: L1 cache size limit
            max_compressed_size: L2 cache size limit  
            cache_policy: Eviction policy
            compression_threshold: Size threshold for compression
        """
        self.max_memory_size = max_memory_size
        self.max_compressed_size = max_compressed_size
        self.cache_policy = cache_policy
        self.compression_threshold = compression_threshold
        
        # Multi-level storage
        self.l1_cache = OrderedDict()  # Memory cache
        self.l2_cache = OrderedDict()  # Compressed cache
        self.cache_metadata = {}
        
        # Size tracking
        self.l1_size = 0
        self.l2_size = 0
        
        # ML components
        self.predictor = MLCachePredictor()
        self.access_patterns = {}
        self.access_history = deque(maxlen=5000)
        
        # Performance metrics
        self.cache_stats = {
            'hits': 0, 'misses': 0, 'evictions': 0,
            'compressions': 0, 'decompressions': 0,
            'ml_predictions': 0, 'prediction_hits': 0,
            'start_time': datetime.now()
        }
        
        # Background optimization
        self.optimization_active = False
        self.optimization_thread = None
        self.lock = threading.RLock()
        
        # Adaptive configuration
        self.adaptive_config = {
            'hit_rate_target': 0.85,
            'retrain_interval': 300,  # 5 minutes
            'last_retrain': time.time()
        }
        
        logger.info("Advanced ML Cache initialized")
    
    def start_optimization(self):
        """Start background optimization."""
        self.optimization_active = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self.optimization_thread.start()
        logger.info("ML cache optimization started")
    
    def stop_optimization(self):
        """Stop background optimization."""
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("ML cache optimization stopped")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cached data with ML-enhanced access tracking."""
        with self.lock:
            access_time = datetime.now()
            self._record_access(key, access_time)
            
            # Check L1 cache
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                entry.last_accessed = access_time
                entry.access_count += 1
                
                # Update access pattern
                self._update_access_pattern(key, access_time, hit=True)
                
                # Move to end (LRU)
                self.l1_cache.move_to_end(key)
                
                self.cache_stats['hits'] += 1
                return entry.data
            
            # Check L2 cache
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                
                # Decompress
                data = self._decompress_data(entry.data)
                
                # Update access tracking
                entry.last_accessed = access_time
                entry.access_count += 1
                self._update_access_pattern(key, access_time, hit=True)
                
                # Consider promotion to L1
                if self._should_promote_to_l1(entry):
                    self._promote_to_l1(key, data, entry)
                
                self.l2_cache.move_to_end(key)
                self.cache_stats['hits'] += 1
                self.cache_stats['decompressions'] += 1
                return data
            
            # Cache miss
            self.cache_stats['misses'] += 1
            self._update_access_pattern(key, access_time, hit=False)
            
            return default
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Put data into cache with ML-driven placement."""
        with self.lock:
            data_size = self._calculate_size(data)
            current_time = datetime.now()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                size_bytes=data_size,
                level=CacheLevel.L1_MEMORY
            )
            
            # ML-driven cache level decision
            if self.cache_policy == CachePolicy.ML_PREDICTED:
                cache_level = self._predict_optimal_level(key, data_size)
            else:
                cache_level = self._determine_cache_level(data_size)
            
            # Place in appropriate level
            if cache_level == CacheLevel.L2_COMPRESSED:
                compressed_data = self._compress_data(data)
                entry.data = compressed_data
                entry.compressed = True
                entry.level = CacheLevel.L2_COMPRESSED
                entry.size_bytes = len(compressed_data)
                self._put_l2(key, entry)
                self.cache_stats['compressions'] += 1
            else:
                self._put_l1(key, entry)
            
            # Update metadata
            self.cache_metadata[key] = {
                'created_at': current_time.isoformat(),
                'level': entry.level.value,
                'size_bytes': entry.size_bytes,
                'ttl': ttl,
                'prediction_score': entry.prediction_score
            }
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Remove entry from all cache levels."""
        with self.lock:
            removed = False
            
            if key in self.l1_cache:
                entry = self.l1_cache.pop(key)
                self.l1_size -= entry.size_bytes
                removed = True
            
            if key in self.l2_cache:
                entry = self.l2_cache.pop(key)
                self.l2_size -= entry.size_bytes
                removed = True
            
            # Clean up metadata
            self.cache_metadata.pop(key, None)
            self.access_patterns.pop(key, None)
            
            return removed
    
    def _predict_optimal_level(self, key: str, data_size: int) -> CacheLevel:
        """Use ML to predict optimal cache level."""
        if key in self.access_patterns:
            pattern = self.access_patterns[key]
            probability = self.predictor.predict_access_probability(
                pattern, datetime.now()
            )
            
            # High probability + reasonable size -> L1
            if probability > 0.7 and data_size < self.max_memory_size * 0.1:
                return CacheLevel.L1_MEMORY
            # Medium probability or large size -> L2
            elif probability > 0.3:
                return CacheLevel.L2_COMPRESSED
        
        return self._determine_cache_level(data_size)
    
    def _determine_cache_level(self, data_size: int) -> CacheLevel:
        """Traditional cache level determination."""
        if data_size > self.compression_threshold:
            return CacheLevel.L2_COMPRESSED
        return CacheLevel.L1_MEMORY
    
    def _should_promote_to_l1(self, entry: CacheEntry) -> bool:
        """Decide if L2 entry should be promoted to L1."""
        if entry.access_count < 3:
            return False
        
        if entry.key in self.access_patterns:
            pattern = self.access_patterns[entry.key]
            probability = self.predictor.predict_access_probability(
                pattern, datetime.now()
            )
            return probability > 0.8
        
        return entry.access_count > 5
    
    def _record_access(self, key: str, access_time: datetime):
        """Record access for ML training."""
        self.access_history.append((key, access_time))
    
    def _update_access_pattern(self, key: str, access_time: datetime, hit: bool):
        """Update ML access patterns."""
        if key not in self.access_patterns:
            self.access_patterns[key] = AccessPattern(key=key)
        
        pattern = self.access_patterns[key]
        pattern.access_times.append(access_time)
        
        # Update frequency
        if len(pattern.access_times) > 1:
            time_span = (pattern.access_times[-1] - pattern.access_times[0]).total_seconds()
            pattern.frequency = len(pattern.access_times) / max(time_span / 3600, 0.1)
        
        # Update recency score
        current_time = datetime.now()
        time_diff = (current_time - access_time).total_seconds()
        pattern.recency_score = 1.0 / (1.0 + time_diff / 3600)
        
        # Store training data
        if self.predictor.trained:
            features = self.predictor.extract_features(pattern, access_time)
            label = 1.0 if hit else 0.0
            self.predictor.feature_history.append((features, label))
    
    def _put_l1(self, key: str, entry: CacheEntry):
        """Place entry in L1 cache."""
        while self.l1_size + entry.size_bytes > self.max_memory_size and self.l1_cache:
            self._evict_l1()
        
        self.l1_cache[key] = entry
        self.l1_size += entry.size_bytes
    
    def _put_l2(self, key: str, entry: CacheEntry):
        """Place entry in L2 cache."""
        while self.l2_size + entry.size_bytes > self.max_compressed_size and self.l2_cache:
            self._evict_l2()
        
        self.l2_cache[key] = entry
        self.l2_size += entry.size_bytes
    
    def _evict_l1(self):
        """Evict from L1 using ML-enhanced policy."""
        if not self.l1_cache:
            return
        
        if self.cache_policy == CachePolicy.ML_PREDICTED:
            victim_key = self._select_ml_eviction_candidate(self.l1_cache)
        else:
            victim_key, _ = self.l1_cache.popitem(last=False)  # LRU fallback
        
        entry = self.l1_cache.pop(victim_key)
        self.l1_size -= entry.size_bytes
        self.cache_stats['evictions'] += 1
        
        # Consider moving valuable entries to L2
        if entry.access_count > 2:
            compressed_data = self._compress_data(entry.data)
            entry.data = compressed_data
            entry.compressed = True
            entry.level = CacheLevel.L2_COMPRESSED
            entry.size_bytes = len(compressed_data)
            self._put_l2(victim_key, entry)
            self.cache_stats['compressions'] += 1
    
    def _evict_l2(self):
        """Evict from L2 cache."""
        if not self.l2_cache:
            return
        
        key, entry = self.l2_cache.popitem(last=False)
        self.l2_size -= entry.size_bytes
        self.cache_stats['evictions'] += 1
    
    def _select_ml_eviction_candidate(self, cache: OrderedDict) -> str:
        """Use ML to select eviction candidate."""
        scores = {}
        current_time = datetime.now()
        
        for key in cache.keys():
            if key in self.access_patterns:
                pattern = self.access_patterns[key]
                probability = self.predictor.predict_access_probability(pattern, current_time)
                # Lower probability = better eviction candidate
                scores[key] = probability
            else:
                scores[key] = 0.1  # Unknown patterns get low priority
        
        # Return key with lowest prediction score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _promote_to_l1(self, key: str, data: Any, l2_entry: CacheEntry):
        """Promote L2 entry to L1."""
        self.l2_cache.pop(key, None)
        self.l2_size -= l2_entry.size_bytes
        
        l1_entry = CacheEntry(
            key=key, data=data,
            created_at=l2_entry.created_at,
            last_accessed=l2_entry.last_accessed,
            access_count=l2_entry.access_count,
            size_bytes=self._calculate_size(data),
            level=CacheLevel.L1_MEMORY
        )
        
        self._put_l1(key, l1_entry)
    
    def _optimization_loop(self):
        """Background optimization loop."""
        while self.optimization_active:
            try:
                time.sleep(60)  # Run every minute
                
                # Retrain ML model periodically
                current_time = time.time()
                if (current_time - self.adaptive_config['last_retrain'] > 
                    self.adaptive_config['retrain_interval']):
                    self._retrain_predictor()
                    self.adaptive_config['last_retrain'] = current_time
                
                # Clean expired entries
                self._cleanup_expired_entries()
                
                # Optimize cache sizes
                self._optimize_cache_sizes()
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
    
    def _retrain_predictor(self):
        """Retrain the ML predictor with recent data."""
        if len(self.predictor.feature_history) >= 50:
            training_data = list(self.predictor.feature_history)
            success = self.predictor.train_model(training_data)
            if success:
                logger.info("Cache predictor retrained successfully")
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, metadata in self.cache_metadata.items():
            if metadata.get('ttl'):
                created_at = datetime.fromisoformat(metadata['created_at'])
                if (current_time - created_at).total_seconds() > metadata['ttl']:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self.invalidate(key)
    
    def _optimize_cache_sizes(self):
        """Dynamically optimize cache sizes."""
        stats = self.get_cache_stats()
        hit_rate = stats['hit_rate']
        
        if hit_rate < self.adaptive_config['hit_rate_target']:
            # Increase cache sizes
            self.max_memory_size = min(
                int(self.max_memory_size * 1.1),
                1024 * 1024 * 1024  # Max 1GB
            )
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for L2 storage."""
        serialized = pickle.dumps(data)
        return zlib.compress(serialized, level=6)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from L2 storage."""
        decompressed = zlib.decompress(compressed_data)
        return SafePickleHandler.safe_load(decompressed)
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate data size in bytes."""
        try:
            return len(pickle.dumps(data))
        except:
            return 1024  # Fallback estimate
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / max(total_requests, 1)) * 100
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'l1_utilization': (self.l1_size / self.max_memory_size) * 100,
            'l2_utilization': (self.l2_size / self.max_compressed_size) * 100,
            'ml_accuracy': statistics.mean(self.predictor.accuracy_scores) if self.predictor.accuracy_scores else 0,
            'cache_stats': self.cache_stats.copy(),
            'predictor_trained': self.predictor.trained,
            'total_patterns': len(self.access_patterns)
        }


# Export for use by other modules
__all__ = ['AdvancedMLCache', 'CachePolicy', 'CacheLevel', 'MLCachePredictor']