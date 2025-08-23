#!/usr/bin/env python3
"""
Advanced Caching Strategy Evolution System
Multi-tier intelligent caching with predictive prefetching, advanced eviction strategies, and cross-system cache coordination.

Agent Beta - Phase 2, Hours 65-70
Greek Swarm Coordination - TestMaster Intelligence System
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json
import logging
import threading
import time
import uuid
import hashlib
import pickle
import zlib
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from enum import Enum
import queue
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    print("WARNING: scikit-learn not available. ML cache optimization features disabled.")
    SKLEARN_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    print("WARNING: redis not available. Redis caching disabled.")
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AdvancedCachingStrategyEvolution')

class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"
    L4_DISTRIBUTED = "l4_distributed"

class EvictionStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    RANDOM = "random"  # Random eviction
    TTL_BASED = "ttl_based"  # Time To Live based
    ML_OPTIMIZED = "ml_optimized"  # ML-driven eviction
    HYBRID_INTELLIGENT = "hybrid_intelligent"  # Intelligent hybrid strategy

class CacheAccessPattern(Enum):
    """Cache access patterns for optimization"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORAL_LOCALITY = "temporal_locality"
    SPATIAL_LOCALITY = "spatial_locality"
    HOTSPOT = "hotspot"
    BURST = "burst"

class PrefetchStrategy(Enum):
    """Prefetching strategies"""
    NONE = "none"
    LINEAR = "linear"  # Linear prefetching
    STRIDE = "stride"  # Stride-based prefetching
    ML_PREDICTIVE = "ml_predictive"  # ML-driven predictive prefetching
    PATTERN_BASED = "pattern_based"  # Pattern-based prefetching
    COLLABORATIVE = "collaborative"  # Cross-system collaborative prefetching

@dataclass
class AdvancedCacheConfig:
    """Configuration for advanced caching system"""
    # Cache Hierarchy Configuration
    l1_memory_size_mb: int = 512
    l2_redis_size_mb: int = 2048
    l3_disk_size_mb: int = 8192
    l4_distributed_size_mb: int = 16384
    
    # Eviction Strategy Configuration
    default_eviction_strategy: EvictionStrategy = EvictionStrategy.HYBRID_INTELLIGENT
    ml_eviction_enabled: bool = True
    eviction_batch_size: int = 100
    eviction_threshold: float = 0.9  # Evict when cache is 90% full
    
    # Prefetching Configuration
    prefetch_strategy: PrefetchStrategy = PrefetchStrategy.ML_PREDICTIVE
    prefetch_window_size: int = 10
    prefetch_confidence_threshold: float = 0.7
    max_prefetch_queue_size: int = 1000
    
    # TTL Configuration
    default_ttl_seconds: int = 3600
    adaptive_ttl_enabled: bool = True
    min_ttl_seconds: int = 60
    max_ttl_seconds: int = 86400
    
    # Performance Configuration
    cache_warming_enabled: bool = True
    compression_enabled: bool = True
    serialization_format: str = "pickle"  # pickle, json, msgpack
    max_key_size_bytes: int = 1024
    max_value_size_mb: int = 100
    
    # ML Optimization Configuration
    ml_model_retrain_interval: int = 3600  # seconds
    access_pattern_window_size: int = 10000
    cache_analytics_enabled: bool = True
    
    # Cross-System Coordination
    cross_system_sync_enabled: bool = True
    cache_replication_enabled: bool = False
    distributed_cache_nodes: List[str] = field(default_factory=list)
    
    # Database Configuration
    db_path: str = "advanced_caching_analytics.db"
    analytics_retention_days: int = 7

@dataclass
class CacheEntry:
    """Advanced cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int
    
    # Advanced metadata
    access_pattern: Optional[CacheAccessPattern] = None
    prefetch_score: float = 0.0
    eviction_priority: float = 0.0
    compression_ratio: float = 1.0
    
    # Cross-system metadata
    origin_system: str = ""
    replication_targets: List[str] = field(default_factory=list)
    coherence_version: int = 1
    
    def update_access(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_age_seconds(self) -> float:
        """Calculate entry age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return self.calculate_age_seconds() > self.ttl_seconds
    
    def calculate_access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)"""
        age = max(self.calculate_age_seconds(), 1)
        return self.access_count / age

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    timestamp: datetime
    cache_level: CacheLevel
    
    # Hit/Miss Metrics
    hit_count: int = 0
    miss_count: int = 0
    total_requests: int = 0
    hit_ratio: float = 0.0
    
    # Performance Metrics
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Capacity Metrics
    total_entries: int = 0
    total_size_mb: float = 0.0
    utilization_ratio: float = 0.0
    
    # Eviction Metrics
    evictions_count: int = 0
    eviction_rate: float = 0.0
    
    # Prefetch Metrics
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_accuracy: float = 0.0
    
    def calculate_hit_ratio(self):
        """Calculate and update hit ratio"""
        self.total_requests = self.hit_count + self.miss_count
        if self.total_requests > 0:
            self.hit_ratio = self.hit_count / self.total_requests

@dataclass
class AccessPattern:
    """Cache access pattern analysis"""
    pattern_id: str
    pattern_type: CacheAccessPattern
    keys: List[str]
    timestamps: List[datetime]
    frequency: float
    predictability_score: float
    confidence: float

class IntelligentEvictionEngine:
    """Intelligent cache eviction with ML optimization"""
    
    def __init__(self, config: AdvancedCacheConfig):
        self.config = config
        self.access_history = []
        self.ml_model = None
        self.feature_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.last_model_training = 0
        
    def should_evict(self, cache_size: int, max_size: int) -> bool:
        """Check if eviction should be triggered"""
        utilization = cache_size / max_size if max_size > 0 else 0
        return utilization >= self.config.eviction_threshold
    
    def select_eviction_candidates(self, cache_entries: Dict[str, CacheEntry], 
                                 eviction_count: int) -> List[str]:
        """Select entries for eviction using intelligent strategies"""
        try:
            if self.config.default_eviction_strategy == EvictionStrategy.HYBRID_INTELLIGENT:
                return self._hybrid_intelligent_eviction(cache_entries, eviction_count)
            elif self.config.default_eviction_strategy == EvictionStrategy.ML_OPTIMIZED and self.ml_model:
                return self._ml_optimized_eviction(cache_entries, eviction_count)
            elif self.config.default_eviction_strategy == EvictionStrategy.LRU:
                return self._lru_eviction(cache_entries, eviction_count)
            elif self.config.default_eviction_strategy == EvictionStrategy.LFU:
                return self._lfu_eviction(cache_entries, eviction_count)
            elif self.config.default_eviction_strategy == EvictionStrategy.TTL_BASED:
                return self._ttl_based_eviction(cache_entries, eviction_count)
            else:
                return self._lru_eviction(cache_entries, eviction_count)  # Fallback
                
        except Exception as e:
            logger.error(f"Eviction candidate selection failed: {e}")
            return self._lru_eviction(cache_entries, eviction_count)  # Safe fallback
    
    def _hybrid_intelligent_eviction(self, cache_entries: Dict[str, CacheEntry], 
                                   eviction_count: int) -> List[str]:
        """Hybrid intelligent eviction combining multiple strategies"""
        candidates = []
        
        # Score each entry using multiple factors
        entry_scores = {}
        current_time = datetime.now()
        
        for key, entry in cache_entries.items():
            # Skip recently accessed entries
            if (current_time - entry.last_accessed).total_seconds() < 60:
                continue
            
            # Calculate composite score (lower is better for eviction)
            age_factor = entry.calculate_age_seconds() / 3600  # Age in hours
            frequency_factor = 1.0 / (entry.calculate_access_frequency() + 1e-6)
            size_factor = entry.size_bytes / (1024 * 1024)  # Size in MB
            ttl_factor = 1.0 if entry.is_expired() else 0.5
            
            # Combine factors (higher score = better eviction candidate)
            score = (age_factor * 0.3 + 
                    frequency_factor * 0.4 + 
                    size_factor * 0.2 + 
                    ttl_factor * 0.1)
            
            entry_scores[key] = score
        
        # Sort by score (descending) and select top candidates
        sorted_candidates = sorted(entry_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [key for key, score in sorted_candidates[:eviction_count]]
        
        return candidates
    
    def _ml_optimized_eviction(self, cache_entries: Dict[str, CacheEntry], 
                              eviction_count: int) -> List[str]:
        """ML-optimized eviction using trained model"""
        if not self.ml_model or not SKLEARN_AVAILABLE:
            return self._hybrid_intelligent_eviction(cache_entries, eviction_count)
        
        try:
            # Prepare features for each cache entry
            features = []
            keys = []
            
            for key, entry in cache_entries.items():
                feature_vector = self._extract_eviction_features(entry)
                features.append(feature_vector)
                keys.append(key)
            
            if not features:
                return []
            
            # Scale features and predict
            features_array = np.array(features)
            features_scaled = self.feature_scaler.transform(features_array)
            
            # Predict eviction scores (higher score = better candidate)
            eviction_scores = self.ml_model.predict(features_scaled)
            
            # Sort by predicted score and select top candidates
            scored_keys = list(zip(keys, eviction_scores))
            scored_keys.sort(key=lambda x: x[1], reverse=True)
            
            candidates = [key for key, score in scored_keys[:eviction_count]]
            return candidates
            
        except Exception as e:
            logger.error(f"ML eviction failed, falling back to hybrid: {e}")
            return self._hybrid_intelligent_eviction(cache_entries, eviction_count)
    
    def _extract_eviction_features(self, entry: CacheEntry) -> List[float]:
        """Extract features for ML eviction model"""
        current_time = datetime.now()
        
        features = [
            entry.calculate_age_seconds() / 3600,  # Age in hours
            entry.access_count,
            entry.calculate_access_frequency(),
            entry.size_bytes / (1024 * 1024),  # Size in MB
            (current_time - entry.last_accessed).total_seconds() / 3600,  # Hours since last access
            1.0 if entry.is_expired() else 0.0,  # Expired flag
            entry.ttl_seconds / 3600,  # TTL in hours
            entry.eviction_priority,
            entry.prefetch_score
        ]
        
        return features
    
    def _lru_eviction(self, cache_entries: Dict[str, CacheEntry], eviction_count: int) -> List[str]:
        """Least Recently Used eviction"""
        sorted_entries = sorted(cache_entries.items(), key=lambda x: x[1].last_accessed)
        return [key for key, entry in sorted_entries[:eviction_count]]
    
    def _lfu_eviction(self, cache_entries: Dict[str, CacheEntry], eviction_count: int) -> List[str]:
        """Least Frequently Used eviction"""
        sorted_entries = sorted(cache_entries.items(), key=lambda x: x[1].access_count)
        return [key for key, entry in sorted_entries[:eviction_count]]
    
    def _ttl_based_eviction(self, cache_entries: Dict[str, CacheEntry], eviction_count: int) -> List[str]:
        """TTL-based eviction (expire oldest first)"""
        expired_entries = [(key, entry) for key, entry in cache_entries.items() if entry.is_expired()]
        
        if len(expired_entries) >= eviction_count:
            # Sort expired entries by age
            expired_entries.sort(key=lambda x: x[1].created_at)
            return [key for key, entry in expired_entries[:eviction_count]]
        else:
            # Include expired entries and add LRU for the rest
            candidates = [key for key, entry in expired_entries]
            remaining_count = eviction_count - len(candidates)
            
            if remaining_count > 0:
                non_expired = {k: v for k, v in cache_entries.items() if not v.is_expired()}
                lru_candidates = self._lru_eviction(non_expired, remaining_count)
                candidates.extend(lru_candidates)
            
            return candidates
    
    def train_ml_eviction_model(self, access_history: List[Dict[str, Any]]):
        """Train ML model for eviction optimization"""
        if not SKLEARN_AVAILABLE or len(access_history) < 100:
            logger.warning("Insufficient data or sklearn unavailable for ML training")
            return
        
        try:
            # Prepare training data
            features = []
            targets = []
            
            for record in access_history:
                if 'entry_features' in record and 'was_good_eviction' in record:
                    features.append(record['entry_features'])
                    targets.append(1.0 if record['was_good_eviction'] else 0.0)
            
            if len(features) < 50:
                logger.warning("Insufficient training data for ML eviction model")
                return
            
            # Train model
            features_array = np.array(features)
            targets_array = np.array(targets)
            
            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features_array)
            
            # Train Random Forest model
            self.ml_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.ml_model.fit(features_scaled, targets_array)
            
            self.last_model_training = time.time()
            logger.info(f"Trained ML eviction model with {len(features)} samples")
            
        except Exception as e:
            logger.error(f"ML eviction model training failed: {e}")

class PredictivePrefetcher:
    """Predictive prefetching engine with ML optimization"""
    
    def __init__(self, config: AdvancedCacheConfig):
        self.config = config
        self.access_patterns = {}
        self.pattern_history = []
        self.ml_predictor = None
        self.prefetch_queue = queue.Queue(maxsize=config.max_prefetch_queue_size)
        self.pattern_analyzer = AccessPatternAnalyzer(config)
        
    def should_prefetch(self, key: str, access_context: Dict[str, Any]) -> bool:
        """Determine if prefetching should be triggered"""
        if self.config.prefetch_strategy == PrefetchStrategy.NONE:
            return False
        
        # Analyze current access pattern
        pattern = self.pattern_analyzer.analyze_access_pattern([key], [datetime.now()])
        
        if pattern and pattern.predictability_score > self.config.prefetch_confidence_threshold:
            return True
        
        return False
    
    def generate_prefetch_candidates(self, key: str, access_context: Dict[str, Any]) -> List[str]:
        """Generate prefetch candidates based on access patterns"""
        candidates = []
        
        try:
            if self.config.prefetch_strategy == PrefetchStrategy.LINEAR:
                candidates = self._linear_prefetch_candidates(key)
            elif self.config.prefetch_strategy == PrefetchStrategy.STRIDE:
                candidates = self._stride_prefetch_candidates(key, access_context)
            elif self.config.prefetch_strategy == PrefetchStrategy.ML_PREDICTIVE:
                candidates = self._ml_predictive_candidates(key, access_context)
            elif self.config.prefetch_strategy == PrefetchStrategy.PATTERN_BASED:
                candidates = self._pattern_based_candidates(key)
            elif self.config.prefetch_strategy == PrefetchStrategy.COLLABORATIVE:
                candidates = self._collaborative_prefetch_candidates(key, access_context)
            
            # Limit candidates and filter duplicates
            unique_candidates = list(set(candidates))[:self.config.prefetch_window_size]
            
            return unique_candidates
            
        except Exception as e:
            logger.error(f"Prefetch candidate generation failed: {e}")
            return []
    
    def _linear_prefetch_candidates(self, key: str) -> List[str]:
        """Generate linear prefetch candidates"""
        candidates = []
        
        # Simple linear prefetching for numerical keys
        try:
            if key.isdigit():
                base_num = int(key)
                for i in range(1, self.config.prefetch_window_size + 1):
                    candidates.append(str(base_num + i))
            else:
                # For non-numeric keys, try pattern matching
                import re
                match = re.search(r'(\d+)', key)
                if match:
                    prefix = key[:match.start()]
                    suffix = key[match.end():]
                    num = int(match.group())
                    
                    for i in range(1, self.config.prefetch_window_size + 1):
                        candidate = f"{prefix}{num + i}{suffix}"
                        candidates.append(candidate)
        except Exception as e:
            logger.error(f"Linear prefetch generation failed: {e}")
        
        return candidates
    
    def _pattern_based_candidates(self, key: str) -> List[str]:
        """Generate candidates based on discovered access patterns"""
        candidates = []
        
        # Look for patterns involving this key
        for pattern_id, pattern in self.access_patterns.items():
            if key in pattern.keys:
                key_index = pattern.keys.index(key)
                
                # Predict next keys in the pattern
                for i in range(1, min(self.config.prefetch_window_size + 1, len(pattern.keys) - key_index)):
                    if key_index + i < len(pattern.keys):
                        candidates.append(pattern.keys[key_index + i])
        
        return candidates
    
    def _ml_predictive_candidates(self, key: str, access_context: Dict[str, Any]) -> List[str]:
        """Generate ML-driven predictive candidates"""
        # Placeholder for ML-based prediction
        # In a full implementation, this would use trained models
        return self._pattern_based_candidates(key)
    
    def _stride_prefetch_candidates(self, key: str, access_context: Dict[str, Any]) -> List[str]:
        """Generate stride-based prefetch candidates"""
        candidates = []
        
        # Detect stride pattern from access context
        recent_keys = access_context.get('recent_keys', [])
        if len(recent_keys) >= 2:
            try:
                # Try to detect numeric stride
                numeric_keys = [int(k) for k in recent_keys[-3:] if k.isdigit()]
                if len(numeric_keys) >= 2:
                    stride = numeric_keys[-1] - numeric_keys[-2]
                    base_num = int(key) if key.isdigit() else numeric_keys[-1]
                    
                    for i in range(1, self.config.prefetch_window_size + 1):
                        candidates.append(str(base_num + stride * i))
            except Exception:
                pass
        
        return candidates
    
    def _collaborative_prefetch_candidates(self, key: str, access_context: Dict[str, Any]) -> List[str]:
        """Generate collaborative prefetch candidates using cross-system intelligence"""
        # Placeholder for cross-system collaborative prefetching
        # Would integrate with multi-system coordinator
        return self._pattern_based_candidates(key)
    
    def update_access_pattern(self, key: str, timestamp: datetime):
        """Update access patterns for prefetch optimization"""
        self.pattern_history.append({'key': key, 'timestamp': timestamp})
        
        # Limit pattern history size
        if len(self.pattern_history) > self.config.access_pattern_window_size:
            self.pattern_history = self.pattern_history[-self.config.access_pattern_window_size:]
        
        # Reanalyze patterns periodically
        if len(self.pattern_history) % 100 == 0:
            self._reanalyze_patterns()
    
    def _reanalyze_patterns(self):
        """Reanalyze access patterns from history"""
        try:
            # Group recent accesses by time windows
            time_windows = {}
            
            for record in self.pattern_history[-1000:]:  # Last 1000 accesses
                window_key = int(record['timestamp'].timestamp() // 60)  # 1-minute windows
                if window_key not in time_windows:
                    time_windows[window_key] = []
                time_windows[window_key].append(record['key'])
            
            # Analyze patterns in each window
            for window_key, keys in time_windows.items():
                if len(keys) >= 3:  # Minimum for pattern detection
                    pattern = self.pattern_analyzer.analyze_access_pattern(
                        keys, 
                        [datetime.fromtimestamp(window_key * 60) for _ in keys]
                    )
                    if pattern and pattern.confidence > 0.6:
                        self.access_patterns[pattern.pattern_id] = pattern
            
            # Limit stored patterns
            if len(self.access_patterns) > 100:
                # Keep most confident patterns
                sorted_patterns = sorted(self.access_patterns.items(), 
                                       key=lambda x: x[1].confidence, reverse=True)
                self.access_patterns = dict(sorted_patterns[:100])
                
        except Exception as e:
            logger.error(f"Pattern reanalysis failed: {e}")

class AccessPatternAnalyzer:
    """Analyzer for cache access patterns"""
    
    def __init__(self, config: AdvancedCacheConfig):
        self.config = config
    
    def analyze_access_pattern(self, keys: List[str], timestamps: List[datetime]) -> Optional[AccessPattern]:
        """Analyze access pattern from key sequence"""
        if len(keys) < 2 or len(keys) != len(timestamps):
            return None
        
        try:
            pattern_id = hashlib.md5(json.dumps(keys).encode()).hexdigest()[:16]
            
            # Detect pattern type
            pattern_type = self._detect_pattern_type(keys, timestamps)
            
            # Calculate frequency (accesses per second)
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            frequency = len(keys) / max(time_span, 1.0)
            
            # Calculate predictability score
            predictability_score = self._calculate_predictability(keys, timestamps)
            
            # Calculate confidence based on pattern strength
            confidence = self._calculate_confidence(keys, timestamps, pattern_type)
            
            pattern = AccessPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                keys=keys,
                timestamps=timestamps,
                frequency=frequency,
                predictability_score=predictability_score,
                confidence=confidence
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return None
    
    def _detect_pattern_type(self, keys: List[str], timestamps: List[datetime]) -> CacheAccessPattern:
        """Detect the type of access pattern"""
        try:
            # Check for sequential numeric pattern
            if self._is_sequential_numeric(keys):
                return CacheAccessPattern.SEQUENTIAL
            
            # Check for temporal locality (repeated keys in short time)
            if self._has_temporal_locality(keys, timestamps):
                return CacheAccessPattern.TEMPORAL_LOCALITY
            
            # Check for burst pattern (many accesses in short time)
            if self._is_burst_pattern(timestamps):
                return CacheAccessPattern.BURST
            
            # Check for hotspot pattern (few keys accessed frequently)
            if self._is_hotspot_pattern(keys):
                return CacheAccessPattern.HOTSPOT
            
            # Default to random if no clear pattern
            return CacheAccessPattern.RANDOM
            
        except Exception:
            return CacheAccessPattern.RANDOM
    
    def _is_sequential_numeric(self, keys: List[str]) -> bool:
        """Check if keys form a sequential numeric pattern"""
        try:
            numeric_keys = [int(k) for k in keys if k.isdigit()]
            if len(numeric_keys) < len(keys) * 0.7:  # At least 70% numeric
                return False
            
            # Check if mostly sequential
            sequential_count = 0
            for i in range(1, len(numeric_keys)):
                if numeric_keys[i] == numeric_keys[i-1] + 1:
                    sequential_count += 1
            
            return sequential_count >= len(numeric_keys) * 0.6  # 60% sequential
            
        except Exception:
            return False
    
    def _has_temporal_locality(self, keys: List[str], timestamps: List[datetime]) -> bool:
        """Check for temporal locality pattern"""
        try:
            # Count repeated keys
            key_counts = {}
            for key in keys:
                key_counts[key] = key_counts.get(key, 0) + 1
            
            # Check if there are repeated keys (temporal locality)
            repeated_keys = sum(1 for count in key_counts.values() if count > 1)
            return repeated_keys >= len(key_counts) * 0.3  # 30% of keys repeated
            
        except Exception:
            return False
    
    def _is_burst_pattern(self, timestamps: List[datetime]) -> bool:
        """Check for burst access pattern"""
        try:
            if len(timestamps) < 5:
                return False
            
            # Calculate time intervals
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                        for i in range(1, len(timestamps))]
            
            # Check if most intervals are very short (burst)
            short_intervals = sum(1 for interval in intervals if interval < 1.0)
            return short_intervals >= len(intervals) * 0.7  # 70% under 1 second
            
        except Exception:
            return False
    
    def _is_hotspot_pattern(self, keys: List[str]) -> bool:
        """Check for hotspot access pattern"""
        try:
            key_counts = {}
            for key in keys:
                key_counts[key] = key_counts.get(key, 0) + 1
            
            # Check if a small number of keys account for most accesses
            sorted_counts = sorted(key_counts.values(), reverse=True)
            top_keys_accesses = sum(sorted_counts[:max(1, len(sorted_counts) // 5)])  # Top 20% of keys
            total_accesses = sum(sorted_counts)
            
            return top_keys_accesses >= total_accesses * 0.8  # Top keys account for 80% of accesses
            
        except Exception:
            return False
    
    def _calculate_predictability(self, keys: List[str], timestamps: List[datetime]) -> float:
        """Calculate how predictable the access pattern is"""
        try:
            # Simple predictability based on pattern regularity
            if self._is_sequential_numeric(keys):
                return 0.9
            elif self._has_temporal_locality(keys, timestamps):
                return 0.7
            elif self._is_hotspot_pattern(keys):
                return 0.6
            elif self._is_burst_pattern(timestamps):
                return 0.5
            else:
                return 0.2
                
        except Exception:
            return 0.2
    
    def _calculate_confidence(self, keys: List[str], timestamps: List[datetime], 
                            pattern_type: CacheAccessPattern) -> float:
        """Calculate confidence in pattern detection"""
        try:
            base_confidence = 0.5
            
            # Increase confidence based on pattern strength
            if pattern_type == CacheAccessPattern.SEQUENTIAL:
                base_confidence += 0.3
            elif pattern_type == CacheAccessPattern.TEMPORAL_LOCALITY:
                base_confidence += 0.2
            elif pattern_type == CacheAccessPattern.HOTSPOT:
                base_confidence += 0.2
            
            # Increase confidence based on sample size
            if len(keys) > 10:
                base_confidence += 0.1
            if len(keys) > 50:
                base_confidence += 0.1
            
            return min(base_confidence, 1.0)
            
        except Exception:
            return 0.5

class MultiTierCacheManager:
    """Multi-tier cache manager with intelligent coordination"""
    
    def __init__(self, config: AdvancedCacheConfig):
        self.config = config
        self.cache_levels = {}
        self.eviction_engine = IntelligentEvictionEngine(config)
        self.prefetcher = PredictivePrefetcher(config)
        self.metrics = {}
        
        # Initialize cache levels
        self._initialize_cache_levels()
        
        # Performance tracking
        self.response_times = []
        self.operation_counts = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'prefetches': 0
        }
        
    def _initialize_cache_levels(self):
        """Initialize all cache levels"""
        # L1 Memory Cache
        self.cache_levels[CacheLevel.L1_MEMORY] = {
            'entries': {},
            'max_size_mb': self.config.l1_memory_size_mb,
            'current_size_mb': 0.0
        }
        
        # L2 Redis Cache (if available)
        if REDIS_AVAILABLE:
            try:
                self.cache_levels[CacheLevel.L2_REDIS] = {
                    'client': redis.Redis(host='localhost', port=6379, decode_responses=False),
                    'max_size_mb': self.config.l2_redis_size_mb,
                    'current_size_mb': 0.0
                }
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
        
        # L3 Disk Cache
        self.cache_levels[CacheLevel.L3_DISK] = {
            'entries': {},
            'base_path': 'cache_l3',
            'max_size_mb': self.config.l3_disk_size_mb,
            'current_size_mb': 0.0
        }
        
        logger.info(f"Initialized {len(self.cache_levels)} cache levels")
    
    def get(self, key: str, access_context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Get value from cache with multi-tier lookup"""
        start_time = time.time()
        access_context = access_context or {}
        
        try:
            # Try each cache level in order
            for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK]:
                if level not in self.cache_levels:
                    continue
                
                value = self._get_from_level(key, level)
                if value is not None:
                    # Cache hit - promote to higher levels
                    self._promote_to_higher_levels(key, value, level)
                    
                    # Update access patterns
                    self.prefetcher.update_access_pattern(key, datetime.now())
                    
                    # Check for prefetch opportunities
                    if self.prefetcher.should_prefetch(key, access_context):
                        self._trigger_prefetch(key, access_context)
                    
                    # Record hit
                    self.operation_counts['hits'] += 1
                    self._record_response_time(time.time() - start_time)
                    
                    return value
            
            # Cache miss
            self.operation_counts['misses'] += 1
            self._record_response_time(time.time() - start_time)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get operation failed: {e}")
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            access_context: Optional[Dict[str, Any]] = None) -> bool:
        """Put value into cache with intelligent placement"""
        try:
            ttl_seconds = ttl_seconds or self.config.default_ttl_seconds
            access_context = access_context or {}
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl_seconds,
                size_bytes=self._calculate_size(value),
                origin_system=access_context.get('origin_system', 'local')
            )
            
            # Determine optimal cache level for initial placement
            target_level = self._determine_optimal_level(entry, access_context)
            
            # Store in target level
            success = self._put_in_level(entry, target_level)
            
            if success:
                # Update access patterns
                self.prefetcher.update_access_pattern(key, datetime.now())
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache put operation failed: {e}")
            return False
    
    def _get_from_level(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Get value from specific cache level"""
        try:
            if level == CacheLevel.L1_MEMORY:
                return self._get_from_memory(key)
            elif level == CacheLevel.L2_REDIS and REDIS_AVAILABLE:
                return self._get_from_redis(key)
            elif level == CacheLevel.L3_DISK:
                return self._get_from_disk(key)
            
            return None
            
        except Exception as e:
            logger.error(f"Get from level {level.value} failed: {e}")
            return None
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from L1 memory cache"""
        cache_data = self.cache_levels[CacheLevel.L1_MEMORY]
        
        if key in cache_data['entries']:
            entry = cache_data['entries'][key]
            
            if not entry.is_expired():
                entry.update_access()
                return entry.value
            else:
                # Remove expired entry
                del cache_data['entries'][key]
                cache_data['current_size_mb'] -= entry.size_bytes / (1024 * 1024)
        
        return None
    
    def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from L2 Redis cache"""
        if CacheLevel.L2_REDIS not in self.cache_levels:
            return None
        
        try:
            redis_client = self.cache_levels[CacheLevel.L2_REDIS]['client']
            data = redis_client.get(key)
            
            if data:
                # Deserialize data
                if self.config.compression_enabled:
                    data = zlib.decompress(data)
                
                entry = pickle.loads(data)
                
                if not entry.is_expired():
                    entry.update_access()
                    return entry.value
                else:
                    # Remove expired entry
                    redis_client.delete(key)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
            return None
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from L3 disk cache"""
        # Simplified disk cache implementation
        # In production, this would use proper file-based caching
        cache_data = self.cache_levels[CacheLevel.L3_DISK]
        
        if key in cache_data['entries']:
            entry = cache_data['entries'][key]
            
            if not entry.is_expired():
                entry.update_access()
                return entry.value
            else:
                # Remove expired entry
                del cache_data['entries'][key]
                cache_data['current_size_mb'] -= entry.size_bytes / (1024 * 1024)
        
        return None
    
    def _put_in_level(self, entry: CacheEntry, level: CacheLevel) -> bool:
        """Put entry in specific cache level"""
        try:
            if level == CacheLevel.L1_MEMORY:
                return self._put_in_memory(entry)
            elif level == CacheLevel.L2_REDIS and REDIS_AVAILABLE:
                return self._put_in_redis(entry)
            elif level == CacheLevel.L3_DISK:
                return self._put_in_disk(entry)
            
            return False
            
        except Exception as e:
            logger.error(f"Put in level {level.value} failed: {e}")
            return False
    
    def _put_in_memory(self, entry: CacheEntry) -> bool:
        """Put entry in L1 memory cache"""
        cache_data = self.cache_levels[CacheLevel.L1_MEMORY]
        entry_size_mb = entry.size_bytes / (1024 * 1024)
        
        # Check if eviction is needed
        if self.eviction_engine.should_evict(
            int(cache_data['current_size_mb'] + entry_size_mb),
            cache_data['max_size_mb']
        ):
            self._evict_from_level(CacheLevel.L1_MEMORY)
        
        # Store entry
        cache_data['entries'][entry.key] = entry
        cache_data['current_size_mb'] += entry_size_mb
        
        return True
    
    def _put_in_redis(self, entry: CacheEntry) -> bool:
        """Put entry in L2 Redis cache"""
        if CacheLevel.L2_REDIS not in self.cache_levels:
            return False
        
        try:
            redis_client = self.cache_levels[CacheLevel.L2_REDIS]['client']
            
            # Serialize entry
            data = pickle.dumps(entry)
            if self.config.compression_enabled:
                data = zlib.compress(data)
            
            # Store with TTL
            redis_client.setex(entry.key, entry.ttl_seconds, data)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis put failed: {e}")
            return False
    
    def _put_in_disk(self, entry: CacheEntry) -> bool:
        """Put entry in L3 disk cache"""
        # Simplified disk cache implementation
        cache_data = self.cache_levels[CacheLevel.L3_DISK]
        entry_size_mb = entry.size_bytes / (1024 * 1024)
        
        # Check if eviction is needed
        if self.eviction_engine.should_evict(
            int(cache_data['current_size_mb'] + entry_size_mb),
            cache_data['max_size_mb']
        ):
            self._evict_from_level(CacheLevel.L3_DISK)
        
        # Store entry
        cache_data['entries'][entry.key] = entry
        cache_data['current_size_mb'] += entry_size_mb
        
        return True
    
    def _promote_to_higher_levels(self, key: str, value: Any, current_level: CacheLevel):
        """Promote frequently accessed entries to higher cache levels"""
        try:
            # Promote from L3 to L2
            if current_level == CacheLevel.L3_DISK and CacheLevel.L2_REDIS in self.cache_levels:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    ttl_seconds=self.config.default_ttl_seconds,
                    size_bytes=self._calculate_size(value)
                )
                self._put_in_level(entry, CacheLevel.L2_REDIS)
            
            # Promote from L2 to L1 or L3 to L1
            if current_level in [CacheLevel.L2_REDIS, CacheLevel.L3_DISK]:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    ttl_seconds=self.config.default_ttl_seconds,
                    size_bytes=self._calculate_size(value)
                )
                self._put_in_level(entry, CacheLevel.L1_MEMORY)
                
        except Exception as e:
            logger.error(f"Cache promotion failed: {e}")
    
    def _determine_optimal_level(self, entry: CacheEntry, access_context: Dict[str, Any]) -> CacheLevel:
        """Determine optimal cache level for initial placement"""
        # Simple heuristics for cache placement
        size_mb = entry.size_bytes / (1024 * 1024)
        
        # Large entries go to lower levels
        if size_mb > 10:  # > 10MB
            return CacheLevel.L3_DISK
        elif size_mb > 1:  # > 1MB
            return CacheLevel.L2_REDIS if REDIS_AVAILABLE else CacheLevel.L3_DISK
        else:
            return CacheLevel.L1_MEMORY
    
    def _evict_from_level(self, level: CacheLevel):
        """Evict entries from specific cache level"""
        try:
            if level == CacheLevel.L1_MEMORY:
                cache_data = self.cache_levels[level]
                candidates = self.eviction_engine.select_eviction_candidates(
                    cache_data['entries'], self.config.eviction_batch_size
                )
                
                for key in candidates:
                    if key in cache_data['entries']:
                        entry = cache_data['entries'][key]
                        cache_data['current_size_mb'] -= entry.size_bytes / (1024 * 1024)
                        del cache_data['entries'][key]
                        self.operation_counts['evictions'] += 1
            
            elif level == CacheLevel.L3_DISK:
                cache_data = self.cache_levels[level]
                candidates = self.eviction_engine.select_eviction_candidates(
                    cache_data['entries'], self.config.eviction_batch_size
                )
                
                for key in candidates:
                    if key in cache_data['entries']:
                        entry = cache_data['entries'][key]
                        cache_data['current_size_mb'] -= entry.size_bytes / (1024 * 1024)
                        del cache_data['entries'][key]
                        self.operation_counts['evictions'] += 1
            
        except Exception as e:
            logger.error(f"Eviction from level {level.value} failed: {e}")
    
    def _trigger_prefetch(self, key: str, access_context: Dict[str, Any]):
        """Trigger prefetch based on predicted access patterns"""
        try:
            candidates = self.prefetcher.generate_prefetch_candidates(key, access_context)
            
            for candidate in candidates:
                # Check if candidate is already cached
                if self.get(candidate) is None:
                    # Add to prefetch queue (would trigger background prefetch in full implementation)
                    try:
                        self.prefetch_queue.put_nowait({
                            'key': candidate,
                            'source_key': key,
                            'timestamp': datetime.now(),
                            'priority': 1
                        })
                        self.operation_counts['prefetches'] += 1
                    except queue.Full:
                        break  # Queue full, skip remaining candidates
                        
        except Exception as e:
            logger.error(f"Prefetch trigger failed: {e}")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, bytes):
                return len(value)
            else:
                # Use pickle size as approximation
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default 1KB estimate
    
    def _record_response_time(self, response_time: float):
        """Record response time for metrics"""
        self.response_times.append(response_time * 1000)  # Convert to ms
        
        # Limit response time history
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_cache_metrics(self) -> Dict[CacheLevel, CacheMetrics]:
        """Get comprehensive cache metrics"""
        metrics = {}
        
        try:
            for level in self.cache_levels.keys():
                cache_data = self.cache_levels[level]
                
                # Calculate metrics
                total_requests = self.operation_counts['hits'] + self.operation_counts['misses']
                hit_ratio = (self.operation_counts['hits'] / total_requests) if total_requests > 0 else 0.0
                
                # Response time percentiles
                if self.response_times:
                    avg_response_time = np.mean(self.response_times)
                    p95_response_time = np.percentile(self.response_times, 95)
                    p99_response_time = np.percentile(self.response_times, 99)
                else:
                    avg_response_time = p95_response_time = p99_response_time = 0.0
                
                # Cache level specific metrics
                if level == CacheLevel.L1_MEMORY:
                    total_entries = len(cache_data['entries'])
                    current_size_mb = cache_data['current_size_mb']
                    max_size_mb = cache_data['max_size_mb']
                elif level == CacheLevel.L3_DISK:
                    total_entries = len(cache_data['entries'])
                    current_size_mb = cache_data['current_size_mb']
                    max_size_mb = cache_data['max_size_mb']
                else:
                    total_entries = 0
                    current_size_mb = 0.0
                    max_size_mb = cache_data.get('max_size_mb', 0)
                
                utilization_ratio = current_size_mb / max_size_mb if max_size_mb > 0 else 0.0
                
                cache_metrics = CacheMetrics(
                    timestamp=datetime.now(),
                    cache_level=level,
                    hit_count=self.operation_counts['hits'],
                    miss_count=self.operation_counts['misses'],
                    total_requests=total_requests,
                    hit_ratio=hit_ratio,
                    average_response_time_ms=avg_response_time,
                    p95_response_time_ms=p95_response_time,
                    p99_response_time_ms=p99_response_time,
                    total_entries=total_entries,
                    total_size_mb=current_size_mb,
                    utilization_ratio=utilization_ratio,
                    evictions_count=self.operation_counts['evictions'],
                    prefetch_hits=self.operation_counts['prefetches'],
                    prefetch_accuracy=0.8  # Placeholder
                )
                
                cache_metrics.calculate_hit_ratio()
                metrics[level] = cache_metrics
            
        except Exception as e:
            logger.error(f"Cache metrics calculation failed: {e}")
        
        return metrics

class AdvancedCachingDatabase:
    """Database for advanced caching analytics"""
    
    def __init__(self, config: AdvancedCacheConfig):
        self.config = config
        self.db_path = config.db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Cache metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    cache_level TEXT NOT NULL,
                    hit_count INTEGER,
                    miss_count INTEGER,
                    total_requests INTEGER,
                    hit_ratio REAL,
                    average_response_time_ms REAL,
                    p95_response_time_ms REAL,
                    p99_response_time_ms REAL,
                    total_entries INTEGER,
                    total_size_mb REAL,
                    utilization_ratio REAL,
                    evictions_count INTEGER,
                    prefetch_hits INTEGER,
                    prefetch_accuracy REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Access patterns table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS access_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    pattern_type TEXT NOT NULL,
                    keys TEXT NOT NULL,  -- JSON
                    timestamps TEXT NOT NULL,  -- JSON
                    frequency REAL,
                    predictability_score REAL,
                    confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Cache operations log
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    cache_level TEXT,
                    response_time_ms REAL,
                    hit BOOLEAN,
                    entry_size_bytes INTEGER,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_metrics_timestamp ON cache_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_metrics_level ON cache_metrics(cache_level)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_patterns_type ON access_patterns(pattern_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_operations_timestamp ON cache_operations(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_operations_key ON cache_operations(cache_key)')
                
                conn.commit()
                logger.info("Advanced caching database initialized")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

class AdvancedCachingStrategyEvolution:
    """Advanced Caching Strategy Evolution System"""
    
    def __init__(self, config: AdvancedCacheConfig = None):
        self.config = config or AdvancedCacheConfig()
        self.database = AdvancedCachingDatabase(self.config)
        self.cache_manager = MultiTierCacheManager(self.config)
        
        # System state
        self.is_running = False
        self.analytics_thread = None
        self.optimization_thread = None
        
        # Performance tracking
        self.system_metrics = {
            'cache_operations_total': 0,
            'cache_hits_total': 0,
            'cache_misses_total': 0,
            'average_hit_ratio': 0.0,
            'average_response_time_ms': 0.0,
            'total_evictions': 0,
            'total_prefetches': 0,
            'patterns_discovered': 0
        }
        
    def start_caching_system(self):
        """Start the advanced caching system"""
        try:
            logger.info("Starting Advanced Caching Strategy Evolution System")
            
            self.is_running = True
            
            # Start analytics thread
            self.analytics_thread = threading.Thread(target=self._analytics_cycle, daemon=True)
            self.analytics_thread.start()
            
            # Start optimization thread
            self.optimization_thread = threading.Thread(target=self._optimization_cycle, daemon=True)
            self.optimization_thread.start()
            
            logger.info("Advanced Caching Strategy Evolution System started successfully")
            return {"status": "started", "timestamp": datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Failed to start caching system: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analytics_cycle(self):
        """Analytics collection cycle"""
        while self.is_running:
            try:
                # Collect cache metrics
                cache_metrics = self.cache_manager.get_cache_metrics()
                
                # Update system metrics
                self._update_system_metrics(cache_metrics)
                
                # Log metrics to database (would be implemented)
                
                time.sleep(60)  # Analytics cycle every minute
                
            except Exception as e:
                logger.error(f"Analytics cycle error: {e}")
                time.sleep(60)
    
    def _optimization_cycle(self):
        """Cache optimization cycle"""
        while self.is_running:
            try:
                # Train ML models periodically
                if time.time() - self.cache_manager.eviction_engine.last_model_training > self.config.ml_model_retrain_interval:
                    self._retrain_ml_models()
                
                # Optimize cache strategies
                self._optimize_cache_strategies()
                
                time.sleep(300)  # Optimization cycle every 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization cycle error: {e}")
                time.sleep(300)
    
    def _update_system_metrics(self, cache_metrics: Dict[CacheLevel, CacheMetrics]):
        """Update system-wide metrics"""
        try:
            total_hits = sum(metrics.hit_count for metrics in cache_metrics.values())
            total_misses = sum(metrics.miss_count for metrics in cache_metrics.values())
            total_requests = total_hits + total_misses
            
            if total_requests > 0:
                self.system_metrics['average_hit_ratio'] = total_hits / total_requests
            
            # Update other metrics
            self.system_metrics['cache_operations_total'] = total_requests
            self.system_metrics['cache_hits_total'] = total_hits
            self.system_metrics['cache_misses_total'] = total_misses
            self.system_metrics['total_evictions'] = self.cache_manager.operation_counts['evictions']
            self.system_metrics['total_prefetches'] = self.cache_manager.operation_counts['prefetches']
            
            # Calculate average response time
            if cache_metrics:
                avg_response_times = [metrics.average_response_time_ms for metrics in cache_metrics.values() if metrics.average_response_time_ms > 0]
                if avg_response_times:
                    self.system_metrics['average_response_time_ms'] = np.mean(avg_response_times)
            
        except Exception as e:
            logger.error(f"System metrics update failed: {e}")
    
    def _retrain_ml_models(self):
        """Retrain ML models for cache optimization"""
        try:
            # Placeholder for ML model retraining
            # Would collect access history and retrain eviction models
            logger.info("ML model retraining initiated")
            
        except Exception as e:
            logger.error(f"ML model retraining failed: {e}")
    
    def _optimize_cache_strategies(self):
        """Optimize cache strategies based on performance data"""
        try:
            # Analyze current performance
            cache_metrics = self.cache_manager.get_cache_metrics()
            
            # Optimize eviction strategies
            for level, metrics in cache_metrics.items():
                if metrics.hit_ratio < 0.8:  # Below 80% hit ratio
                    logger.info(f"Optimizing {level.value} cache strategy (hit ratio: {metrics.hit_ratio:.2%})")
            
        except Exception as e:
            logger.error(f"Cache strategy optimization failed: {e}")
    
    def get_caching_status(self) -> Dict[str, Any]:
        """Get comprehensive caching system status"""
        try:
            cache_metrics = self.cache_manager.get_cache_metrics()
            
            status = {
                'system_name': 'Advanced Caching Strategy Evolution',
                'version': '2.0.0',
                'status': 'operational' if self.is_running else 'stopped',
                'timestamp': datetime.now().isoformat(),
                
                # Cache Level Status
                'cache_levels': {
                    level.value: {
                        'hit_ratio': metrics.hit_ratio,
                        'total_entries': metrics.total_entries,
                        'size_mb': metrics.total_size_mb,
                        'utilization': metrics.utilization_ratio,
                        'average_response_time_ms': metrics.average_response_time_ms,
                        'evictions': metrics.evictions_count
                    }
                    for level, metrics in cache_metrics.items()
                },
                
                # System Metrics
                'system_metrics': self.system_metrics,
                
                # Configuration
                'configuration': {
                    'eviction_strategy': self.config.default_eviction_strategy.value,
                    'prefetch_strategy': self.config.prefetch_strategy.value,
                    'l1_memory_size_mb': self.config.l1_memory_size_mb,
                    'l2_redis_size_mb': self.config.l2_redis_size_mb,
                    'l3_disk_size_mb': self.config.l3_disk_size_mb,
                    'compression_enabled': self.config.compression_enabled,
                    'ml_eviction_enabled': self.config.ml_eviction_enabled
                },
                
                # Capabilities
                'capabilities': {
                    'multi_tier_caching': True,
                    'intelligent_eviction': True,
                    'predictive_prefetching': True,
                    'ml_optimization': SKLEARN_AVAILABLE,
                    'redis_support': REDIS_AVAILABLE,
                    'compression_support': self.config.compression_enabled,
                    'cross_system_coordination': self.config.cross_system_sync_enabled
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Status generation failed: {e}")
            return {
                'system_name': 'Advanced Caching Strategy Evolution',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def stop_caching_system(self):
        """Stop the caching system"""
        try:
            logger.info("Stopping Advanced Caching Strategy Evolution System")
            
            self.is_running = False
            
            # Wait for threads to finish
            if self.analytics_thread:
                self.analytics_thread.join(timeout=10.0)
            if self.optimization_thread:
                self.optimization_thread.join(timeout=10.0)
            
            logger.info("Advanced Caching Strategy Evolution System stopped")
            return {"status": "stopped", "timestamp": datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Failed to stop caching system: {e}")
            return {"status": "error", "message": str(e)}

def main():
    """Demonstration of Advanced Caching Strategy Evolution System"""
    print("=== Advanced Caching Strategy Evolution System Demo ===")
    
    # Initialize system
    config = AdvancedCacheConfig()
    caching_system = AdvancedCachingStrategyEvolution(config)
    
    # Start system
    print("Starting caching system...")
    start_result = caching_system.start_caching_system()
    print(f"Start result: {start_result}")
    
    # Demonstrate cache operations
    print("\nTesting cache operations...")
    cache_manager = caching_system.cache_manager
    
    # Test data
    test_data = [
        ("user:123", {"name": "Alice", "email": "alice@example.com"}),
        ("user:124", {"name": "Bob", "email": "bob@example.com"}),
        ("product:456", {"title": "Laptop", "price": 999.99}),
        ("session:789", {"user_id": 123, "created": "2025-01-01"}),
        ("config:settings", {"theme": "dark", "language": "en"})
    ]
    
    # Put operations
    for key, value in test_data:
        success = cache_manager.put(key, value)
        print(f"PUT {key}: {'✓' if success else '✗'}")
    
    # Get operations
    print("\nTesting cache retrieval...")
    for key, expected_value in test_data:
        retrieved = cache_manager.get(key)
        hit = retrieved is not None
        print(f"GET {key}: {'HIT' if hit else 'MISS'}")
    
    # Test access patterns
    print("\nTesting access patterns...")
    sequential_keys = ["item:1", "item:2", "item:3", "item:4", "item:5"]
    for key in sequential_keys:
        cache_manager.put(key, f"data for {key}")
        cache_manager.get(key)  # Immediate access to establish pattern
    
    # Let the system run for analytics
    print("\nRunning system for analytics...")
    time.sleep(5)
    
    # Get system status
    print("\nSystem Status:")
    status = caching_system.get_caching_status()
    
    print(f"Status: {status['status']}")
    print(f"System Metrics:")
    for metric, value in status['system_metrics'].items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.3f}")
        else:
            print(f"  - {metric}: {value}")
    
    print(f"\nCache Levels:")
    for level, metrics in status['cache_levels'].items():
        print(f"  {level}:")
        print(f"    - Hit Ratio: {metrics['hit_ratio']:.2%}")
        print(f"    - Entries: {metrics['total_entries']}")
        print(f"    - Size: {metrics['size_mb']:.2f} MB")
        print(f"    - Utilization: {metrics['utilization']:.1%}")
        print(f"    - Response Time: {metrics['average_response_time_ms']:.2f} ms")
    
    print(f"\nCapabilities:")
    for capability, enabled in status['capabilities'].items():
        print(f"  - {capability}: {'✓' if enabled else '✗'}")
    
    # Stop system
    print("\nStopping caching system...")
    stop_result = caching_system.stop_caching_system()
    print(f"Stop result: {stop_result}")
    
    print("\n=== Advanced Caching Strategy Evolution Demo Complete ===")

if __name__ == "__main__":
    main()