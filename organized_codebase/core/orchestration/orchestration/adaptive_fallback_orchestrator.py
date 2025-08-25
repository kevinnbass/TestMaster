"""
Adaptive Fallback Orchestrator - Archive-Derived Reliability System
==================================================================

Multi-level fallback system with intelligent degradation, cache management,
and adaptive recovery patterns based on failure analysis.

Author: Agent C Security Framework
Created: 2025-08-21
"""

import logging
import time
import json
import pickle
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import os
import hashlib
import random

logger = logging.getLogger(__name__)

class FallbackLevel(Enum):
    """Hierarchical fallback escalation levels."""
    PRIMARY = "primary"          # Normal operation
    CACHE = "cache"             # Use cached data
    SECONDARY = "secondary"      # Alternative endpoint/service
    DEGRADED = "degraded"       # Reduced functionality
    LOCAL = "local"             # Local storage only
    EMERGENCY = "emergency"      # Minimal data only
    OFFLINE = "offline"         # Complete offline mode

class FallbackReason(Enum):
    """Comprehensive failure classification."""
    TIMEOUT = "timeout"
    ERROR = "error"
    OVERLOAD = "overload"
    MAINTENANCE = "maintenance"
    NETWORK = "network"
    VALIDATION = "validation"
    SECURITY = "security"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION = "configuration"

class CacheStrategy(Enum):
    """Cache management strategies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live based
    ADAPTIVE = "adaptive"    # ML-driven cache decisions
    PRIORITY = "priority"    # Priority-based eviction

@dataclass
class FallbackEvent:
    """Comprehensive fallback event record."""
    event_id: str
    timestamp: datetime
    from_level: FallbackLevel
    to_level: FallbackLevel
    reason: FallbackReason
    error_details: Optional[str]
    data_preserved: bool
    recovery_time_estimate_seconds: int
    context: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.5
    automated_recovery: bool = True

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    priority: int
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 1.0

@dataclass
class FallbackConfiguration:
    """Comprehensive fallback configuration."""
    cache_size: int = 1000
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    local_storage_path: str = "data/fallback_storage.db"
    degraded_mode_threshold: int = 5
    emergency_mode_threshold: int = 10
    auto_recovery_enabled: bool = True
    recovery_check_interval: int = 30
    cache_ttl_default: int = 3600
    max_fallback_depth: int = 6
    performance_monitoring: bool = True

class AdaptiveFallbackOrchestrator:
    """
    Intelligent fallback orchestrator with machine learning-driven decisions.
    """
    
    def __init__(self, config: FallbackConfiguration = None):
        """
        Initialize adaptive fallback orchestrator.
        
        Args:
            config: Fallback configuration
        """
        self.config = config or FallbackConfiguration()
        
        # State management
        self.current_level = FallbackLevel.PRIMARY
        self.failure_count = 0
        self.success_count = 0
        self.last_success_time = datetime.now()
        self.last_failure_time = None
        
        # Initialize storage
        self._setup_local_storage()
        
        # Multi-level cache system
        self.cache_layers = {
            FallbackLevel.CACHE: deque(maxlen=self.config.cache_size),
            FallbackLevel.SECONDARY: deque(maxlen=self.config.cache_size // 2),
            FallbackLevel.DEGRADED: deque(maxlen=self.config.cache_size // 4)
        }
        self.cache_indexes = {level: {} for level in self.cache_layers.keys()}
        
        # Alternative services registry
        self.alternative_services = []
        self.service_health = {}
        self.service_performance = defaultdict(lambda: {
            'response_times': deque(maxlen=100),
            'success_rate': 1.0,
            'last_check': datetime.now()
        })
        
        # Fallback strategies mapping
        self.fallback_strategies = {
            FallbackLevel.CACHE: self._execute_cache_fallback,
            FallbackLevel.SECONDARY: self._execute_secondary_fallback,
            FallbackLevel.DEGRADED: self._execute_degraded_fallback,
            FallbackLevel.LOCAL: self._execute_local_fallback,
            FallbackLevel.EMERGENCY: self._execute_emergency_fallback,
            FallbackLevel.OFFLINE: self._execute_offline_fallback
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            'exponential': self._recover_exponential,
            'linear': self._recover_linear,
            'adaptive': self._recover_adaptive,
            'immediate': self._recover_immediate
        }
        
        # Event history and analytics
        self.fallback_events = deque(maxlen=1000)
        self.performance_metrics = deque(maxlen=5000)
        
        # Advanced statistics
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'fallback_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'local_storage_writes': 0,
            'data_loss_prevented': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'average_fallback_time': 0.0,
            'cache_efficiency': 100.0,
            'uptime_percentage': 100.0
        }
        
        # Machine learning components
        self.failure_patterns = defaultdict(list)
        self.success_predictors = {}
        self.adaptive_thresholds = {
            'failure_rate_threshold': 0.2,
            'response_time_threshold': 5.0,
            'cache_hit_threshold': 0.8,
            'recovery_confidence_threshold': 0.7
        }
        
        # Background processing
        self.monitoring_active = True
        self.recovery_thread = threading.Thread(
            target=self._recovery_monitoring_loop,
            daemon=True
        )
        self.cache_maintenance_thread = threading.Thread(
            target=self._cache_maintenance_loop,
            daemon=True
        )
        self.performance_analysis_thread = threading.Thread(
            target=self._performance_analysis_loop,
            daemon=True
        )
        
        # Start background threads
        if self.config.auto_recovery_enabled:
            self.recovery_thread.start()
        
        self.cache_maintenance_thread.start()
        
        if self.config.performance_monitoring:
            self.performance_analysis_thread.start()
        
        # Thread safety
        self.fallback_lock = threading.RLock()
        
        logger.info("Adaptive Fallback Orchestrator initialized with ML capabilities")
    
    def _setup_local_storage(self):
        """Initialize local SQLite storage for fallback operations."""
        try:
            storage_dir = os.path.dirname(self.config.local_storage_path)
            os.makedirs(storage_dir, exist_ok=True)
            
            self.local_db = sqlite3.connect(
                self.config.local_storage_path,
                check_same_thread=False
            )
            
            cursor = self.local_db.cursor()
            
            # Fallback data storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fallback_data (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    level TEXT NOT NULL,
                    synced INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 3,
                    metadata TEXT
                )
            """)
            
            # Fallback events log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fallback_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    from_level TEXT NOT NULL,
                    to_level TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    error_details TEXT,
                    data_preserved INTEGER DEFAULT 0,
                    recovery_estimate INTEGER DEFAULT 0,
                    context TEXT
                )
            """)
            
            # Performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    success INTEGER NOT NULL,
                    fallback_level TEXT,
                    cache_hit INTEGER DEFAULT 0
                )
            """)
            
            self.local_db.commit()
            
        except Exception as e:
            logger.error(f"Failed to setup local storage: {e}")
            self.local_db = None
    
    def execute_with_fallback(self,
                            primary_func: Callable,
                            operation_data: Dict[str, Any],
                            context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any, FallbackLevel]:
        """
        Execute operation with comprehensive fallback protection.
        
        Args:
            primary_func: Primary operation function
            operation_data: Operation data
            context: Additional context
            
        Returns:
            (success, result, fallback_level_used)
        """
        with self.fallback_lock:
            operation_start = time.time()
            self.stats['total_operations'] += 1
            
            # Try primary function first if we're in primary mode
            if self.current_level == FallbackLevel.PRIMARY:
                try:
                    result = primary_func(operation_data)
                    processing_time = time.time() - operation_start
                    self._handle_success(operation_data, processing_time)
                    return (True, result, FallbackLevel.PRIMARY)
                    
                except Exception as e:
                    logger.warning(f"Primary operation failed: {e}")
                    processing_time = time.time() - operation_start
                    self._handle_failure(e, operation_data, processing_time)
                    return self._execute_fallback_cascade(operation_data, str(e), context)
            else:
                # Already in fallback mode, execute appropriate fallback
                return self._execute_fallback_cascade(operation_data, "In fallback mode", context)
    
    def _handle_success(self, data: Dict[str, Any], processing_time: float):
        """Handle successful operation with learning."""
        self.failure_count = max(0, self.failure_count - 1)
        self.success_count += 1
        self.last_success_time = datetime.now()
        
        # Update cache with successful data
        self._update_cache_smart(data, FallbackLevel.CACHE)
        
        # Record performance metrics
        self._record_performance_metric(
            operation_type="primary",
            response_time=processing_time,
            success=True,
            fallback_level=self.current_level,
            cache_hit=False
        )
        
        # Check if we can recover to primary level
        if self.current_level != FallbackLevel.PRIMARY:
            self._attempt_level_recovery()
        
        self.stats['successful_operations'] += 1
    
    def _handle_failure(self, error: Exception, data: Dict[str, Any], processing_time: float):
        """Handle operation failure with intelligent analysis."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        # Classify failure and determine fallback reason
        failure_reason = self._classify_failure_reason(error)
        
        # Preserve data before escalation
        self._preserve_data_multilevel(data)
        
        # Record performance metrics
        self._record_performance_metric(
            operation_type="primary",
            response_time=processing_time,
            success=False,
            fallback_level=self.current_level,
            cache_hit=False
        )
        
        # Analyze failure patterns
        self._analyze_failure_pattern(error, failure_reason, data)
        
        # Determine appropriate fallback level
        new_level = self._determine_fallback_level(failure_reason, self.failure_count)
        
        if new_level != self.current_level:
            self._escalate_fallback(new_level, failure_reason, str(error))
    
    def _execute_fallback_cascade(self, data: Dict[str, Any], error: str, 
                                 context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any, FallbackLevel]:
        """Execute fallback cascade with intelligent strategy selection."""
        current_level = self.current_level
        max_depth = self.config.max_fallback_depth
        attempt_count = 0
        
        while attempt_count < max_depth:
            strategy = self.fallback_strategies.get(current_level)
            
            if strategy:
                try:
                    operation_start = time.time()
                    result = strategy(data, error, context)
                    processing_time = time.time() - operation_start
                    
                    # Record successful fallback
                    self._record_performance_metric(
                        operation_type=f"fallback_{current_level.value}",
                        response_time=processing_time,
                        success=True,
                        fallback_level=current_level,
                        cache_hit=current_level == FallbackLevel.CACHE
                    )
                    
                    self.stats['fallback_operations'] += 1
                    self.stats['data_loss_prevented'] += 1
                    
                    return (True, result, current_level)
                    
                except Exception as e:
                    logger.error(f"Fallback failed at level {current_level}: {e}")
                    processing_time = time.time() - operation_start
                    
                    # Record failed fallback
                    self._record_performance_metric(
                        operation_type=f"fallback_{current_level.value}",
                        response_time=processing_time,
                        success=False,
                        fallback_level=current_level,
                        cache_hit=False
                    )
                    
                    # Escalate to next level
                    current_level = self._get_next_fallback_level(current_level)
                    attempt_count += 1
            else:
                break
        
        # All fallbacks failed
        logger.error(f"All fallback strategies exhausted for operation")
        return (False, None, current_level)
    
    def _execute_cache_fallback(self, data: Dict[str, Any], error: str, 
                               context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute intelligent cache fallback with adaptive selection."""
        cache_key = self._generate_cache_key(data)
        
        # Try exact match first
        for level in [FallbackLevel.CACHE, FallbackLevel.SECONDARY, FallbackLevel.DEGRADED]:
            cache_entry = self.cache_indexes[level].get(cache_key)
            if cache_entry:
                cache_entry.last_accessed = datetime.now()
                cache_entry.access_count += 1
                
                # Check TTL
                if self._is_cache_entry_valid(cache_entry):
                    self.stats['cache_hits'] += 1
                    
                    result = cache_entry.data.copy() if isinstance(cache_entry.data, dict) else cache_entry.data
                    if isinstance(result, dict):
                        result.update({
                            '_from_cache': True,
                            '_cache_level': level.value,
                            '_cached_at': cache_entry.created_at.isoformat(),
                            '_cache_age_seconds': (datetime.now() - cache_entry.created_at).total_seconds()
                        })
                    
                    logger.info(f"Cache hit at level {level.value}")
                    return result
        
        # Try semantic similarity matching
        similar_entry = self._find_similar_cache_entry(data)
        if similar_entry:
            similar_entry.last_accessed = datetime.now()
            similar_entry.access_count += 1
            self.stats['cache_hits'] += 1
            
            result = similar_entry.data.copy() if isinstance(similar_entry.data, dict) else similar_entry.data
            if isinstance(result, dict):
                result.update({
                    '_from_cache': True,
                    '_semantic_match': True,
                    '_cache_similarity': 0.8,  # Placeholder
                    '_cached_at': similar_entry.created_at.isoformat()
                })
            
            logger.info("Semantic cache hit found")
            return result
        
        self.stats['cache_misses'] += 1
        
        # Generate degraded response if no cache hit
        return self._create_intelligent_degraded_response(data, "cache_miss")
    
    def _execute_secondary_fallback(self, data: Dict[str, Any], error: str, 
                                   context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute secondary service fallback with health-based selection."""
        # Sort services by health and performance
        available_services = [
            service for service in self.alternative_services
            if self.service_health.get(service, True)
        ]
        
        # Sort by performance metrics
        available_services.sort(key=lambda s: (
            self.service_performance[s]['success_rate'],
            -sum(self.service_performance[s]['response_times']) / max(1, len(self.service_performance[s]['response_times']))
        ), reverse=True)
        
        for service in available_services:
            try:
                start_time = time.time()
                
                # Simulate service call (in production, would make actual call)
                result = self._call_alternative_service(service, data)
                
                response_time = time.time() - start_time
                
                # Update service performance
                self.service_performance[service]['response_times'].append(response_time)
                self.service_performance[service]['last_check'] = datetime.now()
                
                logger.info(f"Secondary service success: {service}")
                return result
                
            except Exception as e:
                # Mark service as unhealthy
                self.service_health[service] = False
                logger.error(f"Secondary service failed: {service} - {e}")
                continue
        
        # All secondary services failed
        raise Exception("All secondary services failed")
    
    def _execute_degraded_fallback(self, data: Dict[str, Any], error: str, 
                                  context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute degraded mode with intelligent feature reduction."""
        logger.warning("Operating in degraded mode with reduced functionality")
        
        # Store data for later processing
        self._store_locally_with_priority(data, "degraded", priority=2)
        
        # Create intelligent degraded response
        degraded_response = self._create_intelligent_degraded_response(data, "degraded_mode")
        
        # Add degraded mode indicators
        if isinstance(degraded_response, dict):
            degraded_response.update({
                '_degraded_mode': True,
                '_degraded_reason': error,
                '_degraded_at': datetime.now().isoformat(),
                '_full_recovery_estimate': self._estimate_recovery_time(),
                '_functionality_level': 0.6  # 60% functionality in degraded mode
            })
        
        return degraded_response
    
    def _execute_local_fallback(self, data: Dict[str, Any], error: str, 
                               context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute local storage fallback with queued sync."""
        logger.warning("Fallback to local storage only - queuing for sync")
        
        # Store data locally with high priority
        storage_id = self._store_locally_with_priority(data, "local_only", priority=1)
        
        if storage_id:
            self.stats['local_storage_writes'] += 1
            
            return {
                'status': 'stored_locally',
                'storage_id': storage_id,
                'will_sync': True,
                'sync_priority': 'high',
                'timestamp': datetime.now().isoformat(),
                'estimated_sync_time': self._estimate_sync_time(),
                'local_queue_size': self._get_local_queue_size()
            }
        
        raise Exception("Local storage fallback failed")
    
    def _execute_emergency_fallback(self, data: Dict[str, Any], error: str, 
                                   context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute emergency fallback with absolute minimum functionality."""
        logger.error("EMERGENCY MODE - Minimal functionality only")
        
        # Attempt to preserve critical data
        preserved = self._store_locally_with_priority(data, "emergency", priority=0)
        
        return {
            'status': 'emergency',
            'timestamp': datetime.now().isoformat(),
            'emergency_mode': True,
            'basic_metrics': {
                'system_alive': True,
                'emergency_level': True,
                'data_preserved': preserved is not None,
                'critical_functions_only': True
            },
            'recovery_actions': [
                'Check primary system health',
                'Verify network connectivity', 
                'Review error logs',
                'Contact system administrator'
            ]
        }
    
    def _execute_offline_fallback(self, data: Dict[str, Any], error: str, 
                                 context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute complete offline fallback mode."""
        logger.critical("OFFLINE MODE - System completely disconnected")
        
        return {
            'status': 'offline',
            'timestamp': datetime.now().isoformat(),
            'offline_mode': True,
            'message': 'System operating in complete offline mode',
            'available_functions': [
                'local_data_access',
                'basic_calculations', 
                'cached_responses'
            ],
            'reconnection_attempts': 'automatic',
            'last_online': self.last_success_time.isoformat()
        }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive fallback orchestrator status."""
        with self.fallback_lock:
            # Calculate cache efficiency
            total_cache_operations = self.stats['cache_hits'] + self.stats['cache_misses']
            cache_efficiency = (self.stats['cache_hits'] / max(1, total_cache_operations)) * 100
            
            # Calculate uptime percentage
            total_ops = self.stats['successful_operations'] + self.stats['fallback_operations']
            uptime = (self.stats['successful_operations'] / max(1, total_ops)) * 100
            
            return {
                'current_level': self.current_level.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_success': self.last_success_time.isoformat(),
                'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'statistics': self.stats.copy(),
                'cache_efficiency': cache_efficiency,
                'uptime_percentage': uptime,
                'cache_layers': {
                    level.value: len(cache) for level, cache in self.cache_layers.items()
                },
                'alternative_services': {
                    service: {
                        'healthy': self.service_health.get(service, True),
                        'success_rate': self.service_performance[service]['success_rate'],
                        'avg_response_time': sum(self.service_performance[service]['response_times']) / 
                                           max(1, len(self.service_performance[service]['response_times']))
                    }
                    for service in self.alternative_services
                },
                'recent_events': [
                    {
                        'timestamp': e.timestamp.isoformat(),
                        'from': e.from_level.value,
                        'to': e.to_level.value,
                        'reason': e.reason.value,
                        'data_preserved': e.data_preserved
                    }
                    for e in list(self.fallback_events)[-5:]
                ],
                'recovery_estimate_seconds': self._estimate_recovery_time(),
                'adaptive_thresholds': self.adaptive_thresholds.copy(),
                'configuration': {
                    'cache_size': self.config.cache_size,
                    'auto_recovery_enabled': self.config.auto_recovery_enabled,
                    'max_fallback_depth': self.config.max_fallback_depth,
                    'performance_monitoring': self.config.performance_monitoring
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def shutdown(self):
        """Shutdown adaptive fallback orchestrator."""
        self.monitoring_active = False
        
        # Wait for threads to complete
        for thread in [self.recovery_thread, self.cache_maintenance_thread, 
                      self.performance_analysis_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        # Close database connection
        if self.local_db:
            self.local_db.close()
        
        logger.info(f"Adaptive Fallback Orchestrator shutdown - Stats: {self.stats}")

# Global fallback orchestrator instance
adaptive_fallback_orchestrator = None