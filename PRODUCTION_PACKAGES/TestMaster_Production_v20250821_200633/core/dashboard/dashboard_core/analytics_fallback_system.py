"""
Analytics Fallback System
=========================

Provides multiple fallback mechanisms for analytics failures including
caching, alternative endpoints, degraded mode, and local storage.

Author: TestMaster Team
"""

import logging
import time
import json
import pickle
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import os
import sqlite3

logger = logging.getLogger(__name__)

class FallbackLevel(Enum):
    """Fallback escalation levels."""
    PRIMARY = "primary"          # Normal operation
    CACHE = "cache"             # Use cached data
    SECONDARY = "secondary"      # Alternative endpoint
    DEGRADED = "degraded"       # Reduced functionality
    LOCAL = "local"             # Local storage only
    EMERGENCY = "emergency"      # Minimal data only

class FallbackReason(Enum):
    """Reasons for fallback activation."""
    TIMEOUT = "timeout"
    ERROR = "error"
    OVERLOAD = "overload"
    MAINTENANCE = "maintenance"
    NETWORK = "network"
    VALIDATION = "validation"

@dataclass
class FallbackEvent:
    """Represents a fallback event."""
    event_id: str
    timestamp: datetime
    from_level: FallbackLevel
    to_level: FallbackLevel
    reason: FallbackReason
    error_details: Optional[str]
    data_preserved: bool
    recovery_time_estimate_seconds: int

class AnalyticsFallbackSystem:
    """
    Comprehensive fallback system for analytics failures.
    """
    
    def __init__(self,
                 cache_size: int = 1000,
                 local_storage_path: str = "fallback_analytics.db",
                 degraded_mode_threshold: int = 5):
        """
        Initialize fallback system.
        
        Args:
            cache_size: Maximum cache entries
            local_storage_path: Path for local storage fallback
            degraded_mode_threshold: Failures before degraded mode
        """
        self.cache_size = cache_size
        self.local_storage_path = local_storage_path
        self.degraded_mode_threshold = degraded_mode_threshold
        
        # Current fallback level
        self.current_level = FallbackLevel.PRIMARY
        self.failure_count = 0
        self.last_success_time = datetime.now()
        
        # Fallback cache
        self.cache = deque(maxlen=cache_size)
        self.cache_index = {}
        
        # Local storage setup
        self._setup_local_storage()
        
        # Alternative endpoints
        self.alternative_endpoints = []
        self.endpoint_health = {}
        
        # Fallback strategies
        self.fallback_strategies = {
            FallbackLevel.CACHE: self._fallback_to_cache,
            FallbackLevel.SECONDARY: self._fallback_to_secondary,
            FallbackLevel.DEGRADED: self._fallback_to_degraded,
            FallbackLevel.LOCAL: self._fallback_to_local,
            FallbackLevel.EMERGENCY: self._fallback_to_emergency
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            'exponential': self._recover_exponential,
            'linear': self._recover_linear,
            'immediate': self._recover_immediate
        }
        
        # Fallback history
        self.fallback_events = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'total_fallbacks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'local_storage_writes': 0,
            'data_loss_prevented': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
        
        # Recovery thread
        self.recovery_active = True
        self.recovery_thread = threading.Thread(
            target=self._recovery_loop,
            daemon=True
        )
        self.recovery_thread.start()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Analytics Fallback System initialized")
    
    def process_analytics(self,
                         analytics_func: Callable,
                         analytics_data: Dict[str, Any],
                         context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
        """
        Process analytics with fallback protection.
        
        Args:
            analytics_func: Primary analytics function
            analytics_data: Analytics data
            context: Optional context
            
        Returns:
            (success, result)
        """
        with self.lock:
            # Try primary function
            if self.current_level == FallbackLevel.PRIMARY:
                try:
                    result = analytics_func(analytics_data)
                    self._handle_success(analytics_data)
                    return (True, result)
                except Exception as e:
                    logger.error(f"Primary analytics failed: {e}")
                    self._handle_failure(e, analytics_data)
                    return self._execute_fallback(analytics_data, str(e))
            else:
                # Already in fallback mode
                return self._execute_fallback(analytics_data, "In fallback mode")
    
    def _handle_success(self, data: Dict[str, Any]):
        """Handle successful analytics processing."""
        self.failure_count = 0
        self.last_success_time = datetime.now()
        
        # Cache successful result
        self._update_cache(data)
        
        # Check if we can recover to primary
        if self.current_level != FallbackLevel.PRIMARY:
            self._attempt_recovery()
    
    def _handle_failure(self, error: Exception, data: Dict[str, Any]):
        """Handle analytics failure."""
        self.failure_count += 1
        
        # Determine fallback level
        if self.failure_count >= self.degraded_mode_threshold * 2:
            self._escalate_fallback(FallbackLevel.EMERGENCY, FallbackReason.ERROR, str(error))
        elif self.failure_count >= self.degraded_mode_threshold:
            self._escalate_fallback(FallbackLevel.DEGRADED, FallbackReason.ERROR, str(error))
        else:
            self._escalate_fallback(FallbackLevel.CACHE, FallbackReason.ERROR, str(error))
        
        # Preserve data
        self._preserve_data(data)
    
    def _execute_fallback(self, data: Dict[str, Any], error: str) -> Tuple[bool, Any]:
        """Execute current fallback strategy."""
        strategy = self.fallback_strategies.get(self.current_level)
        
        if strategy:
            try:
                result = strategy(data, error)
                self.stats['data_loss_prevented'] += 1
                return (True, result)
            except Exception as e:
                logger.error(f"Fallback failed at level {self.current_level}: {e}")
                # Escalate to next level
                self._escalate_to_next_level(str(e))
                return (False, None)
        
        return (False, None)
    
    def _fallback_to_cache(self, data: Dict[str, Any], error: str) -> Any:
        """Fallback to cached data."""
        # Try to find similar cached data
        cache_key = self._generate_cache_key(data)
        
        if cache_key in self.cache_index:
            self.stats['cache_hits'] += 1
            cached_entry = self.cache_index[cache_key]
            
            # Update with current timestamp
            cached_entry['timestamp'] = datetime.now().isoformat()
            cached_entry['from_cache'] = True
            
            logger.info("Using cached analytics data")
            return cached_entry
        
        self.stats['cache_misses'] += 1
        
        # If no cache, try to return degraded data
        return self._create_degraded_response(data)
    
    def _fallback_to_secondary(self, data: Dict[str, Any], error: str) -> Any:
        """Fallback to secondary endpoint."""
        for endpoint in self.alternative_endpoints:
            if self.endpoint_health.get(endpoint, True):
                try:
                    # Simplified - would call actual endpoint
                    logger.info(f"Using secondary endpoint: {endpoint}")
                    return self._create_degraded_response(data)
                except Exception as e:
                    self.endpoint_health[endpoint] = False
                    logger.error(f"Secondary endpoint failed: {e}")
        
        # All secondaries failed
        raise Exception("All secondary endpoints failed")
    
    def _fallback_to_degraded(self, data: Dict[str, Any], error: str) -> Any:
        """Fallback to degraded mode with minimal data."""
        logger.warning("Operating in degraded mode")
        
        response = self._create_degraded_response(data)
        
        # Store locally for later processing
        self._store_locally(data, "degraded")
        
        return response
    
    def _fallback_to_local(self, data: Dict[str, Any], error: str) -> Any:
        """Fallback to local storage only."""
        logger.warning("Fallback to local storage only")
        
        # Store data locally
        stored = self._store_locally(data, "local_only")
        
        if stored:
            self.stats['local_storage_writes'] += 1
            return {
                'status': 'stored_locally',
                'storage_id': stored,
                'will_sync': True,
                'timestamp': datetime.now().isoformat()
            }
        
        raise Exception("Local storage failed")
    
    def _fallback_to_emergency(self, data: Dict[str, Any], error: str) -> Any:
        """Emergency fallback with absolute minimum."""
        logger.error("EMERGENCY MODE - Minimal analytics only")
        
        # Return absolute minimum
        return {
            'status': 'emergency',
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': {
                'alive': True,
                'emergency_mode': True,
                'data_preserved': self._store_locally(data, "emergency") is not None
            }
        }
    
    def _create_degraded_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create degraded response with essential data."""
        return {
            'status': 'degraded',
            'timestamp': datetime.now().isoformat(),
            'degraded_mode': True,
            'essential_metrics': {
                'system_status': 'operational',
                'data_points': len(data) if isinstance(data, dict) else 0,
                'last_success': self.last_success_time.isoformat()
            },
            'full_data_available': False
        }
    
    def _escalate_fallback(self, 
                          new_level: FallbackLevel,
                          reason: FallbackReason,
                          error: str):
        """Escalate to new fallback level."""
        if self.current_level == new_level:
            return
        
        old_level = self.current_level
        self.current_level = new_level
        
        # Record event
        event = FallbackEvent(
            event_id=f"fb_{int(time.time() * 1000000)}",
            timestamp=datetime.now(),
            from_level=old_level,
            to_level=new_level,
            reason=reason,
            error_details=error,
            data_preserved=True,
            recovery_time_estimate_seconds=self._estimate_recovery_time()
        )
        
        self.fallback_events.append(event)
        self.stats['total_fallbacks'] += 1
        
        logger.warning(f"Fallback escalation: {old_level.value} -> {new_level.value} ({reason.value})")
    
    def _escalate_to_next_level(self, error: str):
        """Escalate to next fallback level."""
        level_order = [
            FallbackLevel.PRIMARY,
            FallbackLevel.CACHE,
            FallbackLevel.SECONDARY,
            FallbackLevel.DEGRADED,
            FallbackLevel.LOCAL,
            FallbackLevel.EMERGENCY
        ]
        
        current_index = level_order.index(self.current_level)
        if current_index < len(level_order) - 1:
            self._escalate_fallback(
                level_order[current_index + 1],
                FallbackReason.ERROR,
                error
            )
    
    def _attempt_recovery(self):
        """Attempt to recover to better fallback level."""
        self.stats['recovery_attempts'] += 1
        
        # Simple recovery check - in production would test actual endpoint
        time_since_failure = (datetime.now() - self.last_success_time).total_seconds()
        
        if time_since_failure < 60:  # Recent success
            if self.current_level != FallbackLevel.PRIMARY:
                self.current_level = FallbackLevel.PRIMARY
                self.failure_count = 0
                self.stats['successful_recoveries'] += 1
                logger.info("Recovered to primary mode")
    
    def _setup_local_storage(self):
        """Setup local SQLite storage for fallback."""
        try:
            self.local_db = sqlite3.connect(
                self.local_storage_path,
                check_same_thread=False
            )
            
            cursor = self.local_db.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fallback_analytics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    data TEXT,
                    status TEXT,
                    synced INTEGER DEFAULT 0
                )
            """)
            self.local_db.commit()
            
        except Exception as e:
            logger.error(f"Failed to setup local storage: {e}")
            self.local_db = None
    
    def _store_locally(self, data: Dict[str, Any], status: str) -> Optional[str]:
        """Store data locally."""
        if not self.local_db:
            return None
        
        try:
            storage_id = f"local_{int(time.time() * 1000000)}"
            
            cursor = self.local_db.cursor()
            cursor.execute("""
                INSERT INTO fallback_analytics (id, timestamp, data, status, synced)
                VALUES (?, ?, ?, ?, 0)
            """, (
                storage_id,
                datetime.now().isoformat(),
                json.dumps(data),
                status
            ))
            
            self.local_db.commit()
            return storage_id
            
        except Exception as e:
            logger.error(f"Local storage failed: {e}")
            return None
    
    def _preserve_data(self, data: Dict[str, Any]):
        """Preserve data during failure."""
        # Try multiple preservation methods
        preserved = False
        
        # Try cache
        if self._update_cache(data):
            preserved = True
        
        # Try local storage
        if self._store_locally(data, "preserved"):
            preserved = True
        
        if preserved:
            self.stats['data_loss_prevented'] += 1
    
    def _update_cache(self, data: Dict[str, Any]) -> bool:
        """Update fallback cache."""
        try:
            cache_key = self._generate_cache_key(data)
            cache_entry = {
                'key': cache_key,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'hits': 0
            }
            
            self.cache.append(cache_entry)
            self.cache_index[cache_key] = cache_entry
            
            return True
        except Exception as e:
            logger.error(f"Cache update failed: {e}")
            return False
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for data."""
        # Simple key generation - would be more sophisticated in production
        key_parts = []
        
        for k in sorted(['type', 'source', 'category']):
            if k in data:
                key_parts.append(f"{k}:{data[k]}")
        
        return "_".join(key_parts) if key_parts else "default"
    
    def _estimate_recovery_time(self) -> int:
        """Estimate time to recovery in seconds."""
        # Simple estimation based on failure count
        base_time = 30  # 30 seconds base
        return min(base_time * (2 ** self.failure_count), 3600)  # Max 1 hour
    
    def _recovery_loop(self):
        """Background recovery monitoring."""
        while self.recovery_active:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                with self.lock:
                    # Sync local storage if primary is restored
                    if self.current_level == FallbackLevel.PRIMARY:
                        self._sync_local_storage()
                    
                    # Attempt recovery if not primary
                    if self.current_level != FallbackLevel.PRIMARY:
                        self._attempt_recovery()
                
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
    
    def _sync_local_storage(self):
        """Sync locally stored data when connection restored."""
        if not self.local_db:
            return
        
        try:
            cursor = self.local_db.cursor()
            cursor.execute("""
                SELECT id, data FROM fallback_analytics
                WHERE synced = 0
                LIMIT 10
            """)
            
            rows = cursor.fetchall()
            
            for row_id, data_str in rows:
                try:
                    # In production, would send to actual endpoint
                    logger.info(f"Syncing local data: {row_id}")
                    
                    # Mark as synced
                    cursor.execute("""
                        UPDATE fallback_analytics
                        SET synced = 1
                        WHERE id = ?
                    """, (row_id,))
                except Exception as e:
                    logger.error(f"Failed to sync {row_id}: {e}")
            
            self.local_db.commit()
            
        except Exception as e:
            logger.error(f"Local storage sync failed: {e}")
    
    def _recover_exponential(self, attempts: int) -> int:
        """Exponential backoff recovery."""
        return min(30 * (2 ** attempts), 3600)
    
    def _recover_linear(self, attempts: int) -> int:
        """Linear recovery timing."""
        return min(30 * attempts, 600)
    
    def _recover_immediate(self, attempts: int) -> int:
        """Immediate recovery attempt."""
        return 1
    
    def register_alternative_endpoint(self, endpoint: str, priority: int = 5):
        """Register alternative endpoint for fallback."""
        self.alternative_endpoints.append((priority, endpoint))
        self.alternative_endpoints.sort(key=lambda x: x[0], reverse=True)
        self.endpoint_health[endpoint] = True
        logger.info(f"Registered alternative endpoint: {endpoint}")
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get current fallback status."""
        with self.lock:
            return {
                'current_level': self.current_level.value,
                'failure_count': self.failure_count,
                'last_success': self.last_success_time.isoformat(),
                'statistics': self.stats,
                'cache_size': len(self.cache),
                'recent_events': [
                    {
                        'timestamp': e.timestamp.isoformat(),
                        'from': e.from_level.value,
                        'to': e.to_level.value,
                        'reason': e.reason.value
                    }
                    for e in list(self.fallback_events)[-5:]
                ],
                'recovery_estimate_seconds': self._estimate_recovery_time()
            }
    
    def force_recovery(self):
        """Force attempt to recover to primary."""
        with self.lock:
            logger.info("Forcing recovery attempt")
            self.failure_count = 0
            self.current_level = FallbackLevel.PRIMARY
            self.stats['recovery_attempts'] += 1
    
    def shutdown(self):
        """Shutdown fallback system."""
        self.recovery_active = False
        
        if self.recovery_thread and self.recovery_thread.is_alive():
            self.recovery_thread.join(timeout=5)
        
        if self.local_db:
            self.local_db.close()
        
        logger.info(f"Fallback System shutdown - Stats: {self.stats}")

# Global fallback system instance
fallback_system = AnalyticsFallbackSystem()