"""
Metrics Caching Module
======================

Provides caching layer for performance optimization.

Author: TestMaster Team
"""

import time
import threading
from typing import Any, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsCache:
    """
    Time-based cache for metrics data.
    
    Provides TTL-based caching with thread-safe operations.
    """
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize the cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        logger.info(f"MetricsCache initialized with {default_ttl}s default TTL")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cached value.
        
        Args:
            key: Cache key
            default: Default value if not found or expired
            
        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._cache:
                return default
            
            entry = self._cache[key]
            
            # Check if expired
            if time.time() > entry['expires']:
                del self._cache[key]
                return default
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        with self._lock:
            self._cache[key] = {
                'value': value,
                'expires': time.time() + ttl,
                'created': time.time()
            }
    
    def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match (simple string matching)
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            
            for key in keys_to_remove:
                del self._cache[key]
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = 0
            
            for entry in self._cache.values():
                if current_time > entry['expires']:
                    expired_count += 1
            
            return {
                'total_entries': len(self._cache),
                'expired_entries': expired_count,
                'active_entries': len(self._cache) - expired_count,
                'default_ttl': self.default_ttl
            }