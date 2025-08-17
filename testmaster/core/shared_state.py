"""
Shared State Management System

Cross-agent state sharing with thread-safe persistence.
Inspired by Agency-Swarm's shared_state.py pattern.

This enables coordination between test generation, verification,
and optimization agents by providing a centralized state store.
"""

import json
import pickle
import threading
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .feature_flags import FeatureFlags
try:
    from cache.intelligent_cache import IntelligentCache, CacheStrategy
    INTELLIGENT_CACHE_AVAILABLE = True
except ImportError:
    INTELLIGENT_CACHE_AVAILABLE = False


class SharedState:
    """
    Thread-safe shared state management for TestMaster components.
    
    Provides a centralized state store that can be accessed by all
    test generation, verification, and monitoring components.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern for shared state."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize shared state with configured backend."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # Get configuration
            config = FeatureFlags.get_config('layer1_test_foundation', 'shared_state')
            self.backend = config.get('backend', 'intelligent_cache' if INTELLIGENT_CACHE_AVAILABLE else 'memory')
            self.ttl_seconds = config.get('ttl', 3600)  # Default 1 hour TTL
            
            # Initialize intelligent cache if available
            self.intelligent_cache = None
            if INTELLIGENT_CACHE_AVAILABLE and self.backend == 'intelligent_cache':
                try:
                    self.intelligent_cache = IntelligentCache(
                        cache_dir="cache/shared",
                        max_size_mb=1000,
                        default_ttl=self.ttl_seconds,
                        strategy=CacheStrategy.LRU,
                        enable_compression=True,
                        enable_persistence=True
                    )
                    print("SharedState: Connected to IntelligentCache")
                except Exception as e:
                    print(f"Failed to initialize IntelligentCache: {e}")
                    self.backend = 'memory'
            
            # Initialize backend
            if self.backend == 'memory' or (self.backend == 'intelligent_cache' and not self.intelligent_cache):
                self._store = {}
                self._metadata = {}
            elif self.backend == 'file':
                self._file_path = Path('.testmaster_state.json')
                self._load_from_file()
            elif self.backend == 'redis':
                self._init_redis()
            else:
                # Fallback to memory
                self._store = {}
                self._metadata = {}
            
            # Workflow contexts
            self.contexts = {}
            self.active_workflows = set()
            
            # Statistics
            self._stats = {
                'reads': 0,
                'writes': 0,
                'hits': 0,
                'misses': 0,
                'llm_cache_hits': 0,
                'test_cache_hits': 0
            }
            
            self._initialized = True
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the shared state.
        
        Args:
            key: The key to store the value under
            value: The value to store (must be serializable)
            ttl: Time to live in seconds (overrides default)
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                self._stats['writes'] += 1
                
                # Use intelligent cache if available
                if self.intelligent_cache:
                    return self.intelligent_cache.set(key, value, ttl=ttl or self.ttl_seconds)
                
                # Calculate expiry
                expiry = None
                if ttl or self.ttl_seconds:
                    expiry = datetime.now() + timedelta(seconds=ttl or self.ttl_seconds)
                
                if self.backend == 'memory':
                    self._store[key] = value
                    self._metadata[key] = {
                        'created': datetime.now(),
                        'expiry': expiry,
                        'access_count': 0
                    }
                elif self.backend == 'file':
                    self._store[key] = value
                    self._metadata[key] = {
                        'created': datetime.now().isoformat(),
                        'expiry': expiry.isoformat() if expiry else None,
                        'access_count': 0
                    }
                    self._save_to_file()
                elif self.backend == 'redis':
                    self._redis_set(key, value, ttl)
                
                return True
            except Exception as e:
                print(f"SharedState.set error: {e}")
                return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the shared state.
        
        Args:
            key: The key to retrieve
            default: Default value if key not found or expired
            
        Returns:
            The stored value or default
        """
        with self._lock:
            try:
                self._stats['reads'] += 1
                
                # Use intelligent cache if available
                if self.intelligent_cache:
                    value = self.intelligent_cache.get(key, default)
                    if value != default:
                        self._stats['hits'] += 1
                    else:
                        self._stats['misses'] += 1
                    return value
                
                if self.backend == 'memory':
                    if key in self._store:
                        # Check expiry
                        metadata = self._metadata.get(key, {})
                        if metadata.get('expiry') and datetime.now() > metadata['expiry']:
                            del self._store[key]
                            del self._metadata[key]
                            self._stats['misses'] += 1
                            return default
                        
                        # Update access count
                        metadata['access_count'] = metadata.get('access_count', 0) + 1
                        self._stats['hits'] += 1
                        return self._store[key]
                    
                elif self.backend == 'file':
                    self._load_from_file()
                    if key in self._store:
                        metadata = self._metadata.get(key, {})
                        if metadata.get('expiry'):
                            expiry = datetime.fromisoformat(metadata['expiry'])
                            if datetime.now() > expiry:
                                del self._store[key]
                                del self._metadata[key]
                                self._save_to_file()
                                self._stats['misses'] += 1
                                return default
                        
                        self._stats['hits'] += 1
                        return self._store[key]
                
                elif self.backend == 'redis':
                    value = self._redis_get(key)
                    if value is not None:
                        self._stats['hits'] += 1
                        return value
                
                self._stats['misses'] += 1
                return default
                
            except Exception as e:
                print(f"SharedState.get error: {e}")
                return default
    
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a numeric value in the shared state.
        
        Args:
            key: The key to increment
            amount: Amount to increment by
            
        Returns:
            The new value
        """
        with self._lock:
            current = self.get(key, 0)
            if not isinstance(current, (int, float)):
                current = 0
            new_value = current + amount
            self.set(key, new_value)
            return new_value
    
    def append(self, key: str, value: Any) -> list:
        """
        Append to a list in the shared state.
        
        Args:
            key: The key containing the list
            value: Value to append
            
        Returns:
            The updated list
        """
        with self._lock:
            current = self.get(key, [])
            if not isinstance(current, list):
                current = []
            current.append(value)
            self.set(key, current)
            return current
    
    def update_dict(self, key: str, updates: Dict[str, Any]) -> dict:
        """
        Update a dictionary in the shared state.
        
        Args:
            key: The key containing the dictionary
            updates: Dictionary of updates to apply
            
        Returns:
            The updated dictionary
        """
        with self._lock:
            current = self.get(key, {})
            if not isinstance(current, dict):
                current = {}
            current.update(updates)
            self.set(key, current)
            return current
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the shared state.
        
        Args:
            key: The key to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            try:
                if self.backend == 'memory':
                    if key in self._store:
                        del self._store[key]
                        if key in self._metadata:
                            del self._metadata[key]
                        return True
                    
                elif self.backend == 'file':
                    if key in self._store:
                        del self._store[key]
                        if key in self._metadata:
                            del self._metadata[key]
                        self._save_to_file()
                        return True
                
                elif self.backend == 'redis':
                    return self._redis_delete(key)
                
                return False
                
            except Exception as e:
                print(f"SharedState.delete error: {e}")
                return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the shared state.
        
        Args:
            key: The key to check
            
        Returns:
            True if exists and not expired, False otherwise
        """
        return self.get(key) is not None
    
    def keys(self, pattern: str = "*") -> list:
        """
        Get all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            List of matching keys
        """
        with self._lock:
            if self.backend in ['memory', 'file']:
                if self.backend == 'file':
                    self._load_from_file()
                
                all_keys = list(self._store.keys())
                
                if pattern == "*":
                    return all_keys
                
                # Simple pattern matching
                import fnmatch
                return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
            
            elif self.backend == 'redis':
                return self._redis_keys(pattern)
            
            return []
    
    def clear(self, pattern: str = "*") -> int:
        """
        Clear keys matching a pattern.
        
        Args:
            pattern: Pattern to match for deletion
            
        Returns:
            Number of keys deleted
        """
        with self._lock:
            keys_to_delete = self.keys(pattern)
            count = 0
            for key in keys_to_delete:
                if self.delete(key):
                    count += 1
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats['hit_rate'] = (stats['hits'] / max(stats['reads'], 1)) * 100
            stats['backend'] = self.backend
            stats['total_keys'] = len(self.keys())
            return stats
    
    def _load_from_file(self):
        """Load state from file backend."""
        try:
            if self._file_path.exists():
                with open(self._file_path, 'r') as f:
                    data = json.load(f)
                    self._store = data.get('store', {})
                    self._metadata = data.get('metadata', {})
            else:
                self._store = {}
                self._metadata = {}
        except Exception as e:
            print(f"Error loading shared state from file: {e}")
            self._store = {}
            self._metadata = {}
    
    def _save_to_file(self):
        """Save state to file backend."""
        try:
            data = {
                'store': self._store,
                'metadata': self._metadata,
                'saved_at': datetime.now().isoformat()
            }
            with open(self._file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving shared state to file: {e}")
    
    def _init_redis(self):
        """Initialize Redis backend."""
        try:
            import redis
            config = FeatureFlags.get_config('layer1_test_foundation', 'shared_state')
            self._redis = redis.Redis(
                host=config.get('redis_host', 'localhost'),
                port=config.get('redis_port', 6379),
                db=config.get('redis_db', 0),
                decode_responses=False
            )
            self._redis.ping()
        except Exception as e:
            print(f"Redis initialization failed: {e}")
            print("   Falling back to memory backend")
            self.backend = 'memory'
            self._store = {}
            self._metadata = {}
    
    def _redis_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis."""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                self._redis.setex(f"testmaster:{key}", ttl, serialized)
            else:
                self._redis.set(f"testmaster:{key}", serialized)
        except Exception as e:
            print(f"Redis set error: {e}")
    
    def _redis_get(self, key: str) -> Any:
        """Get value from Redis."""
        try:
            value = self._redis.get(f"testmaster:{key}")
            if value:
                return pickle.loads(value)
        except Exception as e:
            print(f"Redis get error: {e}")
        return None
    
    def _redis_delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            return self._redis.delete(f"testmaster:{key}") > 0
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def _redis_keys(self, pattern: str) -> list:
        """Get keys from Redis."""
        try:
            redis_pattern = f"testmaster:{pattern}"
            keys = self._redis.keys(redis_pattern)
            return [k.decode().replace('testmaster:', '') for k in keys]
        except Exception as e:
            print(f"Redis keys error: {e}")
            return []


# Global instance for easy access
_shared_state = None


def get_shared_state() -> SharedState:
    """
    Get the global shared state instance.
    
    Returns:
        SharedState instance
    """
    global _shared_state
    if _shared_state is None:
        _shared_state = SharedState()
    return _shared_state


# Convenience functions
def state_get(key: str, default: Any = None) -> Any:
    """Get value from shared state."""
    return get_shared_state().get(key, default)


def state_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set value in shared state."""
    return get_shared_state().set(key, value, ttl)


def state_increment(key: str, amount: int = 1) -> int:
    """Increment numeric value in shared state."""
    return get_shared_state().increment(key, amount)


def state_append(key: str, value: Any) -> list:
    """Append to list in shared state."""
    return get_shared_state().append(key, value)


def state_exists(key: str) -> bool:
    """Check if key exists in shared state."""
    return get_shared_state().exists(key)


@dataclass
class SharedContext:
    """Shared context across test generation pipeline."""
    workflow_id: str
    created_at: datetime = field(default_factory=datetime.now)
    modules_processed: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    llm_responses: Dict[str, Any] = field(default_factory=dict)
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def cache_llm_response(prompt: str, response: str, model: str = "default") -> None:
    """Cache LLM response for reuse."""
    state = get_shared_state()
    if state.intelligent_cache:
        cache_key = state.intelligent_cache.generate_key("llm", model, prompt)
        state.intelligent_cache.set(cache_key, {
            "prompt": prompt,
            "response": response,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }, ttl=3600)
    else:
        # Fallback to regular state
        state.set(f"llm_{model}_{hash(prompt)}", response, ttl=3600)


def get_cached_llm_response(prompt: str, model: str = "default") -> Optional[str]:
    """Get cached LLM response if available."""
    state = get_shared_state()
    if state.intelligent_cache:
        cache_key = state.intelligent_cache.generate_key("llm", model, prompt)
        cached = state.intelligent_cache.get(cache_key)
        if cached:
            state._stats['llm_cache_hits'] += 1
            return cached.get("response")
    else:
        # Fallback to regular state
        response = state.get(f"llm_{model}_{hash(prompt)}")
        if response:
            state._stats['llm_cache_hits'] += 1
            return response
    return None


def cache_test_result(module_path: str, test_code: str, score: float) -> None:
    """Cache test generation result."""
    state = get_shared_state()
    if state.intelligent_cache:
        cache_key = state.intelligent_cache.generate_key("test", module_path)
        state.intelligent_cache.set(cache_key, {
            "module_path": module_path,
            "test_code": test_code,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }, ttl=86400)
    else:
        state.set(f"test_{module_path}", {
            "test_code": test_code,
            "score": score
        }, ttl=86400)


def get_cached_test_result(module_path: str) -> Optional[Dict[str, Any]]:
    """Get cached test result if available."""
    state = get_shared_state()
    if state.intelligent_cache:
        cache_key = state.intelligent_cache.generate_key("test", module_path)
        cached = state.intelligent_cache.get(cache_key)
        if cached:
            state._stats['test_cache_hits'] += 1
            return cached
    else:
        result = state.get(f"test_{module_path}")
        if result:
            state._stats['test_cache_hits'] += 1
            return result
    return None


def get_context_manager() -> SharedState:
    """Alias for get_shared_state() for compatibility with base.py."""
    return get_shared_state()