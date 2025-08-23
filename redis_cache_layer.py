#!/usr/bin/env python3
"""
🏗️ MODULE: Redis Cache Layer - Redis L2 Cache with Async Support
===============================================================

📋 PURPOSE:
    Redis L2 cache layer implementation with compression and async operations.
    Provides distributed caching capabilities with metadata tracking.

🎯 CORE FUNCTIONALITY:
    • Redis-based L2 cache with synchronous and asynchronous operations
    • Data compression for large values with automatic threshold detection
    • Comprehensive metadata tracking for access patterns and statistics
    • TTL-based expiration with automatic cleanup and health monitoring

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 10:20:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract Redis cache layer from advanced_caching_architecture.py via STEELCLAD
   └─ Changes: Created dedicated module for Redis L2 cache operations
   └─ Impact: Clean separation of Redis caching from memory cache and orchestration

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: pickle, zlib, time, logging, typing
🎯 Integration Points: Redis connection manager, cache metrics tracker
⚡ Performance Notes: Async operations with connection pooling and compression
🔒 Security Notes: Secure Redis connections with authentication and SSL support

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via caching system validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: caching_models, cache_metrics_tracker, redis_connection_manager
📤 Provides: Distributed Redis caching for L2 cache operations
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import time
import pickle
import zlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Import models
from caching_models import CacheConfig, CacheLayer
from cache_metrics_tracker import CacheMetrics
from redis_connection_manager import RedisConnectionManager

class RedisCacheLayer:
    """Redis L2 cache layer with async support"""
    
    def __init__(self, config: CacheConfig, metrics: CacheMetrics, 
                 connection_manager: RedisConnectionManager):
        self.config = config
        self.metrics = metrics
        self.connection_manager = connection_manager
        self.logger = logging.getLogger('RedisCacheLayer')
        
        # Key prefixes for organization
        self.key_prefix = "testmaster:cache:"
        self.meta_prefix = "testmaster:meta:"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        try:
            serialized = pickle.dumps(value)
            
            if self.config.enable_compression and len(serialized) > self.config.compression_threshold:
                compressed = zlib.compress(serialized)
                # Add compression marker
                return b"COMPRESSED:" + compressed
            
            return serialized
        except Exception as e:
            self.logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis storage"""
        try:
            if data.startswith(b"COMPRESSED:"):
                compressed_data = data[11:]  # Remove "COMPRESSED:" prefix
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            
            return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache (sync)"""
        start_time = time.perf_counter()
        
        client = self.connection_manager.get_sync_client()
        if not client:
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_error(CacheLayer.REDIS, "connection_error")
            return None
        
        try:
            redis_key = self.key_prefix + key
            data = client.get(redis_key)
            
            if data is None:
                response_time = (time.perf_counter() - start_time) * 1000
                self.metrics.record_miss(CacheLayer.REDIS, response_time)
                return None
            
            # Update access metadata
            meta_key = self.meta_prefix + key
            client.hincrby(meta_key, "access_count", 1)
            client.hset(meta_key, "last_accessed", datetime.now(timezone.utc).isoformat())
            
            value = self._deserialize(data)
            
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_hit(CacheLayer.REDIS, response_time)
            
            return value
            
        except Exception as e:
            self.logger.error(f"Redis get error for key {key}: {e}")
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_error(CacheLayer.REDIS, "get_error")
            return None
    
    async def get_async(self, key: str) -> Optional[Any]:
        """Get value from Redis cache (async)"""
        start_time = time.perf_counter()
        
        client = await self.connection_manager.get_async_client()
        if not client:
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_error(CacheLayer.REDIS, "connection_error")
            return None
        
        try:
            redis_key = self.key_prefix + key
            data = await client.get(redis_key)
            
            if data is None:
                response_time = (time.perf_counter() - start_time) * 1000
                self.metrics.record_miss(CacheLayer.REDIS, response_time)
                await client.close()
                return None
            
            # Update access metadata
            meta_key = self.meta_prefix + key
            await client.hincrby(meta_key, "access_count", 1)
            await client.hset(meta_key, "last_accessed", datetime.now(timezone.utc).isoformat())
            
            value = self._deserialize(data)
            
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_hit(CacheLayer.REDIS, response_time)
            
            await client.close()
            return value
            
        except Exception as e:
            self.logger.error(f"Redis async get error for key {key}: {e}")
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_error(CacheLayer.REDIS, "get_error")
            if client:
                await client.close()
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache (sync)"""
        client = self.connection_manager.get_sync_client()
        if not client:
            self.metrics.record_error(CacheLayer.REDIS, "connection_error")
            return False
        
        try:
            redis_key = self.key_prefix + key
            meta_key = self.meta_prefix + key
            
            # Serialize value
            data = self._serialize(value)
            
            # Set with TTL
            ttl_seconds = ttl or self.config.default_ttl
            client.setex(redis_key, ttl_seconds, data)
            
            # Set metadata
            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "access_count": "1",
                "size_bytes": str(len(data)),
                "ttl": str(ttl_seconds)
            }
            
            client.hset(meta_key, mapping=metadata)
            client.expire(meta_key, ttl_seconds)
            
            # Record memory usage
            self.metrics.record_memory_usage(len(data), CacheLayer.REDIS)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis set error for key {key}: {e}")
            self.metrics.record_error(CacheLayer.REDIS, "set_error")
            return False
    
    async def set_async(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache (async)"""
        client = await self.connection_manager.get_async_client()
        if not client:
            self.metrics.record_error(CacheLayer.REDIS, "connection_error")
            return False
        
        try:
            redis_key = self.key_prefix + key
            meta_key = self.meta_prefix + key
            
            # Serialize value
            data = self._serialize(value)
            
            # Set with TTL
            ttl_seconds = ttl or self.config.default_ttl
            await client.setex(redis_key, ttl_seconds, data)
            
            # Set metadata
            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "access_count": "1",
                "size_bytes": str(len(data)),
                "ttl": str(ttl_seconds)
            }
            
            await client.hset(meta_key, mapping=metadata)
            await client.expire(meta_key, ttl_seconds)
            
            # Record memory usage
            self.metrics.record_memory_usage(len(data), CacheLayer.REDIS)
            
            await client.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Redis async set error for key {key}: {e}")
            self.metrics.record_error(CacheLayer.REDIS, "set_error")
            if client:
                await client.close()
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        client = self.connection_manager.get_sync_client()
        if not client:
            self.metrics.record_error(CacheLayer.REDIS, "connection_error")
            return False
        
        try:
            redis_key = self.key_prefix + key
            meta_key = self.meta_prefix + key
            
            deleted = client.delete(redis_key, meta_key)
            return deleted > 0
            
        except Exception as e:
            self.logger.error(f"Redis delete error for key {key}: {e}")
            self.metrics.record_error(CacheLayer.REDIS, "delete_error")
            return False
    
    async def delete_async(self, key: str) -> bool:
        """Delete key from Redis cache (async)"""
        client = await self.connection_manager.get_async_client()
        if not client:
            self.metrics.record_error(CacheLayer.REDIS, "connection_error")
            return False
        
        try:
            redis_key = self.key_prefix + key
            meta_key = self.meta_prefix + key
            
            deleted = await client.delete(redis_key, meta_key)
            await client.close()
            return deleted > 0
            
        except Exception as e:
            self.logger.error(f"Redis async delete error for key {key}: {e}")
            self.metrics.record_error(CacheLayer.REDIS, "delete_error")
            if client:
                await client.close()
            return False
    
    def clear_cache(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern"""
        client = self.connection_manager.get_sync_client()
        if not client:
            return 0
        
        try:
            # Find matching cache keys
            cache_pattern = self.key_prefix + pattern
            cache_keys = client.keys(cache_pattern)
            
            # Find matching meta keys
            meta_pattern = self.meta_prefix + pattern
            meta_keys = client.keys(meta_pattern)
            
            # Delete all keys
            all_keys = cache_keys + meta_keys
            if all_keys:
                deleted = client.delete(*all_keys)
                self.logger.info(f"Cleared {deleted} Redis cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")
            self.metrics.record_error(CacheLayer.REDIS, "clear_error")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        client = self.connection_manager.get_sync_client()
        if not client:
            return {}
        
        try:
            info = client.info('memory')
            keyspace = client.info('keyspace')
            
            # Count our keys
            cache_keys = client.keys(self.key_prefix + "*")
            meta_keys = client.keys(self.meta_prefix + "*")
            
            # Get sample metadata for analysis
            sample_metadata = []
            for key in cache_keys[:10]:  # Sample first 10 keys
                meta_key = key.decode().replace(self.key_prefix, self.meta_prefix)
                metadata = client.hgetall(meta_key)
                if metadata:
                    sample_metadata.append({
                        k.decode(): v.decode() for k, v in metadata.items()
                    })
            
            return {
                'total_keys': len(cache_keys),
                'meta_keys': len(meta_keys),
                'memory_usage_bytes': info.get('used_memory', 0),
                'memory_usage_human': info.get('used_memory_human', '0B'),
                'keyspace_info': keyspace,
                'sample_metadata': sample_metadata,
                'connection_info': self.connection_manager.get_connection_info()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get Redis stats: {e}")
            return {}
    
    def get_key_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a specific cache key"""
        client = self.connection_manager.get_sync_client()
        if not client:
            return {}
        
        try:
            meta_key = self.meta_prefix + key
            metadata = client.hgetall(meta_key)
            
            if metadata:
                return {k.decode(): v.decode() for k, v in metadata.items()}
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for key {key}: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check"""
        health = {
            'redis_available': False,
            'connection_pool_healthy': False,
            'ping_successful': False,
            'error_message': None
        }
        
        try:
            # Check connection manager
            connection_healthy = self.connection_manager.test_connection()
            health['connection_pool_healthy'] = connection_healthy
            
            if connection_healthy:
                client = self.connection_manager.get_sync_client()
                if client:
                    # Test ping
                    client.ping()
                    health['ping_successful'] = True
                    health['redis_available'] = True
                    
        except Exception as e:
            health['error_message'] = str(e)
            self.logger.error(f"Redis health check failed: {e}")
        
        return health