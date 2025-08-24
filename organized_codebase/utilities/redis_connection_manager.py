#!/usr/bin/env python3
"""
🏗️ MODULE: Redis Connection Manager - Redis Connection Pooling and Management
==============================================================================

📋 PURPOSE:
    Redis connection management with pooling, failover, and async support.
    Handles both synchronous and asynchronous Redis connections.

🎯 CORE FUNCTIONALITY:
    • Redis connection pool initialization and management
    • Synchronous and asynchronous client creation
    • Connection health monitoring and failover handling
    • Graceful connection cleanup and resource management

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 10:10:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract Redis connection management from advanced_caching_architecture.py via STEELCLAD
   └─ Changes: Created dedicated module for Redis connection pooling and management
   └─ Impact: Clean separation of Redis connectivity from caching business logic

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: redis, aioredis, asyncio, threading, logging
🎯 Integration Points: All Redis-based cache operations and layers
⚡ Performance Notes: Optimized connection pooling with health checks
🔒 Security Notes: SSL support and authentication for secure Redis connections

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via caching system validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: redis, aioredis, caching_models
📤 Provides: Redis connection management for all cache layers
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import asyncio
import logging
import threading
from typing import Optional

# Import models
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.caching_models import CacheConfig

# Redis imports with graceful fallback
try:
    import redis
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class RedisConnectionManager:
    """Manages Redis connections with pooling and failover"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.pool = None
        self.async_pool = None
        self.logger = logging.getLogger('RedisConnectionManager')
        self._lock = threading.Lock()
    
    def initialize_sync_pool(self) -> Optional['redis.ConnectionPool']:
        """Initialize synchronous Redis connection pool"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available - using memory-only caching")
            return None
        
        try:
            self.pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                max_connections=self.config.redis_pool_size,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            client = redis.Redis(connection_pool=self.pool)
            client.ping()
            
            self.logger.info(f"Redis sync pool initialized: {self.config.redis_host}:{self.config.redis_port}")
            return self.pool
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis sync pool: {e}")
            return None
    
    async def initialize_async_pool(self) -> Optional['aioredis.ConnectionPool']:
        """Initialize asynchronous Redis connection pool"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available - using memory-only caching")
            return None
        
        try:
            self.async_pool = aioredis.ConnectionPool.from_url(
                f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}",
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                max_connections=self.config.redis_pool_size,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            client = aioredis.Redis(connection_pool=self.async_pool)
            await client.ping()
            await client.close()
            
            self.logger.info(f"Redis async pool initialized: {self.config.redis_host}:{self.config.redis_port}")
            return self.async_pool
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis async pool: {e}")
            return None
    
    def get_sync_client(self) -> Optional['redis.Redis']:
        """Get synchronous Redis client"""
        if not self.pool:
            return None
        
        try:
            return redis.Redis(connection_pool=self.pool)
        except Exception as e:
            self.logger.error(f"Failed to get sync Redis client: {e}")
            return None
    
    async def get_async_client(self) -> Optional['aioredis.Redis']:
        """Get asynchronous Redis client"""
        if not self.async_pool:
            return None
        
        try:
            return aioredis.Redis(connection_pool=self.async_pool)
        except Exception as e:
            self.logger.error(f"Failed to get async Redis client: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test Redis connection health"""
        try:
            client = self.get_sync_client()
            if client:
                client.ping()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            return False
    
    async def test_async_connection(self) -> bool:
        """Test asynchronous Redis connection health"""
        try:
            client = await self.get_async_client()
            if client:
                await client.ping()
                await client.close()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Redis async connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get connection pool information"""
        info = {
            'redis_available': REDIS_AVAILABLE,
            'sync_pool_initialized': self.pool is not None,
            'async_pool_initialized': self.async_pool is not None,
            'host': self.config.redis_host,
            'port': self.config.redis_port,
            'db': self.config.redis_db,
            'ssl_enabled': self.config.redis_ssl,
            'pool_size': self.config.redis_pool_size
        }
        
        if self.pool:
            try:
                info.update({
                    'created_connections': self.pool.created_connections,
                    'available_connections': len(self.pool._available_connections),
                    'in_use_connections': len(self.pool._in_use_connections)
                })
            except AttributeError:
                # Some Redis versions may not have these attributes
                pass
        
        return info
    
    def close(self):
        """Close connection pools"""
        try:
            if self.pool:
                self.pool.disconnect()
                self.logger.info("Redis sync pool closed")
            
            if self.async_pool:
                # Schedule async pool closure
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.async_pool.disconnect())
                    else:
                        loop.run_until_complete(self.async_pool.disconnect())
                    self.logger.info("Redis async pool closed")
                except Exception as e:
                    self.logger.warning(f"Error closing async pool: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error closing Redis pools: {e}")
    
    def restart_pools(self) -> bool:
        """Restart connection pools"""
        self.logger.info("Restarting Redis connection pools")
        
        # Close existing pools
        self.close()
        
        # Reinitialize sync pool
        sync_success = self.initialize_sync_pool() is not None
        
        # Reinitialize async pool
        async_success = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._restart_async_pool())
                async_success = True
            else:
                async_success = loop.run_until_complete(self.initialize_async_pool()) is not None
        except Exception as e:
            self.logger.error(f"Error restarting async pool: {e}")
        
        success = sync_success or async_success
        if success:
            self.logger.info("Redis connection pools restarted successfully")
        else:
            self.logger.error("Failed to restart Redis connection pools")
        
        return success
    
    async def _restart_async_pool(self):
        """Helper method to restart async pool"""
        await self.initialize_async_pool()