#!/usr/bin/env python3
"""
🏗️ MODULE: Connection Pool Manager - Database Connection Pool Management
==================================================================

📋 PURPOSE:
    Manages database connection pooling for optimal performance and resource utilization.
    Handles connection lifecycle, validation, monitoring, and thread-safe access.

🎯 CORE FUNCTIONALITY:
    • Database connection pool initialization and management
    • Thread-safe connection borrowing and returning
    • Connection validation and health monitoring
    • Pool statistics and performance metrics

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 08:30:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract connection pool manager from database_performance_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for database connection pool management
   └─ Impact: Improved modularity and single responsibility for connection management

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: sqlite3, threading, logging, time
🎯 Integration Points: database_optimization_models.py, database_optimization_core.py
⚡ Performance Notes: Thread-safe connection pooling with validation
🔒 Security Notes: Connection validation and timeout protection

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via connection pool metrics | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: database_optimization_models for ConnectionPoolConfig
📤 Provides: Connection pool management for database optimization
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import sqlite3
import threading
import logging
import time
from typing import Dict, Optional, Set
from collections import deque
from contextlib import contextmanager

# Import data models
from database_optimization_models import ConnectionPoolConfig


class ConnectionPoolManager:
    """Manages database connection pooling for optimal performance"""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.available_connections: deque = deque()
        self.active_connections: Set = set()
        self.connection_stats = {
            'created': 0,
            'destroyed': 0,
            'borrowed': 0,
            'returned': 0,
            'validation_failures': 0,
            'timeouts': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Monitoring
        self.logger = logging.getLogger('ConnectionPoolManager')
        self.start_time = time.time()
        
        # Initialize minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize pool with minimum connections"""
        with self.lock:
            for _ in range(self.config.min_connections):
                conn = self._create_connection()
                if conn:
                    self.available_connections.append({
                        'connection': conn,
                        'created_time': time.time(),
                        'last_used': time.time()
                    })

    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection"""
        try:
            # In production, this would connect to PostgreSQL, MySQL, etc.
            conn = sqlite3.connect(':memory:', check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            
            self.connection_stats['created'] += 1
            self.logger.debug("Created new database connection")
            
            return conn
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return None

    def _validate_connection(self, conn_data: Dict) -> bool:
        """Validate connection is still working"""
        try:
            conn = conn_data['connection']
            cursor = conn.execute(self.config.validation_query)
            cursor.fetchone()
            return True
        except Exception as e:
            self.connection_stats['validation_failures'] += 1
            self.logger.warning(f"Connection validation failed: {e}")
            return False

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn_data = None
        
        try:
            # Get connection from pool
            conn_data = self._borrow_connection()
            if not conn_data:
                raise Exception("Unable to get connection from pool")
            
            yield conn_data['connection']
            
        finally:
            # Return connection to pool
            if conn_data:
                self._return_connection(conn_data)

    def _borrow_connection(self) -> Optional[Dict]:
        """Borrow a connection from the pool"""
        start_time = time.time()
        
        while time.time() - start_time < self.config.connection_timeout:
            with self.lock:
                # Try to get available connection
                if self.available_connections:
                    conn_data = self.available_connections.popleft()
                    
                    # Validate connection
                    if self._validate_connection(conn_data):
                        conn_data['last_used'] = time.time()
                        self.active_connections.add(id(conn_data['connection']))
                        self.connection_stats['borrowed'] += 1
                        return conn_data
                    else:
                        # Connection invalid, destroy it
                        self._destroy_connection(conn_data)
                
                # Create new connection if under max limit
                total_connections = len(self.available_connections) + len(self.active_connections)
                if total_connections < self.config.max_connections:
                    conn = self._create_connection()
                    if conn:
                        conn_data = {
                            'connection': conn,
                            'created_time': time.time(),
                            'last_used': time.time()
                        }
                        self.active_connections.add(id(conn))
                        self.connection_stats['borrowed'] += 1
                        return conn_data
            
            # Wait before retrying
            time.sleep(0.01)
        
        # Timeout occurred
        self.connection_stats['timeouts'] += 1
        self.logger.warning("Connection pool timeout")
        return None

    def _return_connection(self, conn_data: Dict):
        """Return a connection to the pool"""
        with self.lock:
            conn_id = id(conn_data['connection'])
            if conn_id in self.active_connections:
                self.active_connections.remove(conn_id)
                
                # Check if connection is still valid and not too old
                current_time = time.time()
                connection_age = current_time - conn_data['created_time']
                
                if (self._validate_connection(conn_data) and 
                    connection_age < self.config.idle_timeout):
                    # Return to pool
                    conn_data['last_used'] = current_time
                    self.available_connections.append(conn_data)
                    self.connection_stats['returned'] += 1
                else:
                    # Connection too old or invalid, destroy it
                    self._destroy_connection(conn_data)

    def _destroy_connection(self, conn_data: Dict):
        """Destroy a connection"""
        try:
            conn_data['connection'].close()
            self.connection_stats['destroyed'] += 1
            self.logger.debug("Destroyed database connection")
        except Exception as e:
            self.logger.error(f"Error destroying connection: {e}")

    def cleanup_idle_connections(self):
        """Clean up idle connections that exceed idle timeout"""
        current_time = time.time()
        
        with self.lock:
            active_connections = deque()
            
            while self.available_connections:
                conn_data = self.available_connections.popleft()
                
                # Check if connection has been idle too long
                idle_time = current_time - conn_data['last_used']
                if idle_time > self.config.idle_timeout:
                    self._destroy_connection(conn_data)
                else:
                    active_connections.append(conn_data)
            
            self.available_connections = active_connections

    def get_pool_status(self) -> Dict[str, any]:
        """Get current pool status and statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            total_connections = len(self.available_connections) + len(self.active_connections)
            
            return {
                'total_connections': total_connections,
                'available_connections': len(self.available_connections),
                'active_connections': len(self.active_connections),
                'max_connections': self.config.max_connections,
                'min_connections': self.config.min_connections,
                'pool_utilization_percent': (len(self.active_connections) / self.config.max_connections) * 100,
                'uptime_seconds': uptime,
                'statistics': self.connection_stats.copy(),
                'configuration': {
                    'connection_timeout': self.config.connection_timeout,
                    'idle_timeout': self.config.idle_timeout,
                    'validation_query': self.config.validation_query,
                    'retry_attempts': self.config.retry_attempts,
                    'retry_delay': self.config.retry_delay
                }
            }

    def shutdown(self):
        """Shutdown connection pool and close all connections"""
        with self.lock:
            # Close all available connections
            while self.available_connections:
                conn_data = self.available_connections.popleft()
                self._destroy_connection(conn_data)
            
            # Note: Active connections will be closed when returned
            self.logger.info(f"Connection pool shutdown. Total connections created: {self.connection_stats['created']}")

    def execute_with_connection(self, query: str, parameters: tuple = None):
        """Execute a query using a connection from the pool"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            # Return results for SELECT queries
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                conn.commit()
                return cursor.rowcount

    def execute_transaction(self, queries_and_params: list):
        """Execute multiple queries in a transaction"""
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                for query, params in queries_and_params:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                
                conn.commit()
                return True
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Transaction failed: {e}")
                raise