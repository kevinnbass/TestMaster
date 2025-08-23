#!/usr/bin/env python3
"""
AGENT BETA - DATABASE PERFORMANCE OPTIMIZER
Phase 1, Hours 10-15: Database Performance Optimization
======================================================

Advanced database optimization system with query analysis, index optimization,
connection pool configuration, and database-level caching implementation.

Created: 2025-08-23 02:45:00 UTC
Agent: Beta (Performance Optimization Specialist)
Phase: 1 (Hours 10-15)
"""

import os
import sys
import time
import json
import sqlite3
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import re
import statistics
from contextlib import contextmanager
import psutil

# Import monitoring infrastructure from previous phase
try:
    from performance_monitoring_infrastructure import PerformanceMonitoringSystem, MonitoringConfig
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

@dataclass
class QueryAnalysis:
    """Analysis results for a database query"""
    query_hash: str
    query_text: str
    execution_count: int
    total_execution_time: float
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    rows_examined: int = 0
    rows_returned: int = 0
    index_usage: List[str] = None
    optimization_suggestions: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.index_usage is None:
            self.index_usage = []
        if self.optimization_suggestions is None:
            self.optimization_suggestions = []
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class IndexRecommendation:
    """Database index recommendation"""
    table_name: str
    columns: List[str]
    index_type: str  # 'btree', 'hash', 'composite'
    estimated_benefit: float
    queries_affected: List[str]
    creation_sql: str
    impact_score: float
    rationale: str

@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration"""
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    validation_query: str = "SELECT 1"
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_monitoring: bool = True

class QueryPerformanceAnalyzer:
    """Analyzes database query performance and provides optimization recommendations"""
    
    def __init__(self, analysis_db_path: str = "query_analysis.db"):
        self.analysis_db = Path(analysis_db_path)
        self.query_cache: Dict[str, QueryAnalysis] = {}
        self.active_queries: Dict[str, Dict] = {}
        
        # Performance tracking
        self.slow_query_threshold = 0.1  # 100ms
        self.very_slow_query_threshold = 1.0  # 1 second
        
        # Initialize analysis database
        self._init_analysis_db()
        
        # Set up logging
        self.logger = logging.getLogger('QueryPerformanceAnalyzer')
        
    def _init_analysis_db(self):
        """Initialize analysis database schema"""
        with sqlite3.connect(self.analysis_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    execution_count INTEGER DEFAULT 0,
                    total_execution_time REAL DEFAULT 0.0,
                    avg_execution_time REAL DEFAULT 0.0,
                    min_execution_time REAL DEFAULT 0.0,
                    max_execution_time REAL DEFAULT 0.0,
                    rows_examined INTEGER DEFAULT 0,
                    rows_returned INTEGER DEFAULT 0,
                    index_usage TEXT,
                    optimization_suggestions TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    UNIQUE(query_hash)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    rows_examined INTEGER DEFAULT 0,
                    rows_returned INTEGER DEFAULT 0,
                    execution_plan TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(query_hash) REFERENCES query_analysis(query_hash)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_hash ON query_analysis(query_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_timestamp ON query_executions(timestamp)
            """)

    def normalize_query(self, query: str) -> str:
        """Normalize query for analysis (remove literals, etc.)"""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query.strip())
        
        # Replace literals with placeholders
        # String literals
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        
        # Numeric literals
        normalized = re.sub(r'\b\d+\.?\d*\b', '?', normalized)
        
        # IN clauses with multiple values
        normalized = re.sub(r'IN\s*\([^)]+\)', 'IN (?)', normalized, flags=re.IGNORECASE)
        
        return normalized.upper()

    def generate_query_hash(self, normalized_query: str) -> str:
        """Generate hash for normalized query"""
        return hashlib.md5(normalized_query.encode()).hexdigest()[:16]

    @contextmanager
    def monitor_query(self, query: str, connection=None):
        """Context manager to monitor query execution"""
        normalized_query = self.normalize_query(query)
        query_hash = self.generate_query_hash(normalized_query)
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        execution_data = {
            'query_hash': query_hash,
            'query_text': normalized_query,
            'original_query': query,
            'start_time': start_time,
            'start_memory': start_memory
        }
        
        self.active_queries[query_hash] = execution_data
        
        try:
            yield execution_data
        except Exception as e:
            execution_data['error'] = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Record query execution
            self._record_query_execution(
                query_hash=query_hash,
                normalized_query=normalized_query,
                original_query=query,
                execution_time=execution_time,
                memory_delta=memory_delta,
                execution_data=execution_data
            )
            
            # Remove from active queries
            self.active_queries.pop(query_hash, None)

    def _record_query_execution(self, query_hash: str, normalized_query: str, 
                               original_query: str, execution_time: float, 
                               memory_delta: int, execution_data: Dict):
        """Record query execution in analysis database"""
        timestamp = datetime.now(timezone.utc)
        
        # Get execution details
        rows_examined = execution_data.get('rows_examined', 0)
        rows_returned = execution_data.get('rows_returned', 0)
        execution_plan = execution_data.get('execution_plan', '')
        
        # Update or insert query analysis
        with sqlite3.connect(self.analysis_db) as conn:
            # Check if query exists
            cursor = conn.execute(
                "SELECT * FROM query_analysis WHERE query_hash = ?",
                (query_hash,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing analysis
                new_count = existing[3] + 1
                new_total_time = existing[4] + execution_time
                new_avg_time = new_total_time / new_count
                new_min_time = min(existing[6], execution_time)
                new_max_time = max(existing[7], execution_time)
                
                conn.execute("""
                    UPDATE query_analysis 
                    SET execution_count = ?, 
                        total_execution_time = ?,
                        avg_execution_time = ?,
                        min_execution_time = ?,
                        max_execution_time = ?,
                        last_seen = ?
                    WHERE query_hash = ?
                """, (new_count, new_total_time, new_avg_time, new_min_time, 
                     new_max_time, timestamp.isoformat(), query_hash))
            else:
                # Insert new analysis
                conn.execute("""
                    INSERT INTO query_analysis (
                        query_hash, query_text, execution_count, total_execution_time,
                        avg_execution_time, min_execution_time, max_execution_time,
                        rows_examined, rows_returned, first_seen, last_seen
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (query_hash, normalized_query, 1, execution_time, execution_time,
                     execution_time, execution_time, rows_examined, rows_returned,
                     timestamp.isoformat(), timestamp.isoformat()))
            
            # Insert execution record
            conn.execute("""
                INSERT INTO query_executions (
                    query_hash, execution_time, rows_examined, rows_returned,
                    execution_plan, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (query_hash, execution_time, rows_examined, rows_returned,
                 execution_plan, timestamp.isoformat()))
        
        # Log slow queries
        if execution_time > self.very_slow_query_threshold:
            self.logger.warning(f"Very slow query detected: {execution_time:.4f}s - {normalized_query[:100]}...")
        elif execution_time > self.slow_query_threshold:
            self.logger.info(f"Slow query detected: {execution_time:.4f}s - {normalized_query[:100]}...")

    def get_slow_queries(self, limit: int = 20) -> List[QueryAnalysis]:
        """Get slowest queries by average execution time"""
        with sqlite3.connect(self.analysis_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM query_analysis 
                ORDER BY avg_execution_time DESC 
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                analysis = QueryAnalysis(
                    query_hash=row[1],
                    query_text=row[2],
                    execution_count=row[3],
                    total_execution_time=row[4],
                    avg_execution_time=row[5],
                    min_execution_time=row[6],
                    max_execution_time=row[7],
                    rows_examined=row[8],
                    rows_returned=row[9],
                    index_usage=json.loads(row[10]) if row[10] else [],
                    optimization_suggestions=json.loads(row[11]) if row[11] else [],
                    timestamp=datetime.fromisoformat(row[13]) if row[13] else None
                )
                results.append(analysis)
            
            return results

    def get_frequent_queries(self, limit: int = 20) -> List[QueryAnalysis]:
        """Get most frequently executed queries"""
        with sqlite3.connect(self.analysis_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM query_analysis 
                ORDER BY execution_count DESC 
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                analysis = QueryAnalysis(
                    query_hash=row[1],
                    query_text=row[2],
                    execution_count=row[3],
                    total_execution_time=row[4],
                    avg_execution_time=row[5],
                    min_execution_time=row[6],
                    max_execution_time=row[7],
                    rows_examined=row[8],
                    rows_returned=row[9]
                )
                results.append(analysis)
            
            return results

    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns and identify optimization opportunities"""
        with sqlite3.connect(self.analysis_db) as conn:
            # Basic statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_unique_queries,
                    SUM(execution_count) as total_executions,
                    AVG(avg_execution_time) as overall_avg_time,
                    MAX(max_execution_time) as slowest_query_time,
                    SUM(CASE WHEN avg_execution_time > ? THEN 1 ELSE 0 END) as slow_queries,
                    SUM(CASE WHEN avg_execution_time > ? THEN 1 ELSE 0 END) as very_slow_queries
                FROM query_analysis
            """, (self.slow_query_threshold, self.very_slow_query_threshold))
            
            stats_row = cursor.fetchone()
            
            # Query type distribution
            cursor = conn.execute("""
                SELECT 
                    CASE 
                        WHEN query_text LIKE 'SELECT%' THEN 'SELECT'
                        WHEN query_text LIKE 'INSERT%' THEN 'INSERT'
                        WHEN query_text LIKE 'UPDATE%' THEN 'UPDATE'
                        WHEN query_text LIKE 'DELETE%' THEN 'DELETE'
                        ELSE 'OTHER'
                    END as query_type,
                    COUNT(*) as count,
                    AVG(avg_execution_time) as avg_time
                FROM query_analysis
                GROUP BY query_type
                ORDER BY count DESC
            """)
            
            query_types = {}
            for row in cursor.fetchall():
                query_types[row[0]] = {
                    'count': row[1],
                    'avg_time': row[2]
                }
            
            return {
                'total_unique_queries': stats_row[0],
                'total_executions': stats_row[1],
                'overall_avg_time': stats_row[2],
                'slowest_query_time': stats_row[3],
                'slow_queries': stats_row[4],
                'very_slow_queries': stats_row[5],
                'query_type_distribution': query_types,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }

class IndexOptimizer:
    """Generates index recommendations based on query analysis"""
    
    def __init__(self, query_analyzer: QueryPerformanceAnalyzer):
        self.query_analyzer = query_analyzer
        self.logger = logging.getLogger('IndexOptimizer')
        
    def analyze_table_access_patterns(self, database_path: str) -> Dict[str, Any]:
        """Analyze table access patterns from query logs"""
        table_patterns = defaultdict(lambda: {
            'select_count': 0,
            'where_columns': defaultdict(int),
            'join_columns': defaultdict(int),
            'order_columns': defaultdict(int),
            'group_columns': defaultdict(int),
            'total_execution_time': 0.0,
            'avg_rows_examined': 0
        })
        
        # Analyze existing queries
        slow_queries = self.query_analyzer.get_slow_queries(100)
        frequent_queries = self.query_analyzer.get_frequent_queries(100)
        
        all_queries = {}
        for q in slow_queries + frequent_queries:
            all_queries[q.query_hash] = q
        
        for query in all_queries.values():
            self._analyze_query_for_patterns(query.query_text, table_patterns)
        
        return dict(table_patterns)

    def _analyze_query_for_patterns(self, query: str, patterns: Dict):
        """Extract table access patterns from a single query"""
        # Simple SQL parsing (would use proper SQL parser in production)
        query_upper = query.upper()
        
        # Extract table names from FROM clauses
        from_match = re.search(r'FROM\s+(\w+)', query_upper)
        if from_match:
            table_name = from_match.group(1).lower()
            patterns[table_name]['select_count'] += 1
            
            # Extract WHERE clause columns
            where_columns = self._extract_where_columns(query_upper)
            for col in where_columns:
                patterns[table_name]['where_columns'][col] += 1
            
            # Extract ORDER BY columns
            order_columns = self._extract_order_columns(query_upper)
            for col in order_columns:
                patterns[table_name]['order_columns'][col] += 1
            
            # Extract GROUP BY columns
            group_columns = self._extract_group_columns(query_upper)
            for col in group_columns:
                patterns[table_name]['group_columns'][col] += 1

    def _extract_where_columns(self, query: str) -> List[str]:
        """Extract column names from WHERE clauses"""
        columns = []
        
        # Find WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|\s+GROUP\s+BY|\s+LIMIT|\s*$)', query)
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract column names (simple pattern matching)
            col_matches = re.findall(r'\b(\w+)\s*[=<>!]', where_clause)
            columns.extend([col.lower() for col in col_matches])
            
            # Extract LIKE patterns
            like_matches = re.findall(r'\b(\w+)\s+LIKE', where_clause)
            columns.extend([col.lower() for col in like_matches])
        
        return columns

    def _extract_order_columns(self, query: str) -> List[str]:
        """Extract column names from ORDER BY clauses"""
        columns = []
        
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s*$)', query)
        if order_match:
            order_clause = order_match.group(1)
            col_matches = re.findall(r'\b(\w+)', order_clause)
            columns.extend([col.lower() for col in col_matches if col not in ('ASC', 'DESC')])
        
        return columns

    def _extract_group_columns(self, query: str) -> List[str]:
        """Extract column names from GROUP BY clauses"""
        columns = []
        
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|\s+HAVING|\s+LIMIT|\s*$)', query)
        if group_match:
            group_clause = group_match.group(1)
            col_matches = re.findall(r'\b(\w+)', group_clause)
            columns.extend([col.lower() for col in col_matches])
        
        return columns

    def generate_index_recommendations(self, database_path: str) -> List[IndexRecommendation]:
        """Generate index recommendations based on query analysis"""
        recommendations = []
        
        # Analyze access patterns
        patterns = self.analyze_table_access_patterns(database_path)
        
        for table_name, pattern_data in patterns.items():
            # Recommend indexes for frequently used WHERE columns
            for column, usage_count in pattern_data['where_columns'].items():
                if usage_count >= 3:  # Used in at least 3 queries
                    impact_score = usage_count * 10
                    
                    recommendation = IndexRecommendation(
                        table_name=table_name,
                        columns=[column],
                        index_type='btree',
                        estimated_benefit=impact_score,
                        queries_affected=[],
                        creation_sql=f"CREATE INDEX idx_{table_name}_{column} ON {table_name} ({column})",
                        impact_score=impact_score,
                        rationale=f"Column '{column}' used in WHERE clauses {usage_count} times"
                    )
                    recommendations.append(recommendation)
            
            # Recommend composite indexes for ORDER BY patterns
            order_columns = list(pattern_data['order_columns'].keys())
            if len(order_columns) >= 2:
                column_list = order_columns[:3]  # Limit to 3 columns
                column_str = ', '.join(column_list)
                impact_score = sum(pattern_data['order_columns'].values())
                
                recommendation = IndexRecommendation(
                    table_name=table_name,
                    columns=column_list,
                    index_type='composite',
                    estimated_benefit=impact_score,
                    queries_affected=[],
                    creation_sql=f"CREATE INDEX idx_{table_name}_{'_'.join(column_list)} ON {table_name} ({column_str})",
                    impact_score=impact_score,
                    rationale=f"Composite index for ORDER BY patterns: {column_str}"
                )
                recommendations.append(recommendation)
        
        # Sort by impact score
        recommendations.sort(key=lambda x: x.impact_score, reverse=True)
        
        return recommendations[:10]  # Return top 10 recommendations

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
                        # Connection invalid, try to create new one
                        self._destroy_connection(conn_data)
                        continue
                
                # No available connections, try to create new one
                if len(self.active_connections) < self.config.max_connections:
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
            
            # Wait before retry
            time.sleep(0.1)
        
        # Timeout reached
        self.connection_stats['timeouts'] += 1
        self.logger.error("Connection pool timeout")
        return None

    def _return_connection(self, conn_data: Dict):
        """Return connection to the pool"""
        with self.lock:
            conn = conn_data['connection']
            conn_id = id(conn)
            
            if conn_id in self.active_connections:
                self.active_connections.remove(conn_id)
                
                # Check if connection is still valid and not too old
                max_age = 3600  # 1 hour
                if (time.time() - conn_data['created_time'] < max_age and 
                    self._validate_connection(conn_data)):
                    
                    # Return to available pool
                    conn_data['last_used'] = time.time()
                    self.available_connections.append(conn_data)
                    self.connection_stats['returned'] += 1
                else:
                    # Connection too old or invalid, destroy it
                    self._destroy_connection(conn_data)

    def _destroy_connection(self, conn_data: Dict):
        """Destroy a database connection"""
        try:
            conn_data['connection'].close()
            self.connection_stats['destroyed'] += 1
            self.logger.debug("Destroyed database connection")
        except Exception as e:
            self.logger.error(f"Error destroying connection: {e}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self.lock:
            return {
                'available_connections': len(self.available_connections),
                'active_connections': len(self.active_connections),
                'total_connections': len(self.available_connections) + len(self.active_connections),
                'max_connections': self.config.max_connections,
                'min_connections': self.config.min_connections,
                'connection_stats': self.connection_stats.copy(),
                'uptime_seconds': time.time() - self.start_time
            }

    def cleanup(self):
        """Clean up all connections"""
        with self.lock:
            # Close available connections
            while self.available_connections:
                conn_data = self.available_connections.popleft()
                self._destroy_connection(conn_data)
            
            # Note: Active connections will be closed when returned
            self.logger.info("Connection pool cleaned up")

class DatabasePerformanceOptimizer:
    """Main database performance optimization orchestrator"""
    
    def __init__(self, database_path: str = "testmaster.db", 
                 enable_monitoring: bool = True):
        self.database_path = Path(database_path)
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.query_analyzer = QueryPerformanceAnalyzer()
        self.index_optimizer = IndexOptimizer(self.query_analyzer)
        self.connection_pool = ConnectionPoolManager(ConnectionPoolConfig())
        
        # Performance monitoring integration
        self.monitoring_system = None
        if enable_monitoring and MONITORING_AVAILABLE:
            config = MonitoringConfig(
                collection_interval=5.0,
                alert_channels=['console'],
                enable_prometheus=False,  # Don't conflict with main monitoring
                enable_alerting=False
            )
            self.monitoring_system = PerformanceMonitoringSystem(config)
            self._setup_database_metrics()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('DatabasePerformanceOptimizer')
        
        # Create sample database for testing
        self._create_sample_database()

    def _setup_database_metrics(self):
        """Set up database-specific metrics"""
        if not self.monitoring_system:
            return
        
        # Add custom database metrics
        self.monitoring_system.add_custom_metric(
            "db_active_connections",
            lambda: len(self.connection_pool.active_connections),
            unit="count",
            help_text="Number of active database connections"
        )
        
        self.monitoring_system.add_custom_metric(
            "db_available_connections", 
            lambda: len(self.connection_pool.available_connections),
            unit="count",
            help_text="Number of available database connections"
        )
        
        self.monitoring_system.add_custom_metric(
            "db_query_rate",
            lambda: self.connection_pool.connection_stats['borrowed'],
            unit="count",
            help_text="Total database queries executed"
        )

    def _create_sample_database(self):
        """Create sample database with test data for optimization"""
        if self.database_path.exists():
            return
            
        with sqlite3.connect(self.database_path) as conn:
            # Create sample tables
            conn.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    email TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.execute("""
                CREATE TABLE test_results (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    test_name TEXT NOT NULL,
                    score INTEGER,
                    duration_ms INTEGER,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE performance_metrics (
                    id INTEGER PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT
                )
            """)
            
            # Insert sample data
            import random
            from datetime import datetime, timedelta
            
            # Sample users
            for i in range(1000):
                created_at = datetime.now() - timedelta(days=random.randint(1, 365))
                last_login = created_at + timedelta(days=random.randint(0, 30))
                
                conn.execute("""
                    INSERT INTO users (username, email, created_at, last_login, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"user_{i}", f"user_{i}@test.com", created_at.isoformat(),
                     last_login.isoformat(), random.choice(['active', 'inactive'])))
            
            # Sample test results
            for i in range(5000):
                user_id = random.randint(1, 1000)
                test_name = random.choice(['unit_test', 'integration_test', 'performance_test'])
                score = random.randint(0, 100)
                duration_ms = random.randint(10, 5000)
                executed_at = datetime.now() - timedelta(days=random.randint(0, 90))
                
                conn.execute("""
                    INSERT INTO test_results (user_id, test_name, score, duration_ms, executed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, test_name, score, duration_ms, executed_at.isoformat()))
            
            # Sample performance metrics
            for i in range(10000):
                metric_name = random.choice(['cpu_usage', 'memory_usage', 'response_time'])
                metric_value = random.uniform(0, 100)
                timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))
                
                conn.execute("""
                    INSERT INTO performance_metrics (metric_name, metric_value, timestamp, tags)
                    VALUES (?, ?, ?, ?)
                """, (metric_name, metric_value, timestamp.isoformat(), '{}'))
        
        self.logger.info(f"Created sample database: {self.database_path}")

    def benchmark_queries(self) -> Dict[str, Any]:
        """Benchmark common query patterns"""
        benchmark_results = {}
        
        # Define benchmark queries
        queries = {
            'simple_select': "SELECT * FROM users WHERE status = 'active'",
            'complex_join': """
                SELECT u.username, COUNT(tr.id) as test_count, AVG(tr.score) as avg_score
                FROM users u 
                LEFT JOIN test_results tr ON u.id = tr.user_id
                WHERE u.created_at > '2024-01-01'
                GROUP BY u.id, u.username
                ORDER BY avg_score DESC
            """,
            'aggregation': """
                SELECT metric_name, AVG(metric_value), MAX(metric_value), MIN(metric_value)
                FROM performance_metrics 
                WHERE timestamp > datetime('now', '-1 day')
                GROUP BY metric_name
            """,
            'range_query': """
                SELECT * FROM test_results 
                WHERE executed_at BETWEEN '2024-01-01' AND '2024-12-31'
                AND score > 80
                ORDER BY score DESC
                LIMIT 100
            """,
            'count_query': "SELECT COUNT(*) FROM users WHERE last_login > datetime('now', '-30 days')"
        }
        
        self.logger.info("Starting query benchmarks...")
        
        with sqlite3.connect(self.database_path) as conn:
            for query_name, query_sql in queries.items():
                # Warm up
                conn.execute(query_sql).fetchall()
                
                # Benchmark multiple executions
                execution_times = []
                for _ in range(10):
                    with self.query_analyzer.monitor_query(query_sql) as query_data:
                        start_time = time.perf_counter()
                        result = conn.execute(query_sql).fetchall()
                        end_time = time.perf_counter()
                        
                        execution_time = end_time - start_time
                        execution_times.append(execution_time)
                        
                        # Record additional data
                        query_data['rows_returned'] = len(result)
                
                # Calculate statistics
                benchmark_results[query_name] = {
                    'avg_time': statistics.mean(execution_times),
                    'min_time': min(execution_times),
                    'max_time': max(execution_times),
                    'median_time': statistics.median(execution_times),
                    'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    'executions': len(execution_times),
                    'query_sql': query_sql
                }
                
                self.logger.info(f"Benchmarked {query_name}: {benchmark_results[query_name]['avg_time']:.4f}s avg")
        
        return benchmark_results

    def optimize_database(self) -> Dict[str, Any]:
        """Perform comprehensive database optimization"""
        optimization_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'phases': {}
        }
        
        # Phase 1: Benchmark current performance
        self.logger.info("Phase 1: Benchmarking current performance...")
        baseline_benchmarks = self.benchmark_queries()
        optimization_results['phases']['baseline_benchmark'] = baseline_benchmarks
        
        # Phase 2: Analyze query patterns
        self.logger.info("Phase 2: Analyzing query patterns...")
        query_patterns = self.query_analyzer.analyze_query_patterns()
        optimization_results['phases']['query_analysis'] = query_patterns
        
        # Phase 3: Generate index recommendations
        self.logger.info("Phase 3: Generating index recommendations...")
        index_recommendations = self.index_optimizer.generate_index_recommendations(str(self.database_path))
        optimization_results['phases']['index_recommendations'] = [
            asdict(rec) for rec in index_recommendations
        ]
        
        # Phase 4: Apply optimizations
        self.logger.info("Phase 4: Applying database optimizations...")
        applied_optimizations = self._apply_optimizations(index_recommendations)
        optimization_results['phases']['applied_optimizations'] = applied_optimizations
        
        # Phase 5: Benchmark optimized performance
        self.logger.info("Phase 5: Benchmarking optimized performance...")
        optimized_benchmarks = self.benchmark_queries()
        optimization_results['phases']['optimized_benchmark'] = optimized_benchmarks
        
        # Phase 6: Calculate improvement
        improvement_analysis = self._calculate_improvements(baseline_benchmarks, optimized_benchmarks)
        optimization_results['phases']['improvement_analysis'] = improvement_analysis
        
        return optimization_results

    def _apply_optimizations(self, recommendations: List[IndexRecommendation]) -> Dict[str, Any]:
        """Apply database optimizations"""
        applied = {
            'indexes_created': 0,
            'pragma_optimizations': 0,
            'errors': []
        }
        
        with sqlite3.connect(self.database_path) as conn:
            # Apply PRAGMA optimizations
            pragma_settings = [
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=NORMAL", 
                "PRAGMA cache_size=10000",
                "PRAGMA temp_store=MEMORY",
                "PRAGMA mmap_size=268435456"  # 256MB
            ]
            
            for pragma in pragma_settings:
                try:
                    conn.execute(pragma)
                    applied['pragma_optimizations'] += 1
                    self.logger.info(f"Applied: {pragma}")
                except Exception as e:
                    applied['errors'].append(f"Failed to apply {pragma}: {str(e)}")
            
            # Create recommended indexes (top 5 only)
            for recommendation in recommendations[:5]:
                try:
                    conn.execute(recommendation.creation_sql)
                    applied['indexes_created'] += 1
                    self.logger.info(f"Created index: {recommendation.creation_sql}")
                except Exception as e:
                    applied['errors'].append(f"Failed to create index: {str(e)}")
            
            conn.commit()
        
        return applied

    def _calculate_improvements(self, baseline: Dict, optimized: Dict) -> Dict[str, Any]:
        """Calculate performance improvements"""
        improvements = {}
        
        for query_name in baseline.keys():
            if query_name in optimized:
                baseline_time = baseline[query_name]['avg_time']
                optimized_time = optimized[query_name]['avg_time']
                
                improvement_percent = ((baseline_time - optimized_time) / baseline_time) * 100
                speedup_factor = baseline_time / optimized_time if optimized_time > 0 else float('inf')
                
                improvements[query_name] = {
                    'baseline_avg_time': baseline_time,
                    'optimized_avg_time': optimized_time,
                    'improvement_percent': improvement_percent,
                    'speedup_factor': speedup_factor,
                    'time_saved_seconds': baseline_time - optimized_time
                }
        
        # Overall statistics
        if improvements:
            all_improvements = [imp['improvement_percent'] for imp in improvements.values()]
            improvements['overall'] = {
                'avg_improvement_percent': statistics.mean(all_improvements),
                'best_improvement_percent': max(all_improvements),
                'total_queries_improved': sum(1 for imp in all_improvements if imp > 0),
                'total_queries_analyzed': len(all_improvements)
            }
        
        return improvements

    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        # Run optimization
        results = self.optimize_database()
        
        # Get pool stats
        pool_stats = self.connection_pool.get_pool_stats()
        
        # Generate report
        report_lines = [
            "DATABASE PERFORMANCE OPTIMIZATION REPORT",
            "=" * 50,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Database: {self.database_path}",
            "",
            "CONNECTION POOL STATUS:",
            f"  Active Connections: {pool_stats['active_connections']}/{pool_stats['max_connections']}",
            f"  Available Connections: {pool_stats['available_connections']}",
            f"  Total Queries: {pool_stats['connection_stats']['borrowed']}",
            f"  Connection Timeouts: {pool_stats['connection_stats']['timeouts']}",
            "",
            "QUERY ANALYSIS:",
        ]
        
        # Add query analysis
        query_analysis = results['phases']['query_analysis']
        report_lines.extend([
            f"  Total Unique Queries: {query_analysis['total_unique_queries']}",
            f"  Total Executions: {query_analysis['total_executions']}",
            f"  Overall Avg Time: {query_analysis['overall_avg_time']:.4f}s",
            f"  Slow Queries (>100ms): {query_analysis['slow_queries']}",
            f"  Very Slow Queries (>1s): {query_analysis['very_slow_queries']}",
            ""
        ])
        
        # Add improvements
        if 'improvement_analysis' in results['phases']:
            improvements = results['phases']['improvement_analysis']
            if 'overall' in improvements:
                overall = improvements['overall']
                report_lines.extend([
                    "PERFORMANCE IMPROVEMENTS:",
                    f"  Average Improvement: {overall['avg_improvement_percent']:.1f}%",
                    f"  Best Improvement: {overall['best_improvement_percent']:.1f}%",
                    f"  Queries Improved: {overall['total_queries_improved']}/{overall['total_queries_analyzed']}",
                    ""
                ])
        
        # Add index recommendations
        if results['phases']['index_recommendations']:
            report_lines.extend([
                "INDEX RECOMMENDATIONS APPLIED:",
                f"  Indexes Created: {results['phases']['applied_optimizations']['indexes_created']}",
                f"  PRAGMA Optimizations: {results['phases']['applied_optimizations']['pragma_optimizations']}",
                ""
            ])
        
        return "\n".join(report_lines)

def main():
    """Main function to demonstrate database optimization"""
    print("AGENT BETA - Database Performance Optimizer")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = DatabasePerformanceOptimizer(enable_monitoring=True)
    
    try:
        # Start monitoring if available
        if optimizer.monitoring_system:
            optimizer.monitoring_system.start()
        
        # Run optimization
        print("Running comprehensive database optimization...")
        report = optimizer.generate_optimization_report()
        
        print("\n" + report)
        
        # Additional analysis
        print("\nTOP SLOW QUERIES:")
        slow_queries = optimizer.query_analyzer.get_slow_queries(5)
        for i, query in enumerate(slow_queries, 1):
            print(f"{i}. {query.avg_execution_time:.4f}s avg - {query.query_text[:80]}...")
        
        print("\nMOST FREQUENT QUERIES:")
        frequent_queries = optimizer.query_analyzer.get_frequent_queries(5)
        for i, query in enumerate(frequent_queries, 1):
            print(f"{i}. {query.execution_count} executions - {query.query_text[:80]}...")
        
        print("\nDatabase optimization completed successfully!")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if optimizer.monitoring_system:
            optimizer.monitoring_system.stop()
        optimizer.connection_pool.cleanup()

if __name__ == "__main__":
    main()