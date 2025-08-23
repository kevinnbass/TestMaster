#!/usr/bin/env python3
"""
🏗️ MODULE: Query Performance Analyzer - Query Analysis Engine
==================================================================

📋 PURPOSE:
    Analyzes database query performance and provides optimization recommendations.
    Handles query normalization, execution monitoring, and performance tracking.

🎯 CORE FUNCTIONALITY:
    • Query normalization and hashing for consistent analysis
    • Real-time query performance monitoring
    • Execution time tracking and statistical analysis
    • Query optimization suggestions generation

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 08:05:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract query analyzer from database_performance_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for query performance analysis
   └─ Impact: Improved modularity and single responsibility for query analysis

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: sqlite3, psutil, logging, hashlib
🎯 Integration Points: database_optimization_models.py, index_optimizer.py
⚡ Performance Notes: Context manager for efficient query monitoring
🔒 Security Notes: SQL injection protection via parameterized queries

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Self-monitoring via query analysis | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: database_optimization_models for data structures
📤 Provides: Query analysis capabilities for database optimization
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import sqlite3
import logging
import time
import hashlib
import re
import psutil
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
from contextlib import contextmanager

# Import data models
from database_optimization_models import QueryAnalysis


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
                """, (new_count, new_total_time, new_avg_time, 
                      new_min_time, new_max_time, 
                      timestamp.isoformat(), query_hash))
                
            else:
                # Insert new query analysis
                conn.execute("""
                    INSERT INTO query_analysis (
                        query_hash, query_text, execution_count, 
                        total_execution_time, avg_execution_time,
                        min_execution_time, max_execution_time,
                        first_seen, last_seen
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (query_hash, normalized_query, 1, execution_time, 
                      execution_time, execution_time, execution_time,
                      timestamp.isoformat(), timestamp.isoformat()))
            
            # Record individual execution
            conn.execute("""
                INSERT INTO query_executions (
                    query_hash, execution_time, rows_examined, 
                    rows_returned, execution_plan, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (query_hash, execution_time, rows_examined, 
                  rows_returned, execution_plan, timestamp.isoformat()))

    def get_slow_queries(self, limit: int = 10, min_executions: int = 5) -> List[QueryAnalysis]:
        """Get slowest queries based on average execution time"""
        with sqlite3.connect(self.analysis_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM query_analysis 
                WHERE execution_count >= ? 
                ORDER BY avg_execution_time DESC 
                LIMIT ?
            """, (min_executions, limit))
            
            slow_queries = []
            for row in cursor.fetchall():
                slow_queries.append(self._row_to_analysis(row))
            
            return slow_queries
    
    def get_frequent_queries(self, limit: int = 10) -> List[QueryAnalysis]:
        """Get most frequently executed queries"""
        with sqlite3.connect(self.analysis_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM query_analysis 
                ORDER BY execution_count DESC 
                LIMIT ?
            """, (limit,))
            
            frequent_queries = []
            for row in cursor.fetchall():
                frequent_queries.append(self._row_to_analysis(row))
            
            return frequent_queries
    
    def _row_to_analysis(self, row) -> QueryAnalysis:
        """Convert database row to QueryAnalysis object"""
        return QueryAnalysis(
            query_hash=row[1],
            query_text=row[2],
            execution_count=row[3],
            total_execution_time=row[4],
            avg_execution_time=row[5],
            min_execution_time=row[6],
            max_execution_time=row[7],
            rows_examined=row[8],
            rows_returned=row[9],
            index_usage=row[10].split(',') if row[10] else [],
            optimization_suggestions=row[11].split(',') if row[11] else [],
            timestamp=datetime.fromisoformat(row[13]) if row[13] else None
        )
    
    def generate_optimization_suggestions(self, query_analysis: QueryAnalysis) -> List[str]:
        """Generate optimization suggestions for a query"""
        suggestions = []
        
        # Check for slow execution time
        if query_analysis.avg_execution_time > self.very_slow_query_threshold:
            suggestions.append("Query execution time is very slow (>1s) - consider adding indexes")
        elif query_analysis.avg_execution_time > self.slow_query_threshold:
            suggestions.append("Query execution time is slow (>100ms) - review query structure")
        
        # Check for high execution count
        if query_analysis.execution_count > 1000:
            suggestions.append("High-frequency query - consider caching results")
        
        # Check for large variance in execution time
        time_variance = query_analysis.max_execution_time - query_analysis.min_execution_time
        if time_variance > query_analysis.avg_execution_time * 2:
            suggestions.append("Inconsistent execution times - check for locking or resource contention")
        
        # Check for full table scans (basic heuristic)
        if 'SELECT' in query_analysis.query_text.upper() and 'WHERE' not in query_analysis.query_text.upper():
            suggestions.append("Query may be performing full table scan - add WHERE clause if possible")
        
        return suggestions
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get overall query analysis statistics"""
        with sqlite3.connect(self.analysis_db) as conn:
            # Total queries analyzed
            cursor = conn.execute("SELECT COUNT(*) FROM query_analysis")
            total_queries = cursor.fetchone()[0]
            
            # Total executions
            cursor = conn.execute("SELECT SUM(execution_count) FROM query_analysis")
            total_executions = cursor.fetchone()[0] or 0
            
            # Slow queries
            cursor = conn.execute(
                "SELECT COUNT(*) FROM query_analysis WHERE avg_execution_time > ?",
                (self.slow_query_threshold,)
            )
            slow_queries = cursor.fetchone()[0]
            
            # Average execution time
            cursor = conn.execute("SELECT AVG(avg_execution_time) FROM query_analysis")
            avg_exec_time = cursor.fetchone()[0] or 0
            
            return {
                'total_unique_queries': total_queries,
                'total_executions': total_executions,
                'slow_queries_count': slow_queries,
                'average_execution_time_ms': avg_exec_time * 1000,
                'analysis_period_days': 30,  # Could be calculated from data
                'performance_score': max(0, 100 - (slow_queries / max(total_queries, 1)) * 100)
            }