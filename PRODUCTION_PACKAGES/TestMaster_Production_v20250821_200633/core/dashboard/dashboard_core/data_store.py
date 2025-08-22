from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
Data Store for Dashboard Metrics
=================================

Provides persistent storage for metrics and analytics data.
Maintains historical data for trend analysis.

Author: TestMaster Team
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import pickle

logger = logging.getLogger(__name__)

class MetricsDataStore:
    """
    Persistent storage for dashboard metrics using SQLite.
    """
    
    def __init__(self, db_path: str = None, max_history_days: int = 30):
        """
        Initialize the data store.
        
        Args:
            db_path: Path to SQLite database file
            max_history_days: Days of history to retain
        """
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, 'metrics.db')
        
        self.db_path = db_path
        self.max_history_days = max_history_days
        self._lock = threading.Lock()
        
        # In-memory cache for fast access
        self._cache = {
            'performance': deque(maxlen=10000),
            'test_results': deque(maxlen=1000),
            'analytics': deque(maxlen=1000),
            'events': deque(maxlen=5000)
        }
        
        # Initialize optimizer
        try:
            from .analytics_optimizer import AnalyticsOptimizer
            self.optimizer = AnalyticsOptimizer()
            logger.info("Analytics optimizer initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize optimizer: {e}")
            self.optimizer = None
        
        # Initialize backup manager
        try:
            from .analytics_backup import AnalyticsBackupManager
            self.backup_manager = AnalyticsBackupManager()
            # Start auto backup every 6 hours
            self.backup_manager.start_auto_backup(self.db_path)
            logger.info("Analytics backup manager initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize backup manager: {e}")
            self.backup_manager = None
        
        self._init_database()
        logger.info(f"MetricsDataStore initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage REAL,
                    memory_mb REAL,
                    network_kb_s REAL,
                    disk_usage REAL,
                    response_time_ms REAL,
                    codebase TEXT
                )
            ''')
            
            # Test results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_tests INTEGER,
                    passed INTEGER,
                    failed INTEGER,
                    skipped INTEGER,
                    coverage_percent REAL,
                    quality_score REAL,
                    duration_seconds REAL
                )
            ''')
            
            # Analytics snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    snapshot_type TEXT,
                    data TEXT
                )
            ''')
            
            # Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT,
                    severity TEXT,
                    message TEXT,
                    details TEXT
                )
            ''')
            
            # Workflow metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    workflow_name TEXT,
                    status TEXT,
                    duration_seconds REAL,
                    success BOOLEAN
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_timestamp ON test_results(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics_snapshots(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflow_timestamp ON workflow_metrics(timestamp)')
            
            conn.commit()
    
    def store_performance_metrics(self, metrics: Dict[str, Any]):
        """Store performance metrics."""
        with self._lock:
            try:
                # Add to cache
                self._cache['performance'].append(metrics)
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO performance_metrics 
                        (cpu_usage, memory_mb, network_kb_s, disk_usage, response_time_ms, codebase)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.get('cpu_usage', 0),
                        metrics.get('memory_usage_mb', 0),
                        metrics.get('network_kb_s', 0),
                        metrics.get('disk_usage_percent', 0),
                        metrics.get('response_time_ms', 0),
                        metrics.get('codebase', '/testmaster')
                    ))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Failed to store performance metrics: {e}")
    
    def store_test_results(self, results: Dict[str, Any]):
        """Store test execution results."""
        with self._lock:
            try:
                # Add to cache
                self._cache['test_results'].append(results)
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO test_results 
                        (total_tests, passed, failed, skipped, coverage_percent, quality_score, duration_seconds)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        results.get('total_tests', 0),
                        results.get('passed', 0),
                        results.get('failed', 0),
                        results.get('skipped', 0),
                        results.get('coverage_percent', 0),
                        results.get('quality_score', 0),
                        results.get('duration', 0)
                    ))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Failed to store test results: {e}")
    
    def store_analytics_snapshot(self, snapshot_type: str, data: Dict[str, Any]):
        """Store analytics snapshot with optimization."""
        with self._lock:
            try:
                # Optimize and compress data if optimizer available
                if self.optimizer:
                    compressed_data = self.optimizer.compress_analytics_data(data, 'gzip')
                    stored_data = json.dumps(compressed_data)
                    
                    # Cache optimized query for future retrieval
                    cache_params = {'snapshot_type': snapshot_type, 'latest': True}
                    self.optimizer.cache_data(cache_params, data, ttl_seconds=300)
                else:
                    stored_data = json.dumps(data)
                
                # Add to cache
                self._cache['analytics'].append({'type': snapshot_type, 'data': data})
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO analytics_snapshots (snapshot_type, data)
                        VALUES (?, ?)
                    ''', (snapshot_type, stored_data))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Failed to store analytics snapshot: {e}")
    
    def log_event(self, event_type: str, severity: str, message: str, details: Dict = None):
        """Log an event."""
        with self._lock:
            try:
                event = {
                    'type': event_type,
                    'severity': severity,
                    'message': message,
                    'details': details,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add to cache
                self._cache['events'].append(event)
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO events (event_type, severity, message, details)
                        VALUES (?, ?, ?, ?)
                    ''', (event_type, severity, message, json.dumps(details or {})))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Failed to log event: {e}")
    
    def get_performance_history(self, hours: int = 24, codebase: str = None) -> List[Dict[str, Any]]:
        """Get performance metrics history with optimization."""
        # Check optimizer cache first
        if self.optimizer:
            cache_params = {'type': 'performance_history', 'hours': hours, 'codebase': codebase}
            cached_data = self.optimizer.optimize_data_retriSafeCodeExecutor.safe_SafeCodeExecutor.safe_eval(cache_params)
            if cached_data is not None:
                return cached_data
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT timestamp, cpu_usage, memory_mb, network_kb_s, disk_usage, response_time_ms
                    FROM performance_metrics
                    WHERE timestamp > datetime('now', '-{} hours')
                '''.format(hours)
                
                if codebase:
                    query += f" AND codebase = '{codebase}'"
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                result = [
                    {
                        'timestamp': row[0],
                        'cpu_usage': row[1],
                        'memory_mb': row[2],
                        'network_kb_s': row[3],
                        'disk_usage': row[4],
                        'response_time_ms': row[5]
                    }
                    for row in rows
                ]
                
                # Cache the result for future queries
                if self.optimizer:
                    cache_params = {'type': 'performance_history', 'hours': hours, 'codebase': codebase}
                    self.optimizer.cache_data(cache_params, result, ttl_seconds=300)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []
    
    def get_test_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get test results history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, total_tests, passed, failed, skipped, 
                           coverage_percent, quality_score, duration_seconds
                    FROM test_results
                    WHERE timestamp > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                '''.format(days))
                
                rows = cursor.fetchall()
                
                return [
                    {
                        'timestamp': row[0],
                        'total_tests': row[1],
                        'passed': row[2],
                        'failed': row[3],
                        'skipped': row[4],
                        'coverage_percent': row[5],
                        'quality_score': row[6],
                        'duration': row[7]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get test history: {e}")
            return []
    
    def get_recent_events(self, limit: int = 100, severity: str = None) -> List[Dict[str, Any]]:
        """Get recent events."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT timestamp, event_type, severity, message, details
                    FROM events
                '''
                
                if severity:
                    query += f" WHERE severity = '{severity}'"
                
                query += f" ORDER BY timestamp DESC LIMIT {limit}"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                return [
                    {
                        'timestamp': row[0],
                        'type': row[1],
                        'severity': row[2],
                        'message': row[3],
                        'details': json.loads(row[4]) if row[4] else {}
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    def get_analytics_snapshot(self, snapshot_type: str = None) -> Optional[Dict[str, Any]]:
        """Get the latest analytics snapshot."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if snapshot_type:
                    cursor.execute('''
                        SELECT timestamp, data FROM analytics_snapshots
                        WHERE snapshot_type = ?
                        ORDER BY timestamp DESC LIMIT 1
                    ''', (snapshot_type,))
                else:
                    cursor.execute('''
                        SELECT timestamp, snapshot_type, data FROM analytics_snapshots
                        ORDER BY timestamp DESC LIMIT 1
                    ''')
                
                row = cursor.fetchone()
                
                if row:
                    if snapshot_type:
                        return {
                            'timestamp': row[0],
                            'data': json.loads(row[1])
                        }
                    else:
                        return {
                            'timestamp': row[0],
                            'type': row[1],
                            'data': json.loads(row[2])
                        }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get analytics snapshot: {e}")
            return None
    
    def calculate_trends(self, metric_type: str, hours: int = 24) -> Dict[str, Any]:
        """Calculate trends for a specific metric type."""
        if metric_type == 'performance':
            history = self.get_performance_history(hours)
            if len(history) < 2:
                return {'trend': 'stable', 'change_percent': 0}
            
            # Compare first half with second half
            mid = len(history) // 2
            first_half_avg = sum(h.get('cpu_usage', 0) for h in history[:mid]) / mid if mid > 0 else 0
            second_half_avg = sum(h.get('cpu_usage', 0) for h in history[mid:]) / len(history[mid:]) if len(history[mid:]) > 0 else 0
            
            change_percent = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
            
            if change_percent > 10:
                trend = 'increasing'
            elif change_percent < -10:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'change_percent': round(change_percent, 1),
                'current_avg': round(second_half_avg, 1),
                'previous_avg': round(first_half_avg, 1)
            }
        
        elif metric_type == 'tests':
            history = self.get_test_history(7)
            if len(history) < 2:
                return {'trend': 'stable', 'change_percent': 0}
            
            # Compare pass rates
            recent = history[:len(history)//2]
            older = history[len(history)//2:]
            
            recent_pass_rate = sum(h['passed'] / h['total_tests'] * 100 for h in recent if h['total_tests'] > 0) / len(recent) if recent else 0
            older_pass_rate = sum(h['passed'] / h['total_tests'] * 100 for h in older if h['total_tests'] > 0) / len(older) if older else 0
            
            change = recent_pass_rate - older_pass_rate
            
            return {
                'trend': 'improving' if change > 0 else 'declining' if change < 0 else 'stable',
                'change_percent': round(change, 1),
                'current_rate': round(recent_pass_rate, 1),
                'previous_rate': round(older_pass_rate, 1)
            }
        
        return {'trend': 'unknown', 'change_percent': 0}
    
    def cleanup_old_data(self):
        """Remove data older than max_history_days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
                
                tables = ['performance_metrics', 'test_results', 'analytics_snapshots', 'events', 'workflow_metrics']
                
                for table in tables:
                    cursor.execute(f'''
                        DELETE FROM {table}
                        WHERE timestamp < ?
                    ''', (cutoff_date.isoformat(),))
                
                conn.commit()
                
                # Vacuum to reclaim space
                cursor.execute('VACUUM')
                
                logger.info(f"Cleaned up data older than {self.max_history_days} days")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'performance_cached': len(self._cache['performance']),
            'test_results_cached': len(self._cache['test_results']),
            'analytics_cached': len(self._cache['analytics']),
            'events_cached': len(self._cache['events'])
        }