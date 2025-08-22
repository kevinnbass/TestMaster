"""
Analytics Data Persistence Engine
=================================

Advanced data persistence with historical trending, time-series analysis,
and intelligent data retention policies.

Author: TestMaster Team
"""

import sqlite3
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics
import zlib
import pickle

logger = logging.getLogger(__name__)

class AnalyticsPersistenceEngine:
    """
    Manages persistent storage of analytics data with advanced querying capabilities.
    """
    
    def __init__(self, db_path: str = None, retention_days: int = 90):
        """
        Initialize the persistence engine.
        
        Args:
            db_path: Path to the SQLite database
            retention_days: Number of days to retain data
        """
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'analytics_history.db')
        
        self.db_path = db_path
        self.retention_days = retention_days
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connection pool for thread safety
        self._local = threading.local()
        
        # Compression settings
        self.compression_threshold = 1024  # Compress data larger than 1KB
        self.compression_level = 6
        
        # Initialize database
        self._init_database()
        
        # Start background maintenance
        self.maintenance_active = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        
        logger.info(f"Analytics Persistence Engine initialized: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path, timeout=30)
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=10000")
        return self._local.connection
    
    def _init_database(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        
        # Analytics snapshots table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                snapshot_type TEXT NOT NULL,
                data_compressed BLOB,
                data_size INTEGER,
                is_compressed BOOLEAN DEFAULT FALSE,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes separately
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON analytics_snapshots(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_type ON analytics_snapshots(snapshot_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_created ON analytics_snapshots(created_at)")
        
        # Time series metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS time_series_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_type TEXT,
                source_component TEXT,
                tags TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes separately
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON time_series_metrics(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON time_series_metrics(metric_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_component ON time_series_metrics(source_component)")
        
        # Analytics events table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                severity TEXT,
                component TEXT,
                message TEXT,
                event_data TEXT,
                correlation_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes separately
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON analytics_events(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON analytics_events(event_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON analytics_events(severity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_component ON analytics_events(component)")
        
        # Trend analysis results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trend_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                trend_direction TEXT,
                trend_strength REAL,
                analysis_window_hours INTEGER,
                prediction_data TEXT,
                confidence_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes separately
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trends_timestamp ON trend_analysis(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trends_metric ON trend_analysis(metric_name)")
        
        # Performance baselines table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL UNIQUE,
                baseline_value REAL,
                baseline_std_dev REAL,
                sample_count INTEGER,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL
            )
        """)
        
        conn.commit()
        logger.info("Database schema initialized")
    
    def store_analytics_snapshot(self, snapshot_type: str, analytics_data: Dict[str, Any], 
                                metadata: Dict[str, Any] = None) -> int:
        """
        Store a complete analytics snapshot.
        
        Args:
            snapshot_type: Type of snapshot (e.g., 'comprehensive', 'system_metrics')
            analytics_data: The analytics data to store
            metadata: Additional metadata about the snapshot
        
        Returns:
            The ID of the stored snapshot
        """
        conn = self._get_connection()
        
        # Serialize and potentially compress the data
        data_json = json.dumps(analytics_data)
        data_size = len(data_json.encode('utf-8'))
        
        if data_size > self.compression_threshold:
            # Compress the data
            compressed_data = zlib.compress(data_json.encode('utf-8'), self.compression_level)
            is_compressed = True
            stored_data = compressed_data
        else:
            is_compressed = False
            stored_data = data_json.encode('utf-8')
        
        # Store metadata
        metadata_json = json.dumps(metadata or {})
        
        cursor = conn.execute("""
            INSERT INTO analytics_snapshots 
            (snapshot_type, data_compressed, data_size, is_compressed, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (snapshot_type, stored_data, data_size, is_compressed, metadata_json))
        
        conn.commit()
        
        snapshot_id = cursor.lastrowid
        logger.debug(f"Stored analytics snapshot {snapshot_id} ({snapshot_type}): {data_size} bytes")
        
        return snapshot_id
    
    def store_time_series_metric(self, metric_name: str, value: float, metric_type: str = None,
                                source_component: str = None, tags: Dict[str, str] = None):
        """
        Store a time-series metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric (gauge, counter, histogram, etc.)
            source_component: Component that generated the metric
            tags: Additional tags for the metric
        """
        conn = self._get_connection()
        
        tags_json = json.dumps(tags or {})
        
        conn.execute("""
            INSERT INTO time_series_metrics 
            (metric_name, metric_value, metric_type, source_component, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (metric_name, value, metric_type, source_component, tags_json))
        
        conn.commit()
        
        # Update performance baseline
        self._update_performance_baseline(metric_name, value)
    
    def store_analytics_event(self, event_type: str, message: str, severity: str = 'info',
                             component: str = None, event_data: Dict[str, Any] = None,
                             correlation_id: str = None):
        """
        Store an analytics event.
        
        Args:
            event_type: Type of event
            message: Event message
            severity: Event severity (debug, info, warning, error, critical)
            component: Component that generated the event
            event_data: Additional event data
            correlation_id: Correlation ID for tracking related events
        """
        conn = self._get_connection()
        
        event_data_json = json.dumps(event_data or {})
        
        conn.execute("""
            INSERT INTO analytics_events 
            (event_type, severity, component, message, event_data, correlation_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (event_type, severity, component, message, event_data_json, correlation_id))
        
        conn.commit()
    
    def get_historical_snapshots(self, snapshot_type: str = None, hours: int = 24,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve historical analytics snapshots.
        
        Args:
            snapshot_type: Filter by snapshot type
            hours: Number of hours to look back
            limit: Maximum number of snapshots to return
        
        Returns:
            List of analytics snapshots
        """
        conn = self._get_connection()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if snapshot_type:
            query = """
                SELECT * FROM analytics_snapshots 
                WHERE snapshot_type = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT ?
            """
            params = (snapshot_type, cutoff_time, limit)
        else:
            query = """
                SELECT * FROM analytics_snapshots 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC LIMIT ?
            """
            params = (cutoff_time, limit)
        
        rows = conn.execute(query, params).fetchall()
        
        snapshots = []
        for row in rows:
            # Decompress data if needed
            if row['is_compressed']:
                data_json = zlib.decompress(row['data_compressed']).decode('utf-8')
            else:
                data_json = row['data_compressed'].decode('utf-8')
            
            snapshot = {
                'id': row['id'],
                'timestamp': row['timestamp'],
                'snapshot_type': row['snapshot_type'],
                'data': json.loads(data_json),
                'data_size': row['data_size'],
                'metadata': json.loads(row['metadata'] or '{}'),
                'created_at': row['created_at']
            }
            snapshots.append(snapshot)
        
        return snapshots
    
    def get_time_series_data(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get time series data for a specific metric.
        
        Args:
            metric_name: Name of the metric
            hours: Number of hours to look back
        
        Returns:
            List of metric data points
        """
        conn = self._get_connection()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        rows = conn.execute("""
            SELECT * FROM time_series_metrics 
            WHERE metric_name = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (metric_name, cutoff_time)).fetchall()
        
        return [dict(row) for row in rows]
    
    def calculate_trend_analysis(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Calculate trend analysis for a specific metric.
        
        Args:
            metric_name: Name of the metric
            hours: Analysis window in hours
        
        Returns:
            Trend analysis results
        """
        time_series = self.get_time_series_data(metric_name, hours)
        
        if len(time_series) < 5:
            return {
                'metric_name': metric_name,
                'trend_direction': 'insufficient_data',
                'trend_strength': 0.0,
                'confidence_score': 0.0,
                'analysis_window_hours': hours,
                'sample_count': len(time_series)
            }
        
        # Extract values and timestamps
        values = [point['metric_value'] for point in time_series]
        timestamps = [datetime.fromisoformat(point['timestamp']) for point in time_series]
        
        # Calculate trend using linear regression
        n = len(values)
        x = list(range(n))
        
        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        # Calculate slope (trend)
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Calculate correlation coefficient (trend strength)
        value_std = statistics.stdev(values) if len(values) > 1 else 0
        if value_std == 0:
            correlation = 0
        else:
            correlation = abs(slope * statistics.stdev(x) / value_std)
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Threshold for "stable"
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        # Calculate confidence score based on sample size and consistency
        confidence_score = min(1.0, len(time_series) / 50) * min(1.0, correlation)
        
        # Generate prediction for next few data points
        prediction_data = self._generate_predictions(values, slope, 5)
        
        trend_analysis = {
            'metric_name': metric_name,
            'trend_direction': trend_direction,
            'trend_strength': correlation,
            'slope': slope,
            'confidence_score': confidence_score,
            'analysis_window_hours': hours,
            'sample_count': len(time_series),
            'value_range': {'min': min(values), 'max': max(values), 'mean': y_mean},
            'predictions': prediction_data,
            'calculated_at': datetime.now().isoformat()
        }
        
        # Store the trend analysis
        self._store_trend_analysis(trend_analysis)
        
        return trend_analysis
    
    def get_performance_baseline(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get performance baseline for a metric."""
        conn = self._get_connection()
        
        row = conn.execute("""
            SELECT * FROM performance_baselines WHERE metric_name = ?
        """, (metric_name,)).fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_anomaly_candidates(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Identify potential anomalies based on deviation from baselines.
        
        Args:
            hours: Number of hours to analyze
        
        Returns:
            List of potential anomalies
        """
        conn = self._get_connection()
        
        # Get all metrics with baselines
        baselines = conn.execute("SELECT * FROM performance_baselines").fetchall()
        
        anomalies = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for baseline in baselines:
            metric_name = baseline['metric_name']
            baseline_value = baseline['baseline_value']
            baseline_std = baseline['baseline_std_dev']
            
            if baseline_std == 0:
                continue  # Skip metrics with no variation
            
            # Get recent values
            recent_values = conn.execute("""
                SELECT * FROM time_series_metrics 
                WHERE metric_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (metric_name, cutoff_time)).fetchall()
            
            for value_row in recent_values:
                value = value_row['metric_value']
                z_score = abs(value - baseline_value) / baseline_std
                
                # Consider z-score > 2.5 as potential anomaly
                if z_score > 2.5:
                    anomalies.append({
                        'metric_name': metric_name,
                        'timestamp': value_row['timestamp'],
                        'value': value,
                        'baseline_value': baseline_value,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3.5 else 'medium',
                        'deviation_percent': ((value - baseline_value) / baseline_value) * 100
                    })
        
        # Sort by z-score (most anomalous first)
        anomalies.sort(key=lambda x: x['z_score'], reverse=True)
        
        return anomalies[:50]  # Return top 50 anomalies
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence engine statistics."""
        conn = self._get_connection()
        
        # Count records in each table
        snapshot_count = conn.execute("SELECT COUNT(*) FROM analytics_snapshots").fetchone()[0]
        metrics_count = conn.execute("SELECT COUNT(*) FROM time_series_metrics").fetchone()[0]
        events_count = conn.execute("SELECT COUNT(*) FROM analytics_events").fetchone()[0]
        trends_count = conn.execute("SELECT COUNT(*) FROM trend_analysis").fetchone()[0]
        baselines_count = conn.execute("SELECT COUNT(*) FROM performance_baselines").fetchone()[0]
        
        # Database size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        # Data retention info
        oldest_snapshot = conn.execute("""
            SELECT MIN(timestamp) FROM analytics_snapshots
        """).fetchone()[0]
        
        oldest_metric = conn.execute("""
            SELECT MIN(timestamp) FROM time_series_metrics
        """).fetchone()[0]
        
        return {
            'database_path': self.db_path,
            'database_size_mb': db_size / (1024 * 1024),
            'retention_days': self.retention_days,
            'record_counts': {
                'analytics_snapshots': snapshot_count,
                'time_series_metrics': metrics_count,
                'analytics_events': events_count,
                'trend_analysis': trends_count,
                'performance_baselines': baselines_count
            },
            'oldest_data': {
                'snapshots': oldest_snapshot,
                'metrics': oldest_metric
            },
            'compression_enabled': True,
            'compression_threshold_bytes': self.compression_threshold
        }
    
    def _update_performance_baseline(self, metric_name: str, value: float):
        """Update the performance baseline for a metric."""
        conn = self._get_connection()
        
        # Get existing baseline
        existing = conn.execute("""
            SELECT * FROM performance_baselines WHERE metric_name = ?
        """, (metric_name,)).fetchone()
        
        if existing:
            # Update existing baseline using exponential smoothing
            current_value = existing['baseline_value']
            current_std = existing['baseline_std_dev']
            sample_count = existing['sample_count']
            
            # Exponential smoothing factor (adjust based on sample count)
            alpha = min(0.1, 2.0 / (sample_count + 1))
            
            new_value = alpha * value + (1 - alpha) * current_value
            
            # Update standard deviation
            variance = current_std ** 2
            new_variance = alpha * (value - current_value) ** 2 + (1 - alpha) * variance
            new_std = new_variance ** 0.5
            
            # Calculate confidence intervals (assuming normal distribution)
            ci_lower = new_value - 1.96 * new_std
            ci_upper = new_value + 1.96 * new_std
            
            conn.execute("""
                UPDATE performance_baselines 
                SET baseline_value = ?, baseline_std_dev = ?, sample_count = ?,
                    last_updated = CURRENT_TIMESTAMP,
                    confidence_interval_lower = ?, confidence_interval_upper = ?
                WHERE metric_name = ?
            """, (new_value, new_std, sample_count + 1, ci_lower, ci_upper, metric_name))
        else:
            # Create new baseline
            conn.execute("""
                INSERT INTO performance_baselines 
                (metric_name, baseline_value, baseline_std_dev, sample_count,
                 confidence_interval_lower, confidence_interval_upper)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (metric_name, value, 0.0, 1, value, value))
        
        conn.commit()
    
    def _store_trend_analysis(self, trend_analysis: Dict[str, Any]):
        """Store trend analysis results."""
        conn = self._get_connection()
        
        prediction_json = json.dumps(trend_analysis.get('predictions', []))
        
        conn.execute("""
            INSERT INTO trend_analysis 
            (metric_name, trend_direction, trend_strength, analysis_window_hours,
             prediction_data, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            trend_analysis['metric_name'],
            trend_analysis['trend_direction'],
            trend_analysis['trend_strength'],
            trend_analysis['analysis_window_hours'],
            prediction_json,
            trend_analysis['confidence_score']
        ))
        
        conn.commit()
    
    def _generate_predictions(self, values: List[float], slope: float, num_predictions: int) -> List[Dict[str, Any]]:
        """Generate predictions based on trend analysis."""
        if not values:
            return []
        
        last_value = values[-1]
        predictions = []
        
        for i in range(1, num_predictions + 1):
            predicted_value = last_value + (slope * i)
            prediction_time = datetime.now() + timedelta(minutes=i * 5)  # 5-minute intervals
            
            predictions.append({
                'timestamp': prediction_time.isoformat(),
                'predicted_value': predicted_value,
                'confidence': max(0.1, 1.0 - (i * 0.15))  # Decreasing confidence over time
            })
        
        return predictions
    
    def _maintenance_loop(self):
        """Background maintenance loop for data cleanup and optimization."""
        while self.maintenance_active:
            try:
                # Run maintenance every hour
                time.sleep(3600)
                
                if not self.maintenance_active:
                    break
                
                logger.info("Running analytics persistence maintenance")
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Optimize database
                self._optimize_database()
                
            except Exception as e:
                logger.error(f"Error in persistence maintenance: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        conn = self._get_connection()
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Clean up old snapshots
        deleted_snapshots = conn.execute("""
            DELETE FROM analytics_snapshots WHERE timestamp < ?
        """, (cutoff_date,)).rowcount
        
        # Clean up old time series data
        deleted_metrics = conn.execute("""
            DELETE FROM time_series_metrics WHERE timestamp < ?
        """, (cutoff_date,)).rowcount
        
        # Clean up old events
        deleted_events = conn.execute("""
            DELETE FROM analytics_events WHERE timestamp < ?
        """, (cutoff_date,)).rowcount
        
        # Clean up old trend analysis
        deleted_trends = conn.execute("""
            DELETE FROM trend_analysis WHERE timestamp < ?
        """, (cutoff_date,)).rowcount
        
        conn.commit()
        
        if deleted_snapshots + deleted_metrics + deleted_events + deleted_trends > 0:
            logger.info(f"Cleaned up old data: {deleted_snapshots} snapshots, "
                       f"{deleted_metrics} metrics, {deleted_events} events, "
                       f"{deleted_trends} trends")
    
    def _optimize_database(self):
        """Optimize database performance."""
        conn = self._get_connection()
        
        # Analyze tables for query optimization
        conn.execute("ANALYZE")
        
        # Vacuum to reclaim space (only if significant space can be reclaimed)
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        conn.execute("PRAGMA incremental_vacuum")
        
        conn.commit()
        
        logger.debug("Database optimization completed")
    
    def shutdown(self):
        """Shutdown the persistence engine."""
        self.maintenance_active = False
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
        logger.info("Analytics Persistence Engine shutdown")