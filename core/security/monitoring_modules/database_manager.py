#!/usr/bin/env python3
"""
Database Management Module
Agent D Hour 5 - Modularized Database Operations

Handles all database operations for security monitoring system
following STEELCLAD Anti-Regression Modularization Protocol.
"""

import sqlite3
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
import logging

from .security_events import SecurityEvent, SecurityMetrics, ThreatLevel

class DatabaseManager:
    """Centralized database management for security monitoring"""
    
    def __init__(self, db_path: str = None):
        """Initialize database manager"""
        if db_path is None:
            db_path = Path(__file__).parent.parent / "monitoring_data.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema with enhanced security tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Security events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    response_action TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TEXT,
                    correlation_id TEXT,
                    severity_score INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    active_processes INTEGER,
                    network_connections INTEGER,
                    threat_detection_rate REAL,
                    response_time_avg REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # File integrity table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_integrity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    permissions TEXT NOT NULL,
                    scan_timestamp TEXT NOT NULL,
                    threat_detected BOOLEAN DEFAULT FALSE,
                    threat_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_path, scan_timestamp)
                )
            ''')
            
            # Threat patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT UNIQUE NOT NULL,
                    pattern_regex TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    description TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    detection_count INTEGER DEFAULT 0,
                    false_positive_count INTEGER DEFAULT 0,
                    last_detection TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Response actions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS response_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_status TEXT NOT NULL,
                    execution_time TEXT NOT NULL,
                    duration_ms INTEGER,
                    success BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES security_events (event_id)
                )
            ''')
            
            # Security dashboard metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dashboard_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON security_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_threat_level ON security_events(threat_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_resolved ON security_events(resolved)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_integrity_path ON file_integrity(file_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_response_actions_event_id ON response_actions(event_id)')
            
            conn.commit()
            
        self.logger.info("Database initialized with enhanced security schema")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def store_security_event(self, event: SecurityEvent) -> bool:
        """Store security event in database"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO security_events (
                        event_id, timestamp, event_type, threat_level, source_file,
                        description, evidence, response_action, resolved, resolution_time,
                        correlation_id, severity_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id, event.timestamp, event.event_type, 
                    event.threat_level.value, event.source_file, event.description,
                    json.dumps(event.evidence), event.response_action.value,
                    event.resolved, event.resolution_time, event.correlation_id,
                    event.severity_score
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store security event: {e}")
            return False
    
    def get_security_events(self, limit: int = 100, 
                           threat_level: ThreatLevel = None,
                           resolved: bool = None,
                           start_time: str = None,
                           end_time: str = None) -> List[SecurityEvent]:
        """Retrieve security events with filtering"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM security_events WHERE 1=1"
                params = []
                
                if threat_level:
                    query += " AND threat_level = ?"
                    params.append(threat_level.value)
                
                if resolved is not None:
                    query += " AND resolved = ?"
                    params.append(resolved)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = SecurityEvent(
                        timestamp=row['timestamp'],
                        event_type=row['event_type'],
                        threat_level=ThreatLevel(row['threat_level']),
                        source_file=row['source_file'],
                        description=row['description'],
                        evidence=json.loads(row['evidence']),
                        response_action=ResponseAction(row['response_action']),
                        resolved=bool(row['resolved']),
                        resolution_time=row['resolution_time'],
                        event_id=row['event_id'],
                        correlation_id=row['correlation_id'],
                        severity_score=row['severity_score']
                    )
                    events.append(event)
                
                return events
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve security events: {e}")
            return []
    
    def update_event_resolution(self, event_id: str, resolved: bool = True,
                              resolution_time: str = None) -> bool:
        """Update event resolution status"""
        try:
            if resolution_time is None:
                resolution_time = datetime.datetime.now().isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE security_events 
                    SET resolved = ?, resolution_time = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE event_id = ?
                ''', (resolved, resolution_time, event_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Failed to update event resolution: {e}")
            return False
    
    def store_system_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Store system performance metrics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_metrics (
                        timestamp, cpu_usage, memory_usage, disk_usage,
                        active_processes, network_connections, threat_detection_rate,
                        response_time_avg
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.get('timestamp', datetime.datetime.now().isoformat()),
                    metrics.get('cpu_usage', 0.0),
                    metrics.get('memory_usage', 0.0),
                    metrics.get('disk_usage', 0.0),
                    metrics.get('active_processes', 0),
                    metrics.get('network_connections', 0),
                    metrics.get('threat_detection_rate', 0.0),
                    metrics.get('response_time_avg', 0.0)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store system metrics: {e}")
            return False
    
    def store_response_action(self, event_id: str, action_type: str, 
                            action_status: str, success: bool,
                            duration_ms: int = None, error_message: str = None,
                            details: Dict[str, Any] = None) -> bool:
        """Store response action execution details"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO response_actions (
                        event_id, action_type, action_status, execution_time,
                        duration_ms, success, error_message, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_id, action_type, action_status,
                    datetime.datetime.now().isoformat(),
                    duration_ms, success, error_message,
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store response action: {e}")
            return False
    
    def get_security_metrics(self, hours: int = 24) -> SecurityMetrics:
        """Calculate security metrics for dashboard"""
        try:
            cutoff_time = (datetime.datetime.now() - 
                          datetime.timedelta(hours=hours)).isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get event counts by threat level
                cursor.execute('''
                    SELECT threat_level, COUNT(*) as count, 
                           SUM(CASE WHEN resolved THEN 1 ELSE 0 END) as resolved_count
                    FROM security_events 
                    WHERE timestamp >= ?
                    GROUP BY threat_level
                ''', (cutoff_time,))
                
                threat_counts = {}
                total_events = 0
                total_resolved = 0
                
                for row in cursor.fetchall():
                    level = row['count']
                    count = row['count']
                    resolved = row['resolved_count']
                    
                    threat_counts[level] = {'total': count, 'resolved': resolved}
                    total_events += count
                    total_resolved += resolved
                
                # Get threat types
                cursor.execute('''
                    SELECT event_type, COUNT(*) as count
                    FROM security_events 
                    WHERE timestamp >= ?
                    GROUP BY event_type
                ''', (cutoff_time,))
                
                threat_types = {row['event_type']: row['count'] 
                              for row in cursor.fetchall()}
                
                # Get response effectiveness
                cursor.execute('''
                    SELECT ra.action_type, 
                           AVG(CASE WHEN ra.success THEN 1.0 ELSE 0.0 END) as success_rate,
                           AVG(ra.duration_ms) as avg_duration
                    FROM response_actions ra
                    INNER JOIN security_events se ON ra.event_id = se.event_id
                    WHERE se.timestamp >= ?
                    GROUP BY ra.action_type
                ''', (cutoff_time,))
                
                response_effectiveness = {}
                for row in cursor.fetchall():
                    response_effectiveness[row['action_type']] = {
                        'success_rate': row['success_rate'] or 0.0,
                        'avg_duration_ms': row['avg_duration'] or 0.0
                    }
                
                # Calculate average resolution time
                cursor.execute('''
                    SELECT AVG(
                        CASE WHEN resolved AND resolution_time IS NOT NULL 
                        THEN (julianday(resolution_time) - julianday(timestamp)) * 86400
                        ELSE NULL END
                    ) as avg_resolution_time
                    FROM security_events 
                    WHERE timestamp >= ?
                ''', (cutoff_time,))
                
                avg_resolution_time = cursor.fetchone()['avg_resolution_time'] or 0.0
                
                # Build SecurityMetrics object
                metrics = SecurityMetrics(
                    total_events=total_events,
                    critical_events=threat_counts.get('CRITICAL', {}).get('total', 0),
                    high_events=threat_counts.get('HIGH', {}).get('total', 0),
                    medium_events=threat_counts.get('MEDIUM', {}).get('total', 0),
                    low_events=threat_counts.get('LOW', {}).get('total', 0),
                    resolved_events=total_resolved,
                    unresolved_events=total_events - total_resolved,
                    average_resolution_time=avg_resolution_time,
                    threat_types=threat_types,
                    response_effectiveness=response_effectiveness
                )
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to calculate security metrics: {e}")
            return SecurityMetrics()
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """Clean up old data to maintain database performance"""
        try:
            cutoff_date = (datetime.datetime.now() - 
                          datetime.timedelta(days=days_to_keep)).isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up old resolved events
                cursor.execute('''
                    DELETE FROM security_events 
                    WHERE timestamp < ? AND resolved = TRUE
                ''', (cutoff_date,))
                events_cleaned = cursor.rowcount
                
                # Clean up old system metrics
                cursor.execute('''
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                metrics_cleaned = cursor.rowcount
                
                # Clean up old response actions for deleted events
                cursor.execute('''
                    DELETE FROM response_actions 
                    WHERE event_id NOT IN (SELECT event_id FROM security_events)
                ''')
                actions_cleaned = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"Cleaned up {events_cleaned} events, "
                               f"{metrics_cleaned} metrics, "
                               f"{actions_cleaned} response actions")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health metrics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Table row counts
                for table in ['security_events', 'system_metrics', 'file_integrity', 
                             'threat_patterns', 'response_actions', 'dashboard_metrics']:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()['count']
                
                # Database file size
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
                # Most recent event timestamp
                cursor.execute("SELECT MAX(timestamp) as latest FROM security_events")
                stats['latest_event'] = cursor.fetchone()['latest']
                
                # Unresolved events count
                cursor.execute("SELECT COUNT(*) as count FROM security_events WHERE resolved = FALSE")
                stats['unresolved_events'] = cursor.fetchone()['count']
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}