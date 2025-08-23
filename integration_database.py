"""
Integration Database Handler
Extracted from advanced_system_integration.py for modularization

Handles all database operations for the integration system.
"""

import sqlite3
import json
import logging
from datetime import datetime
from dataclasses import asdict

from integration_models import ServiceHealth, IntegrationEndpoint, SystemMetrics

logger = logging.getLogger(__name__)


class IntegrationDatabase:
    """Handles all database operations for integration system"""
    
    def __init__(self, db_path: str = "system_integration.db"):
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize the integration database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Service registry table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS service_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    url TEXT NOT NULL,
                    config TEXT NOT NULL,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Health monitoring table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_monitoring (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time_ms REAL NOT NULL,
                    error_message TEXT,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (service_name) REFERENCES service_registry (name)
                )
            ''')
            
            # Integration events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS integration_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    service_name TEXT,
                    details TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_services INTEGER NOT NULL,
                    healthy_services INTEGER NOT NULL,
                    degraded_services INTEGER NOT NULL,
                    unhealthy_services INTEGER NOT NULL,
                    average_response_time REAL NOT NULL,
                    uptime_percentage REAL NOT NULL,
                    integration_score REAL NOT NULL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Integration database initialized")
    
    def register_service(self, service: IntegrationEndpoint):
        """Register a service in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO service_registry 
                    (name, type, url, config, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    service.name,
                    service.type.value,
                    service.url,
                    json.dumps(asdict(service)),
                    datetime.now()
                ))
                conn.commit()
            
            logger.info(f"Service '{service.name}' registered in database")
        except Exception as e:
            logger.error(f"Error registering service '{service.name}' in database: {e}")
    
    def unregister_service(self, service_name: str):
        """Unregister a service from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM service_registry WHERE name = ?', (service_name,))
                conn.commit()
            
            logger.info(f"Service '{service_name}' unregistered from database")
        except Exception as e:
            logger.error(f"Error unregistering service '{service_name}' from database: {e}")
    
    def store_health_result(self, health: ServiceHealth):
        """Store health check result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO health_monitoring 
                    (service_name, status, response_time_ms, error_message)
                    VALUES (?, ?, ?, ?)
                ''', (
                    health.service_name,
                    health.status.value,
                    health.response_time_ms,
                    health.error_message
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing health result: {e}")
    
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (total_services, healthy_services, degraded_services, unhealthy_services,
                     average_response_time, uptime_percentage, integration_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.total_services,
                    metrics.healthy_services,
                    metrics.degraded_services,
                    metrics.unhealthy_services,
                    metrics.average_response_time,
                    metrics.uptime_percentage,
                    metrics.integration_score
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
    
    def get_service_count(self) -> int:
        """Get count of registered services"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM service_registry")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting service count: {e}")
            return 0