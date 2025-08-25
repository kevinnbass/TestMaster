#!/usr/bin/env python3
"""
Multi-Database Integration System
Agent B Hours 110-120: Enterprise Integration & Advanced Analytics

Advanced integration system supporting multiple database types and environments.
"""

import json
import sqlite3
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import hashlib

@dataclass
class DatabaseConnection:
    """Database connection configuration"""
    name: str
    type: str  # 'sqlite', 'mysql', 'postgresql', 'mongodb'
    connection_string: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    enabled: bool = True
    last_connected: Optional[datetime] = None
    connection_status: str = "unknown"  # 'connected', 'disconnected', 'error'
    metadata: Dict[str, Any] = None

@dataclass
class DatabaseMetrics:
    """Database metrics data structure"""
    database_name: str
    timestamp: datetime
    connection_status: str
    size_mb: float
    table_count: int
    record_count: int
    index_count: int
    active_connections: int
    query_performance_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_io_rate: float
    replication_lag_ms: float = 0
    custom_metrics: Dict[str, float] = None

class DatabaseAdapter(ABC):
    """Abstract base class for database adapters"""
    
    @abstractmethod
    def connect(self, connection_config: DatabaseConnection) -> bool:
        """Establish database connection"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close database connection"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> DatabaseMetrics:
        """Get database metrics"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        pass
    
    @abstractmethod
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test database connection"""
        pass

class SQLiteAdapter(DatabaseAdapter):
    """SQLite database adapter"""
    
    def __init__(self):
        self.connection = None
        self.config = None
        self.last_metrics = None
    
    def connect(self, connection_config: DatabaseConnection) -> bool:
        """Connect to SQLite database"""
        try:
            db_path = connection_config.connection_string
            if not Path(db_path).exists():
                print(f"[WARNING] SQLite database not found: {db_path}")
                return False
            
            self.connection = sqlite3.connect(db_path)
            self.config = connection_config
            self.config.connection_status = "connected"
            self.config.last_connected = datetime.now()
            
            print(f"[OK] Connected to SQLite database: {connection_config.name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to connect to SQLite {connection_config.name}: {e}")
            self.config.connection_status = "error"
            return False
    
    def disconnect(self):
        """Disconnect from SQLite database"""
        if self.connection:
            self.connection.close()
            self.connection = None
            if self.config:
                self.config.connection_status = "disconnected"
    
    def get_metrics(self) -> DatabaseMetrics:
        """Get SQLite database metrics"""
        if not self.connection or not self.config:
            return None
        
        try:
            cursor = self.connection.cursor()
            
            # Get database size
            db_path = self.config.connection_string
            size_mb = Path(db_path).stat().st_size / (1024 * 1024)
            
            # Get table count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            # Get index count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
            index_count = cursor.fetchone()[0]
            
            # Get approximate record count
            record_count = 0
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                    record_count += cursor.fetchone()[0]
                except:
                    pass  # Skip tables that can't be counted
            
            # Simulate performance metrics (SQLite doesn't provide direct access)
            query_start = time.time()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master")
            cursor.fetchone()
            query_performance_ms = (time.time() - query_start) * 1000
            
            metrics = DatabaseMetrics(
                database_name=self.config.name,
                timestamp=datetime.now(),
                connection_status="connected",
                size_mb=size_mb,
                table_count=table_count,
                record_count=record_count,
                index_count=index_count,
                active_connections=1,  # SQLite is single-connection
                query_performance_ms=query_performance_ms,
                cpu_usage_percent=0,  # Not available for SQLite
                memory_usage_mb=0,    # Not available for SQLite
                disk_io_rate=0,       # Not available for SQLite
                custom_metrics={
                    'database_file_size_bytes': Path(db_path).stat().st_size,
                    'page_size': self._get_page_size(),
                    'cache_size': self._get_cache_size()
                }
            )
            
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            print(f"[ERROR] Failed to get SQLite metrics for {self.config.name}: {e}")
            return None
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query on SQLite database"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                row_dict = {columns[i]: row[i] for i in range(len(columns))}
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Query execution failed on {self.config.name}: {e}")
            return []
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get SQLite schema information"""
        if not self.connection:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Get tables
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            # Get indexes
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index'")
            indexes = cursor.fetchall()
            
            # Get views
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='view'")
            views = cursor.fetchall()
            
            schema_info = {
                'database_type': 'sqlite',
                'database_name': self.config.name,
                'tables': [{'name': t[0], 'sql': t[1]} for t in tables],
                'indexes': [{'name': i[0], 'sql': i[1]} for i in indexes],
                'views': [{'name': v[0], 'sql': v[1]} for v in views],
                'table_count': len(tables),
                'index_count': len(indexes),
                'view_count': len(views)
            }
            
            # Get table details
            table_details = {}
            for table_name, _ in tables:
                try:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    
                    table_details[table_name] = {
                        'columns': [{'name': c[1], 'type': c[2], 'nullable': not c[3], 'primary_key': bool(c[5])} for c in columns],
                        'row_count': row_count,
                        'column_count': len(columns)
                    }
                except:
                    pass  # Skip problematic tables
            
            schema_info['table_details'] = table_details
            return schema_info
            
        except Exception as e:
            print(f"[ERROR] Failed to get schema info for {self.config.name}: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """Test SQLite connection"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except:
            return False
    
    def _get_page_size(self) -> int:
        """Get SQLite page size"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("PRAGMA page_size")
            return cursor.fetchone()[0]
        except:
            return 4096  # Default
    
    def _get_cache_size(self) -> int:
        """Get SQLite cache size"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("PRAGMA cache_size")
            return cursor.fetchone()[0]
        except:
            return 2000  # Default

class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter (placeholder)"""
    
    def __init__(self):
        self.connection = None
        self.config = None
    
    def connect(self, connection_config: DatabaseConnection) -> bool:
        """Connect to PostgreSQL database"""
        # This would require psycopg2 or similar library
        print(f"[INFO] PostgreSQL adapter not implemented - would connect to {connection_config.name}")
        return False
    
    def disconnect(self):
        """Disconnect from PostgreSQL database"""
        pass
    
    def get_metrics(self) -> DatabaseMetrics:
        """Get PostgreSQL database metrics"""
        return None
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query on PostgreSQL database"""
        return []
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get PostgreSQL schema information"""
        return {}
    
    def test_connection(self) -> bool:
        """Test PostgreSQL connection"""
        return False

class MySQLAdapter(DatabaseAdapter):
    """MySQL database adapter (placeholder)"""
    
    def __init__(self):
        self.connection = None
        self.config = None
    
    def connect(self, connection_config: DatabaseConnection) -> bool:
        """Connect to MySQL database"""
        # This would require mysql-connector-python or similar library
        print(f"[INFO] MySQL adapter not implemented - would connect to {connection_config.name}")
        return False
    
    def disconnect(self):
        """Disconnect from MySQL database"""
        pass
    
    def get_metrics(self) -> DatabaseMetrics:
        """Get MySQL database metrics"""
        return None
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query on MySQL database"""
        return []
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get MySQL schema information"""
        return {}
    
    def test_connection(self) -> bool:
        """Test MySQL connection"""
        return False

class MultiDatabaseIntegrationSystem:
    """Advanced multi-database integration system"""
    
    def __init__(self, config_file: str = "multi_db_config.json"):
        self.config_file = Path(config_file)
        self.database_connections = {}
        self.database_adapters = {}
        self.metrics_history = {}
        self.monitoring_active = False
        
        # Initialize adapter registry
        self.adapter_registry = {
            'sqlite': SQLiteAdapter,
            'postgresql': PostgreSQLAdapter,
            'mysql': MySQLAdapter
        }
        
        # Load configuration
        self.load_configuration()
        
        # Performance monitoring
        self.performance_stats = {
            'total_queries_executed': 0,
            'total_connections_made': 0,
            'failed_connections': 0,
            'avg_query_time_ms': 0,
            'databases_online': 0
        }
        
        print("[OK] Multi-Database Integration System initialized")
        print(f"[OK] Supporting database types: {list(self.adapter_registry.keys())}")
    
    def load_configuration(self):
        """Load multi-database configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                for db_config in config_data.get('databases', []):
                    # Convert to DatabaseConnection object
                    if 'last_connected' in db_config and db_config['last_connected']:
                        db_config['last_connected'] = datetime.fromisoformat(db_config['last_connected'])
                    
                    db_connection = DatabaseConnection(**db_config)
                    self.database_connections[db_connection.name] = db_connection
                
                print(f"[OK] Loaded {len(self.database_connections)} database configurations")
                
            except Exception as e:
                print(f"[WARNING] Failed to load multi-database config: {e}")
    
    def save_configuration(self):
        """Save multi-database configuration"""
        try:
            config_data = {
                'databases': []
            }
            
            for db_connection in self.database_connections.values():
                db_dict = asdict(db_connection)
                if db_dict['last_connected']:
                    db_dict['last_connected'] = db_dict['last_connected'].isoformat()
                config_data['databases'].append(db_dict)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            print(f"[ERROR] Failed to save multi-database config: {e}")
    
    def add_database(self, name: str, db_type: str, connection_string: str, **kwargs) -> bool:
        """Add a new database to the system"""
        if name in self.database_connections:
            print(f"[WARNING] Database {name} already exists")
            return False
        
        if db_type not in self.adapter_registry:
            print(f"[ERROR] Unsupported database type: {db_type}")
            return False
        
        # Create database connection configuration
        db_connection = DatabaseConnection(
            name=name,
            type=db_type,
            connection_string=connection_string,
            **kwargs
        )
        
        self.database_connections[name] = db_connection
        
        # Test connection
        if self.connect_database(name):
            self.save_configuration()
            print(f"[OK] Added database: {name} ({db_type})")
            return True
        else:
            del self.database_connections[name]
            print(f"[ERROR] Failed to add database: {name}")
            return False
    
    def connect_database(self, database_name: str) -> bool:
        """Connect to a specific database"""
        if database_name not in self.database_connections:
            print(f"[ERROR] Database {database_name} not configured")
            return False
        
        db_config = self.database_connections[database_name]
        
        if not db_config.enabled:
            print(f"[WARNING] Database {database_name} is disabled")
            return False
        
        # Create adapter if not exists
        if database_name not in self.database_adapters:
            adapter_class = self.adapter_registry[db_config.type]
            self.database_adapters[database_name] = adapter_class()
        
        adapter = self.database_adapters[database_name]
        
        # Attempt connection
        if adapter.connect(db_config):
            self.performance_stats['total_connections_made'] += 1
            self.performance_stats['databases_online'] = len([a for a in self.database_adapters.values() if a.test_connection()])
            return True
        else:
            self.performance_stats['failed_connections'] += 1
            return False
    
    def connect_all_databases(self):
        """Connect to all configured databases"""
        print("[OK] Connecting to all configured databases...")
        
        connected_count = 0
        for db_name in self.database_connections:
            if self.connect_database(db_name):
                connected_count += 1
        
        print(f"[OK] Connected to {connected_count}/{len(self.database_connections)} databases")
        return connected_count
    
    def collect_all_metrics(self) -> Dict[str, DatabaseMetrics]:
        """Collect metrics from all connected databases"""
        all_metrics = {}
        
        for db_name, adapter in self.database_adapters.items():
            if adapter.test_connection():
                metrics = adapter.get_metrics()
                if metrics:
                    all_metrics[db_name] = metrics
                    
                    # Store in history
                    if db_name not in self.metrics_history:
                        self.metrics_history[db_name] = []
                    
                    self.metrics_history[db_name].append(metrics)
                    
                    # Keep only recent history (last 1000 points)
                    if len(self.metrics_history[db_name]) > 1000:
                        self.metrics_history[db_name] = self.metrics_history[db_name][-1000:]
        
        return all_metrics
    
    def execute_query_on_database(self, database_name: str, query: str) -> List[Dict[str, Any]]:
        """Execute query on specific database"""
        if database_name not in self.database_adapters:
            print(f"[ERROR] Database {database_name} not connected")
            return []
        
        adapter = self.database_adapters[database_name]
        
        if not adapter.test_connection():
            print(f"[ERROR] Database {database_name} connection lost")
            return []
        
        start_time = time.time()
        results = adapter.execute_query(query)
        execution_time = (time.time() - start_time) * 1000
        
        # Update performance stats
        self.performance_stats['total_queries_executed'] += 1
        current_avg = self.performance_stats['avg_query_time_ms']
        query_count = self.performance_stats['total_queries_executed']
        self.performance_stats['avg_query_time_ms'] = (current_avg * (query_count - 1) + execution_time) / query_count
        
        print(f"[OK] Executed query on {database_name} in {execution_time:.2f}ms, returned {len(results)} rows")
        return results
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        # Connection status
        connection_status = {}
        online_databases = []
        
        for db_name, db_config in self.database_connections.items():
            if db_name in self.database_adapters:
                is_connected = self.database_adapters[db_name].test_connection()
                connection_status[db_name] = {
                    'status': 'online' if is_connected else 'offline',
                    'type': db_config.type,
                    'last_connected': db_config.last_connected.isoformat() if db_config.last_connected else None
                }
                if is_connected:
                    online_databases.append(db_name)
            else:
                connection_status[db_name] = {
                    'status': 'not_connected',
                    'type': db_config.type,
                    'last_connected': None
                }
        
        # Collect current metrics
        current_metrics = self.collect_all_metrics()
        
        # Calculate aggregate metrics
        total_size_mb = sum(m.size_mb for m in current_metrics.values())
        total_tables = sum(m.table_count for m in current_metrics.values())
        total_records = sum(m.record_count for m in current_metrics.values())
        avg_query_performance = statistics.mean([m.query_performance_ms for m in current_metrics.values()]) if current_metrics else 0
        
        return {
            'system_status': {
                'total_databases_configured': len(self.database_connections),
                'databases_online': len(online_databases),
                'databases_offline': len(self.database_connections) - len(online_databases),
                'monitoring_active': self.monitoring_active
            },
            'connection_status': connection_status,
            'aggregate_metrics': {
                'total_size_mb': total_size_mb,
                'total_tables': total_tables,
                'total_records': total_records,
                'avg_query_performance_ms': avg_query_performance
            },
            'performance_stats': self.performance_stats,
            'database_metrics': {name: asdict(metrics) for name, metrics in current_metrics.items()}
        }
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring of all databases"""
        if self.monitoring_active:
            print("[WARNING] Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            print("[OK] Started multi-database monitoring")
            while self.monitoring_active:
                try:
                    # Collect metrics from all databases
                    metrics = self.collect_all_metrics()
                    
                    if metrics:
                        print(f"[MONITORING] Collected metrics from {len(metrics)} databases")
                        
                        # Update performance stats
                        self.performance_stats['databases_online'] = len(metrics)
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    print(f"[ERROR] Monitoring loop error: {e}")
                    time.sleep(interval_seconds * 2)  # Wait longer on error
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        print(f"[OK] Multi-database monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        print("[OK] Multi-database monitoring stopped")
    
    def disconnect_all_databases(self):
        """Disconnect from all databases"""
        for adapter in self.database_adapters.values():
            adapter.disconnect()
        
        self.database_adapters.clear()
        print("[OK] Disconnected from all databases")
    
    def get_database_schema(self, database_name: str) -> Dict[str, Any]:
        """Get schema information for specific database"""
        if database_name not in self.database_adapters:
            return {"error": f"Database {database_name} not connected"}
        
        adapter = self.database_adapters[database_name]
        return adapter.get_schema_info()
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive integration report"""
        overview = self.get_system_overview()
        
        report = f"""
MULTI-DATABASE INTEGRATION REPORT
==================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM OVERVIEW:
- Total Databases Configured: {overview['system_status']['total_databases_configured']}
- Databases Online: {overview['system_status']['databases_online']}
- Databases Offline: {overview['system_status']['databases_offline']}
- Monitoring Active: {overview['system_status']['monitoring_active']}

AGGREGATE METRICS:
- Total Database Size: {overview['aggregate_metrics']['total_size_mb']:.2f} MB
- Total Tables: {overview['aggregate_metrics']['total_tables']}
- Total Records: {overview['aggregate_metrics']['total_records']:,}
- Average Query Performance: {overview['aggregate_metrics']['avg_query_performance_ms']:.2f} ms

PERFORMANCE STATISTICS:
- Total Queries Executed: {overview['performance_stats']['total_queries_executed']}
- Total Connections Made: {overview['performance_stats']['total_connections_made']}
- Failed Connections: {overview['performance_stats']['failed_connections']}
- Average Query Time: {overview['performance_stats']['avg_query_time_ms']:.2f} ms

DATABASE STATUS:
"""
        
        for db_name, status in overview['connection_status'].items():
            report += f"- {db_name} ({status['type']}): {status['status'].upper()}\n"
        
        # Add individual database metrics
        if overview['database_metrics']:
            report += "\nDATABASE METRICS:\n"
            for db_name, metrics in overview['database_metrics'].items():
                report += f"\n{db_name.upper()}:\n"
                report += f"  Size: {metrics['size_mb']:.2f} MB\n"
                report += f"  Tables: {metrics['table_count']}\n"
                report += f"  Records: {metrics['record_count']:,}\n"
                report += f"  Query Performance: {metrics['query_performance_ms']:.2f} ms\n"
        
        return report

import statistics

def main():
    """Main function for testing multi-database integration"""
    system = MultiDatabaseIntegrationSystem()
    
    print("[OK] Multi-Database Integration System ready for testing")
    
    # Add test databases (using existing SQLite databases)
    test_databases = [
        {
            'name': 'cache_db',
            'type': 'sqlite',
            'connection_string': './cache/cache.db'
        },
        {
            'name': 'deduplication_db', 
            'type': 'sqlite',
            'connection_string': './deduplication/deduplication.db'
        }
    ]
    
    # Add databases if files exist
    for db_config in test_databases:
        if Path(db_config['connection_string']).exists():
            success = system.add_database(
                name=db_config['name'],
                db_type=db_config['type'],
                connection_string=db_config['connection_string']
            )
            if success:
                print(f"[OK] Successfully added {db_config['name']}")
            else:
                print(f"[WARNING] Failed to add {db_config['name']}")
    
    # Connect to all databases
    connected_count = system.connect_all_databases()
    
    if connected_count > 0:
        # Collect metrics
        print("\n[OK] Collecting metrics from all databases...")
        metrics = system.collect_all_metrics()
        
        if metrics:
            print(f"[OK] Collected metrics from {len(metrics)} databases")
            
            # Test query execution
            for db_name in metrics.keys():
                print(f"\n[TEST] Testing query execution on {db_name}...")
                results = system.execute_query_on_database(db_name, "SELECT name FROM sqlite_master WHERE type='table' LIMIT 3")
                print(f"[OK] Query returned {len(results)} results")
                
                # Get schema info
                schema = system.get_database_schema(db_name)
                if schema:
                    print(f"[OK] Schema info: {schema.get('table_count', 0)} tables, {schema.get('view_count', 0)} views")
        
        # Generate system overview
        overview = system.get_system_overview()
        print(f"\n[OVERVIEW] System managing {overview['system_status']['total_databases_configured']} databases")
        print(f"[OVERVIEW] {overview['system_status']['databases_online']} online, {overview['system_status']['databases_offline']} offline")
        
        # Generate report
        report = system.generate_integration_report()
        print("\n" + "="*60)
        print(report)
        
        # Start monitoring for a brief test
        print("\n[TEST] Starting monitoring for 10 seconds...")
        system.start_monitoring(interval_seconds=5)
        time.sleep(10)
        system.stop_monitoring()
    
    # Cleanup
    system.disconnect_all_databases()
    print("\n[OK] Multi-Database Integration System test completed")

if __name__ == "__main__":
    main()