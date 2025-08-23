#!/usr/bin/env python3
"""
🏗️ MODULE: Database Optimization Core - Main Orchestration Engine
==================================================================

📋 PURPOSE:
    Main orchestration layer for database performance optimization system.
    Coordinates all database optimization components and provides unified interface.

🎯 CORE FUNCTIONALITY:
    • Database optimization system initialization and configuration
    • Component orchestration and coordination between analyzers and optimizers
    • Comprehensive database performance optimization workflow
    • Result aggregation and reporting

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 08:35:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract main orchestration from database_performance_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for database optimization orchestration
   └─ Impact: Clean separation of orchestration from individual optimization components

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: All database optimization modules, logging, sqlite3
🎯 Integration Points: All database optimization child modules
⚡ Performance Notes: Orchestration layer with minimal processing overhead
🔒 Security Notes: Configuration validation and safe database operations

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via database optimization validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: All database optimization child modules
📤 Provides: Complete database optimization framework capabilities
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import sqlite3
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Import all child modules
from database_optimization_models import ConnectionPoolConfig, QueryAnalysis, IndexRecommendation
from query_performance_analyzer import QueryPerformanceAnalyzer
from index_optimizer import IndexOptimizer
from connection_pool_manager import ConnectionPoolManager


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
        try:
            from performance_monitoring_infrastructure import PerformanceMonitoringSystem, MonitoringConfig
            if enable_monitoring:
                config = MonitoringConfig(
                    collection_interval=5.0,
                    alert_channels=['console'],
                    enable_prometheus=False,  # Don't conflict with main monitoring
                    enable_alerting=False
                )
                self.monitoring_system = PerformanceMonitoringSystem(config)
                self._setup_database_metrics()
        except ImportError:
            self.monitoring_system = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('DatabasePerformanceOptimizer')
        
        # Optimization history
        self.optimization_history = []
        
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
                    result TEXT NOT NULL,
                    score REAL,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE test_coverage (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    line_count INTEGER,
                    covered_lines INTEGER,
                    coverage_percent REAL,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Insert sample data
            import random
            from datetime import datetime, timedelta
            
            # Sample users
            users_data = [
                ('alice', 'alice@example.com', 'active'),
                ('bob', 'bob@example.com', 'active'),
                ('charlie', 'charlie@example.com', 'inactive'),
                ('diana', 'diana@example.com', 'active'),
                ('eve', 'eve@example.com', 'active')
            ]
            
            for i, (username, email, status) in enumerate(users_data, 1):
                created = (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
                last_login = (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat() if status == 'active' else None
                
                conn.execute("""
                    INSERT INTO users (id, username, email, created_at, last_login, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (i, username, email, created, last_login, status))
            
            # Sample test results
            test_names = ['unit_test', 'integration_test', 'performance_test', 'security_test']
            results = ['pass', 'fail', 'skip']
            
            for i in range(100):
                user_id = random.randint(1, 5)
                test_name = random.choice(test_names)
                result = random.choice(results)
                score = random.uniform(0, 100) if result == 'pass' else 0
                executed = (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat()
                
                conn.execute("""
                    INSERT INTO test_results (user_id, test_name, result, score, executed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, test_name, result, score, executed))
            
            # Sample coverage data
            files = [
                'src/main.py', 'src/utils.py', 'src/models.py', 
                'tests/test_main.py', 'tests/test_utils.py'
            ]
            
            for file_path in files:
                line_count = random.randint(50, 500)
                covered_lines = random.randint(int(line_count * 0.5), line_count)
                coverage_percent = (covered_lines / line_count) * 100
                last_updated = (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat()
                
                conn.execute("""
                    INSERT INTO test_coverage (file_path, line_count, covered_lines, coverage_percent, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (file_path, line_count, covered_lines, coverage_percent, last_updated))
            
            conn.commit()
            self.logger.info(f"Created sample database: {self.database_path}")

    def run_optimization_analysis(self) -> Dict[str, Any]:
        """Run comprehensive database optimization analysis"""
        optimization_id = f"optimization_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting database optimization analysis: {optimization_id}")
        
        analysis_results = {
            'optimization_id': optimization_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query_analysis': {},
            'index_recommendations': [],
            'connection_pool_status': {},
            'summary': {}
        }
        
        try:
            # Generate some sample queries for analysis
            self._generate_sample_queries()
            
            # Analyze query performance
            slow_queries = self.query_analyzer.get_slow_queries()
            frequent_queries = self.query_analyzer.get_frequent_queries()
            query_stats = self.query_analyzer.get_query_statistics()
            
            analysis_results['query_analysis'] = {
                'slow_queries': [q.__dict__ for q in slow_queries],
                'frequent_queries': [q.__dict__ for q in frequent_queries],
                'statistics': query_stats
            }
            
            # Generate index recommendations
            table_patterns = self.index_optimizer.analyze_table_access_patterns(str(self.database_path))
            index_recommendations = self.index_optimizer.generate_index_recommendations(table_patterns)
            
            analysis_results['index_recommendations'] = [
                {
                    'table_name': rec.table_name,
                    'columns': rec.columns,
                    'index_type': rec.index_type,
                    'estimated_benefit': rec.estimated_benefit,
                    'impact_score': rec.impact_score,
                    'creation_sql': rec.creation_sql,
                    'rationale': rec.rationale
                }
                for rec in index_recommendations
            ]
            
            # Get connection pool status
            analysis_results['connection_pool_status'] = self.connection_pool.get_pool_status()
            
            # Generate summary
            analysis_results['summary'] = self._generate_optimization_summary(analysis_results)
            
            # Store in history
            self.optimization_history.append(analysis_results)
            
        except Exception as e:
            self.logger.error(f"Database optimization analysis failed: {str(e)}")
            analysis_results['error'] = str(e)
            raise
        
        self.logger.info(f"Database optimization analysis completed: {optimization_id}")
        return analysis_results

    def _generate_sample_queries(self):
        """Generate sample queries for analysis"""
        sample_queries = [
            "SELECT * FROM users WHERE status = 'active'",
            "SELECT username, email FROM users WHERE last_login > '2025-01-01'",
            "SELECT COUNT(*) FROM test_results WHERE result = 'pass'",
            "SELECT AVG(score) FROM test_results WHERE test_name = 'performance_test'",
            "SELECT u.username, COUNT(tr.id) FROM users u JOIN test_results tr ON u.id = tr.user_id GROUP BY u.username",
            "SELECT * FROM test_coverage WHERE coverage_percent < 80",
            "UPDATE users SET last_login = '2025-08-23' WHERE username = 'alice'"
        ]
        
        # Execute queries through the query analyzer for monitoring
        for query in sample_queries:
            try:
                with self.query_analyzer.monitor_query(query):
                    with self.connection_pool.get_connection() as conn:
                        conn.execute(query)
                        if query.strip().upper().startswith('SELECT'):
                            conn.fetchall()
                        else:
                            conn.commit()
            except Exception as e:
                self.logger.debug(f"Sample query execution failed: {e}")

    def _generate_optimization_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization analysis summary"""
        summary = {
            'optimization_id': analysis_results['optimization_id'],
            'timestamp': analysis_results['timestamp'],
            'queries_analyzed': 0,
            'slow_queries_count': 0,
            'index_recommendations_count': 0,
            'high_impact_recommendations': 0,
            'connection_pool_utilization': 0,
            'overall_health_score': 100
        }
        
        # Query analysis summary
        if 'query_analysis' in analysis_results:
            qa = analysis_results['query_analysis']
            summary['queries_analyzed'] = qa.get('statistics', {}).get('total_unique_queries', 0)
            summary['slow_queries_count'] = len(qa.get('slow_queries', []))
        
        # Index recommendations summary
        if 'index_recommendations' in analysis_results:
            recommendations = analysis_results['index_recommendations']
            summary['index_recommendations_count'] = len(recommendations)
            summary['high_impact_recommendations'] = len([r for r in recommendations if r.get('impact_score', 0) > 50])
        
        # Connection pool summary
        if 'connection_pool_status' in analysis_results:
            pool_status = analysis_results['connection_pool_status']
            summary['connection_pool_utilization'] = pool_status.get('pool_utilization_percent', 0)
        
        # Calculate overall health score
        health_penalties = 0
        if summary['slow_queries_count'] > 5:
            health_penalties += 20
        if summary['index_recommendations_count'] > 10:
            health_penalties += 15
        if summary['connection_pool_utilization'] > 80:
            health_penalties += 10
        
        summary['overall_health_score'] = max(0, 100 - health_penalties)
        
        return summary

    def apply_index_recommendations(self, recommendations: List[Dict[str, Any]], 
                                   confirm: bool = False) -> Dict[str, Any]:
        """Apply index recommendations to the database"""
        if not confirm:
            return {
                'status': 'confirmation_required',
                'message': 'Set confirm=True to apply index recommendations',
                'recommendations_count': len(recommendations)
            }
        
        results = {
            'applied': [],
            'failed': [],
            'skipped': []
        }
        
        with self.connection_pool.get_connection() as conn:
            for rec in recommendations:
                try:
                    # Check if index already exists (simplified check)
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE ?",
                        (f"idx_{rec['table_name']}_%",)
                    )
                    if cursor.fetchone():
                        results['skipped'].append({
                            'recommendation': rec,
                            'reason': 'Similar index already exists'
                        })
                        continue
                    
                    # Apply the index
                    conn.execute(rec['creation_sql'])
                    results['applied'].append(rec)
                    self.logger.info(f"Applied index: {rec['creation_sql']}")
                    
                except Exception as e:
                    results['failed'].append({
                        'recommendation': rec,
                        'error': str(e)
                    })
                    self.logger.error(f"Failed to apply index: {e}")
            
            conn.commit()
        
        return results

    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent optimization history"""
        return self.optimization_history[-limit:]

    def cleanup_resources(self):
        """Cleanup database optimization resources"""
        # Cleanup connection pool
        self.connection_pool.cleanup_idle_connections()
        
        # Stop monitoring if enabled
        if self.monitoring_system:
            try:
                self.monitoring_system.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error stopping monitoring: {e}")
        
        self.logger.info("Database optimization resources cleaned up")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'database_path': str(self.database_path),
            'database_exists': self.database_path.exists(),
            'monitoring_enabled': self.monitoring_system is not None,
            'connection_pool_status': self.connection_pool.get_pool_status(),
            'query_analyzer_stats': self.query_analyzer.get_query_statistics(),
            'optimization_runs': len(self.optimization_history),
            'last_optimization': self.optimization_history[-1]['timestamp'] if self.optimization_history else None
        }
        
        return status