#!/usr/bin/env python3
"""
🏗️ MODULE: Database Optimization Models - Data Structures and Type Definitions
==================================================================

📋 PURPOSE:
    Data classes and type definitions for database performance optimization system.
    Contains all configuration classes, analysis results, and recommendation structures.

🎯 CORE FUNCTIONALITY:
    • QueryAnalysis data structure for query performance metrics
    • IndexRecommendation for index optimization suggestions
    • ConnectionPoolConfig for database connection management

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 08:00:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract data models from database_performance_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for database optimization data structures
   └─ Impact: Improved modularity and testability of database optimization components

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python  
📦 Dependencies: typing, datetime, dataclasses
🎯 Integration Points: query_performance_analyzer.py, index_optimizer.py
⚡ Performance Notes: Lightweight data structures for database optimization
🔒 Security Notes: No security-sensitive operations

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: N/A | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: N/A | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Standard library only
📤 Provides: Data structures for database optimization system
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


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