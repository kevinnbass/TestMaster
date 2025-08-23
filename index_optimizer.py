#!/usr/bin/env python3
"""
🏗️ MODULE: Index Optimizer - Database Index Recommendation Engine
==================================================================

📋 PURPOSE:
    Generates database index recommendations based on query analysis patterns.
    Analyzes table access patterns and provides automated index suggestions.

🎯 CORE FUNCTIONALITY:
    • Table access pattern analysis from query logs
    • Column usage pattern extraction and frequency analysis
    • Index recommendation generation with impact scoring
    • SQL DDL generation for recommended indexes

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 08:20:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract index optimizer from database_performance_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for database index optimization
   └─ Impact: Improved modularity and single responsibility for index management

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: logging, re, typing, collections
🎯 Integration Points: query_performance_analyzer.py, database_optimization_models.py
⚡ Performance Notes: Pattern analysis optimized for large query sets
🔒 Security Notes: SQL parsing uses safe pattern matching

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via index impact analysis | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: query_performance_analyzer for query data
📤 Provides: Index optimization capabilities for database performance
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import logging
import re
from typing import Dict, List, Any
from collections import defaultdict

# Import data models and dependencies
from database_optimization_models import IndexRecommendation


class IndexOptimizer:
    """Generates index recommendations based on query analysis"""
    
    def __init__(self, query_analyzer):
        """Initialize with reference to query analyzer"""
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

    def generate_index_recommendations(self, table_patterns: Dict[str, Any] = None) -> List[IndexRecommendation]:
        """Generate index recommendations based on usage patterns"""
        if table_patterns is None:
            table_patterns = self.analyze_table_access_patterns("")
        
        recommendations = []
        
        for table_name, patterns in table_patterns.items():
            # Skip tables with minimal activity
            if patterns['select_count'] < 5:
                continue
            
            # Recommend indexes for frequently used WHERE columns
            for column, frequency in patterns['where_columns'].items():
                if frequency >= 3:  # Column used in WHERE clause 3+ times
                    recommendation = IndexRecommendation(
                        table_name=table_name,
                        columns=[column],
                        index_type='btree',
                        estimated_benefit=self._calculate_where_index_benefit(frequency, patterns),
                        queries_affected=[],  # Would be populated from actual query analysis
                        creation_sql=f"CREATE INDEX idx_{table_name}_{column} ON {table_name} ({column});",
                        impact_score=frequency * 10,
                        rationale=f"Column '{column}' used in WHERE clause {frequency} times"
                    )
                    recommendations.append(recommendation)
            
            # Recommend composite indexes for ORDER BY columns
            order_columns = [(col, freq) for col, freq in patterns['order_columns'].items() if freq >= 2]
            if len(order_columns) > 1:
                # Create composite index for most frequently ordered columns
                top_order_columns = sorted(order_columns, key=lambda x: x[1], reverse=True)[:3]
                column_names = [col for col, _ in top_order_columns]
                
                recommendation = IndexRecommendation(
                    table_name=table_name,
                    columns=column_names,
                    index_type='composite',
                    estimated_benefit=sum(freq for _, freq in top_order_columns) * 0.8,
                    queries_affected=[],
                    creation_sql=f"CREATE INDEX idx_{table_name}_order ON {table_name} ({', '.join(column_names)});",
                    impact_score=sum(freq for _, freq in top_order_columns) * 5,
                    rationale=f"Composite index for ORDER BY columns: {', '.join(column_names)}"
                )
                recommendations.append(recommendation)
        
        # Sort recommendations by impact score
        recommendations.sort(key=lambda x: x.impact_score, reverse=True)
        
        return recommendations

    def _calculate_where_index_benefit(self, frequency: int, patterns: Dict) -> float:
        """Calculate estimated benefit of a WHERE clause index"""
        base_benefit = frequency * 2.0  # Base benefit from frequency
        
        # Increase benefit for tables with high select activity
        if patterns['select_count'] > 50:
            base_benefit *= 1.5
        elif patterns['select_count'] > 20:
            base_benefit *= 1.2
        
        return base_benefit

    def analyze_existing_indexes(self, database_path: str) -> Dict[str, List[str]]:
        """Analyze existing indexes in the database"""
        # This would query actual database metadata in production
        # For now, return empty structure
        existing_indexes = defaultdict(list)
        
        # In a real implementation, this would execute:
        # SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index';
        # and parse the results
        
        return dict(existing_indexes)

    def prioritize_recommendations(self, recommendations: List[IndexRecommendation], 
                                 existing_indexes: Dict[str, List[str]] = None) -> List[IndexRecommendation]:
        """Prioritize index recommendations based on impact and existing indexes"""
        if existing_indexes is None:
            existing_indexes = {}
        
        prioritized = []
        
        for rec in recommendations:
            # Skip if index already exists (simplified check)
            table_indexes = existing_indexes.get(rec.table_name, [])
            if any(rec.columns[0] in idx for idx in table_indexes):
                continue
            
            # Adjust priority based on table activity and impact
            priority_multiplier = 1.0
            
            if rec.index_type == 'composite':
                priority_multiplier *= 1.3  # Composite indexes often more beneficial
            
            if rec.estimated_benefit > 10:
                priority_multiplier *= 1.2  # High benefit indexes
            
            # Update impact score with priority
            rec.impact_score *= priority_multiplier
            prioritized.append(rec)
        
        # Re-sort by updated impact score
        prioritized.sort(key=lambda x: x.impact_score, reverse=True)
        
        return prioritized

    def generate_index_creation_script(self, recommendations: List[IndexRecommendation]) -> str:
        """Generate SQL script for creating recommended indexes"""
        script_lines = [
            "-- Database Index Optimization Script",
            f"-- Generated: {self.logger.name}",
            "-- Execute with caution and test in development environment first",
            "",
        ]
        
        for i, rec in enumerate(recommendations, 1):
            script_lines.extend([
                f"-- Recommendation {i}: {rec.rationale}",
                f"-- Impact Score: {rec.impact_score:.1f}",
                f"-- Estimated Benefit: {rec.estimated_benefit:.2f}",
                rec.creation_sql,
                "",
            ])
        
        return "\n".join(script_lines)