#!/usr/bin/env python3
"""
SQL Query Analyzer
Agent B Hours 90-100: Query Performance Analysis

Analyzes SQL queries for performance issues and optimization opportunities.
"""

import sqlite3
import time
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter

@dataclass
class QueryAnalysis:
    """Analysis of a SQL query"""
    query: str
    query_type: str
    execution_time_ms: float
    rows_examined: int
    rows_returned: int
    tables_accessed: List[str]
    indexes_used: List[str]
    optimization_suggestions: List[str]
    complexity_score: int
    timestamp: datetime

@dataclass
class QueryPattern:
    """Common query pattern"""
    pattern: str
    count: int
    avg_execution_time: float
    total_time: float
    examples: List[str]

class QueryAnalyzer:
    """Analyzes SQL queries for performance optimization"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.query_history: List[QueryAnalysis] = []
        self.slow_query_threshold = 100  # ms
        
        # Common performance patterns
        self.optimization_patterns = {
            r'SELECT \* FROM': 'Avoid SELECT * - specify only needed columns',
            r'WHERE.*LIKE.*%.*%': 'LIKE with leading wildcard prevents index usage',
            r'ORDER BY.*LIMIT': 'Consider using indexed columns for ORDER BY',
            r'WHERE.*OR.*OR': 'Multiple ORs may benefit from UNION instead',
            r'SELECT.*FROM.*WHERE.*=.*AND.*=': 'Consider composite index',
            r'COUNT\(\*\).*WHERE': 'COUNT(*) can be slow on large tables',
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a single SQL query"""
        start_time = time.time()
        
        # Clean and normalize query
        normalized_query = self._normalize_query(query)
        query_type = self._get_query_type(normalized_query)
        
        # Execute with timing
        execution_time_ms, rows_examined, rows_returned = self._execute_with_timing(query)
        
        # Extract table and index information
        tables_accessed = self._extract_tables(normalized_query)
        indexes_used = self._get_indexes_used(query, tables_accessed)
        
        # Generate optimization suggestions
        suggestions = self._analyze_for_optimizations(normalized_query, execution_time_ms)
        
        # Calculate complexity score
        complexity = self._calculate_complexity(normalized_query)
        
        analysis = QueryAnalysis(
            query=query,
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            rows_examined=rows_examined,
            rows_returned=rows_returned,
            tables_accessed=tables_accessed,
            indexes_used=indexes_used,
            optimization_suggestions=suggestions,
            complexity_score=complexity,
            timestamp=datetime.now()
        )
        
        self.query_history.append(analysis)
        return analysis
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for analysis"""
        # Remove extra whitespace and comments
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        return query.upper()
    
    def _get_query_type(self, query: str) -> str:
        """Determine query type"""
        query = query.strip().upper()
        if query.startswith('SELECT'):
            return 'SELECT'
        elif query.startswith('INSERT'):
            return 'INSERT'
        elif query.startswith('UPDATE'):
            return 'UPDATE'
        elif query.startswith('DELETE'):
            return 'DELETE'
        elif query.startswith('CREATE'):
            return 'CREATE'
        elif query.startswith('DROP'):
            return 'DROP'
        elif query.startswith('ALTER'):
            return 'ALTER'
        else:
            return 'OTHER'
    
    def _execute_with_timing(self, query: str) -> Tuple[float, int, int]:
        """Execute query and measure timing"""
        try:
            if not self.db_path.exists():
                return 0.0, 0, 0
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            start_time = time.time()
            cursor.execute(query)
            results = cursor.fetchall()
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            rows_returned = len(results)
            rows_examined = rows_returned  # Simplified for SQLite
            
            conn.close()
            return execution_time, rows_examined, rows_returned
            
        except Exception as e:
            # Query failed, return minimal info
            return 0.0, 0, 0
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query"""
        tables = []
        
        # Simple regex to find table names after FROM and JOIN
        from_pattern = r'FROM\s+(\w+)'
        join_pattern = r'JOIN\s+(\w+)'
        
        tables.extend(re.findall(from_pattern, query, re.IGNORECASE))
        tables.extend(re.findall(join_pattern, query, re.IGNORECASE))
        
        return list(set(tables))  # Remove duplicates
    
    def _get_indexes_used(self, query: str, tables: List[str]) -> List[str]:
        """Get indexes that could be used by this query"""
        indexes = []
        
        try:
            if not self.db_path.exists():
                return indexes
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for table in tables:
                try:
                    cursor.execute(f"PRAGMA index_list({table})")
                    table_indexes = cursor.fetchall()
                    for index_info in table_indexes:
                        indexes.append(f"{table}.{index_info[1]}")  # table.index_name
                except:
                    pass
            
            conn.close()
            
        except Exception as e:
            pass
        
        return indexes
    
    def _analyze_for_optimizations(self, query: str, execution_time: float) -> List[str]:
        """Analyze query for optimization opportunities"""
        suggestions = []
        
        # Check execution time
        if execution_time > self.slow_query_threshold:
            suggestions.append(f"Slow query ({execution_time:.1f}ms) - consider optimization")
        
        # Check against optimization patterns
        for pattern, suggestion in self.optimization_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                suggestions.append(suggestion)
        
        # Specific SQLite optimizations
        if 'SELECT *' in query:
            suggestions.append("SELECT * can be slow - specify only needed columns")
        
        if 'ORDER BY' in query and 'LIMIT' in query:
            suggestions.append("ORDER BY + LIMIT benefits from indexed columns")
        
        if query.count('JOIN') > 2:
            suggestions.append("Multiple JOINs - ensure proper indexing on join columns")
        
        if 'WHERE' in query and 'INDEX' not in query:
            suggestions.append("WHERE clause detected - ensure indexed columns are used")
        
        return suggestions
    
    def _calculate_complexity(self, query: str) -> int:
        """Calculate query complexity score (1-10)"""
        score = 1
        
        # Add points for various complexity factors
        if 'JOIN' in query:
            score += query.count('JOIN')
        
        if 'SUBQUERY' in query or '(' in query:
            score += 2
        
        if 'GROUP BY' in query:
            score += 1
        
        if 'ORDER BY' in query:
            score += 1
        
        if 'HAVING' in query:
            score += 2
        
        if 'UNION' in query:
            score += 2
        
        # Count conditions
        where_conditions = query.count('AND') + query.count('OR')
        score += min(where_conditions, 3)
        
        return min(score, 10)  # Cap at 10
    
    def get_slow_queries(self, threshold_ms: float = None) -> List[QueryAnalysis]:
        """Get queries slower than threshold"""
        threshold = threshold_ms or self.slow_query_threshold
        return [q for q in self.query_history if q.execution_time_ms > threshold]
    
    def get_query_patterns(self) -> List[QueryPattern]:
        """Analyze query patterns"""
        pattern_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'examples': []
        })
        
        for query_analysis in self.query_history:
            # Simplify query to pattern
            pattern = self._simplify_to_pattern(query_analysis.query)
            
            pattern_stats[pattern]['count'] += 1
            pattern_stats[pattern]['total_time'] += query_analysis.execution_time_ms
            
            if len(pattern_stats[pattern]['examples']) < 3:
                pattern_stats[pattern]['examples'].append(query_analysis.query[:100])
        
        patterns = []
        for pattern, stats in pattern_stats.items():
            if stats['count'] > 1:  # Only patterns that appear multiple times
                patterns.append(QueryPattern(
                    pattern=pattern,
                    count=stats['count'],
                    avg_execution_time=stats['total_time'] / stats['count'],
                    total_time=stats['total_time'],
                    examples=stats['examples']
                ))
        
        return sorted(patterns, key=lambda x: x.total_time, reverse=True)
    
    def _simplify_to_pattern(self, query: str) -> str:
        """Simplify query to a pattern for grouping"""
        # Remove specific values and normalize
        pattern = re.sub(r"'[^']*'", "'?'", query)  # String literals
        pattern = re.sub(r'\b\d+\b', '?', pattern)  # Numbers
        pattern = re.sub(r'=\s*\?', '= ?', pattern)  # Normalize equals
        pattern = re.sub(r'\s+', ' ', pattern.strip())  # Normalize whitespace
        return pattern[:200]  # Limit length
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.query_history:
            return {"message": "No queries analyzed yet"}
        
        slow_queries = self.get_slow_queries()
        patterns = self.get_query_patterns()
        
        # Query type distribution
        type_distribution = Counter(q.query_type for q in self.query_history)
        
        # Complexity distribution
        complexity_distribution = Counter(q.complexity_score for q in self.query_history)
        
        # Most common suggestions
        all_suggestions = []
        for q in self.query_history:
            all_suggestions.extend(q.optimization_suggestions)
        suggestion_frequency = Counter(all_suggestions)
        
        return {
            "summary": {
                "total_queries_analyzed": len(self.query_history),
                "slow_queries": len(slow_queries),
                "avg_execution_time": sum(q.execution_time_ms for q in self.query_history) / len(self.query_history),
                "total_execution_time": sum(q.execution_time_ms for q in self.query_history),
                "unique_patterns": len(patterns)
            },
            "slow_queries": [asdict(q) for q in slow_queries[:5]],  # Top 5 slow queries
            "query_patterns": [asdict(p) for p in patterns[:10]],   # Top 10 patterns
            "query_type_distribution": dict(type_distribution),
            "complexity_distribution": dict(complexity_distribution),
            "common_optimizations": [
                {"suggestion": suggestion, "frequency": count}
                for suggestion, count in suggestion_frequency.most_common(5)
            ],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate general recommendations"""
        recommendations = []
        
        slow_queries = self.get_slow_queries()
        if len(slow_queries) > len(self.query_history) * 0.1:  # More than 10% slow
            recommendations.append("High percentage of slow queries - review indexing strategy")
        
        complex_queries = [q for q in self.query_history if q.complexity_score > 7]
        if len(complex_queries) > 5:
            recommendations.append("Many complex queries detected - consider query simplification")
        
        select_star_queries = [q for q in self.query_history if 'SELECT *' in q.query.upper()]
        if len(select_star_queries) > 0:
            recommendations.append("SELECT * queries detected - specify only needed columns")
        
        if not recommendations:
            recommendations.append("Query performance looks good - continue monitoring")
        
        return recommendations

def analyze_database_queries(db_path: str, sample_queries: List[str] = None) -> Dict[str, Any]:
    """Quick function to analyze database queries"""
    analyzer = QueryAnalyzer(db_path)
    
    # If no sample queries provided, use common ones
    if not sample_queries:
        sample_queries = [
            "SELECT * FROM sqlite_master WHERE type='table'",
            "SELECT name FROM sqlite_master WHERE type='table'",
            "SELECT COUNT(*) FROM sqlite_master",
        ]
    
    # Analyze each query
    for query in sample_queries:
        try:
            analyzer.analyze_query(query)
        except Exception as e:
            print(f"[WARNING] Failed to analyze query: {query[:50]}... Error: {e}")
    
    return analyzer.generate_optimization_report()

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python query_analyzer.py <database_path> [query1] [query2] ...")
        print("Example: python query_analyzer.py ./cache/cache.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    custom_queries = sys.argv[2:] if len(sys.argv) > 2 else None
    
    print(f"Analyzing queries for database: {db_path}")
    report = analyze_database_queries(db_path, custom_queries)
    
    print("\nQUERY ANALYSIS REPORT")
    print("=" * 50)
    print(json.dumps(report, indent=2, default=str))