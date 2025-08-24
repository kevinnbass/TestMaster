#!/usr/bin/env python3
"""
Query Optimization Engine
Agent B Hours 100-110: Advanced User Experience & Enhancement

Intelligent SQL query optimization with automatic suggestions and performance tuning.
"""

import json
import sqlite3
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib

@dataclass
class QueryOptimization:
    """Query optimization result"""
    original_query: str
    optimized_query: str
    optimization_type: str
    performance_gain: float  # Estimated % improvement
    explanation: str
    confidence: float  # 0-1 confidence in optimization
    applicable: bool

@dataclass
class QueryProfile:
    """Query execution profile"""
    query_hash: str
    original_query: str
    execution_count: int
    avg_execution_time_ms: float
    min_execution_time_ms: float
    max_execution_time_ms: float
    total_execution_time_ms: float
    last_execution: datetime
    optimization_applied: bool
    performance_trend: str  # 'improving', 'degrading', 'stable'

class QueryOptimizationEngine:
    """Advanced query optimization engine with ML-like pattern recognition"""
    
    def __init__(self, db_path: str = None, profile_file: str = "query_profiles.json"):
        self.db_path = Path(db_path) if db_path else None
        self.profile_file = Path(profile_file)
        self.query_profiles = {}
        self.optimization_patterns = {}
        self.performance_baselines = {}
        
        # Load existing profiles
        self.load_profiles()
        self.setup_optimization_patterns()
        
        # Optimization statistics
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'avg_performance_gain': 0.0,
            'patterns_learned': 0
        }
    
    def load_profiles(self):
        """Load query profiles from file"""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert back to QueryProfile objects
                for query_hash, profile_data in data.get('profiles', {}).items():
                    profile_data['last_execution'] = datetime.fromisoformat(profile_data['last_execution'])
                    self.query_profiles[query_hash] = QueryProfile(**profile_data)
                    
                # Load stats
                self.optimization_stats = data.get('stats', self.optimization_stats)
                
            except Exception as e:
                print(f"[WARNING] Failed to load query profiles: {e}")
    
    def save_profiles(self):
        """Save query profiles to file"""
        try:
            # Convert to serializable format
            serializable_profiles = {}
            for query_hash, profile in self.query_profiles.items():
                profile_dict = asdict(profile)
                profile_dict['last_execution'] = profile_dict['last_execution'].isoformat()
                serializable_profiles[query_hash] = profile_dict
            
            data = {
                'profiles': serializable_profiles,
                'stats': self.optimization_stats
            }
            
            with open(self.profile_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save query profiles: {e}")
    
    def setup_optimization_patterns(self):
        """Setup optimization patterns with confidence scores"""
        self.optimization_patterns = {
            # SELECT * optimizations
            'select_star': {
                'pattern': r'SELECT\s+\*\s+FROM',
                'optimization': self._optimize_select_star,
                'confidence': 0.9,
                'category': 'column_selection',
                'description': 'Replace SELECT * with specific columns'
            },
            
            # Index usage optimizations
            'unindexed_where': {
                'pattern': r'WHERE\s+(\w+)\s*=',
                'optimization': self._optimize_where_clause,
                'confidence': 0.8,
                'category': 'indexing',
                'description': 'Suggest index creation for WHERE clauses'
            },
            
            # JOIN optimizations
            'inefficient_join': {
                'pattern': r'JOIN\s+(\w+)\s+ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)',
                'optimization': self._optimize_join,
                'confidence': 0.7,
                'category': 'joins',
                'description': 'Optimize JOIN operations'
            },
            
            # Subquery optimizations
            'subquery_in_select': {
                'pattern': r'SELECT.*\(SELECT.*\).*FROM',
                'optimization': self._optimize_subquery,
                'confidence': 0.8,
                'category': 'subqueries',
                'description': 'Convert correlated subqueries to JOINs'
            },
            
            # LIKE optimizations
            'leading_wildcard': {
                'pattern': r"LIKE\s+'%.*'",
                'optimization': self._optimize_like_clause,
                'confidence': 0.9,
                'category': 'pattern_matching',
                'description': 'Optimize LIKE patterns with leading wildcards'
            },
            
            # ORDER BY + LIMIT optimizations
            'order_limit': {
                'pattern': r'ORDER\s+BY.*LIMIT',
                'optimization': self._optimize_order_limit,
                'confidence': 0.8,
                'category': 'sorting',
                'description': 'Optimize ORDER BY with LIMIT'
            },
            
            # COUNT optimizations
            'count_star': {
                'pattern': r'COUNT\(\*\)',
                'optimization': self._optimize_count,
                'confidence': 0.7,
                'category': 'aggregation',
                'description': 'Optimize COUNT(*) operations'
            },
            
            # Redundant DISTINCT
            'unnecessary_distinct': {
                'pattern': r'SELECT\s+DISTINCT.*FROM.*WHERE.*=',
                'optimization': self._optimize_distinct,
                'confidence': 0.6,
                'category': 'redundancy',
                'description': 'Remove unnecessary DISTINCT'
            }
        }
    
    def analyze_and_optimize_query(self, query: str) -> List[QueryOptimization]:
        """Analyze query and generate optimizations"""
        optimizations = []
        normalized_query = self._normalize_query(query)
        
        # Check each optimization pattern
        for pattern_name, pattern_info in self.optimization_patterns.items():
            if re.search(pattern_info['pattern'], normalized_query, re.IGNORECASE):
                optimization = pattern_info['optimization'](query, normalized_query)
                if optimization and optimization.applicable:
                    optimizations.append(optimization)
        
        # Sort by confidence and performance gain
        optimizations.sort(key=lambda x: (x.confidence * x.performance_gain), reverse=True)
        
        return optimizations
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        # Remove comments
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        return query.upper()
    
    def _optimize_select_star(self, original: str, normalized: str) -> Optional[QueryOptimization]:
        """Optimize SELECT * queries"""
        if not self.db_path or not self.db_path.exists():
            return QueryOptimization(
                original_query=original,
                optimized_query=original,
                optimization_type='select_star',
                performance_gain=25.0,
                explanation="Replace SELECT * with specific column names to reduce I/O",
                confidence=0.9,
                applicable=False
            )
        
        try:
            # Extract table name
            table_match = re.search(r'FROM\s+(\w+)', normalized, re.IGNORECASE)
            if not table_match:
                return None
            
            table_name = table_match.group(1)
            
            # Get table schema
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            conn.close()
            
            if not columns:
                return None
            
            # Generate optimized query
            column_names = [col[1] for col in columns[:5]]  # Limit to first 5 columns
            optimized_query = re.sub(
                r'SELECT\s+\*',
                f"SELECT {', '.join(column_names)}",
                original,
                flags=re.IGNORECASE
            )
            
            return QueryOptimization(
                original_query=original,
                optimized_query=optimized_query,
                optimization_type='select_star',
                performance_gain=30.0,
                explanation=f"Replaced SELECT * with specific columns: {', '.join(column_names)}",
                confidence=0.9,
                applicable=True
            )
            
        except Exception as e:
            print(f"[WARNING] Failed to optimize SELECT *: {e}")
            return None
    
    def _optimize_where_clause(self, original: str, normalized: str) -> Optional[QueryOptimization]:
        """Optimize WHERE clause with index suggestions"""
        # Extract WHERE conditions
        where_matches = re.findall(r'WHERE\s+(\w+)\s*=', normalized, re.IGNORECASE)
        if not where_matches:
            return None
        
        column_name = where_matches[0]
        
        suggestion = f"-- Consider creating index: CREATE INDEX idx_{column_name} ON table_name({column_name});\n{original}"
        
        return QueryOptimization(
            original_query=original,
            optimized_query=suggestion,
            optimization_type='indexing',
            performance_gain=50.0,
            explanation=f"Adding index on column '{column_name}' could improve WHERE clause performance",
            confidence=0.8,
            applicable=True
        )
    
    def _optimize_join(self, original: str, normalized: str) -> Optional[QueryOptimization]:
        """Optimize JOIN operations"""
        # Extract JOIN information
        join_matches = re.findall(r'JOIN\s+(\w+)\s+ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)', normalized, re.IGNORECASE)
        if not join_matches:
            return None
        
        table_name, left_col, right_col = join_matches[0]
        
        suggestion = f"-- Consider indexes on join columns: {left_col}, {right_col}\n{original}"
        
        return QueryOptimization(
            original_query=original,
            optimized_query=suggestion,
            optimization_type='join_optimization',
            performance_gain=40.0,
            explanation=f"Index on join columns {left_col} and {right_col} could improve JOIN performance",
            confidence=0.7,
            applicable=True
        )
    
    def _optimize_subquery(self, original: str, normalized: str) -> Optional[QueryOptimization]:
        """Optimize subqueries"""
        # Simple placeholder - would need more sophisticated analysis
        return QueryOptimization(
            original_query=original,
            optimized_query=original,
            optimization_type='subquery',
            performance_gain=35.0,
            explanation="Consider converting correlated subquery to JOIN for better performance",
            confidence=0.8,
            applicable=True
        )
    
    def _optimize_like_clause(self, original: str, normalized: str) -> Optional[QueryOptimization]:
        """Optimize LIKE clauses with leading wildcards"""
        # Find LIKE patterns with leading wildcards
        like_patterns = re.findall(r"LIKE\s+'(%[^']*)'", normalized, re.IGNORECASE)
        if not like_patterns:
            return None
        
        pattern = like_patterns[0]
        if not pattern.startswith('%'):
            return None
        
        return QueryOptimization(
            original_query=original,
            optimized_query=original,
            optimization_type='like_optimization',
            performance_gain=20.0,
            explanation="LIKE with leading wildcard prevents index usage. Consider full-text search or reverse indexing",
            confidence=0.9,
            applicable=True
        )
    
    def _optimize_order_limit(self, original: str, normalized: str) -> Optional[QueryOptimization]:
        """Optimize ORDER BY with LIMIT"""
        order_match = re.search(r'ORDER\s+BY\s+(\w+)', normalized, re.IGNORECASE)
        if not order_match:
            return None
        
        order_column = order_match.group(1)
        
        suggestion = f"-- Consider index for ORDER BY: CREATE INDEX idx_{order_column}_sort ON table_name({order_column});\n{original}"
        
        return QueryOptimization(
            original_query=original,
            optimized_query=suggestion,
            optimization_type='sorting',
            performance_gain=45.0,
            explanation=f"Index on '{order_column}' could significantly improve ORDER BY + LIMIT performance",
            confidence=0.8,
            applicable=True
        )
    
    def _optimize_count(self, original: str, normalized: str) -> Optional[QueryOptimization]:
        """Optimize COUNT operations"""
        if 'WHERE' in normalized:
            return QueryOptimization(
                original_query=original,
                optimized_query=original,
                optimization_type='count_optimization',
                performance_gain=15.0,
                explanation="COUNT(*) with WHERE clause can be slow. Consider maintaining count tables for frequently counted data",
                confidence=0.7,
                applicable=True
            )
        return None
    
    def _optimize_distinct(self, original: str, normalized: str) -> Optional[QueryOptimization]:
        """Optimize unnecessary DISTINCT"""
        return QueryOptimization(
            original_query=original,
            optimized_query=original,
            optimization_type='distinct_optimization',
            performance_gain=10.0,
            explanation="DISTINCT may be unnecessary if query results are already unique",
            confidence=0.6,
            applicable=True
        )
    
    def profile_query(self, query: str, execution_time_ms: float):
        """Profile query execution"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self.query_profiles:
            # Update existing profile
            profile = self.query_profiles[query_hash]
            profile.execution_count += 1
            profile.total_execution_time_ms += execution_time_ms
            profile.avg_execution_time_ms = profile.total_execution_time_ms / profile.execution_count
            profile.min_execution_time_ms = min(profile.min_execution_time_ms, execution_time_ms)
            profile.max_execution_time_ms = max(profile.max_execution_time_ms, execution_time_ms)
            profile.last_execution = datetime.now()
            
            # Update performance trend
            profile.performance_trend = self._calculate_performance_trend(profile)
        else:
            # Create new profile
            self.query_profiles[query_hash] = QueryProfile(
                query_hash=query_hash,
                original_query=query,
                execution_count=1,
                avg_execution_time_ms=execution_time_ms,
                min_execution_time_ms=execution_time_ms,
                max_execution_time_ms=execution_time_ms,
                total_execution_time_ms=execution_time_ms,
                last_execution=datetime.now(),
                optimization_applied=False,
                performance_trend='stable'
            )
        
        self.save_profiles()
    
    def _calculate_performance_trend(self, profile: QueryProfile) -> str:
        """Calculate performance trend for a query"""
        # Simplified trend calculation
        # In a real system, this would analyze execution times over time
        recent_avg = profile.avg_execution_time_ms
        
        if recent_avg < profile.min_execution_time_ms * 1.1:
            return 'improving'
        elif recent_avg > profile.max_execution_time_ms * 0.9:
            return 'degrading'
        else:
            return 'stable'
    
    def get_optimization_report(self, query: str = None) -> Dict[str, Any]:
        """Generate optimization report"""
        if query:
            # Single query optimization
            optimizations = self.analyze_and_optimize_query(query)
            return {
                'query': query,
                'optimizations': [asdict(opt) for opt in optimizations],
                'optimization_count': len(optimizations),
                'max_performance_gain': max([opt.performance_gain for opt in optimizations], default=0)
            }
        else:
            # System-wide optimization report
            total_profiles = len(self.query_profiles)
            slow_queries = [p for p in self.query_profiles.values() if p.avg_execution_time_ms > 100]
            degrading_queries = [p for p in self.query_profiles.values() if p.performance_trend == 'degrading']
            
            return {
                'system_stats': self.optimization_stats,
                'query_profiles': {
                    'total_queries': total_profiles,
                    'slow_queries': len(slow_queries),
                    'degrading_queries': len(degrading_queries),
                    'optimized_queries': sum(1 for p in self.query_profiles.values() if p.optimization_applied)
                },
                'recommendations': self._generate_system_recommendations()
            }
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide optimization recommendations"""
        recommendations = []
        
        slow_queries = [p for p in self.query_profiles.values() if p.avg_execution_time_ms > 100]
        if slow_queries:
            recommendations.append(f"Found {len(slow_queries)} slow queries (>100ms) - consider optimization")
        
        degrading_queries = [p for p in self.query_profiles.values() if p.performance_trend == 'degrading']
        if degrading_queries:
            recommendations.append(f"Found {len(degrading_queries)} queries with degrading performance - investigate")
        
        frequent_queries = [p for p in self.query_profiles.values() if p.execution_count > 100]
        if frequent_queries:
            recommendations.append(f"Found {len(frequent_queries)} frequently executed queries - ensure they're optimized")
        
        if not recommendations:
            recommendations.append("Query performance looks good - continue monitoring")
        
        return recommendations

def main():
    """Main function for testing optimization engine"""
    # Initialize engine
    engine = QueryOptimizationEngine()
    
    print("[OK] Query Optimization Engine initialized")
    print(f"[OK] Loaded {len(engine.optimization_patterns)} optimization patterns")
    
    # Test queries
    test_queries = [
        "SELECT * FROM users WHERE name = 'John'",
        "SELECT COUNT(*) FROM orders WHERE status = 'pending'",
        "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id ORDER BY o.total DESC LIMIT 10",
        "SELECT * FROM products WHERE description LIKE '%expensive%'"
    ]
    
    print("\n[TEST] Analyzing test queries for optimizations...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Original: {query}")
        
        optimizations = engine.analyze_and_optimize_query(query)
        
        if optimizations:
            for opt in optimizations:
                print(f"Optimization: {opt.optimization_type}")
                print(f"Performance gain: {opt.performance_gain}%")
                print(f"Explanation: {opt.explanation}")
                print(f"Confidence: {opt.confidence}")
                if opt.optimized_query != opt.original_query:
                    print(f"Optimized query: {opt.optimized_query}")
        else:
            print("No optimizations found")
        
        # Profile the query (simulate execution time)
        execution_time = 50.0 + (i * 25)  # Simulated execution time
        engine.profile_query(query, execution_time)
        print(f"Profiled execution time: {execution_time}ms")
    
    # Generate optimization report
    print("\n" + "="*50)
    report = engine.get_optimization_report()
    print("OPTIMIZATION REPORT:")
    print(f"- Total Query Profiles: {report['query_profiles']['total_queries']}")
    print(f"- Slow Queries: {report['query_profiles']['slow_queries']}")
    print(f"- Degrading Queries: {report['query_profiles']['degrading_queries']}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main()