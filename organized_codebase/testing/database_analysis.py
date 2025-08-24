"""
Database Query Analyzer for N+1 and Performance Issues
=======================================================

Implements comprehensive database query analysis:
- N+1 query problem detection
- Query complexity analysis
- Index usage assessment
- Transaction pattern analysis
- Connection pooling detection
- Query optimization recommendations
- ORM-specific pattern detection (SQLAlchemy, Django ORM)
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class DatabaseAnalyzer(BaseAnalyzer):
    """Analyzer for database query patterns and performance issues."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        
        # ORM patterns
        self.orm_patterns = {
            "sqlalchemy": {
                "query_methods": ["query", "filter", "filter_by", "all", "first", "one", "scalar"],
                "relationship_loading": ["joinedload", "subqueryload", "selectinload", "lazyload"],
                "session_methods": ["add", "commit", "rollback", "flush", "expire"],
            },
            "django": {
                "query_methods": ["objects", "filter", "exclude", "get", "all", "values", "values_list"],
                "optimization": ["select_related", "prefetch_related", "only", "defer"],
                "aggregation": ["aggregate", "annotate", "count", "sum", "avg"],
            },
            "peewee": {
                "query_methods": ["select", "where", "join", "get", "create"],
                "optimization": ["prefetch", "switch"],
            }
        }
        
        # SQL anti-patterns
        self.anti_patterns = {
            "n_plus_one": "Multiple queries in loops",
            "missing_index": "Queries without proper indexing",
            "cartesian_product": "Missing JOIN conditions",
            "implicit_conversion": "Type mismatch in comparisons",
            "wildcard_select": "SELECT * usage",
            "no_pagination": "Unbounded result sets",
            "transaction_abuse": "Long-running transactions"
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive database query analysis."""
        print("[INFO] Analyzing Database Query Patterns...")
        
        results = {
            "n_plus_one_detection": self._detect_n_plus_one_queries(),
            "query_complexity": self._analyze_query_complexity(),
            "index_analysis": self._analyze_index_usage(),
            "transaction_patterns": self._analyze_transaction_patterns(),
            "connection_pooling": self._analyze_connection_pooling(),
            "orm_patterns": self._analyze_orm_patterns(),
            "raw_sql_analysis": self._analyze_raw_sql(),
            "optimization_opportunities": self._identify_optimization_opportunities(),
            "database_metrics": self._calculate_database_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} database aspects")
        return results
    
    def _detect_n_plus_one_queries(self) -> Dict[str, Any]:
        """Detect N+1 query problems in the code."""
        n_plus_one = {
            "detected_patterns": [],
            "risk_locations": [],
            "orm_specific": defaultdict(list),
            "severity_distribution": defaultdict(int)
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect loop-based query patterns
                loop_queries = self._detect_queries_in_loops(tree, content)
                for query in loop_queries:
                    query["file"] = file_key
                    n_plus_one["detected_patterns"].append(query)
                    n_plus_one["severity_distribution"][query["severity"]] += 1
                
                # Detect ORM-specific N+1 patterns
                if "sqlalchemy" in content.lower():
                    sqlalchemy_patterns = self._detect_sqlalchemy_n_plus_one(tree, content)
                    n_plus_one["orm_specific"]["sqlalchemy"].extend(sqlalchemy_patterns)
                
                if "django" in content or "objects" in content:
                    django_patterns = self._detect_django_n_plus_one(tree, content)
                    n_plus_one["orm_specific"]["django"].extend(django_patterns)
                
            except:
                continue
        
        # Identify high-risk locations
        n_plus_one["risk_locations"] = self._identify_n_plus_one_hotspots(n_plus_one)
        
        return n_plus_one
    
    def _detect_queries_in_loops(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect database queries inside loops."""
        queries_in_loops = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check for query operations inside loops
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        if self._is_database_query(inner, content):
                            severity = self._assess_query_severity(node, inner)
                            queries_in_loops.append({
                                "line": inner.lineno,
                                "loop_line": node.lineno,
                                "type": "query_in_loop",
                                "severity": severity,
                                "pattern": self._identify_query_pattern(inner),
                                "recommendation": self._get_n_plus_one_fix(inner, content)
                            })
        
        return queries_in_loops
    
    def _is_database_query(self, node: ast.Call, content: str) -> bool:
        """Check if a call is a database query."""
        # Check for ORM query methods
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            
            # Check all ORM patterns
            for orm, patterns in self.orm_patterns.items():
                if attr_name in patterns["query_methods"]:
                    return True
            
            # Check for common query patterns
            if any(keyword in attr_name.lower() for keyword in ["query", "select", "fetch", "find", "get"]):
                return True
        
        # Check for execute() calls (raw SQL)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ["execute", "executemany", "fetchall", "fetchone"]:
                return True
        
        return False
    
    def _detect_sqlalchemy_n_plus_one(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect SQLAlchemy-specific N+1 patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Look for relationship access without eager loading
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Attribute):
                        # Check if accessing a relationship attribute
                        if self._is_relationship_access(inner, content):
                            # Check if parent query has eager loading
                            if not self._has_eager_loading(tree, node):
                                patterns.append({
                                    "line": inner.lineno,
                                    "type": "lazy_loading_in_loop",
                                    "orm": "sqlalchemy",
                                    "severity": "high",
                                    "fix": "Use joinedload() or selectinload()"
                                })
        
        return patterns
    
    def _detect_django_n_plus_one(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect Django ORM-specific N+1 patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check iterator for missing select_related/prefetch_related
                if isinstance(node.iter, ast.Call):
                    if self._is_django_queryset(node.iter):
                        # Check for foreign key access in loop
                        for inner in ast.walk(node):
                            if isinstance(inner, ast.Attribute):
                                if self._is_foreign_key_access(inner):
                                    if not self._has_django_optimization(node.iter):
                                        patterns.append({
                                            "line": inner.lineno,
                                            "type": "missing_select_related",
                                            "orm": "django",
                                            "severity": "high",
                                            "fix": "Add select_related() or prefetch_related()"
                                        })
        
        return patterns
    
    def _analyze_query_complexity(self) -> Dict[str, Any]:
        """Analyze complexity of database queries."""
        complexity = {
            "complex_queries": [],
            "join_analysis": defaultdict(list),
            "subquery_patterns": [],
            "aggregation_patterns": [],
            "complexity_scores": defaultdict(int)
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Analyze raw SQL complexity
                sql_queries = self._extract_sql_queries(content)
                for query in sql_queries:
                    complexity_score = self._calculate_sql_complexity(query)
                    if complexity_score > 5:
                        complexity["complex_queries"].append({
                            "file": file_key,
                            "query": query[:100] + "..." if len(query) > 100 else query,
                            "complexity_score": complexity_score,
                            "issues": self._identify_query_issues(query)
                        })
                
                # Analyze ORM query chains
                orm_chains = self._analyze_orm_query_chains(tree)
                for chain in orm_chains:
                    chain["file"] = file_key
                    if chain["chain_length"] > 5:
                        complexity["complex_queries"].append(chain)
                
                # Detect JOIN patterns
                joins = self._detect_join_patterns(tree, content)
                for join in joins:
                    complexity["join_analysis"][join["type"]].append({
                        "file": file_key,
                        "line": join["line"]
                    })
                
                # Detect subqueries
                subqueries = self._detect_subqueries(tree, content)
                complexity["subquery_patterns"].extend(subqueries)
                
            except:
                continue
        
        return complexity
    
    def _extract_sql_queries(self, content: str) -> List[str]:
        """Extract SQL queries from code."""
        queries = []
        
        # Find SQL strings (multi-line and single-line)
        sql_patterns = [
            r'"""[\s]*SELECT.*?"""',
            r"'''[\s]*SELECT.*?'''",
            r'"SELECT.*?"',
            r"'SELECT.*?'",
            r'"""[\s]*INSERT.*?"""',
            r'"""[\s]*UPDATE.*?"""',
            r'"""[\s]*DELETE.*?"""',
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            queries.extend(matches)
        
        return queries
    
    def _calculate_sql_complexity(self, query: str) -> int:
        """Calculate complexity score for SQL query."""
        complexity = 1
        query_upper = query.upper()
        
        # Count JOINs
        complexity += query_upper.count(' JOIN ') * 2
        complexity += query_upper.count(' LEFT JOIN ') * 2
        complexity += query_upper.count(' RIGHT JOIN ') * 2
        complexity += query_upper.count(' INNER JOIN ') * 2
        
        # Count subqueries
        complexity += query_upper.count('(SELECT') * 3
        
        # Count UNION operations
        complexity += query_upper.count(' UNION ') * 2
        
        # Count GROUP BY
        if ' GROUP BY ' in query_upper:
            complexity += 2
        
        # Count HAVING
        if ' HAVING ' in query_upper:
            complexity += 2
        
        # Count CASE statements
        complexity += query_upper.count(' CASE ') * 2
        
        # Count CTEs (WITH clause)
        if query_upper.startswith('WITH '):
            complexity += 3
        
        return complexity
    
    def _analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze index usage patterns."""
        index_analysis = {
            "missing_indexes": [],
            "index_hints": [],
            "full_table_scans": [],
            "index_recommendations": []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect queries without WHERE clauses
                no_where = self._detect_queries_without_where(content)
                for query in no_where:
                    index_analysis["full_table_scans"].append({
                        "file": file_key,
                        "pattern": query,
                        "risk": "high",
                        "recommendation": "Add WHERE clause or LIMIT"
                    })
                
                # Detect filter patterns that need indexes
                filter_patterns = self._detect_filter_patterns(tree)
                for pattern in filter_patterns:
                    if self._needs_index(pattern):
                        index_analysis["missing_indexes"].append({
                            "file": file_key,
                            "field": pattern["field"],
                            "operation": pattern["operation"],
                            "line": pattern["line"],
                            "recommendation": f"Consider adding index on {pattern['field']}"
                        })
                
                # Look for explicit index hints
                if "index" in content.lower():
                    index_analysis["index_hints"].append({
                        "file": file_key,
                        "has_index_references": True
                    })
                
            except:
                continue
        
        # Generate index recommendations
        index_analysis["index_recommendations"] = self._generate_index_recommendations(index_analysis)
        
        return index_analysis
    
    def _detect_queries_without_where(self, content: str) -> List[str]:
        """Detect SELECT queries without WHERE clause."""
        queries = []
        
        # Simple pattern to detect SELECT without WHERE
        pattern = r'SELECT\s+.*?\s+FROM\s+\w+(?!\s+WHERE)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        for match in matches:
            if 'LIMIT' not in match.upper():
                queries.append(match)
        
        return queries
    
    def _detect_filter_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect filtering patterns in ORM queries."""
        filter_patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["filter", "filter_by", "where"]:
                        # Extract filter fields
                        for keyword in node.keywords:
                            filter_patterns.append({
                                "field": keyword.arg,
                                "operation": "equality",
                                "line": node.lineno
                            })
        
        return filter_patterns
    
    def _analyze_transaction_patterns(self) -> Dict[str, Any]:
        """Analyze database transaction patterns."""
        transactions = {
            "transaction_blocks": [],
            "long_transactions": [],
            "nested_transactions": [],
            "commit_patterns": defaultdict(int),
            "rollback_patterns": []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect transaction blocks
                tx_blocks = self._detect_transaction_blocks(tree, content)
                for tx in tx_blocks:
                    tx["file"] = file_key
                    transactions["transaction_blocks"].append(tx)
                    
                    # Check for long transactions
                    if tx.get("statement_count", 0) > 10:
                        transactions["long_transactions"].append({
                            "file": file_key,
                            "line": tx["line"],
                            "statements": tx["statement_count"],
                            "risk": "high",
                            "recommendation": "Break into smaller transactions"
                        })
                
                # Detect nested transactions
                nested = self._detect_nested_transactions(tree)
                for n in nested:
                    n["file"] = file_key
                    transactions["nested_transactions"].append(n)
                
                # Analyze commit patterns
                commits = self._analyze_commit_patterns(tree)
                for pattern, count in commits.items():
                    transactions["commit_patterns"][pattern] += count
                
                # Detect rollback patterns
                rollbacks = self._detect_rollback_patterns(tree, content)
                transactions["rollback_patterns"].extend(rollbacks)
                
            except:
                continue
        
        return transactions
    
    def _detect_transaction_blocks(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect transaction block patterns."""
        tx_blocks = []
        
        for node in ast.walk(tree):
            # Detect context manager transactions
            if isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Attribute):
                            if "transaction" in item.context_expr.func.attr.lower():
                                tx_blocks.append({
                                    "line": node.lineno,
                                    "type": "context_manager",
                                    "statement_count": len(node.body)
                                })
            
            # Detect explicit begin/commit patterns
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["begin", "begin_nested", "start_transaction"]:
                        tx_blocks.append({
                            "line": node.lineno,
                            "type": "explicit",
                            "statement_count": self._count_statements_until_commit(tree, node)
                        })
        
        return tx_blocks
    
    def _analyze_connection_pooling(self) -> Dict[str, Any]:
        """Analyze database connection pooling patterns."""
        pooling = {
            "pool_configurations": [],
            "connection_leaks": [],
            "pool_exhaustion_risks": [],
            "connection_patterns": defaultdict(int)
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect pool configurations
                pool_configs = self._detect_pool_configurations(tree, content)
                for config in pool_configs:
                    config["file"] = file_key
                    pooling["pool_configurations"].append(config)
                
                # Detect connection leaks
                leaks = self._detect_connection_leaks(tree)
                for leak in leaks:
                    leak["file"] = file_key
                    pooling["connection_leaks"].append(leak)
                
                # Detect pool exhaustion risks
                exhaustion = self._detect_pool_exhaustion_risks(tree, content)
                for risk in exhaustion:
                    risk["file"] = file_key
                    pooling["pool_exhaustion_risks"].append(risk)
                
                # Analyze connection patterns
                patterns = self._analyze_connection_patterns(tree)
                for pattern, count in patterns.items():
                    pooling["connection_patterns"][pattern] += count
                
            except:
                continue
        
        return pooling
    
    def _detect_pool_configurations(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect connection pool configurations."""
        configurations = []
        
        # Look for SQLAlchemy pool configurations
        if "create_engine" in content:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "create_engine":
                        pool_config = {}
                        for keyword in node.keywords:
                            if keyword.arg in ["pool_size", "max_overflow", "pool_timeout", "pool_recycle"]:
                                if isinstance(keyword.value, ast.Constant):
                                    pool_config[keyword.arg] = keyword.value.value
                        
                        if pool_config:
                            configurations.append({
                                "line": node.lineno,
                                "type": "sqlalchemy",
                                "config": pool_config,
                                "analysis": self._analyze_pool_config(pool_config)
                            })
        
        # Look for Django database configurations
        if "DATABASES" in content:
            configurations.append({
                "type": "django",
                "detected": True,
                "note": "Check settings.py for CONN_MAX_AGE configuration"
            })
        
        return configurations
    
    def _detect_connection_leaks(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect potential connection leaks."""
        leaks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for connect() without close()
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["connect", "connection", "get_connection"]:
                        # Check if connection is properly closed
                        if not self._has_connection_cleanup(tree, node):
                            leaks.append({
                                "line": node.lineno,
                                "type": "unclosed_connection",
                                "severity": "high",
                                "fix": "Use context manager or ensure close() is called"
                            })
        
        return leaks
    
    def _analyze_orm_patterns(self) -> Dict[str, Any]:
        """Analyze ORM-specific patterns and practices."""
        orm_analysis = {
            "detected_orms": [],
            "query_patterns": defaultdict(list),
            "bulk_operations": [],
            "lazy_loading": [],
            "eager_loading": [],
            "orm_best_practices": {}
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect which ORMs are used
                detected = self._detect_orm_usage(content)
                orm_analysis["detected_orms"].extend(detected)
                
                # Analyze query patterns
                patterns = self._analyze_query_patterns(tree, content)
                for pattern in patterns:
                    pattern["file"] = file_key
                    orm_analysis["query_patterns"][pattern["type"]].append(pattern)
                
                # Detect bulk operations
                bulk_ops = self._detect_bulk_operations(tree, content)
                for op in bulk_ops:
                    op["file"] = file_key
                    orm_analysis["bulk_operations"].append(op)
                
                # Detect lazy vs eager loading
                lazy, eager = self._detect_loading_strategies(tree, content)
                orm_analysis["lazy_loading"].extend(lazy)
                orm_analysis["eager_loading"].extend(eager)
                
            except:
                continue
        
        # Analyze best practices compliance
        orm_analysis["orm_best_practices"] = self._check_orm_best_practices(orm_analysis)
        
        return orm_analysis
    
    def _detect_orm_usage(self, content: str) -> List[str]:
        """Detect which ORMs are being used."""
        orms = []
        
        if "sqlalchemy" in content.lower():
            orms.append("SQLAlchemy")
        if "django.db" in content or "models.Model" in content:
            orms.append("Django ORM")
        if "peewee" in content.lower():
            orms.append("Peewee")
        if "tortoise" in content.lower():
            orms.append("Tortoise ORM")
        
        return orms
    
    def _analyze_raw_sql(self) -> Dict[str, Any]:
        """Analyze raw SQL usage."""
        raw_sql = {
            "raw_queries": [],
            "sql_injection_risks": [],
            "parameterization": defaultdict(int),
            "sql_quality_issues": []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Extract raw SQL queries
                queries = self._extract_sql_queries(content)
                for query in queries:
                    raw_sql["raw_queries"].append({
                        "file": file_key,
                        "query_preview": query[:100],
                        "complexity": self._calculate_sql_complexity(query)
                    })
                
                # Check for SQL injection risks
                injection_risks = self._detect_sql_injection_risks(tree, content)
                for risk in injection_risks:
                    risk["file"] = file_key
                    raw_sql["sql_injection_risks"].append(risk)
                
                # Check parameterization
                param_usage = self._check_query_parameterization(tree, content)
                for pattern, count in param_usage.items():
                    raw_sql["parameterization"][pattern] += count
                
                # Check SQL quality
                quality_issues = self._check_sql_quality(queries)
                raw_sql["sql_quality_issues"].extend(quality_issues)
                
            except:
                continue
        
        return raw_sql
    
    def _detect_sql_injection_risks(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect potential SQL injection vulnerabilities."""
        risks = []
        
        for node in ast.walk(tree):
            # Check for string formatting in SQL queries
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Mod):  # % formatting
                    if isinstance(node.left, ast.Str):
                        if any(keyword in node.left.s.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                            risks.append({
                                "line": node.lineno,
                                "type": "string_formatting",
                                "severity": "critical",
                                "pattern": "SQL % formatting",
                                "fix": "Use parameterized queries"
                            })
            
            # Check for f-strings with SQL
            elif isinstance(node, ast.JoinedStr):
                # Check if it contains SQL keywords
                sql_in_fstring = False
                for value in node.values:
                    if isinstance(value, ast.Constant):
                        if any(keyword in str(value.value).upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                            sql_in_fstring = True
                
                if sql_in_fstring:
                    risks.append({
                        "line": node.lineno,
                        "type": "f_string_sql",
                        "severity": "critical",
                        "pattern": "SQL in f-string",
                        "fix": "Use parameterized queries"
                    })
        
        return risks
    
    def _identify_optimization_opportunities(self) -> Dict[str, Any]:
        """Identify database optimization opportunities."""
        optimizations = {
            "query_optimizations": [],
            "index_suggestions": [],
            "caching_opportunities": [],
            "batch_processing": [],
            "recommendations": []
        }
        
        # Analyze all collected data
        n_plus_one = self._detect_n_plus_one_queries()
        complexity = self._analyze_query_complexity()
        transactions = self._analyze_transaction_patterns()
        
        # Generate query optimizations
        if n_plus_one["detected_patterns"]:
            optimizations["query_optimizations"].append({
                "type": "n_plus_one",
                "count": len(n_plus_one["detected_patterns"]),
                "priority": "high",
                "recommendation": "Use eager loading to reduce query count"
            })
        
        # Generate index suggestions
        if complexity["complex_queries"]:
            for query in complexity["complex_queries"]:
                if query["complexity_score"] > 10:
                    optimizations["index_suggestions"].append({
                        "query_complexity": query["complexity_score"],
                        "recommendation": "Consider adding indexes for complex queries"
                    })
        
        # Identify caching opportunities
        optimizations["caching_opportunities"] = self._identify_caching_opportunities()
        
        # Identify batch processing opportunities
        optimizations["batch_processing"] = self._identify_batch_opportunities()
        
        # Generate overall recommendations
        optimizations["recommendations"] = self._generate_db_recommendations(optimizations)
        
        return optimizations
    
    def _identify_caching_opportunities(self) -> List[Dict[str, Any]]:
        """Identify queries that could benefit from caching."""
        opportunities = []
        
        # Look for repeated queries pattern
        opportunities.append({
            "pattern": "repeated_queries",
            "recommendation": "Implement query result caching for frequently accessed data"
        })
        
        # Look for read-heavy patterns
        opportunities.append({
            "pattern": "read_heavy",
            "recommendation": "Consider Redis or Memcached for read-heavy workloads"
        })
        
        return opportunities
    
    def _identify_batch_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for batch processing."""
        opportunities = []
        
        opportunities.append({
            "pattern": "multiple_inserts",
            "recommendation": "Use bulk_insert_mappings() or executemany() for batch inserts"
        })
        
        opportunities.append({
            "pattern": "multiple_updates",
            "recommendation": "Use bulk_update_mappings() for batch updates"
        })
        
        return opportunities
    
    def _calculate_database_metrics(self) -> Dict[str, Any]:
        """Calculate overall database metrics."""
        n_plus_one = self._detect_n_plus_one_queries()
        complexity = self._analyze_query_complexity()
        pooling = self._analyze_connection_pooling()
        raw_sql = self._analyze_raw_sql()
        
        metrics = {
            "n_plus_one_count": len(n_plus_one["detected_patterns"]),
            "complex_query_count": len(complexity["complex_queries"]),
            "connection_leak_count": len(pooling["connection_leaks"]),
            "sql_injection_risk_count": len(raw_sql["sql_injection_risks"]),
            "database_health_score": 0,
            "critical_issues": [],
            "performance_impact": "low"
        }
        
        # Calculate health score
        health_score = 100
        health_score -= metrics["n_plus_one_count"] * 5
        health_score -= metrics["complex_query_count"] * 2
        health_score -= metrics["connection_leak_count"] * 10
        health_score -= metrics["sql_injection_risk_count"] * 20
        metrics["database_health_score"] = max(0, health_score)
        
        # Identify critical issues
        if metrics["sql_injection_risk_count"] > 0:
            metrics["critical_issues"].append("SQL injection vulnerabilities detected")
        
        if metrics["connection_leak_count"] > 0:
            metrics["critical_issues"].append("Connection leaks detected")
        
        if metrics["n_plus_one_count"] > 5:
            metrics["critical_issues"].append("Multiple N+1 query problems")
        
        # Assess performance impact
        if metrics["n_plus_one_count"] > 10 or metrics["complex_query_count"] > 5:
            metrics["performance_impact"] = "high"
        elif metrics["n_plus_one_count"] > 5 or metrics["complex_query_count"] > 2:
            metrics["performance_impact"] = "medium"
        
        return metrics
    
    # Helper methods
    def _assess_query_severity(self, loop_node: ast.AST, query_node: ast.AST) -> str:
        """Assess severity of query in loop."""
        # Check loop size/complexity
        loop_size = len(list(ast.walk(loop_node)))
        
        if loop_size > 50:
            return "critical"
        elif loop_size > 20:
            return "high"
        elif loop_size > 10:
            return "medium"
        return "low"
    
    def _identify_query_pattern(self, node: ast.Call) -> str:
        """Identify the pattern of database query."""
        if isinstance(node.func, ast.Attribute):
            return f"{node.func.attr}_query"
        return "unknown_query"
    
    def _get_n_plus_one_fix(self, node: ast.Call, content: str) -> str:
        """Get recommended fix for N+1 query."""
        if "sqlalchemy" in content.lower():
            return "Use joinedload() or selectinload() for eager loading"
        elif "django" in content:
            return "Use select_related() or prefetch_related()"
        return "Use eager loading or batch queries"
    
    def _is_relationship_access(self, node: ast.Attribute, content: str) -> bool:
        """Check if accessing a database relationship."""
        # Simplified check - looks for common relationship patterns
        relationship_indicators = ["user", "author", "items", "children", "parent", "related"]
        return any(ind in node.attr.lower() for ind in relationship_indicators)
    
    def _has_eager_loading(self, tree: ast.AST, loop_node: ast.AST) -> bool:
        """Check if query has eager loading configured."""
        # Look for eager loading patterns before the loop
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["joinedload", "subqueryload", "selectinload"]:
                        return True
        return False
    
    def _is_django_queryset(self, node: ast.Call) -> bool:
        """Check if node is a Django queryset."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in ["all", "filter", "exclude", "get"]
        return False
    
    def _is_foreign_key_access(self, node: ast.Attribute) -> bool:
        """Check if accessing a foreign key relationship."""
        # Common foreign key patterns
        fk_patterns = ["_id", "user", "author", "category", "group"]
        return any(pattern in node.attr.lower() for pattern in fk_patterns)
    
    def _has_django_optimization(self, node: ast.Call) -> bool:
        """Check if Django query has optimization."""
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                if isinstance(inner.func, ast.Attribute):
                    if inner.func.attr in ["select_related", "prefetch_related"]:
                        return True
        return False
    
    def _identify_query_issues(self, query: str) -> List[str]:
        """Identify issues in SQL query."""
        issues = []
        query_upper = query.upper()
        
        if "SELECT *" in query_upper:
            issues.append("Uses SELECT * (specify columns)")
        
        if " OR " in query_upper and "WHERE" in query_upper:
            issues.append("Complex OR conditions (consider indexes)")
        
        if "NOT IN" in query_upper:
            issues.append("Uses NOT IN (consider NOT EXISTS)")
        
        if "LIKE '%'" in query_upper or "LIKE '%%" in query_upper:
            issues.append("Leading wildcard in LIKE (cannot use index)")
        
        return issues
    
    def _analyze_orm_query_chains(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze chained ORM query methods."""
        chains = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain_length = 0
                current = node
                
                while isinstance(current, ast.Attribute):
                    if current.attr in ["filter", "exclude", "annotate", "order_by", "values"]:
                        chain_length += 1
                    current = current.value if isinstance(current.value, ast.Attribute) else None
                
                if chain_length > 3:
                    chains.append({
                        "line": node.lineno if hasattr(node, 'lineno') else 0,
                        "chain_length": chain_length,
                        "complexity": "high" if chain_length > 5 else "medium"
                    })
        
        return chains
    
    def _detect_join_patterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect JOIN patterns in queries."""
        joins = []
        
        # SQL JOINs
        for join_type in ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN"]:
            if join_type in content.upper():
                joins.append({"type": join_type.lower().replace(" ", "_")})
        
        # ORM JOINs
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["join", "outerjoin"]:
                        joins.append({
                            "type": "orm_join",
                            "line": node.lineno
                        })
        
        return joins
    
    def _detect_subqueries(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect subquery patterns."""
        subqueries = []
        
        # SQL subqueries
        if "(SELECT" in content.upper():
            count = content.upper().count("(SELECT")
            subqueries.append({
                "type": "sql_subquery",
                "count": count,
                "risk": "high" if count > 3 else "medium"
            })
        
        # ORM subqueries
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["subquery", "as_scalar"]:
                        subqueries.append({
                            "type": "orm_subquery",
                            "line": node.lineno
                        })
        
        return subqueries
    
    def _needs_index(self, pattern: Dict[str, Any]) -> bool:
        """Check if a field pattern needs an index."""
        # Common fields that should be indexed
        index_fields = ["id", "user_id", "created_at", "updated_at", "status", "type"]
        return pattern.get("field") in index_fields
    
    def _generate_index_recommendations(self, index_analysis: Dict[str, Any]) -> List[str]:
        """Generate index recommendations."""
        recommendations = []
        
        if index_analysis["full_table_scans"]:
            recommendations.append("Add WHERE clauses or LIMIT to prevent full table scans")
        
        if index_analysis["missing_indexes"]:
            recommendations.append("Create indexes on frequently queried fields")
        
        return recommendations
    
    def _detect_nested_transactions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect nested transaction patterns."""
        nested = []
        
        # Look for nested context managers with transaction
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.With) and inner != node:
                        # Check if both are transaction-related
                        nested.append({
                            "line": node.lineno,
                            "type": "nested_transaction",
                            "risk": "medium"
                        })
        
        return nested
    
    def _analyze_commit_patterns(self, tree: ast.AST) -> Dict[str, int]:
        """Analyze commit patterns in code."""
        patterns = defaultdict(int)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "commit":
                        patterns["explicit_commit"] += 1
                    elif node.func.attr == "flush":
                        patterns["flush"] += 1
        
        return patterns
    
    def _detect_rollback_patterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect rollback patterns."""
        rollbacks = []
        
        if "rollback" in content.lower():
            rollbacks.append({"has_rollback_handling": True})
        
        return rollbacks
    
    def _count_statements_until_commit(self, tree: ast.AST, begin_node: ast.Call) -> int:
        """Count statements between begin and commit."""
        # Simplified implementation
        return 5  # Placeholder
    
    def _detect_pool_exhaustion_risks(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect connection pool exhaustion risks."""
        risks = []
        
        # Look for patterns that might exhaust pool
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for connection creation in loops
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        if isinstance(inner.func, ast.Attribute):
                            if inner.func.attr in ["connect", "connection"]:
                                risks.append({
                                    "line": node.lineno,
                                    "pattern": "connection_in_loop",
                                    "risk": "high"
                                })
        
        return risks
    
    def _analyze_connection_patterns(self, tree: ast.AST) -> Dict[str, int]:
        """Analyze database connection patterns."""
        patterns = defaultdict(int)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                patterns["context_manager"] += 1
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "close":
                        patterns["explicit_close"] += 1
        
        return patterns
    
    def _analyze_pool_config(self, config: Dict[str, Any]) -> str:
        """Analyze pool configuration for issues."""
        issues = []
        
        pool_size = config.get("pool_size", 5)
        if pool_size < 5:
            issues.append("Pool size might be too small")
        elif pool_size > 20:
            issues.append("Pool size might be too large")
        
        return "; ".join(issues) if issues else "Configuration looks good"
    
    def _has_connection_cleanup(self, tree: ast.AST, connect_node: ast.Call) -> bool:
        """Check if connection has proper cleanup."""
        # Check for close() call or context manager
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                return True
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "close":
                        return True
        return False
    
    def _analyze_query_patterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Analyze general query patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["all", "first", "one"]:
                        patterns.append({
                            "type": "data_fetching",
                            "method": node.func.attr,
                            "line": node.lineno
                        })
        
        return patterns
    
    def _detect_bulk_operations(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect bulk operation patterns."""
        bulk_ops = []
        
        # SQLAlchemy bulk operations
        if "bulk_insert_mappings" in content or "bulk_save_objects" in content:
            bulk_ops.append({"type": "sqlalchemy_bulk", "detected": True})
        
        # Django bulk operations
        if "bulk_create" in content or "bulk_update" in content:
            bulk_ops.append({"type": "django_bulk", "detected": True})
        
        return bulk_ops
    
    def _detect_loading_strategies(self, tree: ast.AST, content: str) -> Tuple[List, List]:
        """Detect lazy and eager loading strategies."""
        lazy = []
        eager = []
        
        # SQLAlchemy loading
        if "lazyload" in content:
            lazy.append({"type": "sqlalchemy_lazy"})
        if "joinedload" in content or "subqueryload" in content:
            eager.append({"type": "sqlalchemy_eager"})
        
        # Django loading
        if "select_related" in content or "prefetch_related" in content:
            eager.append({"type": "django_eager"})
        
        return lazy, eager
    
    def _check_orm_best_practices(self, orm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check ORM best practices compliance."""
        best_practices = {
            "uses_eager_loading": len(orm_analysis["eager_loading"]) > 0,
            "uses_bulk_operations": len(orm_analysis["bulk_operations"]) > 0,
            "avoids_n_plus_one": True,  # Will be set based on other analysis
            "score": 0
        }
        
        score = 0
        if best_practices["uses_eager_loading"]:
            score += 30
        if best_practices["uses_bulk_operations"]:
            score += 30
        
        best_practices["score"] = score
        
        return best_practices
    
    def _check_query_parameterization(self, tree: ast.AST, content: str) -> Dict[str, int]:
        """Check query parameterization usage."""
        patterns = defaultdict(int)
        
        # Check for parameterized queries
        if "?" in content or "%s" in content:
            patterns["parameterized"] += content.count("?") + content.count("%s")
        
        # Check for named parameters
        if ":param" in content or "%(param)" in content:
            patterns["named_params"] += 1
        
        return patterns
    
    def _check_sql_quality(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Check SQL query quality issues."""
        issues = []
        
        for query in queries:
            query_issues = self._identify_query_issues(query)
            if query_issues:
                issues.append({
                    "query_preview": query[:50],
                    "issues": query_issues
                })
        
        return issues
    
    def _identify_n_plus_one_hotspots(self, n_plus_one_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify hotspots with high N+1 query risk."""
        hotspots = []
        
        # Group by file
        file_counts = defaultdict(int)
        for pattern in n_plus_one_data["detected_patterns"]:
            if "file" in pattern:
                file_counts[pattern["file"]] += 1
        
        # Identify hotspots
        for file, count in file_counts.items():
            if count > 2:
                hotspots.append({
                    "file": file,
                    "n_plus_one_count": count,
                    "risk_level": "critical" if count > 5 else "high"
                })
        
        return hotspots
    
    def _generate_db_recommendations(self, optimizations: Dict[str, Any]) -> List[str]:
        """Generate database optimization recommendations."""
        recommendations = []
        
        if optimizations["query_optimizations"]:
            recommendations.append("Fix N+1 query problems using eager loading")
        
        if optimizations["index_suggestions"]:
            recommendations.append("Add database indexes for complex queries")
        
        if optimizations["caching_opportunities"]:
            recommendations.append("Implement query result caching")
        
        if optimizations["batch_processing"]:
            recommendations.append("Use batch operations for bulk data processing")
        
        return recommendations