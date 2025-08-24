#!/usr/bin/env python3
"""
Agent C - Database Relationship Analysis Tool (Hours 19-21)
Comprehensive database schema, connection, and relationship analysis
"""

import os
import ast
import json
import logging
import argparse
import time
import re
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import sqlite3


@dataclass
class DatabaseConnection:
    """Database connection data structure"""
    connection_type: str
    host: str
    port: int
    database_name: str
    username: str
    file_path: str
    line_number: int
    connection_string: str


@dataclass
class DatabaseTable:
    """Database table structure"""
    name: str
    columns: List[Dict[str, Any]]
    indexes: List[str]
    foreign_keys: List[Dict[str, str]]
    primary_key: List[str]
    schema: str
    file_path: str


@dataclass
class DatabaseQuery:
    """Database query analysis"""
    query_type: str
    tables_accessed: List[str]
    operation: str
    file_path: str
    line_number: int
    complexity_score: float
    raw_query: str


@dataclass
class DatabaseRelationship:
    """Database relationship mapping"""
    source_table: str
    target_table: str
    relationship_type: str
    foreign_key: str
    strength: float


class DatabaseAnalyzer(ast.NodeVisitor):
    """AST visitor for database analysis"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.connections = []
        self.queries = []
        self.tables = []
        self.orm_models = []
        
    def visit_Call(self, node):
        """Visit function calls for database operations"""
        # Database connection patterns
        self._analyze_db_connection(node)
        
        # SQL query patterns
        self._analyze_sql_query(node)
        
        # ORM operations
        self._analyze_orm_operations(node)
        
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Visit class definitions for ORM models"""
        self._analyze_orm_model(node)
        self.generic_visit(node)
        
    def _analyze_db_connection(self, node):
        """Analyze database connection calls"""
        connection_patterns = {
            'sqlite3.connect': 'sqlite',
            'psycopg2.connect': 'postgresql',
            'mysql.connector.connect': 'mysql',
            'pymongo.MongoClient': 'mongodb',
            'redis.Redis': 'redis',
            'cx_Oracle.connect': 'oracle',
            'pyodbc.connect': 'mssql',
            'create_engine': 'sqlalchemy'
        }
        
        call_name = self._get_call_name(node)
        
        for pattern, db_type in connection_patterns.items():
            if pattern in call_name:
                connection = self._extract_connection_details(node, db_type)
                if connection:
                    self.connections.append(connection)
                break
                
    def _analyze_sql_query(self, node):
        """Analyze SQL query patterns"""
        query_patterns = [
            'execute', 'query', 'fetchall', 'fetchone', 'cursor',
            'raw', 'sql', 'select', 'insert', 'update', 'delete'
        ]
        
        call_name = self._get_call_name(node).lower()
        
        if any(pattern in call_name for pattern in query_patterns):
            query = self._extract_query_details(node)
            if query:
                self.queries.append(query)
                
    def _analyze_orm_operations(self, node):
        """Analyze ORM operations"""
        orm_patterns = [
            'filter', 'get', 'create', 'save', 'delete', 'update',
            'objects.', 'session.', 'Model.', 'query.'
        ]
        
        call_name = self._get_call_name(node)
        
        if any(pattern in call_name for pattern in orm_patterns):
            query = self._extract_orm_query(node)
            if query:
                self.queries.append(query)
                
    def _analyze_orm_model(self, node):
        """Analyze ORM model definitions"""
        # Check for Django models
        if self._is_django_model(node):
            table = self._extract_django_model(node)
            if table:
                self.tables.append(table)
                
        # Check for SQLAlchemy models
        elif self._is_sqlalchemy_model(node):
            table = self._extract_sqlalchemy_model(node)
            if table:
                self.tables.append(table)
                
    def _get_call_name(self, node):
        """Extract call name from AST node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attr_chain(node.func)
        return ""
        
    def _get_attr_chain(self, node):
        """Get attribute chain"""
        if isinstance(node, ast.Attribute):
            base = self._get_attr_chain(node.value) if hasattr(node.value, 'attr') else getattr(node.value, 'id', '')
            return f"{base}.{node.attr}" if base else node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return ""
        
    def _extract_connection_details(self, node, db_type):
        """Extract database connection details"""
        try:
            connection_string = ""
            host = "localhost"
            port = 0
            database_name = ""
            username = ""
            
            # Extract from arguments
            for i, arg in enumerate(node.args):
                if isinstance(arg, ast.Constant):
                    if i == 0:
                        connection_string = str(arg.value)
                        if db_type == 'sqlite':
                            database_name = arg.value
                            
            # Extract from keywords
            for keyword in node.keywords:
                if keyword.arg == 'host' and isinstance(keyword.value, ast.Constant):
                    host = keyword.value.value
                elif keyword.arg == 'port' and isinstance(keyword.value, ast.Constant):
                    port = keyword.value.value
                elif keyword.arg == 'database' and isinstance(keyword.value, ast.Constant):
                    database_name = keyword.value.value
                elif keyword.arg == 'user' and isinstance(keyword.value, ast.Constant):
                    username = keyword.value.value
                    
            return DatabaseConnection(
                connection_type=db_type,
                host=host,
                port=port or self._get_default_port(db_type),
                database_name=database_name,
                username=username,
                file_path=self.file_path,
                line_number=node.lineno,
                connection_string=connection_string
            )
            
        except Exception as e:
            logging.warning(f"Error extracting connection details: {e}")
            return None
            
    def _extract_query_details(self, node):
        """Extract SQL query details"""
        try:
            raw_query = ""
            query_type = "unknown"
            operation = "read"
            
            # Extract query string
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    raw_query = arg.value
                    break
                elif isinstance(arg, ast.Str):
                    raw_query = arg.s
                    break
                    
            if raw_query:
                query_type = self._detect_query_type(raw_query)
                operation = self._detect_operation_type(raw_query)
                tables = self._extract_tables_from_query(raw_query)
                complexity = self._calculate_query_complexity(raw_query)
                
                return DatabaseQuery(
                    query_type=query_type,
                    tables_accessed=tables,
                    operation=operation,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    complexity_score=complexity,
                    raw_query=raw_query
                )
                
        except Exception as e:
            logging.warning(f"Error extracting query details: {e}")
            
        return None
        
    def _extract_orm_query(self, node):
        """Extract ORM query details"""
        try:
            call_name = self._get_call_name(node)
            operation = "read"
            tables = []
            
            # Determine operation type
            if any(op in call_name.lower() for op in ['create', 'save', 'insert']):
                operation = "write"
            elif any(op in call_name.lower() for op in ['update', 'modify']):
                operation = "update"
            elif any(op in call_name.lower() for op in ['delete', 'remove']):
                operation = "delete"
                
            # Extract model/table name
            if 'objects.' in call_name:
                model_name = call_name.split('.objects.')[0].split('.')[-1]
                tables.append(model_name)
            elif 'session.' in call_name:
                # Look for model class in arguments
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        tables.append(arg.id)
                        
            return DatabaseQuery(
                query_type="orm",
                tables_accessed=tables,
                operation=operation,
                file_path=self.file_path,
                line_number=node.lineno,
                complexity_score=1.0,
                raw_query=call_name
            )
            
        except Exception as e:
            logging.warning(f"Error extracting ORM query: {e}")
            
        return None
        
    def _is_django_model(self, node):
        """Check if class is Django model"""
        for base in node.bases:
            if isinstance(base, ast.Attribute):
                if 'models.Model' in self._get_attr_chain(base):
                    return True
        return False
        
    def _is_sqlalchemy_model(self, node):
        """Check if class is SQLAlchemy model"""
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id in ['Base', 'Model', 'DeclarativeBase']:
                    return True
        return False
        
    def _extract_django_model(self, node):
        """Extract Django model structure"""
        try:
            columns = []
            foreign_keys = []
            indexes = []
            
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            field_info = self._analyze_django_field(item.value)
                            if field_info:
                                columns.append({
                                    'name': target.id,
                                    'type': field_info['type'],
                                    'constraints': field_info['constraints']
                                })
                                
                                if field_info['type'] == 'ForeignKey':
                                    foreign_keys.append({
                                        'field': target.id,
                                        'references': field_info.get('references', '')
                                    })
                                    
            return DatabaseTable(
                name=node.name,
                columns=columns,
                indexes=indexes,
                foreign_keys=foreign_keys,
                primary_key=['id'],  # Default Django primary key
                schema='default',
                file_path=self.file_path
            )
            
        except Exception as e:
            logging.warning(f"Error extracting Django model: {e}")
            
        return None
        
    def _extract_sqlalchemy_model(self, node):
        """Extract SQLAlchemy model structure"""
        try:
            columns = []
            foreign_keys = []
            indexes = []
            primary_key = []
            
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            field_info = self._analyze_sqlalchemy_field(item.value)
                            if field_info:
                                columns.append({
                                    'name': target.id,
                                    'type': field_info['type'],
                                    'constraints': field_info['constraints']
                                })
                                
                                if field_info.get('is_primary_key'):
                                    primary_key.append(target.id)
                                    
                                if field_info.get('foreign_key'):
                                    foreign_keys.append({
                                        'field': target.id,
                                        'references': field_info['foreign_key']
                                    })
                                    
            return DatabaseTable(
                name=node.name,
                columns=columns,
                indexes=indexes,
                foreign_keys=foreign_keys,
                primary_key=primary_key,
                schema='default',
                file_path=self.file_path
            )
            
        except Exception as e:
            logging.warning(f"Error extracting SQLAlchemy model: {e}")
            
        return None
        
    def _analyze_django_field(self, node):
        """Analyze Django field definition"""
        if isinstance(node, ast.Call):
            call_name = self._get_call_name(node)
            if 'Field' in call_name or call_name in ['CharField', 'IntegerField', 'ForeignKey', 'DateTimeField']:
                field_type = call_name.split('.')[-1]
                constraints = []
                references = ""
                
                # Extract constraints from keywords
                for keyword in node.keywords:
                    constraints.append(f"{keyword.arg}={self._extract_value(keyword.value)}")
                    
                # Extract ForeignKey reference
                if field_type == 'ForeignKey' and node.args:
                    references = self._extract_value(node.args[0])
                    
                return {
                    'type': field_type,
                    'constraints': constraints,
                    'references': references
                }
                
        return None
        
    def _analyze_sqlalchemy_field(self, node):
        """Analyze SQLAlchemy field definition"""
        if isinstance(node, ast.Call):
            call_name = self._get_call_name(node)
            if 'Column' in call_name:
                field_type = "Column"
                constraints = []
                is_primary_key = False
                foreign_key = ""
                
                # Extract type and constraints from arguments
                for arg in node.args:
                    arg_name = self._get_call_name(arg) if isinstance(arg, ast.Call) else str(arg)
                    if 'Integer' in arg_name or 'String' in arg_name or 'DateTime' in arg_name:
                        field_type = arg_name
                    elif 'ForeignKey' in arg_name:
                        foreign_key = self._extract_value(arg.args[0]) if arg.args else ""
                        
                # Check for primary_key constraint
                for keyword in node.keywords:
                    if keyword.arg == 'primary_key':
                        is_primary_key = True
                    constraints.append(f"{keyword.arg}={self._extract_value(keyword.value)}")
                    
                return {
                    'type': field_type,
                    'constraints': constraints,
                    'is_primary_key': is_primary_key,
                    'foreign_key': foreign_key
                }
                
        return None
        
    def _extract_value(self, node):
        """Extract value from AST node"""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Name):
            return node.id
        return ""
        
    def _get_default_port(self, db_type):
        """Get default port for database type"""
        ports = {
            'postgresql': 5432,
            'mysql': 3306,
            'mongodb': 27017,
            'redis': 6379,
            'oracle': 1521,
            'mssql': 1433
        }
        return ports.get(db_type, 0)
        
    def _detect_query_type(self, query):
        """Detect SQL query type"""
        query_lower = query.lower().strip()
        if query_lower.startswith('select'):
            return 'select'
        elif query_lower.startswith('insert'):
            return 'insert'
        elif query_lower.startswith('update'):
            return 'update'
        elif query_lower.startswith('delete'):
            return 'delete'
        elif query_lower.startswith('create'):
            return 'create'
        elif query_lower.startswith('drop'):
            return 'drop'
        elif query_lower.startswith('alter'):
            return 'alter'
        return 'unknown'
        
    def _detect_operation_type(self, query):
        """Detect operation type"""
        query_lower = query.lower()
        if any(op in query_lower for op in ['select', 'show', 'describe']):
            return 'read'
        elif any(op in query_lower for op in ['insert', 'create']):
            return 'write'
        elif 'update' in query_lower:
            return 'update'
        elif 'delete' in query_lower:
            return 'delete'
        return 'unknown'
        
    def _extract_tables_from_query(self, query):
        """Extract table names from SQL query"""
        tables = []
        
        # Simple regex patterns for table extraction
        patterns = [
            r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'INSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'DELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            tables.extend(matches)
            
        return list(set(tables))
        
    def _calculate_query_complexity(self, query):
        """Calculate query complexity score"""
        complexity = 1.0
        query_lower = query.lower()
        
        # Add complexity for joins
        complexity += query_lower.count('join') * 0.5
        
        # Add complexity for subqueries
        complexity += query_lower.count('select') * 0.3
        
        # Add complexity for aggregations
        complexity += len(re.findall(r'(count|sum|avg|min|max)\s*\(', query_lower)) * 0.2
        
        # Add complexity for conditions
        complexity += query_lower.count('where') * 0.1
        complexity += query_lower.count('and') * 0.05
        complexity += query_lower.count('or') * 0.05
        
        return round(complexity, 2)


class DatabaseRelationshipAnalyzer:
    """Main database relationship analysis tool"""
    
    def __init__(self, root_dir: str, output_file: str):
        self.root_dir = Path(root_dir)
        self.output_file = output_file
        self.connections = []
        self.queries = []
        self.tables = []
        self.relationships = []
        self.statistics = {
            'total_connections': 0,
            'total_queries': 0,
            'total_tables': 0,
            'total_relationships': 0,
            'database_types': set(),
            'files_analyzed': 0,
            'query_complexity_avg': 0.0,
            'security_issues': [],
            'performance_issues': []
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def analyze_codebase(self):
        """Analyze entire codebase for database relationships"""
        print("Agent C - Database Relationship Analysis (Hours 19-21)")
        print(f"Analyzing: {self.root_dir}")
        print(f"Output: {self.output_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        self.logger.info(f"Starting database relationship analysis for {self.root_dir}")
        
        # Find Python files
        python_files = list(self.root_dir.rglob("*.py"))
        self.logger.info(f"Analyzing database relationships in {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
                self.statistics['files_analyzed'] += 1
                
                if self.statistics['files_analyzed'] % 100 == 0:
                    print(f"   Processed {self.statistics['files_analyzed']} files...")
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing {file_path}: {e}")
                
        # Analyze relationships
        self._analyze_table_relationships()
        
        # Analyze schema files
        self._analyze_schema_files()
        
        # Security analysis
        self._analyze_security_issues()
        
        # Performance analysis
        self._analyze_performance_issues()
        
        duration = time.time() - start_time
        
        # Update statistics
        self._update_statistics()
        
        self._print_results(duration)
        self._save_results()
        
        self.logger.info(f"Database relationship analysis completed in {duration:.2f} seconds")
        self.logger.info(f"Database relationship analysis report saved to {self.output_file}")
        
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Analyze with DatabaseAnalyzer
            analyzer = DatabaseAnalyzer(str(file_path))
            analyzer.visit(tree)
            
            # Add results
            self.connections.extend(analyzer.connections)
            self.queries.extend(analyzer.queries)
            self.tables.extend(analyzer.tables)
            
        except SyntaxError:
            self.logger.warning(f"Syntax error in {file_path}, skipping")
        except Exception as e:
            self.logger.warning(f"Error parsing {file_path}: {e}")
            
    def _analyze_table_relationships(self):
        """Analyze relationships between tables"""
        # Create relationships from foreign keys
        for table in self.tables:
            for fk in table.foreign_keys:
                relationship = DatabaseRelationship(
                    source_table=table.name,
                    target_table=fk['references'],
                    relationship_type='foreign_key',
                    foreign_key=fk['field'],
                    strength=0.9
                )
                self.relationships.append(relationship)
                
        # Infer relationships from queries
        for query in self.queries:
            if len(query.tables_accessed) > 1:
                for i, table1 in enumerate(query.tables_accessed):
                    for table2 in query.tables_accessed[i+1:]:
                        relationship = DatabaseRelationship(
                            source_table=table1,
                            target_table=table2,
                            relationship_type='query_join',
                            foreign_key='',
                            strength=0.6
                        )
                        self.relationships.append(relationship)
                        
    def _analyze_schema_files(self):
        """Analyze SQL schema files"""
        schema_files = list(self.root_dir.rglob("*.sql")) + list(self.root_dir.rglob("*.ddl"))
        
        for schema_file in schema_files:
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract table definitions
                tables = self._extract_sql_tables(content, str(schema_file))
                self.tables.extend(tables)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing schema file {schema_file}: {e}")
                
    def _extract_sql_tables(self, content: str, file_path: str):
        """Extract table definitions from SQL"""
        tables = []
        
        # Simple regex for CREATE TABLE statements
        table_pattern = r'CREATE\s+TABLE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\);'
        matches = re.findall(table_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for table_name, table_def in matches:
            columns = self._parse_sql_columns(table_def)
            
            table = DatabaseTable(
                name=table_name,
                columns=columns,
                indexes=[],
                foreign_keys=[],
                primary_key=[],
                schema='default',
                file_path=file_path
            )
            tables.append(table)
            
        return tables
        
    def _parse_sql_columns(self, table_def: str):
        """Parse SQL column definitions"""
        columns = []
        
        # Split by commas (simplified)
        column_defs = [col.strip() for col in table_def.split(',')]
        
        for col_def in column_defs:
            if col_def.strip():
                parts = col_def.strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    col_type = parts[1]
                    constraints = ' '.join(parts[2:]) if len(parts) > 2 else ''
                    
                    columns.append({
                        'name': name,
                        'type': col_type,
                        'constraints': [constraints] if constraints else []
                    })
                    
        return columns
        
    def _analyze_security_issues(self):
        """Analyze database security issues"""
        security_issues = []
        
        # Check for hardcoded credentials
        for connection in self.connections:
            if connection.username and not connection.username.startswith('${'):
                security_issues.append({
                    'type': 'hardcoded_credentials',
                    'file': connection.file_path,
                    'line': connection.line_number,
                    'details': f'Hardcoded username: {connection.username}'
                })
                
        # Check for SQL injection risks
        for query in self.queries:
            if '%' in query.raw_query or 'format(' in query.raw_query:
                security_issues.append({
                    'type': 'sql_injection_risk',
                    'file': query.file_path,
                    'line': query.line_number,
                    'details': 'String formatting in SQL query'
                })
                
        self.statistics['security_issues'] = security_issues
        
    def _analyze_performance_issues(self):
        """Analyze database performance issues"""
        performance_issues = []
        
        # Check for missing indexes on foreign keys
        for table in self.tables:
            for fk in table.foreign_keys:
                if fk['field'] not in [idx.split('_')[0] for idx in table.indexes]:
                    performance_issues.append({
                        'type': 'missing_fk_index',
                        'table': table.name,
                        'field': fk['field'],
                        'recommendation': f'Add index on {fk["field"]} in {table.name}'
                    })
                    
        # Check for complex queries without optimization
        for query in self.queries:
            if query.complexity_score > 3.0:
                performance_issues.append({
                    'type': 'complex_query',
                    'file': query.file_path,
                    'line': query.line_number,
                    'complexity': query.complexity_score,
                    'recommendation': 'Consider query optimization'
                })
                
        self.statistics['performance_issues'] = performance_issues
        
    def _update_statistics(self):
        """Update analysis statistics"""
        self.statistics['total_connections'] = len(self.connections)
        self.statistics['total_queries'] = len(self.queries)
        self.statistics['total_tables'] = len(self.tables)
        self.statistics['total_relationships'] = len(self.relationships)
        
        # Database types
        for connection in self.connections:
            self.statistics['database_types'].add(connection.connection_type)
        self.statistics['database_types'] = list(self.statistics['database_types'])
        
        # Average query complexity
        if self.queries:
            total_complexity = sum(q.complexity_score for q in self.queries)
            self.statistics['query_complexity_avg'] = round(total_complexity / len(self.queries), 2)
            
    def _print_results(self, duration):
        """Print analysis results"""
        print(f"\nDatabase Relationship Analysis Results:")
        print(f"   Database Connections: {self.statistics['total_connections']}")
        print(f"   Database Queries: {self.statistics['total_queries']}")
        print(f"   Database Tables: {self.statistics['total_tables']}")
        print(f"   Table Relationships: {self.statistics['total_relationships']}")
        print(f"   Database Types: {', '.join(self.statistics['database_types']) if self.statistics['database_types'] else 'None detected'}")
        print(f"   Files Analyzed: {self.statistics['files_analyzed']}")
        print(f"   Avg Query Complexity: {self.statistics['query_complexity_avg']}")
        print(f"   Security Issues: {len(self.statistics['security_issues'])}")
        print(f"   Performance Issues: {len(self.statistics['performance_issues'])}")
        print(f"   Scan Duration: {duration:.2f} seconds")
        
        if self.statistics['security_issues']:
            print(f"\nSecurity Recommendations:")
            for issue in self.statistics['security_issues'][:3]:
                print(f"   - {issue['type']}: {issue['details']}")
                
        if self.statistics['performance_issues']:
            print(f"\nPerformance Recommendations:")
            for issue in self.statistics['performance_issues'][:3]:
                print(f"   - {issue['type']}: {issue['recommendation']}")
                
        print(f"\nDatabase relationship analysis complete! Report saved to {self.output_file}")
        
    def _save_results(self):
        """Save analysis results to JSON file"""
        results = {
            'metadata': {
                'analysis_type': 'database_relationship_analysis',
                'timestamp': datetime.now().isoformat(),
                'root_directory': str(self.root_dir),
                'agent': 'Agent C',
                'phase': 'Hours 19-21: Database Relationship Analysis'
            },
            'statistics': self.statistics,
            'connections': [asdict(conn) for conn in self.connections],
            'queries': [asdict(query) for query in self.queries],
            'tables': [asdict(table) for table in self.tables],
            'relationships': [asdict(rel) for rel in self.relationships],
            'recommendations': {
                'security': self.statistics['security_issues'],
                'performance': self.statistics['performance_issues'],
                'optimization': self._generate_optimization_recommendations()
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    def _generate_optimization_recommendations(self):
        """Generate database optimization recommendations"""
        recommendations = []
        
        # Connection pooling recommendations
        if len(self.connections) > 5:
            recommendations.append({
                'type': 'connection_pooling',
                'priority': 'high',
                'recommendation': f'Consider implementing connection pooling for {len(self.connections)} database connections'
            })
            
        # Query optimization recommendations
        complex_queries = [q for q in self.queries if q.complexity_score > 2.0]
        if len(complex_queries) > 10:
            recommendations.append({
                'type': 'query_optimization',
                'priority': 'medium',
                'recommendation': f'Review and optimize {len(complex_queries)} complex database queries'
            })
            
        # Schema normalization recommendations
        if len(self.tables) > 20 and len(self.relationships) < len(self.tables) * 0.5:
            recommendations.append({
                'type': 'schema_normalization',
                'priority': 'low',
                'recommendation': 'Consider reviewing database schema for normalization opportunities'
            })
            
        return recommendations


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agent C Database Relationship Analyzer')
    parser.add_argument('--root', required=True, help='Root directory to analyze')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    analyzer = DatabaseRelationshipAnalyzer(args.root, args.output)
    analyzer.analyze_codebase()


if __name__ == "__main__":
    main()