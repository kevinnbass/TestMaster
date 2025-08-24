"""
Resource and I/O Analysis Module
=================================

Implements comprehensive resource and I/O analysis:
- File I/O pattern analysis and leak detection
- Network call pattern analysis with retry and timeout
- Database connection analysis and pooling
- Memory allocation patterns
- Cache effectiveness analysis
- Stream processing analysis
- Resource cleanup patterns
- External service dependency analysis
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class ResourceIOAnalyzer(BaseAnalyzer):
    """Analyzer for resource usage and I/O patterns."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive resource and I/O analysis."""
        print("[INFO] Analyzing Resource and I/O Patterns...")
        
        results = {
            "file_io_analysis": self._analyze_file_io_patterns(),
            "network_analysis": self._analyze_network_patterns(),
            "database_connections": self._analyze_database_connections(),
            "memory_allocations": self._analyze_memory_allocations(),
            "cache_analysis": self._analyze_cache_effectiveness(),
            "stream_processing": self._analyze_stream_processing(),
            "resource_cleanup": self._analyze_resource_cleanup(),
            "external_services": self._analyze_external_services(),
            "resource_metrics": self._calculate_resource_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} resource aspects")
        return results
    
    def _analyze_file_io_patterns(self) -> Dict[str, Any]:
        """Analyze file I/O patterns and potential issues."""
        file_io_analysis = {
            "file_operations": [],
            "file_handle_leaks": [],
            "inefficient_operations": [],
            "large_file_handling": [],
            "file_locking": [],
            "temp_file_usage": [],
            "path_safety": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Analyze file operations
                file_ops = self._detect_file_operations(tree)
                for op in file_ops:
                    op["file"] = file_key
                    file_io_analysis["file_operations"].append(op)
                    
                    # Check for potential handle leaks
                    if op.get("leak_risk"):
                        file_io_analysis["file_handle_leaks"].append({
                            "file": file_key,
                            "line": op["line"],
                            "type": op["type"],
                            "risk": "File handle may not be closed properly",
                            "recommendation": "Use context manager (with statement)"
                        })
                
                # Detect inefficient file operations
                inefficient = self._detect_inefficient_file_ops(tree, content)
                for ineff in inefficient:
                    ineff["file"] = file_key
                    file_io_analysis["inefficient_operations"].append(ineff)
                
                # Detect large file handling
                large_files = self._detect_large_file_handling(tree, content)
                for lf in large_files:
                    lf["file"] = file_key
                    file_io_analysis["large_file_handling"].append(lf)
                
                # Detect file locking patterns
                locks = self._detect_file_locking(tree, content)
                for lock in locks:
                    lock["file"] = file_key
                    file_io_analysis["file_locking"].append(lock)
                
                # Detect temporary file usage
                temp_files = self._detect_temp_file_usage(tree, content)
                for temp in temp_files:
                    temp["file"] = file_key
                    file_io_analysis["temp_file_usage"].append(temp)
                
                # Check path safety
                path_issues = self._check_path_safety(tree)
                for issue in path_issues:
                    issue["file"] = file_key
                    file_io_analysis["path_safety"].append(issue)
                
            except:
                continue
        
        return file_io_analysis
    
    def _detect_file_operations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect file I/O operations."""
        file_ops = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # open() function
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    op_info = {
                        "type": "open",
                        "line": node.lineno,
                        "mode": "r",  # default
                        "leak_risk": True,  # Assume risk unless in with statement
                        "context_manager": False
                    }
                    
                    # Extract mode if specified
                    if len(node.args) > 1:
                        if isinstance(node.args[1], ast.Constant):
                            op_info["mode"] = node.args[1].value
                    
                    # Check for mode in keywords
                    for keyword in node.keywords:
                        if keyword.arg == 'mode':
                            if isinstance(keyword.value, ast.Constant):
                                op_info["mode"] = keyword.value.value
                    
                    # Determine operation type from mode
                    if 'w' in op_info["mode"]:
                        op_info["operation"] = "write"
                    elif 'a' in op_info["mode"]:
                        op_info["operation"] = "append"
                    elif 'r' in op_info["mode"]:
                        op_info["operation"] = "read"
                    
                    if 'b' in op_info["mode"]:
                        op_info["binary"] = True
                    
                    file_ops.append(op_info)
                
                # Path operations
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['read', 'write', 'readlines', 'writelines']:
                        file_ops.append({
                            "type": node.func.attr,
                            "line": node.lineno,
                            "operation": "read" if 'read' in node.func.attr else "write"
                        })
                    
                    # pathlib operations
                    elif node.func.attr in ['read_text', 'write_text', 'read_bytes', 'write_bytes']:
                        file_ops.append({
                            "type": f"pathlib_{node.func.attr}",
                            "line": node.lineno,
                            "operation": "read" if 'read' in node.func.attr else "write",
                            "context_manager": False,  # pathlib handles cleanup
                            "leak_risk": False
                        })
            
            # Check for with statements
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name):
                            if item.context_expr.func.id == 'open':
                                # Mark the open operation as safe
                                for op in file_ops:
                                    if op["line"] == item.context_expr.lineno:
                                        op["context_manager"] = True
                                        op["leak_risk"] = False
        
        return file_ops
    
    def _detect_inefficient_file_ops(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect inefficient file operations."""
        inefficient = []
        
        # Reading entire file into memory
        if '.read()' in content and not '.read(' in content:
            inefficient.append({
                "type": "unbounded_read",
                "description": "Reading entire file into memory",
                "recommendation": "Use streaming or chunked reading for large files"
            })
        
        # Multiple small writes
        write_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['write', 'writelines']:
                        write_count += 1
        
        if write_count > 10:
            inefficient.append({
                "type": "multiple_small_writes",
                "count": write_count,
                "recommendation": "Buffer writes or use writelines() for better performance"
            })
        
        # Line-by-line processing without iterator
        if 'readlines()' in content:
            inefficient.append({
                "type": "readlines_memory",
                "description": "readlines() loads entire file into memory",
                "recommendation": "Use file iterator: 'for line in file:'"
            })
        
        return inefficient
    
    def _detect_large_file_handling(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect large file handling patterns."""
        large_file_patterns = []
        
        # Check for chunked reading
        if 'chunk' in content.lower() or 'buffer_size' in content.lower():
            large_file_patterns.append({
                "type": "chunked_reading",
                "status": "good",
                "description": "Uses chunked reading for efficiency"
            })
        
        # Check for streaming
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if iterating over file
                if isinstance(node.iter, ast.Name):
                    large_file_patterns.append({
                        "type": "streaming",
                        "line": node.lineno,
                        "status": "good",
                        "description": "Uses file iterator for streaming"
                    })
        
        # Check for memory mapping
        if 'mmap' in content:
            large_file_patterns.append({
                "type": "memory_mapped",
                "status": "good",
                "description": "Uses memory mapping for large files"
            })
        
        return large_file_patterns
    
    def _detect_file_locking(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect file locking patterns."""
        locking_patterns = []
        
        # Check for fcntl usage
        if 'fcntl' in content:
            locking_patterns.append({
                "type": "fcntl_locking",
                "platform": "unix",
                "description": "Uses fcntl for file locking"
            })
        
        # Check for msvcrt usage (Windows)
        if 'msvcrt' in content:
            locking_patterns.append({
                "type": "msvcrt_locking",
                "platform": "windows",
                "description": "Uses msvcrt for file locking"
            })
        
        # Check for lockfile libraries
        if 'lockfile' in content.lower() or 'filelock' in content.lower():
            locking_patterns.append({
                "type": "library_locking",
                "description": "Uses file locking library"
            })
        
        return locking_patterns
    
    def _detect_temp_file_usage(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect temporary file usage."""
        temp_patterns = []
        
        # Check for tempfile module usage
        if 'tempfile' in content:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['NamedTemporaryFile', 'TemporaryFile', 'mkstemp', 'mkdtemp']:
                            temp_patterns.append({
                                "type": node.func.attr,
                                "line": node.lineno,
                                "cleanup": "automatic" if 'Temporary' in node.func.attr else "manual"
                            })
        
        # Check for manual temp file patterns
        if '/tmp/' in content or 'temp_' in content or '.tmp' in content:
            temp_patterns.append({
                "type": "manual_temp",
                "warning": "Manual temp file management detected",
                "recommendation": "Use tempfile module for safe temp file handling"
            })
        
        return temp_patterns
    
    def _check_path_safety(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for path traversal and safety issues."""
        safety_issues = []
        
        for node in ast.walk(tree):
            # Check for path concatenation
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Add):
                    # String concatenation for paths
                    left_str = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
                    right_str = isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                    
                    if left_str or right_str:
                        # Check if it looks like a path
                        if isinstance(node.left, ast.Constant):
                            if '/' in str(node.left.value) or '\\' in str(node.left.value):
                                safety_issues.append({
                                    "type": "unsafe_path_concatenation",
                                    "line": node.lineno,
                                    "recommendation": "Use os.path.join() or pathlib"
                                })
            
            # Check for user input in paths
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    # Check if filename comes from input
                    if node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.Call):
                            if isinstance(arg.func, ast.Name) and arg.func.id == 'input':
                                safety_issues.append({
                                    "type": "path_traversal_risk",
                                    "line": node.lineno,
                                    "severity": "HIGH",
                                    "description": "User input directly used in file path",
                                    "recommendation": "Validate and sanitize user input paths"
                                })
        
        return safety_issues
    
    def _analyze_network_patterns(self) -> Dict[str, Any]:
        """Analyze network call patterns."""
        network_analysis = {
            "http_calls": [],
            "timeout_configuration": [],
            "retry_logic": [],
            "connection_pooling": [],
            "rate_limiting": [],
            "circuit_breakers": [],
            "batch_opportunities": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect HTTP calls
                http_calls = self._detect_http_calls(tree, content)
                for call in http_calls:
                    call["file"] = file_key
                    network_analysis["http_calls"].append(call)
                
                # Check timeout configuration
                timeouts = self._check_timeout_configuration(tree, content)
                for timeout in timeouts:
                    timeout["file"] = file_key
                    network_analysis["timeout_configuration"].append(timeout)
                
                # Detect retry logic
                retries = self._detect_retry_logic(tree, content)
                for retry in retries:
                    retry["file"] = file_key
                    network_analysis["retry_logic"].append(retry)
                
                # Check connection pooling
                pooling = self._check_connection_pooling(tree, content)
                for pool in pooling:
                    pool["file"] = file_key
                    network_analysis["connection_pooling"].append(pool)
                
                # Detect rate limiting
                rate_limits = self._detect_rate_limiting(tree, content)
                for limit in rate_limits:
                    limit["file"] = file_key
                    network_analysis["rate_limiting"].append(limit)
                
                # Detect circuit breakers
                breakers = self._detect_circuit_breakers(tree, content)
                for breaker in breakers:
                    breaker["file"] = file_key
                    network_analysis["circuit_breakers"].append(breaker)
                
                # Identify batch opportunities
                batch_ops = self._identify_batch_opportunities(tree)
                for batch in batch_ops:
                    batch["file"] = file_key
                    network_analysis["batch_opportunities"].append(batch)
                
            except:
                continue
        
        return network_analysis
    
    def _detect_http_calls(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect HTTP calls."""
        http_calls = []
        
        # requests library
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                        # Check if it's requests module
                        if isinstance(node.func.value, ast.Name):
                            if node.func.value.id == 'requests':
                                call_info = {
                                    "library": "requests",
                                    "method": node.func.attr.upper(),
                                    "line": node.lineno,
                                    "has_timeout": False,
                                    "has_retry": False
                                }
                                
                                # Check for timeout parameter
                                for keyword in node.keywords:
                                    if keyword.arg == 'timeout':
                                        call_info["has_timeout"] = True
                                
                                http_calls.append(call_info)
        
        # urllib
        if 'urlopen' in content or 'urllib' in content:
            http_calls.append({
                "library": "urllib",
                "warning": "Using urllib, consider requests for better API"
            })
        
        # httpx
        if 'httpx' in content:
            http_calls.append({
                "library": "httpx",
                "feature": "Async HTTP client"
            })
        
        # aiohttp
        if 'aiohttp' in content:
            http_calls.append({
                "library": "aiohttp",
                "feature": "Async HTTP client"
            })
        
        return http_calls
    
    def _check_timeout_configuration(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Check timeout configuration in network calls."""
        timeouts = []
        
        # Check for missing timeouts
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['get', 'post', 'put', 'delete']:
                        has_timeout = False
                        
                        for keyword in node.keywords:
                            if keyword.arg == 'timeout':
                                has_timeout = True
                                # Check timeout value
                                if isinstance(keyword.value, ast.Constant):
                                    timeout_value = keyword.value.value
                                    if timeout_value > 30:
                                        timeouts.append({
                                            "line": node.lineno,
                                            "value": timeout_value,
                                            "warning": "Long timeout value",
                                            "recommendation": "Consider shorter timeout with retry"
                                        })
                        
                        if not has_timeout:
                            timeouts.append({
                                "line": node.lineno,
                                "issue": "missing_timeout",
                                "severity": "HIGH",
                                "recommendation": "Always set timeouts for network calls"
                            })
        
        return timeouts
    
    def _detect_retry_logic(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect retry logic implementation."""
        retry_patterns = []
        
        # Check for retry decorators
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if 'retry' in decorator.id.lower():
                            retry_patterns.append({
                                "type": "decorator",
                                "function": node.name,
                                "line": node.lineno
                            })
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            if 'retry' in decorator.func.id.lower():
                                retry_patterns.append({
                                    "type": "decorator_with_config",
                                    "function": node.name,
                                    "line": node.lineno
                                })
        
        # Check for manual retry loops
        for node in ast.walk(tree):
            if isinstance(node, ast.While) or isinstance(node, ast.For):
                # Look for retry pattern
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Name):
                        if 'retry' in inner_node.id.lower() or 'attempt' in inner_node.id.lower():
                            retry_patterns.append({
                                "type": "manual_loop",
                                "line": node.lineno,
                                "warning": "Manual retry implementation",
                                "recommendation": "Consider using tenacity or retrying library"
                            })
                            break
        
        # Check for exponential backoff
        if 'backoff' in content.lower() or 'exponential' in content.lower():
            retry_patterns.append({
                "type": "exponential_backoff",
                "status": "good",
                "description": "Uses exponential backoff strategy"
            })
        
        return retry_patterns
    
    def _check_connection_pooling(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Check for connection pooling usage."""
        pooling_patterns = []
        
        # requests.Session
        if 'Session()' in content:
            pooling_patterns.append({
                "type": "requests_session",
                "description": "Uses requests.Session for connection pooling"
            })
        
        # urllib3 PoolManager
        if 'PoolManager' in content:
            pooling_patterns.append({
                "type": "urllib3_pool",
                "description": "Uses urllib3 PoolManager"
            })
        
        # Check for multiple requests without session
        request_count = content.count('requests.get') + content.count('requests.post')
        if request_count > 5 and 'Session' not in content:
            pooling_patterns.append({
                "issue": "no_connection_pooling",
                "request_count": request_count,
                "recommendation": "Use requests.Session() for connection pooling"
            })
        
        return pooling_patterns
    
    def _detect_rate_limiting(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect rate limiting implementation."""
        rate_limiting = []
        
        # Check for rate limiting libraries
        if 'ratelimit' in content.lower():
            rate_limiting.append({
                "type": "library",
                "description": "Uses rate limiting library"
            })
        
        # Check for sleep patterns (basic rate limiting)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'sleep':
                        rate_limiting.append({
                            "type": "sleep_based",
                            "line": node.lineno,
                            "warning": "Basic rate limiting with sleep",
                            "recommendation": "Consider token bucket or sliding window algorithm"
                        })
        
        return rate_limiting
    
    def _detect_circuit_breakers(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect circuit breaker patterns."""
        circuit_breakers = []
        
        # Check for circuit breaker libraries
        if 'circuit' in content.lower() and 'breaker' in content.lower():
            circuit_breakers.append({
                "type": "library",
                "description": "Uses circuit breaker pattern"
            })
        
        # Check for manual implementation
        if 'failure_count' in content or 'error_threshold' in content:
            circuit_breakers.append({
                "type": "manual",
                "description": "Manual circuit breaker implementation"
            })
        
        return circuit_breakers
    
    def _identify_batch_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify opportunities for batch API calls."""
        batch_opportunities = []
        
        # Look for loops with API calls
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Attribute):
                            if inner_node.func.attr in ['get', 'post', 'put', 'delete']:
                                batch_opportunities.append({
                                    "type": "api_call_in_loop",
                                    "line": node.lineno,
                                    "recommendation": "Consider batching API calls"
                                })
                                break
        
        return batch_opportunities
    
    def _analyze_database_connections(self) -> Dict[str, Any]:
        """Analyze database connection patterns."""
        db_analysis = {
            "connections": [],
            "connection_pools": [],
            "transaction_management": [],
            "cursor_management": [],
            "prepared_statements": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect database connections
                connections = self._detect_db_connections(tree, content)
                for conn in connections:
                    conn["file"] = file_key
                    db_analysis["connections"].append(conn)
                
                # Check connection pooling
                pools = self._check_db_connection_pooling(tree, content)
                for pool in pools:
                    pool["file"] = file_key
                    db_analysis["connection_pools"].append(pool)
                
                # Check transaction management
                transactions = self._check_transaction_management(tree, content)
                for trans in transactions:
                    trans["file"] = file_key
                    db_analysis["transaction_management"].append(trans)
                
                # Check cursor management
                cursors = self._check_cursor_management(tree, content)
                for cursor in cursors:
                    cursor["file"] = file_key
                    db_analysis["cursor_management"].append(cursor)
                
            except:
                continue
        
        return db_analysis
    
    def _detect_db_connections(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect database connections."""
        connections = []
        
        # SQLAlchemy
        if 'create_engine' in content:
            connections.append({
                "type": "sqlalchemy",
                "orm": True,
                "description": "SQLAlchemy engine detected"
            })
        
        # Django
        if 'django.db' in content:
            connections.append({
                "type": "django",
                "orm": True,
                "description": "Django database connection"
            })
        
        # psycopg2 (PostgreSQL)
        if 'psycopg2' in content:
            connections.append({
                "type": "psycopg2",
                "database": "postgresql",
                "orm": False
            })
        
        # pymongo (MongoDB)
        if 'pymongo' in content or 'MongoClient' in content:
            connections.append({
                "type": "pymongo",
                "database": "mongodb",
                "orm": False
            })
        
        # sqlite3
        if 'sqlite3' in content:
            connections.append({
                "type": "sqlite3",
                "database": "sqlite",
                "orm": False
            })
        
        return connections
    
    def _check_db_connection_pooling(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Check database connection pooling."""
        pooling = []
        
        # SQLAlchemy pooling
        if 'pool_size' in content or 'max_overflow' in content:
            pooling.append({
                "type": "sqlalchemy_pool",
                "configured": True
            })
        
        # Check for connection reuse
        connection_count = content.count('.connect(') + content.count('Connection(')
        if connection_count > 3:
            pooling.append({
                "issue": "multiple_connections",
                "count": connection_count,
                "recommendation": "Use connection pooling to reuse connections"
            })
        
        return pooling
    
    def _check_transaction_management(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Check transaction management patterns."""
        transactions = []
        
        # Check for explicit transactions
        if 'begin()' in content or 'commit()' in content or 'rollback()' in content:
            transactions.append({
                "type": "explicit_transactions",
                "description": "Uses explicit transaction management"
            })
        
        # Check for atomic decorators (Django)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id == 'atomic':
                            transactions.append({
                                "type": "django_atomic",
                                "function": node.name,
                                "line": node.lineno
                            })
        
        # Check for context managers
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Attribute):
                            if item.context_expr.func.attr in ['transaction', 'atomic']:
                                transactions.append({
                                    "type": "context_manager",
                                    "line": node.lineno
                                })
        
        return transactions
    
    def _check_cursor_management(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Check cursor management patterns."""
        cursors = []
        
        # Check for cursor usage
        if 'cursor()' in content:
            # Check if cursors are properly closed
            if 'cursor.close()' not in content:
                cursors.append({
                    "issue": "unclosed_cursor",
                    "recommendation": "Use context manager or ensure cursors are closed"
                })
        
        # Check for cursor in with statement
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Attribute):
                            if item.context_expr.func.attr == 'cursor':
                                cursors.append({
                                    "type": "context_managed_cursor",
                                    "line": node.lineno,
                                    "status": "good"
                                })
        
        return cursors
    
    def _analyze_memory_allocations(self) -> Dict[str, Any]:
        """Analyze memory allocation patterns."""
        memory_analysis = {
            "allocations": [],
            "large_allocations": [],
            "memory_pools": [],
            "object_reuse": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect memory allocations
                allocations = self._detect_memory_allocations(tree)
                for alloc in allocations:
                    alloc["file"] = file_key
                    memory_analysis["allocations"].append(alloc)
                
                # Detect large allocations
                large = self._detect_large_allocations(tree)
                for l in large:
                    l["file"] = file_key
                    memory_analysis["large_allocations"].append(l)
                
                # Check for object reuse patterns
                reuse = self._check_object_reuse(tree)
                for r in reuse:
                    r["file"] = file_key
                    memory_analysis["object_reuse"].append(r)
                
            except:
                continue
        
        return memory_analysis
    
    def _detect_memory_allocations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect memory allocation patterns."""
        allocations = []
        
        for node in ast.walk(tree):
            # List comprehensions
            if isinstance(node, ast.ListComp):
                allocations.append({
                    "type": "list_comprehension",
                    "line": node.lineno if hasattr(node, 'lineno') else 0
                })
            
            # Dict comprehensions
            elif isinstance(node, ast.DictComp):
                allocations.append({
                    "type": "dict_comprehension",
                    "line": node.lineno if hasattr(node, 'lineno') else 0
                })
            
            # String concatenation in loops
            elif isinstance(node, ast.For):
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.AugAssign):
                        if isinstance(inner_node.op, ast.Add):
                            allocations.append({
                                "type": "string_concatenation_in_loop",
                                "line": inner_node.lineno if hasattr(inner_node, 'lineno') else 0,
                                "warning": "Inefficient string building",
                                "recommendation": "Use list.append() and ''.join()"
                            })
                            break
        
        return allocations
    
    def _detect_large_allocations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect large memory allocations."""
        large_allocations = []
        
        for node in ast.walk(tree):
            # Large list literals
            if isinstance(node, ast.List):
                if len(node.elts) > 1000:
                    large_allocations.append({
                        "type": "large_list",
                        "size": len(node.elts),
                        "line": node.lineno if hasattr(node, 'lineno') else 0
                    })
            
            # Large dict literals
            elif isinstance(node, ast.Dict):
                if len(node.keys) > 1000:
                    large_allocations.append({
                        "type": "large_dict",
                        "size": len(node.keys),
                        "line": node.lineno if hasattr(node, 'lineno') else 0
                    })
            
            # List multiplication
            elif isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Mult):
                    if isinstance(node.left, ast.List) or isinstance(node.right, ast.List):
                        large_allocations.append({
                            "type": "list_multiplication",
                            "line": node.lineno if hasattr(node, 'lineno') else 0,
                            "warning": "Large list creation"
                        })
        
        return large_allocations
    
    def _check_object_reuse(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for object reuse patterns."""
        reuse_patterns = []
        
        for node in ast.walk(tree):
            # Check for object pooling patterns
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        if 'pool' in method.name.lower() or 'cache' in method.name.lower():
                            reuse_patterns.append({
                                "type": "object_pooling",
                                "class": node.name,
                                "method": method.name,
                                "line": method.lineno
                            })
        
        return reuse_patterns
    
    def _analyze_cache_effectiveness(self) -> Dict[str, Any]:
        """Analyze cache effectiveness and patterns."""
        cache_analysis = {
            "cache_implementations": [],
            "cache_strategies": [],
            "invalidation_patterns": [],
            "cache_opportunities": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect cache implementations
                implementations = self._detect_cache_implementations(tree, content)
                for impl in implementations:
                    impl["file"] = file_key
                    cache_analysis["cache_implementations"].append(impl)
                
                # Detect cache strategies
                strategies = self._detect_cache_strategies(tree, content)
                for strategy in strategies:
                    strategy["file"] = file_key
                    cache_analysis["cache_strategies"].append(strategy)
                
                # Detect invalidation patterns
                invalidation = self._detect_cache_invalidation(tree, content)
                for inv in invalidation:
                    inv["file"] = file_key
                    cache_analysis["invalidation_patterns"].append(inv)
                
                # Identify cache opportunities
                opportunities = self._identify_cache_opportunities(tree)
                for opp in opportunities:
                    opp["file"] = file_key
                    cache_analysis["cache_opportunities"].append(opp)
                
            except:
                continue
        
        return cache_analysis
    
    def _detect_cache_implementations(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect cache implementations."""
        implementations = []
        
        # functools.lru_cache
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id == 'lru_cache':
                            implementations.append({
                                "type": "lru_cache",
                                "function": node.name,
                                "line": node.lineno
                            })
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            if decorator.func.id == 'lru_cache':
                                implementations.append({
                                    "type": "lru_cache_configured",
                                    "function": node.name,
                                    "line": node.lineno
                                })
        
        # Redis cache
        if 'redis' in content.lower():
            implementations.append({
                "type": "redis",
                "description": "Redis cache implementation"
            })
        
        # Memcached
        if 'memcache' in content.lower():
            implementations.append({
                "type": "memcached",
                "description": "Memcached implementation"
            })
        
        # Dictionary cache
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if 'cache' in node.id.lower():
                    implementations.append({
                        "type": "dictionary_cache",
                        "variable": node.id,
                        "line": node.lineno if hasattr(node, 'lineno') else 0
                    })
                    break
        
        return implementations
    
    def _detect_cache_strategies(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect caching strategies."""
        strategies = []
        
        # TTL-based caching
        if 'ttl' in content.lower() or 'expire' in content.lower():
            strategies.append({
                "type": "ttl_based",
                "description": "Time-to-live based caching"
            })
        
        # LRU strategy
        if 'lru' in content.lower():
            strategies.append({
                "type": "lru",
                "description": "Least Recently Used eviction"
            })
        
        # Write-through cache
        if 'write_through' in content.lower():
            strategies.append({
                "type": "write_through",
                "description": "Write-through caching strategy"
            })
        
        # Write-back cache
        if 'write_back' in content.lower() or 'write_behind' in content.lower():
            strategies.append({
                "type": "write_back",
                "description": "Write-back caching strategy"
            })
        
        return strategies
    
    def _detect_cache_invalidation(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect cache invalidation patterns."""
        invalidation = []
        
        # Manual invalidation
        if 'cache.clear()' in content or 'cache.delete' in content:
            invalidation.append({
                "type": "manual",
                "description": "Manual cache invalidation"
            })
        
        # Event-based invalidation
        if 'on_update' in content or 'on_change' in content:
            invalidation.append({
                "type": "event_based",
                "description": "Event-driven cache invalidation"
            })
        
        return invalidation
    
    def _identify_cache_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify caching opportunities."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function is pure (cacheable)
                is_pure = True
                has_expensive_ops = False
                
                for inner_node in ast.walk(node):
                    # Check for side effects
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Name):
                            if inner_node.func.id in ['print', 'open', 'write']:
                                is_pure = False
                                break
                    
                    # Check for expensive operations
                    if isinstance(inner_node, ast.For):
                        has_expensive_ops = True
                
                if is_pure and has_expensive_ops:
                    opportunities.append({
                        "function": node.name,
                        "line": node.lineno,
                        "reason": "Pure function with expensive operations",
                        "recommendation": "Consider adding @lru_cache decorator"
                    })
        
        return opportunities
    
    def _analyze_stream_processing(self) -> Dict[str, Any]:
        """Analyze stream processing patterns."""
        stream_analysis = {
            "stream_patterns": [],
            "backpressure_handling": [],
            "buffer_management": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect streaming patterns
                if 'stream' in content.lower() or 'generator' in content:
                    stream_analysis["stream_patterns"].append({
                        "file": file_key,
                        "type": "streaming_detected"
                    })
                
                # Check for generators
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for inner_node in ast.walk(node):
                            if isinstance(inner_node, ast.Yield):
                                stream_analysis["stream_patterns"].append({
                                    "file": file_key,
                                    "function": node.name,
                                    "type": "generator",
                                    "line": node.lineno
                                })
                                break
                
            except:
                continue
        
        return stream_analysis
    
    def _analyze_resource_cleanup(self) -> Dict[str, Any]:
        """Analyze resource cleanup patterns."""
        cleanup_analysis = {
            "context_managers": [],
            "finally_blocks": [],
            "cleanup_issues": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Check for context managers
                for node in ast.walk(tree):
                    if isinstance(node, ast.With):
                        cleanup_analysis["context_managers"].append({
                            "file": file_key,
                            "line": node.lineno,
                            "status": "good"
                        })
                
                # Check for finally blocks
                for node in ast.walk(tree):
                    if isinstance(node, ast.Try):
                        if node.finalbody:
                            cleanup_analysis["finally_blocks"].append({
                                "file": file_key,
                                "line": node.lineno
                            })
                
                # Check for resource leaks
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id == 'open':
                                # Check if in with statement
                                # Simplified check
                                cleanup_analysis["cleanup_issues"].append({
                                    "file": file_key,
                                    "line": node.lineno,
                                    "type": "potential_file_leak",
                                    "recommendation": "Use context manager"
                                })
                
            except:
                continue
        
        return cleanup_analysis
    
    def _analyze_external_services(self) -> Dict[str, Any]:
        """Analyze external service dependencies."""
        external_analysis = {
            "services": [],
            "health_checks": [],
            "fallback_mechanisms": []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect external services
                services = self._detect_external_services(content)
                for service in services:
                    service["file"] = file_key
                    external_analysis["services"].append(service)
                
                # Check for health checks
                if 'health' in content.lower() and 'check' in content.lower():
                    external_analysis["health_checks"].append({
                        "file": file_key,
                        "detected": True
                    })
                
                # Check for fallback mechanisms
                if 'fallback' in content.lower() or 'default' in content:
                    external_analysis["fallback_mechanisms"].append({
                        "file": file_key,
                        "detected": True
                    })
                
            except:
                continue
        
        return external_analysis
    
    def _detect_external_services(self, content: str) -> List[Dict[str, Any]]:
        """Detect external service dependencies."""
        services = []
        
        # AWS services
        if 's3' in content.lower() or 'boto' in content:
            services.append({
                "type": "aws",
                "service": "S3/AWS"
            })
        
        # Google Cloud
        if 'google.cloud' in content:
            services.append({
                "type": "gcp",
                "service": "Google Cloud"
            })
        
        # Azure
        if 'azure' in content.lower():
            services.append({
                "type": "azure",
                "service": "Microsoft Azure"
            })
        
        # Elasticsearch
        if 'elasticsearch' in content.lower():
            services.append({
                "type": "elasticsearch",
                "service": "Elasticsearch"
            })
        
        # Message queues
        if 'rabbitmq' in content.lower() or 'amqp' in content.lower():
            services.append({
                "type": "message_queue",
                "service": "RabbitMQ/AMQP"
            })
        
        if 'kafka' in content.lower():
            services.append({
                "type": "message_queue",
                "service": "Apache Kafka"
            })
        
        return services
    
    def _calculate_resource_metrics(self) -> Dict[str, Any]:
        """Calculate overall resource usage metrics."""
        file_io = self._analyze_file_io_patterns()
        network = self._analyze_network_patterns()
        database = self._analyze_database_connections()
        cache = self._analyze_cache_effectiveness()
        
        metrics = {
            "resource_score": 100,
            "issues": [],
            "recommendations": []
        }
        
        # File I/O issues
        if file_io["file_handle_leaks"]:
            metrics["resource_score"] -= 15
            metrics["issues"].append(f"{len(file_io['file_handle_leaks'])} potential file handle leaks")
        
        # Network issues
        missing_timeouts = [t for t in network["timeout_configuration"] if t.get("issue") == "missing_timeout"]
        if missing_timeouts:
            metrics["resource_score"] -= 10
            metrics["issues"].append(f"{len(missing_timeouts)} network calls without timeouts")
        
        # Database issues
        if not database["connection_pools"]:
            metrics["resource_score"] -= 10
            metrics["issues"].append("No database connection pooling detected")
        
        # Cache opportunities
        if cache["cache_opportunities"]:
            metrics["resource_score"] -= 5
            metrics["recommendations"].append(f"{len(cache['cache_opportunities'])} functions could benefit from caching")
        
        # Generate recommendations
        metrics["recommendations"].extend([
            "Always use context managers for resource management",
            "Implement connection pooling for database and HTTP connections",
            "Set appropriate timeouts for all network operations",
            "Use caching for expensive, pure functions",
            "Implement retry logic with exponential backoff",
            "Monitor resource usage in production"
        ])
        
        return metrics