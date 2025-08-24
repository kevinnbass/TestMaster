"""
Concurrency Analysis for GIL, Async/Await, and Thread Safety
=============================================================

Implements comprehensive concurrency analysis:
- GIL bottleneck detection
- Thread safety analysis
- Race condition detection
- Deadlock potential identification
- Async/await pattern analysis
- Lock contention analysis
- Concurrent data structure usage
- Parallelization opportunities
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class ConcurrencyAnalyzer(BaseAnalyzer):
    """Analyzer for concurrency patterns and thread safety."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        
        # Threading primitives
        self.threading_primitives = {
            "locks": ["Lock", "RLock", "Semaphore", "BoundedSemaphore"],
            "events": ["Event", "Condition"],
            "barriers": ["Barrier"],
            "thread_local": ["local"],
        }
        
        # Async patterns
        self.async_patterns = {
            "coroutines": ["async def", "await"],
            "tasks": ["create_task", "gather", "wait", "as_completed"],
            "sync_primitives": ["Lock", "Event", "Condition", "Semaphore", "Queue"],
        }
        
        # Concurrent data structures
        self.concurrent_structures = {
            "safe": ["queue.Queue", "queue.PriorityQueue", "queue.LifoQueue", "collections.deque"],
            "unsafe": ["list", "dict", "set"],
        }
        
        # GIL-bound operations
        self.gil_bound_ops = {
            "cpu_intensive": ["math operations", "string manipulation", "list comprehension"],
            "io_bound": ["file operations", "network calls", "database queries"],
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive concurrency analysis."""
        print("[INFO] Analyzing Concurrency Patterns...")
        
        results = {
            "gil_analysis": self._analyze_gil_bottlenecks(),
            "thread_safety": self._analyze_thread_safety(),
            "race_conditions": self._detect_race_conditions(),
            "deadlock_analysis": self._detect_deadlock_potential(),
            "async_patterns": self._analyze_async_await_patterns(),
            "lock_analysis": self._analyze_lock_contention(),
            "concurrent_structures": self._analyze_concurrent_data_structures(),
            "parallelization": self._identify_parallelization_opportunities(),
            "concurrency_metrics": self._calculate_concurrency_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} concurrency aspects")
        return results
    
    def _analyze_gil_bottlenecks(self) -> Dict[str, Any]:
        """Analyze potential GIL bottlenecks."""
        gil_analysis = {
            "cpu_bound_operations": [],
            "gil_contentious_patterns": [],
            "multiprocessing_candidates": [],
            "gil_bypass_opportunities": [],
            "gil_impact_score": 0
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect CPU-bound operations
                cpu_ops = self._detect_cpu_bound_operations(tree, content)
                for op in cpu_ops:
                    op["file"] = file_key
                    gil_analysis["cpu_bound_operations"].append(op)
                
                # Detect GIL-contentious patterns
                contentious = self._detect_gil_contentious_patterns(tree, content)
                for pattern in contentious:
                    pattern["file"] = file_key
                    gil_analysis["gil_contentious_patterns"].append(pattern)
                
                # Identify multiprocessing candidates
                mp_candidates = self._identify_multiprocessing_candidates(tree, content)
                for candidate in mp_candidates:
                    candidate["file"] = file_key
                    gil_analysis["multiprocessing_candidates"].append(candidate)
                
                # Find GIL bypass opportunities (C extensions, NumPy, etc.)
                bypass_ops = self._find_gil_bypass_opportunities(content)
                gil_analysis["gil_bypass_opportunities"].extend(bypass_ops)
                
            except:
                continue
        
        # Calculate GIL impact score
        gil_analysis["gil_impact_score"] = self._calculate_gil_impact(gil_analysis)
        
        return gil_analysis
    
    def _detect_cpu_bound_operations(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect CPU-bound operations that might cause GIL contention."""
        cpu_operations = []
        
        for node in ast.walk(tree):
            # Heavy mathematical computations
            if isinstance(node, ast.For):
                loop_body_ops = self._count_math_operations(node)
                if loop_body_ops > 5:
                    cpu_operations.append({
                        "line": node.lineno,
                        "type": "heavy_computation_loop",
                        "math_ops": loop_body_ops,
                        "gil_impact": "high",
                        "recommendation": "Consider using NumPy or multiprocessing"
                    })
            
            # List comprehensions with complex operations
            elif isinstance(node, ast.ListComp):
                complexity = self._analyze_comprehension_complexity(node)
                if complexity > 3:
                    cpu_operations.append({
                        "line": node.lineno,
                        "type": "complex_list_comprehension",
                        "complexity": complexity,
                        "gil_impact": "medium",
                        "recommendation": "Consider generator or parallel processing"
                    })
            
            # Recursive functions
            elif isinstance(node, ast.FunctionDef):
                if self._is_recursive_function(node):
                    cpu_operations.append({
                        "function": node.name,
                        "line": node.lineno,
                        "type": "recursive_function",
                        "gil_impact": "medium",
                        "recommendation": "Consider iterative approach or memoization"
                    })
            
            # String operations in loops
            elif isinstance(node, ast.While):
                if self._has_string_operations(node):
                    cpu_operations.append({
                        "line": node.lineno,
                        "type": "string_processing_loop",
                        "gil_impact": "high",
                        "recommendation": "Use str.join() or StringIO"
                    })
        
        return cpu_operations
    
    def _analyze_thread_safety(self) -> Dict[str, Any]:
        """Analyze thread safety issues."""
        thread_safety = {
            "shared_state_access": [],
            "unprotected_mutations": [],
            "thread_safe_patterns": [],
            "unsafe_patterns": [],
            "synchronization_analysis": {}
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect shared state access
                shared_state = self._detect_shared_state(tree, content)
                for state in shared_state:
                    state["file"] = file_key
                    thread_safety["shared_state_access"].append(state)
                
                # Detect unprotected mutations
                unprotected = self._detect_unprotected_mutations(tree, content)
                for mutation in unprotected:
                    mutation["file"] = file_key
                    thread_safety["unprotected_mutations"].append(mutation)
                
                # Identify thread-safe patterns
                safe_patterns = self._identify_thread_safe_patterns(tree, content)
                thread_safety["thread_safe_patterns"].extend(safe_patterns)
                
                # Identify unsafe patterns
                unsafe = self._identify_unsafe_patterns(tree, content)
                for pattern in unsafe:
                    pattern["file"] = file_key
                    thread_safety["unsafe_patterns"].append(pattern)
                
            except:
                continue
        
        # Analyze synchronization coverage
        thread_safety["synchronization_analysis"] = self._analyze_synchronization_coverage(thread_safety)
        
        return thread_safety
    
    def _detect_shared_state(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect shared state that might be accessed by multiple threads."""
        shared_state = []
        
        # Global variables
        global_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                global_vars.update(node.names)
        
        # Class attributes (potential shared state)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Class variables
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                shared_state.append({
                                    "type": "class_variable",
                                    "class": node.name,
                                    "variable": target.id,
                                    "line": item.lineno,
                                    "risk": "high" if not self._has_synchronization(node) else "low"
                                })
        
        # Module-level variables
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if not target.id.startswith('_'):  # Public variables
                            shared_state.append({
                                "type": "module_variable",
                                "variable": target.id,
                                "line": node.lineno,
                                "risk": "medium"
                            })
        
        return shared_state
    
    def _detect_unprotected_mutations(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect mutations of shared state without proper synchronization."""
        unprotected = []
        
        # Find all mutations
        for node in ast.walk(tree):
            # Check for mutations without locks
            if isinstance(node, ast.AugAssign):  # +=, -=, etc.
                if not self._is_protected_by_lock(tree, node):
                    unprotected.append({
                        "line": node.lineno,
                        "type": "augmented_assignment",
                        "operation": type(node.op).__name__,
                        "risk": "high",
                        "fix": "Use lock around mutation"
                    })
            
            # List/dict mutations
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['append', 'extend', 'pop', 'remove', 'clear']:
                        if not self._is_protected_by_lock(tree, node):
                            unprotected.append({
                                "line": node.lineno,
                                "type": "collection_mutation",
                                "method": node.func.attr,
                                "risk": "high",
                                "fix": "Use thread-safe collection or lock"
                            })
        
        return unprotected
    
    def _detect_race_conditions(self) -> Dict[str, Any]:
        """Detect potential race conditions."""
        race_conditions = {
            "check_then_act": [],
            "read_modify_write": [],
            "test_and_set": [],
            "double_checked_locking": [],
            "race_condition_score": 0
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect check-then-act patterns
                check_then_act = self._detect_check_then_act(tree)
                for pattern in check_then_act:
                    pattern["file"] = file_key
                    race_conditions["check_then_act"].append(pattern)
                
                # Detect read-modify-write patterns
                rmw = self._detect_read_modify_write(tree)
                for pattern in rmw:
                    pattern["file"] = file_key
                    race_conditions["read_modify_write"].append(pattern)
                
                # Detect test-and-set patterns
                test_set = self._detect_test_and_set(tree)
                for pattern in test_set:
                    pattern["file"] = file_key
                    race_conditions["test_and_set"].append(pattern)
                
                # Detect double-checked locking
                dcl = self._detect_double_checked_locking(tree)
                for pattern in dcl:
                    pattern["file"] = file_key
                    race_conditions["double_checked_locking"].append(pattern)
                
            except:
                continue
        
        # Calculate race condition risk score
        race_conditions["race_condition_score"] = self._calculate_race_condition_score(race_conditions)
        
        return race_conditions
    
    def _detect_check_then_act(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect check-then-act race condition patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Look for patterns like: if not exists: create
                test = node.test
                body = node.body
                
                # Check if test involves existence/state check
                if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                    # Check if body performs action based on test
                    if body and self._performs_state_change(body[0]):
                        patterns.append({
                            "line": node.lineno,
                            "pattern": "check_then_act",
                            "risk": "high",
                            "description": "State check followed by action",
                            "fix": "Use atomic operations or locks"
                        })
                
                # Check for file existence patterns
                elif isinstance(test, ast.Call):
                    if self._is_existence_check(test):
                        if body and self._creates_resource(body[0]):
                            patterns.append({
                                "line": node.lineno,
                                "pattern": "file_check_create",
                                "risk": "medium",
                                "description": "File existence check before creation",
                                "fix": "Use try/except with EAFP principle"
                            })
        
        return patterns
    
    def _detect_read_modify_write(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect read-modify-write patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Pattern: x = x + 1 or similar
                if len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.BinOp):
                        if isinstance(node.value.left, ast.Name):
                            if target.id == node.value.left.id:
                                patterns.append({
                                    "line": node.lineno,
                                    "variable": target.id,
                                    "pattern": "read_modify_write",
                                    "risk": "high",
                                    "fix": "Use atomic operations or locks"
                                })
        
        return patterns
    
    def _detect_deadlock_potential(self) -> Dict[str, Any]:
        """Detect potential deadlock situations."""
        deadlock_analysis = {
            "lock_ordering_issues": [],
            "nested_locks": [],
            "circular_dependencies": [],
            "resource_hierarchies": [],
            "deadlock_risk_score": 0
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect lock ordering issues
                ordering_issues = self._detect_lock_ordering_issues(tree)
                for issue in ordering_issues:
                    issue["file"] = file_key
                    deadlock_analysis["lock_ordering_issues"].append(issue)
                
                # Detect nested locks
                nested = self._detect_nested_locks(tree)
                for nest in nested:
                    nest["file"] = file_key
                    deadlock_analysis["nested_locks"].append(nest)
                
                # Detect circular dependencies
                circular = self._detect_circular_lock_dependencies(tree, content)
                deadlock_analysis["circular_dependencies"].extend(circular)
                
            except:
                continue
        
        # Calculate deadlock risk score
        deadlock_analysis["deadlock_risk_score"] = self._calculate_deadlock_risk(deadlock_analysis)
        
        return deadlock_analysis
    
    def _detect_nested_locks(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect nested lock acquisitions."""
        nested_locks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                # Check if it's a lock acquisition
                if self._is_lock_acquisition(node):
                    # Check for nested with statements
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.With) and inner != node:
                            if self._is_lock_acquisition(inner):
                                nested_locks.append({
                                    "line": node.lineno,
                                    "inner_line": inner.lineno,
                                    "pattern": "nested_locks",
                                    "risk": "high",
                                    "description": "Nested lock acquisition detected",
                                    "fix": "Use single lock or ensure consistent ordering"
                                })
        
        return nested_locks
    
    def _analyze_async_await_patterns(self) -> Dict[str, Any]:
        """Analyze async/await patterns and coroutines."""
        async_analysis = {
            "coroutines": [],
            "async_context_managers": [],
            "concurrent_tasks": [],
            "blocking_operations": [],
            "async_antipatterns": [],
            "async_best_practices": {}
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect coroutines
                coroutines = self._detect_coroutines(tree)
                for coro in coroutines:
                    coro["file"] = file_key
                    async_analysis["coroutines"].append(coro)
                
                # Detect async context managers
                async_cms = self._detect_async_context_managers(tree)
                async_analysis["async_context_managers"].extend(async_cms)
                
                # Detect concurrent task patterns
                tasks = self._detect_concurrent_tasks(tree, content)
                for task in tasks:
                    task["file"] = file_key
                    async_analysis["concurrent_tasks"].append(task)
                
                # Detect blocking operations in async code
                blocking = self._detect_blocking_in_async(tree)
                for block in blocking:
                    block["file"] = file_key
                    async_analysis["blocking_operations"].append(block)
                
                # Detect async antipatterns
                antipatterns = self._detect_async_antipatterns(tree, content)
                for anti in antipatterns:
                    anti["file"] = file_key
                    async_analysis["async_antipatterns"].append(anti)
                
            except:
                continue
        
        # Analyze async best practices
        async_analysis["async_best_practices"] = self._check_async_best_practices(async_analysis)
        
        return async_analysis
    
    def _detect_coroutines(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect async function definitions."""
        coroutines = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                # Analyze coroutine complexity
                await_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Await))
                
                coroutines.append({
                    "name": node.name,
                    "line": node.lineno,
                    "await_count": await_count,
                    "complexity": self._calculate_async_complexity(node),
                    "has_error_handling": self._has_async_error_handling(node)
                })
        
        return coroutines
    
    def _detect_blocking_in_async(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect blocking operations in async functions."""
        blocking_ops = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                # Check for blocking I/O
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        if isinstance(inner.func, ast.Name):
                            # Check for blocking functions
                            if inner.func.id in ['open', 'read', 'write']:
                                blocking_ops.append({
                                    "function": node.name,
                                    "line": inner.lineno,
                                    "operation": inner.func.id,
                                    "type": "blocking_io",
                                    "risk": "high",
                                    "fix": "Use aiofiles or async alternatives"
                                })
                            
                            # Check for time.sleep instead of asyncio.sleep
                            elif inner.func.id == 'sleep':
                                blocking_ops.append({
                                    "function": node.name,
                                    "line": inner.lineno,
                                    "operation": "time.sleep",
                                    "type": "blocking_sleep",
                                    "risk": "critical",
                                    "fix": "Use await asyncio.sleep()"
                                })
                        
                        # Check for requests library
                        elif isinstance(inner.func, ast.Attribute):
                            if inner.func.attr in ['get', 'post', 'put', 'delete']:
                                blocking_ops.append({
                                    "function": node.name,
                                    "line": inner.lineno,
                                    "operation": f"requests.{inner.func.attr}",
                                    "type": "blocking_http",
                                    "risk": "high",
                                    "fix": "Use aiohttp or httpx"
                                })
        
        return blocking_ops
    
    def _analyze_lock_contention(self) -> Dict[str, Any]:
        """Analyze lock contention patterns."""
        lock_analysis = {
            "lock_usage": [],
            "lock_duration": [],
            "fine_grained_locks": [],
            "coarse_grained_locks": [],
            "lock_free_alternatives": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Analyze lock usage patterns
                locks = self._analyze_lock_usage(tree)
                for lock in locks:
                    lock["file"] = file_key
                    lock_analysis["lock_usage"].append(lock)
                
                # Estimate lock duration
                duration = self._estimate_lock_duration(tree)
                lock_analysis["lock_duration"].extend(duration)
                
                # Identify fine vs coarse grained locks
                fine, coarse = self._classify_lock_granularity(tree)
                lock_analysis["fine_grained_locks"].extend(fine)
                lock_analysis["coarse_grained_locks"].extend(coarse)
                
                # Suggest lock-free alternatives
                alternatives = self._suggest_lock_free_alternatives(tree, content)
                lock_analysis["lock_free_alternatives"].extend(alternatives)
                
            except:
                continue
        
        return lock_analysis
    
    def _analyze_concurrent_data_structures(self) -> Dict[str, Any]:
        """Analyze usage of concurrent data structures."""
        structure_analysis = {
            "thread_safe_structures": [],
            "unsafe_structures": [],
            "queue_usage": [],
            "atomic_operations": [],
            "recommendations": []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect thread-safe data structures
                safe = self._detect_thread_safe_structures(tree, content)
                for struct in safe:
                    struct["file"] = file_key
                    structure_analysis["thread_safe_structures"].append(struct)
                
                # Detect unsafe data structure usage
                unsafe = self._detect_unsafe_structures(tree, content)
                for struct in unsafe:
                    struct["file"] = file_key
                    structure_analysis["unsafe_structures"].append(struct)
                
                # Analyze queue usage
                queues = self._analyze_queue_usage(tree, content)
                structure_analysis["queue_usage"].extend(queues)
                
                # Detect atomic operations
                atomic = self._detect_atomic_operations(tree, content)
                structure_analysis["atomic_operations"].extend(atomic)
                
            except:
                continue
        
        # Generate recommendations
        structure_analysis["recommendations"] = self._generate_structure_recommendations(structure_analysis)
        
        return structure_analysis
    
    def _identify_parallelization_opportunities(self) -> Dict[str, Any]:
        """Identify opportunities for parallelization."""
        parallelization = {
            "embarrassingly_parallel": [],
            "map_reduce_candidates": [],
            "pipeline_opportunities": [],
            "vectorization_candidates": [],
            "recommendations": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Find embarrassingly parallel loops
                parallel_loops = self._find_parallel_loops(tree)
                for loop in parallel_loops:
                    loop["file"] = file_key
                    parallelization["embarrassingly_parallel"].append(loop)
                
                # Find map-reduce candidates
                map_reduce = self._find_map_reduce_patterns(tree)
                for pattern in map_reduce:
                    pattern["file"] = file_key
                    parallelization["map_reduce_candidates"].append(pattern)
                
                # Find pipeline opportunities
                pipelines = self._find_pipeline_opportunities(tree)
                parallelization["pipeline_opportunities"].extend(pipelines)
                
                # Find vectorization candidates
                vectorizable = self._find_vectorization_candidates(tree, content)
                parallelization["vectorization_candidates"].extend(vectorizable)
                
            except:
                continue
        
        # Generate parallelization recommendations
        parallelization["recommendations"] = self._generate_parallelization_recommendations(parallelization)
        
        return parallelization
    
    def _find_parallel_loops(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find loops that can be parallelized."""
        parallel_candidates = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if loop iterations are independent
                if self._has_loop_independence(node):
                    parallel_candidates.append({
                        "line": node.lineno,
                        "pattern": "independent_iterations",
                        "parallelization_method": "multiprocessing.Pool or concurrent.futures",
                        "speedup_potential": self._estimate_speedup(node)
                    })
                
                # Check for map-like patterns
                if self._is_map_pattern(node):
                    parallel_candidates.append({
                        "line": node.lineno,
                        "pattern": "map_operation",
                        "parallelization_method": "Pool.map() or ThreadPoolExecutor",
                        "speedup_potential": "high"
                    })
        
        return parallel_candidates
    
    def _calculate_concurrency_metrics(self) -> Dict[str, Any]:
        """Calculate overall concurrency metrics."""
        gil_analysis = self._analyze_gil_bottlenecks()
        thread_safety = self._analyze_thread_safety()
        race_conditions = self._detect_race_conditions()
        deadlock = self._detect_deadlock_potential()
        
        metrics = {
            "gil_bottleneck_count": len(gil_analysis["cpu_bound_operations"]),
            "thread_safety_issues": len(thread_safety["unprotected_mutations"]),
            "race_condition_count": len(race_conditions["check_then_act"]) + len(race_conditions["read_modify_write"]),
            "deadlock_risk_score": deadlock["deadlock_risk_score"],
            "concurrency_health_score": 0,
            "critical_issues": [],
            "recommendations": []
        }
        
        # Calculate health score
        health_score = 100
        health_score -= metrics["gil_bottleneck_count"] * 3
        health_score -= metrics["thread_safety_issues"] * 5
        health_score -= metrics["race_condition_count"] * 10
        health_score -= metrics["deadlock_risk_score"] * 5
        metrics["concurrency_health_score"] = max(0, health_score)
        
        # Identify critical issues
        if metrics["race_condition_count"] > 0:
            metrics["critical_issues"].append("Race conditions detected")
        
        if metrics["deadlock_risk_score"] > 5:
            metrics["critical_issues"].append("High deadlock risk")
        
        if metrics["thread_safety_issues"] > 5:
            metrics["critical_issues"].append("Multiple thread safety violations")
        
        # Generate recommendations
        if metrics["gil_bottleneck_count"] > 3:
            metrics["recommendations"].append("Consider multiprocessing for CPU-bound operations")
        
        if metrics["thread_safety_issues"] > 0:
            metrics["recommendations"].append("Add proper synchronization to shared state access")
        
        if metrics["race_condition_count"] > 0:
            metrics["recommendations"].append("Use atomic operations or locks to prevent race conditions")
        
        return metrics
    
    # Helper methods
    def _count_math_operations(self, node: ast.AST) -> int:
        """Count mathematical operations in a node."""
        count = 0
        for inner in ast.walk(node):
            if isinstance(inner, ast.BinOp):
                if isinstance(inner.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    count += 1
            elif isinstance(inner, ast.Call):
                if isinstance(inner.func, ast.Attribute):
                    if inner.func.attr in ['sqrt', 'pow', 'exp', 'log']:
                        count += 1
        return count
    
    def _analyze_comprehension_complexity(self, comp: ast.ListComp) -> int:
        """Analyze complexity of list comprehension."""
        complexity = 1
        
        # Count generators
        complexity += len(comp.generators)
        
        # Count conditions
        for gen in comp.generators:
            complexity += len(gen.ifs)
        
        # Check for nested operations
        for node in ast.walk(comp.elt):
            if isinstance(node, ast.Call):
                complexity += 1
        
        return complexity
    
    def _is_recursive_function(self, func: ast.FunctionDef) -> bool:
        """Check if function is recursive."""
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == func.name:
                        return True
        return False
    
    def _has_string_operations(self, node: ast.AST) -> bool:
        """Check if node contains string operations."""
        for inner in ast.walk(node):
            if isinstance(inner, ast.BinOp):
                if isinstance(inner.op, ast.Add):
                    # Check if it's string concatenation
                    return True
        return False
    
    def _detect_gil_contentious_patterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect patterns that cause GIL contention."""
        patterns = []
        
        # Threading with CPU-bound work
        if "threading" in content:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if inherits from Thread
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "Thread":
                            # Check for CPU-intensive run method
                            for method in node.body:
                                if isinstance(method, ast.FunctionDef) and method.name == "run":
                                    if self._count_math_operations(method) > 3:
                                        patterns.append({
                                            "class": node.name,
                                            "line": node.lineno,
                                            "pattern": "cpu_bound_thread",
                                            "risk": "high",
                                            "fix": "Use multiprocessing instead"
                                        })
        
        return patterns
    
    def _identify_multiprocessing_candidates(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Identify code that would benefit from multiprocessing."""
        candidates = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function is CPU-intensive
                math_ops = self._count_math_operations(node)
                if math_ops > 10:
                    candidates.append({
                        "function": node.name,
                        "line": node.lineno,
                        "math_operations": math_ops,
                        "recommendation": "Convert to multiprocessing.Process"
                    })
        
        return candidates
    
    def _find_gil_bypass_opportunities(self, content: str) -> List[Dict[str, Any]]:
        """Find opportunities to bypass GIL using C extensions."""
        opportunities = []
        
        if "numpy" in content:
            opportunities.append({
                "library": "NumPy",
                "benefit": "Releases GIL for array operations"
            })
        
        if "pandas" in content:
            opportunities.append({
                "library": "Pandas",
                "benefit": "Optimized C operations"
            })
        
        if "numba" in content:
            opportunities.append({
                "library": "Numba",
                "benefit": "JIT compilation with nogil option"
            })
        
        return opportunities
    
    def _calculate_gil_impact(self, gil_analysis: Dict[str, Any]) -> int:
        """Calculate GIL impact score."""
        score = 0
        score += len(gil_analysis["cpu_bound_operations"]) * 3
        score += len(gil_analysis["gil_contentious_patterns"]) * 5
        return min(score, 10)
    
    def _has_synchronization(self, class_node: ast.ClassDef) -> bool:
        """Check if class has synchronization mechanisms."""
        for node in ast.walk(class_node):
            if isinstance(node, ast.Name):
                if node.id in ["Lock", "RLock", "Semaphore"]:
                    return True
        return False
    
    def _is_protected_by_lock(self, tree: ast.AST, mutation: ast.AST) -> bool:
        """Check if mutation is protected by a lock."""
        # Check if mutation is inside a with statement
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                if mutation in list(ast.walk(node)):
                    # Check if with statement uses a lock
                    return self._is_lock_acquisition(node)
        return False
    
    def _is_lock_acquisition(self, with_node: ast.With) -> bool:
        """Check if with statement acquires a lock."""
        for item in with_node.items:
            if isinstance(item.context_expr, ast.Call):
                if isinstance(item.context_expr.func, ast.Attribute):
                    if item.context_expr.func.attr in ["acquire", "__enter__"]:
                        return True
            elif isinstance(item.context_expr, ast.Name):
                if "lock" in item.context_expr.id.lower():
                    return True
        return False
    
    def _identify_thread_safe_patterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Identify thread-safe patterns."""
        patterns = []
        
        if "Queue" in content:
            patterns.append({"pattern": "thread_safe_queue", "safe": True})
        
        if "threading.local" in content:
            patterns.append({"pattern": "thread_local_storage", "safe": True})
        
        return patterns
    
    def _identify_unsafe_patterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Identify thread-unsafe patterns."""
        patterns = []
        
        # Global variable modification
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                patterns.append({
                    "line": node.lineno,
                    "pattern": "global_modification",
                    "variables": node.names,
                    "risk": "high"
                })
        
        return patterns
    
    def _analyze_synchronization_coverage(self, thread_safety: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze synchronization coverage."""
        total_mutations = len(thread_safety["unprotected_mutations"])
        protected = len(thread_safety["thread_safe_patterns"])
        
        coverage = 0 if total_mutations == 0 else protected / (total_mutations + protected)
        
        return {
            "coverage_ratio": coverage,
            "protected_operations": protected,
            "unprotected_operations": total_mutations,
            "recommendation": "Increase synchronization coverage" if coverage < 0.8 else "Good coverage"
        }
    
    def _performs_state_change(self, node: ast.AST) -> bool:
        """Check if node performs a state change."""
        return isinstance(node, (ast.Assign, ast.AugAssign, ast.Call))
    
    def _is_existence_check(self, node: ast.Call) -> bool:
        """Check if call is an existence check."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in ["exists", "isfile", "isdir"]
        return False
    
    def _creates_resource(self, node: ast.AST) -> bool:
        """Check if node creates a resource."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id in ["open", "create", "mkdir"]
        return False
    
    def _detect_test_and_set(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect test-and-set patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for: if flag: flag = False; do_something()
                if len(node.body) >= 2:
                    first = node.body[0]
                    if isinstance(first, ast.Assign):
                        patterns.append({
                            "line": node.lineno,
                            "pattern": "test_and_set",
                            "risk": "medium"
                        })
        
        return patterns
    
    def _detect_double_checked_locking(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect double-checked locking patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for nested if with same condition
                for inner in node.body:
                    if isinstance(inner, ast.With):
                        for inner2 in inner.body:
                            if isinstance(inner2, ast.If):
                                patterns.append({
                                    "line": node.lineno,
                                    "pattern": "double_checked_locking",
                                    "risk": "low",
                                    "note": "Generally safe in Python"
                                })
        
        return patterns
    
    def _calculate_race_condition_score(self, race_conditions: Dict[str, Any]) -> int:
        """Calculate race condition risk score."""
        score = 0
        score += len(race_conditions["check_then_act"]) * 3
        score += len(race_conditions["read_modify_write"]) * 4
        score += len(race_conditions["test_and_set"]) * 2
        return min(score, 10)
    
    def _detect_lock_ordering_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect potential lock ordering issues."""
        issues = []
        
        # Track lock acquisition order
        lock_sequences = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                locks_in_function = []
                for inner in ast.walk(node):
                    if isinstance(inner, ast.With):
                        if self._is_lock_acquisition(inner):
                            locks_in_function.append(inner.lineno)
                
                if len(locks_in_function) > 1:
                    lock_sequences.append({
                        "function": node.name,
                        "locks": locks_in_function
                    })
        
        # Check for inconsistent ordering
        if len(lock_sequences) > 1:
            issues.append({
                "pattern": "multiple_lock_sequences",
                "risk": "medium",
                "recommendation": "Ensure consistent lock ordering"
            })
        
        return issues
    
    def _detect_circular_lock_dependencies(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect circular lock dependencies."""
        # Simplified detection
        return []
    
    def _calculate_deadlock_risk(self, deadlock_analysis: Dict[str, Any]) -> int:
        """Calculate deadlock risk score."""
        score = 0
        score += len(deadlock_analysis["lock_ordering_issues"]) * 3
        score += len(deadlock_analysis["nested_locks"]) * 4
        score += len(deadlock_analysis["circular_dependencies"]) * 5
        return min(score, 10)
    
    def _detect_async_context_managers(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect async context managers."""
        managers = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncWith):
                managers.append({
                    "line": node.lineno,
                    "type": "async_context_manager"
                })
        
        return managers
    
    def _detect_concurrent_tasks(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect concurrent task patterns."""
        tasks = []
        
        if "asyncio.gather" in content:
            tasks.append({"pattern": "gather", "concurrency": "high"})
        
        if "asyncio.create_task" in content:
            tasks.append({"pattern": "create_task", "concurrency": "medium"})
        
        if "asyncio.as_completed" in content:
            tasks.append({"pattern": "as_completed", "concurrency": "high"})
        
        return tasks
    
    def _detect_async_antipatterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect async antipatterns."""
        antipatterns = []
        
        # Forgetting await
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["gather", "sleep", "wait"]:
                        # Check if it's awaited
                        parent = self._get_parent(tree, node)
                        if not isinstance(parent, ast.Await):
                            antipatterns.append({
                                "line": node.lineno,
                                "pattern": "missing_await",
                                "function": node.func.id,
                                "risk": "critical"
                            })
        
        return antipatterns
    
    def _calculate_async_complexity(self, func: ast.AsyncFunctionDef) -> int:
        """Calculate async function complexity."""
        complexity = 1
        
        for node in ast.walk(func):
            if isinstance(node, ast.Await):
                complexity += 1
            elif isinstance(node, ast.AsyncFor):
                complexity += 2
            elif isinstance(node, ast.AsyncWith):
                complexity += 1
        
        return complexity
    
    def _has_async_error_handling(self, func: ast.AsyncFunctionDef) -> bool:
        """Check if async function has error handling."""
        for node in ast.walk(func):
            if isinstance(node, ast.Try):
                return True
        return False
    
    def _check_async_best_practices(self, async_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check async best practices compliance."""
        practices = {
            "uses_gather_for_concurrency": len([t for t in async_analysis["concurrent_tasks"] if t.get("pattern") == "gather"]) > 0,
            "avoids_blocking_operations": len(async_analysis["blocking_operations"]) == 0,
            "proper_error_handling": True,  # Simplified
            "score": 0
        }
        
        score = 0
        if practices["uses_gather_for_concurrency"]:
            score += 30
        if practices["avoids_blocking_operations"]:
            score += 40
        if practices["proper_error_handling"]:
            score += 30
        
        practices["score"] = score
        
        return practices
    
    def _analyze_lock_usage(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze lock usage patterns."""
        lock_usage = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                if self._is_lock_acquisition(node):
                    lock_usage.append({
                        "line": node.lineno,
                        "statements_protected": len(node.body),
                        "estimated_duration": "short" if len(node.body) < 5 else "long"
                    })
        
        return lock_usage
    
    def _estimate_lock_duration(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Estimate lock hold duration."""
        durations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                if self._is_lock_acquisition(node):
                    # Count operations inside lock
                    op_count = len(list(ast.walk(node)))
                    duration = "short" if op_count < 20 else "medium" if op_count < 50 else "long"
                    
                    durations.append({
                        "line": node.lineno,
                        "operation_count": op_count,
                        "estimated_duration": duration,
                        "risk": "high" if duration == "long" else "low"
                    })
        
        return durations
    
    def _classify_lock_granularity(self, tree: ast.AST) -> Tuple[List, List]:
        """Classify locks as fine or coarse grained."""
        fine = []
        coarse = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                if self._is_lock_acquisition(node):
                    scope_size = len(list(ast.walk(node)))
                    if scope_size < 10:
                        fine.append({"line": node.lineno, "scope": "fine"})
                    else:
                        coarse.append({"line": node.lineno, "scope": "coarse"})
        
        return fine, coarse
    
    def _suggest_lock_free_alternatives(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Suggest lock-free alternatives."""
        suggestions = []
        
        # Suggest atomic operations
        suggestions.append({
            "pattern": "counter_increment",
            "alternative": "Use threading.local() or atomic operations"
        })
        
        # Suggest concurrent collections
        suggestions.append({
            "pattern": "shared_list",
            "alternative": "Use queue.Queue or collections.deque"
        })
        
        return suggestions
    
    def _detect_thread_safe_structures(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect thread-safe data structures."""
        safe_structures = []
        
        if "queue.Queue" in content or "Queue()" in content:
            safe_structures.append({
                "type": "Queue",
                "thread_safe": True
            })
        
        if "collections.deque" in content:
            safe_structures.append({
                "type": "deque",
                "thread_safe": "partially"
            })
        
        return safe_structures
    
    def _detect_unsafe_structures(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect unsafe data structure usage in concurrent context."""
        unsafe = []
        
        # Check for list/dict/set in threaded context
        if "threading" in content or "Thread" in content:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["list", "dict", "set"]:
                            unsafe.append({
                                "line": node.lineno,
                                "type": node.func.id,
                                "risk": "high",
                                "fix": "Use thread-safe alternative"
                            })
        
        return unsafe
    
    def _analyze_queue_usage(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Analyze queue usage patterns."""
        queue_usage = []
        
        if "queue" in content.lower():
            queue_usage.append({
                "uses_queue": True,
                "recommendation": "Good choice for thread communication"
            })
        
        return queue_usage
    
    def _detect_atomic_operations(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect atomic operations."""
        atomic_ops = []
        
        # Threading.local usage
        if "threading.local" in content:
            atomic_ops.append({"type": "thread_local", "safe": True})
        
        return atomic_ops
    
    def _generate_structure_recommendations(self, structure_analysis: Dict[str, Any]) -> List[str]:
        """Generate data structure recommendations."""
        recommendations = []
        
        if structure_analysis["unsafe_structures"]:
            recommendations.append("Replace thread-unsafe collections with Queue or thread-safe alternatives")
        
        if not structure_analysis["thread_safe_structures"]:
            recommendations.append("Consider using thread-safe data structures for concurrent access")
        
        return recommendations
    
    def _has_loop_independence(self, loop: ast.For) -> bool:
        """Check if loop iterations are independent."""
        # Simplified check - look for shared state modifications
        for node in ast.walk(loop):
            if isinstance(node, ast.AugAssign):
                return False
        return True
    
    def _is_map_pattern(self, loop: ast.For) -> bool:
        """Check if loop follows a map pattern."""
        # Check if loop appends transformed values
        for node in ast.walk(loop):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "append":
                        return True
        return False
    
    def _estimate_speedup(self, loop: ast.For) -> str:
        """Estimate potential speedup from parallelization."""
        loop_size = len(list(ast.walk(loop)))
        if loop_size > 50:
            return "high"
        elif loop_size > 20:
            return "medium"
        return "low"
    
    def _find_map_reduce_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find map-reduce patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["map", "filter", "reduce"]:
                        patterns.append({
                            "line": node.lineno,
                            "function": node.func.id,
                            "parallelizable": True
                        })
        
        return patterns
    
    def _find_pipeline_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find pipeline parallelization opportunities."""
        # Simplified implementation
        return []
    
    def _find_vectorization_candidates(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Find vectorization candidates."""
        candidates = []
        
        # NumPy vectorization opportunities
        if "for" in content and "numpy" in content:
            candidates.append({
                "pattern": "loop_over_array",
                "recommendation": "Use NumPy vectorized operations"
            })
        
        return candidates
    
    def _generate_parallelization_recommendations(self, parallelization: Dict[str, Any]) -> List[str]:
        """Generate parallelization recommendations."""
        recommendations = []
        
        if parallelization["embarrassingly_parallel"]:
            recommendations.append("Use multiprocessing.Pool for embarrassingly parallel loops")
        
        if parallelization["map_reduce_candidates"]:
            recommendations.append("Consider using concurrent.futures for map-reduce operations")
        
        if parallelization["vectorization_candidates"]:
            recommendations.append("Use NumPy vectorization for array operations")
        
        return recommendations
    
    def _get_parent(self, tree: ast.AST, node: ast.AST) -> Optional[ast.AST]:
        """Get parent node of a given node."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                if child == node:
                    return parent
        return None