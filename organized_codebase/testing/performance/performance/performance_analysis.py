"""
Performance Analysis Engine
============================

Implements comprehensive performance analysis:
- Algorithmic complexity detection (Big-O notation)
- Memory usage patterns and leak detection
- Database query analysis (N+1, inefficient queries)
- Concurrency and GIL impact analysis
- Performance benchmarking and profiling points
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import math

from .base_analyzer import BaseAnalyzer


class PerformanceAnalyzer(BaseAnalyzer):
    """Analyzer for performance patterns and optimization opportunities."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        print("[INFO] Analyzing Performance Patterns...")
        
        results = {
            "algorithmic_complexity": self._analyze_algorithmic_complexity(),
            "memory_patterns": self._analyze_memory_patterns(),
            "database_performance": self._analyze_database_performance(),
            "concurrency_performance": self._analyze_concurrency_performance(),
            "cpu_intensive_operations": self._identify_cpu_intensive_operations(),
            "performance_anti_patterns": self._detect_performance_anti_patterns(),
            "optimization_opportunities": self._identify_optimization_opportunities(),
            "performance_metrics": self._calculate_performance_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} performance aspects")
        return results
    
    def _analyze_algorithmic_complexity(self) -> Dict[str, Any]:
        """Analyze algorithmic complexity and Big-O notation."""
        complexity_analysis = {
            "functions": [],
            "complexity_distribution": defaultdict(int),
            "nested_loops": [],
            "recursive_functions": [],
            "sorting_algorithms": [],
            "search_algorithms": [],
            "dynamic_programming": [],
            "graph_algorithms": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Analyze function complexity
                        complexity = self._calculate_big_o_complexity(node)
                        
                        complexity_analysis["functions"].append({
                            "function": node.name,
                            "file": file_key,
                            "line": node.lineno,
                            "complexity": complexity["notation"],
                            "complexity_class": complexity["class"],
                            "nested_loops": complexity["nested_loops"],
                            "recursive": complexity["recursive"],
                            "analysis_details": complexity["details"]
                        })
                        
                        complexity_analysis["complexity_distribution"][complexity["class"]] += 1
                        
                        # Track specific patterns
                        if complexity["nested_loops"] > 0:
                            complexity_analysis["nested_loops"].append({
                                "function": node.name,
                                "file": file_key,
                                "depth": complexity["nested_loops"],
                                "estimated_complexity": f"O(n^{complexity['nested_loops']})"
                            })
                        
                        if complexity["recursive"]:
                            complexity_analysis["recursive_functions"].append({
                                "function": node.name,
                                "file": file_key,
                                "type": self._classify_recursion(node),
                                "has_memoization": self._has_memoization(node)
                            })
                        
                        # Detect specific algorithm types
                        if self._is_sorting_algorithm(node):
                            complexity_analysis["sorting_algorithms"].append({
                                "function": node.name,
                                "file": file_key,
                                "type": self._classify_sorting_algorithm(node),
                                "complexity": complexity["notation"]
                            })
                        
                        if self._is_search_algorithm(node):
                            complexity_analysis["search_algorithms"].append({
                                "function": node.name,
                                "file": file_key,
                                "type": self._classify_search_algorithm(node),
                                "complexity": complexity["notation"]
                            })
                        
                        if self._uses_dynamic_programming(node):
                            complexity_analysis["dynamic_programming"].append({
                                "function": node.name,
                                "file": file_key,
                                "has_memoization": self._has_memoization(node),
                                "space_complexity": self._estimate_space_complexity(node)
                            })
                        
                        if self._is_graph_algorithm(node):
                            complexity_analysis["graph_algorithms"].append({
                                "function": node.name,
                                "file": file_key,
                                "type": self._classify_graph_algorithm(node)
                            })
                
            except:
                continue
        
        return complexity_analysis
    
    def _calculate_big_o_complexity(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Calculate Big-O complexity for a function."""
        complexity = {
            "notation": "O(1)",
            "class": "constant",
            "nested_loops": 0,
            "recursive": False,
            "details": []
        }
        
        # Check for recursion
        if self._is_recursive(func_node):
            complexity["recursive"] = True
            recursion_type = self._classify_recursion(func_node)
            
            if recursion_type == "linear":
                complexity["notation"] = "O(n)"
                complexity["class"] = "linear"
            elif recursion_type == "binary":
                complexity["notation"] = "O(log n)"
                complexity["class"] = "logarithmic"
            elif recursion_type == "exponential":
                complexity["notation"] = "O(2^n)"
                complexity["class"] = "exponential"
            
            complexity["details"].append(f"Recursive function ({recursion_type})")
        
        # Analyze loops
        loop_analysis = self._analyze_loops(func_node)
        max_nesting = loop_analysis["max_nesting"]
        
        if max_nesting > 0:
            complexity["nested_loops"] = max_nesting
            
            # Check for logarithmic patterns
            if loop_analysis["has_logarithmic"]:
                complexity["notation"] = f"O(n log n)" if max_nesting == 1 else f"O(n^{max_nesting} log n)"
                complexity["class"] = "linearithmic" if max_nesting == 1 else "polynomial"
                complexity["details"].append("Contains logarithmic loop pattern")
            else:
                if max_nesting == 1:
                    complexity["notation"] = "O(n)"
                    complexity["class"] = "linear"
                elif max_nesting == 2:
                    complexity["notation"] = "O(n²)"
                    complexity["class"] = "quadratic"
                elif max_nesting == 3:
                    complexity["notation"] = "O(n³)"
                    complexity["class"] = "cubic"
                else:
                    complexity["notation"] = f"O(n^{max_nesting})"
                    complexity["class"] = "polynomial"
                
                complexity["details"].append(f"{max_nesting} nested loop(s)")
        
        # Check for specific patterns
        if self._has_binary_search_pattern(func_node):
            complexity["notation"] = "O(log n)"
            complexity["class"] = "logarithmic"
            complexity["details"].append("Binary search pattern detected")
        
        if self._has_divide_and_conquer_pattern(func_node):
            if not complexity["recursive"]:
                complexity["notation"] = "O(n log n)"
                complexity["class"] = "linearithmic"
            complexity["details"].append("Divide and conquer pattern")
        
        # Check for factorial/permutation patterns
        if self._has_factorial_pattern(func_node):
            complexity["notation"] = "O(n!)"
            complexity["class"] = "factorial"
            complexity["details"].append("Factorial/permutation pattern")
        
        return complexity
    
    def _analyze_loops(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze loop patterns in a function."""
        loop_info = {
            "max_nesting": 0,
            "has_logarithmic": False,
            "loop_types": []
        }
        
        def analyze_node(node, depth=0):
            nonlocal loop_info
            
            if isinstance(node, (ast.For, ast.While)):
                loop_info["max_nesting"] = max(loop_info["max_nesting"], depth + 1)
                
                # Check for logarithmic patterns (i *= 2, i /= 2, etc.)
                if isinstance(node, ast.While):
                    for inner_node in ast.walk(node):
                        if isinstance(inner_node, ast.AugAssign):
                            if isinstance(inner_node.op, (ast.Mult, ast.Div, ast.FloorDiv)):
                                if isinstance(inner_node.value, ast.Constant):
                                    if inner_node.value.value in [2, 10]:
                                        loop_info["has_logarithmic"] = True
                
                # Check for range with step
                if isinstance(node, ast.For):
                    if isinstance(node.iter, ast.Call):
                        if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                            if len(node.iter.args) >= 3:
                                # Has step parameter
                                step = node.iter.args[2]
                                if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
                                    loop_info["loop_types"].append("reverse")
                
                # Recursively analyze nested loops
                for child in ast.iter_child_nodes(node):
                    analyze_node(child, depth + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    analyze_node(child, depth)
        
        for stmt in func_node.body:
            analyze_node(stmt)
        
        return loop_info
    
    def _is_recursive(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is recursive."""
        func_name = func_node.name
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return True
        return False
    
    def _classify_recursion(self, func_node: ast.FunctionDef) -> str:
        """Classify type of recursion."""
        func_name = func_node.name
        recursive_calls = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    recursive_calls.append(node)
        
        if len(recursive_calls) == 1:
            # Check if it's tail recursion
            last_stmt = func_node.body[-1] if func_node.body else None
            if isinstance(last_stmt, ast.Return) and isinstance(last_stmt.value, ast.Call):
                if isinstance(last_stmt.value.func, ast.Name) and last_stmt.value.func.id == func_name:
                    return "tail"
            return "linear"
        elif len(recursive_calls) == 2:
            return "binary"
        elif len(recursive_calls) > 2:
            return "exponential"
        
        return "unknown"
    
    def _has_memoization(self, func_node: ast.FunctionDef) -> bool:
        """Check if function uses memoization."""
        # Check for cache decorator
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ['cache', 'lru_cache', 'memoize']:
                    return True
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in ['cache', 'lru_cache', 'memoize']:
                    return True
        
        # Check for manual memoization pattern
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name):
                if 'memo' in node.id.lower() or 'cache' in node.id.lower():
                    return True
        
        return False
    
    def _has_binary_search_pattern(self, func_node: ast.FunctionDef) -> bool:
        """Check for binary search pattern."""
        has_mid_calculation = False
        has_comparison = False
        has_recursive_or_loop = False
        
        for node in ast.walk(func_node):
            # Check for mid = (left + right) // 2 or similar
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and 'mid' in target.id.lower():
                        has_mid_calculation = True
            
            # Check for comparison
            if isinstance(node, ast.Compare):
                has_comparison = True
            
            # Check for loop or recursion
            if isinstance(node, (ast.While, ast.For)) or self._is_recursive(func_node):
                has_recursive_or_loop = True
        
        return has_mid_calculation and has_comparison and has_recursive_or_loop
    
    def _has_divide_and_conquer_pattern(self, func_node: ast.FunctionDef) -> bool:
        """Check for divide and conquer pattern."""
        # Look for splitting/dividing operations
        has_split = False
        has_merge = False
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Subscript):
                # Check for array slicing
                if isinstance(node.slice, ast.Slice):
                    has_split = True
            
            # Check for merge/combine operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if any(keyword in node.func.id.lower() for keyword in ['merge', 'combine', 'join']):
                        has_merge = True
        
        return has_split and (has_merge or self._is_recursive(func_node))
    
    def _has_factorial_pattern(self, func_node: ast.FunctionDef) -> bool:
        """Check for factorial/permutation pattern."""
        # Look for patterns like n * factorial(n-1)
        for node in ast.walk(func_node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if isinstance(node.right, ast.Call):
                    if isinstance(node.right.func, ast.Name) and node.right.func.id == func_node.name:
                        return True
        
        # Check for permutation/combination function names
        if any(keyword in func_node.name.lower() for keyword in ['factorial', 'permut', 'combin']):
            return True
        
        return False
    
    def _is_sorting_algorithm(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is a sorting algorithm."""
        # Check function name
        if any(keyword in func_node.name.lower() for keyword in ['sort', 'bubble', 'quick', 'merge', 'heap', 'insertion']):
            return True
        
        # Check for swap operations
        has_swap = False
        has_comparison = False
        
        for node in ast.walk(func_node):
            # Check for swap pattern: a, b = b, a
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Tuple) and len(node.targets) == 1:
                    if isinstance(node.targets[0], ast.Tuple):
                        has_swap = True
            
            # Check for comparisons
            if isinstance(node, ast.Compare):
                has_comparison = True
        
        return has_swap and has_comparison
    
    def _classify_sorting_algorithm(self, func_node: ast.FunctionDef) -> str:
        """Classify type of sorting algorithm."""
        name_lower = func_node.name.lower()
        
        if 'bubble' in name_lower:
            return "bubble_sort"
        elif 'quick' in name_lower:
            return "quicksort"
        elif 'merge' in name_lower:
            return "merge_sort"
        elif 'heap' in name_lower:
            return "heapsort"
        elif 'insertion' in name_lower:
            return "insertion_sort"
        elif 'selection' in name_lower:
            return "selection_sort"
        
        # Analyze implementation patterns
        if self._has_divide_and_conquer_pattern(func_node):
            if self._is_recursive(func_node):
                return "quicksort" if has_pivot else "merge_sort"
        
        loop_analysis = self._analyze_loops(func_node)
        if loop_analysis["max_nesting"] == 2:
            return "bubble_sort"  # Default for O(n²) sorting
        
        return "unknown_sort"
    
    def _is_search_algorithm(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is a search algorithm."""
        return any(keyword in func_node.name.lower() for keyword in ['search', 'find', 'lookup', 'binary'])
    
    def _classify_search_algorithm(self, func_node: ast.FunctionDef) -> str:
        """Classify type of search algorithm."""
        if self._has_binary_search_pattern(func_node):
            return "binary_search"
        
        loop_analysis = self._analyze_loops(func_node)
        if loop_analysis["max_nesting"] == 1:
            return "linear_search"
        
        if 'bfs' in func_node.name.lower():
            return "breadth_first_search"
        elif 'dfs' in func_node.name.lower():
            return "depth_first_search"
        
        return "unknown_search"
    
    def _uses_dynamic_programming(self, func_node: ast.FunctionDef) -> bool:
        """Check if function uses dynamic programming."""
        # Check for DP table/array initialization
        has_table = False
        has_iteration = False
        
        for node in ast.walk(func_node):
            # Check for 2D array initialization
            if isinstance(node, ast.ListComp):
                if isinstance(node.elt, ast.ListComp):
                    has_table = True
            
            # Check for bottom-up iteration
            if isinstance(node, ast.For):
                has_iteration = True
        
        return has_table and has_iteration or self._has_memoization(func_node)
    
    def _estimate_space_complexity(self, func_node: ast.FunctionDef) -> str:
        """Estimate space complexity of a function."""
        # Count data structure allocations
        arrays = 0
        matrices = 0
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.List):
                arrays += 1
            elif isinstance(node, ast.ListComp):
                if isinstance(node.elt, (ast.List, ast.ListComp)):
                    matrices += 1
                else:
                    arrays += 1
        
        if matrices > 0:
            return "O(n²)"
        elif arrays > 1:
            return "O(n)"
        else:
            return "O(1)"
    
    def _is_graph_algorithm(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is a graph algorithm."""
        graph_keywords = ['graph', 'node', 'edge', 'vertex', 'adjacency', 'bfs', 'dfs', 'dijkstra', 'prim', 'kruskal']
        return any(keyword in func_node.name.lower() for keyword in graph_keywords)
    
    def _classify_graph_algorithm(self, func_node: ast.FunctionDef) -> str:
        """Classify type of graph algorithm."""
        name_lower = func_node.name.lower()
        
        if 'bfs' in name_lower or 'breadth' in name_lower:
            return "breadth_first_search"
        elif 'dfs' in name_lower or 'depth' in name_lower:
            return "depth_first_search"
        elif 'dijkstra' in name_lower:
            return "dijkstra"
        elif 'prim' in name_lower:
            return "prim"
        elif 'kruskal' in name_lower:
            return "kruskal"
        elif 'topological' in name_lower:
            return "topological_sort"
        
        return "unknown_graph"
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns and potential leaks."""
        memory_analysis = {
            "memory_allocations": [],
            "potential_leaks": [],
            "large_objects": [],
            "circular_references": [],
            "object_pooling_opportunities": [],
            "memory_optimization_suggestions": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Analyze memory allocations
                allocations = self._detect_memory_allocations(tree)
                for alloc in allocations:
                    alloc["file"] = file_key
                    memory_analysis["memory_allocations"].append(alloc)
                
                # Detect potential memory leaks
                leaks = self._detect_memory_leaks(tree, content)
                for leak in leaks:
                    leak["file"] = file_key
                    memory_analysis["potential_leaks"].append(leak)
                
                # Detect large object allocations
                large_objects = self._detect_large_objects(tree)
                for obj in large_objects:
                    obj["file"] = file_key
                    memory_analysis["large_objects"].append(obj)
                
                # Detect circular references
                circular_refs = self._detect_circular_references(tree)
                for ref in circular_refs:
                    ref["file"] = file_key
                    memory_analysis["circular_references"].append(ref)
                
                # Identify object pooling opportunities
                pooling_ops = self._identify_pooling_opportunities(tree)
                for op in pooling_ops:
                    op["file"] = file_key
                    memory_analysis["object_pooling_opportunities"].append(op)
                
            except:
                continue
        
        # Generate optimization suggestions
        memory_analysis["memory_optimization_suggestions"] = self._generate_memory_optimization_suggestions(memory_analysis)
        
        return memory_analysis
    
    def _detect_memory_allocations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect memory allocation patterns."""
        allocations = []
        
        for node in ast.walk(tree):
            # List/Dict/Set comprehensions
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                allocations.append({
                    "type": "comprehension",
                    "line": node.lineno if hasattr(node, 'lineno') else 0,
                    "kind": node.__class__.__name__
                })
            
            # Large list/dict literals
            elif isinstance(node, ast.List):
                if len(node.elts) > 100:
                    allocations.append({
                        "type": "large_list",
                        "line": node.lineno if hasattr(node, 'lineno') else 0,
                        "size": len(node.elts)
                    })
            
            elif isinstance(node, ast.Dict):
                if len(node.keys) > 100:
                    allocations.append({
                        "type": "large_dict",
                        "line": node.lineno if hasattr(node, 'lineno') else 0,
                        "size": len(node.keys)
                    })
            
            # String concatenation in loops
            elif isinstance(node, ast.For):
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.AugAssign):
                        if isinstance(inner_node.op, ast.Add):
                            if isinstance(inner_node.target, ast.Name):
                                allocations.append({
                                    "type": "string_concatenation_in_loop",
                                    "line": inner_node.lineno if hasattr(inner_node, 'lineno') else 0,
                                    "variable": inner_node.target.id,
                                    "recommendation": "Use list.append() and ''.join() instead"
                                })
        
        return allocations
    
    def _detect_memory_leaks(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        
        # Check for unclosed resources
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # File operations without context manager
                    if node.func.id == 'open':
                        # Check if it's in a with statement
                        parent = None  # Would need parent tracking
                        if not self._is_in_with_statement(node, tree):
                            leaks.append({
                                "type": "unclosed_file",
                                "line": node.lineno,
                                "recommendation": "Use 'with open() as f:' pattern"
                            })
        
        # Check for global accumulating data structures
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    if any(keyword in name.lower() for keyword in ['cache', 'buffer', 'pool']):
                        leaks.append({
                            "type": "global_accumulator",
                            "line": node.lineno,
                            "variable": name,
                            "recommendation": "Consider bounded cache or periodic cleanup"
                        })
        
        # Check for event listeners without cleanup
        if 'addEventListener' in content or 'on(' in content or '.bind(' in content:
            leaks.append({
                "type": "event_listener_leak",
                "recommendation": "Ensure event listeners are properly removed"
            })
        
        return leaks
    
    def _is_in_with_statement(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if a node is within a with statement."""
        # Simplified check - would need proper parent tracking
        for with_node in ast.walk(tree):
            if isinstance(with_node, ast.With):
                for item in with_node.items:
                    if item.context_expr == node:
                        return True
        return False
    
    def _detect_large_objects(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect large object allocations."""
        large_objects = []
        
        for node in ast.walk(tree):
            # Multiplication creating large lists
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Mult):
                    if isinstance(node.left, ast.List) or isinstance(node.right, ast.List):
                        large_objects.append({
                            "type": "list_multiplication",
                            "line": node.lineno if hasattr(node, 'lineno') else 0,
                            "warning": "Large list creation via multiplication"
                        })
            
            # Range with large values
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'range':
                    if node.args:
                        if isinstance(node.args[-1], ast.Constant):
                            if node.args[-1].value > 1000000:
                                large_objects.append({
                                    "type": "large_range",
                                    "line": node.lineno,
                                    "size": node.args[-1].value,
                                    "recommendation": "Consider using generator or itertools"
                                })
        
        return large_objects
    
    def _detect_circular_references(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect potential circular references."""
        circular_refs = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for parent-child circular references
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        if method.name == '__init__':
                            for stmt in ast.walk(method):
                                if isinstance(stmt, ast.Assign):
                                    # Check for self.parent = parent pattern
                                    for target in stmt.targets:
                                        if isinstance(target, ast.Attribute):
                                            if target.attr in ['parent', 'owner', 'container']:
                                                circular_refs.append({
                                                    "type": "parent_child_reference",
                                                    "class": node.name,
                                                    "line": stmt.lineno,
                                                    "attribute": target.attr,
                                                    "recommendation": "Consider using weakref"
                                                })
        
        return circular_refs
    
    def _identify_pooling_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify object pooling opportunities."""
        opportunities = []
        
        for node in ast.walk(tree):
            # Repeated object creation in loops
            if isinstance(node, ast.For):
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Name):
                            # Creating objects in loop
                            if inner_node.func.id in ['dict', 'list', 'set'] or inner_node.func.id[0].isupper():
                                opportunities.append({
                                    "type": "repeated_allocation",
                                    "line": inner_node.lineno if hasattr(inner_node, 'lineno') else 0,
                                    "object_type": inner_node.func.id,
                                    "recommendation": "Consider object pooling or pre-allocation"
                                })
                                break
        
        return opportunities
    
    def _generate_memory_optimization_suggestions(self, memory_data: Dict[str, Any]) -> List[str]:
        """Generate memory optimization suggestions."""
        suggestions = []
        
        if memory_data["potential_leaks"]:
            suggestions.append("Fix potential memory leaks by properly closing resources")
        
        if memory_data["circular_references"]:
            suggestions.append("Use weakref to break circular references")
        
        if any(alloc["type"] == "string_concatenation_in_loop" for alloc in memory_data["memory_allocations"]):
            suggestions.append("Replace string concatenation in loops with list.append() and join()")
        
        if memory_data["large_objects"]:
            suggestions.append("Consider using generators or lazy evaluation for large data structures")
        
        if memory_data["object_pooling_opportunities"]:
            suggestions.append("Implement object pooling for frequently created objects")
        
        suggestions.extend([
            "Use __slots__ in classes to reduce memory overhead",
            "Consider using array.array for homogeneous numeric data",
            "Use sys.getsizeof() to profile memory usage",
            "Implement memory profiling with memory_profiler or tracemalloc"
        ])
        
        return suggestions
    
    def _analyze_database_performance(self) -> Dict[str, Any]:
        """Analyze database query patterns and performance issues."""
        db_analysis = {
            "n_plus_one_queries": [],
            "missing_indexes": [],
            "inefficient_queries": [],
            "transaction_issues": [],
            "connection_pooling": [],
            "query_optimization_opportunities": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect N+1 query problems
                n_plus_one = self._detect_n_plus_one_queries(tree)
                for issue in n_plus_one:
                    issue["file"] = file_key
                    db_analysis["n_plus_one_queries"].append(issue)
                
                # Detect missing indexes
                missing_indexes = self._detect_missing_indexes(tree, content)
                for index in missing_indexes:
                    index["file"] = file_key
                    db_analysis["missing_indexes"].append(index)
                
                # Detect inefficient queries
                inefficient = self._detect_inefficient_queries(tree, content)
                for query in inefficient:
                    query["file"] = file_key
                    db_analysis["inefficient_queries"].append(query)
                
                # Detect transaction issues
                trans_issues = self._detect_transaction_issues(tree)
                for issue in trans_issues:
                    issue["file"] = file_key
                    db_analysis["transaction_issues"].append(issue)
                
            except:
                continue
        
        return db_analysis
    
    def _detect_n_plus_one_queries(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect N+1 query problems."""
        n_plus_one = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Look for database queries inside loops
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Attribute):
                            # Common ORM query methods
                            if inner_node.func.attr in ['get', 'filter', 'all', 'first', 'query', 'select']:
                                n_plus_one.append({
                                    "type": "potential_n_plus_one",
                                    "line": inner_node.lineno,
                                    "method": inner_node.func.attr,
                                    "recommendation": "Use select_related() or prefetch_related() for Django, joinedload() for SQLAlchemy"
                                })
        
        return n_plus_one
    
    def _detect_missing_indexes(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect potential missing database indexes."""
        missing_indexes = []
        
        # Look for filter/where conditions on non-indexed fields
        filter_fields = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['filter', 'where', 'filter_by']:
                        # Extract field names from arguments
                        for arg in node.args:
                            if isinstance(arg, ast.Compare):
                                if isinstance(arg.left, ast.Attribute):
                                    filter_fields.add(arg.left.attr)
                        
                        for keyword in node.keywords:
                            filter_fields.add(keyword.arg)
        
        # Check if these fields are likely indexed
        for field in filter_fields:
            if field not in ['id', 'pk', 'primary_key'] and not field.endswith('_id'):
                missing_indexes.append({
                    "field": field,
                    "recommendation": f"Consider adding index on '{field}' field"
                })
        
        return missing_indexes
    
    def _detect_inefficient_queries(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect inefficient database queries."""
        inefficient = []
        
        # Look for SELECT * patterns
        if 'SELECT *' in content or '.all()' in content:
            inefficient.append({
                "type": "select_all",
                "recommendation": "Select only required fields to reduce data transfer"
            })
        
        # Look for missing limit/pagination
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['all', 'filter']:
                        # Check if followed by limit/slice
                        has_limit = False
                        # Simplified check - would need more context
                        if not has_limit:
                            inefficient.append({
                                "type": "missing_pagination",
                                "line": node.lineno,
                                "recommendation": "Add pagination or limit to queries"
                            })
        
        # Look for multiple similar queries
        query_patterns = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['get', 'filter', 'query']:
                        key = f"{node.func.attr}"
                        query_patterns[key].append(node.lineno)
        
        for pattern, lines in query_patterns.items():
            if len(lines) > 3:
                inefficient.append({
                    "type": "repeated_queries",
                    "pattern": pattern,
                    "occurrences": len(lines),
                    "recommendation": "Consider batching similar queries"
                })
        
        return inefficient
    
    def _detect_transaction_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect database transaction issues."""
        issues = []
        
        # Look for missing transaction blocks
        has_write_operations = False
        has_transaction = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['save', 'update', 'delete', 'insert', 'execute']:
                        has_write_operations = True
                    if node.func.attr in ['transaction', 'atomic', 'begin', 'commit']:
                        has_transaction = True
        
        if has_write_operations and not has_transaction:
            issues.append({
                "type": "missing_transaction",
                "recommendation": "Wrap multiple write operations in a transaction"
            })
        
        return issues
    
    def _analyze_concurrency_performance(self) -> Dict[str, Any]:
        """Analyze concurrency and GIL impact."""
        concurrency_analysis = {
            "gil_bottlenecks": [],
            "async_opportunities": [],
            "thread_safety_issues": [],
            "parallelization_opportunities": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect GIL bottlenecks
                gil_issues = self._detect_gil_bottlenecks(tree)
                for issue in gil_issues:
                    issue["file"] = file_key
                    concurrency_analysis["gil_bottlenecks"].append(issue)
                
                # Identify async opportunities
                async_ops = self._identify_async_opportunities(tree)
                for op in async_ops:
                    op["file"] = file_key
                    concurrency_analysis["async_opportunities"].append(op)
                
                # Detect thread safety issues
                safety_issues = self._detect_thread_safety_issues(tree)
                for issue in safety_issues:
                    issue["file"] = file_key
                    concurrency_analysis["thread_safety_issues"].append(issue)
                
                # Identify parallelization opportunities
                parallel_ops = self._identify_parallelization_opportunities(tree)
                for op in parallel_ops:
                    op["file"] = file_key
                    concurrency_analysis["parallelization_opportunities"].append(op)
                
            except:
                continue
        
        return concurrency_analysis
    
    def _detect_gil_bottlenecks(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect potential GIL bottlenecks."""
        bottlenecks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for CPU-intensive operations
                has_heavy_computation = False
                
                for inner_node in ast.walk(node):
                    # Nested loops with computation
                    if isinstance(inner_node, ast.For):
                        for inner_inner in ast.walk(inner_node):
                            if isinstance(inner_inner, ast.BinOp):
                                has_heavy_computation = True
                                break
                
                if has_heavy_computation:
                    # Check if using threading
                    uses_threading = any('thread' in str(n).lower() for n in ast.walk(node))
                    
                    if uses_threading:
                        bottlenecks.append({
                            "function": node.name,
                            "line": node.lineno,
                            "issue": "CPU-bound operation with threading",
                            "recommendation": "Use multiprocessing instead of threading for CPU-bound tasks"
                        })
        
        return bottlenecks
    
    def _identify_async_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify opportunities for async/await."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_io = False
                is_async = isinstance(node, ast.AsyncFunctionDef)
                
                for inner_node in ast.walk(node):
                    # Check for I/O operations
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Name):
                            if inner_node.func.id in ['open', 'read', 'write']:
                                has_io = True
                        elif isinstance(inner_node.func, ast.Attribute):
                            if inner_node.func.attr in ['get', 'post', 'request', 'query']:
                                has_io = True
                
                if has_io and not is_async:
                    opportunities.append({
                        "function": node.name,
                        "line": node.lineno,
                        "type": "io_bound",
                        "recommendation": "Consider making this function async"
                    })
        
        return opportunities
    
    def _detect_thread_safety_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect thread safety issues."""
        issues = []
        
        # Look for shared mutable state without locks
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    issues.append({
                        "type": "global_mutable_state",
                        "variable": name,
                        "line": node.lineno,
                        "recommendation": "Use threading.Lock() or consider thread-local storage"
                    })
        
        return issues
    
    def _identify_parallelization_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify parallelization opportunities."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if loop iterations are independent
                loop_var = None
                if isinstance(node.target, ast.Name):
                    loop_var = node.target.id
                
                # Simple heuristic: if loop doesn't modify shared state
                modifies_external = False
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Assign):
                        for target in inner_node.targets:
                            if isinstance(target, ast.Name):
                                if target.id != loop_var:
                                    modifies_external = True
                
                if not modifies_external:
                    opportunities.append({
                        "type": "parallelizable_loop",
                        "line": node.lineno,
                        "recommendation": "Consider using multiprocessing.Pool or concurrent.futures"
                    })
        
        return opportunities
    
    def _identify_cpu_intensive_operations(self) -> Dict[str, Any]:
        """Identify CPU-intensive operations."""
        cpu_operations = {
            "mathematical_operations": [],
            "string_operations": [],
            "data_processing": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    # Mathematical operations
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute):
                            if node.func.attr in ['sqrt', 'pow', 'exp', 'log', 'sin', 'cos']:
                                cpu_operations["mathematical_operations"].append({
                                    "operation": node.func.attr,
                                    "file": file_key,
                                    "line": node.lineno
                                })
                    
                    # Regular expressions
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute):
                            if node.func.attr in ['match', 'search', 'findall', 'compile']:
                                if isinstance(node.func.value, ast.Name):
                                    if node.func.value.id == 're':
                                        cpu_operations["string_operations"].append({
                                            "type": "regex",
                                            "file": file_key,
                                            "line": node.lineno
                                        })
                    
                    # JSON/XML parsing
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in ['loads', 'dumps', 'parse']:
                                cpu_operations["data_processing"].append({
                                    "type": "serialization",
                                    "file": file_key,
                                    "line": node.lineno
                                })
                
            except:
                continue
        
        return cpu_operations
    
    def _detect_performance_anti_patterns(self) -> Dict[str, Any]:
        """Detect common performance anti-patterns."""
        anti_patterns = {
            "premature_optimization": [],
            "inefficient_string_operations": [],
            "unnecessary_list_comprehensions": [],
            "excessive_function_calls": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    # String concatenation in loops
                    if isinstance(node, ast.For):
                        for inner_node in ast.walk(node):
                            if isinstance(inner_node, ast.AugAssign):
                                if isinstance(inner_node.op, ast.Add):
                                    if isinstance(inner_node.target, ast.Name):
                                        anti_patterns["inefficient_string_operations"].append({
                                            "type": "string_concatenation_in_loop",
                                            "file": file_key,
                                            "line": inner_node.lineno,
                                            "variable": inner_node.target.id
                                        })
                    
                    # List comprehension for side effects only
                    if isinstance(node, ast.ListComp):
                        # Check if result is not used
                        anti_patterns["unnecessary_list_comprehensions"].append({
                            "file": file_key,
                            "line": node.lineno if hasattr(node, 'lineno') else 0,
                            "recommendation": "Use a regular for loop for side effects"
                        })
                
            except:
                continue
        
        return anti_patterns
    
    def _identify_optimization_opportunities(self) -> Dict[str, Any]:
        """Identify optimization opportunities."""
        optimizations = {
            "cacheable_functions": [],
            "vectorization_opportunities": [],
            "lazy_evaluation_opportunities": [],
            "batch_processing_opportunities": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if function is pure and cacheable
                        if self._is_pure_function(node):
                            optimizations["cacheable_functions"].append({
                                "function": node.name,
                                "file": file_key,
                                "line": node.lineno,
                                "recommendation": "Consider using @lru_cache decorator"
                            })
                        
                        # Check for vectorization opportunities
                        if self._has_vectorization_opportunity(node):
                            optimizations["vectorization_opportunities"].append({
                                "function": node.name,
                                "file": file_key,
                                "line": node.lineno,
                                "recommendation": "Consider using NumPy for vectorized operations"
                            })
                
            except:
                continue
        
        return optimizations
    
    def _is_pure_function(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is pure (no side effects)."""
        for node in ast.walk(func_node):
            # Check for I/O operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['print', 'open', 'input']:
                        return False
            
            # Check for global/nonlocal modifications
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return False
            
            # Check for attribute assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        return False
        
        return True
    
    def _has_vectorization_opportunity(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has vectorization opportunities."""
        # Look for element-wise operations on lists
        for node in ast.walk(func_node):
            if isinstance(node, ast.For):
                # Check for mathematical operations in loop
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.BinOp):
                        if isinstance(inner_node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                            return True
        return False
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        complexity = self._analyze_algorithmic_complexity()
        memory = self._analyze_memory_patterns()
        database = self._analyze_database_performance()
        concurrency = self._analyze_concurrency_performance()
        
        metrics = {
            "performance_score": 0,
            "bottlenecks": [],
            "critical_issues": [],
            "recommendations": []
        }
        
        # Calculate performance score
        score = 100
        
        # Penalize for high complexity
        high_complexity_count = complexity["complexity_distribution"].get("exponential", 0) + \
                               complexity["complexity_distribution"].get("factorial", 0)
        if high_complexity_count > 0:
            score -= 20
            metrics["critical_issues"].append(f"{high_complexity_count} functions with exponential/factorial complexity")
        
        # Penalize for memory issues
        if memory["potential_leaks"]:
            score -= 15
            metrics["critical_issues"].append(f"{len(memory['potential_leaks'])} potential memory leaks")
        
        # Penalize for database issues
        if database["n_plus_one_queries"]:
            score -= 10
            metrics["bottlenecks"].append(f"{len(database['n_plus_one_queries'])} N+1 query problems")
        
        # Penalize for concurrency issues
        if concurrency["gil_bottlenecks"]:
            score -= 10
            metrics["bottlenecks"].append(f"{len(concurrency['gil_bottlenecks'])} GIL bottlenecks")
        
        metrics["performance_score"] = max(score, 0)
        
        # Generate recommendations
        metrics["recommendations"] = [
            "Profile code with cProfile to identify actual bottlenecks",
            "Use memory_profiler to track memory usage",
            "Implement caching for expensive computations",
            "Consider async/await for I/O-bound operations",
            "Use multiprocessing for CPU-bound parallel tasks"
        ]
        
        return metrics