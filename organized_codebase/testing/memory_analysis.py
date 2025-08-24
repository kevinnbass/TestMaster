"""
Memory Usage Pattern Analysis and Leak Detection Module
========================================================

Implements comprehensive memory analysis capabilities:
- Memory allocation pattern detection
- Leak detection in common scenarios
- Reference cycle identification
- Memory growth pattern analysis
- Object lifetime analysis
- Memory optimization recommendations
- GC pressure analysis
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class MemoryAnalyzer(BaseAnalyzer):
    """Analyzer for memory usage patterns and leak detection."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        
        # Memory leak patterns
        self.leak_patterns = {
            "circular_reference": [
                "self reference in __init__",
                "bidirectional parent-child",
                "event listener not removed"
            ],
            "resource_not_freed": [
                "file handle not closed",
                "socket not closed",
                "database connection leak",
                "thread not joined"
            ],
            "cache_unbounded": [
                "unlimited dict/list growth",
                "no cache eviction",
                "memoization without limit"
            ],
            "global_accumulation": [
                "global list append",
                "module-level dict growth",
                "class variable accumulation"
            ]
        }
        
        # Memory-intensive operations
        self.memory_intensive_ops = {
            "large_allocation": ["numpy.zeros", "numpy.ones", "list comprehension", "dict comprehension"],
            "copy_operations": ["copy.deepcopy", "list()", "dict()", "[:]"],
            "string_operations": ["str.join", "+= for strings", "string multiplication"],
            "data_loading": ["pandas.read_csv", "json.load", "pickle.load", "np.load"]
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive memory usage analysis."""
        print("[INFO] Analyzing Memory Usage Patterns...")
        
        results = {
            "memory_allocations": self._analyze_memory_allocations(),
            "leak_detection": self._detect_memory_leaks(),
            "reference_cycles": self._detect_reference_cycles(),
            "memory_growth": self._analyze_memory_growth_patterns(),
            "object_lifetime": self._analyze_object_lifetime(),
            "gc_pressure": self._analyze_gc_pressure(),
            "optimization_opportunities": self._identify_memory_optimizations(),
            "memory_metrics": self._calculate_memory_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} memory aspects")
        return results
    
    def _analyze_memory_allocations(self) -> Dict[str, Any]:
        """Analyze memory allocation patterns in the code."""
        allocations = {
            "large_allocations": [],
            "frequent_allocations": [],
            "allocation_hotspots": defaultdict(list),
            "allocation_patterns": defaultdict(int)
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect large allocations
                large_allocs = self._detect_large_allocations(tree, content)
                for alloc in large_allocs:
                    alloc["file"] = file_key
                    allocations["large_allocations"].append(alloc)
                
                # Detect allocation patterns
                patterns = self._detect_allocation_patterns(tree)
                for pattern, count in patterns.items():
                    allocations["allocation_patterns"][pattern] += count
                
                # Find allocation hotspots
                hotspots = self._find_allocation_hotspots(tree)
                for hotspot in hotspots:
                    allocations["allocation_hotspots"][file_key].append(hotspot)
                
            except:
                continue
        
        # Identify frequently allocated objects
        allocations["frequent_allocations"] = self._identify_frequent_allocations(allocations)
        
        return allocations
    
    def _detect_large_allocations(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect potentially large memory allocations."""
        large_allocations = []
        
        for node in ast.walk(tree):
            # NumPy array allocations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['zeros', 'ones', 'empty', 'full']:
                        size_info = self._extract_array_size(node)
                        if size_info and size_info["estimated_size"] > 1000000:  # > 1MB
                            large_allocations.append({
                                "type": "numpy_array",
                                "line": node.lineno,
                                "size": size_info,
                                "risk": "high" if size_info["estimated_size"] > 100000000 else "medium"
                            })
                    
                    # Pandas operations
                    elif node.func.attr in ['read_csv', 'read_excel', 'read_json']:
                        large_allocations.append({
                            "type": "data_loading",
                            "operation": node.func.attr,
                            "line": node.lineno,
                            "risk": "medium",
                            "note": "Consider chunking or streaming"
                        })
            
            # List comprehensions with large ranges
            elif isinstance(node, ast.ListComp):
                if self._is_large_comprehension(node):
                    large_allocations.append({
                        "type": "list_comprehension",
                        "line": node.lineno,
                        "risk": "medium",
                        "recommendation": "Consider generator expression"
                    })
            
            # String concatenation in loops
            elif isinstance(node, ast.For):
                if self._has_string_concatenation_in_loop(node):
                    large_allocations.append({
                        "type": "string_concatenation",
                        "line": node.lineno,
                        "risk": "high",
                        "recommendation": "Use str.join() or StringIO"
                    })
        
        return large_allocations
    
    def _detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        leaks = {
            "resource_leaks": [],
            "reference_leaks": [],
            "cache_leaks": [],
            "global_leaks": [],
            "leak_risk_score": 0
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect resource leaks
                resource_leaks = self._detect_resource_leaks(tree, content)
                for leak in resource_leaks:
                    leak["file"] = file_key
                    leaks["resource_leaks"].append(leak)
                
                # Detect reference leaks
                ref_leaks = self._detect_reference_leaks(tree)
                for leak in ref_leaks:
                    leak["file"] = file_key
                    leaks["reference_leaks"].append(leak)
                
                # Detect cache-related leaks
                cache_leaks = self._detect_cache_leaks(tree, content)
                for leak in cache_leaks:
                    leak["file"] = file_key
                    leaks["cache_leaks"].append(leak)
                
                # Detect global accumulation
                global_leaks = self._detect_global_accumulation(tree)
                for leak in global_leaks:
                    leak["file"] = file_key
                    leaks["global_leaks"].append(leak)
                
            except:
                continue
        
        # Calculate overall leak risk score
        leaks["leak_risk_score"] = self._calculate_leak_risk_score(leaks)
        
        return leaks
    
    def _detect_resource_leaks(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect resource leaks (files, sockets, etc.)."""
        resource_leaks = []
        
        # Track resource allocations and deallocations
        resources = defaultdict(list)
        deallocations = set()
        
        for node in ast.walk(tree):
            # File operations without context manager
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    # Check if it's in a with statement
                    if not self._is_in_with_statement(tree, node):
                        # Check if close() is called
                        if not self._has_matching_close(tree, node):
                            resource_leaks.append({
                                "type": "file_handle",
                                "line": node.lineno,
                                "severity": "high",
                                "fix": "Use 'with open()' context manager"
                            })
                
                # Socket operations
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'socket':
                        if not self._has_matching_close(tree, node):
                            resource_leaks.append({
                                "type": "socket",
                                "line": node.lineno,
                                "severity": "high",
                                "fix": "Ensure socket.close() is called"
                            })
                    
                    # Database connections
                    elif node.func.attr in ['connect', 'create_connection']:
                        if not self._has_matching_close(tree, node):
                            resource_leaks.append({
                                "type": "database_connection",
                                "line": node.lineno,
                                "severity": "high",
                                "fix": "Use connection pooling or ensure close()"
                            })
        
        # Thread leaks
        if 'Thread' in content or 'threading' in content:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr == 'start':
                            if not self._has_thread_join(tree, node):
                                resource_leaks.append({
                                    "type": "thread",
                                    "line": node.lineno,
                                    "severity": "medium",
                                    "fix": "Ensure thread.join() is called"
                                })
        
        return resource_leaks
    
    def _detect_reference_cycles(self) -> Dict[str, Any]:
        """Detect potential reference cycles."""
        cycles = {
            "self_references": [],
            "circular_dependencies": [],
            "event_listener_cycles": [],
            "cycle_patterns": defaultdict(int)
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check for self-references in __init__
                        init_method = self._get_init_method(node)
                        if init_method:
                            self_refs = self._detect_self_references(init_method)
                            for ref in self_refs:
                                ref["file"] = file_key
                                ref["class"] = node.name
                                cycles["self_references"].append(ref)
                        
                        # Check for circular parent-child relationships
                        circular = self._detect_circular_relationships(node)
                        for circ in circular:
                            circ["file"] = file_key
                            circ["class"] = node.name
                            cycles["circular_dependencies"].append(circ)
                
                # Check for event listener patterns
                listeners = self._detect_event_listener_cycles(tree)
                for listener in listeners:
                    listener["file"] = file_key
                    cycles["event_listener_cycles"].append(listener)
                
            except:
                continue
        
        return cycles
    
    def _detect_self_references(self, init_method: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Detect self-references that might cause cycles."""
        self_refs = []
        
        for node in ast.walk(init_method):
            if isinstance(node, ast.Assign):
                # Check for patterns like self.parent = self
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == 'self':
                            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                                self_refs.append({
                                    "line": node.lineno,
                                    "attribute": target.attr,
                                    "pattern": "direct_self_reference",
                                    "risk": "high"
                                })
                            
                            # Check for self in collections
                            elif isinstance(node.value, (ast.List, ast.Set)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Name) and elt.id == 'self':
                                        self_refs.append({
                                            "line": node.lineno,
                                            "attribute": target.attr,
                                            "pattern": "self_in_collection",
                                            "risk": "high"
                                        })
        
        return self_refs
    
    def _detect_circular_relationships(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Detect circular parent-child relationships."""
        circular_patterns = []
        
        init_method = self._get_init_method(class_node)
        if not init_method:
            return circular_patterns
        
        # Look for parent-child patterns
        for node in ast.walk(init_method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if target.attr in ['parent', 'owner', 'container']:
                            # Check if child adds itself to parent
                            for inner_node in ast.walk(init_method):
                                if isinstance(inner_node, ast.Call):
                                    if isinstance(inner_node.func, ast.Attribute):
                                        if inner_node.func.attr in ['append', 'add']:
                                            circular_patterns.append({
                                                "line": node.lineno,
                                                "pattern": "bidirectional_parent_child",
                                                "risk": "medium"
                                            })
        
        return circular_patterns
    
    def _detect_event_listener_cycles(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect event listener patterns that might cause cycles."""
        listener_cycles = []
        
        # Track addEventListener/removeEventListener patterns
        add_listeners = []
        remove_listeners = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if 'add' in node.func.attr.lower() and 'listener' in node.func.attr.lower():
                        add_listeners.append(node.lineno)
                    elif 'remove' in node.func.attr.lower() and 'listener' in node.func.attr.lower():
                        remove_listeners.append(node.lineno)
                    elif node.func.attr in ['subscribe', 'on', 'bind']:
                        add_listeners.append(node.lineno)
                    elif node.func.attr in ['unsubscribe', 'off', 'unbind']:
                        remove_listeners.append(node.lineno)
        
        # Check for unbalanced listeners
        if len(add_listeners) > len(remove_listeners):
            listener_cycles.append({
                "pattern": "unbalanced_listeners",
                "add_count": len(add_listeners),
                "remove_count": len(remove_listeners),
                "risk": "medium",
                "recommendation": "Ensure all event listeners are properly removed"
            })
        
        return listener_cycles
    
    def _detect_cache_leaks(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect unbounded cache growth."""
        cache_leaks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for cache attributes
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                if 'cache' in target.id.lower() or 'memo' in target.id.lower():
                                    # Check if there's a size limit
                                    if not self._has_cache_eviction(node):
                                        cache_leaks.append({
                                            "class": node.name,
                                            "line": item.lineno,
                                            "type": "unbounded_cache",
                                            "risk": "high",
                                            "recommendation": "Implement LRU cache or size limit"
                                        })
            
            # Check for @lru_cache without maxsize
            elif isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'lru_cache':
                        cache_leaks.append({
                            "function": node.name,
                            "line": node.lineno,
                            "type": "unlimited_lru_cache",
                            "risk": "medium",
                            "recommendation": "Add maxsize parameter to @lru_cache"
                        })
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name) and decorator.func.id == 'lru_cache':
                            # Check if maxsize is specified
                            has_maxsize = False
                            for keyword in decorator.keywords:
                                if keyword.arg == 'maxsize':
                                    has_maxsize = True
                            if not has_maxsize:
                                cache_leaks.append({
                                    "function": node.name,
                                    "line": node.lineno,
                                    "type": "lru_cache_no_maxsize",
                                    "risk": "medium",
                                    "recommendation": "Specify maxsize for @lru_cache"
                                })
        
        return cache_leaks
    
    def _detect_global_accumulation(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect global variable accumulation patterns."""
        global_leaks = []
        
        # Find global lists/dicts
        global_collections = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if it's at module level
                if self._is_module_level(tree, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                                global_collections[target.id] = {
                                    "type": type(node.value).__name__,
                                    "line": node.lineno
                                }
        
        # Check for append/add operations on globals
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['append', 'add', 'extend', 'update']:
                        if isinstance(node.func.value, ast.Name):
                            if node.func.value.id in global_collections:
                                global_leaks.append({
                                    "variable": node.func.value.id,
                                    "operation": node.func.attr,
                                    "line": node.lineno,
                                    "type": "global_accumulation",
                                    "risk": "high",
                                    "recommendation": "Consider bounded collection or cleanup"
                                })
        
        return global_leaks
    
    def _analyze_memory_growth_patterns(self) -> Dict[str, Any]:
        """Analyze patterns that lead to memory growth."""
        growth_patterns = {
            "unbounded_growth": [],
            "quadratic_growth": [],
            "recursive_allocations": [],
            "growth_risk_areas": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect unbounded collection growth
                unbounded = self._detect_unbounded_growth(tree)
                for item in unbounded:
                    item["file"] = file_key
                    growth_patterns["unbounded_growth"].append(item)
                
                # Detect quadratic memory growth patterns
                quadratic = self._detect_quadratic_growth(tree)
                for item in quadratic:
                    item["file"] = file_key
                    growth_patterns["quadratic_growth"].append(item)
                
                # Detect recursive allocations
                recursive = self._detect_recursive_allocations(tree)
                for item in recursive:
                    item["file"] = file_key
                    growth_patterns["recursive_allocations"].append(item)
                
            except:
                continue
        
        return growth_patterns
    
    def _detect_unbounded_growth(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect unbounded collection growth patterns."""
        unbounded = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check for while True with append
                if self._is_infinite_loop(node):
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.Call):
                            if isinstance(inner.func, ast.Attribute):
                                if inner.func.attr in ['append', 'add', 'extend']:
                                    unbounded.append({
                                        "line": node.lineno,
                                        "pattern": "infinite_loop_append",
                                        "risk": "critical",
                                        "description": "Infinite loop with collection growth"
                                    })
            
            elif isinstance(node, ast.For):
                # Check for large range with collection building
                if self._has_large_range(node):
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.Call):
                            if isinstance(inner.func, ast.Attribute):
                                if inner.func.attr in ['append', 'add']:
                                    unbounded.append({
                                        "line": node.lineno,
                                        "pattern": "large_range_accumulation",
                                        "risk": "high",
                                        "recommendation": "Consider generator or chunking"
                                    })
        
        return unbounded
    
    def _detect_quadratic_growth(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect O(n²) memory growth patterns."""
        quadratic = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for nested loops with string concatenation
                for inner in ast.walk(node):
                    if isinstance(inner, ast.For) and inner != node:
                        # Nested loop found
                        for op in ast.walk(inner):
                            if isinstance(op, ast.AugAssign):
                                if isinstance(op.op, ast.Add):
                                    if isinstance(op.target, ast.Name):
                                        quadratic.append({
                                            "line": node.lineno,
                                            "pattern": "nested_loop_concatenation",
                                            "complexity": "O(n²)",
                                            "risk": "high",
                                            "recommendation": "Use list and join()"
                                        })
        
        return quadratic
    
    def _detect_recursive_allocations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect recursive functions with allocations."""
        recursive = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function calls itself
                if self._is_recursive_function(node):
                    # Check for allocations in recursive function
                    for inner in ast.walk(node):
                        if isinstance(inner, (ast.List, ast.Dict, ast.Set)):
                            recursive.append({
                                "function": node.name,
                                "line": node.lineno,
                                "pattern": "recursive_allocation",
                                "risk": "medium",
                                "recommendation": "Consider iterative approach or memoization"
                            })
                            break
        
        return recursive
    
    def _analyze_object_lifetime(self) -> Dict[str, Any]:
        """Analyze object lifetime and scope."""
        lifetime_analysis = {
            "short_lived_large_objects": [],
            "long_lived_temporaries": [],
            "scope_issues": [],
            "lifetime_patterns": defaultdict(int)
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for large objects with short lifetime
                        short_lived = self._detect_short_lived_large_objects(node)
                        for obj in short_lived:
                            obj["file"] = file_key
                            obj["function"] = node.name
                            lifetime_analysis["short_lived_large_objects"].append(obj)
                        
                        # Check for temporaries that could be optimized
                        temps = self._detect_unnecessary_temporaries(node)
                        for temp in temps:
                            temp["file"] = file_key
                            temp["function"] = node.name
                            lifetime_analysis["long_lived_temporaries"].append(temp)
                
            except:
                continue
        
        return lifetime_analysis
    
    def _detect_short_lived_large_objects(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Detect large objects created and discarded quickly."""
        short_lived = []
        
        # Track variable creation and last use
        var_lifetime = {}
        
        for i, node in enumerate(func_node.body):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if it's a large allocation
                        if isinstance(node.value, ast.Call):
                            if self._is_large_allocation(node.value):
                                var_lifetime[target.id] = {
                                    "created": i,
                                    "last_used": i,
                                    "line": node.lineno
                                }
        
        # Track variable usage
        for i, node in enumerate(func_node.body):
            for name_node in ast.walk(node):
                if isinstance(name_node, ast.Name):
                    if name_node.id in var_lifetime:
                        var_lifetime[name_node.id]["last_used"] = i
        
        # Find short-lived large objects
        for var, info in var_lifetime.items():
            lifetime = info["last_used"] - info["created"]
            if lifetime <= 2:  # Used within 2 statements
                short_lived.append({
                    "variable": var,
                    "line": info["line"],
                    "lifetime_statements": lifetime,
                    "risk": "optimization_opportunity",
                    "recommendation": "Consider streaming or lazy evaluation"
                })
        
        return short_lived
    
    def _detect_unnecessary_temporaries(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Detect unnecessary temporary variables."""
        temporaries = []
        
        # Look for patterns like: temp = expr; return temp
        for i in range(len(func_node.body) - 1):
            curr = func_node.body[i]
            next_stmt = func_node.body[i + 1]
            
            if isinstance(curr, ast.Assign) and isinstance(next_stmt, ast.Return):
                if isinstance(next_stmt.value, ast.Name):
                    for target in curr.targets:
                        if isinstance(target, ast.Name):
                            if target.id == next_stmt.value.id:
                                temporaries.append({
                                    "variable": target.id,
                                    "line": curr.lineno,
                                    "pattern": "unnecessary_temporary",
                                    "recommendation": "Return expression directly"
                                })
        
        return temporaries
    
    def _analyze_gc_pressure(self) -> Dict[str, Any]:
        """Analyze garbage collection pressure indicators."""
        gc_analysis = {
            "high_allocation_rate": [],
            "reference_counting_issues": [],
            "gc_unfriendly_patterns": [],
            "gc_pressure_score": 0
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Check for high allocation rate patterns
                high_alloc = self._detect_high_allocation_rate(tree)
                for pattern in high_alloc:
                    pattern["file"] = file_key
                    gc_analysis["high_allocation_rate"].append(pattern)
                
                # Check for reference counting issues
                ref_issues = self._detect_reference_counting_issues(tree)
                for issue in ref_issues:
                    issue["file"] = file_key
                    gc_analysis["reference_counting_issues"].append(issue)
                
                # Check for GC-unfriendly patterns
                if '__del__' in content:
                    gc_analysis["gc_unfriendly_patterns"].append({
                        "file": file_key,
                        "pattern": "__del__method",
                        "risk": "high",
                        "reason": "__del__ can interfere with garbage collection"
                    })
                
            except:
                continue
        
        # Calculate GC pressure score
        gc_analysis["gc_pressure_score"] = self._calculate_gc_pressure_score(gc_analysis)
        
        return gc_analysis
    
    def _detect_high_allocation_rate(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect patterns causing high allocation rate."""
        high_allocation = []
        
        for node in ast.walk(tree):
            # Allocations in tight loops
            if isinstance(node, (ast.For, ast.While)):
                allocation_count = 0
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.List, ast.Dict, ast.Set)):
                        allocation_count += 1
                    elif isinstance(inner, ast.Call):
                        if isinstance(inner.func, ast.Name):
                            if inner.func.id in ['list', 'dict', 'set']:
                                allocation_count += 1
                
                if allocation_count > 2:
                    high_allocation.append({
                        "line": node.lineno,
                        "pattern": "loop_allocations",
                        "allocation_count": allocation_count,
                        "risk": "high" if allocation_count > 5 else "medium",
                        "recommendation": "Move allocations outside loop"
                    })
        
        return high_allocation
    
    def _detect_reference_counting_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect patterns that complicate reference counting."""
        ref_issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for __slots__ usage (good for memory)
                has_slots = False
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == '__slots__':
                                has_slots = True
                
                # Large classes without __slots__
                method_count = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
                if method_count > 10 and not has_slots:
                    ref_issues.append({
                        "class": node.name,
                        "line": node.lineno,
                        "pattern": "large_class_no_slots",
                        "risk": "medium",
                        "recommendation": "Consider using __slots__ to reduce memory"
                    })
        
        return ref_issues
    
    def _identify_memory_optimizations(self) -> Dict[str, Any]:
        """Identify memory optimization opportunities."""
        optimizations = {
            "generator_opportunities": [],
            "string_optimization": [],
            "collection_optimization": [],
            "caching_opportunities": [],
            "recommendations": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Find generator opportunities
                gen_ops = self._find_generator_opportunities(tree)
                for op in gen_ops:
                    op["file"] = file_key
                    optimizations["generator_opportunities"].append(op)
                
                # Find string optimization opportunities
                str_ops = self._find_string_optimizations(tree)
                for op in str_ops:
                    op["file"] = file_key
                    optimizations["string_optimization"].append(op)
                
                # Find collection optimization opportunities
                coll_ops = self._find_collection_optimizations(tree)
                for op in coll_ops:
                    op["file"] = file_key
                    optimizations["collection_optimization"].append(op)
                
            except:
                continue
        
        # Generate recommendations
        optimizations["recommendations"] = self._generate_memory_recommendations(optimizations)
        
        return optimizations
    
    def _find_generator_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find opportunities to use generators instead of lists."""
        opportunities = []
        
        for node in ast.walk(tree):
            # List comprehensions that could be generators
            if isinstance(node, ast.ListComp):
                # Check if the result is only iterated once
                parent = self._get_parent_node(tree, node)
                if isinstance(parent, ast.For):
                    opportunities.append({
                        "line": node.lineno,
                        "pattern": "list_comp_in_loop",
                        "recommendation": "Use generator expression instead",
                        "memory_savings": "high"
                    })
            
            # Functions returning large lists
            elif isinstance(node, ast.FunctionDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        if isinstance(stmt.value, ast.ListComp):
                            opportunities.append({
                                "function": node.name,
                                "line": stmt.lineno,
                                "pattern": "return_list_comp",
                                "recommendation": "Consider yielding values",
                                "memory_savings": "medium"
                            })
        
        return opportunities
    
    def _find_string_optimizations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find string operation optimization opportunities."""
        optimizations = []
        
        for node in ast.walk(tree):
            # String concatenation in loops
            if isinstance(node, ast.For):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.AugAssign):
                        if isinstance(inner.op, ast.Add):
                            if isinstance(inner.target, ast.Name):
                                # Check if it's a string
                                optimizations.append({
                                    "line": inner.lineno,
                                    "pattern": "string_concat_in_loop",
                                    "recommendation": "Use list.append() and ''.join()",
                                    "memory_savings": "high"
                                })
        
        return optimizations
    
    def _find_collection_optimizations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find collection usage optimization opportunities."""
        optimizations = []
        
        for node in ast.walk(tree):
            # Check for list where set would be better
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'list':
                        # Check if it's used for membership testing
                        parent = self._get_parent_node(tree, node)
                        if isinstance(parent, ast.Compare):
                            for op in parent.ops:
                                if isinstance(op, ast.In):
                                    optimizations.append({
                                        "line": node.lineno,
                                        "pattern": "list_for_membership",
                                        "recommendation": "Use set for O(1) membership testing",
                                        "memory_savings": "medium"
                                    })
        
        return optimizations
    
    def _calculate_memory_metrics(self) -> Dict[str, Any]:
        """Calculate overall memory metrics and risk assessment."""
        allocations = self._analyze_memory_allocations()
        leaks = self._detect_memory_leaks()
        gc_pressure = self._analyze_gc_pressure()
        
        metrics = {
            "total_large_allocations": len(allocations.get("large_allocations", [])),
            "leak_risk_score": leaks.get("leak_risk_score", 0),
            "gc_pressure_score": gc_pressure.get("gc_pressure_score", 0),
            "memory_health_score": 0,
            "critical_issues": [],
            "recommendations": []
        }
        
        # Calculate memory health score
        health_score = 100
        health_score -= metrics["total_large_allocations"] * 2
        health_score -= metrics["leak_risk_score"] * 10
        health_score -= metrics["gc_pressure_score"] * 5
        metrics["memory_health_score"] = max(0, health_score)
        
        # Identify critical issues
        if metrics["leak_risk_score"] > 5:
            metrics["critical_issues"].append("High memory leak risk detected")
        
        if metrics["gc_pressure_score"] > 7:
            metrics["critical_issues"].append("High GC pressure patterns detected")
        
        # Generate recommendations
        if metrics["total_large_allocations"] > 10:
            metrics["recommendations"].append("Review and optimize large memory allocations")
        
        if len(leaks.get("resource_leaks", [])) > 0:
            metrics["recommendations"].append("Fix resource leaks using context managers")
        
        if len(gc_pressure.get("high_allocation_rate", [])) > 0:
            metrics["recommendations"].append("Reduce allocation rate in hot loops")
        
        return metrics
    
    # Helper methods
    def _extract_array_size(self, call_node: ast.Call) -> Optional[Dict[str, Any]]:
        """Extract size information from array allocation."""
        if not call_node.args:
            return None
        
        size_arg = call_node.args[0]
        if isinstance(size_arg, ast.Constant):
            size = size_arg.value
            return {
                "dimensions": [size],
                "estimated_size": size * 8  # Assume 8 bytes per element
            }
        elif isinstance(size_arg, ast.Tuple):
            dims = []
            for elt in size_arg.elts:
                if isinstance(elt, ast.Constant):
                    dims.append(elt.value)
            if dims:
                size = 1
                for d in dims:
                    size *= d
                return {
                    "dimensions": dims,
                    "estimated_size": size * 8
                }
        return None
    
    def _is_large_comprehension(self, comp_node: ast.ListComp) -> bool:
        """Check if a list comprehension is potentially large."""
        for generator in comp_node.generators:
            if isinstance(generator.iter, ast.Call):
                if isinstance(generator.iter.func, ast.Name):
                    if generator.iter.func.id == 'range':
                        if generator.iter.args:
                            if isinstance(generator.iter.args[0], ast.Constant):
                                if generator.iter.args[0].value > 10000:
                                    return True
        return False
    
    def _has_string_concatenation_in_loop(self, loop_node: ast.For) -> bool:
        """Check if loop contains string concatenation."""
        for node in ast.walk(loop_node):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.op, ast.Add):
                    return True
        return False
    
    def _is_in_with_statement(self, tree: ast.AST, node: ast.Call) -> bool:
        """Check if a call is within a with statement."""
        for with_node in ast.walk(tree):
            if isinstance(with_node, ast.With):
                for item in with_node.items:
                    if item.context_expr == node:
                        return True
        return False
    
    def _has_matching_close(self, tree: ast.AST, open_node: ast.Call) -> bool:
        """Check if a resource has a matching close call."""
        # Simplified check - looks for .close() in the same scope
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'close':
                        return True
        return False
    
    def _has_thread_join(self, tree: ast.AST, start_node: ast.Call) -> bool:
        """Check if a thread has a join call."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'join':
                        return True
        return False
    
    def _get_init_method(self, class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Get the __init__ method of a class."""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == '__init__':
                    return node
        return None
    
    def _has_cache_eviction(self, class_node: ast.ClassDef) -> bool:
        """Check if a class implements cache eviction."""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if any(word in node.name.lower() for word in ['evict', 'clear', 'cleanup', 'limit']):
                    return True
        return False
    
    def _is_module_level(self, tree: ast.AST, node: ast.AST) -> bool:
        """Check if a node is at module level."""
        return any(node in tree.body for node in ast.walk(tree) if node == node)
    
    def _is_infinite_loop(self, while_node: ast.While) -> bool:
        """Check if a while loop is infinite."""
        if isinstance(while_node.test, ast.Constant):
            return bool(while_node.test.value)
        elif isinstance(while_node.test, ast.NameConstant):
            return while_node.test.value is True
        return False
    
    def _has_large_range(self, for_node: ast.For) -> bool:
        """Check if a for loop has a large range."""
        if isinstance(for_node.iter, ast.Call):
            if isinstance(for_node.iter.func, ast.Name):
                if for_node.iter.func.id == 'range':
                    if for_node.iter.args:
                        if isinstance(for_node.iter.args[0], ast.Constant):
                            return for_node.iter.args[0].value > 10000
        return False
    
    def _is_recursive_function(self, func_node: ast.FunctionDef) -> bool:
        """Check if a function is recursive."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == func_node.name:
                        return True
        return False
    
    def _is_large_allocation(self, node: ast.Call) -> bool:
        """Check if a call represents a large allocation."""
        if isinstance(node.func, ast.Name):
            return node.func.id in ['zeros', 'ones', 'empty', 'list', 'dict']
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr in ['zeros', 'ones', 'empty', 'read_csv', 'load']
        return False
    
    def _get_parent_node(self, tree: ast.AST, target: ast.AST) -> Optional[ast.AST]:
        """Get the parent node of a target node."""
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if child == target:
                    return node
        return None
    
    def _calculate_leak_risk_score(self, leaks: Dict[str, Any]) -> int:
        """Calculate overall memory leak risk score."""
        score = 0
        score += len(leaks.get("resource_leaks", [])) * 3
        score += len(leaks.get("reference_leaks", [])) * 2
        score += len(leaks.get("cache_leaks", [])) * 2
        score += len(leaks.get("global_leaks", [])) * 1
        return min(score, 10)  # Cap at 10
    
    def _calculate_gc_pressure_score(self, gc_analysis: Dict[str, Any]) -> int:
        """Calculate GC pressure score."""
        score = 0
        score += len(gc_analysis.get("high_allocation_rate", [])) * 2
        score += len(gc_analysis.get("reference_counting_issues", [])) * 1
        score += len(gc_analysis.get("gc_unfriendly_patterns", [])) * 3
        return min(score, 10)  # Cap at 10
    
    def _generate_memory_recommendations(self, optimizations: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if optimizations["generator_opportunities"]:
            recommendations.append("Replace list comprehensions with generators where appropriate")
        
        if optimizations["string_optimization"]:
            recommendations.append("Optimize string operations using join() instead of concatenation")
        
        if optimizations["collection_optimization"]:
            recommendations.append("Use appropriate collection types (set for membership, deque for queue)")
        
        return recommendations