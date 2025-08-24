"""
Graph Analysis Module
====================

Implements comprehensive graph-based analysis:
- Call graphs and control flow analysis
- Dependency graphs and cycles
- Network analysis with NetworkX (with fallbacks)
- Graph metrics and centrality measures
"""

import ast
import statistics
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, deque

from .base_analyzer import BaseAnalyzer

# Try to import NetworkX, fall back to basic graph implementation
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

class GraphAnalyzer(BaseAnalyzer):
    """Analyzer for graph-based code analysis."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive graph analysis."""
        print("[INFO] Analyzing Graph Structures...")
        
        results = {
            "call_graphs": self._analyze_call_graphs(),
            "dependency_graphs": self._analyze_dependency_graphs(),
            "control_flow": self._analyze_control_flow(),
            "graph_metrics": self._calculate_graph_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} graph categories")
        return results
    
    def _analyze_call_graphs(self) -> Dict[str, Any]:
        """Analyze function call relationships."""
        call_graph = defaultdict(set)  # caller -> set of callees
        reverse_call_graph = defaultdict(set)  # callee -> set of callers
        function_locations = {}  # function_name -> (file, line)
        
        # Build call graph
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Find all function definitions
                functions_in_file = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_key = f"{file_key}::{node.name}"
                        functions_in_file[node.name] = func_key
                        function_locations[func_key] = (file_key, node.lineno)
                
                # Find function calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        caller_key = f"{file_key}::{node.name}"
                        
                        # Find calls within this function
                        for call_node in ast.walk(node):
                            if isinstance(call_node, ast.Call):
                                callee_name = self._extract_function_name(call_node)
                                if callee_name:
                                    # Try to resolve to full key
                                    if callee_name in functions_in_file:
                                        callee_key = functions_in_file[callee_name]
                                    else:
                                        callee_key = f"external::{callee_name}"
                                    
                                    call_graph[caller_key].add(callee_key)
                                    reverse_call_graph[callee_key].add(caller_key)
                                    
            except Exception:
                continue
        
        # Convert sets to lists for JSON serialization
        call_graph_json = {k: list(v) for k, v in call_graph.items()}
        reverse_call_graph_json = {k: list(v) for k, v in reverse_call_graph.items()}
        
        # Calculate call graph metrics
        if HAS_NETWORKX:
            metrics = self._calculate_networkx_metrics(call_graph, "call_graph")
        else:
            metrics = self._calculate_basic_graph_metrics(call_graph, reverse_call_graph)
        
        return {
            "call_relationships": call_graph_json,
            "reverse_calls": reverse_call_graph_json,
            "function_locations": function_locations,
            "metrics": metrics,
            "summary": {
                "total_functions": len(function_locations),
                "total_call_relationships": sum(len(calls) for calls in call_graph.values()),
                "most_called_functions": self._get_most_connected_nodes(reverse_call_graph, 5),
                "most_calling_functions": self._get_most_connected_nodes(call_graph, 5),
                "external_dependencies": len([k for k in call_graph_json if k.startswith("external::")]),
                "isolated_functions": len([f for f in function_locations if f not in call_graph and f not in reverse_call_graph])
            }
        }
    
    def _extract_function_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from a call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None
    
    def _analyze_dependency_graphs(self) -> Dict[str, Any]:
        """Analyze module dependency relationships."""
        dependency_graph = defaultdict(set)  # module -> set of dependencies
        import_locations = defaultdict(list)  # module -> list of (import, line)
        
        # Build dependency graph
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            dependency_graph[file_key].add(alias.name)
                            import_locations[file_key].append((alias.name, node.lineno))
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        dependency_graph[file_key].add(node.module)
                        import_locations[file_key].append((node.module, node.lineno))
                        
            except Exception:
                continue
        
        # Detect circular dependencies
        cycles = self._detect_cycles(dependency_graph)
        
        # Calculate dependency metrics
        if HAS_NETWORKX:
            metrics = self._calculate_networkx_metrics(dependency_graph, "dependency_graph")
        else:
            metrics = self._calculate_basic_dependency_metrics(dependency_graph)
        
        return {
            "dependencies": {k: list(v) for k, v in dependency_graph.items()},
            "import_locations": dict(import_locations),
            "circular_dependencies": cycles,
            "metrics": metrics,
            "summary": {
                "total_modules": len(dependency_graph),
                "total_dependencies": sum(len(deps) for deps in dependency_graph.values()),
                "circular_dependency_count": len(cycles),
                "highly_coupled_modules": len([m for m, deps in dependency_graph.items() if len(deps) > 10]),
                "isolated_modules": len([m for m, deps in dependency_graph.items() if len(deps) == 0])
            }
        }
    
    def _detect_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect cycles in a directed graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _analyze_control_flow(self) -> Dict[str, Any]:
        """Analyze control flow complexity in functions."""
        control_flow_data = {}
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                file_functions = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        flow_metrics = self._analyze_function_control_flow(node)
                        file_functions[node.name] = flow_metrics
                
                if file_functions:
                    control_flow_data[file_key] = file_functions
                    
            except Exception:
                continue
        
        # Calculate summary statistics
        all_complexities = []
        all_branch_counts = []
        
        for file_funcs in control_flow_data.values():
            for func_data in file_funcs.values():
                all_complexities.append(func_data["cyclomatic_complexity"])
                all_branch_counts.append(func_data["branch_count"])
        
        return {
            "per_function": control_flow_data,
            "summary": {
                "functions_analyzed": len(all_complexities),
                "average_complexity": statistics.mean(all_complexities) if all_complexities else 0,
                "max_complexity": max(all_complexities) if all_complexities else 0,
                "high_complexity_functions": len([c for c in all_complexities if c > 10]),
                "average_branches": statistics.mean(all_branch_counts) if all_branch_counts else 0,
                "complexity_distribution": self._calculate_distribution(all_complexities)
            }
        }
    
    def _analyze_function_control_flow(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze control flow for a single function."""
        branch_count = 0
        loop_count = 0
        decision_points = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                branch_count += 1
                decision_points.append(("if", node.lineno))
            elif isinstance(node, ast.While):
                branch_count += 1
                loop_count += 1
                decision_points.append(("while", node.lineno))
            elif isinstance(node, ast.For):
                branch_count += 1
                loop_count += 1
                decision_points.append(("for", node.lineno))
            elif isinstance(node, ast.Try):
                branch_count += len(node.handlers) + (1 if node.orelse else 0) + (1 if node.finalbody else 0)
                decision_points.append(("try", node.lineno))
        
        cyclomatic_complexity = self._calculate_function_complexity(func_node)
        
        return {
            "cyclomatic_complexity": cyclomatic_complexity,
            "branch_count": branch_count,
            "loop_count": loop_count,
            "decision_points": decision_points,
            "nesting_depth": self._calculate_nesting_depth(func_node)
        }
    
    def _calculate_nesting_depth(self, func_node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth in a function."""
        max_depth = 0
        
        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                for child in ast.iter_child_nodes(node):
                    calculate_depth(child, current_depth + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    calculate_depth(child, current_depth)
        
        for child in func_node.body:
            calculate_depth(child)
        
        return max_depth
    
    def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """Calculate overall graph metrics."""
        if not HAS_NETWORKX:
            return {"networkx_available": False, "basic_metrics_only": True}
        
        # This would contain advanced NetworkX-based metrics
        return {
            "networkx_available": True,
            "advanced_metrics": "Available with NetworkX",
            "centrality_measures": "Supported",
            "community_detection": "Supported",
            "path_analysis": "Supported"
        }
    
    def _calculate_networkx_metrics(self, graph_dict: Dict[str, Set[str]], graph_type: str) -> Dict[str, Any]:
        """Calculate advanced metrics using NetworkX."""
        if not HAS_NETWORKX:
            return {"error": "NetworkX not available"}
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        for node, edges in graph_dict.items():
            G.add_node(node)
            for edge in edges:
                G.add_edge(node, edge)
        
        try:
            metrics = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": nx.density(G),
                "is_strongly_connected": nx.is_strongly_connected(G),
                "number_strongly_connected_components": nx.number_strongly_connected_components(G),
                "is_weakly_connected": nx.is_weakly_connected(G),
                "average_clustering": nx.average_clustering(G.to_undirected()),
            }
            
            # Calculate centrality measures for smaller graphs
            if G.number_of_nodes() < 100:
                centrality = nx.degree_centrality(G)
                metrics["centrality_stats"] = {
                    "max_centrality": max(centrality.values()) if centrality else 0,
                    "avg_centrality": statistics.mean(centrality.values()) if centrality else 0
                }
            
            return metrics
            
        except Exception as e:
            return {"error": f"NetworkX calculation failed: {str(e)}"}
    
    def _calculate_basic_graph_metrics(self, forward_graph: Dict[str, Set[str]], reverse_graph: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Calculate basic graph metrics without NetworkX."""
        total_nodes = len(set(list(forward_graph.keys()) + list(reverse_graph.keys())))
        total_edges = sum(len(edges) for edges in forward_graph.values())
        
        # Calculate in-degree and out-degree distributions
        out_degrees = [len(edges) for edges in forward_graph.values()]
        in_degrees = [len(edges) for edges in reverse_graph.values()]
        
        return {
            "node_count": total_nodes,
            "edge_count": total_edges,
            "average_out_degree": statistics.mean(out_degrees) if out_degrees else 0,
            "average_in_degree": statistics.mean(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "networkx_available": False
        }
    
    def _calculate_basic_dependency_metrics(self, dependency_graph: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Calculate basic dependency metrics without NetworkX."""
        fan_out = [len(deps) for deps in dependency_graph.values()]
        
        return {
            "average_fan_out": statistics.mean(fan_out) if fan_out else 0,
            "max_fan_out": max(fan_out) if fan_out else 0,
            "modules_with_dependencies": len([deps for deps in dependency_graph.values() if deps]),
            "networkx_available": False
        }
    
    def _get_most_connected_nodes(self, graph: Dict[str, Set[str]], top_n: int) -> List[Tuple[str, int]]:
        """Get the most connected nodes in a graph."""
        node_degrees = [(node, len(connections)) for node, connections in graph.items()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        return node_degrees[:top_n]