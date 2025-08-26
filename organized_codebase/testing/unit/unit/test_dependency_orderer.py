#!/usr/bin/env python3
"""
Test Dependency Orderer - Orders tests based on dependency graph analysis.
Ensures tests run in optimal order to minimize failures due to dependencies.
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import ast
import networkx as nx
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create stub implementations for missing analyzers
class DependencyAnalyzer:
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Simple dependency analysis stub."""
        return {
            'dependencies': [],
            'dependents': [],
            'circular_dependencies': []
        }

class ArchitectureAnalyzer:
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Simple architecture analysis stub."""
        return {}

# Import TestComplexityPrioritizer later to avoid circular imports
try:
    from test_complexity_prioritizer import TestComplexityPrioritizer, TestPriority
except ImportError:
    TestComplexityPrioritizer = None
    from dataclasses import dataclass
    @dataclass
    class TestPriority:
        test_path: str
        test_name: str
        priority_score: float = 0.5


@dataclass
class DependencyNode:
    """Represents a test with its dependencies."""
    test_path: str
    test_name: str
    depends_on: Set[str] = field(default_factory=set)
    depended_by: Set[str] = field(default_factory=set)
    layer: int = 0
    module_dependencies: Set[str] = field(default_factory=set)
    external_dependencies: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'test_path': self.test_path,
            'test_name': self.test_name,
            'depends_on': list(self.depends_on),
            'depended_by': list(self.depended_by),
            'layer': self.layer,
            'module_dependencies': list(self.module_dependencies),
            'external_dependencies': list(self.external_dependencies)
        }


@dataclass 
class DependencyGraph:
    """Represents the complete test dependency graph."""
    nodes: Dict[str, DependencyNode]
    edges: List[Tuple[str, str]]
    layers: Dict[int, List[str]]
    cycles: List[List[str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': self.edges,
            'layers': self.layers,
            'cycles': self.cycles
        }


class TestDependencyOrderer:
    """Orders tests based on dependency analysis to minimize cascading failures."""
    
    def __init__(self, project_root: str = '.'):
        """Initialize the test dependency orderer."""
        self.project_root = Path(project_root).resolve()
        self.dependency_analyzer = DependencyAnalyzer()
        self.architecture_analyzer = ArchitectureAnalyzer()
        self.complexity_prioritizer = TestComplexityPrioritizer(project_root)
        
        # Caches
        self._dependency_cache: Dict[str, Dict[str, Any]] = {}
        self._module_map: Dict[str, str] = {}  # Maps module names to file paths
        self._test_module_map: Dict[str, Set[str]] = {}  # Maps tests to tested modules
        
        # Graph
        self.graph = nx.DiGraph()
        
        # Configuration
        self.config = {
            'max_parallel_analysis': 10,
            'cache_ttl_seconds': 300,
            'detect_cycles': True,
            'layer_based_ordering': True
        }
        
        # Threading
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.config['max_parallel_analysis'])
    
    def build_dependency_graph(self, test_paths: List[str]) -> DependencyGraph:
        """Build complete dependency graph for all tests."""
        nodes = {}
        edges = []
        
        # Build module map first
        self._build_module_map()
        
        # Analyze each test's dependencies
        with self._executor as executor:
            futures = []
            for test_path in test_paths:
                future = executor.submit(self._analyze_test_dependencies, test_path)
                futures.append((test_path, future))
            
            for test_path, future in futures:
                try:
                    node = future.result(timeout=30)
                    if node:
                        nodes[test_path] = node
                        # Add to graph
                        self.graph.add_node(test_path)
                except Exception as e:
                    print(f"Error analyzing dependencies for {test_path}: {e}")
        
        # Build edges based on dependencies
        for test_path, node in nodes.items():
            for dep in node.depends_on:
                if dep in nodes:
                    edges.append((dep, test_path))
                    self.graph.add_edge(dep, test_path)
                    nodes[dep].depended_by.add(test_path)
        
        # Detect cycles if configured
        cycles = []
        if self.config['detect_cycles']:
            cycles = self._detect_dependency_cycles()
        
        # Calculate layers (topological levels)
        layers = self._calculate_dependency_layers(nodes)
        
        return DependencyGraph(
            nodes=nodes,
            edges=edges,
            layers=layers,
            cycles=cycles
        )
    
    def order_tests(self, 
                   test_paths: List[str],
                   strategy: str = 'topological',
                   respect_priorities: bool = True) -> List[str]:
        """Order tests based on dependencies using specified strategy."""
        
        # Build dependency graph
        dep_graph = self.build_dependency_graph(test_paths)
        
        # Choose ordering strategy
        if strategy == 'topological':
            ordered = self._topological_order(dep_graph)
        elif strategy == 'layer':
            ordered = self._layer_based_order(dep_graph)
        elif strategy == 'critical_path':
            ordered = self._critical_path_order(dep_graph)
        elif strategy == 'minimal_retest':
            ordered = self._minimal_retest_order(dep_graph)
        else:
            ordered = test_paths  # Default: no ordering
        
        # Apply priority adjustments if requested
        if respect_priorities:
            ordered = self._apply_priority_adjustments(ordered)
        
        return ordered
    
    def _analyze_test_dependencies(self, test_path: str) -> Optional[DependencyNode]:
        """Analyze dependencies for a single test."""
        try:
            # Get test name
            test_name = Path(test_path).stem
            
            # Create node
            node = DependencyNode(
                test_path=test_path,
                test_name=test_name
            )
            
            # Analyze code dependencies
            dep_result = self.dependency_analyzer.analyze(test_path)
            
            # Extract module dependencies
            for dep in dep_result.get('dependencies', []):
                if isinstance(dep, dict):
                    module = dep.get('module', dep.get('name', ''))
                else:
                    module = str(dep)
                
                if module:
                    node.module_dependencies.add(module)
                    
                    # Check if it's an external dependency
                    if not self._is_internal_module(module):
                        node.external_dependencies.add(module)
            
            # Infer tested modules from test name and imports
            tested_modules = self._infer_tested_modules(test_path, test_name)
            node.module_dependencies.update(tested_modules)
            
            # Find dependencies on other tests
            node.depends_on = self._find_test_dependencies(test_path, node.module_dependencies)
            
            return node
            
        except Exception as e:
            print(f"Error analyzing test {test_path}: {e}")
            return None
    
    def _build_module_map(self):
        """Build a map of module names to file paths."""
        self._module_map.clear()
        
        # Scan Python files in project
        for py_file in self.project_root.rglob('*.py'):
            if not any(part.startswith('.') for part in py_file.parts):
                module_name = py_file.stem
                self._module_map[module_name] = str(py_file)
                
                # Also map the full module path
                try:
                    rel_path = py_file.relative_to(self.project_root)
                    module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
                    self._module_map[module_path] = str(py_file)
                except:
                    pass
    
    def _is_internal_module(self, module: str) -> bool:
        """Check if a module is internal to the project."""
        # Check if module is in our module map
        if module in self._module_map:
            return True
        
        # Check if it's a relative import
        if module.startswith('.'):
            return True
        
        # Check if it starts with project name
        project_name = self.project_root.name.lower()
        if module.lower().startswith(project_name):
            return True
        
        # Standard library and common external modules
        external_indicators = [
            'unittest', 'pytest', 'mock', 'numpy', 'pandas', 'requests',
            'flask', 'django', 'sqlalchemy', 'asyncio', 'json', 'os', 'sys'
        ]
        
        for indicator in external_indicators:
            if module.startswith(indicator):
                return False
        
        # Default to internal if uncertain
        return True
    
    def _infer_tested_modules(self, test_path: str, test_name: str) -> Set[str]:
        """Infer which modules are being tested based on test name and content."""
        tested = set()
        
        # Infer from test name
        if test_name.startswith('test_'):
            module_name = test_name[5:]  # Remove 'test_' prefix
            tested.add(module_name)
        
        # Analyze test content for explicit imports of tested modules
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._is_internal_module(alias.name):
                            tested.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and self._is_internal_module(node.module):
                        tested.add(node.module)
        except:
            pass
        
        return tested
    
    def _find_test_dependencies(self, test_path: str, module_dependencies: Set[str]) -> Set[str]:
        """Find which other tests this test depends on."""
        dependencies = set()
        
        # Map modules to tests that test them
        if not self._test_module_map:
            self._build_test_module_map()
        
        # Find tests that test our dependent modules
        for module in module_dependencies:
            if module in self._test_module_map:
                for other_test in self._test_module_map[module]:
                    if other_test != test_path:
                        dependencies.add(other_test)
        
        return dependencies
    
    def _build_test_module_map(self):
        """Build map of modules to tests that test them."""
        self._test_module_map.clear()
        
        # Discover all tests
        test_files = self.complexity_prioritizer.discover_tests()
        
        for test_file in test_files:
            test_name = Path(test_file).stem
            tested_modules = self._infer_tested_modules(test_file, test_name)
            
            for module in tested_modules:
                if module not in self._test_module_map:
                    self._test_module_map[module] = set()
                self._test_module_map[module].add(test_file)
    
    def _detect_dependency_cycles(self) -> List[List[str]]:
        """Detect circular dependencies in the test graph."""
        cycles = []
        
        try:
            # Use NetworkX to find cycles
            simple_cycles = list(nx.simple_cycles(self.graph))
            cycles = [[str(node) for node in cycle] for cycle in simple_cycles]
        except:
            pass
        
        return cycles
    
    def _calculate_dependency_layers(self, nodes: Dict[str, DependencyNode]) -> Dict[int, List[str]]:
        """Calculate dependency layers (topological levels)."""
        layers = defaultdict(list)
        
        # Calculate layer for each node
        visited = set()
        
        def calculate_layer(node_path: str) -> int:
            if node_path in visited:
                return nodes[node_path].layer
            
            visited.add(node_path)
            node = nodes.get(node_path)
            
            if not node or not node.depends_on:
                layer = 0
            else:
                # Layer is max of dependencies + 1
                dep_layers = []
                for dep in node.depends_on:
                    if dep in nodes and dep != node_path:
                        dep_layers.append(calculate_layer(dep))
                
                layer = max(dep_layers) + 1 if dep_layers else 0
            
            if node:
                node.layer = layer
                layers[layer].append(node_path)
            
            return layer
        
        # Calculate layers for all nodes
        for node_path in nodes:
            calculate_layer(node_path)
        
        return dict(layers)
    
    def _topological_order(self, dep_graph: DependencyGraph) -> List[str]:
        """Order tests using topological sort."""
        try:
            # Use NetworkX topological sort
            ordered = list(nx.topological_sort(self.graph))
            
            # Add any tests not in the graph
            all_tests = set(dep_graph.nodes.keys())
            ordered_set = set(ordered)
            remaining = all_tests - ordered_set
            
            ordered.extend(sorted(remaining))
            
            return ordered
            
        except nx.NetworkXError:
            # Graph has cycles, use layer-based ordering instead
            print("Dependency cycles detected, using layer-based ordering")
            return self._layer_based_order(dep_graph)
    
    def _layer_based_order(self, dep_graph: DependencyGraph) -> List[str]:
        """Order tests by dependency layers."""
        ordered = []
        
        # Sort layers by key
        for layer in sorted(dep_graph.layers.keys()):
            # Within each layer, sort alphabetically or by priority
            layer_tests = dep_graph.layers[layer]
            ordered.extend(sorted(layer_tests))
        
        # Add any tests not in layers
        all_tests = set(dep_graph.nodes.keys())
        ordered_set = set(ordered)
        remaining = all_tests - ordered_set
        
        ordered.extend(sorted(remaining))
        
        return ordered
    
    def _critical_path_order(self, dep_graph: DependencyGraph) -> List[str]:
        """Order tests to run critical path first."""
        ordered = []
        
        # Find nodes with most dependencies (critical)
        critical_nodes = []
        for node_path, node in dep_graph.nodes.items():
            score = len(node.depended_by) * 2 + len(node.depends_on)
            critical_nodes.append((score, node_path))
        
        # Sort by criticality
        critical_nodes.sort(reverse=True)
        
        # Build order respecting dependencies
        added = set()
        for _, node_path in critical_nodes:
            if node_path not in added:
                # Add dependencies first
                self._add_with_dependencies(node_path, dep_graph, ordered, added)
        
        return ordered
    
    def _add_with_dependencies(self, node_path: str, dep_graph: DependencyGraph, 
                              ordered: List[str], added: Set[str]):
        """Recursively add node with its dependencies."""
        if node_path in added:
            return
        
        node = dep_graph.nodes.get(node_path)
        if node:
            # Add dependencies first
            for dep in node.depends_on:
                if dep not in added and dep in dep_graph.nodes:
                    self._add_with_dependencies(dep, dep_graph, ordered, added)
        
        # Add this node
        ordered.append(node_path)
        added.add(node_path)
    
    def _minimal_retest_order(self, dep_graph: DependencyGraph) -> List[str]:
        """Order tests to minimize retesting when failures occur."""
        ordered = []
        
        # Start with tests that have no dependencies
        independent = [n for n, node in dep_graph.nodes.items() if not node.depends_on]
        ordered.extend(sorted(independent))
        
        # Add tests with dependencies, minimizing forward dependencies
        remaining = set(dep_graph.nodes.keys()) - set(ordered)
        
        while remaining:
            # Find test with minimum unsatisfied dependencies
            best = None
            min_unsatisfied = float('inf')
            
            for test in remaining:
                node = dep_graph.nodes[test]
                unsatisfied = len(node.depends_on - set(ordered))
                
                if unsatisfied < min_unsatisfied:
                    min_unsatisfied = unsatisfied
                    best = test
            
            if best:
                ordered.append(best)
                remaining.remove(best)
            else:
                # Add remaining tests (may have cycles)
                ordered.extend(sorted(remaining))
                break
        
        return ordered
    
    def _apply_priority_adjustments(self, ordered: List[str]) -> List[str]:
        """Apply priority-based adjustments to the order."""
        # Get priorities for all tests
        suite = self.complexity_prioritizer.prioritize_tests(test_paths=ordered)
        
        # Create priority map
        priority_map = {t.test_path: t.priority_score for t in suite.tests}
        
        # Reorder within dependency constraints
        adjusted = []
        layers = self._group_by_dependencies(ordered)
        
        for layer in layers:
            # Sort layer by priority
            layer.sort(key=lambda x: priority_map.get(x, 0), reverse=True)
            adjusted.extend(layer)
        
        return adjusted
    
    def _group_by_dependencies(self, ordered: List[str]) -> List[List[str]]:
        """Group tests by dependency constraints."""
        layers = []
        current_layer = []
        dependencies_met = set()
        
        for test in ordered:
            node = self.graph.nodes.get(test, {})
            deps = set(node.get('depends_on', []))
            
            # Check if all dependencies are met
            if deps.issubset(dependencies_met):
                current_layer.append(test)
            else:
                # Start new layer
                if current_layer:
                    layers.append(current_layer)
                    dependencies_met.update(current_layer)
                current_layer = [test]
        
        if current_layer:
            layers.append(current_layer)
        
        return layers
    
    def generate_parallel_groups(self, ordered_tests: List[str], max_parallel: int = 4) -> List[List[str]]:
        """Generate parallel execution groups respecting dependencies."""
        groups = []
        remaining = ordered_tests.copy()
        completed = set()
        
        while remaining:
            # Find tests that can run in parallel (dependencies satisfied)
            parallel_batch = []
            
            for test in remaining[:]:
                node = self.graph.nodes.get(test, {})
                deps = set(node.get('depends_on', []))
                
                if deps.issubset(completed):
                    parallel_batch.append(test)
                    remaining.remove(test)
                    
                    if len(parallel_batch) >= max_parallel:
                        break
            
            if parallel_batch:
                groups.append(parallel_batch)
                completed.update(parallel_batch)
            else:
                # No tests can run, might have cycles
                # Add remaining tests in order
                groups.append(remaining[:max_parallel])
                completed.update(remaining[:max_parallel])
                remaining = remaining[max_parallel:]
        
        return groups
    
    def visualize_dependency_graph(self, dep_graph: DependencyGraph, output_file: str = 'test_dependencies.json'):
        """Export dependency graph for visualization."""
        viz_data = {
            'nodes': [],
            'edges': [],
            'layers': dep_graph.layers,
            'cycles': dep_graph.cycles
        }
        
        # Format nodes for visualization
        for node_path, node in dep_graph.nodes.items():
            viz_data['nodes'].append({
                'id': node_path,
                'label': node.test_name,
                'layer': node.layer,
                'dependencies': len(node.depends_on),
                'dependents': len(node.depended_by)
            })
        
        # Format edges
        for source, target in dep_graph.edges:
            viz_data['edges'].append({
                'source': source,
                'target': target
            })
        
        with open(output_file, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        print(f"Dependency graph exported to {output_file}")


def main():
    """Main function to demonstrate dependency-based test ordering."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Order tests based on dependency analysis')
    parser.add_argument('--project', default='.', help='Project root directory')
    parser.add_argument('--strategy', choices=['topological', 'layer', 'critical_path', 'minimal_retest'],
                       default='topological', help='Ordering strategy')
    parser.add_argument('--respect-priorities', action='store_true', 
                       help='Consider test priorities in ordering')
    parser.add_argument('--max-parallel', type=int, default=4,
                       help='Maximum parallel test groups')
    parser.add_argument('--visualize', help='Export dependency graph to file')
    
    args = parser.parse_args()
    
    # Initialize orderer
    orderer = TestDependencyOrderer(args.project)
    
    # Discover tests
    print(f"Discovering tests in {args.project}...")
    test_paths = orderer.complexity_prioritizer.discover_tests()
    print(f"Found {len(test_paths)} test files")
    
    # Build dependency graph
    print("\nBuilding dependency graph...")
    dep_graph = orderer.build_dependency_graph(test_paths)
    
    print(f"\nDependency Analysis:")
    print(f"  Total nodes: {len(dep_graph.nodes)}")
    print(f"  Total edges: {len(dep_graph.edges)}")
    print(f"  Dependency layers: {len(dep_graph.layers)}")
    print(f"  Circular dependencies: {len(dep_graph.cycles)}")
    
    if dep_graph.cycles:
        print("\n  Circular dependencies detected:")
        for cycle in dep_graph.cycles[:5]:  # Show first 5 cycles
            print(f"    {' -> '.join(Path(p).stem for p in cycle)}")
    
    # Order tests
    print(f"\nOrdering tests using {args.strategy} strategy...")
    ordered = orderer.order_tests(
        test_paths,
        strategy=args.strategy,
        respect_priorities=args.respect_priorities
    )
    
    print(f"\nOrdered Test Execution:")
    for i, test in enumerate(ordered[:20], 1):  # Show first 20
        node = dep_graph.nodes.get(test)
        if node:
            print(f"  {i}. {node.test_name} (layer {node.layer}, deps: {len(node.depends_on)})")
    
    # Generate parallel groups
    print(f"\nParallel Execution Groups ({args.max_parallel} max parallel):")
    groups = orderer.generate_parallel_groups(ordered, args.max_parallel)
    for i, group in enumerate(groups[:10], 1):  # Show first 10 groups
        print(f"  Group {i}: {len(group)} tests")
        for test in group[:3]:  # Show first 3 tests in each group
            print(f"    - {Path(test).stem}")
    
    # Visualize if requested
    if args.visualize:
        orderer.visualize_dependency_graph(dep_graph, args.visualize)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())