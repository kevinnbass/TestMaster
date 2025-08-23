#!/usr/bin/env python3
"""
Relationship Core Analyzer
=========================

Core relationship analysis functionality.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
import networkx as nx

from relationship_data import (
    ModuleRelationship, ClassRelationship, FunctionCallGraph,
    CouplingMetrics, RelationshipCluster, RelationshipAnalysis
)


class RelationshipAnalyzer:
    """
    Analyzes relationships between modules, classes, and functions.
    Provides insights into coupling, dependencies, and interaction patterns.
    """

    def __init__(self) -> None:
        """Initialize the relationship analyzer"""
        self.relationship_graph: nx.DiGraph = nx.DiGraph()
        self.module_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.class_hierarchy: Dict[str, Set[str]] = defaultdict(set)

    def analyze_relationships(self, content: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive relationship analysis of code content.

        Args:
            content: The code content to analyze
            file_path: Path to the file for context

        Returns:
            Dictionary containing relationship analysis results
        """
        try:
            tree = ast.parse(content)

            # Analyze different types of relationships
            import_relationships = self._analyze_import_relationships(tree, str(file_path) if file_path else "unknown")
            class_relationships = self._analyze_class_relationships(tree)
            function_relationships = self._analyze_function_relationships(tree)
            data_relationships = self._analyze_data_relationships(tree)

            # Build relationship graph
            relationship_graph = self._build_relationship_graph(
                import_relationships, class_relationships,
                function_relationships, data_relationships
            )

            # Calculate coupling metrics
            coupling_metrics = self._calculate_coupling_metrics(relationship_graph)

            # Identify clusters and communities
            clusters = self._identify_relationship_clusters(relationship_graph)

            # Generate recommendations
            recommendations = self._generate_relationship_recommendations(
                coupling_metrics, clusters, relationship_graph
            )

            result = {
                'import_relationships': import_relationships,
                'class_relationships': class_relationships,
                'function_relationships': function_relationships,
                'data_relationships': data_relationships,
                'coupling_metrics': coupling_metrics,
                'relationship_clusters': clusters,
                'relationship_graph': {
                    'nodes': list(relationship_graph.nodes()),
                    'edges': list(relationship_graph.edges(data=True)),
                    'node_count': relationship_graph.number_of_nodes(),
                    'edge_count': relationship_graph.number_of_edges()
                },
                'recommendations': recommendations,
                'file_path': file_path or 'unknown'
            }

            return result

        except SyntaxError as e:
            return self._fallback_relationship_analysis(content, str(file_path) if file_path else "unknown", e)
        except Exception as e:
            return {
                'error': f'Relationship analysis failed: {e}',
                'import_relationships': [],
                'class_relationships': [],
                'function_relationships': {},
                'data_relationships': {},
                'coupling_metrics': {},
                'relationship_clusters': [],
                'relationship_graph': {'nodes': [], 'edges': [], 'node_count': 0, 'edge_count': 0},
                'recommendations': ['Unable to analyze relationships due to error'],
                'file_path': file_path or 'unknown'
            }

    def _analyze_import_relationships(self, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Analyze import relationships between modules"""
        relationships = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    relationships.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'line_number': node.lineno
                    })

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        relationships.append({
                            'type': 'from_import',
                            'module': node.module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'line_number': node.lineno
                        })

        return relationships

    def _analyze_class_relationships(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze relationships between classes"""
        relationships = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Inheritance relationships
                for base in node.bases:
                    if hasattr(base, 'id'):
                        relationships.append({
                            'type': 'inheritance',
                            'source_class': node.name,
                            'target_class': base.id,
                            'line_number': node.lineno
                        })
                    elif hasattr(base, 'attr'):
                        # Handle dotted names like module.ClassName
                        relationships.append({
                            'type': 'inheritance',
                            'source_class': node.name,
                            'target_class': f"{base.value.id}.{base.attr}",
                            'line_number': node.lineno
                        })

                # Composition relationships (attributes that are class instances)
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and not target.id.startswith('_'):
                                # Check if assignment is a class instantiation
                                if isinstance(item.value, ast.Call):
                                    if isinstance(item.value.func, ast.Name):
                                        relationships.append({
                                            'type': 'composition',
                                            'source_class': node.name,
                                            'target_class': item.value.func.id,
                                            'attribute': target.id,
                                            'line_number': item.lineno
                                        })

        return relationships

    def _analyze_function_relationships(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze function call relationships"""
        function_calls = defaultdict(list)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Find all function calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            function_calls[node.name].append({
                                'called_function': child.func.id,
                                'line_number': child.lineno
                            })
                        elif isinstance(child.func, ast.Attribute):
                            # Handle method calls like obj.method()
                            if isinstance(child.func.value, ast.Name):
                                function_calls[node.name].append({
                                    'called_method': child.func.attr,
                                    'object': child.func.value.id,
                                    'line_number': child.lineno
                                })

        return dict(function_calls)

    def _analyze_data_relationships(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze data flow and variable relationships"""
        data_relationships = defaultdict(list)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Find variable assignments and usages
                assignments = []
                usages = []

                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                assignments.append(target.id)
                    elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        usages.append(child.id)

                data_relationships[node.name] = {
                    'assignments': list(set(assignments)),
                    'usages': list(set(usages))
                }

        return dict(data_relationships)

    def _build_relationship_graph(self, import_relationships, class_relationships,
                                function_relationships, data_relationships) -> nx.DiGraph:
        """Build a graph representation of relationships"""
        G = nx.DiGraph()

        # Add import relationships
        for rel in import_relationships:
            if rel['type'] == 'import':
                G.add_edge(f"file:{self.file_path}", f"module:{rel['module']}", type='import', weight=0.5)
            elif rel['type'] == 'from_import':
                G.add_edge(f"file:{self.file_path}", f"module:{rel['module']}", type='from_import', weight=0.7)

        # Add class relationships
        for rel in class_relationships:
            if rel['type'] == 'inheritance':
                G.add_edge(f"class:{rel['source_class']}", f"class:{rel['target_class']}",
                          type='inherits', weight=1.0)
            elif rel['type'] == 'composition':
                G.add_edge(f"class:{rel['source_class']}", f"class:{rel['target_class']}",
                          type='composes', weight=0.8)

        # Add function relationships
        for func, calls in function_relationships.items():
            for call in calls:
                if 'called_function' in call:
                    G.add_edge(f"func:{func}", f"func:{call['called_function']}",
                              type='calls', weight=0.6)
                elif 'called_method' in call:
                    G.add_edge(f"func:{func}", f"method:{call['called_method']}",
                              type='calls_method', weight=0.4)

        return G

    def _calculate_coupling_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate coupling metrics from relationship graph"""
        if graph.number_of_nodes() == 0:
            return {'coupling_score': 0.0}

        # Calculate various coupling metrics
        density = nx.density(graph)
        average_clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 1 else 0

        # Identify highly connected components
        degrees = dict(graph.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0

        highly_coupled = [node for node, degree in degrees.items()
                         if degree > avg_degree * 1.5]

        return {
            'coupling_score': density,
            'average_clustering': average_clustering,
            'average_degree': avg_degree,
            'highly_coupled_components': highly_coupled,
            'graph_density': density
        }

    def _identify_relationship_clusters(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Identify clusters in the relationship graph"""
        if graph.number_of_nodes() == 0:
            return []

        # Find connected components
        components = list(nx.connected_components(graph.to_undirected()))

        clusters = []
        for i, component in enumerate(components):
            subgraph = graph.subgraph(component)
            cohesion = nx.density(subgraph)

            clusters.append({
                'cluster_id': f"cluster_{i}",
                'components': list(component),
                'size': len(component),
                'cohesion_score': cohesion,
                'coupling_score': nx.density(subgraph)
            })

        return clusters

    def _generate_relationship_recommendations(self, coupling_metrics, clusters, graph) -> List[str]:
        """Generate recommendations based on relationship analysis"""
        recommendations = []

        coupling_score = coupling_metrics.get('coupling_score', 0)

        if coupling_score > 0.7:
            recommendations.append("High coupling detected - consider breaking down into smaller modules")
        elif coupling_score < 0.1:
            recommendations.append("Low coupling - good modular design")

        # Cluster-based recommendations
        large_clusters = [c for c in clusters if c['size'] > 10]
        if large_clusters:
            recommendations.append(f"Found {len(large_clusters)} large clusters - consider further decomposition")

        return recommendations

    def _fallback_relationship_analysis(self, content: str, file_path: str, error) -> Dict[str, Any]:
        """Fallback analysis when AST parsing fails"""
        # Basic regex-based analysis
        import_matches = re.findall(r'^\s*(import|from)\s+(\w+)', content, re.MULTILINE)
        class_matches = re.findall(r'class\s+(\w+)', content)
        function_matches = re.findall(r'def\s+(\w+)', content)

        return {
            'import_relationships': [{'type': 'import', 'module': match[1], 'line_number': 0}
                                   for match in import_matches],
            'class_relationships': [],
            'function_relationships': {},
            'data_relationships': {},
            'coupling_metrics': {'coupling_score': 0.5, 'note': 'Fallback analysis used'},
            'relationship_clusters': [],
            'relationship_graph': {
                'nodes': [f'class:{c}' for c in class_matches] + [f'func:{f}' for f in function_matches],
                'edges': [],
                'node_count': len(class_matches) + len(function_matches),
                'edge_count': 0
            },
            'recommendations': [f'Basic analysis only - full analysis failed: {error}'],
            'file_path': file_path,
            'analysis_type': 'fallback'
        }
