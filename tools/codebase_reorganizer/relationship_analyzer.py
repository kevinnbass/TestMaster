#!/usr/bin/env python3
"""
Relationship Analyzer
====================

Analyzes relationships between modules, classes, and functions to understand
how components interact within the codebase. This intelligence module helps
the reorganizer understand dependencies and coupling patterns.

Key capabilities:
- Import dependency analysis
- Function call relationship mapping
- Data flow analysis
- Module coupling measurement
- Interface analysis
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx


@dataclass
class ModuleRelationship:
    """Relationship between two modules"""
    source_module: Path
    target_module: Path
    relationship_type: str  # 'imports', 'calls', 'shares_data', 'extends'
    strength: float
    evidence: List[str]
    bidirectional: bool


@dataclass
class ClassRelationship:
    """Relationship between two classes"""
    source_class: str
    target_class: str
    relationship_type: str  # 'inherits', 'composes', 'uses', 'associates'
    strength: float
    evidence: List[str]


@dataclass
class FunctionCallGraph:
    """Graph of function calls within and between modules"""
    nodes: List[str]
    edges: List[Tuple[str, str, float]]  # source, target, weight
    clusters: List[List[str]]


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

    def _analyze_function_relationships(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze function call relationships"""
        function_calls = defaultdict(list)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                calls = []

                # Find all function calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            calls.append(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            if isinstance(child.func.value, ast.Name):
                                calls.append(f"{child.func.value.id}.{child.func.attr}")

                function_calls[node.name] = calls

        return dict(function_calls)

    def _analyze_data_relationships(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze data flow and shared data relationships"""
        shared_variables = set()
        data_dependencies = defaultdict(list)

        # Find global variables and shared data
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith('_'):
                        shared_variables.add(target.id)

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Track variable usage
                if node.id in shared_variables:
                    # Find which function uses this variable
                    current_function = None
                    for parent in reversed(list(ast.walk(tree))):
                        if isinstance(parent, ast.FunctionDef):
                            current_function = parent.name
                            break

                    if current_function:
                        data_dependencies[current_function].append(node.id)

        return {
            'shared_variables': list(shared_variables),
            'data_dependencies': dict(data_dependencies),
            'shared_variable_count': len(shared_variables)
        }

    def _build_relationship_graph(self, import_relationships: List[Dict[str, Any]],
                                class_relationships: List[Dict[str, Any]],
                                function_relationships: Dict[str, List[str]],
                                data_relationships: Dict[str, Any]) -> nx.DiGraph:
        """Build a graph representing all relationships"""

        G = nx.DiGraph()

        # Add import relationships
        for rel in import_relationships:
            if rel['type'] == 'import':
                G.add_edge('current_module', rel['module'], weight=1.0, type='import')
            elif rel['type'] == 'from_import':
                G.add_edge('current_module', rel['module'], weight=1.0, type='from_import')

        # Add class relationships
        for rel in class_relationships:
            if rel['type'] == 'inheritance':
                G.add_edge(rel['source_class'], rel['target_class'], weight=3.0, type='inheritance')
            elif rel['type'] == 'composition':
                G.add_edge(rel['source_class'], rel['target_class'], weight=2.0, type='composition')

        # Add function call relationships
        for func, calls in function_relationships.items():
            for call in calls:
                G.add_edge(func, call, weight=1.0, type='function_call')

        return G

    def _calculate_coupling_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate coupling and dependency metrics"""

        if graph.number_of_nodes() == 0:
            return {'coupling_score': 0.0, 'cohesion_score': 0.0}

        # Calculate coupling (connections between nodes)
        total_possible_connections = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
        actual_connections = graph.number_of_edges()

        coupling_score = actual_connections / max(total_possible_connections, 1)

        # Calculate in-degree and out-degree statistics
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]

        avg_in_degree = sum(in_degrees) / len(in_degrees) if in_degrees else 0
        avg_out_degree = sum(out_degrees) / len(out_degrees) if out_degrees else 0

        # Find highly coupled nodes
        high_coupling_threshold = avg_out_degree + 2
        highly_coupled_nodes = [node for node, degree in graph.out_degree()
                              if degree > high_coupling_threshold]

        return {
            'coupling_score': coupling_score,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'highly_coupled_nodes': highly_coupled_nodes,
            'isolated_nodes': [node for node, degree in graph.degree() if degree == 0]
        }

    def _identify_relationship_clusters(self, graph: nx.DiGraph) -> List[List[str]]:
        """Identify clusters of related components"""

        if graph.number_of_nodes() < 2:
            return []

        try:
            # Use community detection if networkx has it
            if hasattr(nx.algorithms.community, 'greedy_modularity_communities'):
                communities = nx.algorithms.community.greedy_modularity_communities(graph)
                return [list(community) for community in communities]
            else:
                # Fallback: use connected components
                components = list(nx.connected_components(graph.to_undirected()))
                return [list(component) for component in components]

        except Exception:
            # Ultimate fallback
            return [list(graph.nodes())]

    def _generate_relationship_recommendations(self, coupling_metrics: Dict[str, Any],
                                             clusters: List[List[str]],
                                             graph: nx.DiGraph) -> List[str]:
        """Generate recommendations based on relationship analysis"""
        recommendations = []

        coupling_score = coupling_metrics.get('coupling_score', 0)

        # Coupling-based recommendations
        if coupling_score > 0.7:
            recommendations.append("HIGH COUPLING: Consider breaking down into smaller modules")
            recommendations.append("Extract common interfaces to reduce coupling")

        elif coupling_score < 0.2:
            recommendations.append("LOW COUPLING: Consider consolidating related modules")

        # Highly coupled nodes
        highly_coupled = coupling_metrics.get('highly_coupled_nodes', [])
        if highly_coupled:
            recommendations.append(f"Highly coupled components: {', '.join(highly_coupled[:3])}")
            recommendations.append("Review dependencies for these components")

        # Isolated nodes
        isolated = coupling_metrics.get('isolated_nodes', [])
        if isolated:
            recommendations.append(f"Isolated components: {', '.join(isolated[:3])}")
            recommendations.append("Consider integrating or removing isolated components")

        # Cluster analysis
        if len(clusters) > 1:
            recommendations.append(f"Found {len(clusters)} relationship clusters")
            recommendations.append("Consider organizing code around these natural clusters")

        # Network analysis
        if graph.number_of_nodes() > 20 and coupling_metrics.get('avg_out_degree', 0) > 5:
            recommendations.append("Complex dependency network detected")
            recommendations.append("Consider introducing interface layers or facades")

        if not recommendations:
            recommendations.append("Relationship structure looks healthy")

        return recommendations

    def _fallback_relationship_analysis(self, content: str, file_path: str,
                                      error: SyntaxError) -> Dict[str, Any]:
        """Fallback analysis for files with syntax errors"""
        # Extract basic relationships from text despite syntax errors
        relationships = []

        # Find import patterns
        import_matches = re.findall(r'^(?:from|import)\s+([^\s;]+)', content, re.MULTILINE)
        for match in import_matches:
            relationships.append({
                'type': 'import',
                'module': match,
                'line_number': 0
            })

        return {
            'import_relationships': relationships,
            'class_relationships': [],
            'function_relationships': {},
            'data_relationships': {'shared_variables': [], 'data_dependencies': {}},
            'coupling_metrics': {'coupling_score': 0.0},
            'relationship_clusters': [],
            'relationship_graph': {'nodes': [], 'edges': [], 'node_count': 0, 'edge_count': 0},
            'recommendations': [f'Contains syntax error: {error}'],
            'file_path': file_path
        }


# Module-level functions for easy integration
def analyze_relationships(content: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Module-level function for relationship analysis"""
    analyzer = RelationshipAnalyzer()
    return analyzer.analyze_relationships(content, file_path)


def calculate_coupling_score(content: str) -> float:
    """Calculate coupling score for code content"""
    analyzer = RelationshipAnalyzer()
    result = analyzer.analyze_relationships(content)
    return result['coupling_metrics'].get('coupling_score', 0.0)


def find_highly_coupled_components(content: str) -> List[str]:
    """Find components with high coupling"""
    analyzer = RelationshipAnalyzer()
    result = analyzer.analyze_relationships(content)
    return result['coupling_metrics'].get('highly_coupled_nodes', [])


if __name__ == "__main__":
    # Example usage
    sample_code = """
import os
import sys
from pathlib import Path
from data_processor import DataProcessor
from model_trainer import ModelTrainer

class MLManager:
    def __init__(self, config: dict):
        self.config = config
        self.processor = DataProcessor(config)
        self.trainer = ModelTrainer(config)

    def run_pipeline(self):
        data = self.processor.load_data()
        model = self.trainer.train_model(data)
        return model

    def evaluate_model(self, model):
        return self.trainer.evaluate(model)

def main():
    config = {"data_path": "data.csv"}
    manager = MLManager(config)
    model = manager.run_pipeline()
    results = manager.evaluate_model(model)
    return results
"""

    analyzer = RelationshipAnalyzer()
    result = analyzer.analyze_relationships(sample_code, "sample.py")

    print("Relationship Analysis Results:")
    print(f"Coupling Score: {result['coupling_metrics']['coupling_score']:.3f}")
    print(f"Import Relationships: {len(result['import_relationships'])}")
    print(f"Class Relationships: {len(result['class_relationships'])}")
    print(f"Function Calls: {len(result['function_relationships'])}")
    print(f"Recommendations: {result['recommendations']}")
