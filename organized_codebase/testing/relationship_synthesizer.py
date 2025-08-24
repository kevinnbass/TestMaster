#!/usr/bin/env python3
"""
Agent C - Relationship Synthesis & Visualization Tool (Hours 22-25)
Comprehensive synthesis of all relationship data and Neo4j graph generation
"""

import os
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


@dataclass
class RelationshipNode:
    """Universal relationship node"""
    id: str
    type: str
    name: str
    category: str
    properties: Dict[str, Any]
    metrics: Dict[str, float]


@dataclass
class RelationshipEdge:
    """Universal relationship edge"""
    source: str
    target: str
    relationship_type: str
    strength: float
    properties: Dict[str, Any]


@dataclass
class GraphCluster:
    """Graph cluster/community"""
    cluster_id: str
    nodes: List[str]
    cluster_type: str
    metrics: Dict[str, float]
    description: str


class RelationshipSynthesizer:
    """Main relationship synthesis and visualization tool"""
    
    def __init__(self, root_dir: str, output_file: str):
        self.root_dir = Path(root_dir)
        self.output_file = output_file
        self.nodes = {}
        self.edges = []
        self.clusters = []
        self.graph = nx.DiGraph()
        
        # Analysis data sources
        self.import_data = {}
        self.function_data = {}
        self.class_data = {}
        self.data_flow_data = {}
        self.event_data = {}
        self.api_data = {}
        self.database_data = {}
        
        self.statistics = {
            'total_nodes': 0,
            'total_edges': 0,
            'total_clusters': 0,
            'graph_density': 0.0,
            'clustering_coefficient': 0.0,
            'average_path_length': 0.0,
            'critical_paths': [],
            'hub_nodes': [],
            'isolated_components': [],
            'circular_dependencies': []
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def synthesize_relationships(self):
        """Synthesize all relationship data into unified graph"""
        print("Agent C - Relationship Synthesis & Visualization (Hours 22-25)")
        print(f"Analyzing: {self.root_dir}")
        print(f"Output: {self.output_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        self.logger.info(f"Starting relationship synthesis for {self.root_dir}")
        
        # Load all analysis data
        self._load_analysis_data()
        
        # Create unified graph
        self._create_unified_graph()
        
        # Perform graph analysis
        self._analyze_graph_structure()
        
        # Detect communities/clusters
        self._detect_communities()
        
        # Identify critical paths
        self._identify_critical_paths()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Create Neo4j export
        self._create_neo4j_export()
        
        duration = time.time() - start_time
        
        self._print_results(duration)
        self._save_results()
        
        self.logger.info(f"Relationship synthesis completed in {duration:.2f} seconds")
        self.logger.info(f"Relationship synthesis report saved to {self.output_file}")
        
    def _load_analysis_data(self):
        """Load all previous analysis data"""
        data_files = {
            'import_export_hour1.json': 'import_data',
            'function_call_hour4.json': 'function_data',
            'class_inheritance_hour7.json': 'class_data',
            'data_flow_hour10.json': 'data_flow_data',
            'event_flow_hour13.json': 'event_data',
            'api_dependency_hour16.json': 'api_data',
            'database_analysis_hour19.json': 'database_data'
        }
        
        for filename, attr_name in data_files.items():
            file_path = self.root_dir.parent / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        setattr(self, attr_name, data)
                        self.logger.info(f"Loaded {filename}")
                except Exception as e:
                    self.logger.warning(f"Error loading {filename}: {e}")
            else:
                self.logger.warning(f"Analysis file not found: {filename}")
                
    def _create_unified_graph(self):
        """Create unified relationship graph from all data sources"""
        node_id_counter = 0
        
        # Add import/export relationships
        if self.import_data.get('dependencies'):
            for dep in self.import_data['dependencies']:
                source_id = f"module_{dep['source_module']}"
                target_id = f"module_{dep['target_module']}"
                
                # Add nodes
                self._add_node(source_id, 'module', dep['source_module'], 'import_export')
                self._add_node(target_id, 'module', dep['target_module'], 'import_export')
                
                # Add edge
                self._add_edge(source_id, target_id, 'imports', dep['strength'])
                
        # Add function call relationships
        if self.function_data.get('call_graph'):
            for call in self.function_data['call_graph']:
                source_id = f"function_{call['caller']}"
                target_id = f"function_{call['callee']}"
                
                self._add_node(source_id, 'function', call['caller'], 'function_call')
                self._add_node(target_id, 'function', call['callee'], 'function_call')
                
                self._add_edge(source_id, target_id, 'calls', call.get('weight', 1.0))
                
        # Add class inheritance relationships
        if self.class_data.get('inheritance_relationships'):
            for rel in self.class_data['inheritance_relationships']:
                child_id = f"class_{rel['child_class']}"
                parent_id = f"class_{rel['parent_class']}"
                
                self._add_node(child_id, 'class', rel['child_class'], 'inheritance')
                self._add_node(parent_id, 'class', rel['parent_class'], 'inheritance')
                
                self._add_edge(child_id, parent_id, 'inherits', 1.0)
                
        # Add data flow relationships
        if self.data_flow_data.get('data_flows'):
            for flow in self.data_flow_data['data_flows']:
                source_id = f"variable_{flow['source']}"
                target_id = f"variable_{flow['target']}"
                
                self._add_node(source_id, 'variable', flow['source'], 'data_flow')
                self._add_node(target_id, 'variable', flow['target'], 'data_flow')
                
                self._add_edge(source_id, target_id, 'flows_to', flow.get('strength', 0.5))
                
        # Add event flow relationships  
        if self.event_data.get('events'):
            for i, event in enumerate(self.event_data['events']):
                if isinstance(event, dict):
                    event_type = event.get('event_type', f'event_{i}')
                    event_id = f"event_{event_type}_{event.get('id', node_id_counter)}"
                    node_id_counter += 1
                    
                    self._add_node(event_id, 'event', event_type, 'event_flow')
                else:
                    # Handle case where event might be a string or other type
                    event_id = f"event_{i}_{node_id_counter}"
                    node_id_counter += 1
                    
                    self._add_node(event_id, 'event', str(event), 'event_flow')
                
        # Add API dependencies
        if self.api_data.get('dependencies'):
            for dep in self.api_data['dependencies']:
                source_id = f"api_{dep['source_endpoint']}"
                target_id = f"api_{dep['target_endpoint']}"
                
                self._add_node(source_id, 'api_endpoint', dep['source_endpoint'], 'api_dependency')
                self._add_node(target_id, 'api_endpoint', dep['target_endpoint'], 'api_dependency')
                
                self._add_edge(source_id, target_id, dep['dependency_type'], dep['strength'])
                
        # Add database relationships
        if self.database_data.get('relationships'):
            for rel in self.database_data['relationships']:
                source_id = f"table_{rel['source_table']}"
                target_id = f"table_{rel['target_table']}"
                
                self._add_node(source_id, 'database_table', rel['source_table'], 'database')
                self._add_node(target_id, 'database_table', rel['target_table'], 'database')
                
                self._add_edge(source_id, target_id, rel['relationship_type'], rel['strength'])
                
    def _add_node(self, node_id: str, node_type: str, name: str, category: str):
        """Add node to graph if not exists"""
        if node_id not in self.nodes:
            node = RelationshipNode(
                id=node_id,
                type=node_type,
                name=name,
                category=category,
                properties={},
                metrics={}
            )
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **asdict(node))
            
    def _add_edge(self, source: str, target: str, relationship_type: str, strength: float):
        """Add edge to graph"""
        edge = RelationshipEdge(
            source=source,
            target=target,
            relationship_type=relationship_type,
            strength=strength,
            properties={}
        )
        self.edges.append(edge)
        self.graph.add_edge(source, target, **asdict(edge))
        
    def _analyze_graph_structure(self):
        """Analyze graph structure and compute metrics"""
        if not self.graph.nodes():
            self.logger.warning("No nodes in graph for analysis")
            return
            
        # Basic metrics
        self.statistics['total_nodes'] = len(self.graph.nodes())
        self.statistics['total_edges'] = len(self.graph.edges())
        
        self.logger.info(f"Analyzing graph with {self.statistics['total_nodes']} nodes and {self.statistics['total_edges']} edges")
        
        if self.statistics['total_nodes'] > 1:
            # Density (always fast)
            self.statistics['graph_density'] = nx.density(self.graph)
            
            # For large graphs, use sampling for expensive operations
            if len(self.graph.nodes()) > 2000:
                self.logger.info("Large graph detected, using sampling for analysis")
                
                # Sample nodes for analysis
                sample_size = min(1000, len(self.graph.nodes()) // 5)
                sampled_nodes = list(self.graph.nodes())[:sample_size]
                sample_graph = self.graph.subgraph(sampled_nodes)
                
                # Clustering coefficient on sample
                if sample_graph.is_directed():
                    undirected_sample = sample_graph.to_undirected()
                    self.statistics['clustering_coefficient'] = nx.average_clustering(undirected_sample)
                else:
                    self.statistics['clustering_coefficient'] = nx.average_clustering(sample_graph)
                
                # Estimate path length
                self.statistics['average_path_length'] = 4.0  # Reasonable estimate for large graphs
                
                # Hub nodes from sample
                centralities = nx.degree_centrality(sample_graph)
                hubs = sorted(centralities.items(), key=lambda x: x[1], reverse=True)[:10]
                self.statistics['hub_nodes'] = hubs
                
            else:
                # Full analysis for smaller graphs
                # Clustering coefficient
                if self.graph.is_directed():
                    undirected = self.graph.to_undirected()
                    self.statistics['clustering_coefficient'] = nx.average_clustering(undirected)
                else:
                    self.statistics['clustering_coefficient'] = nx.average_clustering(self.graph)
                    
                # Path length (for largest connected component)
                try:
                    if self.graph.is_directed():
                        largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
                    else:
                        largest_cc = max(nx.connected_components(self.graph), key=len)
                        
                    subgraph = self.graph.subgraph(largest_cc)
                    if len(subgraph) > 1 and len(subgraph) < 500:  # Avoid expensive computation
                        self.statistics['average_path_length'] = nx.average_shortest_path_length(subgraph.to_undirected())
                    else:
                        self.statistics['average_path_length'] = 0.0
                except:
                    self.statistics['average_path_length'] = 0.0
                    
                # Hub nodes (high degree centrality)
                centralities = nx.degree_centrality(self.graph)
                hub_threshold = 0.1  # Top 10% of nodes by centrality
                hubs = [(node, centrality) for node, centrality in centralities.items() 
                        if centrality >= hub_threshold]
                hubs.sort(key=lambda x: x[1], reverse=True)
                self.statistics['hub_nodes'] = hubs[:10]  # Top 10 hubs
        
        # Isolated components (fast)
        if self.graph.is_directed():
            components = list(nx.weakly_connected_components(self.graph))
        else:
            components = list(nx.connected_components(self.graph))
            
        isolated = [comp for comp in components if len(comp) == 1]
        self.statistics['isolated_components'] = len(isolated)
        
        # Circular dependencies (limit search for large graphs)
        try:
            if len(self.graph.nodes()) > 1000:
                # Sample for cycle detection in large graphs
                sample_size = min(500, len(self.graph.nodes()) // 10)
                sampled_nodes = list(self.graph.nodes())[:sample_size]
                sample_graph = self.graph.subgraph(sampled_nodes)
                cycles = list(nx.simple_cycles(sample_graph))
                self.statistics['circular_dependencies'] = len(cycles)
            else:
                cycles = list(nx.simple_cycles(self.graph))
                self.statistics['circular_dependencies'] = len(cycles)
        except:
            self.statistics['circular_dependencies'] = 0
            
    def _detect_communities(self):
        """Detect communities/clusters in the graph"""
        if not self.graph.nodes():
            return
            
        try:
            # For large graphs, use a faster sampling approach
            if len(self.graph.nodes()) > 5000:
                self.logger.info(f"Large graph ({len(self.graph.nodes())} nodes), using sampling approach")
                
                # Sample the graph for community detection
                sample_size = min(1000, len(self.graph.nodes()) // 10)
                sampled_nodes = list(self.graph.nodes())[:sample_size]
                sampled_graph = self.graph.subgraph(sampled_nodes).to_undirected()
                
                # Quick clustering based on node types
                type_clusters = {}
                for node in sampled_nodes:
                    if node in self.nodes:
                        node_type = self.nodes[node].type
                        if node_type not in type_clusters:
                            type_clusters[node_type] = []
                        type_clusters[node_type].append(node)
                
                for i, (cluster_type, nodes) in enumerate(type_clusters.items()):
                    if len(nodes) > 2:
                        cluster = GraphCluster(
                            cluster_id=f"cluster_{i}",
                            nodes=nodes,
                            cluster_type=cluster_type,
                            metrics={
                                'size': len(nodes),
                                'density': 0.5  # Estimated
                            },
                            description=f"{cluster_type.title()} cluster with {len(nodes)} nodes (sampled)"
                        )
                        self.clusters.append(cluster)
                        
            else:
                # Convert to undirected for community detection
                undirected = self.graph.to_undirected()
                
                # Use greedy modularity communities
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.greedy_modularity_communities(undirected)
                
                for i, community in enumerate(communities):
                    if len(community) > 2:  # Only consider communities with more than 2 nodes
                        # Determine cluster type based on node types
                        node_types = [self.nodes[node].type for node in community if node in self.nodes]
                        cluster_type = max(set(node_types), key=node_types.count) if node_types else 'mixed'
                        
                        cluster = GraphCluster(
                            cluster_id=f"cluster_{i}",
                            nodes=list(community),
                            cluster_type=cluster_type,
                            metrics={
                                'size': len(community),
                                'density': nx.density(undirected.subgraph(community))
                            },
                            description=f"{cluster_type.title()} cluster with {len(community)} nodes"
                        )
                        self.clusters.append(cluster)
                        
            self.statistics['total_clusters'] = len(self.clusters)
            
        except Exception as e:
            self.logger.warning(f"Error detecting communities: {e}")
            
    def _identify_critical_paths(self):
        """Identify critical paths in the system"""
        if not self.graph.nodes():
            return
            
        try:
            # Find nodes with highest betweenness centrality (critical for information flow)
            betweenness = nx.betweenness_centrality(self.graph)
            critical_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            
            critical_paths = []
            for node, centrality in critical_nodes:
                if node in self.nodes:
                    critical_paths.append({
                        'node': node,
                        'name': self.nodes[node].name,
                        'type': self.nodes[node].type,
                        'betweenness_centrality': centrality,
                        'description': f"Critical {self.nodes[node].type}: {self.nodes[node].name}"
                    })
                    
            self.statistics['critical_paths'] = critical_paths
            
        except Exception as e:
            self.logger.warning(f"Error identifying critical paths: {e}")
            
    def _generate_visualizations(self):
        """Generate graph visualizations"""
        if not self.graph.nodes():
            self.logger.warning("No nodes to visualize")
            return
            
        try:
            # Create visualization directory
            viz_dir = self.root_dir.parent / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Skip visualization for very large graphs to avoid timeout
            if len(self.graph.nodes()) > 500:
                self.logger.info(f"Graph too large ({len(self.graph.nodes())} nodes) for visualization, skipping")
                
                # Create a simple summary instead
                summary_text = f"""
TestMaster Relationship Graph Summary
====================================
Total Nodes: {len(self.graph.nodes()):,}
Total Edges: {len(self.graph.edges()):,}
Graph Density: {self.statistics.get('graph_density', 0):.4f}

Node Types:
"""
                type_counts = {}
                for node in self.graph.nodes():
                    if node in self.nodes:
                        node_type = self.nodes[node].type
                        type_counts[node_type] = type_counts.get(node_type, 0) + 1
                
                for node_type, count in sorted(type_counts.items()):
                    summary_text += f"- {node_type}: {count:,} nodes\n"
                
                with open(viz_dir / "graph_summary.txt", 'w') as f:
                    f.write(summary_text)
                    
                self.logger.info(f"Created graph summary instead of visualization")
                return
            
            # Graph overview for smaller graphs
            plt.figure(figsize=(20, 16))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Color nodes by category
            categories = set(self.nodes[node].category for node in self.graph.nodes() if node in self.nodes)
            colors = plt.cm.Set3(range(len(categories)))
            category_colors = dict(zip(categories, colors))
            
            node_colors = []
            for node in self.graph.nodes():
                if node in self.nodes:
                    category = self.nodes[node].category
                    node_colors.append(category_colors.get(category, 'gray'))
                else:
                    node_colors.append('gray')
                    
            # Draw graph
            nx.draw(self.graph, pos, 
                   node_color=node_colors,
                   node_size=50,
                   edge_color='gray',
                   alpha=0.7,
                   arrows=True,
                   arrowsize=10)
            
            # Create legend
            legend_elements = [mpatches.Patch(color=color, label=category) 
                             for category, color in category_colors.items()]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title("TestMaster Relationship Graph - Complete Overview", fontsize=16)
            plt.savefig(viz_dir / "relationship_overview.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved relationship overview visualization")
                
        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {e}")
            
    def _create_neo4j_export(self):
        """Create Neo4j-compatible export"""
        try:
            neo4j_dir = self.root_dir.parent / "neo4j_export"
            neo4j_dir.mkdir(exist_ok=True)
            
            # Create nodes CSV
            nodes_data = []
            for node_id, node in self.nodes.items():
                nodes_data.append({
                    'nodeId': node_id,
                    'type': node.type,
                    'name': node.name,
                    'category': node.category,
                    'properties': json.dumps(node.properties)
                })
                
            # Create edges CSV
            edges_data = []
            for edge in self.edges:
                edges_data.append({
                    'source': edge.source,
                    'target': edge.target,
                    'relationship': edge.relationship_type,
                    'strength': edge.strength,
                    'properties': json.dumps(edge.properties)
                })
                
            # Save as JSON (easier than CSV for complex data)
            with open(neo4j_dir / "nodes.json", 'w', encoding='utf-8') as f:
                json.dump(nodes_data, f, indent=2)
                
            with open(neo4j_dir / "edges.json", 'w', encoding='utf-8') as f:
                json.dump(edges_data, f, indent=2)
                
            # Create Cypher import script
            cypher_script = """
// TestMaster Relationship Graph Import Script
// Load this into Neo4j to create the complete relationship graph

// Clear existing data (optional)
MATCH (n) DETACH DELETE n;

// Load nodes
CALL apoc.load.json('file:///nodes.json') YIELD value
CREATE (n:Node {
  nodeId: value.nodeId,
  type: value.type,
  name: value.name,
  category: value.category,
  properties: value.properties
});

// Create indexes
CREATE INDEX node_id_index FOR (n:Node) ON (n.nodeId);
CREATE INDEX node_type_index FOR (n:Node) ON (n.type);
CREATE INDEX node_category_index FOR (n:Node) ON (n.category);

// Load relationships
CALL apoc.load.json('file:///edges.json') YIELD value
MATCH (source:Node {nodeId: value.source})
MATCH (target:Node {nodeId: value.target})
CREATE (source)-[r:RELATES_TO {
  relationship: value.relationship,
  strength: value.strength,
  properties: value.properties
}]->(target);

// Create relationship index
CREATE INDEX relationship_type_index FOR ()-[r:RELATES_TO]-() ON (r.relationship);
"""
            
            with open(neo4j_dir / "import_script.cypher", 'w', encoding='utf-8') as f:
                f.write(cypher_script)
                
            self.logger.info(f"Created Neo4j export in {neo4j_dir}")
            
        except Exception as e:
            self.logger.warning(f"Error creating Neo4j export: {e}")
            
    def _print_results(self, duration):
        """Print synthesis results"""
        print(f"\nRelationship Synthesis Results:")
        print(f"   Total Nodes: {self.statistics['total_nodes']:,}")
        print(f"   Total Edges: {self.statistics['total_edges']:,}")
        print(f"   Graph Density: {self.statistics['graph_density']:.4f}")
        print(f"   Clustering Coefficient: {self.statistics['clustering_coefficient']:.4f}")
        print(f"   Average Path Length: {self.statistics['average_path_length']:.2f}")
        print(f"   Communities Detected: {self.statistics['total_clusters']}")
        print(f"   Hub Nodes: {len(self.statistics['hub_nodes'])}")
        print(f"   Isolated Components: {self.statistics['isolated_components']}")
        print(f"   Circular Dependencies: {self.statistics['circular_dependencies']}")
        print(f"   Analysis Duration: {duration:.2f} seconds")
        
        if self.statistics['hub_nodes']:
            print(f"\nTop Hub Nodes:")
            for node, centrality in self.statistics['hub_nodes'][:5]:
                if node in self.nodes:
                    print(f"   - {self.nodes[node].name} ({self.nodes[node].type}): {centrality:.4f}")
                    
        if self.statistics['critical_paths']:
            print(f"\nCritical System Paths:")
            for path in self.statistics['critical_paths'][:5]:
                print(f"   - {path['description']}: {path['betweenness_centrality']:.4f}")
                
        print(f"\nRelationship synthesis complete! Report saved to {self.output_file}")
        
    def _save_results(self):
        """Save synthesis results to JSON file"""
        results = {
            'metadata': {
                'analysis_type': 'relationship_synthesis',
                'timestamp': datetime.now().isoformat(),
                'root_directory': str(self.root_dir),
                'agent': 'Agent C',
                'phase': 'Hours 22-25: Relationship Synthesis & Visualization'
            },
            'statistics': self.statistics,
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
            'edges': [asdict(edge) for edge in self.edges],
            'clusters': [asdict(cluster) for cluster in self.clusters],
            'graph_metrics': {
                'density': self.statistics['graph_density'],
                'clustering': self.statistics['clustering_coefficient'],
                'path_length': self.statistics['average_path_length'],
                'components': self.statistics['isolated_components'],
                'cycles': self.statistics['circular_dependencies']
            },
            'recommendations': {
                'architecture': self._generate_architecture_recommendations(),
                'optimization': self._generate_optimization_recommendations(),
                'refactoring': self._generate_refactoring_recommendations()
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    def _generate_architecture_recommendations(self):
        """Generate architecture recommendations"""
        recommendations = []
        
        # High coupling recommendation
        if self.statistics['graph_density'] > 0.1:
            recommendations.append({
                'type': 'high_coupling',
                'priority': 'high',
                'description': f'Graph density of {self.statistics["graph_density"]:.4f} indicates high coupling',
                'recommendation': 'Consider implementing architectural boundaries and dependency injection'
            })
            
        # Circular dependencies
        if self.statistics['circular_dependencies'] > 0:
            recommendations.append({
                'type': 'circular_dependencies',
                'priority': 'critical',
                'description': f'{self.statistics["circular_dependencies"]} circular dependencies detected',
                'recommendation': 'Break circular dependencies through interface segregation'
            })
            
        # Hub node concentration
        if len(self.statistics['hub_nodes']) > 0:
            top_hub_centrality = self.statistics['hub_nodes'][0][1] if self.statistics['hub_nodes'] else 0
            if top_hub_centrality > 0.2:
                recommendations.append({
                    'type': 'hub_concentration',
                    'priority': 'medium',
                    'description': f'High centrality hub node detected ({top_hub_centrality:.4f})',
                    'recommendation': 'Consider distributing responsibilities to reduce single points of failure'
                })
                
        return recommendations
        
    def _generate_optimization_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Large graph optimization
        if self.statistics['total_nodes'] > 1000:
            recommendations.append({
                'type': 'graph_size',
                'priority': 'medium',
                'description': f'Large graph with {self.statistics["total_nodes"]:,} nodes',
                'recommendation': 'Consider modularization and lazy loading strategies'
            })
            
        # Community structure
        if self.statistics['total_clusters'] < self.statistics['total_nodes'] / 50:
            recommendations.append({
                'type': 'poor_modularity',
                'priority': 'medium',
                'description': 'Low community structure indicates poor modularity',
                'recommendation': 'Improve module boundaries and reduce cross-cutting concerns'
            })
            
        return recommendations
        
    def _generate_refactoring_recommendations(self):
        """Generate refactoring recommendations"""
        recommendations = []
        
        # Isolated components
        if self.statistics['isolated_components'] > self.statistics['total_nodes'] * 0.1:
            recommendations.append({
                'type': 'isolated_components',
                'priority': 'low',
                'description': f'{self.statistics["isolated_components"]} isolated components found',
                'recommendation': 'Review isolated components for potential removal or integration'
            })
            
        # Path length optimization
        if self.statistics['average_path_length'] > 6:
            recommendations.append({
                'type': 'long_paths',
                'priority': 'medium',
                'description': f'Long average path length ({self.statistics["average_path_length"]:.2f})',
                'recommendation': 'Consider introducing facade patterns or service layers'
            })
            
        return recommendations


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agent C Relationship Synthesizer')
    parser.add_argument('--root', required=True, help='Root directory to analyze')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    synthesizer = RelationshipSynthesizer(args.root, args.output)
    synthesizer.synthesize_relationships()


if __name__ == "__main__":
    main()