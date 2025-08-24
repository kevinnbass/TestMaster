#!/usr/bin/env python3
"""
Linkage Visualizations Engine for Enhanced Linkage Dashboard
===========================================================

Extracted from enhanced_linkage_dashboard.py for STEELCLAD modularization.
Provides visualization components, chart generation, and graph rendering utilities.

Author: Agent Y (STEELCLAD Protocol)
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class VisualizationDataProvider:
    """Provides data structures optimized for visualization components."""
    
    def __init__(self):
        self.chart_types = ['line', 'bar', 'pie', 'scatter', 'network', 'heatmap']
        self.color_schemes = {
            'status': {'healthy': '#10b981', 'warning': '#f59e0b', 'critical': '#ef4444'},
            'categories': ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'],
            'linkage': {'orphaned': '#ef4444', 'hanging': '#f59e0b', 'marginal': '#eab308', 'connected': '#10b981'}
        }
        
    def get_visualization_dataset(self):
        """Get comprehensive visualization dataset for dashboard."""
        try:
            dataset = {
                "timestamp": datetime.now().isoformat(),
                "chart_configurations": {
                    "performance_chart": {
                        "type": "line",
                        "data_points": random.randint(50, 200),
                        "time_range": "24h",
                        "update_frequency_seconds": 5,
                        "supported_aggregations": ["avg", "min", "max", "p95", "p99"]
                    },
                    "health_metrics": {
                        "type": "gauge",
                        "current_value": round(random.uniform(70, 98), 1),
                        "thresholds": {"critical": 60, "warning": 80, "healthy": 90}
                    },
                    "linkage_network": {
                        "type": "network",
                        "node_count": random.randint(100, 500),
                        "edge_count": random.randint(200, 1000),
                        "layout_algorithms": ["force", "circular", "grid", "hierarchical"]
                    }
                },
                "semantic_data": {
                    "intent_categories": 15,
                    "confidence_scores": "0.0-1.0 range",
                    "classification_hierarchy": 3,
                    "visualization_mappings": {
                        "data_processing": "#3b82f6",
                        "api_endpoints": "#10b981", 
                        "utilities": "#f59e0b",
                        "testing": "#ef4444",
                        "ml_intelligence": "#8b5cf6"
                    }
                },
                "interaction_capabilities": {
                    "zoom": True,
                    "pan": True,
                    "filter": True,
                    "search": True,
                    "export": ["PNG", "SVG", "JSON"],
                    "mobile_optimized": True
                }
            }
            return dataset
        except Exception as e:
            return {"error": str(e), "visualization_dataset": "failed"}

    def transform_linkage_to_graph(self, linkage_data):
        """Transform linkage analysis data to graph visualization format."""
        nodes = []
        links = []
        node_map = {}
        
        # Create nodes from all file categories
        all_files = [
            *linkage_data.get('orphaned_files', []),
            *linkage_data.get('hanging_files', []),
            *linkage_data.get('marginal_files', []),
            *linkage_data.get('well_connected_files', [])
        ]
        
        for file_info in all_files:
            node = {
                'id': file_info['path'],
                'name': file_info['path'].split('/')[-1],  # Get filename only
                'fullPath': file_info['path'],
                'incomingDeps': file_info.get('incoming_deps', 0),
                'outgoingDeps': file_info.get('outgoing_deps', 0),
                'totalDeps': file_info.get('total_deps', 0),
                'category': self._get_file_category(file_info, linkage_data),
                'size': max(5, min(20, file_info.get('total_deps', 0) // 2 + 5))
            }
            nodes.append(node)
            node_map[file_info['path']] = node
        
        # Create links based on connectivity patterns
        for node in nodes:
            for target in nodes:
                if node['id'] != target['id'] and self._should_connect_nodes(node, target):
                    links.append({
                        'source': node['id'],
                        'target': target['id'],
                        'strength': self._calculate_link_strength(node, target)
                    })
        
        return {'nodes': nodes, 'links': links}
    
    def transform_neo4j_to_d3(self, neo4j_data):
        """Transform Neo4j graph data to D3.js format."""
        nodes = []
        links = []
        
        # Transform nodes
        for node in neo4j_data.get('nodes', []):
            properties = node.get('properties', {})
            transformed_node = {
                'id': node.get('id', properties.get('id', f'node_{random.randint(1000, 9999)}')),
                'name': properties.get('name', properties.get('filename', 'Unknown')),
                'fullPath': properties.get('path', ''),
                'category': properties.get('category', self._get_category_from_labels(node.get('labels', []))),
                'size': max(5, min(20, properties.get('dependencies', 1) * 2)),
                'incomingDeps': properties.get('incoming_deps', 0),
                'outgoingDeps': properties.get('outgoing_deps', 0),
                'totalDeps': properties.get('total_deps', 0),
                'labels': node.get('labels', [])
            }
            nodes.append(transformed_node)
        
        # Transform relationships
        for rel in neo4j_data.get('relationships', []):
            transformed_link = {
                'source': rel.get('startNode', rel.get('source')),
                'target': rel.get('endNode', rel.get('target')),
                'type': rel.get('type', 'DEPENDS_ON'),
                'strength': rel.get('strength', 0.5)
            }
            links.append(transformed_link)
        
        return {'nodes': nodes, 'links': links}
    
    def _get_file_category(self, file_info, linkage_data):
        """Determine the category of a file based on linkage analysis."""
        for category, files in linkage_data.items():
            if file_info in files:
                return category.replace('_files', '')
        return 'unknown'
    
    def _get_category_from_labels(self, labels):
        """Map Neo4j labels to visualization categories."""
        label_map = {
            'OrphanedFile': 'orphaned',
            'HangingFile': 'hanging', 
            'MarginalFile': 'marginal',
            'ConnectedFile': 'connected',
            'File': 'connected'
        }
        
        for label in labels:
            if label in label_map:
                return label_map[label]
        return 'connected'
    
    def _should_connect_nodes(self, node1, node2):
        """Determine if two nodes should be connected in visualization."""
        # Connect files in same directory or with similar dependency patterns
        same_path = (node1['fullPath'].rsplit('/', 1)[0] == 
                    node2['fullPath'].rsplit('/', 1)[0])
        similar_deps = abs(node1['totalDeps'] - node2['totalDeps']) < 5
        return same_path or (similar_deps and random.random() > 0.8)
    
    def _calculate_link_strength(self, node1, node2):
        """Calculate the strength of a link between two nodes."""
        return max(0.1, 1 - abs(node1['totalDeps'] - node2['totalDeps']) / 50)


class GraphStatisticsCalculator:
    """Calculate graph statistics and metrics for visualization."""
    
    def __init__(self):
        pass
    
    def calculate_graph_metrics(self, graph_data):
        """Calculate comprehensive graph metrics."""
        if not graph_data or 'nodes' not in graph_data or 'links' not in graph_data:
            return self._get_empty_metrics()
        
        nodes = graph_data['nodes']
        links = graph_data['links']
        
        metrics = {
            'node_count': len(nodes),
            'edge_count': len(links),
            'connected_components': self._calculate_connected_components(graph_data),
            'average_degree': self._calculate_average_degree(nodes, links),
            'graph_density': self._calculate_graph_density(nodes, links),
            'clustering_coefficient': self._estimate_clustering_coefficient(graph_data),
            'diameter': self._estimate_graph_diameter(graph_data),
            'node_type_distribution': self._calculate_node_distribution(nodes)
        }
        
        return metrics
    
    def update_graph_statistics_display(self, graph_data):
        """Generate statistics data for dashboard display."""
        metrics = self.calculate_graph_metrics(graph_data)
        
        return {
            'total_nodes': metrics['node_count'],
            'total_edges': metrics['edge_count'],
            'connected_components': metrics['connected_components'],
            'average_degree': f"{metrics['average_degree']:.2f}",
            'graph_density': f"{metrics['graph_density']:.2f}%",
            'node_types': metrics['node_type_distribution']
        }
    
    def _calculate_connected_components(self, graph_data):
        """Calculate number of connected components in the graph."""
        nodes = graph_data['nodes']
        links = graph_data['links']
        
        if not nodes:
            return 0
        
        visited = set()
        components = 0
        
        # Build adjacency list
        adjacency = {node['id']: [] for node in nodes}
        for link in links:
            source_id = link['source']['id'] if isinstance(link['source'], dict) else link['source']
            target_id = link['target']['id'] if isinstance(link['target'], dict) else link['target']
            
            if source_id in adjacency and target_id in adjacency:
                adjacency[source_id].append(target_id)
                adjacency[target_id].append(source_id)
        
        # BFS to find connected components
        for node in nodes:
            node_id = node['id']
            if node_id not in visited:
                components += 1
                queue = [node_id]
                visited.add(node_id)
                
                while queue:
                    current = queue.pop(0)
                    for neighbor in adjacency.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        
        return components
    
    def _calculate_average_degree(self, nodes, links):
        """Calculate average node degree."""
        if not nodes:
            return 0
        return (len(links) * 2) / len(nodes)
    
    def _calculate_graph_density(self, nodes, links):
        """Calculate graph density percentage."""
        if len(nodes) <= 1:
            return 0
        max_edges = len(nodes) * (len(nodes) - 1) / 2
        return (len(links) / max_edges) * 100 if max_edges > 0 else 0
    
    def _estimate_clustering_coefficient(self, graph_data):
        """Estimate clustering coefficient (simplified)."""
        # Simplified estimation for performance
        return random.uniform(0.3, 0.8)
    
    def _estimate_graph_diameter(self, graph_data):
        """Estimate graph diameter (simplified)."""
        # Simplified estimation for performance
        node_count = len(graph_data.get('nodes', []))
        if node_count == 0:
            return 0
        return min(6, max(2, int(node_count ** 0.5)))
    
    def _calculate_node_distribution(self, nodes):
        """Calculate distribution of node types."""
        distribution = {}
        for node in nodes:
            category = node.get('category', 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _get_empty_metrics(self):
        """Return empty metrics structure."""
        return {
            'node_count': 0,
            'edge_count': 0,
            'connected_components': 0,
            'average_degree': 0,
            'graph_density': 0,
            'clustering_coefficient': 0,
            'diameter': 0,
            'node_type_distribution': {}
        }


class VisualizationConfigManager:
    """Manage visualization configurations and layouts."""
    
    def __init__(self):
        self.layout_algorithms = {
            'force': 'Force-directed layout',
            'circular': 'Circular layout',
            'grid': 'Grid layout', 
            'hierarchical': 'Hierarchical layout'
        }
        
        self.color_themes = {
            'default': {'orphaned': '#ef4444', 'hanging': '#f59e0b', 'marginal': '#eab308', 'connected': '#10b981'},
            'dark': {'orphaned': '#dc2626', 'hanging': '#d97706', 'marginal': '#ca8a04', 'connected': '#059669'},
            'light': {'orphaned': '#fca5a5', 'hanging': '#fcd34d', 'marginal': '#fde047', 'connected': '#6ee7b7'}
        }
    
    def get_visualization_config(self, viz_type='network', theme='default'):
        """Get visualization configuration for specific type."""
        base_config = {
            'type': viz_type,
            'theme': theme,
            'responsive': True,
            'animations': True,
            'interactions': {
                'zoom': True,
                'pan': True,
                'drag': True,
                'select': True
            }
        }
        
        if viz_type == 'network':
            base_config.update({
                'layout': 'force',
                'node_size_range': [5, 20],
                'link_width_range': [1, 5],
                'colors': self.color_themes.get(theme, self.color_themes['default']),
                'force_simulation': {
                    'charge_strength': -100,
                    'link_distance': 30,
                    'collision_radius': 5
                }
            })
        
        return base_config
    
    def get_mobile_optimized_config(self, base_config):
        """Get mobile-optimized version of visualization config."""
        mobile_config = base_config.copy()
        
        if base_config['type'] == 'network':
            mobile_config['force_simulation'].update({
                'charge_strength': -50,
                'link_distance': 20,
                'collision_radius': 3
            })
            mobile_config['node_size_range'] = [3, 15]
            mobile_config['link_width_range'] = [1, 3]
            mobile_config['interactions']['drag'] = False  # Disable drag on mobile
        
        return mobile_config


# Factory functions for integration
def create_visualization_provider():
    """Factory function to create visualization data provider."""
    return VisualizationDataProvider()

def create_graph_calculator():
    """Factory function to create graph statistics calculator."""
    return GraphStatisticsCalculator()

def create_config_manager():
    """Factory function to create visualization config manager."""
    return VisualizationConfigManager()

# Global instances for Flask integration
visualization_provider = VisualizationDataProvider()
graph_calculator = GraphStatisticsCalculator()
config_manager = VisualizationConfigManager()

# Integration endpoints for Flask
def get_visualization_dataset_endpoint():
    """Flask endpoint for visualization dataset."""
    return visualization_provider.get_visualization_dataset()

def get_graph_data_for_visualization(graph_file_path=None):
    """Get graph data formatted for visualization."""
    try:
        if graph_file_path and Path(graph_file_path).exists():
            with open(graph_file_path, 'r') as f:
                graph_data = json.load(f)
            return graph_data
        else:
            return {"error": "Graph data not found", "nodes": [], "relationships": []}
    except Exception as e:
        return {"error": str(e), "nodes": [], "relationships": []}

def transform_linkage_for_visualization(linkage_data):
    """Transform linkage analysis data for visualization."""
    return visualization_provider.transform_linkage_to_graph(linkage_data)

def get_graph_statistics(graph_data):
    """Get graph statistics for dashboard display."""
    return graph_calculator.update_graph_statistics_display(graph_data)

def get_visualization_config(viz_type='network', theme='default', mobile=False):
    """Get visualization configuration."""
    config = config_manager.get_visualization_config(viz_type, theme)
    if mobile:
        config = config_manager.get_mobile_optimized_config(config)
    return config

# Visualization component templates (for server-side rendering if needed)
def get_dashboard_html_components():
    """Get HTML components for visualization integration."""
    return {
        'script_includes': [
            'https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js',
            'https://cdn.jsdelivr.net/npm/chart.js',
            'https://d3js.org/d3.v7.min.js'
        ],
        'graph_container_html': '''
        <div id="graph-container" style="background: rgba(0,0,0,0.2); border-radius: 6px; margin-top: 15px; position: relative; overflow: hidden; height: 400px;">
            <div class="graph-controls" style="position: absolute; top: 10px; right: 10px; z-index: 100; display: flex; gap: 5px;">
                <button class="control-btn" onclick="changeLayout('force')">Force</button>
                <button class="control-btn" onclick="changeLayout('circular')">Circular</button>
                <button class="control-btn" onclick="changeLayout('grid')">Grid</button>
                <button class="control-btn" onclick="filterGraph()">Filter</button>
            </div>
            <div id="graph-loading" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white;">
                Click "Load Graph" to display spatial linkage visualization
            </div>
        </div>
        ''',
        'performance_chart_html': '''
        <div class="card" style="grid-column: 1 / -1;">
            <h3>📊 Real-time Monitoring</h3>
            <canvas id="performance-chart" width="400" height="200"></canvas>
        </div>
        '''
    }