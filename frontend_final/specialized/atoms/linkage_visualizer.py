#!/usr/bin/env python3
"""
Linkage Visualizer Atomic Component
====================================
STEELCLAD Atomized Module (<200 lines)
Extracted from linkage_dashboard_comprehensive.py

Handles linkage graph visualization rendering for the dashboard.
"""

from datetime import datetime
from pathlib import Path
import json
import random
from typing import Dict, List, Any

class LinkageVisualizer:
    """Atomic component for rendering linkage visualizations."""
    
    def __init__(self):
        self.visualization_cache = {}
        self.graph_stats = {
            "nodes": 0,
            "edges": 0,
            "orphaned": 0,
            "hanging": 0
        }
    
    def render_linkage_graph(self, linkage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render linkage analysis as a graph visualization.
        
        Args:
            linkage_data: Linkage analysis results
            
        Returns:
            Graph visualization data for frontend
        """
        visualization = {
            "timestamp": datetime.now().isoformat(),
            "graph": {
                "nodes": [],
                "edges": [],
                "clusters": []
            },
            "statistics": {},
            "render_config": self._get_render_config()
        }
        
        # Process orphaned files as disconnected nodes
        for file_info in linkage_data.get("orphaned_files", []):
            visualization["graph"]["nodes"].append({
                "id": file_info["path"],
                "label": Path(file_info["path"]).name,
                "type": "orphaned",
                "color": "#ff4444",
                "size": 20,
                "x": random.uniform(-500, 500),
                "y": random.uniform(-500, 500)
            })
        
        # Process hanging files as weakly connected nodes
        for file_info in linkage_data.get("hanging_files", []):
            visualization["graph"]["nodes"].append({
                "id": file_info["path"],
                "label": Path(file_info["path"]).name,
                "type": "hanging",
                "color": "#ff9944",
                "size": 25,
                "x": random.uniform(-500, 500),
                "y": random.uniform(-500, 500)
            })
        
        # Process marginal files
        for file_info in linkage_data.get("marginal_files", []):
            visualization["graph"]["nodes"].append({
                "id": file_info["path"],
                "label": Path(file_info["path"]).name,
                "type": "marginal",
                "color": "#ffdd44",
                "size": 18,
                "x": random.uniform(-500, 500),
                "y": random.uniform(-500, 500)
            })
        
        # Process well-connected files
        for file_info in linkage_data.get("well_connected_files", []):
            visualization["graph"]["nodes"].append({
                "id": file_info["path"],
                "label": Path(file_info["path"]).name,
                "type": "well_connected",
                "color": "#44ff44",
                "size": 30,
                "x": random.uniform(-500, 500),
                "y": random.uniform(-500, 500)
            })
        
        # Update statistics
        visualization["statistics"] = {
            "total_nodes": len(visualization["graph"]["nodes"]),
            "orphaned_count": len(linkage_data.get("orphaned_files", [])),
            "hanging_count": len(linkage_data.get("hanging_files", [])),
            "marginal_count": len(linkage_data.get("marginal_files", [])),
            "well_connected_count": len(linkage_data.get("well_connected_files", [])),
            "analysis_coverage": linkage_data.get("analysis_coverage", "0/0")
        }
        
        self.graph_stats.update(visualization["statistics"])
        return visualization
    
    def render_dependency_edges(self, dependencies: List[tuple]) -> List[Dict[str, Any]]:
        """Render dependency relationships as edges."""
        edges = []
        for source, target in dependencies:
            edges.append({
                "source": source,
                "target": target,
                "type": "dependency",
                "color": "#cccccc",
                "width": 1
            })
        return edges
    
    def render_clusters(self, file_groups: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Render file clusters for better organization."""
        clusters = []
        colors = ["#e3f2fd", "#fff3e0", "#f3e5f5", "#e8f5e9", "#fce4ec"]
        
        for idx, (group_name, files) in enumerate(file_groups.items()):
            clusters.append({
                "id": group_name,
                "label": group_name,
                "color": colors[idx % len(colors)],
                "members": files,
                "size": len(files)
            })
        return clusters
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration for the frontend."""
        return self._get_render_config()
    
    def _get_render_config(self) -> Dict[str, Any]:
        """Internal method to get render configuration."""
        return {
            "layout": "force-directed",
            "physics": {
                "enabled": True,
                "gravity": -50,
                "spring_length": 100,
                "spring_strength": 0.01
            },
            "interaction": {
                "hover": True,
                "zoom": True,
                "drag": True,
                "select": True
            },
            "animation": {
                "enabled": True,
                "duration": 500
            }
        }
    
    def export_visualization(self, format: str = "json") -> str:
        """Export visualization data in specified format."""
        if format == "json":
            return json.dumps(self.graph_stats, indent=2)
        elif format == "svg":
            # Simplified SVG export
            return f'<svg><text>Nodes: {self.graph_stats["nodes"]}</text></svg>'
        return ""

# Module exports
__all__ = ['LinkageVisualizer']