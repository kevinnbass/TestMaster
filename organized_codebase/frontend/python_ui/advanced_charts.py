#!/usr/bin/env python3
"""
Advanced Charts Component
=========================
STEELCLAD Atomized Module (<200 lines)
Extracted from advanced_visualization.py

Advanced chart components for complex data visualizations.
"""

import json
from datetime import datetime, timedelta
import random
from typing import Dict, Any, List, Optional, Tuple

class AdvancedCharts:
    """Atomic component for advanced chart visualizations."""
    
    def __init__(self):
        self.chart_registry = {}
        self.color_schemes = self._init_color_schemes()
        self.chart_animations = True
        
    def create_multi_axis_chart(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create multi-axis chart for comparing different metrics."""
        return {
            "type": "multiAxis",
            "data": {
                "labels": self._generate_time_labels(24),
                "datasets": [self._format_dataset(ds) for ds in datasets]
            },
            "options": {
                "responsive": True,
                "interaction": {"mode": "index", "intersect": False},
                "scales": self._create_multi_scales(datasets),
                "plugins": {"legend": {"position": "top"}}
            }
        }
    
    def create_3d_surface_plot(self, matrix_data: List[List[float]], 
                               title: str = "3D Surface") -> Dict[str, Any]:
        """Create 3D surface plot for multidimensional data."""
        return {
            "type": "surface3d",
            "data": {
                "z": matrix_data,
                "x": list(range(len(matrix_data[0]))),
                "y": list(range(len(matrix_data))),
                "colorscale": "Viridis"
            },
            "layout": {
                "title": title,
                "scene": {
                    "xaxis": {"title": "X Axis"},
                    "yaxis": {"title": "Y Axis"},
                    "zaxis": {"title": "Value"},
                    "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}}
                }
            }
        }
    
    def create_sankey_diagram(self, nodes: List[str], links: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create Sankey diagram for flow visualization."""
        return {
            "type": "sankey",
            "data": {
                "nodes": [{"id": i, "label": node} for i, node in enumerate(nodes)],
                "links": [
                    {
                        "source": link["source"],
                        "target": link["target"],
                        "value": link.get("value", 1),
                        "color": link.get("color", "#cccccc")
                    } for link in links
                ]
            },
            "options": {
                "nodeWidth": 15,
                "nodePadding": 10,
                "iterations": 32,
                "interactive": True
            }
        }
    
    def create_treemap(self, hierarchical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create treemap for hierarchical data visualization."""
        return {
            "type": "treemap",
            "data": self._process_hierarchical_data(hierarchical_data),
            "options": {
                "algorithm": "squarify",
                "colorScale": self.color_schemes["categorical"],
                "labels": {"display": True, "font": {"size": 12}},
                "tooltip": {"enabled": True}
            }
        }
    
    def create_polar_area_chart(self, categories: List[str], values: List[float]) -> Dict[str, Any]:
        """Create polar area chart for categorical comparisons."""
        return {
            "type": "polarArea",
            "data": {
                "labels": categories,
                "datasets": [{
                    "data": values,
                    "backgroundColor": self._generate_gradient_colors(len(categories)),
                    "borderWidth": 2,
                    "borderColor": "#ffffff"
                }]
            },
            "options": {
                "responsive": True,
                "animation": {"animateRotate": True, "animateScale": True},
                "scale": {"ticks": {"beginAtZero": True}}
            }
        }
    
    def create_bubble_chart(self, data_points: List[Dict[str, float]]) -> Dict[str, Any]:
        """Create bubble chart for three-dimensional data points."""
        return {
            "type": "bubble",
            "data": {
                "datasets": [{
                    "label": "Data Points",
                    "data": [
                        {
                            "x": point.get("x", 0),
                            "y": point.get("y", 0),
                            "r": point.get("size", 5)
                        } for point in data_points
                    ],
                    "backgroundColor": self._generate_bubble_colors(len(data_points))
                }]
            },
            "options": {
                "responsive": True,
                "scales": {
                    "x": {"type": "linear", "position": "bottom"},
                    "y": {"type": "linear", "position": "left"}
                }
            }
        }
    
    def create_network_graph(self, nodes: List[Dict[str, Any]], 
                            edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create network graph visualization."""
        return {
            "type": "network",
            "data": {
                "nodes": nodes,
                "edges": edges
            },
            "options": {
                "physics": {
                    "enabled": True,
                    "barnesHut": {"gravitationalConstant": -2000, "springConstant": 0.04}
                },
                "interaction": {"hover": True, "tooltipDelay": 200},
                "nodes": {"shape": "dot", "size": 10, "font": {"size": 12}}
            }
        }
    
    def create_combined_chart(self, chart_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create combined chart with multiple visualization types."""
        return {
            "type": "combined",
            "charts": chart_configs,
            "layout": {
                "grid": {"rows": 2, "cols": 2},
                "spacing": 10,
                "responsive": True
            }
        }
    
    def _generate_time_labels(self, hours: int) -> List[str]:
        """Generate time labels for charts."""
        now = datetime.now()
        return [(now - timedelta(hours=i)).strftime("%H:%M") 
                for i in range(hours, 0, -1)]
    
    def _format_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Format dataset for chart rendering."""
        return {
            "label": dataset.get("label", "Dataset"),
            "data": dataset.get("data", []),
            "borderColor": dataset.get("color", "#2196F3"),
            "backgroundColor": dataset.get("bgColor", "rgba(33, 150, 243, 0.1)"),
            "yAxisID": dataset.get("axis", "y"),
            "tension": 0.4
        }
    
    def _create_multi_scales(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create multiple scales for multi-axis chart."""
        scales = {"x": {"type": "category"}}
        for i, ds in enumerate(datasets):
            axis_id = ds.get("axis", f"y{i}")
            scales[axis_id] = {
                "type": "linear",
                "position": "left" if i == 0 else "right",
                "grid": {"drawOnChartArea": i == 0}
            }
        return scales
    
    def _process_hierarchical_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process hierarchical data for treemap."""
        processed = []
        for key, value in data.items():
            if isinstance(value, dict):
                processed.extend(self._process_hierarchical_data(value))
            else:
                processed.append({"name": key, "value": value})
        return processed
    
    def _generate_gradient_colors(self, count: int) -> List[str]:
        """Generate gradient colors for charts."""
        base_hue = 210  # Blue base
        return [f"hsl({base_hue + i * 30}, 70%, 50%)" for i in range(count)]
    
    def _generate_bubble_colors(self, count: int) -> List[str]:
        """Generate colors for bubble chart."""
        return [f"rgba({random.randint(0,255)}, {random.randint(0,255)}, "
                f"{random.randint(0,255)}, 0.6)" for _ in range(count)]
    
    def _init_color_schemes(self) -> Dict[str, List[str]]:
        """Initialize color schemes for charts."""
        return {
            "categorical": ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"],
            "sequential": ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0"],
            "diverging": ["#F44336", "#FF9800", "#FFEB3B", "#8BC34A", "#4CAF50"]
        }

# Module exports
__all__ = ['AdvancedCharts']