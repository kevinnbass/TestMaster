#!/usr/bin/env python3
"""
STEELCLAD MODULE: Chart.js and D3.js Integration Engine
=======================================================

Chart integration classes extracted from unified_dashboard_modular.py
Original: 3,977 lines → Chart Integration Module: ~320 lines

Complete functionality extraction with zero regression.

Author: Agent X (STEELCLAD Anti-Regression Modularization)
"""

import time
from datetime import datetime
from enum import Enum


class ChartType(Enum):
    """Supported chart types for comprehensive visualization."""
    # Chart.js standard charts
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    RADAR = "radar"
    POLAR_AREA = "polarArea"
    SCATTER = "scatter"
    BUBBLE = "bubble"
    
    # D3.js advanced visualizations
    TREEMAP = "treemap"
    FORCE_DIRECTED = "forceDirected"
    SANKEY = "sankey"
    CHORD = "chord"
    HEATMAP = "heatmap"
    NETWORK = "network"
    TIMELINE = "timeline"
    SUNBURST = "sunburst"


class ChartIntegrationEngine:
    """
    Comprehensive chart integration engine supporting both Chart.js and D3.js
    for standard and advanced data visualizations.
    """
    
    def __init__(self):
        self.chart_configs = {}
        self.active_charts = {}
        self.export_formats = ['png', 'svg', 'pdf', 'csv']
        self.performance_metrics = {
            'render_times': [],
            'data_points_processed': 0,
            'charts_created': 0,
            'exports_generated': 0
        }
        
        # Chart.js default configurations
        self.chartjs_defaults = {
            'responsive': True,
            'maintainAspectRatio': False,
            'animation': {'duration': 750},
            'interaction': {
                'intersect': False,
                'mode': 'index'
            }
        }
        
        # D3.js default configurations
        self.d3_defaults = {
            'width': 800,
            'height': 600,
            'margin': {'top': 20, 'right': 30, 'bottom': 40, 'left': 50},
            'animation_duration': 750
        }
    
    def create_chart(self, chart_id: str, chart_type: str, data: dict, options: dict = None):
        """Create a chart with Chart.js or D3.js based on type."""
        try:
            chart_type_enum = ChartType(chart_type)
        except ValueError:
            chart_type_enum = ChartType.LINE  # Default fallback
        
        # Determine library and create configuration
        if chart_type_enum in [ChartType.LINE, ChartType.BAR, ChartType.PIE, 
                              ChartType.DOUGHNUT, ChartType.RADAR, ChartType.POLAR_AREA,
                              ChartType.SCATTER, ChartType.BUBBLE]:
            config = self._create_chartjs_config(chart_type_enum, data, options)
        else:
            config = self._create_d3_config(chart_type_enum, data, options)
        
        # Store chart configuration
        self.chart_configs[chart_id] = config
        self.active_charts[chart_id] = {
            'type': chart_type_enum,
            'created_at': datetime.now(),
            'data_points': self._count_data_points(data)
        }
        
        self.performance_metrics['charts_created'] += 1
        return config
    
    def _create_chartjs_config(self, chart_type: ChartType, data: dict, options: dict = None):
        """Create Chart.js configuration for standard charts."""
        config = {
            'type': chart_type.value,
            'data': data,
            'options': {**self.chartjs_defaults}
        }
        
        # Apply custom options
        if options:
            config['options'].update(options)
        
        # Type-specific configurations
        if chart_type == ChartType.LINE:
            config['options']['scales'] = {
                'x': {'display': True},
                'y': {'display': True, 'beginAtZero': True}
            }
            config['options']['elements'] = {'line': {'tension': 0.4}}
        elif chart_type == ChartType.BAR:
            config['options']['scales'] = {
                'x': {'display': True, 'stacked': False},
                'y': {'display': True, 'stacked': False, 'beginAtZero': True}
            }
        elif chart_type in [ChartType.PIE, ChartType.DOUGHNUT]:
            config['options']['cutout'] = '50%' if chart_type == ChartType.DOUGHNUT else '0%'
            config['options']['plugins'] = {'legend': {'display': True}}
        elif chart_type == ChartType.RADAR:
            config['options']['scales'] = {'r': {'beginAtZero': True}}
        elif chart_type == ChartType.SCATTER:
            config['options']['scales'] = {
                'x': {'type': 'linear', 'position': 'bottom'},
                'y': {'type': 'linear'}
            }
        
        return config
    
    def _create_d3_config(self, chart_type: ChartType, data: dict, options: dict = None):
        """Create D3.js configuration for advanced visualizations."""
        config = {
            'type': chart_type.value,
            'data': data,
            'settings': {**self.d3_defaults}
        }
        
        # Apply custom options
        if options:
            config['settings'].update(options)
        
        # Type-specific configurations for advanced D3.js charts
        if chart_type == ChartType.TREEMAP:
            config['settings'].update({
                'tile': 'd3.treemapSquarify',
                'padding': 1,
                'round': True
            })
        elif chart_type == ChartType.FORCE_DIRECTED:
            config['settings']['force'] = {
                'charge': -300,
                'link_distance': 50,
                'collision_radius': 10,
                'alpha_decay': 0.0228
            }
        elif chart_type == ChartType.SANKEY:
            config['settings'].update({
                'node_width': 15,
                'node_padding': 10,
                'iterations': 32
            })
        elif chart_type == ChartType.HEATMAP:
            config['settings'].update({
                'color_scale': 'd3.interpolateRdYlBu',
                'cell_size': 20,
                'cell_padding': 2
            })
        elif chart_type == ChartType.NETWORK:
            config['settings'].update({
                'node_radius': 5,
                'link_width': 1,
                'charge_strength': -100
            })
        elif chart_type == ChartType.TIMELINE:
            config['settings'].update({
                'axis_format': '%Y-%m-%d',
                'item_height': 20,
                'lane_height': 40
            })
        elif chart_type == ChartType.SUNBURST:
            config['settings'].update({
                'radius_scale': 'd3.scaleSqrt',
                'arc_width': 'd3.arc',
                'color_scale': 'd3.scaleOrdinal(d3.schemeCategory10)'
            })
        
        return config
    
    def export_chart(self, chart_id: str, format: str):
        """Export chart in specified format."""
        if chart_id not in self.chart_configs:
            raise ValueError(f"Chart {chart_id} not found")
        
        if format not in self.export_formats:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.performance_metrics['exports_generated'] += 1
        
        # Mock export functionality
        return {
            "chart_id": chart_id,
            "format": format,
            "status": "exported",
            "timestamp": datetime.now().isoformat(),
            "file_size": "~50KB"  # Simulated
        }
    
    def get_supported_chart_types(self):
        """Get all supported chart types with their capabilities."""
        return {
            "chartjs_types": {
                "line": {"library": "Chart.js", "best_for": "Time series data"},
                "bar": {"library": "Chart.js", "best_for": "Categorical comparisons"},
                "pie": {"library": "Chart.js", "best_for": "Part-to-whole relationships"},
                "scatter": {"library": "Chart.js", "best_for": "Correlation analysis"}
            },
            "d3_types": {
                "treemap": {"library": "D3.js", "best_for": "Hierarchical data"},
                "force_directed": {"library": "D3.js", "best_for": "Network relationships"},
                "sankey": {"library": "D3.js", "best_for": "Flow diagrams"},
                "heatmap": {"library": "D3.js", "best_for": "Matrix data"},
                "network": {"library": "D3.js", "best_for": "Graph networks"},
                "timeline": {"library": "D3.js", "best_for": "Temporal sequences"},
                "sunburst": {"library": "D3.js", "best_for": "Multi-level hierarchies"}
            }
        }
    
    def _count_data_points(self, data: dict):
        """Count total data points in chart data."""
        count = 0
        if 'datasets' in data:
            for dataset in data.get('datasets', []):
                if 'data' in dataset:
                    count += len(dataset['data'])
        return count
    
    def get_performance_metrics(self):
        """Get chart performance metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.performance_metrics,
            "active_charts": len(self.active_charts),
            "total_data_points": sum(chart.get('data_points', 0) for chart in self.active_charts.values())
        }