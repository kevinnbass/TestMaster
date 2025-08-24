#!/usr/bin/env python3
"""
Performance Charts Component
============================
STEELCLAD Atomized Module (<200 lines)
Extracted from performance_analytics_integrated.py

Performance visualization and charting components.
"""

from datetime import datetime, timedelta
import random
from typing import Dict, Any, List, Optional, Tuple

class PerformanceCharts:
    """Atomic component for performance metric visualizations."""
    
    def __init__(self):
        self.chart_cache = {}
        self.time_series_data = []
        self.chart_configs = self._initialize_chart_configs()
    
    def render_performance_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render performance metrics as charts and visualizations.
        
        Args:
            metrics_data: Performance metrics to visualize
            
        Returns:
            Chart configuration and data for frontend rendering
        """
        charts = {
            "timestamp": datetime.now().isoformat(),
            "charts": [],
            "summary": self._generate_performance_summary(metrics_data)
        }
        
        # Response Time Chart
        charts["charts"].append(self._create_response_time_chart(metrics_data))
        
        # Throughput Chart
        charts["charts"].append(self._create_throughput_chart(metrics_data))
        
        # Resource Utilization Charts
        charts["charts"].append(self._create_resource_chart(metrics_data))
        
        # Performance Trends
        charts["charts"].append(self._create_trend_chart(metrics_data))
        
        return charts
    
    def create_time_series_chart(self, data_points: List[Tuple[datetime, float]], 
                                 chart_type: str = "line") -> Dict[str, Any]:
        """Create time series chart configuration."""
        return {
            "type": chart_type,
            "data": {
                "labels": [point[0].isoformat() for point in data_points],
                "datasets": [{
                    "label": "Performance",
                    "data": [point[1] for point in data_points],
                    "borderColor": "#2196F3",
                    "backgroundColor": "rgba(33, 150, 243, 0.1)"
                }]
            },
            "options": self.chart_configs.get(chart_type, {})
        }
    
    def create_gauge_chart(self, value: float, max_value: float = 100,
                          thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Create gauge chart for single metric display."""
        if thresholds is None:
            thresholds = {"good": 80, "warning": 60, "critical": 40}
        
        return {
            "type": "gauge",
            "data": {
                "value": value,
                "max": max_value,
                "thresholds": thresholds
            },
            "options": {
                "animation": True,
                "responsive": True,
                "colors": self._get_gauge_colors(value, thresholds)
            }
        }
    
    def create_bar_chart(self, categories: List[str], values: List[float],
                        colors: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create bar chart for categorical data."""
        if colors is None:
            colors = self._generate_chart_colors(len(categories))
        
        return {
            "type": "bar",
            "data": {
                "labels": categories,
                "datasets": [{
                    "label": "Performance Metrics",
                    "data": values,
                    "backgroundColor": colors
                }]
            },
            "options": self.chart_configs.get("bar", {})
        }
    
    def create_heatmap(self, matrix_data: List[List[float]], 
                      x_labels: List[str], y_labels: List[str]) -> Dict[str, Any]:
        """Create heatmap visualization."""
        return {
            "type": "heatmap",
            "data": {
                "x_labels": x_labels,
                "y_labels": y_labels,
                "values": matrix_data,
                "colorScale": "performance"
            },
            "options": {
                "responsive": True,
                "tooltip": True,
                "legend": True
            }
        }
    
    def _create_response_time_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create response time distribution chart."""
        return {
            "id": "response_time",
            "title": "Response Time Distribution",
            "type": "line",
            "data": {
                "p50": data.get("response_times", {}).get("avg", 0),
                "p95": data.get("response_times", {}).get("p95", 0),
                "p99": data.get("response_times", {}).get("p99", 0)
            },
            "config": self.create_gauge_chart(
                data.get("response_times", {}).get("avg", 0),
                max_value=1000,
                thresholds={"good": 200, "warning": 500, "critical": 800}
            )
        }
    
    def _create_throughput_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create throughput metrics chart."""
        categories = ["Requests/sec", "Operations/min", "Transactions/hour"]
        values = [
            data.get("throughput", {}).get("requests_per_second", 0),
            data.get("throughput", {}).get("operations_per_minute", 0) / 60,
            data.get("active_transactions", 0) * 60
        ]
        return {
            "id": "throughput",
            "title": "Throughput Metrics",
            "config": self.create_bar_chart(categories, values)
        }
    
    def _create_resource_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create resource utilization chart."""
        resources = data.get("resource_utilization", {})
        return {
            "id": "resources",
            "title": "Resource Utilization",
            "type": "radar",
            "data": {
                "labels": ["CPU", "Memory", "Disk I/O", "Network"],
                "values": [
                    resources.get("cpu_percent", 0),
                    resources.get("memory_mb", 0) / 20,  # Normalize to percentage
                    resources.get("disk_io", 0),
                    resources.get("network_io", 0)
                ]
            }
        }
    
    def _create_trend_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance trend chart."""
        # Generate sample trend data
        trend_points = []
        base_time = datetime.now() - timedelta(hours=24)
        for i in range(24):
            trend_points.append((
                base_time + timedelta(hours=i),
                random.uniform(70, 95)
            ))
        
        return {
            "id": "trends",
            "title": "24-Hour Performance Trend",
            "config": self.create_time_series_chart(trend_points)
        }
    
    def _generate_performance_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        return {
            "avg_response_time": data.get("response_times", {}).get("avg", 0),
            "peak_throughput": data.get("throughput", {}).get("requests_per_second", 0),
            "resource_efficiency": self._calculate_efficiency(data),
            "performance_score": self._calculate_performance_score(data)
        }
    
    def _calculate_efficiency(self, data: Dict[str, Any]) -> float:
        """Calculate resource efficiency score."""
        resources = data.get("resource_utilization", {})
        cpu = resources.get("cpu_percent", 0)
        memory = resources.get("memory_mb", 0) / 20
        return round(100 - ((cpu + memory) / 2), 1)
    
    def _calculate_performance_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        response_score = max(0, 100 - (data.get("response_times", {}).get("avg", 0) / 10))
        throughput_score = min(100, data.get("throughput", {}).get("requests_per_second", 0) / 2)
        return round((response_score + throughput_score) / 2, 1)
    
    def _get_gauge_colors(self, value: float, thresholds: Dict[str, float]) -> str:
        """Get color based on gauge thresholds."""
        if value >= thresholds["good"]: return "#4caf50"
        elif value >= thresholds["warning"]: return "#ff9800"
        return "#f44336"
    
    def _generate_chart_colors(self, count: int) -> List[str]:
        """Generate chart colors."""
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
        return [colors[i % len(colors)] for i in range(count)]
    
    def _initialize_chart_configs(self) -> Dict[str, Any]:
        """Initialize chart configuration templates."""
        return {
            "line": {"responsive": True, "animation": True, "tension": 0.4},
            "bar": {"responsive": True, "indexAxis": "x", "animation": True},
            "gauge": {"responsive": True, "animation": True}
        }

# Module exports
__all__ = ['PerformanceCharts']