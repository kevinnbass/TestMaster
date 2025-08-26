#!/usr/bin/env python3
"""
🏗️ MODULE: Data Formatters - Chart Data Processing & Formatting
==================================================================

📋 PURPOSE:
    Data processing and formatting functionality extracted from
    custom_visualization_builder.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Chart data generation and formatting
    • Chart.js compatible configuration building
    • Data source extraction and filtering
    • Format converters for different chart types

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract data formatters from custom_visualization_builder.py
   └─ Source: Lines 343-573 (230 lines)
   └─ Purpose: Separate data processing into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: ChartType, DataSource, ChartConfiguration
📤 Provides: Data formatting and processing functionality
"""

from typing import Dict, List, Any
from .viz_types import ChartType, DataSource, ChartConfiguration


class DataFormatter:
    """Data formatting and processing for chart visualization."""
    
    def generate_chart_data(self, chart: ChartConfiguration, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate chart data for visualization.
        
        Args:
            chart: Chart configuration
            analytics_data: Source data for visualization
        
        Returns:
            Chart.js compatible chart configuration
        """
        # Get data based on source
        source_data = self._get_source_data(chart.data_source, analytics_data)
        
        # Extract requested fields
        chart_data = self._extract_chart_data(chart, source_data)
        
        # Generate chart configuration
        chart_config = self._build_chart_config(chart, chart_data)
        
        return chart_config
    
    def _get_source_data(self, data_source: DataSource, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get data from the specified source."""
        source_map = {
            DataSource.PERSONAL_ANALYTICS: analytics_data,
            DataSource.PREDICTIVE_DATA: analytics_data.get('predictions', {}),
            DataSource.PERFORMANCE_METRICS: analytics_data.get('performance', {}),
            DataSource.QUALITY_METRICS: analytics_data.get('quality_metrics', {}),
            DataSource.PRODUCTIVITY_INSIGHTS: analytics_data.get('productivity_insights', {}),
            DataSource.CUSTOM_DATA: analytics_data.get('custom_data', {})
        }
        
        return source_map.get(data_source, {})
    
    def _extract_chart_data(self, chart: ChartConfiguration, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and transform data for chart visualization."""
        extracted_data = {}
        
        for field in chart.data_fields:
            # Support nested field access (e.g., "metrics.quality_score")
            field_value = source_data
            for key in field.split('.'):
                if isinstance(field_value, dict) and key in field_value:
                    field_value = field_value[key]
                else:
                    field_value = None
                    break
            
            extracted_data[field] = field_value
        
        # Apply filters if specified
        if chart.filters:
            extracted_data = self._apply_filters(extracted_data, chart.filters)
        
        return extracted_data
    
    def _apply_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to the data."""
        # Simple filter implementation
        filtered_data = data.copy()
        
        for filter_key, filter_value in filters.items():
            if filter_key in filtered_data:
                if isinstance(filter_value, dict):
                    # Range filter: {"min": 10, "max": 100}
                    if 'min' in filter_value:
                        filtered_data[filter_key] = max(filtered_data[filter_key], filter_value['min'])
                    if 'max' in filter_value:
                        filtered_data[filter_key] = min(filtered_data[filter_key], filter_value['max'])
        
        return filtered_data
    
    def _build_chart_config(self, chart: ChartConfiguration, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build Chart.js compatible configuration."""
        config = {
            "type": chart.chart_type.value,
            "data": self._format_data_for_chart_type(chart.chart_type, data),
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": chart.title
                    }
                }
            }
        }
        
        # Apply custom styling
        if chart.styling:
            self._apply_chart_styling(config, chart.styling)
        
        # Apply custom options
        if chart.options:
            self._merge_chart_options(config["options"], chart.options)
        
        return config
    
    def _format_data_for_chart_type(self, chart_type: ChartType, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data based on chart type."""
        if chart_type in [ChartType.LINE, ChartType.BAR, ChartType.AREA]:
            return self._format_line_bar_data(data)
        elif chart_type in [ChartType.PIE, ChartType.DOUGHNUT]:
            return self._format_pie_data(data)
        elif chart_type == ChartType.RADAR:
            return self._format_radar_data(data)
        elif chart_type == ChartType.GAUGE:
            return self._format_gauge_data(data)
        elif chart_type == ChartType.SCATTER:
            return self._format_scatter_data(data)
        else:
            # Default format
            return self._format_line_bar_data(data)
    
    def _format_line_bar_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for line/bar charts."""
        labels = list(data.keys())
        values = list(data.values())
        
        # Convert non-numeric values to 0
        numeric_values = []
        for val in values:
            if isinstance(val, (int, float)):
                numeric_values.append(val)
            elif isinstance(val, list) and len(val) > 0:
                numeric_values.append(len(val))  # Count for lists
            else:
                numeric_values.append(0)
        
        return {
            "labels": labels,
            "datasets": [{
                "label": "Data",
                "data": numeric_values,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)"
            }]
        }
    
    def _format_pie_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for pie/doughnut charts."""
        labels = list(data.keys())
        values = [val if isinstance(val, (int, float)) else 1 for val in data.values()]
        
        return {
            "labels": labels,
            "datasets": [{
                "data": values,
                "backgroundColor": [
                    "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0",
                    "#9966FF", "#FF9F40", "#C9CBCF", "#4BC0C0"
                ][:len(labels)]
            }]
        }
    
    def _format_radar_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for radar charts."""
        labels = list(data.keys())
        values = [val if isinstance(val, (int, float)) else 0 for val in data.values()]
        
        return {
            "labels": labels,
            "datasets": [{
                "label": "Metrics",
                "data": values,
                "borderColor": "rgb(54, 162, 235)",
                "backgroundColor": "rgba(54, 162, 235, 0.2)"
            }]
        }
    
    def _format_gauge_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for gauge charts."""
        # For gauge charts, we expect a single value
        value = list(data.values())[0] if data else 0
        if not isinstance(value, (int, float)):
            value = 0
        
        return {
            "value": value,
            "max": 100,
            "label": list(data.keys())[0] if data else "Value"
        }
    
    def _format_scatter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for scatter plots."""
        # Assume data has x and y fields
        points = []
        keys = list(data.keys())
        
        if len(keys) >= 2:
            x_values = data[keys[0]] if isinstance(data[keys[0]], list) else [data[keys[0]]]
            y_values = data[keys[1]] if isinstance(data[keys[1]], list) else [data[keys[1]]]
            
            for i, (x, y) in enumerate(zip(x_values, y_values)):
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    points.append({"x": x, "y": y})
        
        return {
            "datasets": [{
                "label": "Data Points",
                "data": points,
                "backgroundColor": "rgb(255, 99, 132)"
            }]
        }
    
    def _apply_chart_styling(self, config: Dict[str, Any], styling: Dict[str, Any]):
        """Apply custom styling to chart configuration."""
        if "colors" in styling and "datasets" in config["data"]:
            for i, dataset in enumerate(config["data"]["datasets"]):
                if i < len(styling["colors"]):
                    dataset["borderColor"] = styling["colors"][i]
                    dataset["backgroundColor"] = styling["colors"][i] + "33"  # Add alpha
        
        if "line_width" in styling and "datasets" in config["data"]:
            for dataset in config["data"]["datasets"]:
                dataset["borderWidth"] = styling["line_width"]
    
    def _merge_chart_options(self, base_options: Dict[str, Any], custom_options: Dict[str, Any]):
        """Merge custom options into base options."""
        for key, value in custom_options.items():
            if isinstance(value, dict) and key in base_options and isinstance(base_options[key], dict):
                self._merge_chart_options(base_options[key], value)
            else:
                base_options[key] = value