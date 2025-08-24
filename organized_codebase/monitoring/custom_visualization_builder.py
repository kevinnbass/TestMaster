"""
Custom Visualization Builder for Personal Analytics Dashboard
User-configurable chart and visualization creation system for Agent E

Author: Agent E - Latin Swarm
Created: 2025-08-23 23:00:00
Purpose: Enable users to create custom charts, graphs, and visualizations
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types for custom visualization."""
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    RADAR = "radar"
    SCATTER = "scatter"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    TIMELINE = "timeline"
    TREEMAP = "treemap"


class DataSource(Enum):
    """Available data sources for visualizations."""
    PERSONAL_ANALYTICS = "personal_analytics"
    PREDICTIVE_DATA = "predictive_data"
    PERFORMANCE_METRICS = "performance_metrics"
    QUALITY_METRICS = "quality_metrics"
    PRODUCTIVITY_INSIGHTS = "productivity_insights"
    CUSTOM_DATA = "custom_data"


@dataclass
class ChartConfiguration:
    """Configuration for a custom chart."""
    id: str
    title: str
    chart_type: ChartType
    data_source: DataSource
    data_fields: List[str]
    filters: Dict[str, Any] = None
    styling: Dict[str, Any] = None
    options: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class VisualizationPanel:
    """A custom visualization panel containing multiple charts."""
    id: str
    title: str
    description: str
    charts: List[str]  # Chart IDs
    layout: Dict[str, Any]
    position: Dict[str, int]
    size: Dict[str, int]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class CustomVisualizationBuilder:
    """
    Custom visualization builder for creating user-defined charts and panels.
    
    Features:
    - Multiple chart types (line, bar, pie, radar, etc.)
    - Configurable data sources and fields
    - Custom styling and options
    - Panel management for multiple charts
    - Template library for common visualizations
    - Export capabilities
    """
    
    def __init__(self):
        """Initialize the custom visualization builder."""
        self.charts: Dict[str, ChartConfiguration] = {}
        self.panels: Dict[str, VisualizationPanel] = {}
        self.templates: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with default templates
        self._initialize_default_templates()
        
        logger.info("Custom Visualization Builder initialized")
    
    def _initialize_default_templates(self):
        """Initialize default chart templates."""
        self.templates = {
            "productivity_trend": {
                "title": "Productivity Trend",
                "chart_type": ChartType.LINE,
                "data_source": DataSource.PRODUCTIVITY_INSIGHTS,
                "data_fields": ["commits_per_day", "lines_per_day", "files_modified"],
                "styling": {
                    "colors": ["#4CAF50", "#2196F3", "#FF9800"],
                    "line_width": 3,
                    "point_radius": 4
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "y": {"beginAtZero": True}
                    }
                }
            },
            "code_quality_radar": {
                "title": "Code Quality Overview",
                "chart_type": ChartType.RADAR,
                "data_source": DataSource.QUALITY_METRICS,
                "data_fields": ["maintainability", "complexity", "coverage", "documentation"],
                "styling": {
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgb(54, 162, 235)",
                    "pointBackgroundColor": "rgb(54, 162, 235)"
                },
                "options": {
                    "scales": {
                        "r": {
                            "beginAtZero": True,
                            "max": 100
                        }
                    }
                }
            },
            "performance_gauge": {
                "title": "System Performance",
                "chart_type": ChartType.GAUGE,
                "data_source": DataSource.PERFORMANCE_METRICS,
                "data_fields": ["overall_performance_score"],
                "styling": {
                    "colors": {
                        "good": "#4CAF50",
                        "warning": "#FF9800",
                        "critical": "#F44336"
                    }
                },
                "options": {
                    "min": 0,
                    "max": 100,
                    "thresholds": [60, 80, 100]
                }
            },
            "commit_activity_heatmap": {
                "title": "Commit Activity Heatmap",
                "chart_type": ChartType.HEATMAP,
                "data_source": DataSource.PERSONAL_ANALYTICS,
                "data_fields": ["commit_timestamps"],
                "styling": {
                    "colorScale": ["#c6e48b", "#7bc96f", "#49af5d", "#2d8840", "#196127"]
                },
                "options": {
                    "weekStart": 1,  # Monday
                    "tooltip": {
                        "titleFormat": "MMM D, YYYY",
                        "bodyFormat": "{value} commits"
                    }
                }
            }
        }
    
    def create_chart(self, title: str, chart_type: ChartType, data_source: DataSource,
                    data_fields: List[str], **kwargs) -> str:
        """
        Create a new custom chart.
        
        Args:
            title: Chart title
            chart_type: Type of chart to create
            data_source: Data source for the chart
            data_fields: Fields to use from the data source
            **kwargs: Additional configuration (filters, styling, options)
        
        Returns:
            Chart ID
        """
        chart_id = str(uuid.uuid4())
        
        chart_config = ChartConfiguration(
            id=chart_id,
            title=title,
            chart_type=chart_type,
            data_source=data_source,
            data_fields=data_fields,
            filters=kwargs.get('filters', {}),
            styling=kwargs.get('styling', {}),
            options=kwargs.get('options', {})
        )
        
        self.charts[chart_id] = chart_config
        
        logger.info(f"Created custom chart '{title}' with ID: {chart_id}")
        return chart_id
    
    def create_chart_from_template(self, template_name: str, title: str = None, **overrides) -> str:
        """
        Create a chart from a predefined template.
        
        Args:
            template_name: Name of the template to use
            title: Optional custom title (defaults to template title)
            **overrides: Override template configuration
        
        Returns:
            Chart ID
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name].copy()
        
        # Apply overrides
        for key, value in overrides.items():
            template[key] = value
        
        chart_title = title or template['title']
        
        return self.create_chart(
            title=chart_title,
            chart_type=ChartType(template['chart_type']),
            data_source=DataSource(template['data_source']),
            data_fields=template['data_fields'],
            styling=template.get('styling', {}),
            options=template.get('options', {}),
            filters=template.get('filters', {})
        )
    
    def update_chart(self, chart_id: str, **updates) -> bool:
        """
        Update an existing chart configuration.
        
        Args:
            chart_id: ID of chart to update
            **updates: Configuration updates
        
        Returns:
            True if updated successfully
        """
        if chart_id not in self.charts:
            logger.error(f"Chart {chart_id} not found for update")
            return False
        
        chart = self.charts[chart_id]
        
        # Update allowed fields
        updatable_fields = ['title', 'data_fields', 'filters', 'styling', 'options']
        for field in updatable_fields:
            if field in updates:
                setattr(chart, field, updates[field])
        
        chart.updated_at = datetime.now()
        
        logger.info(f"Updated chart {chart_id}")
        return True
    
    def delete_chart(self, chart_id: str) -> bool:
        """
        Delete a chart.
        
        Args:
            chart_id: ID of chart to delete
        
        Returns:
            True if deleted successfully
        """
        if chart_id not in self.charts:
            return False
        
        # Remove from any panels using this chart
        for panel in self.panels.values():
            if chart_id in panel.charts:
                panel.charts.remove(chart_id)
                panel.updated_at = datetime.now()
        
        del self.charts[chart_id]
        logger.info(f"Deleted chart {chart_id}")
        return True
    
    def create_panel(self, title: str, description: str, chart_ids: List[str],
                    layout: Dict[str, Any], position: Dict[str, int] = None,
                    size: Dict[str, int] = None) -> str:
        """
        Create a visualization panel with multiple charts.
        
        Args:
            title: Panel title
            description: Panel description
            chart_ids: List of chart IDs to include
            layout: Layout configuration for charts
            position: Panel position on dashboard
            size: Panel size
        
        Returns:
            Panel ID
        """
        panel_id = str(uuid.uuid4())
        
        # Validate chart IDs
        invalid_charts = [cid for cid in chart_ids if cid not in self.charts]
        if invalid_charts:
            raise ValueError(f"Invalid chart IDs: {invalid_charts}")
        
        panel = VisualizationPanel(
            id=panel_id,
            title=title,
            description=description,
            charts=chart_ids,
            layout=layout,
            position=position or {"x": 0, "y": 0},
            size=size or {"width": 2, "height": 2}
        )
        
        self.panels[panel_id] = panel
        
        logger.info(f"Created visualization panel '{title}' with ID: {panel_id}")
        return panel_id
    
    def generate_chart_data(self, chart_id: str, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate chart data for visualization.
        
        Args:
            chart_id: Chart ID
            analytics_data: Source data for visualization
        
        Returns:
            Chart.js compatible chart configuration
        """
        if chart_id not in self.charts:
            raise ValueError(f"Chart {chart_id} not found")
        
        chart = self.charts[chart_id]
        
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
    
    def get_chart_list(self) -> List[Dict[str, Any]]:
        """Get list of all created charts."""
        return [
            {
                "id": chart.id,
                "title": chart.title,
                "chart_type": chart.chart_type.value,
                "data_source": chart.data_source.value,
                "created_at": chart.created_at.isoformat(),
                "updated_at": chart.updated_at.isoformat()
            }
            for chart in self.charts.values()
        ]
    
    def get_panel_list(self) -> List[Dict[str, Any]]:
        """Get list of all created panels."""
        return [
            {
                "id": panel.id,
                "title": panel.title,
                "description": panel.description,
                "chart_count": len(panel.charts),
                "created_at": panel.created_at.isoformat(),
                "updated_at": panel.updated_at.isoformat()
            }
            for panel in self.panels.values()
        ]
    
    def get_template_list(self) -> List[Dict[str, Any]]:
        """Get list of available templates."""
        return [
            {
                "name": name,
                "title": template["title"],
                "chart_type": template["chart_type"],
                "data_source": template["data_source"],
                "description": f"Template for {template['title'].lower()}"
            }
            for name, template in self.templates.items()
        ]
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export all charts and panels configuration."""
        return {
            "charts": {
                chart_id: asdict(chart) for chart_id, chart in self.charts.items()
            },
            "panels": {
                panel_id: asdict(panel) for panel_id, panel in self.panels.items()
            },
            "exported_at": datetime.now().isoformat()
        }
    
    def import_configuration(self, config: Dict[str, Any]) -> bool:
        """Import charts and panels configuration."""
        try:
            # Import charts
            if "charts" in config:
                for chart_id, chart_data in config["charts"].items():
                    chart_data["chart_type"] = ChartType(chart_data["chart_type"])
                    chart_data["data_source"] = DataSource(chart_data["data_source"])
                    
                    # Handle datetime fields
                    if isinstance(chart_data["created_at"], str):
                        chart_data["created_at"] = datetime.fromisoformat(chart_data["created_at"])
                    if isinstance(chart_data["updated_at"], str):
                        chart_data["updated_at"] = datetime.fromisoformat(chart_data["updated_at"])
                    
                    self.charts[chart_id] = ChartConfiguration(**chart_data)
            
            # Import panels
            if "panels" in config:
                for panel_id, panel_data in config["panels"].items():
                    # Handle datetime fields
                    if isinstance(panel_data["created_at"], str):
                        panel_data["created_at"] = datetime.fromisoformat(panel_data["created_at"])
                    if isinstance(panel_data["updated_at"], str):
                        panel_data["updated_at"] = datetime.fromisoformat(panel_data["updated_at"])
                    
                    self.panels[panel_id] = VisualizationPanel(**panel_data)
            
            logger.info("Successfully imported visualization configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False


def create_custom_visualization_builder() -> CustomVisualizationBuilder:
    """Factory function to create a configured custom visualization builder."""
    builder = CustomVisualizationBuilder()
    
    logger.info("Custom visualization builder created with default templates")
    return builder


if __name__ == '__main__':
    # Demo usage
    builder = create_custom_visualization_builder()
    
    # Create a chart from template
    chart_id = builder.create_chart_from_template("productivity_trend", "My Productivity")
    print(f"Created chart: {chart_id}")
    
    # Create a custom chart
    custom_chart = builder.create_chart(
        title="Code Quality Metrics",
        chart_type=ChartType.BAR,
        data_source=DataSource.QUALITY_METRICS,
        data_fields=["maintainability", "complexity", "coverage"],
        styling={"colors": ["#4CAF50", "#FFC107", "#2196F3"]}
    )
    print(f"Created custom chart: {custom_chart}")
    
    # Demo data generation
    sample_data = {
        "quality_metrics": {
            "maintainability": 85,
            "complexity": 75,
            "coverage": 90
        }
    }
    
    chart_config = builder.generate_chart_data(custom_chart, sample_data)
    print("Generated chart configuration:")
    print(json.dumps(chart_config, indent=2))
    
    # List templates
    templates = builder.get_template_list()
    print(f"\nAvailable templates: {len(templates)}")
    for template in templates:
        print(f"  - {template['name']}: {template['title']}")