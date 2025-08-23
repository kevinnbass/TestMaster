#!/usr/bin/env python3
"""
🏗️ MODULE: Chart Builders - Custom Chart Creation & Management
==================================================================

📋 PURPOSE:
    Chart creation and management functionality extracted from
    custom_visualization_builder.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Chart creation with various types and configurations
    • Chart updating and deletion
    • Panel creation for multiple charts
    • Chart listing and management operations

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract chart builders from custom_visualization_builder.py
   └─ Source: Lines 187-342 (155 lines)
   └─ Purpose: Separate chart creation logic into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: uuid, datetime, logging, dataclasses
📤 Provides: Chart creation and management methods
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from .viz_types import ChartType, DataSource, ChartConfiguration, VisualizationPanel

logger = logging.getLogger(__name__)


class ChartBuilder:
    """Chart creation and management functionality."""
    
    def __init__(self):
        self.charts: Dict[str, ChartConfiguration] = {}
        self.panels: Dict[str, VisualizationPanel] = {}
    
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
    
    def create_chart_from_template(self, template_name: str, templates: Dict[str, Dict[str, Any]], 
                                  title: str = None, **overrides) -> str:
        """
        Create a chart from a predefined template.
        
        Args:
            template_name: Name of the template to use
            templates: Available templates dictionary
            title: Optional custom title (defaults to template title)
            **overrides: Override template configuration
        
        Returns:
            Chart ID
        """
        if template_name not in templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = templates[template_name].copy()
        
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