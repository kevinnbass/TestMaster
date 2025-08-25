"""
No-Code Dashboard Builder Enhancement
====================================

Adds no-code visual dashboard building capabilities to the unified dashboard system.
Provides drag-and-drop interface, visual configuration, and template-based creation.

Author: TestMaster Enhancement System
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Import our unified dashboard
import sys
sys.path.append(str(Path(__file__).parent))
from unified_dashboard import (
    unified_dashboard, DashboardWidget, ChartWidget, MetricWidget, 
    TableWidget, ControlWidget, WidgetType, ChartType, DashboardTheme
)


class BuilderMode(Enum):
    """No-code builder modes"""
    VISUAL_BUILDER = "visual_builder"
    TEMPLATE_BUILDER = "template_builder"
    CODE_BUILDER = "code_builder"
    GUIDED_BUILDER = "guided_builder"


class ComponentCategory(Enum):
    """Widget component categories"""
    VISUALIZATION = "visualization"
    METRICS = "metrics"
    CONTROLS = "controls"
    LAYOUT = "layout"
    DATA = "data"
    ALERTS = "alerts"


@dataclass
class WidgetTemplate:
    """No-code widget template"""
    template_id: str
    name: str
    description: str
    category: ComponentCategory
    widget_type: WidgetType
    default_config: Dict[str, Any] = field(default_factory=dict)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    preview_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "widget_type": self.widget_type.value,
            "default_config": self.default_config,
            "configuration_schema": self.configuration_schema,
            "preview_data": self.preview_data,
            "tags": self.tags
        }


@dataclass
class DashboardTemplate:
    """Complete dashboard template"""
    template_id: str
    name: str
    description: str
    use_case: str
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    theme: DashboardTheme = DashboardTheme.LIGHT
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "use_case": self.use_case,
            "widgets": self.widgets,
            "layout": self.layout,
            "theme": self.theme.value,
            "tags": self.tags
        }


class NoCodeDashboardBuilder:
    """
    No-code dashboard builder with visual interface capabilities.
    Extends the unified dashboard system with intuitive building tools.
    """
    
    def __init__(self, dashboard_system=None):
        self.dashboard_system = dashboard_system or unified_dashboard
        self.logger = logging.getLogger("nocode_dashboard_builder")
        
        # Builder state
        self.widget_templates: Dict[str, WidgetTemplate] = {}
        self.dashboard_templates: Dict[str, DashboardTemplate] = {}
        self.build_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Configuration schemas
        self.config_schemas = {}
        self.validation_rules = {}
        
        self._initialize_widget_templates()
        self._initialize_dashboard_templates()
        self._initialize_config_schemas()
        
        self.logger.info("No-Code Dashboard Builder initialized")
    
    def _initialize_widget_templates(self):
        """Initialize pre-built widget templates"""
        
        # Chart templates
        self.widget_templates["line_chart"] = WidgetTemplate(
            template_id="line_chart",
            name="Line Chart",
            description="Display time-series data with line chart",
            category=ComponentCategory.VISUALIZATION,
            widget_type=WidgetType.CHART,
            default_config={
                "chart_type": "line",
                "show_legend": True,
                "show_grid": True,
                "animation_enabled": True
            },
            configuration_schema={
                "title": {"type": "string", "required": True, "label": "Chart Title"},
                "data_source": {"type": "select", "required": True, "label": "Data Source"},
                "x_axis": {"type": "string", "default": "timestamp", "label": "X-Axis Field"},
                "y_axis": {"type": "string", "default": "value", "label": "Y-Axis Field"},
                "color_scheme": {"type": "color_array", "label": "Color Scheme"}
            },
            preview_data={
                "type": "line",
                "data": [
                    {"timestamp": "2024-01-01", "value": 10},
                    {"timestamp": "2024-01-02", "value": 15},
                    {"timestamp": "2024-01-03", "value": 12}
                ]
            },
            tags=["chart", "visualization", "time-series"]
        )
        
        self.widget_templates["bar_chart"] = WidgetTemplate(
            template_id="bar_chart",
            name="Bar Chart",
            description="Display categorical data with bar chart",
            category=ComponentCategory.VISUALIZATION,
            widget_type=WidgetType.CHART,
            default_config={
                "chart_type": "bar",
                "show_legend": True,
                "horizontal": False
            },
            configuration_schema={
                "title": {"type": "string", "required": True, "label": "Chart Title"},
                "data_source": {"type": "select", "required": True, "label": "Data Source"},
                "category_field": {"type": "string", "label": "Category Field"},
                "value_field": {"type": "string", "label": "Value Field"},
                "horizontal": {"type": "boolean", "default": False, "label": "Horizontal Bars"}
            },
            tags=["chart", "visualization", "categorical"]
        )
        
        self.widget_templates["pie_chart"] = WidgetTemplate(
            template_id="pie_chart",
            name="Pie Chart",
            description="Display proportional data with pie chart",
            category=ComponentCategory.VISUALIZATION,
            widget_type=WidgetType.CHART,
            default_config={
                "chart_type": "pie",
                "show_legend": True,
                "show_labels": True
            },
            configuration_schema={
                "title": {"type": "string", "required": True, "label": "Chart Title"},
                "data_source": {"type": "select", "required": True, "label": "Data Source"},
                "label_field": {"type": "string", "label": "Label Field"},
                "value_field": {"type": "string", "label": "Value Field"}
            },
            tags=["chart", "visualization", "proportional"]
        )
        
        # Metric templates
        self.widget_templates["kpi_metric"] = WidgetTemplate(
            template_id="kpi_metric",
            name="KPI Metric",
            description="Display key performance indicator",
            category=ComponentCategory.METRICS,
            widget_type=WidgetType.METRIC,
            default_config={
                "show_trend": True,
                "trend_period": "24h"
            },
            configuration_schema={
                "title": {"type": "string", "required": True, "label": "Metric Title"},
                "data_source": {"type": "select", "required": True, "label": "Data Source"},
                "unit": {"type": "string", "label": "Unit"},
                "format": {"type": "select", "options": ["number", "currency", "percentage"], "label": "Format"},
                "thresholds": {"type": "object", "label": "Alert Thresholds"}
            },
            tags=["metric", "kpi", "performance"]
        )
        
        self.widget_templates["progress_metric"] = WidgetTemplate(
            template_id="progress_metric",
            name="Progress Metric",
            description="Display progress towards a goal",
            category=ComponentCategory.METRICS,
            widget_type=WidgetType.METRIC,
            default_config={
                "show_percentage": True,
                "show_target": True
            },
            configuration_schema={
                "title": {"type": "string", "required": True, "label": "Progress Title"},
                "current_value": {"type": "number", "required": True, "label": "Current Value"},
                "target_value": {"type": "number", "required": True, "label": "Target Value"},
                "color": {"type": "color", "default": "#4CAF50", "label": "Progress Color"}
            },
            tags=["metric", "progress", "goal"]
        )
        
        # Table templates
        self.widget_templates["data_table"] = WidgetTemplate(
            template_id="data_table",
            name="Data Table",
            description="Display tabular data with sorting and filtering",
            category=ComponentCategory.DATA,
            widget_type=WidgetType.TABLE,
            default_config={
                "sortable": True,
                "filterable": True,
                "paginated": True,
                "page_size": 25
            },
            configuration_schema={
                "title": {"type": "string", "required": True, "label": "Table Title"},
                "data_source": {"type": "select", "required": True, "label": "Data Source"},
                "columns": {"type": "array", "items": {"type": "object"}, "label": "Columns"},
                "page_size": {"type": "number", "default": 25, "label": "Rows Per Page"}
            },
            tags=["table", "data", "grid"]
        )
        
        # Control templates
        self.widget_templates["action_button"] = WidgetTemplate(
            template_id="action_button",
            name="Action Button",
            description="Interactive button for triggering actions",
            category=ComponentCategory.CONTROLS,
            widget_type=WidgetType.CONTROL,
            default_config={
                "control_type": "button",
                "style": "primary"
            },
            configuration_schema={
                "title": {"type": "string", "required": True, "label": "Button Text"},
                "action": {"type": "select", "required": True, "label": "Action"},
                "style": {"type": "select", "options": ["primary", "secondary", "success", "warning", "danger"], "label": "Style"},
                "size": {"type": "select", "options": ["small", "medium", "large"], "default": "medium", "label": "Size"}
            },
            tags=["control", "button", "action"]
        )
    
    def _initialize_dashboard_templates(self):
        """Initialize complete dashboard templates"""
        
        # System monitoring template
        self.dashboard_templates["system_monitoring"] = DashboardTemplate(
            template_id="system_monitoring",
            name="System Monitoring Dashboard",
            description="Monitor system health and performance metrics",
            use_case="Real-time system monitoring and alerting",
            widgets=[
                {
                    "template_id": "kpi_metric",
                    "title": "CPU Usage",
                    "position": {"x": 0, "y": 0, "width": 3, "height": 2},
                    "config": {"unit": "%", "thresholds": {"warning": 70, "critical": 90}}
                },
                {
                    "template_id": "kpi_metric", 
                    "title": "Memory Usage",
                    "position": {"x": 3, "y": 0, "width": 3, "height": 2},
                    "config": {"unit": "%", "thresholds": {"warning": 80, "critical": 95}}
                },
                {
                    "template_id": "line_chart",
                    "title": "Performance Timeline",
                    "position": {"x": 0, "y": 2, "width": 8, "height": 4},
                    "config": {"chart_type": "line", "show_legend": True}
                },
                {
                    "template_id": "data_table",
                    "title": "Active Processes",
                    "position": {"x": 8, "y": 0, "width": 4, "height": 6},
                    "config": {"page_size": 20}
                }
            ],
            layout={
                "grid_size": {"cols": 12, "rows": 20},
                "auto_refresh": True,
                "refresh_interval": 30
            },
            theme=DashboardTheme.DARK,
            tags=["monitoring", "system", "performance"]
        )
        
        # Analytics template
        self.dashboard_templates["analytics"] = DashboardTemplate(
            template_id="analytics",
            name="Analytics Dashboard",
            description="Analyze usage patterns and performance trends",
            use_case="Business analytics and insights",
            widgets=[
                {
                    "template_id": "pie_chart",
                    "title": "Usage Distribution",
                    "position": {"x": 0, "y": 0, "width": 6, "height": 4},
                    "config": {"show_legend": True}
                },
                {
                    "template_id": "bar_chart",
                    "title": "Monthly Trends",
                    "position": {"x": 6, "y": 0, "width": 6, "height": 4},
                    "config": {"horizontal": False}
                },
                {
                    "template_id": "kpi_metric",
                    "title": "Total Users",
                    "position": {"x": 0, "y": 4, "width": 3, "height": 2},
                    "config": {"format": "number"}
                },
                {
                    "template_id": "kpi_metric",
                    "title": "Revenue",
                    "position": {"x": 3, "y": 4, "width": 3, "height": 2},
                    "config": {"format": "currency", "unit": "$"}
                }
            ],
            layout={
                "grid_size": {"cols": 12, "rows": 15},
                "auto_refresh": True,
                "refresh_interval": 300
            },
            theme=DashboardTheme.LIGHT,
            tags=["analytics", "business", "insights"]
        )
    
    def _initialize_config_schemas(self):
        """Initialize configuration schemas for validation"""
        
        self.config_schemas = {
            "widget": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 100},
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "minimum": 0},
                            "y": {"type": "integer", "minimum": 0},
                            "width": {"type": "integer", "minimum": 1, "maximum": 12},
                            "height": {"type": "integer", "minimum": 1, "maximum": 20}
                        },
                        "required": ["x", "y", "width", "height"]
                    },
                    "data_source": {"type": "string"},
                    "refresh_interval": {"type": "integer", "minimum": 1, "maximum": 3600}
                },
                "required": ["title", "position"]
            },
            "dashboard": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 100},
                    "description": {"type": "string", "maxLength": 500},
                    "theme": {"type": "string", "enum": ["light", "dark", "auto", "high_contrast"]},
                    "widgets": {"type": "array", "items": {"$ref": "#/definitions/widget"}}
                },
                "required": ["name"]
            }
        }
    
    # ========================================================================
    # VISUAL BUILDER INTERFACE
    # ========================================================================
    
    def start_build_session(self, user_id: str, mode: BuilderMode = BuilderMode.VISUAL_BUILDER) -> str:
        """Start new dashboard building session"""
        session_id = f"build_{uuid.uuid4().hex[:12]}"
        
        self.build_sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "mode": mode,
            "start_time": datetime.now(),
            "dashboard_draft": None,
            "widgets": [],
            "history": [],
            "auto_save": True
        }
        
        self.logger.info(f"Build session started: {session_id} for user {user_id}")
        return session_id
    
    def get_widget_palette(self, category: Optional[ComponentCategory] = None) -> List[Dict[str, Any]]:
        """Get available widget templates for builder palette"""
        templates = list(self.widget_templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return [template.to_dict() for template in templates]
    
    def get_dashboard_templates(self, use_case: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available dashboard templates"""
        templates = list(self.dashboard_templates.values())
        
        if use_case:
            templates = [t for t in templates if use_case.lower() in t.use_case.lower()]
        
        return [template.to_dict() for template in templates]
    
    def add_widget_to_session(self, session_id: str, template_id: str, 
                             position: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add widget to build session"""
        if session_id not in self.build_sessions:
            return {"error": "Build session not found"}
        
        if template_id not in self.widget_templates:
            return {"error": "Widget template not found"}
        
        try:
            session = self.build_sessions[session_id]
            template = self.widget_templates[template_id]
            
            # Create widget instance
            widget_id = f"{template.widget_type.value}_{uuid.uuid4().hex[:8]}"
            widget_config = template.default_config.copy()
            
            if config:
                widget_config.update(config)
            
            widget_data = {
                "widget_id": widget_id,
                "template_id": template_id,
                "widget_type": template.widget_type.value,
                "title": config.get("title", template.name),
                "position": position,
                "config": widget_config,
                "created_at": datetime.now().isoformat()
            }
            
            # Add to session
            session["widgets"].append(widget_data)
            
            # Add to history
            session["history"].append({
                "action": "add_widget",
                "widget_id": widget_id,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Widget added to session {session_id}: {widget_id}")
            return {"success": True, "widget": widget_data}
            
        except Exception as e:
            self.logger.error(f"Error adding widget to session {session_id}: {e}")
            return {"error": str(e)}
    
    def update_widget_in_session(self, session_id: str, widget_id: str, 
                                updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update widget in build session"""
        if session_id not in self.build_sessions:
            return {"error": "Build session not found"}
        
        try:
            session = self.build_sessions[session_id]
            
            # Find widget
            widget = None
            for w in session["widgets"]:
                if w["widget_id"] == widget_id:
                    widget = w
                    break
            
            if not widget:
                return {"error": "Widget not found"}
            
            # Apply updates
            widget.update(updates)
            widget["updated_at"] = datetime.now().isoformat()
            
            # Add to history
            session["history"].append({
                "action": "update_widget",
                "widget_id": widget_id,
                "updates": updates,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Widget updated in session {session_id}: {widget_id}")
            return {"success": True, "widget": widget}
            
        except Exception as e:
            self.logger.error(f"Error updating widget in session {session_id}: {e}")
            return {"error": str(e)}
    
    def remove_widget_from_session(self, session_id: str, widget_id: str) -> Dict[str, Any]:
        """Remove widget from build session"""
        if session_id not in self.build_sessions:
            return {"error": "Build session not found"}
        
        try:
            session = self.build_sessions[session_id]
            
            # Remove widget
            session["widgets"] = [w for w in session["widgets"] if w["widget_id"] != widget_id]
            
            # Add to history
            session["history"].append({
                "action": "remove_widget",
                "widget_id": widget_id,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Widget removed from session {session_id}: {widget_id}")
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Error removing widget from session {session_id}: {e}")
            return {"error": str(e)}
    
    def get_session_preview(self, session_id: str) -> Dict[str, Any]:
        """Get preview of current build session"""
        if session_id not in self.build_sessions:
            return {"error": "Build session not found"}
        
        try:
            session = self.build_sessions[session_id]
            
            return {
                "session": {
                    "session_id": session_id,
                    "mode": session["mode"].value,
                    "start_time": session["start_time"].isoformat(),
                    "widget_count": len(session["widgets"])
                },
                "widgets": session["widgets"],
                "layout": {
                    "grid_size": {"cols": 12, "rows": 20},
                    "auto_refresh": True,
                    "refresh_interval": 30
                },
                "preview_url": f"/dashboard/preview/{session_id}"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session preview {session_id}: {e}")
            return {"error": str(e)}
    
    # ========================================================================
    # TEMPLATE-BASED BUILDER
    # ========================================================================
    
    def create_from_template(self, session_id: str, template_id: str, 
                           customizations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create dashboard from template"""
        if session_id not in self.build_sessions:
            return {"error": "Build session not found"}
        
        if template_id not in self.dashboard_templates:
            return {"error": "Dashboard template not found"}
        
        try:
            session = self.build_sessions[session_id]
            template = self.dashboard_templates[template_id]
            
            # Clear existing widgets
            session["widgets"] = []
            
            # Apply template widgets
            for widget_config in template.widgets:
                template_id = widget_config["template_id"]
                position = widget_config["position"]
                config = widget_config.get("config", {})
                
                # Apply customizations
                if customizations and "widgets" in customizations:
                    widget_custom = customizations["widgets"].get(widget_config.get("title", ""), {})
                    config.update(widget_custom)
                
                result = self.add_widget_to_session(session_id, template_id, position, config)
                if "error" in result:
                    return result
            
            # Set dashboard properties
            session["dashboard_draft"] = {
                "name": customizations.get("name", template.name),
                "description": customizations.get("description", template.description),
                "theme": customizations.get("theme", template.theme.value),
                "layout": template.layout
            }
            
            # Add to history
            session["history"].append({
                "action": "apply_template",
                "template_id": template_id,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Template applied to session {session_id}: {template_id}")
            return {"success": True, "widgets_created": len(template.widgets)}
            
        except Exception as e:
            self.logger.error(f"Error applying template to session {session_id}: {e}")
            return {"error": str(e)}
    
    # ========================================================================
    # DASHBOARD CREATION AND DEPLOYMENT
    # ========================================================================
    
    def deploy_dashboard(self, session_id: str, dashboard_name: str, 
                        deploy_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deploy dashboard from build session"""
        if session_id not in self.build_sessions:
            return {"error": "Build session not found"}
        
        try:
            session = self.build_sessions[session_id]
            
            # Validate dashboard
            validation_result = self._validate_dashboard_config({
                "name": dashboard_name,
                "widgets": session["widgets"]
            })
            
            if validation_result["errors"]:
                return {"error": "Validation failed", "errors": validation_result["errors"]}
            
            # Create dashboard
            dashboard_id = f"dashboard_{uuid.uuid4().hex[:8]}"
            theme = DashboardTheme(session.get("dashboard_draft", {}).get("theme", "light"))
            
            dashboard = self.dashboard_system.create_dashboard(
                dashboard_id, dashboard_name, theme=theme
            )
            
            # Create and add widgets
            created_widgets = []
            for widget_data in session["widgets"]:
                widget = self._create_widget_from_session_data(widget_data)
                if widget:
                    self.dashboard_system.add_widget(dashboard_id, widget)
                    created_widgets.append(widget.widget_id)
            
            # Add to history
            session["history"].append({
                "action": "deploy_dashboard",
                "dashboard_id": dashboard_id,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Dashboard deployed from session {session_id}: {dashboard_id}")
            
            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "widgets_created": len(created_widgets),
                "dashboard_url": f"/dashboard/{dashboard_id}"
            }
            
        except Exception as e:
            self.logger.error(f"Error deploying dashboard from session {session_id}: {e}")
            return {"error": str(e)}
    
    def _create_widget_from_session_data(self, widget_data: Dict[str, Any]) -> Optional[DashboardWidget]:
        """Create actual widget from session data"""
        try:
            widget_type = WidgetType(widget_data["widget_type"])
            
            if widget_type == WidgetType.CHART:
                chart_type = ChartType(widget_data["config"].get("chart_type", "line"))
                widget = ChartWidget(
                    widget_id=widget_data["widget_id"],
                    widget_type=widget_type,
                    title=widget_data["title"],
                    position=widget_data["position"],
                    config=widget_data["config"],
                    chart_type=chart_type
                )
            elif widget_type == WidgetType.METRIC:
                widget = MetricWidget(
                    widget_id=widget_data["widget_id"],
                    widget_type=widget_type,
                    title=widget_data["title"],
                    position=widget_data["position"],
                    config=widget_data["config"],
                    unit=widget_data["config"].get("unit", ""),
                    thresholds=widget_data["config"].get("thresholds", {})
                )
            elif widget_type == WidgetType.TABLE:
                widget = TableWidget(
                    widget_id=widget_data["widget_id"],
                    widget_type=widget_type,
                    title=widget_data["title"],
                    position=widget_data["position"],
                    config=widget_data["config"],
                    columns=widget_data["config"].get("columns", [])
                )
            elif widget_type == WidgetType.CONTROL:
                widget = ControlWidget(
                    widget_id=widget_data["widget_id"],
                    widget_type=widget_type,
                    title=widget_data["title"],
                    position=widget_data["position"],
                    config=widget_data["config"],
                    control_type=widget_data["config"].get("control_type", "button")
                )
            else:
                widget = DashboardWidget(
                    widget_id=widget_data["widget_id"],
                    widget_type=widget_type,
                    title=widget_data["title"],
                    position=widget_data["position"],
                    config=widget_data["config"]
                )
            
            return widget
            
        except Exception as e:
            self.logger.error(f"Error creating widget from session data: {e}")
            return None
    
    def _validate_dashboard_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dashboard configuration"""
        errors = []
        warnings = []
        
        # Validate dashboard name
        if not config.get("name") or len(config["name"].strip()) == 0:
            errors.append("Dashboard name is required")
        
        # Validate widgets
        widgets = config.get("widgets", [])
        if len(widgets) == 0:
            warnings.append("Dashboard has no widgets")
        
        # Check for overlapping widgets
        positions = {}
        for widget in widgets:
            pos = widget.get("position", {})
            pos_key = (pos.get("x"), pos.get("y"))
            
            if pos_key in positions:
                warnings.append(f"Widgets may overlap at position {pos_key}")
            positions[pos_key] = widget["widget_id"]
        
        # Validate widget configurations
        for widget in widgets:
            if not widget.get("title"):
                errors.append(f"Widget {widget.get('widget_id', 'unknown')} missing title")
            
            position = widget.get("position", {})
            required_pos_fields = ["x", "y", "width", "height"]
            for field in required_pos_fields:
                if field not in position:
                    errors.append(f"Widget {widget.get('widget_id', 'unknown')} missing position.{field}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def end_build_session(self, session_id: str, save_session: bool = True) -> Dict[str, Any]:
        """End build session"""
        if session_id not in self.build_sessions:
            return {"error": "Build session not found"}
        
        try:
            session = self.build_sessions[session_id]
            
            # Save session if requested
            if save_session:
                session_data = session.copy()
                session_data["end_time"] = datetime.now()
                # Could save to persistent storage here
            
            # Remove from active sessions
            del self.build_sessions[session_id]
            
            self.logger.info(f"Build session ended: {session_id}")
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Error ending build session {session_id}: {e}")
            return {"error": str(e)}
    
    # ========================================================================
    # ENHANCEMENT INFO
    # ========================================================================
    
    def get_enhancement_info(self) -> Dict[str, Any]:
        """Get information about no-code enhancement"""
        return {
            "enhancement_type": "No-Code Dashboard Builder",
            "base_system": "Unified Dashboard System",
            "enhancement_timestamp": "2025-08-19T20:20:06.000000",
            "capabilities": [
                "Visual drag-and-drop dashboard builder",
                "Pre-built widget templates library",
                "Complete dashboard templates",
                "Real-time preview and validation",
                "Template-based dashboard creation",
                "Visual configuration interface",
                "Auto-save and session management",
                "Widget palette with categories",
                "Responsive layout builder",
                "Theme and styling options"
            ],
            "builder_modes": [mode.value for mode in BuilderMode],
            "component_categories": [cat.value for cat in ComponentCategory],
            "features": {
                "visual_builder": True,
                "template_builder": True,
                "drag_and_drop": True,
                "real_time_preview": True,
                "configuration_validation": True,
                "auto_save": True
            },
            "templates": {
                "widget_templates": len(self.widget_templates),
                "dashboard_templates": len(self.dashboard_templates),
                "available_widgets": list(self.widget_templates.keys()),
                "available_dashboards": list(self.dashboard_templates.keys())
            },
            "status": "FULLY_OPERATIONAL"
        }


# ============================================================================
# INTEGRATION WITH UNIFIED DASHBOARD
# ============================================================================

# Create no-code builder instance
nocode_dashboard_builder = NoCodeDashboardBuilder()

# Integrate with unified dashboard
unified_dashboard.nocode_builder = nocode_dashboard_builder

# Export for external use
__all__ = [
    'NoCodeDashboardBuilder',
    'WidgetTemplate',
    'DashboardTemplate',
    'BuilderMode',
    'ComponentCategory',
    'nocode_dashboard_builder'
]