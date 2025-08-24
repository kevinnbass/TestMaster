"""
Interactive Dashboard System
============================

Real-time interactive dashboard with advanced widgets and controls.
Provides comprehensive monitoring and control interface.

Author: TestMaster Team
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

# Configure logging
logger = logging.getLogger(__name__)


class WidgetType(Enum):
    """Types of dashboard widgets"""
    CHART = "chart"
    METRIC = "metric"
    TABLE = "table"
    MAP = "map"
    TIMELINE = "timeline"
    LOG = "log"
    CONTROL = "control"
    ALERT = "alert"


class ChartType(Enum):
    """Types of charts"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    RADAR = "radar"


@dataclass
class DashboardWidget:
    """Base dashboard widget"""
    widget_id: str = field(default_factory=lambda: f"widget_{uuid.uuid4().hex[:8]}")
    widget_type: WidgetType = WidgetType.METRIC
    title: str = ""
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    size: Dict[str, int] = field(default_factory=lambda: {"width": 4, "height": 3})
    config: Dict[str, Any] = field(default_factory=dict)
    data: Any = None
    refresh_interval: int = 5000  # milliseconds
    last_update: datetime = field(default_factory=datetime.now)
    visible: bool = True
    interactive: bool = True
    
    def update_data(self, new_data: Any):
        """Update widget data"""
        self.data = new_data
        self.last_update = datetime.now()
    
    def get_age_seconds(self) -> float:
        """Get data age in seconds"""
        return (datetime.now() - self.last_update).total_seconds()


@dataclass
class RealtimeChart(DashboardWidget):
    """Real-time chart widget"""
    chart_type: ChartType = ChartType.LINE
    max_points: int = 100
    data_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    axes: Dict[str, Any] = field(default_factory=dict)
    series: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_point(self, value: float, series_name: str = "default", 
                  timestamp: Optional[datetime] = None):
        """Add a data point to the chart"""
        if timestamp is None:
            timestamp = datetime.now()
        
        point = {
            "timestamp": timestamp.isoformat(),
            "value": value,
            "series": series_name
        }
        self.data_buffer.append(point)
        self.last_update = datetime.now()
    
    def get_chart_data(self) -> Dict[str, Any]:
        """Get formatted chart data"""
        return {
            "type": self.chart_type.value,
            "data": list(self.data_buffer),
            "series": self.series,
            "axes": self.axes,
            "config": self.config
        }


@dataclass
class MetricsPanel(DashboardWidget):
    """Metrics display panel"""
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    trends: Dict[str, List[float]] = field(default_factory=dict)
    
    def update_metric(self, name: str, value: Any, unit: str = ""):
        """Update a metric value"""
        if name not in self.metrics:
            self.metrics[name] = {
                "value": value,
                "unit": unit,
                "history": [],
                "status": "normal"
            }
        else:
            # Store history
            self.metrics[name]["history"].append({
                "value": self.metrics[name]["value"],
                "timestamp": self.last_update.isoformat()
            })
            
            # Update current value
            self.metrics[name]["value"] = value
            self.metrics[name]["unit"] = unit
        
        # Check thresholds
        if name in self.thresholds:
            threshold = self.thresholds[name]
            if "critical" in threshold and value > threshold["critical"]:
                self.metrics[name]["status"] = "critical"
            elif "warning" in threshold and value > threshold["warning"]:
                self.metrics[name]["status"] = "warning"
            else:
                self.metrics[name]["status"] = "normal"
        
        # Update trends
        if name not in self.trends:
            self.trends[name] = []
        self.trends[name].append(float(value) if isinstance(value, (int, float)) else 0)
        
        # Keep trend history limited
        if len(self.trends[name]) > 20:
            self.trends[name] = self.trends[name][-20:]
        
        self.last_update = datetime.now()
    
    def get_metric_status(self, name: str) -> str:
        """Get metric status"""
        if name in self.metrics:
            return self.metrics[name].get("status", "unknown")
        return "unknown"


@dataclass
class ControlPanel(DashboardWidget):
    """Interactive control panel"""
    controls: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    callbacks: Dict[str, Callable] = field(default_factory=dict)
    states: Dict[str, Any] = field(default_factory=dict)
    
    def add_control(self, name: str, control_type: str, 
                   config: Dict[str, Any] = None, callback: Optional[Callable] = None):
        """Add a control to the panel"""
        control_id = f"control_{name}_{uuid.uuid4().hex[:4]}"
        
        self.controls[control_id] = {
            "name": name,
            "type": control_type,  # button, slider, switch, dropdown, input
            "config": config or {},
            "enabled": True,
            "value": None
        }
        
        if callback:
            self.callbacks[control_id] = callback
        
        return control_id
    
    def trigger_control(self, control_id: str, value: Any = None) -> Any:
        """Trigger a control action"""
        if control_id not in self.controls:
            return None
        
        # Update control value
        self.controls[control_id]["value"] = value
        
        # Execute callback if exists
        if control_id in self.callbacks:
            try:
                result = self.callbacks[control_id](value)
                self.states[control_id] = value
                self.last_update = datetime.now()
                return result
            except Exception as e:
                logger.error(f"Control callback error: {e}")
                return None
        
        return value
    
    def get_control_state(self, control_id: str) -> Any:
        """Get current control state"""
        return self.states.get(control_id)


class InteractiveDashboard:
    """
    Main interactive dashboard system.
    
    Features:
    - Customizable widget layout
    - Real-time data updates
    - Interactive controls
    - Responsive grid system
    - Theme support
    - Export/Import layouts
    - Alert management
    """
    
    def __init__(self, name: str = "TestMaster Dashboard"):
        self.dashboard_id = f"dashboard_{uuid.uuid4().hex[:12]}"
        self.name = name
        self.widgets: Dict[str, DashboardWidget] = {}
        self.layout_grid = {"columns": 12, "rows": 8}
        self.theme = "dark"
        self.refresh_rate = 1000  # milliseconds
        self.alerts: List[Dict[str, Any]] = []
        self.data_sources: Dict[str, Callable] = {}
        self.created_at = datetime.now()
        self.last_modified = datetime.now()
        
        # Initialize default widgets
        self._create_default_widgets()
    
    def _create_default_widgets(self):
        """Create default dashboard widgets"""
        # System metrics
        metrics = MetricsPanel(
            title="System Metrics",
            position={"x": 0, "y": 0},
            size={"width": 4, "height": 2}
        )
        metrics.update_metric("CPU Usage", 45, "%")
        metrics.update_metric("Memory", 2.3, "GB")
        metrics.update_metric("Active Agents", 12, "")
        self.add_widget(metrics)
        
        # Performance chart
        chart = RealtimeChart(
            title="Performance Trends",
            position={"x": 4, "y": 0},
            size={"width": 4, "height": 3},
            chart_type=ChartType.LINE
        )
        self.add_widget(chart)
        
        # Control panel
        controls = ControlPanel(
            title="System Controls",
            position={"x": 8, "y": 0},
            size={"width": 4, "height": 2}
        )
        controls.add_control("Start", "button")
        controls.add_control("Stop", "button")
        controls.add_control("Scale", "slider", {"min": 1, "max": 10})
        self.add_widget(controls)
    
    def add_widget(self, widget: DashboardWidget) -> str:
        """Add a widget to the dashboard"""
        self.widgets[widget.widget_id] = widget
        self.last_modified = datetime.now()
        
        logger.info(f"Added widget {widget.widget_id} to dashboard")
        return widget.widget_id
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard"""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            self.last_modified = datetime.now()
            return True
        return False
    
    def update_widget_position(self, widget_id: str, 
                              position: Dict[str, int]) -> bool:
        """Update widget position"""
        if widget_id in self.widgets:
            self.widgets[widget_id].position = position
            self.last_modified = datetime.now()
            return True
        return False
    
    def update_widget_size(self, widget_id: str, 
                          size: Dict[str, int]) -> bool:
        """Update widget size"""
        if widget_id in self.widgets:
            self.widgets[widget_id].size = size
            self.last_modified = datetime.now()
            return True
        return False
    
    def add_data_source(self, name: str, source: Callable):
        """Add a data source for widgets"""
        self.data_sources[name] = source
    
    async def refresh_data(self):
        """Refresh all widget data"""
        for widget in self.widgets.values():
            if widget.get_age_seconds() * 1000 > widget.refresh_interval:
                # Check if widget has associated data source
                source_name = widget.config.get("data_source")
                if source_name and source_name in self.data_sources:
                    try:
                        new_data = await self.data_sources[source_name]()
                        widget.update_data(new_data)
                    except Exception as e:
                        logger.error(f"Data refresh error for widget {widget.widget_id}: {e}")
    
    def add_alert(self, level: str, message: str, 
                  source: Optional[str] = None) -> str:
        """Add an alert to the dashboard"""
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        alert = {
            "id": alert_id,
            "level": level,  # info, warning, error, critical
            "message": message,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        return alert_id
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                return True
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active (unacknowledged) alerts"""
        return [a for a in self.alerts if not a["acknowledged"]]
    
    def set_theme(self, theme: str):
        """Set dashboard theme"""
        self.theme = theme
        self.last_modified = datetime.now()
    
    def export_layout(self) -> Dict[str, Any]:
        """Export dashboard layout"""
        return {
            "dashboard_id": self.dashboard_id,
            "name": self.name,
            "theme": self.theme,
            "layout_grid": self.layout_grid,
            "widgets": [
                {
                    "widget_id": w.widget_id,
                    "type": w.widget_type.value,
                    "title": w.title,
                    "position": w.position,
                    "size": w.size,
                    "config": w.config
                }
                for w in self.widgets.values()
            ],
            "exported_at": datetime.now().isoformat()
        }
    
    def import_layout(self, layout_data: Dict[str, Any]) -> bool:
        """Import dashboard layout"""
        try:
            self.name = layout_data.get("name", self.name)
            self.theme = layout_data.get("theme", self.theme)
            self.layout_grid = layout_data.get("layout_grid", self.layout_grid)
            
            # Clear existing widgets
            self.widgets.clear()
            
            # Import widgets
            for widget_data in layout_data.get("widgets", []):
                widget_type = WidgetType(widget_data["type"])
                
                if widget_type == WidgetType.CHART:
                    widget = RealtimeChart(
                        title=widget_data["title"],
                        position=widget_data["position"],
                        size=widget_data["size"],
                        config=widget_data.get("config", {})
                    )
                elif widget_type == WidgetType.METRIC:
                    widget = MetricsPanel(
                        title=widget_data["title"],
                        position=widget_data["position"],
                        size=widget_data["size"],
                        config=widget_data.get("config", {})
                    )
                elif widget_type == WidgetType.CONTROL:
                    widget = ControlPanel(
                        title=widget_data["title"],
                        position=widget_data["position"],
                        size=widget_data["size"],
                        config=widget_data.get("config", {})
                    )
                else:
                    widget = DashboardWidget(
                        widget_type=widget_type,
                        title=widget_data["title"],
                        position=widget_data["position"],
                        size=widget_data["size"],
                        config=widget_data.get("config", {})
                    )
                
                self.add_widget(widget)
            
            self.last_modified = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Layout import error: {e}")
            return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        return {
            "dashboard_id": self.dashboard_id,
            "name": self.name,
            "theme": self.theme,
            "layout_grid": self.layout_grid,
            "widgets": {
                widget_id: {
                    "widget_id": widget.widget_id,
                    "type": widget.widget_type.value,
                    "title": widget.title,
                    "position": widget.position,
                    "size": widget.size,
                    "data": widget.data,
                    "last_update": widget.last_update.isoformat(),
                    "visible": widget.visible,
                    "interactive": widget.interactive
                }
                for widget_id, widget in self.widgets.items()
            },
            "alerts": self.get_active_alerts(),
            "refresh_rate": self.refresh_rate,
            "last_modified": self.last_modified.isoformat()
        }
    
    def get_widget_data(self, widget_id: str) -> Optional[Dict[str, Any]]:
        """Get specific widget data"""
        if widget_id not in self.widgets:
            return None
        
        widget = self.widgets[widget_id]
        
        data = {
            "widget_id": widget.widget_id,
            "type": widget.widget_type.value,
            "title": widget.title,
            "data": widget.data,
            "last_update": widget.last_update.isoformat()
        }
        
        # Add type-specific data
        if isinstance(widget, RealtimeChart):
            data["chart_data"] = widget.get_chart_data()
        elif isinstance(widget, MetricsPanel):
            data["metrics"] = widget.metrics
            data["trends"] = widget.trends
        elif isinstance(widget, ControlPanel):
            data["controls"] = widget.controls
            data["states"] = widget.states
        
        return data
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status"""
        return {
            "dashboard_id": self.dashboard_id,
            "name": self.name,
            "widget_count": len(self.widgets),
            "active_alerts": len(self.get_active_alerts()),
            "total_alerts": len(self.alerts),
            "data_sources": list(self.data_sources.keys()),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
            "last_modified": self.last_modified.isoformat(),
            "features": {
                "real_time_updates": True,
                "interactive_controls": True,
                "customizable_layout": True,
                "data_export": True,
                "alert_management": True,
                "theme_support": True,
                "responsive_grid": True
            }
        }