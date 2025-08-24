"""
Dashboard Data Types and Structures

This module defines the core data types, enumerations, and data structures
used throughout the unified dashboard system.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable


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
    GRID = "grid"
    CARD = "card"
    TABS = "tabs"
    IFRAME = "iframe"
    TEXT = "text"
    IMAGE = "image"


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
    TREEMAP = "treemap"
    SANKEY = "sankey"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"


class DashboardTheme(Enum):
    """Dashboard themes"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"
    CUSTOM = "custom"


class LayoutMode(Enum):
    """Dashboard layout modes"""
    FIXED = "fixed"
    RESPONSIVE = "responsive"
    FLUID = "fluid"
    ADAPTIVE = "adaptive"
    GRID = "grid"


class InteractionMode(Enum):
    """Dashboard interaction modes"""
    VIEW_ONLY = "view_only"
    INTERACTIVE = "interactive"
    EDIT_MODE = "edit_mode"
    ADMIN_MODE = "admin_mode"
    DEMO_MODE = "demo_mode"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"


class DataSourceType(Enum):
    """Data source types"""
    STATIC = "static"
    API = "api"
    DATABASE = "database"
    WEBSOCKET = "websocket"
    FILE = "file"
    STREAM = "stream"


class UpdateFrequency(Enum):
    """Widget update frequencies"""
    REAL_TIME = "real_time"
    FAST = "fast"        # 1 second
    NORMAL = "normal"    # 5 seconds
    SLOW = "slow"        # 30 seconds
    MANUAL = "manual"


@dataclass
class DashboardWidget:
    """Unified dashboard widget definition"""
    widget_id: str = field(default_factory=lambda: f"widget_{uuid.uuid4().hex[:8]}")
    widget_type: WidgetType = WidgetType.METRIC
    title: str = "Untitled Widget"
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "width": 4, "height": 3})
    data_source: Optional[str] = None
    data_source_type: DataSourceType = DataSourceType.STATIC
    update_frequency: UpdateFrequency = UpdateFrequency.NORMAL
    configuration: Dict[str, Any] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_visible: bool = True
    is_interactive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary"""
        return {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type.value,
            "title": self.title,
            "position": self.position,
            "data_source": self.data_source,
            "data_source_type": self.data_source_type.value,
            "update_frequency": self.update_frequency.value,
            "configuration": self.configuration,
            "styling": self.styling,
            "filters": self.filters,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_visible": self.is_visible,
            "is_interactive": self.is_interactive
        }


@dataclass
class ChartWidget(DashboardWidget):
    """Chart-specific widget configuration"""
    chart_type: ChartType = ChartType.LINE
    x_axis: str = ""
    y_axis: str = ""
    series: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    show_legend: bool = True
    show_grid: bool = True
    is_animated: bool = True
    
    def __post_init__(self):
        self.widget_type = WidgetType.CHART


@dataclass
class MetricWidget(DashboardWidget):
    """Metric display widget"""
    value: Union[int, float, str] = 0
    unit: str = ""
    format_string: str = "{value}"
    trend: Optional[str] = None  # "up", "down", "flat"
    trend_value: Optional[float] = None
    target_value: Optional[Union[int, float]] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def __post_init__(self):
        self.widget_type = WidgetType.METRIC


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    layout_id: str = field(default_factory=lambda: f"layout_{uuid.uuid4().hex[:8]}")
    name: str = "Default Layout"
    mode: LayoutMode = LayoutMode.RESPONSIVE
    theme: DashboardTheme = DashboardTheme.LIGHT
    columns: int = 12
    row_height: int = 50
    margin: Dict[str, int] = field(default_factory=lambda: {"x": 10, "y": 10})
    padding: Dict[str, int] = field(default_factory=lambda: {"x": 5, "y": 5})
    breakpoints: Dict[str, int] = field(default_factory=lambda: {
        "xs": 480, "sm": 768, "md": 1024, "lg": 1200, "xl": 1400
    })
    is_responsive: bool = True
    auto_resize: bool = True


@dataclass
class DashboardConfig:
    """Complete dashboard configuration"""
    dashboard_id: str = field(default_factory=lambda: f"dashboard_{uuid.uuid4().hex[:8]}")
    name: str = "New Dashboard"
    description: str = ""
    layout: DashboardLayout = field(default_factory=DashboardLayout)
    widgets: List[DashboardWidget] = field(default_factory=list)
    interaction_mode: InteractionMode = InteractionMode.INTERACTIVE
    refresh_interval: int = 30  # seconds
    auto_refresh: bool = True
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_public: bool = False
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard config to dictionary"""
        return {
            "dashboard_id": self.dashboard_id,
            "name": self.name,
            "description": self.description,
            "layout": self.layout.__dict__,
            "widgets": [w.to_dict() for w in self.widgets],
            "interaction_mode": self.interaction_mode.value,
            "refresh_interval": self.refresh_interval,
            "auto_refresh": self.auto_refresh,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_public": self.is_public,
            "tags": self.tags
        }


@dataclass
class DashboardAlert:
    """Dashboard alert definition"""
    alert_id: str = field(default_factory=lambda: f"alert_{uuid.uuid4().hex[:8]}")
    title: str = "Alert"
    message: str = ""
    level: AlertLevel = AlertLevel.INFO
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_persistent: bool = False
    is_dismissible: bool = True
    actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if alert has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class DashboardEvent:
    """Dashboard event for real-time updates"""
    event_id: str = field(default_factory=lambda: f"event_{uuid.uuid4().hex[:8]}")
    event_type: str = "update"
    target_widget: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "target_widget": self.target_widget,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }