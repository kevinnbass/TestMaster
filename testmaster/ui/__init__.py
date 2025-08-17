"""
TestMaster UI Module - Layer 2

Real-time web dashboard inspired by framework analysis:
- Agency-Swarm Gradio: Web-based UI with queue updates
- PraisonAI: Performance metrics visualization  
- LangGraph: Graph-based workflow display

Provides:
- Real-time web dashboard with WebSocket updates
- Coverage and quality metrics display
- Alert and notification system
- Module health visualization
"""

from .dashboard import TestMasterDashboard, DashboardConfig
from .metrics_display import MetricsDisplay, CoverageWidget, QualityWidget
from .alert_system import AlertSystem, AlertLevel, Alert

__all__ = [
    "TestMasterDashboard",
    "DashboardConfig", 
    "MetricsDisplay",
    "CoverageWidget",
    "QualityWidget",
    "AlertSystem",
    "AlertLevel",
    "Alert"
]