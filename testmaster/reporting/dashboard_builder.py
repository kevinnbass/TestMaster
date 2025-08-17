"""
Dashboard Builder for TestMaster Reporting

Creates interactive performance dashboards with visual charts and analytics.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.feature_flags import FeatureFlags

class ChartType(Enum):
    """Chart types for dashboards."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"

@dataclass
class DashboardSection:
    """Dashboard section configuration."""
    section_id: str
    title: str
    chart_type: ChartType
    data_source: str
    metrics: List[str]

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    title: str
    sections: List[DashboardSection]
    refresh_interval: int = 30
    theme: str = "light"

class DashboardBuilder:
    """Dashboard builder for performance visualization."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'performance_reporting')
    
    def build_dashboard(self, config: DashboardConfig) -> str:
        """Build performance dashboard."""
        if not self.enabled:
            return "Dashboard building disabled"
        
        dashboard_html = f"""
        <div class="dashboard">
            <h1>{config.title}</h1>
            <div class="sections">
        """
        
        for section in config.sections:
            dashboard_html += f"""
                <div class="section">
                    <h2>{section.title}</h2>
                    <div class="chart" data-type="{section.chart_type.value}">
                        Chart placeholder for {section.title}
                    </div>
                </div>
            """
        
        dashboard_html += """
            </div>
        </div>
        """
        
        return dashboard_html

def get_dashboard_builder() -> DashboardBuilder:
    """Get dashboard builder instance."""
    return DashboardBuilder()

def build_performance_dashboard(title: str = "Performance Dashboard") -> str:
    """Build performance dashboard."""
    builder = get_dashboard_builder()
    config = DashboardConfig(
        title=title,
        sections=[
            DashboardSection("overview", "System Overview", ChartType.LINE, "system", ["cpu", "memory"]),
            DashboardSection("performance", "Performance Metrics", ChartType.BAR, "performance", ["throughput"])
        ]
    )
    return builder.build_dashboard(config)