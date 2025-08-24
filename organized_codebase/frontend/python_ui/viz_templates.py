#!/usr/bin/env python3
"""
🏗️ MODULE: Visualization Templates - Predefined Chart Templates
==================================================================

📋 PURPOSE:
    Template management and initialization extracted from
    custom_visualization_builder.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Default chart template definitions
    • Template initialization and management
    • Template listing and access methods
    • Predefined visualization configurations

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract visualization templates from custom_visualization_builder.py
   └─ Source: Lines 111-185 (74 lines)
   └─ Purpose: Separate template management into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: ChartType, DataSource enums
📤 Provides: Template management functionality
"""

from typing import Dict, List, Any
from .viz_types import ChartType, DataSource


class TemplateManager:
    """Manages visualization chart templates."""
    
    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.initialize_default_templates()
    
    def initialize_default_templates(self):
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
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """Get a specific template by name."""
        return self.templates.get(template_name, {})
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available templates."""
        return self.templates.copy()
    
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
    
    def add_template(self, name: str, template: Dict[str, Any]) -> bool:
        """
        Add a new template.
        
        Args:
            name: Template name
            template: Template configuration
        
        Returns:
            True if added successfully
        """
        required_fields = ["title", "chart_type", "data_source", "data_fields"]
        if not all(field in template for field in required_fields):
            return False
        
        self.templates[name] = template
        return True
    
    def remove_template(self, name: str) -> bool:
        """
        Remove a template.
        
        Args:
            name: Template name to remove
        
        Returns:
            True if removed successfully
        """
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def template_exists(self, name: str) -> bool:
        """Check if a template exists."""
        return name in self.templates