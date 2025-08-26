#!/usr/bin/env python3
"""
Linkage UI Controls Component
=============================
STEELCLAD Atomized Module (<150 lines)
Extracted from linkage_dashboard_comprehensive.py

UI controls and interaction handlers for linkage analysis.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

class LinkageUIControls:
    """Atomic component for linkage analysis UI controls."""
    
    def __init__(self):
        self.filters = {}
        self.selected_files = []
        self.view_mode = "graph"
        self.control_state = {}
        
    def render_filter_controls(self) -> Dict[str, Any]:
        """Render filter controls for linkage analysis."""
        return {
            "filters": [
                {
                    "id": "file_type",
                    "label": "File Type",
                    "type": "select",
                    "options": ["All", "Orphaned", "Hanging", "Marginal", "Well Connected"],
                    "value": self.filters.get("file_type", "All")
                },
                {
                    "id": "dependency_count",
                    "label": "Dependency Count",
                    "type": "range",
                    "min": 0,
                    "max": 50,
                    "value": self.filters.get("dependency_count", [0, 50])
                },
                {
                    "id": "path_filter",
                    "label": "Path Filter",
                    "type": "text",
                    "placeholder": "Enter path pattern...",
                    "value": self.filters.get("path_filter", "")
                }
            ],
            "actions": [
                {"id": "apply", "label": "Apply Filters", "type": "primary"},
                {"id": "reset", "label": "Reset", "type": "secondary"}
            ]
        }
    
    def render_view_controls(self) -> Dict[str, Any]:
        """Render view mode controls."""
        return {
            "view_modes": [
                {"id": "graph", "label": "Graph View", "icon": "graph"},
                {"id": "tree", "label": "Tree View", "icon": "tree"},
                {"id": "list", "label": "List View", "icon": "list"},
                {"id": "matrix", "label": "Matrix View", "icon": "grid"}
            ],
            "current_view": self.view_mode,
            "view_options": self._get_view_options()
        }
    
    def render_action_buttons(self) -> List[Dict[str, Any]]:
        """Render action buttons for selected files."""
        return [
            {
                "id": "analyze",
                "label": "Analyze Selected",
                "icon": "search",
                "enabled": len(self.selected_files) > 0,
                "action": "analyze_selection"
            },
            {
                "id": "export",
                "label": "Export Results",
                "icon": "download",
                "enabled": True,
                "action": "export_results"
            },
            {
                "id": "refresh",
                "label": "Refresh Analysis",
                "icon": "refresh",
                "enabled": True,
                "action": "refresh_analysis"
            }
        ]
    
    def handle_filter_change(self, filter_id: str, value: Any) -> None:
        """Handle filter value changes."""
        self.filters[filter_id] = value
        self.control_state["filters_dirty"] = True
    
    def handle_view_change(self, view_mode: str) -> None:
        """Handle view mode changes."""
        if view_mode in ["graph", "tree", "list", "matrix"]:
            self.view_mode = view_mode
            self.control_state["view_changed"] = True
    
    def handle_file_selection(self, file_path: str, selected: bool) -> None:
        """Handle file selection changes."""
        if selected and file_path not in self.selected_files:
            self.selected_files.append(file_path)
        elif not selected and file_path in self.selected_files:
            self.selected_files.remove(file_path)
    
    def apply_filters(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply current filters to data."""
        filtered_data = data
        
        # Filter by file type
        file_type = self.filters.get("file_type", "All")
        if file_type != "All":
            type_map = {
                "Orphaned": "orphaned",
                "Hanging": "hanging", 
                "Marginal": "marginal",
                "Well Connected": "well_connected"
            }
            filtered_data = [d for d in filtered_data 
                           if d.get("type") == type_map.get(file_type)]
        
        # Filter by dependency count
        dep_range = self.filters.get("dependency_count", [0, 50])
        if dep_range:
            filtered_data = [d for d in filtered_data 
                           if dep_range[0] <= d.get("total_deps", 0) <= dep_range[1]]
        
        # Filter by path pattern
        path_pattern = self.filters.get("path_filter", "")
        if path_pattern:
            filtered_data = [d for d in filtered_data 
                           if path_pattern.lower() in d.get("path", "").lower()]
        
        return filtered_data
    
    def get_control_state(self) -> Dict[str, Any]:
        """Get current control state."""
        return {
            "filters": self.filters,
            "selected_files": self.selected_files,
            "view_mode": self.view_mode,
            "timestamp": datetime.now().isoformat(),
            **self.control_state
        }
    
    def reset_controls(self) -> None:
        """Reset all controls to default state."""
        self.filters = {}
        self.selected_files = []
        self.view_mode = "graph"
        self.control_state = {"reset": True}
    
    def _get_view_options(self) -> Dict[str, Any]:
        """Get options for current view mode."""
        options = {
            "graph": {"layout": "force", "physics": True, "clustering": False},
            "tree": {"expanded": False, "show_metrics": True},
            "list": {"sortable": True, "pagination": True, "page_size": 50},
            "matrix": {"cell_size": "auto", "color_scale": "heat"}
        }
        return options.get(self.view_mode, {})

# Module exports
__all__ = ['LinkageUIControls']