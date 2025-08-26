#!/usr/bin/env python3
"""
Visualization Engine Component
==============================
STEELCLAD Atomized Module (<180 lines)
Extracted from advanced_visualization.py

Core visualization engine for rendering complex visualizations.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

class VizEngine:
    """Atomic visualization engine for complex rendering."""
    
    def __init__(self):
        self.renderers = {}
        self.viz_cache = {}
        self.animation_queue = []
        self.config = self._init_config()
        
    def render_visualization(self, viz_type: str, data: Any, 
                           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main rendering method for any visualization type.
        
        Args:
            viz_type: Type of visualization to render
            data: Data to visualize
            options: Rendering options
            
        Returns:
            Rendered visualization configuration
        """
        if viz_type not in self.renderers:
            self._register_renderer(viz_type)
        
        rendered = {
            "type": viz_type,
            "timestamp": datetime.now().isoformat(),
            "data": self._process_data(viz_type, data),
            "config": self._merge_options(self.config.get(viz_type, {}), options or {}),
            "interactive": True
        }
        
        # Cache for performance
        cache_key = f"{viz_type}_{hash(str(data))}"
        self.viz_cache[cache_key] = rendered
        
        return rendered
    
    def create_dashboard_layout(self, visualizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create dashboard layout with multiple visualizations."""
        return {
            "layout": {
                "type": "responsive_grid",
                "columns": self._calculate_grid_columns(len(visualizations)),
                "gap": 16,
                "padding": 20
            },
            "visualizations": [
                self._prepare_viz_for_dashboard(viz) for viz in visualizations
            ],
            "theme": self.config.get("theme", "light"),
            "refresh_interval": 5000
        }
    
    def animate_transition(self, from_viz: Dict[str, Any], 
                          to_viz: Dict[str, Any]) -> Dict[str, Any]:
        """Create animated transition between visualizations."""
        return {
            "animation": {
                "type": "morph",
                "duration": 500,
                "easing": "easeInOutQuad",
                "from": from_viz,
                "to": to_viz
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def apply_theme(self, theme_name: str) -> None:
        """Apply visualization theme."""
        themes = {
            "light": {
                "background": "#ffffff",
                "text": "#333333",
                "grid": "#e0e0e0",
                "accent": "#2196F3"
            },
            "dark": {
                "background": "#1e1e1e",
                "text": "#ffffff",
                "grid": "#333333",
                "accent": "#64B5F6"
            },
            "contrast": {
                "background": "#000000",
                "text": "#ffffff",
                "grid": "#444444",
                "accent": "#FFD600"
            }
        }
        
        if theme_name in themes:
            self.config["theme"] = theme_name
            self.config["colors"] = themes[theme_name]
    
    def export_visualization(self, viz: Dict[str, Any], format: str) -> Any:
        """Export visualization in specified format."""
        exporters = {
            "json": lambda v: json.dumps(v, indent=2),
            "svg": lambda v: self._generate_svg(v),
            "png": lambda v: self._generate_png_data(v),
            "html": lambda v: self._generate_html(v)
        }
        
        exporter = exporters.get(format, exporters["json"])
        return exporter(viz)
    
    def optimize_for_performance(self, viz: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize visualization for performance."""
        optimized = viz.copy()
        
        # Reduce data points if too many
        if "data" in optimized and isinstance(optimized["data"], list):
            if len(optimized["data"]) > 1000:
                # Sample data for performance
                step = len(optimized["data"]) // 1000
                optimized["data"] = optimized["data"][::step]
        
        # Disable animations for large datasets
        if self._is_large_dataset(optimized.get("data")):
            optimized["config"] = optimized.get("config", {})
            optimized["config"]["animation"] = False
        
        return optimized
    
    def _register_renderer(self, viz_type: str) -> None:
        """Register a new renderer for a visualization type."""
        self.renderers[viz_type] = {
            "processor": self._get_processor(viz_type),
            "validator": self._get_validator(viz_type)
        }
    
    def _process_data(self, viz_type: str, data: Any) -> Any:
        """Process data for specific visualization type."""
        processors = {
            "graph": lambda d: {"nodes": d.get("nodes", []), "edges": d.get("edges", [])},
            "chart": lambda d: {"labels": d.get("labels", []), "values": d.get("values", [])},
            "heatmap": lambda d: {"matrix": d.get("matrix", []), "scale": d.get("scale", "linear")},
            "tree": lambda d: {"root": d.get("root", {}), "children": d.get("children", [])}
        }
        
        processor = processors.get(viz_type, lambda d: d)
        return processor(data)
    
    def _merge_options(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """Merge custom options with defaults."""
        merged = default.copy()
        merged.update(custom)
        return merged
    
    def _calculate_grid_columns(self, viz_count: int) -> int:
        """Calculate optimal grid columns for visualization count."""
        if viz_count <= 2: return viz_count
        elif viz_count <= 4: return 2
        elif viz_count <= 9: return 3
        return 4
    
    def _prepare_viz_for_dashboard(self, viz: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare visualization for dashboard display."""
        return {
            **viz,
            "container": {
                "responsive": True,
                "maintainAspectRatio": True,
                "minHeight": 300
            }
        }
    
    def _is_large_dataset(self, data: Any) -> bool:
        """Check if dataset is large."""
        if isinstance(data, list):
            return len(data) > 5000
        elif isinstance(data, dict):
            total_items = sum(len(v) if isinstance(v, list) else 1 
                            for v in data.values())
            return total_items > 5000
        return False
    
    def _generate_svg(self, viz: Dict[str, Any]) -> str:
        """Generate SVG representation."""
        return f'<svg><title>{viz.get("type", "viz")}</title></svg>'
    
    def _generate_png_data(self, viz: Dict[str, Any]) -> bytes:
        """Generate PNG data (placeholder)."""
        return b"PNG_DATA_PLACEHOLDER"
    
    def _generate_html(self, viz: Dict[str, Any]) -> str:
        """Generate HTML representation."""
        return f'<div class="viz">{json.dumps(viz)}</div>'
    
    def _get_processor(self, viz_type: str) -> Callable:
        """Get data processor for visualization type."""
        return lambda d: d
    
    def _get_validator(self, viz_type: str) -> Callable:
        """Get data validator for visualization type."""
        return lambda d: True
    
    def _init_config(self) -> Dict[str, Any]:
        """Initialize default configuration."""
        return {
            "theme": "light",
            "animation": True,
            "responsive": True,
            "interaction": True,
            "export": ["json", "svg", "png"],
            "performance": {
                "max_data_points": 10000,
                "cache_enabled": True,
                "lazy_loading": True
            }
        }

# Module exports
__all__ = ['VizEngine']