#!/usr/bin/env python3
"""
Advanced Interaction Manager - Extracted from Advanced Gamma Dashboard
=====================================================================

STEELCLAD Phase 3 extraction providing advanced user interaction management
and dashboard customization capabilities.

Author: Agent Z (STEELCLAD Protocol)
"""

from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional

class AdvancedInteractionManager:
    """Manages advanced user interactions and customization."""
    
    def __init__(self):
        self.interaction_patterns = defaultdict(list)
        self.customization_presets = self.load_presets()
        
    def load_presets(self):
        """Load dashboard customization presets."""
        return {
            "executive": {
                "layout": "grid-3x2",
                "widgets": ["system_health", "api_costs", "agent_status", "key_metrics"],
                "theme": "professional"
            },
            "developer": {
                "layout": "grid-2x3",
                "widgets": ["3d_visualization", "performance_metrics", "api_usage", "logs"],
                "theme": "dark"
            },
            "analyst": {
                "layout": "fluid",
                "widgets": ["analytics_charts", "predictive_insights", "trends", "export_tools"],
                "theme": "light"
            }
        }
    
    def track_interaction(self, user_id, interaction_data):
        """Track user interaction for personalization."""
        self.interaction_patterns[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "type": interaction_data["type"],
            "element": interaction_data["element"],
            "context": interaction_data.get("context", {})
        })
    
    def suggest_customization(self, user_id):
        """Suggest dashboard customization based on usage patterns."""
        patterns = self.interaction_patterns.get(user_id, [])
        if len(patterns) < 10:
            return {"preset": "default", "confidence": 0.5}
        
        # Analyze usage patterns
        widget_usage = defaultdict(int)
        for pattern in patterns[-50:]:  # Last 50 interactions
            if pattern["type"] == "widget_interaction":
                widget_usage[pattern["element"]] += 1
        
        # Find best matching preset
        best_preset = "executive"
        best_score = 0
        
        for preset_name, preset_config in self.customization_presets.items():
            score = sum(widget_usage.get(widget, 0) for widget in preset_config["widgets"])
            if score > best_score:
                best_score = score
                best_preset = preset_name
        
        return {
            "preset": best_preset,
            "confidence": min(1.0, best_score / 20),
            "customizations": self.customization_presets[best_preset]
        }

def create_interaction_manager(config: Optional[Dict] = None) -> AdvancedInteractionManager:
    """
    Factory function to create a configured interaction manager instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured AdvancedInteractionManager instance
    """
    manager = AdvancedInteractionManager()
    if config:
        # Apply configuration overrides
        if 'custom_presets' in config:
            manager.customization_presets.update(config['custom_presets'])
    return manager

# Export key components
__all__ = ['AdvancedInteractionManager', 'create_interaction_manager']