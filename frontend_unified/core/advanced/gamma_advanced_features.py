#!/usr/bin/env python3
"""
🏗️ MODULE: Gamma Advanced Features - Interaction Management
==================================================================

📋 PURPOSE:
    Advanced user interaction management and customization presets
    extracted from advanced_gamma_dashboard.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • AdvancedInteractionManager - User interaction tracking
    • Dashboard customization presets (executive, developer, analyst)
    • Usage pattern analysis and personalization suggestions
    • Interaction-based dashboard optimization

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION  
   └─ Goal: Extract advanced features from advanced_gamma_dashboard.py
   └─ Source: Lines 161-223 (62 lines)
   └─ Purpose: Separate interaction management into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: collections.defaultdict, datetime
📤 Provides: AdvancedInteractionManager class
"""

from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict


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
    
    def get_interaction_summary(self, user_id: str) -> Dict[str, Any]:
        """Get interaction summary for a user."""
        patterns = self.interaction_patterns.get(user_id, [])
        if not patterns:
            return {"total_interactions": 0, "most_used_features": []}
        
        feature_usage = defaultdict(int)
        for pattern in patterns:
            feature_usage[pattern.get("element", "unknown")] += 1
        
        return {
            "total_interactions": len(patterns),
            "most_used_features": sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)[:5],
            "last_interaction": patterns[-1]["timestamp"] if patterns else None
        }