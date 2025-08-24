#!/usr/bin/env python3
"""
Dashboard Atomic Components
===========================
STEELCLAD Atomized Modules - Agent Y Extraction

Collection of atomic UI and visualization components extracted from
large dashboard files. Each component is under 200 lines and follows
single responsibility principle.
"""

# Linkage Analysis Components
from .linkage_visualizer import LinkageVisualizer
from .linkage_ui_controls import LinkageUIControls

# Analytics Components  
from .dashboard_analytics import DashboardAnalytics

# Security Components
from .security_dashboard_ui import SecurityDashboardUI
from .security_visualizations import SecurityVisualizations

# Performance Components
from .performance_charts import PerformanceCharts

# Advanced Visualization Components
from .advanced_charts import AdvancedCharts
from .viz_engine import VizEngine

# Gamma Enhancement Components
from .gamma_viz_components import GammaVizComponents

# Unified atomic component factory
class AtomicComponentFactory:
    """Factory for creating atomic dashboard components."""
    
    @staticmethod
    def create_linkage_visualizer():
        """Create linkage visualization component."""
        return LinkageVisualizer()
    
    @staticmethod
    def create_dashboard_analytics():
        """Create dashboard analytics component."""
        return DashboardAnalytics()
    
    @staticmethod
    def create_security_ui():
        """Create security dashboard UI component."""
        return SecurityDashboardUI()
    
    @staticmethod
    def create_performance_charts():
        """Create performance charts component."""
        return PerformanceCharts()
    
    @staticmethod
    def create_advanced_charts():
        """Create advanced charts component."""
        return AdvancedCharts()
    
    @staticmethod
    def create_viz_engine():
        """Create visualization engine."""
        return VizEngine()
    
    @staticmethod
    def create_gamma_components():
        """Create Gamma visualization components."""
        return GammaVizComponents()
    
    @staticmethod
    def create_all_components():
        """Create all atomic components."""
        return {
            "linkage_visualizer": LinkageVisualizer(),
            "linkage_controls": LinkageUIControls(),
            "analytics": DashboardAnalytics(),
            "security_ui": SecurityDashboardUI(),
            "security_viz": SecurityVisualizations(),
            "performance_charts": PerformanceCharts(),
            "advanced_charts": AdvancedCharts(),
            "viz_engine": VizEngine(),
            "gamma_viz": GammaVizComponents()
        }

# Export all components
__all__ = [
    # Individual Components
    'LinkageVisualizer',
    'LinkageUIControls',
    'DashboardAnalytics',
    'SecurityDashboardUI',
    'SecurityVisualizations',
    'PerformanceCharts',
    'AdvancedCharts',
    'VizEngine',
    'GammaVizComponents',
    # Factory
    'AtomicComponentFactory'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Agent Y - STEELCLAD Atomization'
__status__ = 'Production'