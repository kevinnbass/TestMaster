#!/usr/bin/env python3
"""
🏗️ ADVANCED MODULE - Gamma Dashboard Components
==================================================================

📋 PURPOSE:
    Module initialization for advanced gamma dashboard components
    extracted via STEELCLAD protocol from advanced_gamma_dashboard.py

🎯 EXPORTS:
    • AdvancedDashboardEngine - Main advanced dashboard engine
    • AdvancedInteractionManager - User interaction management
    • PerformanceOptimizer - Performance monitoring and optimization
    • DashboardCustomizationEngine - Dashboard customization
    • UserBehaviorTracker - User behavior tracking
    • InsightGenerator - Analytics insight generation
    • ExportManager - Report export management

🔄 STEELCLAD EXTRACTION:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 MODULAR ARCHITECTURE
   └─ Source: advanced_gamma_dashboard.py (442 lines)
   └─ Target: 3 focused modules + streamlined main file
   └─ Status: EXTRACTION COMPLETE
"""

from .gamma_dashboard_logic import AdvancedDashboardEngine
from .gamma_advanced_features import AdvancedInteractionManager
from .gamma_data_processing import (
    PerformanceOptimizer,
    DashboardCustomizationEngine,
    UserBehaviorTracker,
    InsightGenerator,
    ExportManager
)

__all__ = [
    'AdvancedDashboardEngine',
    'AdvancedInteractionManager',
    'PerformanceOptimizer',
    'DashboardCustomizationEngine',
    'UserBehaviorTracker',
    'InsightGenerator',
    'ExportManager'
]