#!/usr/bin/env python3
"""
🏗️ ENHANCEMENTS MODULE - Gamma Dashboard Components
==================================================================

📋 PURPOSE:
    Module initialization for gamma dashboard enhancement components
    extracted via STEELCLAD protocol from unified_gamma_dashboard_enhanced.py

🎯 EXPORTS:
    • EnhancedUnifiedDashboard - Main dashboard class
    • APIUsageTracker - API usage tracking with Agent E integration
    • DataIntegrator - Data integration with personal analytics
    • PerformanceMonitor - Performance monitoring and metrics
    • ENHANCED_DASHBOARD_HTML - Complete UI template

🔄 STEELCLAD EXTRACTION:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 MODULAR ARCHITECTURE
   └─ Source: unified_gamma_dashboard_enhanced.py (1,172 lines)
   └─ Target: 3 focused modules + streamlined main file
   └─ Status: EXTRACTION COMPLETE
"""

from .gamma_enhancements import EnhancedUnifiedDashboard
from .performance_enhancements import APIUsageTracker, DataIntegrator, PerformanceMonitor
from .ui_enhancements import ENHANCED_DASHBOARD_HTML

__all__ = [
    'EnhancedUnifiedDashboard',
    'APIUsageTracker', 
    'DataIntegrator',
    'PerformanceMonitor',
    'ENHANCED_DASHBOARD_HTML'
]