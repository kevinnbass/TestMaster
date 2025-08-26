#!/usr/bin/env python3
"""
🏗️ VISUALIZATION COMPONENTS MODULE - Custom Chart Builder Components
==================================================================

📋 PURPOSE:
    Module initialization for visualization components extracted
    via STEELCLAD protocol from custom_visualization_builder.py

🎯 EXPORTS:
    • ChartBuilder - Chart creation and management
    • TemplateManager - Visualization template management
    • DataFormatter - Chart data processing and formatting
    • ChartType, DataSource - Type definitions
    • ChartConfiguration, VisualizationPanel - Data structures

🔄 STEELCLAD EXTRACTION:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 MODULAR ARCHITECTURE
   └─ Source: custom_visualization_builder.py (706 lines)
   └─ Target: 4 focused modules + streamlined main file
   └─ Status: EXTRACTION COMPLETE
"""

from .viz_types import ChartType, DataSource, ChartConfiguration, VisualizationPanel
from .viz_templates import TemplateManager
from .chart_builders import ChartBuilder
from .data_formatters import DataFormatter

__all__ = [
    'ChartType',
    'DataSource', 
    'ChartConfiguration',
    'VisualizationPanel',
    'TemplateManager',
    'ChartBuilder',
    'DataFormatter'
]