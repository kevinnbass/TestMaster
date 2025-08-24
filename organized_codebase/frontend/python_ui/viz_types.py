#!/usr/bin/env python3
"""
🏗️ MODULE: Visualization Types - Chart Types and Data Structures
==================================================================

📋 PURPOSE:
    Chart type definitions and data structures extracted from
    custom_visualization_builder.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • ChartType and DataSource enums
    • ChartConfiguration and VisualizationPanel dataclasses
    • Core data structures for visualization system

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract type definitions from custom_visualization_builder.py
   └─ Source: Lines 21-84 (63 lines)
   └─ Purpose: Separate type definitions into shared module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: typing, dataclasses, datetime, enum
📤 Provides: Core visualization type definitions
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ChartType(Enum):
    """Supported chart types for custom visualization."""
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    RADAR = "radar"
    SCATTER = "scatter"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    TIMELINE = "timeline"
    TREEMAP = "treemap"


class DataSource(Enum):
    """Available data sources for visualizations."""
    PERSONAL_ANALYTICS = "personal_analytics"
    PREDICTIVE_DATA = "predictive_data"
    PERFORMANCE_METRICS = "performance_metrics"
    QUALITY_METRICS = "quality_metrics"
    PRODUCTIVITY_INSIGHTS = "productivity_insights"
    CUSTOM_DATA = "custom_data"


@dataclass
class ChartConfiguration:
    """Configuration for a custom chart."""
    id: str
    title: str
    chart_type: ChartType
    data_source: DataSource
    data_fields: List[str]
    filters: Dict[str, Any] = None
    styling: Dict[str, Any] = None
    options: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class VisualizationPanel:
    """A custom visualization panel containing multiple charts."""
    id: str
    title: str
    description: str
    charts: List[str]  # Chart IDs
    layout: Dict[str, Any]
    position: Dict[str, int]
    size: Dict[str, int]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()