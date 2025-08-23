#!/usr/bin/env python3
"""
🏗️ PREDICTION COMPONENTS MODULE - Predictive Analytics Components
==================================================================

📋 PURPOSE:
    Module initialization for prediction components extracted
    via STEELCLAD protocol from predictive_analytics_engine.py

🎯 EXPORTS:
    • VelocityPredictor - Development velocity prediction
    • QualityForecaster - Code quality forecasting  
    • PatternAnalyzer - Productivity pattern recognition
    • RiskAssessor - Development risk assessment
    • DataAnalysisCore - Core data analysis methods
    • VisualizationFormatter - Chart formatting for dashboard

🔄 STEELCLAD EXTRACTION:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 MODULAR ARCHITECTURE
   └─ Source: predictive_analytics_engine.py (578 lines)
   └─ Target: 2 focused modules + streamlined main file
   └─ Status: EXTRACTION COMPLETE
"""

from .prediction_algorithms import (
    VelocityPredictor,
    QualityForecaster, 
    PatternAnalyzer,
    RiskAssessor
)
from .data_analysis_core import DataAnalysisCore, VisualizationFormatter

__all__ = [
    'VelocityPredictor',
    'QualityForecaster',
    'PatternAnalyzer', 
    'RiskAssessor',
    'DataAnalysisCore',
    'VisualizationFormatter'
]