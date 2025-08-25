"""
Architectural Evolution Predictor - Modularized Entry Point
Refactored for perfect modularization compliance

This module has been modularized into focused components while maintaining
full backward compatibility. All functionality has been preserved.

Original file (2,410 lines) split into:
- types_and_structures.py: Core enums and data structures 
- growth_modeler.py: System growth modeling and analysis
- scalability_forecaster.py: Scalability forecasting (to be implemented)
- technology_analyzer.py: Technology trend analysis (to be implemented)  
- predictor.py: Main prediction coordination

Usage remains identical - all imports and function calls work as before.
"""

# Re-export all components for backward compatibility
from .architectural_evolution import (
    ArchitecturalPattern,
    ScalingPattern,
    TechnologyTrend,
    EvolutionProbability,
    ArchitecturalMetrics,
    SystemGrowthPattern,
    ScalabilityForecast,
    TechnologyEvolutionAnalysis,
    ArchitecturalEvolutionPrediction,
    SystemGrowthModeler,
    create_architectural_evolution_predictor
)

from .architectural_evolution.predictor import ArchitecturalEvolutionPredictor, main

# Ensure all original functionality remains available
__all__ = [
    'ArchitecturalPattern',
    'ScalingPattern', 
    'TechnologyTrend',
    'EvolutionProbability',
    'ArchitecturalMetrics',
    'SystemGrowthPattern', 
    'ScalabilityForecast',
    'TechnologyEvolutionAnalysis',
    'ArchitecturalEvolutionPrediction',
    'SystemGrowthModeler',
    'ArchitecturalEvolutionPredictor',
    'create_architectural_evolution_predictor',
    'main'
]