"""
Architectural Evolution Predictor Package
Modularized from architectural_evolution_predictor.py

This package provides comprehensive architectural evolution prediction capabilities
through focused, modular components.

Components:
- types_and_structures: Core enums and data structures
- growth_modeler: System growth pattern analysis and modeling  
- scalability_forecaster: Scalability forecasting and capacity planning
- technology_analyzer: Technology trend analysis and impact assessment
- predictor: Main architectural evolution predictor

All original functionality has been preserved while achieving perfect modularization.
"""

from .types_and_structures import (
    ArchitecturalPattern,
    ScalingPattern, 
    TechnologyTrend,
    EvolutionProbability,
    ArchitecturalMetrics,
    SystemGrowthPattern,
    ScalabilityForecast,
    TechnologyEvolutionAnalysis,
    ArchitecturalEvolutionPrediction
)

from .growth_modeler import SystemGrowthModeler

# Import factory function to maintain compatibility
def create_architectural_evolution_predictor():
    """Factory function to create an ArchitecturalEvolutionPredictor instance"""
    from .predictor import ArchitecturalEvolutionPredictor
    return ArchitecturalEvolutionPredictor()

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
    'create_architectural_evolution_predictor'
]