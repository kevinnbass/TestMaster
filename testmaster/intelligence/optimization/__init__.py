"""
Multi-Objective Universal Optimization for TestMaster

Provides optimization algorithms for balancing multiple testing objectives.
Adapted from Agency Swarm and PraisonAI's optimization patterns.
"""

from .multi_objective_optimizer import (
    MultiObjectiveOptimizer,
    OptimizationObjective,
    ObjectiveType,
    OptimizationConfig,
    OptimizationResult,
    ParetoFront,
    Solution
)

from .test_optimization_objectives import (
    CoverageObjective,
    PerformanceObjective,
    QualityObjective,
    SecurityObjective,
    MaintainabilityObjective,
    CostObjective,
    BalancedTestObjective
)

from .optimization_algorithms import (
    NSGAIIOptimizer,
    MOEADOptimizer,
    ParticleSwarmOptimizer,
    SimulatedAnnealingOptimizer,
    GeneticAlgorithmOptimizer
)

__all__ = [
    # Core Optimization
    'MultiObjectiveOptimizer',
    'OptimizationObjective',
    'ObjectiveType',
    'OptimizationConfig',
    'OptimizationResult',
    'ParetoFront',
    'Solution',
    
    # Test Objectives
    'CoverageObjective',
    'PerformanceObjective',
    'QualityObjective',
    'SecurityObjective',
    'MaintainabilityObjective',
    'CostObjective',
    'BalancedTestObjective',
    
    # Algorithms
    'NSGAIIOptimizer',
    'MOEADOptimizer',
    'ParticleSwarmOptimizer',
    'SimulatedAnnealingOptimizer',
    'GeneticAlgorithmOptimizer'
]