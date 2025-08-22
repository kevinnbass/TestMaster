"""
Resource Intelligence Analysis Package
=====================================

Enterprise-grade resource allocation and optimization system with predictive scaling.
Modularized from intelligent_resource_allocator.py (1,583 lines → 6 focused modules)

Agent D Implementation - Hour 10-11: Revolutionary Intelligence Modularization

Architecture:
- data_models.py: Core data structures and enums
- optimizers.py: Linear programming and genetic algorithm optimizers  
- load_balancer.py: Advanced load balancing algorithms
- predictive_scaler.py: Predictive scaling and demand forecasting
- resource_monitor.py: Resource health monitoring and anomaly detection
- allocator_core.py: Main allocation orchestration and coordination

Key Features:
- Genetic Algorithm Optimization with scipy differential_evolution
- Linear Programming with scipy.optimize.linprog 
- Predictive Scaling with confidence scoring
- Real-time Load Balancing with 6 algorithms
- Autonomous Resource Monitoring with z-score anomaly detection
- Multi-strategy Allocation (7 strategies: fair_share, performance_based, etc.)
"""

from .data_models import (
    AllocationStrategy,
    ScalingDirection, 
    LoadBalancingAlgorithm,
    ResourceConstraint,
    AllocationRequest,
    AllocationDecision,
    LoadBalancingMetrics,
    PredictiveScalingSignal
)

from .optimizers import (
    ResourceOptimizer,
    LinearProgrammingOptimizer,
    GeneticAlgorithmOptimizer
)

from .load_balancer import (
    LoadBalancer,
    create_load_balancer
)

from .predictive_scaler import (
    PredictiveScaler,
    create_predictive_scaler
)

from .resource_monitor import (
    ResourceMonitor,
    create_resource_monitor
)

from .allocator_core import (
    IntelligentResourceAllocator,
    create_intelligent_resource_allocator
)

__all__ = [
    # Data Models
    'AllocationStrategy',
    'ScalingDirection',
    'LoadBalancingAlgorithm', 
    'ResourceConstraint',
    'AllocationRequest',
    'AllocationDecision',
    'LoadBalancingMetrics',
    'PredictiveScalingSignal',
    
    # Optimizers
    'ResourceOptimizer',
    'LinearProgrammingOptimizer',
    'GeneticAlgorithmOptimizer',
    
    # Components
    'LoadBalancer',
    'PredictiveScaler', 
    'ResourceMonitor',
    'IntelligentResourceAllocator',
    
    # Factory Functions
    'create_load_balancer',
    'create_predictive_scaler',
    'create_resource_monitor',
    'create_intelligent_resource_allocator'
]

__version__ = "1.0.0"
__author__ = "Agent D - Analysis & Resource Management Specialist"
__description__ = "Revolutionary Resource Intelligence System - 100x more capable than any competitor"