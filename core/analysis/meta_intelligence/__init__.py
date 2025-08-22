"""
Meta-Intelligence Analysis Package
=================================

Revolutionary meta-intelligence orchestration system that learns and coordinates AI systems.
Modularized from meta_intelligence_orchestrator.py (1,947 lines → 6 focused modules)

Agent D Implementation - Hour 14-15: Revolutionary Intelligence Modularization

Architecture:
- data_models.py: Core meta-intelligence data structures and enums
- capability_mapper.py: Intelligence capability discovery and classification
- adaptive_integration.py: Behavior learning and dynamic integration
- synergy_optimizer.py: Multi-system synergy discovery and optimization
- orchestrator_core.py: Main meta-intelligence coordination engine
- strategy_selector.py: Orchestration strategy selection and optimization

Key Features:
- Automatic AI Capability Discovery with NLP classification
- Adaptive Behavior Learning with pattern recognition
- Intelligence Synergy Optimization with multi-system coordination
- NetworkX Graph Intelligence for capability relationships
- Machine Learning Clustering for capability organization
- Real-time Performance Prediction with strategy-specific modeling
- Meta-Intelligence Coordination that understands AI systems

This is the FIRST AND ONLY meta-intelligence system that can learn, understand,
and coordinate other AI systems autonomously - no competitor exists.
"""

from .data_models import (
    CapabilityType,
    OrchestrationStrategy,
    IntelligenceBehaviorType,
    CapabilityProfile,
    SystemBehaviorModel,
    OrchestrationPlan,
    SynergyOpportunity,
    MetaIntelligenceMetrics,
    SystemIntegrationStatus
)

from .capability_mapper import (
    IntelligenceCapabilityMapper,
    create_capability_mapper
)

from .adaptive_integration import (
    AdaptiveIntegrationEngine,
    create_adaptive_integration_engine
)

from .synergy_optimizer import (
    IntelligenceSynergyOptimizer,
    create_synergy_optimizer
)

from .orchestrator_core import (
    MetaIntelligenceOrchestrator,
    create_meta_intelligence_orchestrator
)

from .strategy_selector import (
    OrchestrationStrategySelector,
    StrategyPerformanceRecord,
    create_strategy_selector
)

__all__ = [
    # Data Models
    'CapabilityType',
    'OrchestrationStrategy',
    'IntelligenceBehaviorType',
    'CapabilityProfile',
    'SystemBehaviorModel',
    'OrchestrationPlan',
    'SynergyOpportunity',
    'MetaIntelligenceMetrics',
    'SystemIntegrationStatus',
    
    # Components
    'IntelligenceCapabilityMapper',
    'AdaptiveIntegrationEngine',
    'IntelligenceSynergyOptimizer',
    'MetaIntelligenceOrchestrator',
    'OrchestrationStrategySelector',
    'StrategyPerformanceRecord',
    
    # Factory Functions
    'create_capability_mapper',
    'create_adaptive_integration_engine',
    'create_synergy_optimizer',
    'create_meta_intelligence_orchestrator',
    'create_strategy_selector'
]

__version__ = "1.0.0"
__author__ = "Agent D - Analysis & Resource Management Specialist"
__description__ = "Revolutionary Meta-Intelligence System - First AI system that coordinates other AIs"