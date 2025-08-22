"""
Pattern Intelligence Analysis Package
====================================

Revolutionary AI-powered architectural decision-making system with graph theory integration.
Modularized from architectural_decision_engine.py (2,388 lines → 7 focused modules)

Agent D Implementation - Hour 12-13: Revolutionary Intelligence Modularization

Architecture:
- data_models.py: Core decision science data structures and enums
- decision_scorer.py: Multi-criteria decision analysis with weighted scoring
- pattern_evolution.py: Design pattern evolution engine with NetworkX graph analysis
- performance_optimizer.py: Performance architecture optimization with gap analysis
- microservice_analyzer.py: Microservice boundary analysis and migration strategies
- decision_engine_core.py: Main architectural decision orchestration
- implementation_planner.py: Implementation planning with success metrics

Key Features:
- Multi-Criteria Decision Analysis with 12 weighted criteria
- Design Pattern Evolution using NetworkX graph algorithms
- Performance Architecture Optimization with predictive modeling
- Microservice Boundary Analysis with coupling detection
- AI-Powered Decision Making with confidence scoring
- Architectural Pattern Recognition (11 patterns: microservices, monolith, etc.)
- Implementation Planning with timeline estimation and risk assessment
"""

from .data_models import (
    DecisionType,
    DecisionCriteria,
    ArchitecturalPattern,
    ArchitecturalOption,
    DecisionContext,
    DecisionAnalysis,
    PatternEvolution,
    PerformanceMetrics,
    ImplementationPlan
)

from .decision_scorer import (
    DecisionScorer,
    create_decision_scorer
)

from .pattern_evolution import (
    DesignPatternEvolutionEngine,
    create_pattern_evolution_engine
)

from .performance_optimizer import (
    PerformanceArchitectureOptimizer,
    create_performance_optimizer
)

from .microservice_analyzer import (
    MicroserviceEvolutionAnalyzer,
    create_microservice_analyzer
)

from .decision_engine_core import (
    ArchitecturalDecisionEngine,
    create_architectural_decision_engine
)

from .implementation_planner import (
    ImplementationPlanner,
    create_implementation_planner
)

__all__ = [
    # Data Models
    'DecisionType',
    'DecisionCriteria',
    'ArchitecturalPattern',
    'ArchitecturalOption',
    'DecisionContext',
    'DecisionAnalysis',
    'PatternEvolution',
    'PerformanceMetrics',
    'ImplementationPlan',
    
    # Components
    'DecisionScorer',
    'DesignPatternEvolutionEngine',
    'PerformanceArchitectureOptimizer',
    'MicroserviceEvolutionAnalyzer',
    'ArchitecturalDecisionEngine',
    'ImplementationPlanner',
    
    # Factory Functions
    'create_decision_scorer',
    'create_pattern_evolution_engine',
    'create_performance_optimizer',
    'create_microservice_analyzer',
    'create_architectural_decision_engine',
    'create_implementation_planner'
]

__version__ = "1.0.0"
__author__ = "Agent D - Analysis & Resource Management Specialist"
__description__ = "Revolutionary Architectural Decision Intelligence - First AI-powered decision system on Earth"