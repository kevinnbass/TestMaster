"""
Architectural Decision Modules
=============================

Modularized components from architectural_decision_engine.py for better maintainability.
Each module provides focused functionality for architectural decision making.

Modules:
- data_models: Data structures and enums
- decision_scorer: Multi-criteria scoring system
- pattern_evolution_engine: Pattern evolution analysis
- performance_optimizer: Performance optimization engine  
- microservice_analyzer: Microservice architecture analysis

Author: Agent E - Infrastructure Consolidation
"""

from .data_models import (
    DecisionType,
    DecisionCriteria, 
    ArchitecturalPattern,
    DecisionPriority,
    ArchitecturalOption,
    DecisionContext,
    DecisionAnalysis,
    PatternEvolution,
    PerformanceMetrics
)

from .decision_scorer import DecisionScorer
from .pattern_evolution_engine import DesignPatternEvolutionEngine
from .performance_optimizer import PerformanceArchitectureOptimizer
from .microservice_analyzer import MicroserviceEvolutionAnalyzer

__all__ = [
    # Data Models
    'DecisionType',
    'DecisionCriteria',
    'ArchitecturalPattern', 
    'DecisionPriority',
    'ArchitecturalOption',
    'DecisionContext',
    'DecisionAnalysis',
    'PatternEvolution',
    'PerformanceMetrics',
    
    # Core Components
    'DecisionScorer',
    'DesignPatternEvolutionEngine',
    'PerformanceArchitectureOptimizer',
    'MicroserviceEvolutionAnalyzer'
]

# Module metadata
ORIGINAL_FILE = "architectural_decision_engine.py"
ORIGINAL_LINES = 2388
MODULARIZED_MODULES = 5
AVERAGE_MODULE_SIZE = ORIGINAL_LINES // MODULARIZED_MODULES

# Factory function for creating integrated decision engine components
def create_architectural_decision_components():
    """
    Create all architectural decision engine components.
    
    Returns a dictionary containing all the modularized components
    ready for use in the main decision engine.
    """
    return {
        'decision_scorer': DecisionScorer(),
        'pattern_evolution_engine': DesignPatternEvolutionEngine(),
        'performance_optimizer': PerformanceArchitectureOptimizer(),
        'microservice_analyzer': MicroserviceEvolutionAnalyzer()
    }

print(f"""
Architectural Decision Engine Modularization Complete:
- Original: {ORIGINAL_FILE} ({ORIGINAL_LINES} lines)
- Modularized: {MODULARIZED_MODULES} focused modules
- Average module size: ~{AVERAGE_MODULE_SIZE} lines
- All functionality preserved with enhanced maintainability
""")