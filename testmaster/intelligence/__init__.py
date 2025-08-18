"""
TestMaster Intelligence Layer

Provides intelligent test generation using advanced reasoning techniques.
Integrates Hierarchical Test Planning, Multi-Objective Optimization, and LLM management.
"""

# Import from hierarchical_planning which now provides the unified implementation
# with full backward compatibility for Tree-of-Thought names
from .hierarchical_planning import (
    UniversalToTTestGenerator,  # Backward compatibility alias
    ToTGenerationConfig,         # Backward compatibility alias
    ToTGenerationResult,         # Backward compatibility alias
    TreeOfThoughtReasoner,       # Preserved from original
    ReasoningStrategy,           # Preserved from original
    # New unified names (also available)
    UniversalHierarchicalTestGenerator,
    HierarchicalPlanningConfig,
    HierarchicalPlanningResult
)

from .optimization import (
    MultiObjectiveOptimizer,
    OptimizationObjective,
    NSGAIIOptimizer,
    CoverageObjective,
    QualityObjective,
    BalancedTestObjective
)

from .llm_providers import (
    LLMProviderManager,
    UniversalLLMProvider,
    LLMProviderConfig,
    OpenAIProvider,
    AnthropicProvider,
    LocalLLMProvider,
    ProviderOptimizer
)

__all__ = [
    # Hierarchical Test Planning (with backward compatibility)
    'UniversalToTTestGenerator',     # Backward compat
    'ToTGenerationConfig',           # Backward compat  
    'ToTGenerationResult',           # Backward compat
    'TreeOfThoughtReasoner',         # Preserved
    'ReasoningStrategy',             # Preserved
    'UniversalHierarchicalTestGenerator',  # New name
    'HierarchicalPlanningConfig',         # New name
    'HierarchicalPlanningResult',         # New name
    
    # Multi-Objective Optimization
    'MultiObjectiveOptimizer',
    'OptimizationObjective',
    'NSGAIIOptimizer',
    'CoverageObjective',
    'QualityObjective',
    'BalancedTestObjective',
    
    # LLM Providers
    'LLMProviderManager',
    'UniversalLLMProvider',
    'LLMProviderConfig',
    'OpenAIProvider',
    'AnthropicProvider',
    'LocalLLMProvider',
    'ProviderOptimizer'
]