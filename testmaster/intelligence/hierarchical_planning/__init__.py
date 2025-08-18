"""
Hierarchical Test Planning (HTP) System

Implements hierarchical reasoning for intelligent test generation.
Previously mislabeled as "Tree-of-Thought" - corrected as per roadmap specification.

This system provides systematic, multi-level test planning that breaks down
complex testing scenarios into manageable hierarchical components.

IMPORTANT: This module now provides unified integration with the Tree-of-Thought
implementation, preserving ALL functionality while providing the correct naming.
"""

# Import original HTP components
from .htp_reasoning import (
    HierarchicalTestPlanner,
    PlanningStrategy,
    PlanningNode,
    PlanningTree,
    EvaluationCriteria,
    get_hierarchical_planner
)

from .test_plan_generator import (
    TestPlanGenerator,
    TestPlanEvaluator,
    HierarchicalTestGenerator as _BaseHierarchicalTestGenerator
)

from .llm_integration import (
    LLMPoweredPlanGenerator,
    LLMPoweredPlanEvaluator,
    LLMPlanningConfig,
    create_llm_powered_planner
)

# Import unified integration (provides backward compatibility)
from .unified_integration import (
    # New unified classes
    UniversalHierarchicalTestGenerator,
    HierarchicalPlanningConfig,
    HierarchicalPlanningResult,
    
    # Backward compatibility aliases (CRITICAL for orchestrator)
    UniversalToTTestGenerator,
    ToTGenerationConfig,
    ToTGenerationResult,
    
    # Re-exported from tree_of_thought - ALL CLASSES!
    ThoughtNode,
    ThoughtTree,
    ThoughtGenerator,
    ThoughtEvaluator,
    TreeOfThoughtReasoner,
    ReasoningStrategy,
    EvaluationCriteria,
    SimpleThoughtGenerator,
    SimpleThoughtEvaluator,
    
    # Test-specific from tree_of_thought
    TestStrategyType,
    TestGenerationThought,
    TestStrategyThought,
    TestCoverageThought,
    TestThoughtGenerator,
    TestThoughtEvaluator,
    
    # Additional exports
    TestPlanLevel
)

# Make HierarchicalTestGenerator point to the unified version
HierarchicalTestGenerator = UniversalHierarchicalTestGenerator

__all__ = [
    # Original HTP components
    'HierarchicalTestPlanner',
    'PlanningStrategy', 
    'PlanningNode',
    'PlanningTree',
    'get_hierarchical_planner',
    'TestPlanGenerator',
    'TestPlanEvaluator',
    'LLMPoweredPlanGenerator',
    'LLMPoweredPlanEvaluator',
    'LLMPlanningConfig',
    'create_llm_powered_planner',
    
    # Unified components (new names)
    'HierarchicalTestGenerator',
    'UniversalHierarchicalTestGenerator',
    'HierarchicalPlanningConfig',
    'HierarchicalPlanningResult',
    
    # Backward compatibility (CRITICAL - used by orchestrator)
    'UniversalToTTestGenerator',
    'ToTGenerationConfig',
    'ToTGenerationResult',
    
    # Re-exported from tree_of_thought - ALL CLASSES (preserving functionality)
    'ThoughtNode',
    'ThoughtTree',
    'ThoughtGenerator',
    'ThoughtEvaluator',
    'TreeOfThoughtReasoner',
    'ReasoningStrategy',
    'EvaluationCriteria',
    'SimpleThoughtGenerator',
    'SimpleThoughtEvaluator',
    'TestStrategyType',
    'TestGenerationThought',
    'TestStrategyThought',
    'TestCoverageThought',
    'TestThoughtGenerator',
    'TestThoughtEvaluator',
    'TestPlanLevel'
]

# Convenience function to get the best planner based on available resources
def get_best_planner(prefer_llm: bool = True, config: LLMPlanningConfig = None):
    """Get the best available hierarchical test planner."""
    if prefer_llm:
        try:
            return create_llm_powered_planner(config)
        except Exception as e:
            print(f"LLM planner unavailable ({e}), falling back to standard planner")
    
    return get_hierarchical_planner()