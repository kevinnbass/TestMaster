"""
Hierarchical Test Planning (HTP) System

Implements hierarchical reasoning for intelligent test generation.
Previously mislabeled as "Tree-of-Thought" - corrected as per roadmap specification.

This system provides systematic, multi-level test planning that breaks down
complex testing scenarios into manageable hierarchical components.
"""

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
    HierarchicalTestGenerator
)

from .llm_integration import (
    LLMPoweredPlanGenerator,
    LLMPoweredPlanEvaluator,
    LLMPlanningConfig,
    create_llm_powered_planner
)

__all__ = [
    'HierarchicalTestPlanner',
    'PlanningStrategy', 
    'PlanningNode',
    'PlanningTree',
    'EvaluationCriteria',
    'get_hierarchical_planner',
    'TestPlanGenerator',
    'TestPlanEvaluator',
    'HierarchicalTestGenerator',
    'LLMPoweredPlanGenerator',
    'LLMPoweredPlanEvaluator',
    'LLMPlanningConfig',
    'create_llm_powered_planner'
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