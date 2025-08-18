"""
Tree-of-Thought (ToT) Universal Test Generation

Implements Tree-of-Thought reasoning for intelligent test generation.
Adapted from OpenAI Swarm, Agency Swarm, and PraisonAI's reasoning patterns.
"""

from .tot_reasoning import (
    ThoughtNode,
    ThoughtTree,
    ThoughtGenerator,  # Abstract base class
    ThoughtEvaluator,  # Abstract base class
    TreeOfThoughtReasoner,
    ReasoningStrategy,
    EvaluationCriteria,
    SimpleThoughtGenerator,  # Simple implementation
    SimpleThoughtEvaluator   # Simple implementation
)

from .test_thought_generator import (
    TestThoughtGenerator,
    TestThoughtEvaluator,  # Was missing!
    TestGenerationThought,
    TestStrategyThought,
    TestCoverageThought,
    TestStrategyType  # Was missing!
)

from .universal_tot_integration import (
    UniversalToTTestGenerator,
    ToTGenerationConfig,
    ToTGenerationResult
)

__all__ = [
    # Core ToT Components
    'ThoughtNode',
    'ThoughtTree',
    'ThoughtGenerator',
    'ThoughtEvaluator',
    'TreeOfThoughtReasoner',
    'ReasoningStrategy',
    'EvaluationCriteria',
    'SimpleThoughtGenerator',
    'SimpleThoughtEvaluator',
    
    # Test-specific ToT
    'TestThoughtGenerator',
    'TestThoughtEvaluator',
    'TestGenerationThought',
    'TestStrategyThought',
    'TestCoverageThought',
    'TestStrategyType',
    
    # Universal Integration
    'UniversalToTTestGenerator',
    'ToTGenerationConfig',
    'ToTGenerationResult'
]