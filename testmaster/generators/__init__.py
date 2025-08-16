"""
TestMaster Test Generators

Intelligent test generation engines with AI-powered capabilities.
"""

# Base classes
from .base import (
    BaseGenerator, 
    AnalysisBasedGenerator, 
    TemplateBasedGenerator,
    ModuleAnalysis,
    GenerationConfig
)

# Generator implementations
from .intelligent import IntelligentTestGenerator

# Legacy compatibility aliases for existing scripts
IntelligentTestBuilder = IntelligentTestGenerator
IntelligentTestBuilderV2 = IntelligentTestGenerator
OfflineIntelligentTestBuilder = IntelligentTestGenerator

# Placeholder imports for future implementation
try:
    from .context_aware import ContextAwareTestGenerator
except ImportError:
    ContextAwareTestGenerator = IntelligentTestGenerator

try:
    from .specialized import SpecializedTestGenerator
except ImportError:
    SpecializedTestGenerator = IntelligentTestGenerator

try:
    from .integration import IntegrationTestGenerator
except ImportError:
    IntegrationTestGenerator = IntelligentTestGenerator

__all__ = [
    "BaseGenerator",
    "AnalysisBasedGenerator", 
    "TemplateBasedGenerator",
    "ModuleAnalysis",
    "GenerationConfig",
    "IntelligentTestGenerator",
    "IntelligentTestBuilder",
    "IntelligentTestBuilderV2",
    "OfflineIntelligentTestBuilder",
    "ContextAwareTestGenerator",
    "SpecializedTestGenerator",
    "IntegrationTestGenerator"
]