"""
Predictive Code Intelligence Package
===================================

Revolutionary predictive code intelligence system with natural language integration.
Extracted from predictive_code_intelligence.py for enterprise modular architecture.

Agent D Implementation - Hour 15-16: Predictive Intelligence Modularization

Architecture:
- data_models.py: Core predictive intelligence data structures and enums
- evolution_predictor.py: Code evolution prediction and growth analysis
- language_bridge.py: Natural language code translation and explanation
- documentation_generator.py: AI-powered documentation generation
- security_analyzer.py: Predictive security vulnerability analysis
- intelligence_core.py: Master predictive intelligence coordination

Key Features:
- Predictive Code Evolution Analysis with mathematical modeling
- Natural Language Code Translation with bidirectional support
- AI-Powered Documentation Generation with quality metrics
- Security Vulnerability Prediction with proactive threat detection
- Maintenance Hotspot Identification with burden projection
- Performance Degradation Forecasting with complexity analysis
- Feature Addition Likelihood with pattern recognition
- Refactoring Need Prediction with pressure analysis

This is the FIRST AND ONLY predictive code intelligence system that can forecast
code evolution, generate natural language explanations, and create documentation
automatically - representing a breakthrough in AI-powered code analysis.
"""

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.data_models import (
    PredictionType,
    LanguageBridgeDirection,
    DocumentationType,
    PredictionConfidence,
    CodePrediction,
    NaturalLanguageTranslation,
    GeneratedDocumentation,
    CodeEvolutionAnalysis
)

from .evolution_predictor import (
    CodeEvolutionPredictor,
    create_evolution_predictor
)

from .language_bridge import (
    NaturalLanguageBridge,
    create_language_bridge
)

from .documentation_generator import (
    DocumentationGenerator,
    create_documentation_generator
)

from .security_analyzer import (
    SecurityAnalyzer,
    create_security_analyzer
)

from .intelligence_core import (
    PredictiveCodeIntelligence,
    create_predictive_code_intelligence
)

__all__ = [
    # Data Models
    'PredictionType',
    'LanguageBridgeDirection',
    'DocumentationType',
    'PredictionConfidence',
    'CodePrediction',
    'NaturalLanguageTranslation',
    'GeneratedDocumentation',
    'CodeEvolutionAnalysis',
    
    # Components
    'CodeEvolutionPredictor',
    'NaturalLanguageBridge',
    'DocumentationGenerator',
    'SecurityAnalyzer',
    'PredictiveCodeIntelligence',
    
    # Factory Functions
    'create_evolution_predictor',
    'create_language_bridge',
    'create_documentation_generator',
    'create_security_analyzer',
    'create_predictive_code_intelligence'
]

__version__ = "1.0.0"
__author__ = "Agent D - Analysis & Resource Management Specialist"
__description__ = "Revolutionary Predictive Code Intelligence - First AI system that predicts code evolution"