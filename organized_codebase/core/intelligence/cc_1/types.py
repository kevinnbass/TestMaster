"""
Predictive Types and Data Structures
====================================

Core type definitions and data structures for the Predictive Code Intelligence Engine.
Provides enterprise-grade type safety for code prediction, natural language translation,
and intelligent documentation generation with comprehensive prediction modeling.

This module contains all Enum definitions and dataclass structures used throughout
the predictive intelligence system, implementing advanced prediction patterns.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: predictive_types.py (160 lines)
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib


class PredictionType(Enum):
    """Types of predictive analysis for code intelligence"""
    CODE_EVOLUTION = "code_evolution"
    MAINTENANCE_HOTSPOTS = "maintenance_hotspots"
    SECURITY_VULNERABILITIES = "security_vulnerabilities"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FEATURE_ADDITIONS = "feature_additions"
    DEPENDENCY_CHANGES = "dependency_changes"
    REFACTORING_NEEDS = "refactoring_needs"
    TESTING_REQUIREMENTS = "testing_requirements"
    DOCUMENTATION_NEEDS = "documentation_needs"
    ARCHITECTURE_EVOLUTION = "architecture_evolution"
    API_CHANGES = "api_changes"
    COMPATIBILITY_ISSUES = "compatibility_issues"


class LanguageBridgeDirection(Enum):
    """Direction of code-language translation"""
    CODE_TO_LANGUAGE = "code_to_language"
    LANGUAGE_TO_CODE = "language_to_code"
    BIDIRECTIONAL = "bidirectional"


class DocumentationType(Enum):
    """Types of generated documentation"""
    FUNCTION_DOCSTRING = "function_docstring"
    CLASS_DOCSTRING = "class_docstring"
    MODULE_DOCSTRING = "module_docstring"
    API_DOCUMENTATION = "api_documentation"
    INLINE_COMMENTS = "inline_comments"
    README_SECTION = "readme_section"
    ARCHITECTURE_DOCUMENTATION = "architecture_documentation"
    USER_GUIDE = "user_guide"
    TUTORIAL = "tutorial"
    TECHNICAL_SPECIFICATION = "technical_specification"


class PredictionConfidence(Enum):
    """Confidence levels for predictions with probability ranges"""
    VERY_HIGH = "very_high"      # 90-100%
    HIGH = "high"                # 75-89%
    MEDIUM = "medium"            # 50-74%
    LOW = "low"                  # 25-49%
    SPECULATIVE = "speculative"  # 0-24%


class CodeComplexityLevel(Enum):
    """Code complexity levels for analysis"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREMELY_HIGH = "extremely_high"


@dataclass
class CodePrediction:
    """Comprehensive predictive analysis result with detailed insights"""
    prediction_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])
    prediction_type: PredictionType = PredictionType.CODE_EVOLUTION
    target_file: str = ""
    target_element: str = ""
    prediction_summary: str = ""
    detailed_analysis: str = ""
    confidence: PredictionConfidence = PredictionConfidence.MEDIUM
    probability_score: float = 0.0
    timeline_estimate: str = ""
    impact_assessment: Dict[str, str] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    prevention_strategies: List[str] = field(default_factory=list)
    monitoring_indicators: List[str] = field(default_factory=list)
    related_predictions: List[str] = field(default_factory=list)
    evidence_factors: List[str] = field(default_factory=list)
    historical_patterns: List[str] = field(default_factory=list)
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class NaturalLanguageTranslation:
    """Natural language translation of code with context understanding"""
    translation_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])
    direction: LanguageBridgeDirection = LanguageBridgeDirection.CODE_TO_LANGUAGE
    source_code: str = ""
    natural_language: str = ""
    target_audience: str = "general"
    abstraction_level: str = "medium"
    technical_terms: List[str] = field(default_factory=list)
    translation_quality: float = 0.0
    context_understanding: Dict[str, Any] = field(default_factory=dict)
    code_complexity: CodeComplexityLevel = CodeComplexityLevel.MEDIUM
    explanation_depth: str = "standard"
    translation_timestamp: datetime = field(default_factory=datetime.now)
    validation_score: float = 0.0
    user_feedback: Optional[Dict[str, Any]] = None


@dataclass
class DocumentationGeneration:
    """Generated documentation with quality metrics"""
    documentation_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])
    documentation_type: DocumentationType = DocumentationType.FUNCTION_DOCSTRING
    target_element: str = ""
    generated_content: str = ""
    content_quality: float = 0.0
    completeness_score: float = 0.0
    clarity_score: float = 0.0
    technical_accuracy: float = 0.0
    formatting_compliance: bool = False
    generation_method: str = "ai_generated"
    generation_timestamp: datetime = field(default_factory=datetime.now)
    review_status: str = "pending"
    improvements_suggested: List[str] = field(default_factory=list)
    usage_examples: List[str] = field(default_factory=list)


@dataclass
class CodeEvolutionPattern:
    """Pattern of code evolution over time"""
    pattern_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])
    pattern_name: str = ""
    description: str = ""
    frequency: float = 0.0
    growth_rate: float = 0.0
    complexity_trend: str = "stable"
    change_indicators: List[str] = field(default_factory=list)
    historical_occurrences: List[datetime] = field(default_factory=list)
    prediction_accuracy: float = 0.0
    pattern_strength: float = 0.0
    related_patterns: List[str] = field(default_factory=list)
    impact_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MaintenanceBurdenAnalysis:
    """Analysis of maintenance burden and future requirements"""
    analysis_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])
    target_component: str = ""
    current_burden_score: float = 0.0
    projected_burden_score: float = 0.0
    burden_growth_rate: float = 0.0
    complexity_factors: Dict[str, float] = field(default_factory=dict)
    maintenance_hotspots: List[str] = field(default_factory=list)
    effort_estimates: Dict[str, float] = field(default_factory=dict)
    optimization_opportunities: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    timeline_projections: Dict[str, str] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictiveMetrics:
    """Metrics for evaluating predictive accuracy and performance"""
    metrics_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])
    prediction_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confidence_calibration: float = 0.0
    temporal_accuracy: float = 0.0
    model_performance: Dict[str, float] = field(default_factory=dict)
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    improvement_trends: Dict[str, float] = field(default_factory=dict)
    benchmark_comparisons: Dict[str, float] = field(default_factory=dict)


@dataclass
class CodeIntelligenceContext:
    """Context for predictive code intelligence operations"""
    context_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])
    project_name: str = ""
    codebase_size: int = 0
    programming_languages: List[str] = field(default_factory=list)
    frameworks_used: List[str] = field(default_factory=list)
    development_stage: str = "active"
    team_size: int = 1
    project_age: timedelta = field(default_factory=lambda: timedelta(days=0))
    change_frequency: float = 0.0
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    historical_data_available: bool = False
    prediction_scope: str = "full_project"
    analysis_preferences: Dict[str, Any] = field(default_factory=dict)


# Export all predictive types
__all__ = [
    'PredictionType', 'LanguageBridgeDirection', 'DocumentationType', 
    'PredictionConfidence', 'CodeComplexityLevel',
    'CodePrediction', 'NaturalLanguageTranslation', 'DocumentationGeneration',
    'CodeEvolutionPattern', 'MaintenanceBurdenAnalysis', 'PredictiveMetrics',
    'CodeIntelligenceContext'
]