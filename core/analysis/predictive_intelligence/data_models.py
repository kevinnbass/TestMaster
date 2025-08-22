"""
Predictive Intelligence Data Models
==================================

Core data structures for the revolutionary predictive code intelligence system.
Extracted from predictive_code_intelligence.py for enterprise modular architecture.

Agent D Implementation - Hour 15-16: Predictive Intelligence Modularization
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class PredictionType(Enum):
    """Types of predictive analysis"""
    CODE_EVOLUTION = "code_evolution"
    MAINTENANCE_HOTSPOTS = "maintenance_hotspots"
    SECURITY_VULNERABILITIES = "security_vulnerabilities"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FEATURE_ADDITIONS = "feature_additions"
    DEPENDENCY_CHANGES = "dependency_changes"
    REFACTORING_NEEDS = "refactoring_needs"
    TESTING_REQUIREMENTS = "testing_requirements"
    DOCUMENTATION_NEEDS = "documentation_needs"


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


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


@dataclass
class CodePrediction:
    """Represents a predictive analysis result"""
    prediction_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest())
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'prediction_id': self.prediction_id,
            'prediction_type': self.prediction_type.value,
            'target_file': self.target_file,
            'target_element': self.target_element,
            'prediction_summary': self.prediction_summary,
            'detailed_analysis': self.detailed_analysis,
            'confidence': self.confidence.value,
            'probability_score': self.probability_score,
            'timeline_estimate': self.timeline_estimate,
            'impact_assessment': self.impact_assessment,
            'recommended_actions': self.recommended_actions,
            'prevention_strategies': self.prevention_strategies,
            'monitoring_indicators': self.monitoring_indicators,
            'related_predictions': self.related_predictions,
            'evidence_factors': self.evidence_factors,
            'historical_patterns': self.historical_patterns,
            'prediction_timestamp': self.prediction_timestamp.isoformat(),
            'validation_metrics': self.validation_metrics
        }


@dataclass
class NaturalLanguageTranslation:
    """Represents code-language translation result"""
    translation_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    direction: LanguageBridgeDirection = LanguageBridgeDirection.CODE_TO_LANGUAGE
    source_code: str = ""
    natural_language: str = ""
    translation_quality: float = 0.0
    context_understanding: Dict[str, Any] = field(default_factory=dict)
    technical_terms: List[str] = field(default_factory=list)
    abstraction_level: str = ""
    target_audience: str = ""
    explanation_style: str = ""
    code_examples: List[str] = field(default_factory=list)
    alternative_explanations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'translation_id': self.translation_id,
            'direction': self.direction.value,
            'source_code': self.source_code,
            'natural_language': self.natural_language,
            'translation_quality': self.translation_quality,
            'context_understanding': self.context_understanding,
            'technical_terms': self.technical_terms,
            'abstraction_level': self.abstraction_level,
            'target_audience': self.target_audience,
            'explanation_style': self.explanation_style,
            'code_examples': self.code_examples,
            'alternative_explanations': self.alternative_explanations
        }


@dataclass
class GeneratedDocumentation:
    """Represents generated documentation"""
    documentation_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    documentation_type: DocumentationType = DocumentationType.FUNCTION_DOCSTRING
    target_element: str = ""
    generated_content: str = ""
    documentation_quality: float = 0.0
    completeness_score: float = 0.0
    clarity_score: float = 0.0
    technical_accuracy: float = 0.0
    includes_examples: bool = False
    includes_parameters: bool = False
    includes_return_values: bool = False
    includes_exceptions: bool = False
    style_consistency: float = 0.0
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'documentation_id': self.documentation_id,
            'documentation_type': self.documentation_type.value,
            'target_element': self.target_element,
            'generated_content': self.generated_content,
            'documentation_quality': self.documentation_quality,
            'completeness_score': self.completeness_score,
            'clarity_score': self.clarity_score,
            'technical_accuracy': self.technical_accuracy,
            'includes_examples': self.includes_examples,
            'includes_parameters': self.includes_parameters,
            'includes_return_values': self.includes_return_values,
            'includes_exceptions': self.includes_exceptions,
            'style_consistency': self.style_consistency,
            'generation_metadata': self.generation_metadata
        }


@dataclass
class CodeEvolutionAnalysis:
    """Analysis of how code might evolve over time"""
    analysis_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    current_state: Dict[str, Any] = field(default_factory=dict)
    evolution_vectors: List[Dict[str, Any]] = field(default_factory=list)
    growth_patterns: Dict[str, float] = field(default_factory=dict)
    complexity_trends: Dict[str, float] = field(default_factory=dict)
    dependency_evolution: Dict[str, List[str]] = field(default_factory=dict)
    feature_addition_likelihood: Dict[str, float] = field(default_factory=dict)
    refactoring_pressure: Dict[str, float] = field(default_factory=dict)
    maintenance_burden_projection: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'analysis_id': self.analysis_id,
            'current_state': self.current_state,
            'evolution_vectors': self.evolution_vectors,
            'growth_patterns': self.growth_patterns,
            'complexity_trends': self.complexity_trends,
            'dependency_evolution': self.dependency_evolution,
            'feature_addition_likelihood': self.feature_addition_likelihood,
            'refactoring_pressure': self.refactoring_pressure,
            'maintenance_burden_projection': self.maintenance_burden_projection
        }


@dataclass
class SecurityRisk:
    """Represents a security risk identified in code"""
    risk_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    risk_type: str = ""
    severity: str = ""
    description: str = ""
    file_path: str = ""
    line_number: Optional[int] = None
    code_snippet: str = ""
    impact_assessment: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    detection_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'risk_id': self.risk_id,
            'risk_type': self.risk_type,
            'severity': self.severity,
            'description': self.description,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'code_snippet': self.code_snippet,
            'impact_assessment': self.impact_assessment,
            'remediation_steps': self.remediation_steps,
            'confidence_score': self.confidence_score,
            'detection_timestamp': self.detection_timestamp.isoformat()
        }


@dataclass
class EvolutionVector:
    """Represents a direction of code evolution"""
    vector_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    vector_type: str = ""
    target_element: str = ""
    likelihood: float = 0.0
    direction: str = ""
    impact_magnitude: float = 0.0
    time_horizon: str = ""
    driving_factors: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'vector_id': self.vector_id,
            'vector_type': self.vector_type,
            'target_element': self.target_element,
            'likelihood': self.likelihood,
            'direction': self.direction,
            'impact_magnitude': self.impact_magnitude,
            'time_horizon': self.time_horizon,
            'driving_factors': self.driving_factors,
            'prerequisites': self.prerequisites
        }


@dataclass
class PredictionValidation:
    """Validation metrics for prediction accuracy"""
    validation_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    prediction_id: str = ""
    actual_outcome: bool = False
    accuracy_score: float = 0.0
    timeline_accuracy: float = 0.0
    confidence_calibration: float = 0.0
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validation_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'validation_id': self.validation_id,
            'prediction_id': self.prediction_id,
            'actual_outcome': self.actual_outcome,
            'accuracy_score': self.accuracy_score,
            'timeline_accuracy': self.timeline_accuracy,
            'confidence_calibration': self.confidence_calibration,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'validation_notes': self.validation_notes
        }


@dataclass
class PredictiveMetrics:
    """Metrics for predictive intelligence performance"""
    total_predictions: int = 0
    accurate_predictions: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    average_confidence: float = 0.0
    prediction_distribution: Dict[str, int] = field(default_factory=dict)
    accuracy_by_type: Dict[str, float] = field(default_factory=dict)
    timeline_accuracy: Dict[str, float] = field(default_factory=dict)
    confidence_calibration: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def overall_accuracy(self) -> float:
        """Calculate overall prediction accuracy"""
        total = self.accurate_predictions + self.false_positives + self.false_negatives
        return self.accurate_predictions / max(1, total)
    
    def precision(self) -> float:
        """Calculate prediction precision"""
        total_positive = self.accurate_predictions + self.false_positives
        return self.accurate_predictions / max(1, total_positive)
    
    def recall(self) -> float:
        """Calculate prediction recall"""
        total_actual = self.accurate_predictions + self.false_negatives
        return self.accurate_predictions / max(1, total_actual)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'total_predictions': self.total_predictions,
            'accurate_predictions': self.accurate_predictions,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'overall_accuracy': self.overall_accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'average_confidence': self.average_confidence,
            'prediction_distribution': self.prediction_distribution,
            'accuracy_by_type': self.accuracy_by_type,
            'timeline_accuracy': self.timeline_accuracy,
            'confidence_calibration': self.confidence_calibration,
            'last_updated': self.last_updated.isoformat()
        }