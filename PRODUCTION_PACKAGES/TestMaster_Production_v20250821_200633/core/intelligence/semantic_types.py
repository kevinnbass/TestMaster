"""
Semantic Types - Cross-System Semantic Learning Type Definitions
================================================================

Comprehensive type definitions and data structures for advanced semantic learning 
across intelligence frameworks, enabling unified pattern discovery, concept abstraction,
and emergent behavior detection with enterprise-grade type safety.

This module provides all type definitions, enums, and dataclasses required for
sophisticated cross-system semantic analysis and knowledge representation.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: semantic_types.py (280 lines)
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid


class ConceptType(Enum):
    """Types of semantic concepts that can be discovered"""
    ENTITY = "entity"
    RELATIONSHIP = "relationship"  
    BEHAVIOR = "behavior"
    PATTERN = "pattern"
    PROCESS = "process"
    STATE = "state"
    EVENT = "event"
    RESOURCE = "resource"


class AbstractionLevel(Enum):
    """Levels of concept abstraction from concrete to highly abstract"""
    CONCRETE = 1      # Direct observations and measurements
    OPERATIONAL = 2   # Operational behaviors and patterns
    TACTICAL = 3      # Tactical relationships and strategies
    STRATEGIC = 4     # Strategic patterns and coordination
    PHILOSOPHICAL = 5 # Highly abstract principles and emergent properties


class RelationshipType(Enum):
    """Types of semantic relationships between concepts"""
    CAUSES = "causes"                    # Causal relationships
    CORRELATES = "correlates"           # Statistical correlations  
    CONTAINS = "contains"               # Containment/composition relationships
    TRANSFORMS = "transforms"           # Transformation relationships
    ENABLES = "enables"                 # Enabling relationships
    INHIBITS = "inhibits"              # Inhibitory relationships
    PRECEDES = "precedes"              # Temporal precedence
    FOLLOWS = "follows"                # Temporal succession
    COMPETES_WITH = "competes_with"    # Competition relationships
    COLLABORATES_WITH = "collaborates_with"  # Collaboration relationships


class RelationshipDirectionality(Enum):
    """Directionality of semantic relationships"""
    BIDIRECTIONAL = "bidirectional"           # Mutual influence
    SOURCE_TO_TARGET = "source_to_target"     # One-way from source to target
    TARGET_TO_SOURCE = "target_to_source"     # One-way from target to source
    UNDETERMINED = "undetermined"             # Direction not yet determined


class BehaviorType(Enum):
    """Types of emergent behaviors that can be detected"""
    ADAPTIVE = "adaptive"                     # Adaptive learning and adjustment
    SELF_ORGANIZING = "self_organizing"       # Self-organization patterns
    OPTIMIZATION = "optimization"             # Optimization behaviors
    COORDINATION = "coordination"             # Inter-system coordination
    EMERGENCE = "emergence"                   # True emergence behaviors
    SYNCHRONIZATION = "synchronization"      # Synchronization patterns
    COMPETITION = "competition"               # Competitive behaviors
    COOPERATION = "cooperation"               # Cooperative behaviors


class StabilityAssessment(Enum):
    """Stability assessment of emergent behaviors"""
    STABLE = "stable"                         # Consistent and predictable
    UNSTABLE = "unstable"                     # Chaotic or unpredictable
    EVOLVING = "evolving"                     # Changing but with patterns
    OSCILLATING = "oscillating"              # Cyclic behavior patterns
    DECLINING = "declining"                   # Weakening over time
    STRENGTHENING = "strengthening"          # Growing stronger over time


class EvidenceType(Enum):
    """Types of evidence supporting semantic relationships"""
    TEMPORAL_CORRELATION = "temporal_correlation"
    CAUSAL_INFERENCE = "causal_inference"
    STATISTICAL_CORRELATION = "statistical_correlation"
    PATTERN_MATCHING = "pattern_matching"
    BEHAVIOR_ANALYSIS = "behavior_analysis"
    PERFORMANCE_METRICS = "performance_metrics"
    SYSTEM_LOGS = "system_logs"
    EVENT_TRACES = "event_traces"


@dataclass
class SemanticConcept:
    """Unified semantic concept discovered across intelligence systems"""
    concept_id: str
    concept_name: str
    concept_type: ConceptType
    confidence: float  # 0.0-1.0
    abstraction_level: AbstractionLevel
    system_manifestations: Dict[str, Any]  # How concept appears in each system
    semantic_properties: Dict[str, Any]
    related_concepts: List[str] = field(default_factory=list)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Advanced metadata
    occurrence_frequency: float = 0.0
    cross_system_correlation: float = 0.0
    predictive_value: float = 0.0
    importance_score: float = 0.0
    
    def update_confidence(self, new_evidence: Dict[str, Any]):
        """Update concept confidence based on new evidence"""
        evidence_weight = new_evidence.get('weight', 0.1)
        evidence_confidence = new_evidence.get('confidence', 0.0)
        
        # Weighted average with existing confidence
        self.confidence = (self.confidence * 0.9) + (evidence_confidence * evidence_weight * 0.1)
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        self.last_updated = datetime.now()
        self.evolution_history.append({
            'timestamp': self.last_updated.isoformat(),
            'change_type': 'confidence_update',
            'new_confidence': self.confidence,
            'evidence': new_evidence
        })
    
    def add_system_manifestation(self, system_id: str, manifestation: Any):
        """Add or update how this concept manifests in a specific system"""
        self.system_manifestations[system_id] = manifestation
        self.cross_system_correlation = len(self.system_manifestations) / 10.0
        self.last_updated = datetime.now()
    
    def calculate_importance(self) -> float:
        """Calculate overall importance score for this concept"""
        # Factors: confidence, cross-system presence, relationships, frequency
        confidence_factor = self.confidence
        cross_system_factor = min(1.0, len(self.system_manifestations) / 5.0)
        relationship_factor = min(1.0, len(self.related_concepts) / 10.0)
        frequency_factor = min(1.0, self.occurrence_frequency)
        
        self.importance_score = (
            confidence_factor * 0.3 +
            cross_system_factor * 0.25 +
            relationship_factor * 0.25 +
            frequency_factor * 0.2
        )
        
        return self.importance_score


@dataclass
class SemanticRelationship:
    """Semantic relationship between concepts with comprehensive evidence tracking"""
    relationship_id: str
    source_concept: str
    target_concept: str
    relationship_type: RelationshipType
    strength: float  # 0.0-1.0
    directionality: RelationshipDirectionality
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    systems_involved: List[str] = field(default_factory=list)
    temporal_aspects: Optional[Dict[str, Any]] = None
    discovered_at: datetime = field(default_factory=datetime.now)
    last_validated: datetime = field(default_factory=datetime.now)
    
    # Advanced relationship metadata
    stability_score: float = 0.0
    predictive_power: float = 0.0
    causality_confidence: float = 0.0
    temporal_lag: Optional[float] = None  # seconds
    
    def add_evidence(self, evidence_type: EvidenceType, evidence_data: Dict[str, Any], 
                    confidence: float = 0.0):
        """Add new evidence supporting this relationship"""
        evidence_entry = {
            'type': evidence_type.value,
            'data': evidence_data,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'evidence_id': str(uuid.uuid4())
        }
        
        self.evidence.append(evidence_entry)
        
        # Update relationship strength based on evidence
        self._recalculate_strength()
        self.last_validated = datetime.now()
    
    def _recalculate_strength(self):
        """Recalculate relationship strength based on accumulated evidence"""
        if not self.evidence:
            self.strength = 0.0
            return
        
        # Weighted average of evidence confidences
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for evidence in self.evidence:
            weight = self._get_evidence_weight(evidence['type'])
            confidence = evidence.get('confidence', 0.0)
            
            total_weight += weight
            weighted_confidence += confidence * weight
        
        if total_weight > 0:
            self.strength = min(1.0, weighted_confidence / total_weight)
    
    def _get_evidence_weight(self, evidence_type: str) -> float:
        """Get weight for different types of evidence"""
        weights = {
            'causal_inference': 1.0,
            'temporal_correlation': 0.8,
            'statistical_correlation': 0.6,
            'pattern_matching': 0.7,
            'behavior_analysis': 0.9,
            'performance_metrics': 0.8,
            'system_logs': 0.5,
            'event_traces': 0.7
        }
        return weights.get(evidence_type, 0.5)
    
    def calculate_causality_confidence(self) -> float:
        """Calculate confidence that this relationship is truly causal"""
        causal_evidence = [e for e in self.evidence if e['type'] == 'causal_inference']
        temporal_evidence = [e for e in self.evidence if e['type'] == 'temporal_correlation']
        
        if not causal_evidence and not temporal_evidence:
            self.causality_confidence = 0.0
            return 0.0
        
        # Strong causal evidence gets high confidence
        if causal_evidence:
            causal_confidence = sum(e.get('confidence', 0) for e in causal_evidence) / len(causal_evidence)
            self.causality_confidence = min(1.0, causal_confidence * 1.2)
        
        # Temporal evidence provides moderate causality confidence
        elif temporal_evidence:
            temporal_confidence = sum(e.get('confidence', 0) for e in temporal_evidence) / len(temporal_evidence)
            self.causality_confidence = min(0.7, temporal_confidence)
        
        return self.causality_confidence


@dataclass
class EmergentBehavior:
    """Emergent behavior discovered across intelligence systems"""
    behavior_id: str
    behavior_name: str
    behavior_type: BehaviorType
    complexity_score: float  # 0.0-1.0
    emergence_conditions: List[str] = field(default_factory=list)
    participating_systems: List[str] = field(default_factory=list)
    observable_effects: List[str] = field(default_factory=list)
    predictive_indicators: List[str] = field(default_factory=list)
    stability_assessment: StabilityAssessment = StabilityAssessment.UNDETERMINED
    discovered_at: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)
    
    # Behavior analysis metadata
    observation_count: int = 0
    prediction_accuracy: float = 0.0
    impact_magnitude: float = 0.0
    reproducibility_score: float = 0.0
    
    def record_observation(self, observation_data: Dict[str, Any]):
        """Record a new observation of this emergent behavior"""
        self.observation_count += 1
        self.last_observed = datetime.now()
        
        # Update stability assessment based on observations
        self._update_stability_assessment(observation_data)
        
        # Update impact magnitude
        impact = observation_data.get('impact_magnitude', 0.0)
        if impact > 0:
            self.impact_magnitude = (self.impact_magnitude + impact) / 2.0
    
    def _update_stability_assessment(self, observation_data: Dict[str, Any]):
        """Update stability assessment based on observation patterns"""
        if self.observation_count < 5:
            self.stability_assessment = StabilityAssessment.UNDETERMINED
            return
        
        # Simple heuristics for stability assessment
        consistency = observation_data.get('consistency_score', 0.5)
        
        if consistency > 0.8:
            self.stability_assessment = StabilityAssessment.STABLE
        elif consistency < 0.3:
            self.stability_assessment = StabilityAssessment.UNSTABLE
        elif observation_data.get('trend', 'stable') == 'improving':
            self.stability_assessment = StabilityAssessment.STRENGTHENING
        elif observation_data.get('trend', 'stable') == 'declining':
            self.stability_assessment = StabilityAssessment.DECLINING
        else:
            self.stability_assessment = StabilityAssessment.EVOLVING
    
    def calculate_emergence_score(self) -> float:
        """Calculate score indicating how truly emergent this behavior is"""
        # True emergence is characterized by:
        # - High complexity relative to components
        # - Non-predictable from individual systems
        # - Novel properties not present in parts
        
        complexity_factor = self.complexity_score
        system_interaction_factor = min(1.0, len(self.participating_systems) / 5.0)
        unpredictability_factor = 1.0 - self.prediction_accuracy
        novelty_factor = 1.0 / max(1, self.observation_count) * 10  # More novel if less observed
        
        emergence_score = (
            complexity_factor * 0.3 +
            system_interaction_factor * 0.3 +
            unpredictability_factor * 0.2 +
            novelty_factor * 0.2
        )
        
        return min(1.0, emergence_score)


@dataclass
class SemanticLearningContext:
    """Context for semantic learning operations"""
    learning_session_id: str
    start_time: datetime
    systems_monitored: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    temporal_window_hours: float = 24.0
    confidence_threshold: float = 0.6
    
    # Learning progress tracking
    concepts_processed: int = 0
    relationships_analyzed: int = 0
    behaviors_detected: int = 0
    insights_generated: int = 0


@dataclass
class SemanticInsight:
    """Semantic insight derived from cross-system analysis"""
    insight_id: str
    insight_type: str  # "pattern", "anomaly", "optimization", "prediction"
    description: str
    confidence: float
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    actionable_recommendations: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    
    # Insight metadata
    priority_score: float = 0.0
    business_impact: str = "unknown"  # "low", "medium", "high", "critical"
    implementation_complexity: str = "unknown"  # "low", "medium", "high"


# Export all semantic types and enums
__all__ = [
    'ConceptType', 'AbstractionLevel', 'RelationshipType', 'RelationshipDirectionality',
    'BehaviorType', 'StabilityAssessment', 'EvidenceType',
    'SemanticConcept', 'SemanticRelationship', 'EmergentBehavior',
    'SemanticLearningContext', 'SemanticInsight'
]