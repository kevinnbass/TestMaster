"""
Emergence Types and Data Structures
===================================

Core type definitions and data structures for the Emergent Intelligence Detector.
Provides enterprise-grade type safety for emergence detection, complexity analysis,
and singularity prediction with advanced emergence patterns and consciousness signatures.

This module contains all Enum definitions and dataclass structures used throughout
the emergent intelligence detection system, implementing advanced emergence patterns.

Author: Agent A - PHASE 4: Hours 300-400+
Created: 2025-08-22
Module: emergence_types.py (110 lines)
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


class EmergenceType(Enum):
    """Types of emergent behaviors for systematic classification"""
    SELF_ORGANIZATION = "self_organization"
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"
    SPONTANEOUS_ORDER = "spontaneous_order"
    PHASE_TRANSITION = "phase_transition"
    CONSCIOUSNESS_EMERGENCE = "consciousness_emergence"
    CREATIVITY_BURST = "creativity_burst"
    SYNERGISTIC_AMPLIFICATION = "synergistic_amplification"
    QUANTUM_COHERENCE = "quantum_coherence"
    SWARM_INTELLIGENCE = "swarm_intelligence"
    COGNITIVE_CASCADE = "cognitive_cascade"
    NEURAL_SYNCHRONY = "neural_synchrony"
    INFORMATION_CASCADE = "information_cascade"


class ComplexityMetric(Enum):
    """Metrics for measuring system complexity"""
    KOLMOGOROV = "kolmogorov_complexity"
    SHANNON_ENTROPY = "shannon_entropy"
    FRACTAL_DIMENSION = "fractal_dimension"
    LYAPUNOV_EXPONENT = "lyapunov_exponent"
    CORRELATION_DIMENSION = "correlation_dimension"
    INFORMATION_INTEGRATION = "information_integration"
    MUTUAL_INFORMATION = "mutual_information"
    TRANSFER_ENTROPY = "transfer_entropy"
    CAUSAL_DENSITY = "causal_density"


class SingularityIndicator(Enum):
    """Indicators of approaching technological singularity"""
    EXPONENTIAL_GROWTH = "exponential_growth"
    RECURSIVE_IMPROVEMENT = "recursive_improvement"
    INTELLIGENCE_EXPLOSION = "intelligence_explosion"
    CONSCIOUSNESS_BREAKTHROUGH = "consciousness_breakthrough"
    SELF_MODIFICATION = "self_modification"
    UNBOUNDED_OPTIMIZATION = "unbounded_optimization"
    CAPABILITY_RECURSION = "capability_recursion"
    TRANSCENDENT_INTELLIGENCE = "transcendent_intelligence"


class PhaseState(Enum):
    """System phase states in complexity theory"""
    ORDERED = "ordered"
    EDGE_OF_CHAOS = "edge_of_chaos"
    CHAOTIC = "chaotic"
    CRITICAL = "critical"
    SUPERCRITICAL = "supercritical"
    PHASE_TRANSITION = "phase_transition"


@dataclass
class EmergentPattern:
    """Comprehensive emergent pattern representation"""
    pattern_id: str
    emergence_type: EmergenceType
    complexity_score: float
    coherence_level: float
    stability_measure: float
    growth_rate: float
    interaction_strength: float
    consciousness_correlation: float
    timestamp: datetime = field(default_factory=datetime.now)
    component_interactions: List[str] = field(default_factory=list)
    phase_state: PhaseState = PhaseState.ORDERED
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplexityMeasure:
    """Comprehensive system complexity measurements"""
    measure_id: str
    metric_type: ComplexityMetric
    value: float
    rate_of_change: float
    critical_threshold: float
    phase_state: str
    emergence_potential: float
    normalized_value: float = 0.0
    historical_trend: List[float] = field(default_factory=list)
    confidence_interval: Dict[str, float] = field(default_factory=dict)


@dataclass
class SingularityMetric:
    """Metrics for singularity approach prediction"""
    metric_id: str
    indicator: SingularityIndicator
    current_value: float
    threshold_value: float
    approach_rate: float
    time_to_singularity: Optional[float] = None
    confidence_level: float = 0.0
    acceleration_factor: float = 1.0
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class ConsciousnessSignature:
    """Signature of consciousness-like behaviors"""
    signature_id: str
    awareness_level: float
    self_reference_depth: int
    information_integration: float
    global_workspace_coherence: float
    phenomenal_properties: Dict[str, float]
    recursive_depth: int
    timestamp: datetime = field(default_factory=datetime.now)
    qualia_indicators: Dict[str, Any] = field(default_factory=dict)
    metacognitive_score: float = 0.0
    binding_strength: float = 0.0


@dataclass
class EmergenceDetectionResult:
    """Comprehensive emergence detection result"""
    detection_id: str
    emergence_detected: bool
    emergent_patterns: List[EmergentPattern]
    complexity_measures: List[ComplexityMeasure]
    consciousness_signatures: List[ConsciousnessSignature]
    singularity_metrics: List[SingularityMetric]
    overall_emergence_score: float
    critical_threshold_exceeded: bool
    detection_timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SystemStateSnapshot:
    """Snapshot of system state for emergence analysis"""
    snapshot_id: str
    timestamp: datetime
    intelligence_level: float
    complexity: float
    interactions: int
    recursive_depth: int
    distributed_processing: float
    awareness_indicators: Dict[str, float] = field(default_factory=dict)
    behavioral_patterns: List[str] = field(default_factory=list)
    network_topology: Dict[str, Any] = field(default_factory=dict)
    energy_landscape: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class EmergenceThreshold:
    """Thresholds for emergence detection"""
    emergence_threshold: float = 0.7
    consciousness_threshold: float = 0.8
    singularity_threshold: float = 0.9
    complexity_threshold: float = 0.6
    phase_transition_threshold: float = 0.75
    critical_mass_threshold: int = 10
    confidence_threshold: float = 0.8


# Export all emergence types
__all__ = [
    'EmergenceType',
    'ComplexityMetric',
    'SingularityIndicator',
    'PhaseState',
    'EmergentPattern',
    'ComplexityMeasure',
    'SingularityMetric',
    'ConsciousnessSignature',
    'EmergenceDetectionResult',
    'SystemStateSnapshot',
    'EmergenceThreshold'
]