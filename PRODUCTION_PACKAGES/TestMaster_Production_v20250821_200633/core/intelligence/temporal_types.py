"""
Temporal Types and Data Structures
==================================

Core type definitions and data structures for the Temporal Intelligence Engine.
Provides enterprise-grade type safety for temporal analysis, causality detection,
and oracle-level prediction systems with comprehensive time dynamics modeling.

This module contains all Enum definitions and dataclass structures used throughout
the temporal intelligence system, implementing advanced temporal analysis patterns.

Author: Agent A - PHASE 2: Hours 100-200
Created: 2025-08-22
Module: temporal_types.py (100 lines)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TemporalGranularity(Enum):
    """Granularity levels for multi-scale temporal analysis"""
    NANOSECOND = "nanosecond"
    MICROSECOND = "microsecond"
    MILLISECOND = "millisecond"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    DECADE = "decade"
    CENTURY = "century"


class CausalityType(Enum):
    """Types of causal relationships for temporal causality analysis"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    BIDIRECTIONAL = "bidirectional"
    CIRCULAR = "circular"
    PROBABILISTIC = "probabilistic"
    GRANGER = "granger"
    TRANSFER_ENTROPY = "transfer_entropy"
    CONVERGENT_CROSS_MAPPING = "convergent_cross_mapping"


class TemporalPatternType(Enum):
    """Types of temporal patterns for pattern recognition"""
    PERIODIC = "periodic"
    TREND = "trend"
    SEASONAL = "seasonal"
    CYCLIC = "cyclic"
    IRREGULAR = "irregular"
    CHAOTIC = "chaotic"
    FRACTAL = "fractal"
    EMERGENT = "emergent"


@dataclass
class TemporalPattern:
    """Represents a temporal pattern with comprehensive characteristics"""
    pattern_id: str
    pattern_type: TemporalPatternType
    frequency: Optional[float]
    amplitude: Optional[float]
    phase: Optional[float]
    period: Optional[float]
    confidence: float
    start_time: datetime
    end_time: Optional[datetime]
    recurrence_probability: float


@dataclass
class CausalRelationship:
    """Represents a causal relationship with evidence and strength metrics"""
    relationship_id: str
    cause: str
    effect: str
    causality_type: CausalityType
    strength: float
    time_lag: float
    confidence: float
    evidence: List[Dict[str, Any]]


@dataclass
class TimeSeriesPrediction:
    """Represents a time series prediction with uncertainty quantification"""
    prediction_id: str
    target_variable: str
    predicted_values: List[float]
    timestamps: List[datetime]
    confidence_intervals: List[tuple[float, float]]
    prediction_horizon: int
    model_accuracy: float
    uncertainty_bounds: Dict[str, float]


@dataclass
class FutureState:
    """Represents a predicted future state with probability distribution"""
    state_id: str
    scenario_name: str
    state_vector: Dict[str, Any]
    probability: float
    time_to_state: float
    confidence: float
    preconditions: List[str]
    contributing_factors: List[str]


# Export all temporal types
__all__ = [
    'TemporalGranularity',
    'CausalityType',
    'TemporalPatternType',
    'TemporalPattern',
    'CausalRelationship',
    'TimeSeriesPrediction',
    'FutureState'
]