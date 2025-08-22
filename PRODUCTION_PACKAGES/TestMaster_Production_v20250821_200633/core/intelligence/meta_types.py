"""
Meta Intelligence Types and Data Structures
===========================================

Core type definitions and data structures for the Meta-Intelligence Orchestrator.
Provides enterprise-grade type safety for meta-level intelligence coordination,
capability mapping, and orchestration strategy management with advanced patterns.

This module contains all Enum definitions and dataclass structures used throughout
the meta-intelligence orchestration system, implementing advanced coordination patterns.

Author: Agent A - PHASE 3: Hours 200-300
Created: 2025-08-22
Module: meta_types.py (120 lines)
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class CapabilityType(Enum):
    """Types of intelligence capabilities for systematic classification"""
    ANALYTICAL = "analytical"
    PREDICTIVE = "predictive"
    OPTIMIZATION = "optimization"
    PATTERN_RECOGNITION = "pattern_recognition"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    ADAPTATION = "adaptation"
    SYNTHESIS = "synthesis"
    REASONING = "reasoning"
    CREATIVE = "creative"


class OrchestrationStrategy(Enum):
    """Strategies for meta-intelligence orchestration and coordination"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"
    HIERARCHICAL = "hierarchical"


class IntelligenceBehaviorType(Enum):
    """Types of intelligence system behaviors for behavioral modeling"""
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    ADAPTIVE = "adaptive"
    LEARNING = "learning"
    EMERGENT = "emergent"
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    AUTONOMOUS = "autonomous"


@dataclass
class CapabilityProfile:
    """Comprehensive profile of an intelligence capability with performance metrics"""
    capability_id: str
    name: str
    type: CapabilityType
    description: str
    input_types: List[str]
    output_types: List[str]
    processing_time: float  # Average processing time in seconds
    accuracy: float  # Accuracy score 0-1
    reliability: float  # Reliability score 0-1
    scalability: float  # Scalability score 0-1
    resource_requirements: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    complementary_capabilities: List[str] = field(default_factory=list)


@dataclass
class SystemBehaviorModel:
    """Model describing the behavior patterns of an intelligence system"""
    system_id: str
    behavior_type: IntelligenceBehaviorType
    behavior_patterns: Dict[str, Any]
    performance_characteristics: Dict[str, float]
    adaptability_metrics: Dict[str, float]
    learning_parameters: Dict[str, Any]
    interaction_patterns: List[str]
    temporal_behavior: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    """Comprehensive plan for orchestrating intelligence systems"""
    plan_id: str
    objective: str
    strategy: OrchestrationStrategy
    target_systems: List[str]
    execution_timeline: List[Dict[str, Any]]
    resource_allocation: Dict[str, float]
    success_criteria: Dict[str, float]
    risk_factors: List[str] = field(default_factory=list)
    contingency_plans: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SynergyOpportunity:
    """Represents an identified synergy opportunity between intelligence systems"""
    opportunity_id: str
    participating_systems: List[str]
    synergy_type: str
    potential_benefit: float
    implementation_complexity: float
    resource_requirements: Dict[str, float]
    estimated_roi: float
    implementation_timeline: timedelta
    risk_assessment: Dict[str, float] = field(default_factory=dict)


# Export all meta intelligence types
__all__ = [
    'CapabilityType',
    'OrchestrationStrategy',
    'IntelligenceBehaviorType',
    'CapabilityProfile',
    'SystemBehaviorModel',
    'OrchestrationPlan',
    'SynergyOpportunity'
]