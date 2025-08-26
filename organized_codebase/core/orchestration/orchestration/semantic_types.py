"""
Semantic Learning Data Types and Structures

This module defines the core data types and structures used in the
cross-system semantic learning engine.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class SemanticConcept:
    """Unified semantic concept across systems"""
    concept_id: str
    concept_name: str
    concept_type: str  # "entity", "relationship", "behavior", "pattern"
    confidence: float
    abstraction_level: int  # 1=concrete, 5=highly abstract
    system_manifestations: Dict[str, Any]  # How concept appears in each system
    semantic_properties: Dict[str, Any]
    related_concepts: List[str]
    evolution_history: List[Dict[str, Any]]
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class SemanticRelationship:
    """Relationship between semantic concepts"""
    relationship_id: str
    source_concept: str
    target_concept: str
    relationship_type: str  # "causes", "correlates", "contains", "transforms"
    strength: float  # 0-1
    directionality: str  # "bidirectional", "source_to_target", "target_to_source"
    evidence: List[Dict[str, Any]]
    systems_involved: List[str]
    temporal_aspects: Optional[Dict[str, Any]] = None


@dataclass
class EmergentBehavior:
    """Emergent behaviors discovered across systems"""
    behavior_id: str
    behavior_name: str
    behavior_type: str  # "adaptive", "self_organizing", "optimization", "coordination"
    complexity_score: float
    emergence_conditions: List[str]
    participating_systems: List[str]
    observable_effects: List[str]
    predictive_indicators: List[str]
    stability_assessment: str  # "stable", "unstable", "evolving"


@dataclass
class SemanticLearningConfig:
    """Configuration for semantic learning operations"""
    learning_interval: int = 600  # 10 minutes
    concept_confidence_threshold: float = 0.6
    relationship_strength_threshold: float = 0.5
    emergence_detection_threshold: float = 0.7
    abstraction_levels: int = 5
    temporal_window_hours: int = 24
    cross_system_analysis_enabled: bool = True
    emergent_behavior_detection_enabled: bool = True
    semantic_reasoning_enabled: bool = True
    max_concepts_per_cycle: int = 100
    max_relationships_per_cycle: int = 200
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class SemanticLearningMetrics:
    """Metrics for semantic learning operations"""
    concepts_discovered: int = 0
    relationships_discovered: int = 0
    emergent_behaviors_discovered: int = 0
    cross_system_correlations: int = 0
    semantic_inferences_made: int = 0
    learning_cycles_completed: int = 0
    average_cycle_duration: float = 0.0
    last_learning_cycle: Optional[datetime] = None
    start_time: datetime = field(default_factory=datetime.now)