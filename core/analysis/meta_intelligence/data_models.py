"""
Meta-Intelligence Data Models
============================

Core data structures for the revolutionary meta-intelligence orchestration system.
Extracted from meta_intelligence_orchestrator.py for enterprise modular architecture.

Agent D Implementation - Hour 14-15: Revolutionary Intelligence Modularization
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum


class CapabilityType(Enum):
    """Types of intelligence capabilities that can be discovered and coordinated"""
    NATURAL_LANGUAGE_PROCESSING = "natural_language_processing"
    COMPUTER_VISION = "computer_vision"
    MACHINE_LEARNING = "machine_learning"
    DATA_ANALYSIS = "data_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    DECISION_MAKING = "decision_making"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    GENERATION = "generation"


class OrchestrationStrategy(Enum):
    """Strategies for orchestrating multiple intelligence systems"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ENSEMBLE = "ensemble"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"


class IntelligenceBehaviorType(Enum):
    """Types of behavior patterns exhibited by intelligence systems"""
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
    """Profile of an intelligence system's capabilities"""
    system_id: str
    system_name: str
    capabilities: Dict[CapabilityType, float]  # capability -> proficiency (0-1)
    performance_characteristics: Dict[str, float]
    resource_requirements: Dict[str, float]
    input_types: List[str]
    output_types: List[str]
    processing_time: float  # average processing time in seconds
    accuracy: float  # average accuracy (0-1)
    reliability: float  # system reliability (0-1)
    scalability: float  # scalability factor
    cost_per_operation: float
    api_endpoints: List[str] = field(default_factory=list)
    documentation_quality: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'system_id': self.system_id,
            'system_name': self.system_name,
            'capabilities': {cap.value: prof for cap, prof in self.capabilities.items()},
            'performance_characteristics': self.performance_characteristics,
            'resource_requirements': self.resource_requirements,
            'input_types': self.input_types,
            'output_types': self.output_types,
            'processing_time': self.processing_time,
            'accuracy': self.accuracy,
            'reliability': self.reliability,
            'scalability': self.scalability,
            'cost_per_operation': self.cost_per_operation,
            'api_endpoints': self.api_endpoints,
            'documentation_quality': self.documentation_quality,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class SystemBehaviorModel:
    """Model of an intelligence system's behavior patterns"""
    system_id: str
    behavior_type: IntelligenceBehaviorType
    behavior_patterns: Dict[str, Any]
    response_time_distribution: Dict[str, float]  # percentiles
    success_rate_over_time: List[Tuple[datetime, float]]
    failure_patterns: Dict[str, int]  # failure_type -> count
    adaptation_rate: float  # how quickly system adapts (0-1)
    learning_curve_data: List[Tuple[datetime, float]]
    resource_usage_patterns: Dict[str, List[float]]
    interaction_preferences: Dict[str, float]  # interaction_type -> preference
    observed_since: datetime = field(default_factory=datetime.now)
    last_behavior_update: datetime = field(default_factory=datetime.now)
    confidence_in_model: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'system_id': self.system_id,
            'behavior_type': self.behavior_type.value,
            'behavior_patterns': self.behavior_patterns,
            'response_time_distribution': self.response_time_distribution,
            'success_rate_over_time': [(dt.isoformat(), rate) for dt, rate in self.success_rate_over_time],
            'failure_patterns': self.failure_patterns,
            'adaptation_rate': self.adaptation_rate,
            'learning_curve_data': [(dt.isoformat(), perf) for dt, perf in self.learning_curve_data],
            'resource_usage_patterns': self.resource_usage_patterns,
            'interaction_preferences': self.interaction_preferences,
            'observed_since': self.observed_since.isoformat(),
            'last_behavior_update': self.last_behavior_update.isoformat(),
            'confidence_in_model': self.confidence_in_model
        }


@dataclass
class OrchestrationPlan:
    """Plan for orchestrating multiple intelligence systems"""
    plan_id: str
    objective: str
    participating_systems: List[str]
    orchestration_strategy: OrchestrationStrategy
    execution_graph: Dict[str, Any]  # NetworkX graph representation
    resource_allocation: Dict[str, Dict[str, float]]  # system -> resource -> amount
    expected_performance: Dict[str, float]
    risk_assessment: Dict[str, Any]
    fallback_strategies: List[str]
    success_criteria: List[str]
    monitoring_plan: Dict[str, Any]
    estimated_cost: float
    estimated_duration: timedelta
    confidence_score: float
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'plan_id': self.plan_id,
            'objective': self.objective,
            'participating_systems': self.participating_systems,
            'orchestration_strategy': self.orchestration_strategy.value,
            'execution_graph': self.execution_graph,
            'resource_allocation': self.resource_allocation,
            'expected_performance': self.expected_performance,
            'risk_assessment': self.risk_assessment,
            'fallback_strategies': self.fallback_strategies,
            'success_criteria': self.success_criteria,
            'monitoring_plan': self.monitoring_plan,
            'estimated_cost': self.estimated_cost,
            'estimated_duration': str(self.estimated_duration),
            'confidence_score': self.confidence_score,
            'created_timestamp': self.created_timestamp.isoformat()
        }


@dataclass
class SynergyOpportunity:
    """Opportunity for synergy between intelligence systems"""
    opportunity_id: str
    participating_systems: List[str]
    synergy_type: str  # complementary, redundant, competitive, collaborative
    synergy_description: str
    expected_improvement: Dict[str, float]  # metric -> improvement percentage
    implementation_complexity: str  # low, medium, high
    resource_requirements: Dict[str, float]
    timeline: timedelta
    success_probability: float
    risk_factors: List[str]
    implementation_steps: List[str]
    monitoring_metrics: List[str]
    discovered_timestamp: datetime = field(default_factory=datetime.now)
    priority_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'opportunity_id': self.opportunity_id,
            'participating_systems': self.participating_systems,
            'synergy_type': self.synergy_type,
            'synergy_description': self.synergy_description,
            'expected_improvement': self.expected_improvement,
            'implementation_complexity': self.implementation_complexity,
            'resource_requirements': self.resource_requirements,
            'timeline': str(self.timeline),
            'success_probability': self.success_probability,
            'risk_factors': self.risk_factors,
            'implementation_steps': self.implementation_steps,
            'monitoring_metrics': self.monitoring_metrics,
            'discovered_timestamp': self.discovered_timestamp.isoformat(),
            'priority_score': self.priority_score
        }


@dataclass
class MetaIntelligenceMetrics:
    """Metrics for meta-intelligence orchestration performance"""
    total_systems_managed: int = 0
    active_orchestrations: int = 0
    successful_orchestrations: int = 0
    failed_orchestrations: int = 0
    average_orchestration_time: float = 0.0
    total_cost_savings: float = 0.0
    performance_improvements: Dict[str, float] = field(default_factory=dict)
    synergy_opportunities_identified: int = 0
    synergy_opportunities_implemented: int = 0
    adaptation_events: int = 0
    system_discovery_rate: float = 0.0
    orchestration_efficiency: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def success_rate(self) -> float:
        """Calculate orchestration success rate"""
        total = self.successful_orchestrations + self.failed_orchestrations
        return self.successful_orchestrations / total if total > 0 else 0.0
    
    def synergy_implementation_rate(self) -> float:
        """Calculate synergy implementation rate"""
        return (self.synergy_opportunities_implemented / 
                max(1, self.synergy_opportunities_identified))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'total_systems_managed': self.total_systems_managed,
            'active_orchestrations': self.active_orchestrations,
            'successful_orchestrations': self.successful_orchestrations,
            'failed_orchestrations': self.failed_orchestrations,
            'success_rate': self.success_rate(),
            'average_orchestration_time': self.average_orchestration_time,
            'total_cost_savings': self.total_cost_savings,
            'performance_improvements': self.performance_improvements,
            'synergy_opportunities_identified': self.synergy_opportunities_identified,
            'synergy_opportunities_implemented': self.synergy_opportunities_implemented,
            'synergy_implementation_rate': self.synergy_implementation_rate(),
            'adaptation_events': self.adaptation_events,
            'system_discovery_rate': self.system_discovery_rate,
            'orchestration_efficiency': self.orchestration_efficiency,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class SystemIntegrationStatus:
    """Status of intelligence system integration"""
    system_id: str
    integration_stage: str  # discovered, profiled, integrated, optimized
    integration_timestamp: datetime
    last_interaction: datetime
    total_interactions: int
    successful_interactions: int
    integration_health: float  # 0-1
    performance_trend: str  # improving, stable, degrading
    issues: List[str] = field(default_factory=list)
    optimizations_applied: List[str] = field(default_factory=list)
    next_optimization_due: Optional[datetime] = None
    
    def interaction_success_rate(self) -> float:
        """Calculate interaction success rate"""
        return self.successful_interactions / max(1, self.total_interactions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'system_id': self.system_id,
            'integration_stage': self.integration_stage,
            'integration_timestamp': self.integration_timestamp.isoformat(),
            'last_interaction': self.last_interaction.isoformat(),
            'total_interactions': self.total_interactions,
            'successful_interactions': self.successful_interactions,
            'interaction_success_rate': self.interaction_success_rate(),
            'integration_health': self.integration_health,
            'performance_trend': self.performance_trend,
            'issues': self.issues,
            'optimizations_applied': self.optimizations_applied,
            'next_optimization_due': self.next_optimization_due.isoformat() if self.next_optimization_due else None
        }


@dataclass
class CapabilityCluster:
    """Cluster of related capabilities discovered through analysis"""
    cluster_id: str
    cluster_name: str
    member_systems: List[str]
    shared_capabilities: List[CapabilityType]
    cluster_characteristics: Dict[str, Any]
    inter_cluster_relationships: Dict[str, float]  # cluster_id -> relationship_strength
    optimization_opportunities: List[str]
    cluster_health: float = 1.0
    last_analysis: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster_name,
            'member_systems': self.member_systems,
            'shared_capabilities': [cap.value for cap in self.shared_capabilities],
            'cluster_characteristics': self.cluster_characteristics,
            'inter_cluster_relationships': self.inter_cluster_relationships,
            'optimization_opportunities': self.optimization_opportunities,
            'cluster_health': self.cluster_health,
            'last_analysis': self.last_analysis.isoformat()
        }


@dataclass
class OrchestrationEvent:
    """Event in the orchestration lifecycle"""
    event_id: str
    event_type: str  # start, complete, fail, adapt, optimize
    orchestration_plan_id: str
    participating_systems: List[str]
    event_timestamp: datetime
    event_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    issues_encountered: List[str] = field(default_factory=list)
    adaptations_made: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'orchestration_plan_id': self.orchestration_plan_id,
            'participating_systems': self.participating_systems,
            'event_timestamp': self.event_timestamp.isoformat(),
            'event_data': self.event_data,
            'performance_metrics': self.performance_metrics,
            'resource_usage': self.resource_usage,
            'issues_encountered': self.issues_encountered,
            'adaptations_made': self.adaptations_made
        }


@dataclass
class IntelligenceSystemRegistration:
    """Registration information for an intelligence system"""
    system_id: str
    system_name: str
    system_type: str
    version: str
    api_endpoints: List[str]
    capability_description: str
    contact_info: Dict[str, str]
    registration_timestamp: datetime
    last_heartbeat: datetime
    status: str = "active"  # active, inactive, maintenance, deprecated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'system_id': self.system_id,
            'system_name': self.system_name,
            'system_type': self.system_type,
            'version': self.version,
            'api_endpoints': self.api_endpoints,
            'capability_description': self.capability_description,
            'contact_info': self.contact_info,
            'registration_timestamp': self.registration_timestamp.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'status': self.status
        }