"""
Pattern Intelligence Data Models
===============================

Core data structures for AI-powered architectural decision-making system.
Extracted from architectural_decision_engine.py for enterprise modular architecture.

Agent D Implementation - Hour 12-13: Revolutionary Intelligence Modularization
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class DecisionType(Enum):
    """Types of architectural decisions that can be analyzed"""
    TECHNOLOGY_SELECTION = "technology_selection"
    PATTERN_ADOPTION = "pattern_adoption"
    ARCHITECTURE_STYLE = "architecture_style"
    SCALING_STRATEGY = "scaling_strategy"
    DATA_STORAGE = "data_storage"
    COMMUNICATION_PROTOCOL = "communication_protocol"
    DEPLOYMENT_STRATEGY = "deployment_strategy"
    SECURITY_APPROACH = "security_approach"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    INTEGRATION_PATTERN = "integration_pattern"


class DecisionCriteria(Enum):
    """Criteria for evaluating architectural decisions"""
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    COST = "cost"
    COMPLEXITY = "complexity"
    FLEXIBILITY = "flexibility"
    RELIABILITY = "reliability"
    TEAM_EXPERTISE = "team_expertise"
    TIME_TO_MARKET = "time_to_market"
    VENDOR_LOCK_IN = "vendor_lock_in"
    FUTURE_PROOFING = "future_proofing"


class ArchitecturalPattern(Enum):
    """Supported architectural patterns for analysis"""
    MICROSERVICES = "microservices"
    MONOLITH = "monolith"
    MODULAR_MONOLITH = "modular_monolith"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    EVENT_DRIVEN = "event_driven"
    CQRS = "cqrs"
    SERVERLESS = "serverless"
    SERVICE_MESH = "service_mesh"
    PLUGIN_ARCHITECTURE = "plugin_architecture"
    PIPE_AND_FILTER = "pipe_and_filter"


@dataclass
class ArchitecturalOption:
    """Represents an architectural option to be evaluated"""
    option_id: str
    name: str
    description: str
    pattern: ArchitecturalPattern
    estimated_cost: float
    estimated_complexity: float
    implementation_time: timedelta
    required_skills: List[str]
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'option_id': self.option_id,
            'name': self.name,
            'description': self.description,
            'pattern': self.pattern.value,
            'estimated_cost': self.estimated_cost,
            'estimated_complexity': self.estimated_complexity,
            'implementation_time': str(self.implementation_time),
            'required_skills': self.required_skills,
            'advantages': self.advantages,
            'disadvantages': self.disadvantages,
            'risk_factors': self.risk_factors,
            'dependencies': self.dependencies
        }


@dataclass
class DecisionContext:
    """Context information for architectural decision making"""
    project_name: str
    team_size: int
    budget_constraints: float
    timeline_constraints: timedelta
    existing_technology_stack: List[str]
    performance_requirements: Dict[str, float]
    scalability_requirements: Dict[str, int]
    security_requirements: List[str]
    compliance_requirements: List[str] = field(default_factory=list)
    team_expertise_areas: List[str] = field(default_factory=list)
    business_priorities: List[str] = field(default_factory=list)
    risk_tolerance: str = "medium"  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'project_name': self.project_name,
            'team_size': self.team_size,
            'budget_constraints': self.budget_constraints,
            'timeline_constraints': str(self.timeline_constraints),
            'existing_technology_stack': self.existing_technology_stack,
            'performance_requirements': self.performance_requirements,
            'scalability_requirements': self.scalability_requirements,
            'security_requirements': self.security_requirements,
            'compliance_requirements': self.compliance_requirements,
            'team_expertise_areas': self.team_expertise_areas,
            'business_priorities': self.business_priorities,
            'risk_tolerance': self.risk_tolerance
        }


@dataclass
class DecisionAnalysis:
    """Result of architectural decision analysis"""
    analysis_id: str
    decision_type: DecisionType
    context: DecisionContext
    options_evaluated: List[ArchitecturalOption]
    criteria_scores: Dict[str, Dict[DecisionCriteria, float]]  # option_id -> criteria -> score
    weighted_scores: Dict[str, float]  # option_id -> weighted_score
    recommended_option: str  # option_id
    confidence_score: float  # 0-1
    rationale: str
    trade_offs: Dict[str, List[str]]  # criteria -> trade_offs
    implementation_recommendations: List[str]
    risk_mitigation_strategies: List[str]
    monitoring_recommendations: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'analysis_id': self.analysis_id,
            'decision_type': self.decision_type.value,
            'context': self.context.to_dict(),
            'options_evaluated': [opt.to_dict() for opt in self.options_evaluated],
            'criteria_scores': {
                opt_id: {criteria.value: score for criteria, score in scores.items()}
                for opt_id, scores in self.criteria_scores.items()
            },
            'weighted_scores': self.weighted_scores,
            'recommended_option': self.recommended_option,
            'confidence_score': self.confidence_score,
            'rationale': self.rationale,
            'trade_offs': self.trade_offs,
            'implementation_recommendations': self.implementation_recommendations,
            'risk_mitigation_strategies': self.risk_mitigation_strategies,
            'monitoring_recommendations': self.monitoring_recommendations,
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }


@dataclass
class PatternEvolution:
    """Represents evolution path between architectural patterns"""
    from_pattern: ArchitecturalPattern
    to_pattern: ArchitecturalPattern
    evolution_steps: List[str]
    estimated_effort: float  # person-days
    estimated_duration: timedelta
    risk_level: str  # low, medium, high
    success_criteria: List[str]
    rollback_strategy: List[str]
    migration_phases: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'from_pattern': self.from_pattern.value,
            'to_pattern': self.to_pattern.value,
            'evolution_steps': self.evolution_steps,
            'estimated_effort': self.estimated_effort,
            'estimated_duration': str(self.estimated_duration),
            'risk_level': self.risk_level,
            'success_criteria': self.success_criteria,
            'rollback_strategy': self.rollback_strategy,
            'migration_phases': self.migration_phases
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for architectural evaluation"""
    response_time_p95: float  # milliseconds
    throughput_rps: float  # requests per second
    memory_usage_mb: float
    cpu_utilization_percent: float
    network_bandwidth_mbps: float
    storage_iops: float
    error_rate_percent: float
    availability_percent: float
    scalability_factor: float  # how well it scales (1.0 = linear)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'response_time_p95': self.response_time_p95,
            'throughput_rps': self.throughput_rps,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_utilization_percent': self.cpu_utilization_percent,
            'network_bandwidth_mbps': self.network_bandwidth_mbps,
            'storage_iops': self.storage_iops,
            'error_rate_percent': self.error_rate_percent,
            'availability_percent': self.availability_percent,
            'scalability_factor': self.scalability_factor
        }


@dataclass
class ImplementationPlan:
    """Comprehensive implementation plan for architectural decisions"""
    plan_id: str
    decision_analysis_id: str
    selected_option: ArchitecturalOption
    implementation_phases: List[Dict[str, Any]]
    timeline: Dict[str, datetime]  # phase -> completion_date
    resource_requirements: Dict[str, Any]
    success_metrics: List[str]
    risk_mitigation_plans: List[Dict[str, Any]]
    rollback_procedures: List[str]
    monitoring_strategy: Dict[str, Any]
    testing_strategy: Dict[str, Any]
    deployment_strategy: Dict[str, Any]
    maintenance_plan: Dict[str, Any]
    total_estimated_cost: float
    total_estimated_duration: timedelta
    confidence_level: float
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'plan_id': self.plan_id,
            'decision_analysis_id': self.decision_analysis_id,
            'selected_option': self.selected_option.to_dict(),
            'implementation_phases': self.implementation_phases,
            'timeline': {phase: dt.isoformat() for phase, dt in self.timeline.items()},
            'resource_requirements': self.resource_requirements,
            'success_metrics': self.success_metrics,
            'risk_mitigation_plans': self.risk_mitigation_plans,
            'rollback_procedures': self.rollback_procedures,
            'monitoring_strategy': self.monitoring_strategy,
            'testing_strategy': self.testing_strategy,
            'deployment_strategy': self.deployment_strategy,
            'maintenance_plan': self.maintenance_plan,
            'total_estimated_cost': self.total_estimated_cost,
            'total_estimated_duration': str(self.total_estimated_duration),
            'confidence_level': self.confidence_level,
            'created_timestamp': self.created_timestamp.isoformat()
        }


@dataclass
class MicroserviceBoundary:
    """Represents a microservice boundary definition"""
    service_name: str
    business_capabilities: List[str]
    data_entities: List[str]
    dependencies: List[str]
    communication_patterns: List[str]
    coupling_score: float  # 0-1, lower is better
    cohesion_score: float  # 0-1, higher is better
    complexity_score: float  # 0-1, lower is better
    recommended: bool = False
    rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'service_name': self.service_name,
            'business_capabilities': self.business_capabilities,
            'data_entities': self.data_entities,
            'dependencies': self.dependencies,
            'communication_patterns': self.communication_patterns,
            'coupling_score': self.coupling_score,
            'cohesion_score': self.cohesion_score,
            'complexity_score': self.complexity_score,
            'recommended': self.recommended,
            'rationale': self.rationale
        }


@dataclass
class ArchitecturalDecisionRecord:
    """Architectural Decision Record (ADR) for documentation"""
    adr_id: str
    title: str
    status: str  # proposed, accepted, rejected, superseded
    context: str
    decision: str
    consequences: List[str]
    alternatives_considered: List[str]
    created_date: datetime
    last_modified: datetime
    author: str
    stakeholders: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'adr_id': self.adr_id,
            'title': self.title,
            'status': self.status,
            'context': self.context,
            'decision': self.decision,
            'consequences': self.consequences,
            'alternatives_considered': self.alternatives_considered,
            'created_date': self.created_date.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'author': self.author,
            'stakeholders': self.stakeholders,
            'tags': self.tags
        }