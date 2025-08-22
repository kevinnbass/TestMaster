"""
Architectural Types and Data Structures
=======================================

Core type definitions and data structures for the Architectural Decision Engine.
Provides enterprise-grade type safety for architectural decision making, pattern
analysis, and microservice evolution with comprehensive architectural modeling.

This module contains all Enum definitions and dataclass structures used throughout
the architectural decision system, implementing industry best practices and patterns.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: architectural_types.py (180 lines)
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class DecisionType(Enum):
    """Types of architectural decisions with comprehensive coverage"""
    TECHNOLOGY_SELECTION = "technology_selection"
    PATTERN_ADOPTION = "pattern_adoption"
    ARCHITECTURE_STYLE = "architecture_style"
    SCALING_STRATEGY = "scaling_strategy"
    DATA_ARCHITECTURE = "data_architecture"
    INTEGRATION_APPROACH = "integration_approach"
    DEPLOYMENT_STRATEGY = "deployment_strategy"
    SECURITY_APPROACH = "security_approach"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    MICROSERVICE_BOUNDARIES = "microservice_boundaries"
    API_DESIGN = "api_design"
    STATE_MANAGEMENT = "state_management"


class DecisionCriteria(Enum):
    """Comprehensive criteria for evaluating architectural decisions"""
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    COST = "cost"
    COMPLEXITY = "complexity"
    TIME_TO_MARKET = "time_to_market"
    TEAM_EXPERTISE = "team_expertise"
    RISK = "risk"
    FLEXIBILITY = "flexibility"
    RELIABILITY = "reliability"
    TESTABILITY = "testability"
    OBSERVABILITY = "observability"
    COMPLIANCE = "compliance"


class ArchitecturalPattern(Enum):
    """Comprehensive set of architectural patterns"""
    MICROSERVICES = "microservices"
    MONOLITH = "monolith"
    MODULAR_MONOLITH = "modular_monolith"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    EVENT_DRIVEN = "event_driven"
    CQRS = "cqrs"
    SERVERLESS = "serverless"
    SERVICE_MESH = "service_mesh"
    API_GATEWAY = "api_gateway"
    STRANGLER_FIG = "strangler_fig"
    SAGA = "saga"
    BFF = "backend_for_frontend"
    CLEAN_ARCHITECTURE = "clean_architecture"


class DecisionPriority(Enum):
    """Priority levels for architectural decisions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ArchitecturalImpact(Enum):
    """Impact levels of architectural changes"""
    BREAKING = "breaking"
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class PerformanceMetrics:
    """Performance metrics for architectural evaluation"""
    throughput: float
    latency: float
    memory_usage: float
    cpu_usage: float
    availability: float
    error_rate: float
    scalability_factor: float = 1.0
    load_capacity: float = 1000.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0


@dataclass
class CostMetrics:
    """Cost metrics for architectural options"""
    development_cost: float
    operational_cost: float
    maintenance_cost: float
    infrastructure_cost: float
    training_cost: float = 0.0
    migration_cost: float = 0.0
    total_cost: float = 0.0
    
    def __post_init__(self):
        """Calculate total cost"""
        self.total_cost = (
            self.development_cost + 
            self.operational_cost + 
            self.maintenance_cost + 
            self.infrastructure_cost + 
            self.training_cost + 
            self.migration_cost
        )


@dataclass
class ArchitecturalOption:
    """Comprehensive architectural option for decision analysis"""
    id: str
    name: str
    description: str
    technologies: List[str] = field(default_factory=list)
    patterns: List[ArchitecturalPattern] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    implementation_complexity: float = 0.5
    estimated_timeline: timedelta = field(default_factory=lambda: timedelta(weeks=12))
    cost_metrics: Optional[CostMetrics] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    risk_factors: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    impact_level: ArchitecturalImpact = ArchitecturalImpact.MAJOR


@dataclass
class DecisionContext:
    """Context for architectural decision making"""
    project_name: str
    team_size: int
    timeline: timedelta
    budget: float
    current_architecture: Dict[str, Any]
    requirements: Dict[str, Any]
    constraints: List[str] = field(default_factory=list)
    stakeholder_priorities: Dict[DecisionCriteria, float] = field(default_factory=dict)
    technical_debt: float = 0.0
    scalability_requirements: Dict[str, float] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    existing_technologies: Set[str] = field(default_factory=set)


@dataclass
class CriteriaScore:
    """Score for a specific decision criteria"""
    criteria: DecisionCriteria
    score: float
    weight: float
    rationale: str
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)


@dataclass
class OptionEvaluation:
    """Comprehensive evaluation of an architectural option"""
    option: ArchitecturalOption
    criteria_scores: List[CriteriaScore]
    overall_score: float
    weighted_score: float
    rank: int = 0
    recommendation_strength: str = "medium"
    implementation_feasibility: float = 0.5
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    trade_offs: List[str] = field(default_factory=list)


@dataclass
class DecisionAnalysis:
    """Complete analysis of an architectural decision"""
    id: str
    decision_type: DecisionType
    context: DecisionContext
    options: List[ArchitecturalOption]
    evaluations: List[OptionEvaluation]
    recommended_option: Optional[ArchitecturalOption]
    decision_rationale: str
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    confidence_level: float = 0.7
    implementation_plan: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    review_date: Optional[datetime] = None


@dataclass
class MicroserviceMetrics:
    """Metrics for microservice architecture analysis"""
    service_count: int
    average_service_size: float
    coupling_score: float
    cohesion_score: float
    api_complexity: float
    data_consistency_level: float
    deployment_complexity: float
    communication_overhead: float
    fault_tolerance: float = 0.8
    service_autonomy: float = 0.7


@dataclass
class ServiceBoundary:
    """Definition of a microservice boundary"""
    service_name: str
    business_capability: str
    data_entities: List[str]
    api_endpoints: List[str]
    dependencies: List[str] = field(default_factory=list)
    team_ownership: str = "unassigned"
    complexity_score: float = 0.5
    criticality_level: str = "medium"
    scalability_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class PatternRecommendation:
    """Recommendation for architectural pattern adoption"""
    pattern: ArchitecturalPattern
    confidence: float
    rationale: str
    implementation_steps: List[str]
    expected_benefits: List[str]
    potential_challenges: List[str]
    timeline_estimate: timedelta
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationStrategy:
    """Strategy for architectural migration"""
    strategy_name: str
    migration_type: str  # "big_bang", "strangler_fig", "branch_by_abstraction"
    phases: List[Dict[str, Any]]
    timeline: timedelta
    risk_level: str
    rollback_strategy: str
    success_criteria: List[str]
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    monitoring_plan: List[str] = field(default_factory=list)


# Export all architectural types
__all__ = [
    'DecisionType', 'DecisionCriteria', 'ArchitecturalPattern', 'DecisionPriority', 'ArchitecturalImpact',
    'PerformanceMetrics', 'CostMetrics', 'ArchitecturalOption', 'DecisionContext',
    'CriteriaScore', 'OptionEvaluation', 'DecisionAnalysis', 'MicroserviceMetrics',
    'ServiceBoundary', 'PatternRecommendation', 'MigrationStrategy'
]