"""
Architectural Decision Data Models
=================================

Data models, enums, and data structures for architectural decision making.
Modularized from architectural_decision_engine.py for better maintainability.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any


class DecisionType(Enum):
    """Types of architectural decisions"""
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


class DecisionCriteria(Enum):
    """Criteria for evaluating architectural decisions"""
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


class ArchitecturalPattern(Enum):
    """Supported architectural patterns"""
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


class DecisionPriority(Enum):
    """Priority levels for architectural decisions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ArchitecturalOption:
    """Represents an architectural option for decision analysis"""
    id: str
    name: str
    description: str
    technologies: List[str] = field(default_factory=list)
    patterns: List[ArchitecturalPattern] = field(default_factory=list)
    estimated_effort: int = 0  # In person-hours
    estimated_cost: float = 0.0  # In dollars
    implementation_time: int = 0  # In days
    risk_factors: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    trade_offs: List[str] = field(default_factory=list)
    technical_debt: float = 0.0
    scores: Dict[DecisionCriteria, float] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """Context for architectural decision making"""
    decision_id: str
    decision_type: DecisionType
    description: str
    stakeholders: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    current_architecture: Dict[str, Any] = field(default_factory=dict)
    timeline: Optional[datetime] = None
    budget: Optional[float] = None
    team_size: int = 5
    team_expertise: Dict[str, float] = field(default_factory=dict)
    business_context: Dict[str, Any] = field(default_factory=dict)
    priority: DecisionPriority = DecisionPriority.MEDIUM


@dataclass
class DecisionAnalysis:
    """Results of architectural decision analysis"""
    decision_id: str
    recommended_option: ArchitecturalOption
    alternative_options: List[ArchitecturalOption]
    analysis_summary: str
    confidence_score: float
    risk_assessment: Dict[str, float]
    trade_off_analysis: Dict[str, Any]
    implementation_plan: List[str]
    success_metrics: List[str]
    monitoring_recommendations: List[str]
    decision_rationale: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PatternEvolution:
    """Information about pattern evolution"""
    pattern: ArchitecturalPattern
    current_state: Dict[str, Any]
    evolution_path: List[str]
    target_state: Dict[str, Any]
    migration_steps: List[str]
    risk_factors: List[str]
    success_criteria: List[str]
    timeline: timedelta


@dataclass
class PerformanceMetrics:
    """Performance metrics for architectural decisions"""
    throughput: float = 0.0
    latency: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    availability: float = 0.99
    error_rate: float = 0.01
    scalability_factor: float = 1.0


__all__ = [
    'DecisionType',
    'DecisionCriteria',
    'ArchitecturalPattern',
    'DecisionPriority',
    'ArchitecturalOption',
    'DecisionContext',
    'DecisionAnalysis',
    'PatternEvolution',
    'PerformanceMetrics'
]