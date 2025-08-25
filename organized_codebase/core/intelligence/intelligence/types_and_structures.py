"""
Architectural Evolution Predictor - Types and Data Structures
Modularized from architectural_evolution_predictor.py

This module contains all enums, dataclasses, and type definitions used throughout
the architectural evolution prediction system.
"""

from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class ArchitecturalPattern(Enum):
    """Types of architectural patterns"""
    MONOLITHIC = "monolithic"
    MICROSERVICES = "microservices"
    LAYERED = "layered"
    EVENT_DRIVEN = "event_driven"
    HEXAGONAL = "hexagonal"
    SERVERLESS = "serverless"
    SOA = "service_oriented"
    CLEAN_ARCHITECTURE = "clean_architecture"
    CQRS = "cqrs"
    ONION = "onion"


class ScalingPattern(Enum):
    """Patterns of system scaling"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    FUNCTIONAL = "functional"
    DATA_PARTITIONING = "data_partitioning"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"
    LOAD_BASED = "load_based"
    FEATURE_BASED = "feature_based"


class TechnologyTrend(Enum):
    """Emerging technology trends"""
    CLOUD_NATIVE = "cloud_native"
    EDGE_COMPUTING = "edge_computing"
    AI_ML_INTEGRATION = "ai_ml_integration"
    BLOCKCHAIN = "blockchain"
    IOT_INTEGRATION = "iot_integration"
    QUANTUM_COMPUTING = "quantum_computing"
    CONTAINERIZATION = "containerization"
    SERVERLESS_COMPUTING = "serverless_computing"
    WEBASSEMBLY = "webassembly"
    GRAPHQL = "graphql"


class EvolutionProbability(Enum):
    """Probability levels for evolution predictions"""
    VERY_HIGH = "very_high"     # > 90%
    HIGH = "high"               # 70-90%
    MEDIUM = "medium"           # 40-70%
    LOW = "low"                 # 20-40%
    VERY_LOW = "very_low"       # < 20%


@dataclass
class ArchitecturalMetrics:
    """Current architectural metrics for analysis"""
    component_count: int = 0
    service_count: int = 0
    api_endpoint_count: int = 0
    database_count: int = 0
    integration_point_count: int = 0
    lines_of_code: int = 0
    complexity_score: float = 0.0
    coupling_score: float = 0.0
    cohesion_score: float = 0.0
    performance_score: float = 0.0
    scalability_score: float = 0.0
    maintainability_score: float = 0.0
    security_score: float = 0.0
    deployment_complexity: float = 0.0
    monitoring_coverage: float = 0.0
    test_coverage: float = 0.0
    documentation_coverage: float = 0.0


@dataclass
class SystemGrowthPattern:
    """Pattern of system growth over time"""
    growth_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    time_period: str = ""
    component_growth_rate: float = 0.0
    user_growth_rate: float = 0.0
    data_growth_rate: float = 0.0
    transaction_growth_rate: float = 0.0
    feature_addition_rate: float = 0.0
    complexity_growth_rate: float = 0.0
    team_growth_rate: float = 0.0
    deployment_frequency: float = 0.0
    incident_rate: float = 0.0
    performance_degradation_rate: float = 0.0
    growth_acceleration: float = 0.0
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    growth_sustainability: float = 0.0


@dataclass
class ScalabilityForecast:
    """Forecast of system scalability needs"""
    forecast_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    forecast_horizon: int = 12  # months
    current_capacity: Dict[str, float] = field(default_factory=dict)
    predicted_load: Dict[str, float] = field(default_factory=dict)
    capacity_gaps: Dict[str, float] = field(default_factory=dict)
    bottleneck_predictions: List[Dict[str, Any]] = field(default_factory=list)
    scaling_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    resource_requirements: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cost_implications: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    confidence_level: float = 0.0
    recommended_scaling_strategy: ScalingPattern = ScalingPattern.HORIZONTAL


@dataclass
class TechnologyEvolutionAnalysis:
    """Analysis of technology trend impact"""
    analysis_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    current_technology_stack: Dict[str, str] = field(default_factory=dict)
    emerging_trends: List[TechnologyTrend] = field(default_factory=list)
    trend_relevance_scores: Dict[TechnologyTrend, float] = field(default_factory=dict)
    adoption_timeline: Dict[TechnologyTrend, str] = field(default_factory=dict)
    migration_complexity: Dict[TechnologyTrend, float] = field(default_factory=dict)
    business_impact: Dict[TechnologyTrend, Dict[str, str]] = field(default_factory=dict)
    technical_feasibility: Dict[TechnologyTrend, float] = field(default_factory=dict)
    risk_assessment: Dict[TechnologyTrend, Dict[str, float]] = field(default_factory=dict)
    investment_requirements: Dict[TechnologyTrend, Dict[str, float]] = field(default_factory=dict)
    competitive_advantage: Dict[TechnologyTrend, float] = field(default_factory=dict)


@dataclass
class ArchitecturalEvolutionPrediction:
    """Comprehensive architectural evolution prediction"""
    prediction_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest())
    current_architecture: ArchitecturalPattern = ArchitecturalPattern.MONOLITHIC
    predicted_architecture: ArchitecturalPattern = ArchitecturalPattern.MICROSERVICES
    evolution_probability: EvolutionProbability = EvolutionProbability.MEDIUM
    evolution_timeline: str = ""
    evolution_drivers: List[str] = field(default_factory=list)
    evolution_barriers: List[str] = field(default_factory=list)
    migration_strategy: List[str] = field(default_factory=list)
    architectural_metrics: ArchitecturalMetrics = field(default_factory=ArchitecturalMetrics)
    growth_patterns: SystemGrowthPattern = field(default_factory=SystemGrowthPattern)
    scalability_forecast: ScalabilityForecast = field(default_factory=ScalabilityForecast)
    technology_evolution: TechnologyEvolutionAnalysis = field(default_factory=TechnologyEvolutionAnalysis)
    component_evolution: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    integration_evolution: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    data_architecture_evolution: Dict[str, Any] = field(default_factory=dict)
    performance_evolution: Dict[str, float] = field(default_factory=dict)
    security_evolution: Dict[str, Any] = field(default_factory=dict)
    predicted_challenges: List[Dict[str, Any]] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    validation_criteria: List[str] = field(default_factory=list)
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    confidence_factors: Dict[str, float] = field(default_factory=dict)