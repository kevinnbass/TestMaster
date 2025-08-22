"""
Optimization Models - Core data structures for intelligent code optimization

This module provides comprehensive data models for the intelligent code optimization
system, including enums, dataclasses, and factory functions for creating optimization
recommendations, performance metrics, and learning database structures.

Key Components:
- Optimization type definitions and priority levels
- Comprehensive recommendation data structures
- Performance metrics and risk assessment models
- Learning database schemas for ML-powered improvements
- Factory functions for creating optimization artifacts
"""

import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class OptimizationType(Enum):
    """Types of code optimizations supported by the intelligent optimizer"""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    TESTABILITY = "testability"
    ARCHITECTURE = "architecture"
    DESIGN_PATTERN = "design_pattern"
    ALGORITHM = "algorithm"
    CODE_QUALITY = "code_quality"
    REFACTORING = "refactoring"
    BEST_PRACTICES = "best_practices"


class OptimizationPriority(Enum):
    """Priority levels for optimization recommendations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NICE_TO_HAVE = "nice_to_have"


class OptimizationStrategy(Enum):
    """Implementation strategies for optimization recommendations"""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    REFACTOR = "refactor"
    REDESIGN = "redesign"
    PATTERN_APPLICATION = "pattern_application"
    ALGORITHMIC_CHANGE = "algorithmic_change"
    INCREMENTAL_IMPROVEMENT = "incremental_improvement"


class RecommendationStatus(Enum):
    """Status tracking for optimization recommendations"""
    PENDING = "pending"
    APPROVED = "approved"
    IMPLEMENTED = "implemented"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    SUPERSEDED = "superseded"
    IN_REVIEW = "in_review"
    TESTING = "testing"


class AnalysisType(Enum):
    """Types of code analysis performed by the optimizer"""
    STATIC_ANALYSIS = "static_analysis"
    PERFORMANCE_PROFILING = "performance_profiling"
    SECURITY_SCAN = "security_scan"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    QUALITY_METRICS = "quality_metrics"


@dataclass
class PerformanceMetrics:
    """Performance metrics for code analysis and optimization tracking"""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_utilization: float = 0.0
    io_operations: int = 0
    network_calls: int = 0
    database_queries: int = 0
    cache_hit_ratio: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Code quality metrics for comprehensive analysis"""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    technical_debt_ratio: float = 0.0
    code_coverage: float = 0.0
    test_coverage: float = 0.0
    documentation_coverage: float = 0.0
    duplication_percentage: float = 0.0
    lines_of_code: int = 0
    comment_ratio: float = 0.0
    function_complexity_avg: float = 0.0
    class_complexity_avg: float = 0.0


@dataclass
class SecurityMetrics:
    """Security-related metrics and vulnerability assessment"""
    vulnerability_count: int = 0
    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    medium_vulnerabilities: int = 0
    low_vulnerabilities: int = 0
    security_score: float = 0.0
    owasp_compliance: float = 0.0
    encryption_usage: float = 0.0
    authentication_strength: float = 0.0
    data_validation_coverage: float = 0.0


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for optimization recommendations"""
    implementation_risk: float = 0.0
    business_impact_risk: float = 0.0
    technical_debt_risk: float = 0.0
    security_risk: float = 0.0
    performance_risk: float = 0.0
    maintainability_risk: float = 0.0
    testing_risk: float = 0.0
    deployment_risk: float = 0.0
    overall_risk_score: float = 0.0
    risk_mitigation_strategies: List[str] = field(default_factory=list)
    rollback_plan: str = ""


@dataclass
class OptimizationRecommendation:
    """Comprehensive optimization recommendation with AI-powered insights"""
    recommendation_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest())
    optimization_type: OptimizationType = OptimizationType.PERFORMANCE
    priority: OptimizationPriority = OptimizationPriority.MEDIUM
    strategy: OptimizationStrategy = OptimizationStrategy.GRADUAL
    status: RecommendationStatus = RecommendationStatus.PENDING
    
    # Target information
    target_file: str = ""
    target_element: str = ""
    target_lines: Tuple[int, int] = (0, 0)
    target_function: str = ""
    target_class: str = ""
    
    # Recommendation details
    title: str = ""
    description: str = ""
    reasoning: str = ""
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    implementation_effort: str = ""
    estimated_hours: float = 0.0
    
    # Code samples
    original_code: str = ""
    optimized_code: str = ""
    code_diff: str = ""
    alternative_solutions: List[str] = field(default_factory=list)
    
    # Requirements and dependencies
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    testing_requirements: List[str] = field(default_factory=list)
    
    # Assessment and metrics
    risk_assessment: RiskAssessment = field(default_factory=RiskAssessment)
    confidence_score: float = 0.0
    success_probability: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "intelligent_optimizer"
    reviewed_by: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    related_recommendations: List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Results of applying an optimization recommendation"""
    recommendation_id: str
    implementation_status: RecommendationStatus
    before_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    after_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    improvement_percentage: Dict[str, float] = field(default_factory=dict)
    implementation_time: float = 0.0
    testing_results: Dict[str, Any] = field(default_factory=dict)
    rollback_performed: bool = False
    lessons_learned: List[str] = field(default_factory=list)
    implementation_notes: str = ""
    implemented_at: datetime = field(default_factory=datetime.now)


@dataclass
class LearningEntry:
    """Machine learning entry for optimization pattern recognition"""
    pattern_id: str
    pattern_type: str
    optimization_type: OptimizationType
    success_count: int = 0
    failure_count: int = 0
    total_attempts: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_improvement: float = 0.0
    common_prerequisites: List[str] = field(default_factory=list)
    common_issues: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationContext:
    """Context information for optimization analysis"""
    project_name: str = ""
    project_type: str = ""
    language: str = "python"
    framework: str = ""
    version: str = ""
    environment: str = "development"
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    security_requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    available_resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSession:
    """Complete optimization session with comprehensive analysis"""
    session_id: str = field(default_factory=lambda: f"opt_session_{int(time.time())}")
    context: OptimizationContext = field(default_factory=OptimizationContext)
    target_files: List[str] = field(default_factory=list)
    analysis_types: List[AnalysisType] = field(default_factory=list)
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    results: List[OptimizationResult] = field(default_factory=list)
    session_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    security_metrics: SecurityMetrics = field(default_factory=SecurityMetrics)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    status: str = "active"


# Factory Functions
def create_optimization_recommendation(
    optimization_type: OptimizationType,
    title: str,
    description: str,
    target_file: str = "",
    priority: OptimizationPriority = OptimizationPriority.MEDIUM,
    strategy: OptimizationStrategy = OptimizationStrategy.GRADUAL
) -> OptimizationRecommendation:
    """Create a new optimization recommendation with defaults"""
    return OptimizationRecommendation(
        optimization_type=optimization_type,
        title=title,
        description=description,
        target_file=target_file,
        priority=priority,
        strategy=strategy
    )


def create_performance_metrics(
    execution_time: float = 0.0,
    memory_usage: float = 0.0,
    custom_metrics: Optional[Dict[str, float]] = None
) -> PerformanceMetrics:
    """Create performance metrics with optional custom metrics"""
    return PerformanceMetrics(
        execution_time=execution_time,
        memory_usage=memory_usage,
        custom_metrics=custom_metrics or {}
    )


def create_optimization_session(
    project_name: str,
    target_files: List[str],
    analysis_types: Optional[List[AnalysisType]] = None
) -> OptimizationSession:
    """Create a new optimization session"""
    context = OptimizationContext(project_name=project_name)
    return OptimizationSession(
        context=context,
        target_files=target_files,
        analysis_types=analysis_types or [AnalysisType.STATIC_ANALYSIS]
    )


def create_risk_assessment(
    implementation_risk: float = 0.0,
    business_impact_risk: float = 0.0,
    technical_debt_risk: float = 0.0
) -> RiskAssessment:
    """Create risk assessment with calculated overall score"""
    risk = RiskAssessment(
        implementation_risk=implementation_risk,
        business_impact_risk=business_impact_risk,
        technical_debt_risk=technical_debt_risk
    )
    # Calculate overall risk score
    risk.overall_risk_score = (
        implementation_risk + business_impact_risk + technical_debt_risk
    ) / 3.0
    return risk


def create_learning_entry(
    pattern_id: str,
    pattern_type: str,
    optimization_type: OptimizationType
) -> LearningEntry:
    """Create a new learning database entry"""
    return LearningEntry(
        pattern_id=pattern_id,
        pattern_type=pattern_type,
        optimization_type=optimization_type
    )


# Utility Functions
def get_priority_weight(priority: OptimizationPriority) -> float:
    """Get numeric weight for priority comparison"""
    weights = {
        OptimizationPriority.CRITICAL: 1.0,
        OptimizationPriority.HIGH: 0.8,
        OptimizationPriority.MEDIUM: 0.6,
        OptimizationPriority.LOW: 0.4,
        OptimizationPriority.NICE_TO_HAVE: 0.2
    }
    return weights.get(priority, 0.5)


def calculate_improvement_score(before: PerformanceMetrics, after: PerformanceMetrics) -> Dict[str, float]:
    """Calculate improvement percentages between before/after metrics"""
    improvements = {}
    
    if before.execution_time > 0:
        improvements['execution_time'] = ((before.execution_time - after.execution_time) / before.execution_time) * 100
    
    if before.memory_usage > 0:
        improvements['memory_usage'] = ((before.memory_usage - after.memory_usage) / before.memory_usage) * 100
    
    if before.cpu_utilization > 0:
        improvements['cpu_utilization'] = ((before.cpu_utilization - after.cpu_utilization) / before.cpu_utilization) * 100
    
    return improvements


def sort_recommendations_by_impact(recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
    """Sort recommendations by potential impact and priority"""
    def impact_score(rec: OptimizationRecommendation) -> float:
        priority_weight = get_priority_weight(rec.priority)
        confidence_weight = rec.confidence_score
        expected_impact = sum(rec.expected_improvement.values()) / max(len(rec.expected_improvement), 1)
        return priority_weight * confidence_weight * expected_impact
    
    return sorted(recommendations, key=impact_score, reverse=True)


# Constants
DEFAULT_PERFORMANCE_THRESHOLDS = {
    'execution_time': 1.0,  # seconds
    'memory_usage': 100.0,  # MB
    'cpu_utilization': 80.0,  # percentage
    'error_rate': 0.01  # percentage
}

DEFAULT_QUALITY_THRESHOLDS = {
    'cyclomatic_complexity': 10,
    'maintainability_index': 60.0,
    'code_coverage': 80.0,
    'duplication_percentage': 5.0
}

OPTIMIZATION_TYPE_WEIGHTS = {
    OptimizationType.SECURITY: 1.0,
    OptimizationType.PERFORMANCE: 0.95,
    OptimizationType.MAINTAINABILITY: 0.9,
    OptimizationType.ARCHITECTURE: 0.85,
    OptimizationType.CODE_QUALITY: 0.8,
    OptimizationType.DESIGN_PATTERN: 0.75,
    OptimizationType.ALGORITHM: 0.7,
    OptimizationType.TESTABILITY: 0.65,
    OptimizationType.READABILITY: 0.6,
    OptimizationType.REFACTORING: 0.55,
    OptimizationType.BEST_PRACTICES: 0.5,
    OptimizationType.MEMORY: 0.7
}