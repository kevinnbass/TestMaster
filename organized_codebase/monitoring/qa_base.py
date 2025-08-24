"""
Quality Assurance Base Classes and Enums
========================================

Core data structures for agent quality assurance system.
Extracted from agent_qa.py for modularization.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Optional, Union


class AlertType(Enum):
    """Types of quality alerts."""
    QUALITY_DEGRADATION = "quality_degradation"
    PERFORMANCE_ISSUE = "performance_issue"
    ERROR_SPIKE = "error_spike"
    THRESHOLD_BREACH = "threshold_breach"
    TREND_ANOMALY = "trend_anomaly"


class ScoreCategory(Enum):
    """Score categories for quality assessment."""
    FUNCTIONALITY = "functionality"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    USABILITY = "usability"


class BenchmarkType(Enum):
    """Types of performance benchmarks."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    ACCURACY = "accuracy"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"


class ValidationType(Enum):
    """Types of validation checks."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    FORMAT = "format"
    CONTENT = "content"
    STRUCTURE = "structure"
    PERFORMANCE = "performance"


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityThreshold:
    """Quality monitoring threshold configuration."""
    name: str
    metric: str
    value: float
    operator: str  # "gt", "lt", "eq"
    alert_type: AlertType
    severity: str  # "low", "medium", "high", "critical"


@dataclass
class QualityAlert:
    """Quality alert with detailed information."""
    alert_id: str
    alert_type: AlertType
    severity: str
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class QualityMetric:
    """Quality metric data point."""
    name: str
    value: float
    timestamp: datetime
    category: ScoreCategory
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric for benchmarking."""
    name: str
    value: float
    unit: str
    benchmark_type: BenchmarkType
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Validation rule configuration."""
    rule_id: str
    name: str
    validation_type: ValidationType
    pattern: Optional[str] = None
    function: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"
    enabled: bool = True


@dataclass
class ValidationIssue:
    """Issue found during validation."""
    issue_id: str
    validation_type: ValidationType
    severity: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation checks."""
    is_valid: bool
    validation_type: ValidationType
    issues: List[ValidationIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreWeight:
    """Weight configuration for scoring."""
    category: ScoreCategory
    weight: float
    enabled: bool = True


@dataclass
class ScoreBreakdown:
    """Breakdown of quality scores by category."""
    category: ScoreCategory
    score: float
    weight: float
    weighted_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityScore:
    """Overall quality score with breakdown."""
    total_score: float
    quality_level: QualityLevel
    breakdown: List[ScoreBreakdown]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of performance benchmark."""
    benchmark_type: BenchmarkType
    value: float
    baseline: float
    threshold: float
    passed: bool
    improvement: float  # Percentage improvement from baseline
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    report_id: str
    agent_id: str
    timestamp: datetime
    quality_score: QualityScore
    benchmark_results: List[BenchmarkResult]
    validation_results: List[ValidationResult]
    alerts: List[QualityAlert]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Export all classes
__all__ = [
    'AlertType', 'ScoreCategory', 'BenchmarkType', 'ValidationType', 'QualityLevel',
    'QualityThreshold', 'QualityAlert', 'QualityMetric', 'PerformanceMetric',
    'ValidationRule', 'ValidationIssue', 'ValidationResult', 'ScoreWeight',
    'ScoreBreakdown', 'QualityScore', 'BenchmarkResult', 'QualityReport'
]