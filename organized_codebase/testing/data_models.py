"""
Data Models for Quality Assurance System
Extracted from agent_qa.py for better organization.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


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
    """Types of benchmarks for performance testing."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"


class ValidationType(Enum):
    """Types of validation checks."""
    FORMAT = "format"
    CONTENT = "content"
    STRUCTURE = "structure"
    LOGIC = "logic"
    PERFORMANCE = "performance"


class QualityLevel(Enum):
    """Quality levels for assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityThreshold:
    """Quality threshold configuration."""
    category: ScoreCategory
    excellent: float = 90.0
    good: float = 75.0
    acceptable: float = 60.0
    poor: float = 40.0
    alert_below: float = 50.0


@dataclass
class Alert:
    """Quality alert data."""
    type: AlertType
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    category: Optional[ScoreCategory] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    trend: Optional[str] = None


@dataclass
class QualityScore:
    """Quality score for a specific category."""
    category: ScoreCategory
    score: float
    level: QualityLevel
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Determine quality level based on score."""
        if self.score >= 90:
            self.level = QualityLevel.EXCELLENT
        elif self.score >= 75:
            self.level = QualityLevel.GOOD
        elif self.score >= 60:
            self.level = QualityLevel.ACCEPTABLE
        elif self.score >= 40:
            self.level = QualityLevel.POOR
        else:
            self.level = QualityLevel.CRITICAL


@dataclass
class BenchmarkResult:
    """Benchmark test result."""
    type: BenchmarkType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    baseline: Optional[float] = None
    
    @property
    def improvement_ratio(self) -> Optional[float]:
        """Calculate improvement ratio compared to baseline."""
        if self.baseline and self.baseline != 0:
            return self.value / self.baseline
        return None


@dataclass
class ValidationResult:
    """Validation check result."""
    type: ValidationType
    passed: bool
    message: str
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float
    category_scores: Dict[ScoreCategory, QualityScore]
    alerts: List[Alert]
    benchmarks: List[BenchmarkResult]
    validations: List[ValidationResult]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def quality_level(self) -> QualityLevel:
        """Overall quality level based on score."""
        if self.overall_score >= 90:
            return QualityLevel.EXCELLENT
        elif self.overall_score >= 75:
            return QualityLevel.GOOD
        elif self.overall_score >= 60:
            return QualityLevel.ACCEPTABLE
        elif self.overall_score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL