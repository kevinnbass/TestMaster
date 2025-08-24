"""
Unified Agent Quality Assurance System for TestMaster

"""Core Module - Split from agent_qa.py"""


import time
import threading
import statistics
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Core Enums and Data Classes
# =============================================================================


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


# =============================================================================
# Data Classes
# =============================================================================

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
    agent_id: str
    alert_type: AlertType
    severity: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged: bool = False


@dataclass
class QualityMetric:
    """Quality metric data structure."""
    name: str
    value: float
    threshold: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric for benchmarking."""
    name: str
    value: float
    unit: str
    baseline: float
    threshold: float
    status: str


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    type: ValidationType
    description: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationIssue:
    """Validation issue details."""
    rule_name: str
    severity: str
    message: str
    location: str = ""
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Validation result with issues and scoring."""
    agent_id: str
    passed: bool
    score: float
    issues: List[ValidationIssue]
    total_checks: int = 0
    passed_checks: int = 0
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue to result."""
        self.issues.append(issue)


@dataclass
class ScoreWeight:
    """Score weight configuration for categories."""
    category: ScoreCategory
    weight: float
    description: str


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown by category."""
    category: ScoreCategory
    score: float
    weight: float
    weighted_score: float
    details: Dict[str, Any]


@dataclass
class QualityScore:
    """Comprehensive quality score result."""
    agent_id: str
    overall_score: float
    breakdown: List[ScoreBreakdown]
    status: str
    grade: str = ""
    percentile: float = 0.0
    
    def __post_init__(self):
        self.grade = self._calculate_grade()
    
    def _calculate_grade(self) -> str:
        """Calculate letter grade from overall score."""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.85:
            return "A-"
        elif self.overall_score >= 0.8:
            return "B+"
        elif self.overall_score >= 0.75:
            return "B"
        elif self.overall_score >= 0.7:
            return "B-"
        elif self.overall_score >= 0.65:
            return "C+"
        elif self.overall_score >= 0.6:
            return "C"
        elif self.overall_score >= 0.55:
            return "C-"
        elif self.overall_score >= 0.5:
            return "D"
        else:
            return "F"


@dataclass
class BenchmarkResult:
    """Benchmark execution result."""
    agent_id: str
    metrics: List[PerformanceMetric]
    overall_score: float
    status: str
    duration_ms: float = 0.0
    iterations: int = 0


@dataclass
class QualityReport:
    """Comprehensive quality inspection report."""
    agent_id: str
    metrics: List[QualityMetric]
    overall_score: float
    status: str
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Core Agent QA System
# =============================================================================

class AgentQualityAssurance:
    """
    Unified Agent Quality Assurance System
    
    Provides comprehensive quality monitoring, scoring, benchmarking,
    and validation for AI agent operations. Designed to work standalone
    without external dependencies.
    """
    
    def __init__(self, enable_monitoring: bool = True):
        """
        Initialize the Agent QA system.
        
        Args:
            enable_monitoring: Enable continuous quality monitoring
        """
        self.lock = threading.RLock()
        self.enabled = True  # Standalone - always enabled
        
        # Monitoring components
        self.monitoring = False
        self.monitor_thread = None
        self.monitoring_interval = 30.0  # seconds
        
        # Data storage
        self.alerts: List[QualityAlert] = []
        self.thresholds: List[QualityThreshold] = []
        self.agent_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        
        # Scoring and validation
        self.scoring_history: Dict[str, List[QualityScore]] = {}
        self.benchmark_history: Dict[str, List[BenchmarkResult]] = {}
        self.validation_history: Dict[str, List[ValidationResult]] = {}
        self.inspection_history: Dict[str, List[QualityReport]] = {}
        
        # Configuration
        self.score_weights = self._setup_default_weights()
        self.benchmarks: Dict[str, float] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.performance_thresholds = self._setup_performance_thresholds()
        self.quality_standards = self._setup_quality_standards()
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        
        # Initialize system
        self._setup_default_thresholds()
        self._setup_default_validation_rules()
        
        if enable_monitoring:
            self.start_monitoring()
        
        self._log("Agent Quality Assurance system initialized")
        self._log(f"   Monitoring: {'enabled' if enable_monitoring else 'disabled'}")
        self._log(f"   Score categories: {[w.category.value for w in self.score_weights]}")
        self._log(f"   Benchmark types: {[bt.value for bt in BenchmarkType]}")
    
    def _log(self, message: str):
        """Internal logging method."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] AgentQA: {message}")
    
    # =========================================================================
    # Configuration Setup
    # =========================================================================
    
    def _setup_default_thresholds(self):
        """Setup default quality monitoring thresholds."""
        self.thresholds = [
            QualityThreshold(
                name="low_quality_score",
                metric="overall_score",
                value=0.6,
                operator="lt",
                alert_type=AlertType.QUALITY_DEGRADATION,
                severity="high"
            ),
            QualityThreshold(
                name="poor_response_time",
                metric="response_time",
                value=500.0,
                operator="gt",
                alert_type=AlertType.PERFORMANCE_ISSUE,
                severity="medium"
            ),
            QualityThreshold(
                name="high_error_rate",
                metric="error_rate",
                value=0.05,
                operator="gt",
                alert_type=AlertType.ERROR_SPIKE,
                severity="high"
            ),
            QualityThreshold(
                name="memory_usage_high",
                metric="memory_usage",
                value=200.0,
                operator="gt",
                alert_type=AlertType.THRESHOLD_BREACH,
                severity="medium"
            )
        ]
    
    def _setup_default_weights(self) -> List[ScoreWeight]:
        """Setup default scoring weights."""
        return [
            ScoreWeight(
                category=ScoreCategory.FUNCTIONALITY,
                weight=0.25,
                description="Correctness and feature completeness"
            ),
            ScoreWeight(
                category=ScoreCategory.RELIABILITY,
                weight=0.20,
                description="Consistency and error handling"
            ),
            ScoreWeight(
                category=ScoreCategory.PERFORMANCE,
                weight=0.20,
                description="Speed and resource efficiency"
            ),
            ScoreWeight(
                category=ScoreCategory.SECURITY,
                weight=0.15,
                description="Security and vulnerability assessment"
            ),
            ScoreWeight(
                category=ScoreCategory.MAINTAINABILITY,
                weight=0.15,
                description="Code quality and maintainability"
            ),
            ScoreWeight(
                category=ScoreCategory.USABILITY,
                weight=0.05,
                description="User experience and accessibility"
            )
        ]
    
    def _setup_performance_thresholds(self) -> Dict[str, float]:
        """Setup default performance thresholds for benchmarking."""
        return {
            "response_time_ms": 200.0,
            "throughput_ops_sec": 100.0,
            "memory_usage_mb": 100.0,
            "cpu_utilization_percent": 80.0,
            "accuracy_percent": 95.0,
            "reliability_percent": 99.0
        }
    
    def _setup_quality_standards(self) -> Dict[str, float]:
        """Setup quality standards for inspection."""
        return {
            "syntax_score": 0.9,
            "performance_score": 0.8,
            "security_score": 0.95,
            "reliability_score": 0.85
        }
    
    def _setup_default_validation_rules(self):
        """Setup default validation rules."""
        # Syntax validation rules
        syntax_rules = [
            ValidationRule(
                name="no_syntax_errors",
                type=ValidationType.SYNTAX,
                description="Check for syntax errors",
                validator=lambda x: self._check_syntax(x),
                error_message="Syntax errors detected"
            ),
            ValidationRule(
                name="proper_indentation",
                type=ValidationType.SYNTAX,
                description="Check for proper indentation",
                validator=lambda x: self._check_indentation(x),
                error_message="Improper indentation detected"
            )
        ]
        
        # Format validation rules
        format_rules = [
            ValidationRule(
                name="valid_json",
                type=ValidationType.FORMAT,
                description="Check if output is valid JSON when expected",
                validator=lambda x: self._check_json_format(x),
                error_message="Invalid JSON format"
            ),
            ValidationRule(
                name="consistent_naming",
                type=ValidationType.FORMAT,
                description="Check for consistent naming conventions",
                validator=lambda x: self._check_naming_consistency(x),
                error_message="Inconsistent naming conventions"
            )
        ]
        
        # Content validation rules
        content_rules = [
            ValidationRule(
                name="no_empty_output",
                type=ValidationType.CONTENT,
                description="Check that output is not empty",
                validator=lambda x: self._check_not_empty(x),
                error_message="Output is empty"
            ),
            ValidationRule(
                name="relevant_content",
                type=ValidationType.CONTENT,
                description="Check content relevance",
                validator=lambda x: self._check_content_relevance(x),
                error_message="Content appears irrelevant"
            )
        ]
        
        # Performance validation rules
        performance_rules = [
            ValidationRule(
                name="response_time",
                type=ValidationType.PERFORMANCE,
                description="Check response time",
                validator=lambda x: self._check_response_time(x),
                error_message="Response time exceeds threshold",
                severity="warning"
            )
        ]
        
        self.validation_rules = {
            "syntax": syntax_rules,
            "format": format_rules,
            "content": content_rules,
            "performance": performance_rules
        }
    
    # =========================================================================
    # Quality Monitoring
    # =========================================================================
    
    def start_monitoring(self):
        """Start continuous quality monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self._log("Quality monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous quality monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self._log("Quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for continuous quality assessment."""
        while self.monitoring:
            try:
                self._check_all_agents()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self._log(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Brief pause before retrying
    
    def _check_all_agents(self):
        """Check quality for all monitored agents."""
        with self.lock:
            for agent_id in self.agent_metrics.keys():
                self._check_agent_quality(agent_id)
    
    def _check_agent_quality(self, agent_id: str):
        """Check quality thresholds for a specific agent."""
        agent_data = self.agent_metrics.get(agent_id, {})
        
        for threshold in self.thresholds:
            metric_values = agent_data.get(threshold.metric, [])
            if not metric_values:
                continue
            
            current_value = metric_values[-1]  # Latest value
            
            # Check threshold breach
            if self._check_threshold_breach(current_value, threshold):
                alert = self._create_alert(agent_id, threshold, current_value)
                self._raise_alert(alert)
            
            # Check for trend anomalies
            if len(metric_values) >= 5:
                if self._detect_trend_anomaly(metric_values, threshold):
                    alert = self._create_trend_alert(agent_id, threshold, metric_values)
                    self._raise_alert(alert)
    
    def _check_threshold_breach(self, value: float, threshold: QualityThreshold) -> bool:
        """Check if value breaches threshold."""
        if threshold.operator == "gt":
            return value > threshold.value
        elif threshold.operator == "lt":
            return value < threshold.value
        elif threshold.operator == "eq":
            return abs(value - threshold.value) < 0.01
        return False
    
    def _detect_trend_anomaly(self, values: List[float], threshold: QualityThreshold) -> bool:
        """Detect trend anomalies in metric values."""
        if len(values) < 5:
            return False
        
        # Simple trend detection - check if recent values show concerning trend
        recent_values = values[-3:]
        older_values = values[-5:-3]
        
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        # Check for significant degradation
        if threshold.metric in ["overall_score", "accuracy"]:
            # Higher is better - check for decline
            decline_threshold = 0.1  # 10% decline
            return (older_avg - recent_avg) / older_avg > decline_threshold
        else:
            # Lower is better - check for increase
            increase_threshold = 0.2  # 20% increase
            return (recent_avg - older_avg) / older_avg > increase_threshold
    
    def _create_alert(self, agent_id: str, threshold: QualityThreshold, current_value: float) -> QualityAlert:
        """Create quality alert for threshold breach."""
        alert_id = f"alert_{int(time.time())}_{agent_id}_{threshold.name}"
        
        message = f"Agent {agent_id}: {threshold.metric} {threshold.operator} {threshold.value} (current: {current_value:.3f})"
        
        return QualityAlert(
            alert_id=alert_id,
            agent_id=agent_id,
            alert_type=threshold.alert_type,
            severity=threshold.severity,
            message=message,
            metric_name=threshold.metric,
            current_value=current_value,
            threshold_value=threshold.value,
            timestamp=datetime.now()
        )
    
    def _create_trend_alert(self, agent_id: str, threshold: QualityThreshold, values: List[float]) -> QualityAlert:
        """Create trend anomaly alert."""
        alert_id = f"trend_{int(time.time())}_{agent_id}_{threshold.metric}"
        
        recent_avg = sum(values[-3:]) / 3
        message = f"Agent {agent_id}: Trend anomaly detected in {threshold.metric} (recent avg: {recent_avg:.3f})"
        
        return QualityAlert(
            alert_id=alert_id,
            agent_id=agent_id,
            alert_type=AlertType.TREND_ANOMALY,
            severity="medium",
            message=message,
            metric_name=threshold.metric,
            current_value=recent_avg,
            threshold_value=threshold.value,
            timestamp=datetime.now()
        )
    
    def _raise_alert(self, alert: QualityAlert):
        """Raise quality alert and notify callbacks."""
        # Check for duplicate alerts (avoid spam)
        recent_alerts = [a for a in self.alerts if 
                        a.agent_id == alert.agent_id and 
                        a.metric_name == alert.metric_name and
                        (datetime.now() - a.timestamp).seconds < 300]  # 5 minutes
        
        if recent_alerts:
            return  # Skip duplicate alert
        
        with self.lock:
            self.alerts.append(alert)
        
        self._log(f"[{alert.severity.upper()}] Quality Alert: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self._log(f"Error in alert callback: {e}")
    
    def record_metric(self, agent_id: str, metric_name: str, value: float):
        """Record metric value for quality monitoring."""
        with self.lock:
            if agent_id not in self.agent_metrics:
                self.agent_metrics[agent_id] = {}
            
            if metric_name not in self.agent_metrics[agent_id]:
                self.agent_metrics[agent_id][metric_name] = []
            
            # Keep only recent values (last 100 measurements)
            metrics_list = self.agent_metrics[agent_id][metric_name]
            metrics_list.append(value)
            if len(metrics_list) > 100:
                metrics_list.pop(0)
    
    # =========================================================================
    # Quality Inspection
    # =========================================================================
    
    def inspect_agent(
        self,
        agent_id: str,
        test_cases: List[Dict[str, Any]] = None,
        include_benchmarks: bool = True
    ) -> QualityReport:
        """
        Perform comprehensive quality inspection of an agent.
        
        Args:
            agent_id: Agent identifier
            test_cases: Test cases for validation
            include_benchmarks: Include performance benchmarks
            
        Returns:
            Detailed quality report with recommendations
        """
        start_time = time.time()
        metrics = []
        
        # Syntax validation
        syntax_metric = self._check_syntax_quality(agent_id, test_cases)
        metrics.append(syntax_metric)
        
        # Semantic analysis
        semantic_metric = self._check_semantic_quality(agent_id, test_cases)
        metrics.append(semantic_metric)
        
        # Performance testing
        if include_benchmarks:
            performance_metric = self._check_performance_quality(agent_id)
            metrics.append(performance_metric)
        
        # Security scanning
        security_metric = self._check_security_quality(agent_id)
        metrics.append(security_metric)
        
        # Reliability testing
        reliability_metric = self._check_reliability_quality(agent_id)
        metrics.append(reliability_metric)
        
        # Calculate overall score
        overall_score = self._calculate_inspection_score(metrics)
        
        # Determine status
        status = self._determine_quality_status(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        report = QualityReport(
            agent_id=agent_id,
            metrics=metrics,
            overall_score=overall_score,
            status=status,
            recommendations=recommendations
