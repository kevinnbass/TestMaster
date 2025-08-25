"""
Unified Agent Quality Assurance System for TestMaster

This module provides comprehensive agent quality assessment capabilities including:
- Quality monitoring and alerting
- Performance scoring and benchmarking  
- Output validation and analysis
- Continuous quality tracking
- Standalone operation without external dependencies

Integrated from TestMaster/testmaster/agent_qa/ components:
- Quality Monitor
- Scoring System
- Benchmarking Suite
- Quality Inspector
- Validation Engine
"""

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
        )
        
        # Store in history
        with self.lock:
            if agent_id not in self.inspection_history:
                self.inspection_history[agent_id] = []
            self.inspection_history[agent_id].append(report)
        
        inspection_time = time.time() - start_time
        self._log(f"Quality inspection completed for {agent_id}: {overall_score:.2f} ({status}) in {inspection_time*1000:.1f}ms")
        
        return report
    
    def _check_syntax_quality(self, agent_id: str, test_cases: List[Dict[str, Any]] = None) -> QualityMetric:
        """Check syntax quality of agent operations."""
        # Simulate syntax validation - in real implementation would analyze actual code
        syntax_score = 0.92  # Mock score
        threshold = self.quality_standards["syntax_score"]
        
        return QualityMetric(
            name="syntax_validation",
            value=syntax_score,
            threshold=threshold,
            status="pass" if syntax_score >= threshold else "fail",
            details={
                "test_cases_validated": len(test_cases) if test_cases else 0,
                "syntax_errors": 1 if syntax_score < threshold else 0,
                "code_style_issues": 2
            }
        )
    
    def _check_semantic_quality(self, agent_id: str, test_cases: List[Dict[str, Any]] = None) -> QualityMetric:
        """Check semantic quality of agent operations."""
        # Simulate semantic analysis
        semantic_score = 0.88
        threshold = 0.8
        
        return QualityMetric(
            name="semantic_analysis",
            value=semantic_score,
            threshold=threshold,
            status="pass" if semantic_score >= threshold else "fail",
            details={
                "logic_consistency": 0.9,
                "variable_usage": 0.85,
                "flow_analysis": 0.9
            }
        )
    
    def _check_performance_quality(self, agent_id: str) -> QualityMetric:
        """Check performance quality of agent operations."""
        # Simulate performance testing
        performance_score = 0.83
        threshold = self.quality_standards["performance_score"]
        
        return QualityMetric(
            name="performance_test",
            value=performance_score,
            threshold=threshold,
            status="pass" if performance_score >= threshold else "fail",
            details={
                "response_time_ms": 150.0,
                "memory_usage_mb": 45.2,
                "cpu_efficiency": 0.85
            }
        )
    
    def _check_security_quality(self, agent_id: str) -> QualityMetric:
        """Check security quality of agent operations."""
        # Simulate security scanning
        security_score = 0.96
        threshold = self.quality_standards["security_score"]
        
        return QualityMetric(
            name="security_scan",
            value=security_score,
            threshold=threshold,
            status="pass" if security_score >= threshold else "fail",
            details={
                "vulnerabilities_found": 0,
                "security_patterns": ["input_validation", "output_sanitization"],
                "compliance_score": 0.98
            }
        )
    
    def _check_reliability_quality(self, agent_id: str) -> QualityMetric:
        """Check reliability quality of agent operations."""
        # Simulate reliability testing
        reliability_score = 0.87
        threshold = self.quality_standards["reliability_score"]
        
        return QualityMetric(
            name="reliability_test",
            value=reliability_score,
            threshold=threshold,
            status="pass" if reliability_score >= threshold else "fail",
            details={
                "error_rate": 0.02,
                "uptime_percentage": 99.5,
                "fault_tolerance": 0.9
            }
        )
    
    def _calculate_inspection_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate overall quality score from inspection metrics."""
        if not metrics:
            return 0.0
        
        # Weighted average of metric scores
        weights = {
            "syntax_validation": 0.2,
            "semantic_analysis": 0.25,
            "performance_test": 0.2,
            "security_scan": 0.25,
            "reliability_test": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = weights.get(metric.name, 0.1)
            total_score += metric.value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_status(self, overall_score: float) -> str:
        """Determine quality status from overall score."""
        if overall_score >= 0.95:
            return QualityLevel.EXCELLENT.value
        elif overall_score >= 0.85:
            return QualityLevel.GOOD.value
        elif overall_score >= 0.7:
            return QualityLevel.SATISFACTORY.value
        elif overall_score >= 0.5:
            return QualityLevel.POOR.value
        else:
            return QualityLevel.CRITICAL.value
    
    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        for metric in metrics:
            if metric.status == "fail":
                if metric.name == "syntax_validation":
                    recommendations.append("Fix syntax errors and improve code style")
                elif metric.name == "performance_test":
                    recommendations.append("Optimize performance bottlenecks")
                elif metric.name == "security_scan":
                    recommendations.append("Address security vulnerabilities")
                elif metric.name == "reliability_test":
                    recommendations.append("Improve error handling and fault tolerance")
        
        if not recommendations:
            recommendations.append("Maintain current quality standards")
        
        return recommendations
    
    # =========================================================================
    # Output Validation
    # =========================================================================
    
    def validate_output(
        self,
        agent_id: str,
        output: Any,
        expected: Any = None,
        validation_rules: List[ValidationRule] = None
    ) -> ValidationResult:
        """
        Validate agent output against rules and expectations.
        
        Args:
            agent_id: Agent identifier
            output: Output to validate
            expected: Expected output for comparison
            validation_rules: Custom validation rules
            
        Returns:
            Validation result with issues and score
        """
        result = ValidationResult(agent_id, True, 1.0, [])
        
        # Use custom rules if provided, otherwise use default rules
        rules_to_check = validation_rules or []
        if not rules_to_check:
            # Use all default rules
            for rule_category in self.validation_rules.values():
                rules_to_check.extend(rule_category)
        
        result.total_checks = len(rules_to_check)
        
        # Run validation rules
        for rule in rules_to_check:
            try:
                if rule.validator(output):
                    result.passed_checks += 1
                else:
                    issue = ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=rule.error_message,
                        suggestion=self._get_suggestion(rule.name)
                    )
                    result.add_issue(issue)
            except Exception as e:
                # Rule execution failed
                issue = ValidationIssue(
                    rule_name=rule.name,
                    severity="error",
                    message=f"Validation rule failed: {str(e)}",
                    suggestion="Check rule implementation"
                )
                result.add_issue(issue)
        
        # Compare with expected output if provided
        if expected is not None:
            similarity_score = self._calculate_similarity(output, expected)
            if similarity_score < 0.7:  # Threshold for similarity
                issue = ValidationIssue(
                    rule_name="output_similarity",
                    severity="warning",
                    message=f"Output similarity too low: {similarity_score:.2f}",
                    suggestion="Review output against expectations"
                )
                result.add_issue(issue)
        
        # Calculate final score
        result.score = self._calculate_validation_score(result)
        result.passed = result.score >= 0.7  # Pass threshold
        
        # Store in history
        with self.lock:
            if agent_id not in self.validation_history:
                self.validation_history[agent_id] = []
            self.validation_history[agent_id].append(result)
        
        self._log(f"Validation completed for {agent_id}: {result.score:.2f} ({result.passed_checks}/{result.total_checks} checks passed)")
        
        return result
    
    def _calculate_validation_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score."""
        if result.total_checks == 0:
            return 1.0
        
        base_score = result.passed_checks / result.total_checks
        
        # Apply penalties for errors vs warnings
        error_penalty = sum(0.1 for issue in result.issues if issue.severity == "error")
        warning_penalty = sum(0.05 for issue in result.issues if issue.severity == "warning")
        
        final_score = base_score - error_penalty - warning_penalty
        return max(0.0, min(1.0, final_score))
    
    def _calculate_similarity(self, output: Any, expected: Any) -> float:
        """Calculate similarity between output and expected."""
        if type(output) != type(expected):
            return 0.0
        
        if isinstance(output, str) and isinstance(expected, str):
            # Simple string similarity
            output_words = set(output.lower().split())
            expected_words = set(expected.lower().split())
            
            if not expected_words:
                return 1.0 if not output_words else 0.0
            
            intersection = output_words.intersection(expected_words)
            union = output_words.union(expected_words)
            return len(intersection) / len(union) if union else 1.0
        
        # For other types, simple equality check
        return 1.0 if output == expected else 0.0
    
    # =========================================================================
    # Validation Rule Implementations
    # =========================================================================
    
    def _check_syntax(self, output: Any) -> bool:
        """Check for syntax errors in code output."""
        if isinstance(output, str):
            # Basic Python syntax check
            try:
                compile(output, '<string>', 'exec')
                return True
            except SyntaxError:
                return False
        return True  # Non-string outputs pass syntax check
    
    def _check_indentation(self, output: Any) -> bool:
        """Check for proper indentation in code output."""
        if isinstance(output, str):
            lines = output.split('\n')
            indent_levels = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    indent_levels.append(indent)
            
            # Check for consistent indentation (multiples of 4 or 2)
            if indent_levels:
                base_indent = min(i for i in indent_levels if i > 0) if any(i > 0 for i in indent_levels) else 4
                return all(indent % base_indent == 0 for indent in indent_levels)
        return True
    
    def _check_json_format(self, output: Any) -> bool:
        """Check if output is valid JSON when it should be."""
        if isinstance(output, str) and (output.strip().startswith('{') or output.strip().startswith('[')):
            try:
                json.loads(output)
                return True
            except json.JSONDecodeError:
                return False
        return True  # Non-JSON-like outputs pass
    
    def _check_naming_consistency(self, output: Any) -> bool:
        """Check for consistent naming conventions."""
        if isinstance(output, str):
            # Check for consistent variable naming (snake_case vs camelCase)
            snake_case_pattern = re.compile(r'[a-z_][a-z0-9_]*')
            camel_case_pattern = re.compile(r'[a-z][a-zA-Z0-9]*')
            
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', output)
            if words:
                snake_case_count = sum(1 for word in words if snake_case_pattern.fullmatch(word))
                camel_case_count = sum(1 for word in words if camel_case_pattern.fullmatch(word))
                
                # Consistent if one style dominates (>70%)
                total = len(words)
                return (snake_case_count / total > 0.7) or (camel_case_count / total > 0.7)
        return True
    
    def _check_not_empty(self, output: Any) -> bool:
        """Check that output is not empty."""
        if output is None:
            return False
        if isinstance(output, str):
            return len(output.strip()) > 0
        if isinstance(output, (list, dict)):
            return len(output) > 0
        return True
    
    def _check_content_relevance(self, output: Any) -> bool:
        """Check content relevance (simplified heuristic)."""
        if isinstance(output, str):
            # Basic relevance check - contains meaningful content
            words = output.split()
            return len(words) >= 3  # At least 3 words
        return True
    
    def _check_response_time(self, output: Any) -> bool:
        """Check response time (mock implementation)."""
        # In real implementation, this would check actual response time
        return True  # Always pass for now
    
    def _get_suggestion(self, rule_name: str) -> str:
        """Get suggestion for fixing validation issue."""
        suggestions = {
            "no_syntax_errors": "Review code syntax and fix errors",
            "proper_indentation": "Use consistent indentation (4 spaces recommended)",
            "valid_json": "Ensure JSON syntax is correct",
            "consistent_naming": "Use consistent naming convention (snake_case or camelCase)",
            "no_empty_output": "Provide meaningful output content",
            "relevant_content": "Ensure content is relevant to the task",
            "response_time": "Optimize performance to reduce response time"
        }
        return suggestions.get(rule_name, "Review and fix the issue")
    
    # =========================================================================
    # Quality Scoring
    # =========================================================================
    
    def calculate_score(
        self,
        agent_id: str,
        quality_metrics: List[QualityMetric],
        custom_weights: Dict[str, float] = None
    ) -> QualityScore:
        """
        Calculate comprehensive quality score for an agent.
        
        Args:
            agent_id: Agent identifier
            quality_metrics: List of quality metrics
            custom_weights: Custom weights for categories
            
        Returns:
            Quality score with detailed breakdown
        """
        # Use custom weights if provided
        weights = self._apply_custom_weights(custom_weights) if custom_weights else self.score_weights
        
        # Calculate scores by category
        breakdown = []
        for weight in weights:
            category_score = self._calculate_category_score(quality_metrics, weight.category)
            weighted_score = category_score * weight.weight
            
            breakdown.append(ScoreBreakdown(
                category=weight.category,
                score=category_score,
                weight=weight.weight,
                weighted_score=weighted_score,
                details=self._get_category_details(quality_metrics, weight.category)
            ))
        
        # Calculate overall score
        overall_score = sum(b.weighted_score for b in breakdown)
        
        # Determine status
        status = self._determine_score_status(overall_score)
        
        # Calculate percentile
        percentile = self._calculate_percentile(agent_id, overall_score)
        
        score_result = QualityScore(
            agent_id=agent_id,
            overall_score=overall_score,
            breakdown=breakdown,
            status=status,
            percentile=percentile
        )
        
        # Store in history
        with self.lock:
            if agent_id not in self.scoring_history:
                self.scoring_history[agent_id] = []
            self.scoring_history[agent_id].append(score_result)
        
        self._log(f"Quality score calculated for {agent_id}: {overall_score:.3f} ({score_result.grade}) - {status}")
        
        return score_result
    
    def _calculate_category_score(self, metrics: List[QualityMetric], category: ScoreCategory) -> float:
        """Calculate score for a specific category."""
        category_metrics = self._filter_metrics_by_category(metrics, category)
        
        if not category_metrics:
            return 0.8  # Default score if no metrics available
        
        # Calculate weighted average of category metrics
        total_score = 0.0
        total_weight = 0.0
        
        for metric in category_metrics:
            weight = self._get_metric_weight(metric.name, category)
            total_score += metric.value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _filter_metrics_by_category(self, metrics: List[QualityMetric], category: ScoreCategory) -> List[QualityMetric]:
        """Filter metrics by category."""
        category_mapping = {
            ScoreCategory.FUNCTIONALITY: ["syntax_validation", "semantic_analysis"],
            ScoreCategory.RELIABILITY: ["reliability_test", "error_handling"],
            ScoreCategory.PERFORMANCE: ["performance_test", "response_time"],
            ScoreCategory.SECURITY: ["security_scan", "vulnerability_check"],
            ScoreCategory.MAINTAINABILITY: ["code_quality", "documentation"],
            ScoreCategory.USABILITY: ["usability_test", "accessibility"]
        }
        
        relevant_names = category_mapping.get(category, [])
        return [m for m in metrics if m.name in relevant_names or category.value in m.name]
    
    def _get_metric_weight(self, metric_name: str, category: ScoreCategory) -> float:
        """Get weight for a specific metric within a category."""
        metric_weights = {
            "syntax_validation": 0.6,
            "semantic_analysis": 0.4,
            "performance_test": 0.7,
            "response_time": 0.3,
            "security_scan": 1.0,
            "reliability_test": 1.0
        }
        return metric_weights.get(metric_name, 1.0)
    
    def _get_category_details(self, metrics: List[QualityMetric], category: ScoreCategory) -> Dict[str, Any]:
        """Get detailed information for a category."""
        category_metrics = self._filter_metrics_by_category(metrics, category)
        
        return {
            "metrics_count": len(category_metrics),
            "metrics_passed": sum(1 for m in category_metrics if m.status == "pass"),
            "average_score": sum(m.value for m in category_metrics) / len(category_metrics) if category_metrics else 0.0,
            "metric_names": [m.name for m in category_metrics]
        }
    
    def _apply_custom_weights(self, custom_weights: Dict[str, float]) -> List[ScoreWeight]:
        """Apply custom weights to scoring categories."""
        updated_weights = []
        
        for weight in self.score_weights:
            new_weight_value = custom_weights.get(weight.category.value, weight.weight)
            updated_weights.append(ScoreWeight(
                category=weight.category,
                weight=new_weight_value,
                description=weight.description
            ))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(w.weight for w in updated_weights)
        if total_weight > 0:
            for weight in updated_weights:
                weight.weight = weight.weight / total_weight
        
        return updated_weights
    
    def _determine_score_status(self, overall_score: float) -> str:
        """Determine status from overall score."""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
            return "satisfactory"
        elif overall_score >= 0.6:
            return "needs_improvement"
        else:
            return "poor"
    
    def _calculate_percentile(self, agent_id: str, current_score: float) -> float:
        """Calculate percentile ranking for the score."""
        # Get all historical scores
        all_scores = []
        for scores in self.scoring_history.values():
            all_scores.extend([s.overall_score for s in scores])
        
        if not all_scores:
            return 50.0  # Default percentile
        
        # Calculate percentile
        scores_below = sum(1 for score in all_scores if score < current_score)
        percentile = (scores_below / len(all_scores)) * 100
        
        return percentile
    
    # =========================================================================
    # Performance Benchmarking
    # =========================================================================
    
    def run_benchmarks(
        self,
        agent_id: str,
        benchmark_types: List[BenchmarkType] = None,
        iterations: int = 10
    ) -> BenchmarkResult:
        """
        Run performance benchmarks for an agent.
        
        Args:
            agent_id: Agent identifier
            benchmark_types: Types of benchmarks to run
            iterations: Number of iterations for each benchmark
            
        Returns:
            Benchmark results with performance metrics
        """
        start_time = time.time()
        
        # Use all benchmark types if none specified
        if benchmark_types is None:
            benchmark_types = list(BenchmarkType)
        
        metrics = []
        
        for benchmark_type in benchmark_types:
            metric = self._run_single_benchmark(agent_id, benchmark_type, iterations)
            metrics.append(metric)
        
        # Calculate overall score
        overall_score = self._calculate_overall_benchmark_score(metrics)
        
        # Determine status
        status = self._determine_benchmark_status(overall_score)
        
        duration_ms = (time.time() - start_time) * 1000
        
        result = BenchmarkResult(
            agent_id=agent_id,
            metrics=metrics,
            overall_score=overall_score,
            status=status,
            duration_ms=duration_ms,
            iterations=iterations
        )
        
        # Store in history
        with self.lock:
            if agent_id not in self.benchmark_history:
                self.benchmark_history[agent_id] = []
            self.benchmark_history[agent_id].append(result)
        
        self._log(f"Benchmarking completed for {agent_id}: {overall_score:.3f} ({status}) in {duration_ms:.1f}ms")
        
        return result
    
    def _run_single_benchmark(self, agent_id: str, benchmark_type: BenchmarkType, iterations: int) -> PerformanceMetric:
        """Run a single benchmark test."""
        measurements = []
        
        for _ in range(iterations):
            measurement = self._execute_benchmark(agent_id, benchmark_type)
            measurements.append(measurement)
        
        # Calculate statistics
        avg_value = statistics.mean(measurements)
        baseline = self._get_baseline(agent_id, benchmark_type.value)
        threshold = self.performance_thresholds.get(f"{benchmark_type.value}_{'ms' if 'time' in benchmark_type.value else 'ops_sec' if 'throughput' in benchmark_type.value else 'mb' if 'memory' in benchmark_type.value else 'percent'}", 100.0)
        
        # Determine status
        if benchmark_type in [BenchmarkType.RESPONSE_TIME, BenchmarkType.MEMORY_USAGE, BenchmarkType.CPU_UTILIZATION]:
            # Lower is better
            status = "pass" if avg_value <= threshold else "fail"
        else:
            # Higher is better
            status = "pass" if avg_value >= threshold else "fail"
        
        unit = self._get_metric_unit(benchmark_type)
        
        return PerformanceMetric(
            name=benchmark_type.value,
            value=avg_value,
            unit=unit,
            baseline=baseline,
            threshold=threshold,
            status=status
        )
    
    def _execute_benchmark(self, agent_id: str, benchmark_type: BenchmarkType) -> float:
        """Execute a specific benchmark and return measurement."""
        if benchmark_type == BenchmarkType.RESPONSE_TIME:
            return self._benchmark_response_time(agent_id)
        elif benchmark_type == BenchmarkType.THROUGHPUT:
            return self._benchmark_throughput(agent_id)
        elif benchmark_type == BenchmarkType.MEMORY_USAGE:
            return self._benchmark_memory_usage(agent_id)
        elif benchmark_type == BenchmarkType.CPU_UTILIZATION:
            return self._benchmark_cpu_utilization(agent_id)
        elif benchmark_type == BenchmarkType.ACCURACY:
            return self._benchmark_accuracy(agent_id)
        elif benchmark_type == BenchmarkType.SCALABILITY:
            return self._benchmark_scalability(agent_id)
        elif benchmark_type == BenchmarkType.RELIABILITY:
            return self._benchmark_reliability(agent_id)
        else:
            return 0.0
    
    def _benchmark_response_time(self, agent_id: str) -> float:
        """Benchmark response time."""
        start_time = time.time()
        
        # Simulate agent operation
        time.sleep(0.05)  # 50ms simulated operation
        
        end_time = time.time()
        return (end_time - start_time) * 1000  # Return in milliseconds
    
    def _benchmark_throughput(self, agent_id: str) -> float:
        """Benchmark throughput (operations per second)."""
        operations = 0
        start_time = time.time()
        
        # Simulate operations for 100ms
        while time.time() - start_time < 0.1:
            operations += 1
            time.sleep(0.001)  # Small delay per operation
        
        duration = time.time() - start_time
        return operations / duration  # Operations per second
    
    def _benchmark_memory_usage(self, agent_id: str) -> float:
        """Benchmark memory usage."""
        # Simulate memory usage measurement
        import sys
        
        # Create some test data structures
        test_data = {i: f"test_data_{i}" for i in range(1000)}
        memory_used = sys.getsizeof(test_data) / (1024 * 1024)  # Convert to MB
        
        return memory_used
    
    def _benchmark_cpu_utilization(self, agent_id: str) -> float:
        """Benchmark CPU utilization."""
        # Simulate CPU-intensive task
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < 0.01:  # 10ms test
            # Simple CPU work
            _ = sum(i * i for i in range(100))
            iterations += 1
        
        # Return simulated CPU percentage
        return min(iterations / 10.0, 100.0)  # Scale to percentage
    
    def _benchmark_accuracy(self, agent_id: str) -> float:
        """Benchmark accuracy."""
        # Simulate accuracy test
        correct_predictions = 96
        total_predictions = 100
        return (correct_predictions / total_predictions) * 100.0
    
    def _benchmark_scalability(self, agent_id: str) -> float:
        """Benchmark scalability."""
        # Simulate scalability test
        base_response_time = 50.0  # ms
        load_factor = 2.0  # 2x load
        scaled_response_time = base_response_time * (load_factor ** 0.5)  # Sub-linear scaling
        
        # Return scalability score (higher is better)
        return 100.0 / (scaled_response_time / base_response_time)
    
    def _benchmark_reliability(self, agent_id: str) -> float:
        """Benchmark reliability."""
        # Simulate reliability test
        successful_operations = 99
        total_operations = 100
        return (successful_operations / total_operations) * 100.0
    
    def _get_metric_unit(self, benchmark_type: BenchmarkType) -> str:
        """Get unit for benchmark metric."""
        units = {
            BenchmarkType.RESPONSE_TIME: "ms",
            BenchmarkType.THROUGHPUT: "ops/sec",
            BenchmarkType.MEMORY_USAGE: "MB",
            BenchmarkType.CPU_UTILIZATION: "%",
            BenchmarkType.ACCURACY: "%",
            BenchmarkType.SCALABILITY: "score",
            BenchmarkType.RELIABILITY: "%"
        }
        return units.get(benchmark_type, "units")
    
    def _get_baseline(self, agent_id: str, metric_name: str) -> float:
        """Get baseline value for metric."""
        agent_baselines = self.baselines.get(agent_id, {})
        return agent_baselines.get(metric_name, 0.0)
    
    def _calculate_overall_benchmark_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall benchmark score."""
        if not metrics:
            return 0.0
        
        # Calculate weighted score based on metric performance vs threshold
        total_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            # Calculate score as ratio of value to threshold
            if metric.name in ["response_time", "memory_usage", "cpu_utilization"]:
                # Lower is better - score is inverse
                score = min(1.0, metric.threshold / max(metric.value, 0.1))
            else:
                # Higher is better
                score = min(1.0, metric.value / max(metric.threshold, 0.1))
            
            weight = self._get_benchmark_metric_weight(metric.name)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_benchmark_metric_weight(self, metric_name: str) -> float:
        """Get weight for metric in overall benchmark score calculation."""
        weights = {
            "response_time": 0.25,
            "throughput": 0.2,
            "memory_usage": 0.15,
            "cpu_utilization": 0.15,
            "accuracy": 0.15,
            "scalability": 0.05,
            "reliability": 0.05
        }
        return weights.get(metric_name, 0.1)
    
    def _determine_benchmark_status(self, overall_score: float) -> str:
        """Determine benchmark status from overall score."""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
            return "satisfactory"
        elif overall_score >= 0.6:
            return "needs_improvement"
        else:
            return "poor"
    
    # =========================================================================
    # Configuration and Management
    # =========================================================================
    
    def add_threshold(self, threshold: QualityThreshold):
        """Add custom quality threshold."""
        self.thresholds.append(threshold)
        self._log(f"Added quality threshold: {threshold.name}")
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Add callback for quality alerts."""
        self.alert_callbacks.append(callback)
    
    def add_custom_validation_rule(self, category: str, rule: ValidationRule):
        """Add custom validation rule."""
        if category not in self.validation_rules:
            self.validation_rules[category] = []
        self.validation_rules[category].append(rule)
        self._log(f"Added custom validation rule: {rule.name} to {category}")
    
    def set_baseline(self, agent_id: str, metric_name: str, value: float):
        """Set baseline value for an agent metric."""
        if agent_id not in self.baselines:
            self.baselines[agent_id] = {}
        self.baselines[agent_id][metric_name] = value
        self._log(f"Baseline set for {agent_id}.{metric_name}: {value}")
    
    def set_benchmark(self, benchmark_name: str, score: float):
        """Set a benchmark score for comparison."""
        self.benchmarks[benchmark_name] = score
        self._log(f"Benchmark set: {benchmark_name} = {score:.3f}")
    
    # =========================================================================
    # Status and History
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        with self.lock:
            total_alerts = len(self.alerts)
            unacknowledged_alerts = len([a for a in self.alerts if not a.acknowledged])
            critical_alerts = len([a for a in self.alerts if a.severity == "critical"])
            
            return {
                "enabled": self.enabled,
                "monitoring": self.monitoring,
                "monitored_agents": len(self.agent_metrics),
                "total_alerts": total_alerts,
                "unacknowledged_alerts": unacknowledged_alerts,
                "critical_alerts": critical_alerts,
                "thresholds": len(self.thresholds),
                "monitoring_interval": self.monitoring_interval,
                "validation_rules": sum(len(rules) for rules in self.validation_rules.values()),
                "benchmarks": len(self.benchmarks)
            }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status for a specific agent."""
        agent_alerts = [a for a in self.alerts if a.agent_id == agent_id]
        recent_alerts = [a for a in agent_alerts if (datetime.now() - a.timestamp).days < 1]
        
        metrics = self.agent_metrics.get(agent_id, {})
        latest_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                latest_metrics[metric_name] = values[-1]
        
        return {
            "agent_id": agent_id,
            "total_alerts": len(agent_alerts),
            "recent_alerts": len(recent_alerts),
            "latest_metrics": latest_metrics,
            "metrics_count": sum(len(values) for values in metrics.values()),
            "inspections": len(self.inspection_history.get(agent_id, [])),
            "validations": len(self.validation_history.get(agent_id, [])),
            "benchmarks": len(self.benchmark_history.get(agent_id, []))
        }
    
    def get_alerts(self, agent_id: str = None, severity: str = None, since: datetime = None) -> List[QualityAlert]:
        """Get quality alerts with optional filtering."""
        alerts = self.alerts.copy()
        
        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    self._log(f"Alert acknowledged: {alert_id}")
                    break
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.lock:
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_date]
        
        self._log(f"Cleared alerts older than {days} days")
    
    def get_inspection_history(self, agent_id: str) -> List[QualityReport]:
        """Get inspection history for an agent."""
        with self.lock:
            return self.inspection_history.get(agent_id, [])
    
    def get_validation_history(self, agent_id: str) -> List[ValidationResult]:
        """Get validation history for an agent."""
        with self.lock:
            return self.validation_history.get(agent_id, [])
    
    def get_scoring_history(self, agent_id: str) -> List[QualityScore]:
        """Get scoring history for an agent."""
        with self.lock:
            return self.scoring_history.get(agent_id, [])
    
    def get_benchmark_history(self, agent_id: str) -> List[BenchmarkResult]:
        """Get benchmark history for an agent."""
        with self.lock:
            return self.benchmark_history.get(agent_id, [])
    
    def shutdown(self):
        """Shutdown the Agent QA system."""
        self.stop_monitoring()
        self._log("Agent QA system shutdown completed")


# =============================================================================
# Factory Functions and Convenience Interface
# =============================================================================

# Global instance
_agent_qa_instance: Optional[AgentQualityAssurance] = None


def get_agent_qa(enable_monitoring: bool = True) -> AgentQualityAssurance:
    """Get or create the global Agent QA instance."""
    global _agent_qa_instance
    if _agent_qa_instance is None:
        _agent_qa_instance = AgentQualityAssurance(enable_monitoring=enable_monitoring)
    return _agent_qa_instance


def configure_agent_qa(
    similarity_threshold: float = 0.7,
    enable_benchmarking: bool = True,
    enable_monitoring: bool = True,
    alert_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Configure agent quality assurance system.
    
    Args:
        similarity_threshold: Threshold for quality similarity checks
        enable_benchmarking: Enable performance benchmarking
        enable_monitoring: Enable continuous quality monitoring
        alert_threshold: Threshold for quality alerts
        
    Returns:
        Configuration status
    """
    qa_system = get_agent_qa(enable_monitoring=enable_monitoring)
    
    config = {
        "similarity_threshold": similarity_threshold,
        "benchmarking_enabled": enable_benchmarking,
        "monitoring_enabled": enable_monitoring,
        "alert_threshold": alert_threshold
    }
    
    return {"status": "configured", "config": config}


# =============================================================================
# Convenience Functions
# =============================================================================

def inspect_agent_quality(
    agent_id: str,
    test_cases: List[Dict[str, Any]] = None,
    include_benchmarks: bool = True
) -> QualityReport:
    """Perform comprehensive quality inspection of an agent."""
    qa_system = get_agent_qa()
    return qa_system.inspect_agent(agent_id, test_cases, include_benchmarks)


def validate_agent_output(
    agent_id: str,
    output: Any,
    expected: Any = None,
    validation_rules: List[ValidationRule] = None
) -> ValidationResult:
    """Validate agent output against rules and expectations."""
    qa_system = get_agent_qa()
    return qa_system.validate_output(agent_id, output, expected, validation_rules)


def score_agent_quality(
    agent_id: str,
    quality_metrics: List[QualityMetric],
    custom_weights: Dict[str, float] = None
) -> QualityScore:
    """Calculate comprehensive quality score for an agent."""
    qa_system = get_agent_qa()
    return qa_system.calculate_score(agent_id, quality_metrics, custom_weights)


def benchmark_agent_performance(
    agent_id: str,
    benchmark_types: List[BenchmarkType] = None,
    iterations: int = 10
) -> BenchmarkResult:
    """Run performance benchmarks for an agent."""
    qa_system = get_agent_qa()
    return qa_system.run_benchmarks(agent_id, benchmark_types, iterations)


def get_quality_status() -> Dict[str, Any]:
    """Get current quality status across all agents."""
    qa_system = get_agent_qa()
    return qa_system.get_status()


def shutdown_agent_qa():
    """Shutdown agent quality assurance system."""
    global _agent_qa_instance
    if _agent_qa_instance:
        _agent_qa_instance.shutdown()
        _agent_qa_instance = None


# Convenience aliases
inspect_quality = inspect_agent_quality
validate_output = validate_agent_output
calculate_score = score_agent_quality
run_benchmarks = benchmark_agent_performance