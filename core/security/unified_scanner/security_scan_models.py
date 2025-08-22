"""
Security Scan Models - Core data structures for unified security scanning

This module provides comprehensive data models for:
- Scan configuration and management
- Scan results and findings
- Risk assessments and correlations
- Enhanced security intelligence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime


class HTTPMethod(Enum):
    """HTTP methods for API operations"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class AuthenticationLevel(Enum):
    """Authentication levels for security operations"""
    NONE = "none"
    BASIC = "basic"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"


class RiskLevel(Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ScanPhase(Enum):
    """Security scan phases"""
    INITIALIZATION = "initialization"
    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    CORRELATION = "correlation"
    INTELLIGENCE = "intelligence"
    REPORTING = "reporting"


@dataclass
class SecurityFinding:
    """Individual security finding"""
    finding_id: str
    finding_type: str
    severity: RiskLevel
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    confidence: float = 0.0
    remediation: Optional[str] = None
    cve_references: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceViolation:
    """Compliance standard violation"""
    violation_id: str
    standard: str
    control_id: str
    description: str
    severity: RiskLevel
    affected_resources: List[str]
    remediation_steps: List[str]
    compliance_impact: str
    audit_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityIssue:
    """Code quality issue"""
    issue_id: str
    issue_type: str
    severity: str
    metric_name: str
    metric_value: float
    threshold_value: float
    file_path: str
    description: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceConcern:
    """Performance-related concern"""
    concern_id: str
    concern_type: str
    metric: str
    value: float
    expected_value: float
    impact: str
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class SecurityCorrelation:
    """Correlation between different security aspects"""
    correlation_type: str
    correlation_strength: float
    primary_metric: str
    secondary_metric: str
    primary_value: float
    secondary_value: float
    insight: str
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class RemediationAction:
    """Remediation action item"""
    action_id: str
    action: str
    description: str
    priority: str
    estimated_hours: float
    required_skills: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)


@dataclass
class IntelligentTestCase:
    """AI-generated test case"""
    test_id: str
    test_name: str
    test_type: str
    target_vulnerability: str
    test_code: str
    expected_result: str
    confidence: float = 0.0
    framework: str = "pytest"


@dataclass
class IntelligentTestSuite:
    """AI-generated test suite for security validation"""
    suite_id: str
    suite_name: str
    target_findings: List[str]
    test_cases: List[IntelligentTestCase]
    coverage_percentage: float
    estimated_execution_time: float
    recommended_frequency: str
    validation_strategy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedSecurityFinding:
    """Enhanced security finding with AI intelligence"""
    base_finding: SecurityFinding
    ai_confidence: float
    attack_vector: str
    exploitability_score: float
    business_impact: str
    threat_actors: List[str]
    mitre_attack_techniques: List[str]
    contextual_risk_score: float
    historical_occurrences: int
    predicted_evolution: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedScanResult:
    """Comprehensive unified security scan result"""
    # Identification
    scan_id: str
    timestamp: float
    target_path: str
    scan_duration: float
    
    # Core findings
    vulnerabilities: List[Dict[str, Any]]
    compliance_violations: List[Dict[str, Any]]
    quality_issues: List[Dict[str, Any]]
    performance_concerns: List[Dict[str, Any]]
    
    # Risk assessment
    overall_risk_score: float
    risk_distribution: Dict[str, int]
    risk_trends: Dict[str, Any]
    
    # Correlations
    security_quality_correlation: Dict[str, Any]
    security_performance_correlation: Dict[str, Any]
    complexity_vulnerability_correlation: Dict[str, Any]
    
    # Intelligence
    enhanced_findings: List[EnhancedSecurityFinding]
    intelligent_test_suite: Optional[IntelligentTestSuite]
    remediation_plan: Dict[str, Any]
    
    # Metrics
    scan_metrics: Dict[str, Any]
    coverage_metrics: Dict[str, float]
    confidence_score: float
    
    # Additional metadata
    scan_configuration: Optional[Dict[str, Any]] = None
    layer_results: Dict[str, Any] = field(default_factory=dict)
    execution_errors: List[str] = field(default_factory=list)
    

@dataclass
class ScanConfiguration:
    """Configuration for unified security scan"""
    # Scan targets
    target_paths: List[str]
    file_patterns: List[str] = field(default_factory=lambda: ['*.py'])
    exclude_patterns: List[str] = field(default_factory=list)
    
    # Scan options
    enable_real_time_monitoring: bool = True
    enable_classical_analysis: bool = True
    enable_quality_correlation: bool = True
    enable_performance_profiling: bool = True
    enable_compliance_checking: bool = True
    enable_intelligent_testing: bool = True
    
    # Thresholds
    risk_threshold: float = 70.0
    quality_threshold: float = 60.0
    complexity_threshold: float = 15.0
    confidence_threshold: float = 0.7
    
    # Performance
    parallel_workers: int = 4
    scan_timeout: int = 300  # seconds
    cache_results: bool = True
    cache_ttl: int = 300  # seconds
    
    # Output
    generate_report: bool = True
    report_format: str = 'json'  # 'json', 'html', 'markdown'
    output_directory: str = './security_reports'
    verbose_logging: bool = False
    
    # Advanced options
    deep_analysis: bool = True
    cross_reference_findings: bool = True
    generate_test_suite: bool = True
    auto_remediation: bool = False


@dataclass
class ScanProgress:
    """Progress tracking for long-running scans"""
    scan_id: str
    phase: ScanPhase
    current_target: str
    targets_completed: int
    total_targets: int
    findings_count: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    current_operation: str = ""
    percentage_complete: float = 0.0


@dataclass
class SecurityMetrics:
    """Aggregated security metrics"""
    total_scans: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
    total_vulnerabilities: int = 0
    total_scan_time: float = 0.0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    avg_scan_time: float = 0.0
    avg_risk_score: float = 0.0
    avg_confidence_score: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class LayerResult:
    """Result from individual security layer"""
    layer_name: str
    layer_type: str
    execution_time: float
    success: bool
    findings_count: int
    raw_results: Dict[str, Any]
    error_message: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Factory functions for creating common structures
def create_security_finding(
    finding_type: str,
    severity: str,
    title: str,
    description: str,
    file_path: str,
    **kwargs
) -> SecurityFinding:
    """Factory function to create security finding"""
    import uuid
    return SecurityFinding(
        finding_id=str(uuid.uuid4()),
        finding_type=finding_type,
        severity=RiskLevel[severity.upper()],
        title=title,
        description=description,
        file_path=file_path,
        **kwargs
    )


def create_scan_configuration(
    target_paths: List[str],
    **overrides
) -> ScanConfiguration:
    """Factory function to create scan configuration"""
    config = ScanConfiguration(target_paths=target_paths)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_unified_scan_result(
    scan_id: str,
    target_path: str,
    **kwargs
) -> UnifiedScanResult:
    """Factory function to create unified scan result"""
    import time
    return UnifiedScanResult(
        scan_id=scan_id,
        timestamp=time.time(),
        target_path=target_path,
        scan_duration=0.0,
        vulnerabilities=[],
        compliance_violations=[],
        quality_issues=[],
        performance_concerns=[],
        overall_risk_score=0.0,
        risk_distribution={},
        risk_trends={},
        security_quality_correlation={},
        security_performance_correlation={},
        complexity_vulnerability_correlation={},
        enhanced_findings=[],
        intelligent_test_suite=None,
        remediation_plan={},
        scan_metrics={},
        coverage_metrics={},
        confidence_score=0.0,
        **kwargs
    )