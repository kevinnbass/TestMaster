"""
Unified Security Scanner - Comprehensive multi-layer security analysis system

This package provides enterprise-grade security scanning with:
- Multi-layer orchestration across security analysis dimensions
- Advanced correlation algorithms for finding relationships
- Intelligent remediation recommendations
- Comprehensive reporting and metrics
"""

from .security_scan_models import (
    # Core models
    ScanConfiguration,
    UnifiedScanResult,
    SecurityFinding,
    ComplianceViolation,
    QualityIssue,
    PerformanceConcern,
    SecurityCorrelation,
    RemediationAction,
    IntelligentTestCase,
    IntelligentTestSuite,
    EnhancedSecurityFinding,
    
    # Supporting models
    LayerResult,
    ScanProgress,
    SecurityMetrics,
    
    # Enums
    RiskLevel,
    ScanPhase,
    HTTPMethod,
    AuthenticationLevel,
    
    # Factory functions
    create_security_finding,
    create_scan_configuration,
    create_unified_scan_result
)

from .security_correlations import (
    SecurityCorrelationAnalyzer,
    CorrelationResult,
    get_strongest_correlation,
    summarize_correlations
)

from .security_orchestrator import (
    SecurityLayerOrchestrator,
    LayerConfiguration,
    create_security_orchestrator
)

from .unified_scanner_core import (
    UnifiedSecurityScanner,
    create_unified_scanner
)

__all__ = [
    # Main scanner
    'UnifiedSecurityScanner',
    'create_unified_scanner',
    
    # Orchestrator
    'SecurityLayerOrchestrator',
    'LayerConfiguration',
    'create_security_orchestrator',
    
    # Correlation analyzer
    'SecurityCorrelationAnalyzer',
    'CorrelationResult',
    'get_strongest_correlation',
    'summarize_correlations',
    
    # Models
    'ScanConfiguration',
    'UnifiedScanResult',
    'SecurityFinding',
    'ComplianceViolation',
    'QualityIssue',
    'PerformanceConcern',
    'SecurityCorrelation',
    'RemediationAction',
    'IntelligentTestCase',
    'IntelligentTestSuite',
    'EnhancedSecurityFinding',
    'LayerResult',
    'ScanProgress',
    'SecurityMetrics',
    
    # Enums
    'RiskLevel',
    'ScanPhase',
    'HTTPMethod',
    'AuthenticationLevel',
    
    # Factory functions
    'create_security_finding',
    'create_scan_configuration',
    'create_unified_scan_result'
]

# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Security Team'
__description__ = 'Enterprise-grade unified security scanner with multi-layer analysis'


def quick_scan(target_path: str = '.') -> UnifiedScanResult:
    """
    Perform a quick security scan with default settings
    
    Args:
        target_path: Path to scan (default: current directory)
        
    Returns:
        Unified scan result
    
    Example:
        >>> from unified_scanner import quick_scan
        >>> result = quick_scan('src/')
        >>> print(f"Risk Score: {result.overall_risk_score}")
    """
    config = create_scan_configuration(
        target_paths=[target_path],
        parallel_workers=2,
        cache_results=True,
        generate_report=True,
        report_format='markdown'
    )
    scanner = create_unified_scanner(config)
    return scanner.scan()


def deep_scan(target_path: str = '.') -> UnifiedScanResult:
    """
    Perform a comprehensive deep security scan
    
    Args:
        target_path: Path to scan (default: current directory)
        
    Returns:
        Unified scan result with all analysis layers enabled
    
    Example:
        >>> from unified_scanner import deep_scan
        >>> result = deep_scan('src/')
        >>> print(f"Total Findings: {len(result.vulnerabilities)}")
    """
    config = create_scan_configuration(
        target_paths=[target_path],
        enable_real_time_monitoring=True,
        enable_classical_analysis=True,
        enable_quality_correlation=True,
        enable_performance_profiling=True,
        enable_compliance_checking=True,
        enable_intelligent_testing=True,
        deep_analysis=True,
        cross_reference_findings=True,
        generate_test_suite=True,
        parallel_workers=4,
        cache_results=False,  # Fresh analysis
        generate_report=True,
        report_format='html'
    )
    scanner = create_unified_scanner(config)
    return scanner.scan()


# Module initialization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())