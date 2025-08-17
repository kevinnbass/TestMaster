"""
TestMaster Security Intelligence

Provides universal security scanning and compliance testing for any codebase.
Adapted from security patterns in Agency Swarm and PraisonAI.
"""

from .universal_scanner import (
    UniversalSecurityScanner,
    SecurityScanConfig,
    SecurityScanResult,
    VulnerabilityFinding,
    SeverityLevel,
    VulnerabilityType
)

from .compliance_framework import (
    ComplianceFramework,
    ComplianceStandard,
    ComplianceRule,
    ComplianceReport,
    ComplianceStatus
)

from .security_test_generator import (
    SecurityTestGenerator,
    SecurityTestConfig,
    SecurityTestSuite,
    ThreatModel
)

__all__ = [
    # Security Scanning
    'UniversalSecurityScanner',
    'SecurityScanConfig',
    'SecurityScanResult',
    'VulnerabilityFinding',
    'SeverityLevel',
    'VulnerabilityType',
    
    # Compliance Framework
    'ComplianceFramework',
    'ComplianceStandard',
    'ComplianceRule',
    'ComplianceReport',
    'ComplianceStatus',
    
    # Security Test Generation
    'SecurityTestGenerator',
    'SecurityTestConfig',
    'SecurityTestSuite',
    'ThreatModel'
]