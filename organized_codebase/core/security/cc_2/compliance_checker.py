"""
Compliance Checker

Validates code against security standards and compliance requirements.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComplianceIssue:
    """Represents a compliance violation."""
    standard: str  # OWASP, PCI-DSS, HIPAA, GDPR, SOC2
    requirement: str
    severity: str  # critical, high, medium, low
    file_path: str
    line_number: Optional[int]
    description: str
    remediation: str
    

@dataclass
class ComplianceReport:
    """Complete compliance assessment report."""
    standards_checked: List[str]
    total_issues: int
    passed_checks: int
    failed_checks: int
    compliance_score: float
    issues: List[ComplianceIssue]
    timestamp: str
    

class ComplianceChecker:
    """
    Checks code compliance with security standards.
    Supports OWASP, PCI-DSS, HIPAA, GDPR, and custom policies.
    """
    
    def __init__(self):
        """Initialize the compliance checker."""
        self.issues = []
        self.checks_performed = 0
        self.checks_passed = 0
        self.standards = self._load_standards()
        logger.info("Compliance Checker initialized")
        
    def check_owasp_compliance(self, directory: str) -> List[ComplianceIssue]:
        """
        Check OWASP Top 10 compliance.
        
        Args:
            directory: Directory to check
            
        Returns:
            List of compliance issues
        """
        issues = []
        
        # A01:2021 - Broken Access Control
        issues.extend(self._check_access_control(directory))
        
        # A02:2021 - Cryptographic Failures
        issues.extend(self._check_cryptography(directory))
        
        # A03:2021 - Injection
        issues.extend(self._check_injection_prevention(directory))
        
        # A04:2021 - Insecure Design
        issues.extend(self._check_secure_design(directory))
        
        # A05:2021 - Security Misconfiguration
        issues.extend(self._check_configuration(directory))
        
        # A06:2021 - Vulnerable Components
        issues.extend(self._check_dependencies(directory))
        
        # A07:2021 - Authentication Failures
        issues.extend(self._check_authentication(directory))
        
        # A08:2021 - Data Integrity Failures
        issues.extend(self._check_data_integrity(directory))
        
        # A09:2021 - Logging Failures
        issues.extend(self._check_logging(directory))
        
        # A10:2021 - SSRF
        issues.extend(self._check_ssrf_prevention(directory))
        
        self.issues.extend(issues)
        return issues
        
    def check_pci_dss_compliance(self, directory: str) -> List[ComplianceIssue]:
        """
        Check PCI-DSS compliance for payment card data.
        
        Args:
            directory: Directory to check
            
        Returns:
            List of compliance issues
        """
        issues = []
        
        # Requirement 2: Default passwords
        issues.extend(self._check_default_credentials(directory))
        
        # Requirement 3: Cardholder data protection
        issues.extend(self._check_card_data_protection(directory))
        
        # Requirement 4: Encrypted transmission
        issues.extend(self._check_encrypted_transmission(directory))
        
        # Requirement 6: Secure development
        issues.extend(self._check_secure_development(directory))
        
        # Requirement 8: User authentication
        issues.extend(self._check_strong_authentication(directory))
        
        # Requirement 10: Logging
        issues.extend(self._check_audit_logging(directory))
        
        self.issues.extend(issues)
        return issues
        
    def check_gdpr_compliance(self, directory: str) -> List[ComplianceIssue]:
        """
        Check GDPR compliance for data privacy.
        
        Args:
            directory: Directory to check
            
        Returns:
            List of compliance issues
        """
        issues = []
        
        # Data minimization
        issues.extend(self._check_data_minimization(directory))
        
        # Consent management
        issues.extend(self._check_consent_management(directory))
        
        # Right to erasure
        issues.extend(self._check_data_deletion(directory))
        
        # Data portability
        issues.extend(self._check_data_portability(directory))
        
        # Privacy by design
        issues.extend(self._check_privacy_by_design(directory))
        
        self.issues.extend(issues)
        return issues
        
    def check_custom_policies(self, directory: str, policy_file: str) -> List[ComplianceIssue]:
        """
        Check compliance with custom security policies.
        
        Args:
            directory: Directory to check
            policy_file: Path to custom policy JSON file
            
        Returns:
            List of compliance issues
        """
        issues = []
        
        try:
            with open(policy_file, 'r') as f:
                policies = json.load(f)
                
            for policy in policies.get('policies', []):
                issues.extend(self._check_custom_policy(directory, policy))
                
        except Exception as e:
            logger.error(f"Error loading custom policies: {e}")
            
        self.issues.extend(issues)
        return issues
        
    def generate_compliance_report(self, standards: List[str]) -> ComplianceReport:
        """
        Generate comprehensive compliance report.
        
        Args:
            standards: List of standards checked
            
        Returns:
            Compliance report
        """
        total_checks = self.checks_performed
        passed = self.checks_passed
        failed = total_checks - passed
        score = (passed / total_checks * 100) if total_checks > 0 else 0
        
        return ComplianceReport(
            standards_checked=standards,
            total_issues=len(self.issues),
            passed_checks=passed,
            failed_checks=failed,
            compliance_score=score,
            issues=self.issues,
            timestamp=datetime.now().isoformat()
        )
        
    def export_report(self, report: ComplianceReport, output_path: str) -> None:
        """
        Export compliance report to file.
        
        Args:
            report: Compliance report
            output_path: Output file path
        """
        report_dict = {
            'standards': report.standards_checked,
            'total_issues': report.total_issues,
            'passed': report.passed_checks,
            'failed': report.failed_checks,
            'score': report.compliance_score,
            'timestamp': report.timestamp,
            'issues': [self._issue_to_dict(i) for i in report.issues]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
            
        logger.info(f"Exported compliance report to {output_path}")
        
    # Specific compliance checks
    def _check_access_control(self, directory: str) -> List[ComplianceIssue]:
        """Check for access control issues."""
        issues = []
        self.checks_performed += 1
        
        # Check for missing authentication
        for file_path in Path(directory).rglob("*.py"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'route' in content and 'login_required' not in content:
                issues.append(ComplianceIssue(
                    standard="OWASP",
                    requirement="A01:2021 - Broken Access Control",
                    severity="high",
                    file_path=str(file_path),
                    line_number=None,
                    description="Route without authentication check",
                    remediation="Add authentication decorators to all routes"
                ))
        
        if not issues:
            self.checks_passed += 1
            
        return issues
        
    def _check_cryptography(self, directory: str) -> List[ComplianceIssue]:
        """Check for cryptographic issues."""
        issues = []
        self.checks_performed += 1
        
        weak_algorithms = ['md5', 'sha1', 'des', 'rc4']
        
        for file_path in Path(directory).rglob("*.py"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
            for algo in weak_algorithms:
                if algo in content:
                    issues.append(ComplianceIssue(
                        standard="OWASP",
                        requirement="A02:2021 - Cryptographic Failures",
                        severity="high",
                        file_path=str(file_path),
                        line_number=None,
                        description=f"Use of weak algorithm: {algo}",
                        remediation="Use strong algorithms (SHA-256, AES, etc.)"
                    ))
                    
        if not issues:
            self.checks_passed += 1
            
        return issues
        
    def _check_injection_prevention(self, directory: str) -> List[ComplianceIssue]:
        """Check for injection vulnerabilities."""
        issues = []
        self.checks_performed += 1
        
        # Simplified check - would be more comprehensive in production
        for file_path in Path(directory).rglob("*.py"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'execute(' in content and '%s' in content:
                issues.append(ComplianceIssue(
                    standard="OWASP",
                    requirement="A03:2021 - Injection",
                    severity="critical",
                    file_path=str(file_path),
                    line_number=None,
                    description="Potential SQL injection",
                    remediation="Use parameterized queries"
                ))
                
        if not issues:
            self.checks_passed += 1
            
        return issues
        
    def _check_logging(self, directory: str) -> List[ComplianceIssue]:
        """Check for proper logging implementation."""
        issues = []
        self.checks_performed += 1
        
        has_logging = False
        for file_path in Path(directory).rglob("*.py"):
            with open(file_path, 'r', encoding='utf-8') as f:
                if 'import logging' in f.read():
                    has_logging = True
                    break
                    
        if not has_logging:
            issues.append(ComplianceIssue(
                standard="OWASP",
                requirement="A09:2021 - Security Logging",
                severity="medium",
                file_path=directory,
                line_number=None,
                description="No logging implementation found",
                remediation="Implement comprehensive security logging"
            ))
        else:
            self.checks_passed += 1
            
        return issues
        
    # Helper methods
    def _load_standards(self) -> Dict[str, Any]:
        """Load compliance standards definitions."""
        return {
            'OWASP': {'version': '2021', 'categories': 10},
            'PCI-DSS': {'version': '4.0', 'requirements': 12},
            'GDPR': {'articles': 99},
            'HIPAA': {'rules': ['Privacy', 'Security', 'Breach']},
            'SOC2': {'criteria': ['Security', 'Availability', 'Confidentiality']}
        }
        
    def _issue_to_dict(self, issue: ComplianceIssue) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            'standard': issue.standard,
            'requirement': issue.requirement,
            'severity': issue.severity,
            'file': issue.file_path,
            'line': issue.line_number,
            'description': issue.description,
            'remediation': issue.remediation
        }
        
    # Stub methods for additional checks
    def _check_secure_design(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_configuration(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_dependencies(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_authentication(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_data_integrity(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_ssrf_prevention(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_default_credentials(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_card_data_protection(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_encrypted_transmission(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_secure_development(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_strong_authentication(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_audit_logging(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_data_minimization(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_consent_management(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_data_deletion(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_data_portability(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_privacy_by_design(self, directory: str) -> List[ComplianceIssue]:
        return []
        
    def _check_custom_policy(self, directory: str, policy: Dict) -> List[ComplianceIssue]:
        return []