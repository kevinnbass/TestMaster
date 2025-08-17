"""
Universal Compliance Framework

Cross-language compliance checking for various standards (SOX, GDPR, PCI-DSS, etc.).
Adapted from Agency Swarm's compliance patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from datetime import datetime
import json
import re

from .universal_scanner import VulnerabilityFinding, SeverityLevel
from ..core.ast_abstraction import UniversalAST


class ComplianceStatus(Enum):
    """Compliance status for rules and standards."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    SOX = "sox"  # Sarbanes-Oxley Act
    GDPR = "gdpr"  # General Data Protection Regulation
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    ISO_27001 = "iso_27001"  # ISO 27001 Information Security Management
    NIST_CSF = "nist_csf"  # NIST Cybersecurity Framework
    CIS_CONTROLS = "cis_controls"  # CIS Critical Security Controls
    OWASP_ASVS = "owasp_asvs"  # OWASP Application Security Verification Standard
    CCPA = "ccpa"  # California Consumer Privacy Act
    FISMA = "fisma"  # Federal Information Security Management Act


@dataclass
class ComplianceRule:
    """Represents a single compliance rule."""
    rule_id: str
    standard: ComplianceStandard
    title: str
    description: str
    requirement: str
    
    # Implementation details
    check_function: Optional[str] = None
    automated_check: bool = True
    manual_verification_required: bool = False
    
    # Risk and priority
    severity: SeverityLevel = SeverityLevel.MEDIUM
    priority: int = 3  # 1-5, 1 being highest
    
    # Categories
    control_category: str = ""
    technical_control: bool = True
    administrative_control: bool = False
    physical_control: bool = False
    
    # References
    standard_reference: str = ""
    related_cwe: List[str] = field(default_factory=list)
    related_owasp: List[str] = field(default_factory=list)
    
    # Implementation guidance
    implementation_guidance: str = ""
    testing_procedures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_id': self.rule_id,
            'standard': self.standard.value,
            'title': self.title,
            'description': self.description,
            'requirement': self.requirement,
            'check_function': self.check_function,
            'automated_check': self.automated_check,
            'manual_verification_required': self.manual_verification_required,
            'severity': self.severity.value,
            'priority': self.priority,
            'control_category': self.control_category,
            'technical_control': self.technical_control,
            'administrative_control': self.administrative_control,
            'physical_control': self.physical_control,
            'standard_reference': self.standard_reference,
            'related_cwe': self.related_cwe,
            'related_owasp': self.related_owasp,
            'implementation_guidance': self.implementation_guidance,
            'testing_procedures': self.testing_procedures
        }


@dataclass
class ComplianceResult:
    """Result of a compliance rule check."""
    rule_id: str
    status: ComplianceStatus
    score: float  # 0.0 to 1.0
    
    # Details
    findings: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Files and locations
    affected_files: List[str] = field(default_factory=list)
    violations: List[VulnerabilityFinding] = field(default_factory=list)
    
    # Metadata
    check_timestamp: datetime = field(default_factory=datetime.now)
    automated: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_id': self.rule_id,
            'status': self.status.value,
            'score': self.score,
            'findings': self.findings,
            'evidence': self.evidence,
            'gaps': self.gaps,
            'recommendations': self.recommendations,
            'affected_files': self.affected_files,
            'violations': [v.to_dict() for v in self.violations],
            'check_timestamp': self.check_timestamp.isoformat(),
            'automated': self.automated
        }


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""
    standard: ComplianceStandard
    overall_status: ComplianceStatus
    overall_score: float  # 0.0 to 1.0
    
    # Results by rule
    rule_results: Dict[str, ComplianceResult] = field(default_factory=dict)
    
    # Summary statistics
    total_rules: int = 0
    compliant_rules: int = 0
    non_compliant_rules: int = 0
    partially_compliant_rules: int = 0
    not_applicable_rules: int = 0
    
    # Categories
    category_scores: Dict[str, float] = field(default_factory=dict)
    
    # Remediation
    high_priority_gaps: List[str] = field(default_factory=list)
    remediation_plan: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    assessment_date: datetime = field(default_factory=datetime.now)
    assessor: Optional[str] = None
    scope: str = ""
    
    def calculate_summary(self):
        """Calculate summary statistics."""
        self.total_rules = len(self.rule_results)
        
        status_counts = {
            ComplianceStatus.COMPLIANT: 0,
            ComplianceStatus.NON_COMPLIANT: 0,
            ComplianceStatus.PARTIALLY_COMPLIANT: 0,
            ComplianceStatus.NOT_APPLICABLE: 0
        }
        
        total_score = 0.0
        applicable_rules = 0
        
        for result in self.rule_results.values():
            status_counts[result.status] += 1
            
            if result.status != ComplianceStatus.NOT_APPLICABLE:
                total_score += result.score
                applicable_rules += 1
        
        self.compliant_rules = status_counts[ComplianceStatus.COMPLIANT]
        self.non_compliant_rules = status_counts[ComplianceStatus.NON_COMPLIANT]
        self.partially_compliant_rules = status_counts[ComplianceStatus.PARTIALLY_COMPLIANT]
        self.not_applicable_rules = status_counts[ComplianceStatus.NOT_APPLICABLE]
        
        # Calculate overall score
        if applicable_rules > 0:
            self.overall_score = total_score / applicable_rules
        else:
            self.overall_score = 1.0
        
        # Determine overall status
        if self.overall_score >= 0.95:
            self.overall_status = ComplianceStatus.COMPLIANT
        elif self.overall_score >= 0.7:
            self.overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            self.overall_status = ComplianceStatus.NON_COMPLIANT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'standard': self.standard.value,
            'overall_status': self.overall_status.value,
            'overall_score': self.overall_score,
            'summary': {
                'total_rules': self.total_rules,
                'compliant_rules': self.compliant_rules,
                'non_compliant_rules': self.non_compliant_rules,
                'partially_compliant_rules': self.partially_compliant_rules,
                'not_applicable_rules': self.not_applicable_rules
            },
            'category_scores': self.category_scores,
            'rule_results': {k: v.to_dict() for k, v in self.rule_results.items()},
            'high_priority_gaps': self.high_priority_gaps,
            'remediation_plan': self.remediation_plan,
            'metadata': {
                'assessment_date': self.assessment_date.isoformat(),
                'assessor': self.assessor,
                'scope': self.scope
            }
        }


class ComplianceFramework:
    """Universal compliance framework for multiple standards."""
    
    def __init__(self):
        self.rules: Dict[ComplianceStandard, List[ComplianceRule]] = {}
        self._load_compliance_rules()
        
        print(f"Compliance Framework initialized")
        print(f"   Standards loaded: {len(self.rules)}")
        for standard, rules in self.rules.items():
            print(f"      {standard.value}: {len(rules)} rules")
    
    def assess_compliance(self, 
                         standard: ComplianceStandard,
                         universal_ast: UniversalAST,
                         security_findings: List[VulnerabilityFinding] = None) -> ComplianceReport:
        """Assess compliance against a standard."""
        
        print(f"\nAssessing compliance with {standard.value.upper()}...")
        
        # Get rules for this standard
        rules = self.rules.get(standard, [])
        
        # Create report
        report = ComplianceReport(
            standard=standard,
            overall_status=ComplianceStatus.NEEDS_REVIEW,
            overall_score=0.0,
            scope=f"Codebase: {universal_ast.project_path}"
        )
        
        # Check each rule
        for rule in rules:
            result = self._check_compliance_rule(rule, universal_ast, security_findings or [])
            report.rule_results[rule.rule_id] = result
        
        # Calculate summary
        report.calculate_summary()
        
        # Generate remediation plan
        report.remediation_plan = self._generate_remediation_plan(report)
        
        print(f"   Compliance assessment complete:")
        print(f"      Overall score: {report.overall_score:.1%}")
        print(f"      Status: {report.overall_status.value}")
        print(f"      Compliant rules: {report.compliant_rules}/{report.total_rules}")
        
        return report
    
    def _check_compliance_rule(self, 
                              rule: ComplianceRule,
                              universal_ast: UniversalAST,
                              security_findings: List[VulnerabilityFinding]) -> ComplianceResult:
        """Check a single compliance rule."""
        
        result = ComplianceResult(
            rule_id=rule.rule_id,
            status=ComplianceStatus.NEEDS_REVIEW,
            score=0.0,
            automated=rule.automated_check
        )
        
        # Dispatch to specific check function
        if rule.check_function:
            check_method = getattr(self, f"_check_{rule.check_function}", None)
            if check_method:
                try:
                    check_method(rule, universal_ast, security_findings, result)
                except Exception as e:
                    result.findings.append(f"Check failed: {str(e)}")
                    result.status = ComplianceStatus.NEEDS_REVIEW
                    result.score = 0.0
            else:
                result.findings.append(f"Check function {rule.check_function} not implemented")
                result.status = ComplianceStatus.NEEDS_REVIEW
        else:
            # Generic checks based on rule category
            self._generic_compliance_check(rule, universal_ast, security_findings, result)
        
        return result
    
    def _generic_compliance_check(self,
                                 rule: ComplianceRule,
                                 universal_ast: UniversalAST,
                                 security_findings: List[VulnerabilityFinding],
                                 result: ComplianceResult):
        """Generic compliance check based on rule category."""
        
        # Map security findings to compliance
        relevant_findings = []
        for finding in security_findings:
            if self._is_finding_relevant_to_rule(finding, rule):
                relevant_findings.append(finding)
                result.affected_files.append(finding.file_path)
        
        result.violations = relevant_findings
        
        # Calculate score based on findings
        if not relevant_findings:
            result.status = ComplianceStatus.COMPLIANT
            result.score = 1.0
            result.evidence.append("No security violations found related to this rule")
        else:
            # Score based on severity of findings
            critical_count = sum(1 for f in relevant_findings if f.severity == SeverityLevel.CRITICAL)
            high_count = sum(1 for f in relevant_findings if f.severity == SeverityLevel.HIGH)
            medium_count = sum(1 for f in relevant_findings if f.severity == SeverityLevel.MEDIUM)
            
            # Calculate penalty score
            penalty = (critical_count * 0.3) + (high_count * 0.2) + (medium_count * 0.1)
            result.score = max(0.0, 1.0 - penalty)
            
            if result.score >= 0.8:
                result.status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                result.status = ComplianceStatus.NON_COMPLIANT
            
            result.findings.append(f"Found {len(relevant_findings)} security violations")
            result.gaps.append("Security vulnerabilities present that violate compliance requirements")
            result.recommendations.append("Address identified security vulnerabilities")
    
    def _is_finding_relevant_to_rule(self, finding: VulnerabilityFinding, rule: ComplianceRule) -> bool:
        """Check if a security finding is relevant to a compliance rule."""
        
        # Check CWE mappings
        if rule.related_cwe and finding.cwe_id:
            if finding.cwe_id in rule.related_cwe:
                return True
        
        # Check OWASP mappings
        if rule.related_owasp and finding.owasp_category:
            for owasp_ref in rule.related_owasp:
                if owasp_ref in finding.owasp_category:
                    return True
        
        # Check by category keywords
        rule_keywords = rule.title.lower() + " " + rule.description.lower()
        finding_keywords = finding.title.lower() + " " + finding.description.lower()
        
        # Common keywords mapping
        keyword_matches = [
            ('authentication', 'auth'),
            ('authorization', 'access'),
            ('encryption', 'crypto'),
            ('input validation', 'injection'),
            ('logging', 'audit'),
            ('data protection', 'sensitive')
        ]
        
        for rule_kw, finding_kw in keyword_matches:
            if rule_kw in rule_keywords and finding_kw in finding_keywords:
                return True
        
        return False
    
    def _load_compliance_rules(self):
        """Load compliance rules for different standards."""
        
        # SOX (Sarbanes-Oxley) Rules
        self.rules[ComplianceStandard.SOX] = [
            ComplianceRule(
                rule_id="SOX_001",
                standard=ComplianceStandard.SOX,
                title="Access Control",
                description="Restrict access to financial systems and data",
                requirement="Implement proper access controls for financial applications",
                check_function="access_control",
                severity=SeverityLevel.HIGH,
                priority=1,
                control_category="access_control",
                related_cwe=["CWE-285", "CWE-287"],
                related_owasp=["A01:2021", "A07:2021"],
                implementation_guidance="Implement role-based access control and authentication"
            ),
            ComplianceRule(
                rule_id="SOX_002",
                standard=ComplianceStandard.SOX,
                title="Audit Logging",
                description="Maintain comprehensive audit logs",
                requirement="Log all access to financial data and systems",
                check_function="audit_logging",
                severity=SeverityLevel.HIGH,
                priority=1,
                control_category="logging",
                related_owasp=["A09:2021"],
                implementation_guidance="Implement comprehensive audit logging for all financial operations"
            ),
            ComplianceRule(
                rule_id="SOX_003",
                standard=ComplianceStandard.SOX,
                title="Data Integrity",
                description="Ensure data integrity and prevent tampering",
                requirement="Implement controls to ensure financial data integrity",
                check_function="data_integrity",
                severity=SeverityLevel.CRITICAL,
                priority=1,
                control_category="data_protection",
                related_cwe=["CWE-89", "CWE-79"],
                implementation_guidance="Use input validation and secure coding practices"
            )
        ]
        
        # GDPR Rules
        self.rules[ComplianceStandard.GDPR] = [
            ComplianceRule(
                rule_id="GDPR_001",
                standard=ComplianceStandard.GDPR,
                title="Data Protection by Design",
                description="Implement data protection by design and by default",
                requirement="Ensure personal data protection is built into systems",
                check_function="data_protection_design",
                severity=SeverityLevel.HIGH,
                priority=1,
                control_category="data_protection",
                related_cwe=["CWE-200", "CWE-311"],
                implementation_guidance="Implement encryption and access controls for personal data"
            ),
            ComplianceRule(
                rule_id="GDPR_002",
                standard=ComplianceStandard.GDPR,
                title="Consent Management",
                description="Implement proper consent mechanisms",
                requirement="Obtain and manage user consent for data processing",
                check_function="consent_management",
                severity=SeverityLevel.MEDIUM,
                priority=2,
                control_category="consent",
                implementation_guidance="Implement clear consent mechanisms and opt-out options"
            ),
            ComplianceRule(
                rule_id="GDPR_003",
                standard=ComplianceStandard.GDPR,
                title="Data Breach Detection",
                description="Implement data breach detection and notification",
                requirement="Detect and report data breaches within 72 hours",
                check_function="breach_detection",
                severity=SeverityLevel.HIGH,
                priority=1,
                control_category="incident_response",
                related_owasp=["A09:2021"],
                implementation_guidance="Implement monitoring and alerting for security incidents"
            )
        ]
        
        # PCI-DSS Rules
        self.rules[ComplianceStandard.PCI_DSS] = [
            ComplianceRule(
                rule_id="PCI_001",
                standard=ComplianceStandard.PCI_DSS,
                title="Cardholder Data Protection",
                description="Protect stored cardholder data",
                requirement="Encrypt stored cardholder data",
                check_function="cardholder_data_protection",
                severity=SeverityLevel.CRITICAL,
                priority=1,
                control_category="data_protection",
                related_cwe=["CWE-311", "CWE-327"],
                implementation_guidance="Use strong encryption for cardholder data storage"
            ),
            ComplianceRule(
                rule_id="PCI_002",
                standard=ComplianceStandard.PCI_DSS,
                title="Secure Authentication",
                description="Implement strong authentication mechanisms",
                requirement="Use multi-factor authentication for privileged access",
                check_function="secure_authentication",
                severity=SeverityLevel.HIGH,
                priority=1,
                control_category="authentication",
                related_cwe=["CWE-287", "CWE-294"],
                implementation_guidance="Implement multi-factor authentication and strong password policies"
            )
        ]
        
        # OWASP ASVS Rules
        self.rules[ComplianceStandard.OWASP_ASVS] = [
            ComplianceRule(
                rule_id="ASVS_001",
                standard=ComplianceStandard.OWASP_ASVS,
                title="Input Validation",
                description="Implement comprehensive input validation",
                requirement="Validate all input data",
                check_function="input_validation",
                severity=SeverityLevel.HIGH,
                priority=1,
                control_category="input_validation",
                related_cwe=["CWE-20", "CWE-89", "CWE-79"],
                related_owasp=["A03:2021"],
                implementation_guidance="Implement server-side input validation for all user inputs"
            ),
            ComplianceRule(
                rule_id="ASVS_002",
                standard=ComplianceStandard.OWASP_ASVS,
                title="Session Management",
                description="Implement secure session management",
                requirement="Use secure session management practices",
                check_function="session_management",
                severity=SeverityLevel.HIGH,
                priority=1,
                control_category="session_management",
                related_cwe=["CWE-384", "CWE-613"],
                implementation_guidance="Use secure session tokens and proper session lifecycle management"
            )
        ]
    
    def _generate_remediation_plan(self, report: ComplianceReport) -> List[Dict[str, Any]]:
        """Generate remediation plan based on compliance gaps."""
        plan = []
        
        # High priority items first
        high_priority_rules = [
            result for result in report.rule_results.values()
            if result.status == ComplianceStatus.NON_COMPLIANT and
            any(rule.priority == 1 for rule in self.rules[report.standard] if rule.rule_id == result.rule_id)
        ]
        
        for result in high_priority_rules:
            # Find the rule
            rule = next((r for r in self.rules[report.standard] if r.rule_id == result.rule_id), None)
            if rule:
                remediation_item = {
                    'rule_id': result.rule_id,
                    'title': rule.title,
                    'priority': 'HIGH',
                    'estimated_effort': 'Medium',  # Could be calculated based on findings
                    'description': rule.implementation_guidance,
                    'steps': result.recommendations,
                    'affected_files': result.affected_files,
                    'deadline': '30 days'  # Could be calculated based on risk
                }
                plan.append(remediation_item)
        
        return plan
    
    # Specific check functions (simplified implementations)
    
    def _check_access_control(self, rule, ast, findings, result):
        """Check access control implementation."""
        auth_functions = []
        for module in ast.modules:
            for func in module.functions:
                if any(keyword in func.name.lower() for keyword in ['auth', 'login', 'verify']):
                    auth_functions.append(func.name)
        
        if auth_functions:
            result.evidence.append(f"Found authentication functions: {', '.join(auth_functions)}")
            result.score = 0.8
            result.status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            result.gaps.append("No authentication functions found")
            result.score = 0.2
            result.status = ComplianceStatus.NON_COMPLIANT
    
    def _check_audit_logging(self, rule, ast, findings, result):
        """Check audit logging implementation."""
        logging_functions = []
        for module in ast.modules:
            for func in module.functions:
                if any(keyword in func.name.lower() for keyword in ['log', 'audit', 'record']):
                    logging_functions.append(func.name)
        
        if logging_functions:
            result.evidence.append(f"Found logging functions: {', '.join(logging_functions)}")
            result.score = 0.9
            result.status = ComplianceStatus.COMPLIANT
        else:
            result.gaps.append("No audit logging functions found")
            result.score = 0.1
            result.status = ComplianceStatus.NON_COMPLIANT
    
    def _check_data_integrity(self, rule, ast, findings, result):
        """Check data integrity controls."""
        # Look for SQL injection findings
        sql_findings = [f for f in findings if 'sql' in f.type.value.lower()]
        
        if not sql_findings:
            result.evidence.append("No SQL injection vulnerabilities found")
            result.score = 1.0
            result.status = ComplianceStatus.COMPLIANT
        else:
            result.violations = sql_findings
            result.gaps.append("SQL injection vulnerabilities threaten data integrity")
            result.score = 0.3
            result.status = ComplianceStatus.NON_COMPLIANT
    
    def get_supported_standards(self) -> List[ComplianceStandard]:
        """Get list of supported compliance standards."""
        return list(self.rules.keys())
    
    def get_rules_for_standard(self, standard: ComplianceStandard) -> List[ComplianceRule]:
        """Get all rules for a specific standard."""
        return self.rules.get(standard, [])