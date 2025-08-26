"""
Compliance Automation System

Enterprise-grade compliance automation supporting multiple regulatory frameworks
with automated validation, reporting, and remediation workflows.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CCPA = "ccpa"
    SOC2 = "soc2"
    FISMA = "fisma"
    GLBA = "glba"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class ComplianceRule:
    """Individual compliance rule definition."""
    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    automated_check: bool
    severity: str  # critical, high, medium, low
    remediation_guidance: str
    validation_patterns: List[str] = field(default_factory=list)
    exclusion_patterns: List[str] = field(default_factory=list)
    documentation_required: bool = False
    stakeholder_approval: bool = False


@dataclass
class ComplianceViolation:
    """Compliance violation finding."""
    rule_id: str
    framework: ComplianceFramework
    violation_type: str
    severity: str
    file_path: str
    line_number: Optional[int]
    description: str
    evidence: List[str]
    remediation_steps: List[str]
    stakeholders: List[str]
    deadline: Optional[datetime]
    business_impact: str
    detection_time: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class ComplianceReport:
    """Comprehensive compliance assessment report."""
    framework: ComplianceFramework
    assessment_date: datetime
    overall_status: ComplianceStatus
    compliance_score: float  # 0-100
    total_rules_checked: int
    compliant_rules: int
    violations: List[ComplianceViolation]
    recommendations: List[str]
    next_assessment_date: datetime
    certification_status: str


class ComplianceAutomationEngine:
    """
    Enterprise compliance automation engine supporting multiple regulatory frameworks
    with intelligent validation, automated remediation, and stakeholder workflows.
    """
    
    def __init__(self):
        """Initialize compliance automation engine."""
        self.compliance_rules = {}
        self.assessment_history = {}
        self.violation_tracking = {}
        self.remediation_workflows = {}
        self.stakeholder_matrix = {}
        
        # Initialize framework-specific rules
        self._initialize_compliance_rules()
        
        # Automation settings
        self.auto_remediation_enabled = True
        self.stakeholder_notifications = True
        self.continuous_monitoring = True
        
        logger.info("Compliance Automation Engine initialized")
        
    def _initialize_compliance_rules(self) -> None:
        """Initialize compliance rules for all supported frameworks."""
        self.compliance_rules[ComplianceFramework.GDPR] = self._get_gdpr_rules()
        self.compliance_rules[ComplianceFramework.PCI_DSS] = self._get_pci_dss_rules()
        self.compliance_rules[ComplianceFramework.HIPAA] = self._get_hipaa_rules()
        self.compliance_rules[ComplianceFramework.SOX] = self._get_sox_rules()
        self.compliance_rules[ComplianceFramework.ISO_27001] = self._get_iso27001_rules()
        self.compliance_rules[ComplianceFramework.NIST] = self._get_nist_rules()
        self.compliance_rules[ComplianceFramework.SOC2] = self._get_soc2_rules()
        
    def _get_gdpr_rules(self) -> List[ComplianceRule]:
        """Get GDPR compliance rules."""
        return [
            ComplianceRule(
                rule_id="GDPR-001",
                framework=ComplianceFramework.GDPR,
                title="Personal Data Consent",
                description="Personal data collection requires explicit user consent",
                requirement="Article 6(1)(a) - Lawful basis for processing",
                automated_check=True,
                severity="critical",
                remediation_guidance="Implement consent mechanisms for all personal data collection",
                validation_patterns=[
                    r"email.*input",
                    r"phone.*input", 
                    r"address.*input",
                    r"personal.*data"
                ],
                documentation_required=True
            ),
            ComplianceRule(
                rule_id="GDPR-002",
                framework=ComplianceFramework.GDPR,
                title="Data Breach Notification",
                description="Data breaches must be reported within 72 hours",
                requirement="Article 33 - Notification requirements",
                automated_check=True,
                severity="critical",
                remediation_guidance="Implement automated breach detection and notification systems",
                validation_patterns=[
                    r"security.*incident",
                    r"data.*breach",
                    r"unauthorized.*access"
                ]
            ),
            ComplianceRule(
                rule_id="GDPR-003",
                framework=ComplianceFramework.GDPR,
                title="Right to Erasure",
                description="Users must be able to request data deletion",
                requirement="Article 17 - Right to erasure ('right to be forgotten')",
                automated_check=True,
                severity="high",
                remediation_guidance="Implement data deletion workflows and APIs",
                validation_patterns=[
                    r"delete.*user.*data",
                    r"data.*erasure",
                    r"forget.*user"
                ]
            )
        ]
        
    def _get_pci_dss_rules(self) -> List[ComplianceRule]:
        """Get PCI-DSS compliance rules."""
        return [
            ComplianceRule(
                rule_id="PCI-001",
                framework=ComplianceFramework.PCI_DSS,
                title="Cardholder Data Protection",
                description="Protect stored cardholder data",
                requirement="Requirement 3 - Protect stored cardholder data",
                automated_check=True,
                severity="critical",
                remediation_guidance="Encrypt all cardholder data at rest and in transit",
                validation_patterns=[
                    r"credit.*card",
                    r"card.*number",
                    r"cvv",
                    r"payment.*info"
                ]
            ),
            ComplianceRule(
                rule_id="PCI-002",
                framework=ComplianceFramework.PCI_DSS,
                title="Access Control",
                description="Restrict access to cardholder data by business need to know",
                requirement="Requirement 7 - Restrict access to cardholder data",
                automated_check=True,
                severity="high",
                remediation_guidance="Implement role-based access controls",
                validation_patterns=[
                    r"access.*control",
                    r"role.*based",
                    r"permission.*check"
                ]
            )
        ]
        
    def _get_hipaa_rules(self) -> List[ComplianceRule]:
        """Get HIPAA compliance rules."""
        return [
            ComplianceRule(
                rule_id="HIPAA-001",
                framework=ComplianceFramework.HIPAA,
                title="PHI Encryption",
                description="Protected Health Information must be encrypted",
                requirement="164.312(a)(2)(iv) - Encryption and decryption",
                automated_check=True,
                severity="critical",
                remediation_guidance="Encrypt all PHI at rest and in transit",
                validation_patterns=[
                    r"health.*information",
                    r"medical.*record",
                    r"patient.*data",
                    r"phi"
                ]
            ),
            ComplianceRule(
                rule_id="HIPAA-002",
                framework=ComplianceFramework.HIPAA,
                title="Audit Logs",
                description="Maintain audit logs for PHI access",
                requirement="164.312(b) - Audit controls",
                automated_check=True,
                severity="high",
                remediation_guidance="Implement comprehensive audit logging",
                validation_patterns=[
                    r"audit.*log",
                    r"access.*log",
                    r"phi.*access"
                ]
            )
        ]
        
    async def assess_compliance(self, project_path: str, frameworks: List[ComplianceFramework]) -> Dict[ComplianceFramework, ComplianceReport]:
        """
        Comprehensive compliance assessment across multiple frameworks.
        
        Args:
            project_path: Path to project for assessment
            frameworks: List of compliance frameworks to assess
            
        Returns:
            Compliance reports for each framework
        """
        reports = {}
        
        for framework in frameworks:
            logger.info(f"Assessing {framework.value} compliance")
            
            violations = await self._check_framework_compliance(project_path, framework)
            
            # Calculate compliance metrics
            total_rules = len(self.compliance_rules.get(framework, []))
            violation_count = len(violations)
            compliant_rules = total_rules - violation_count
            compliance_score = (compliant_rules / total_rules * 100) if total_rules > 0 else 100
            
            # Determine overall status
            if compliance_score == 100:
                status = ComplianceStatus.COMPLIANT
            elif compliance_score >= 80:
                status = ComplianceStatus.PARTIAL
            else:
                status = ComplianceStatus.NON_COMPLIANT
                
            # Generate recommendations
            recommendations = await self._generate_compliance_recommendations(violations, framework)
            
            report = ComplianceReport(
                framework=framework,
                assessment_date=datetime.now(),
                overall_status=status,
                compliance_score=compliance_score,
                total_rules_checked=total_rules,
                compliant_rules=compliant_rules,
                violations=violations,
                recommendations=recommendations,
                next_assessment_date=datetime.now() + timedelta(days=30),
                certification_status=self._determine_certification_status(compliance_score)
            )
            
            reports[framework] = report
            
        return reports
        
    async def _check_framework_compliance(self, project_path: str, framework: ComplianceFramework) -> List[ComplianceViolation]:
        """Check compliance for a specific framework."""
        violations = []
        rules = self.compliance_rules.get(framework, [])
        
        for rule in rules:
            if rule.automated_check:
                rule_violations = await self._check_rule_compliance(project_path, rule)
                violations.extend(rule_violations)
                
        return violations
        
    async def _check_rule_compliance(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Check compliance for a specific rule."""
        violations = []
        
        # Scan all relevant files
        for file_path in Path(project_path).glob("**/*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for line_num, line in enumerate(lines, 1):
                    # Check validation patterns
                    for pattern in rule.validation_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Check if excluded by exclusion patterns
                            excluded = any(re.search(excl_pattern, line, re.IGNORECASE) 
                                         for excl_pattern in rule.exclusion_patterns)
                            
                            if not excluded:
                                violation = ComplianceViolation(
                                    rule_id=rule.rule_id,
                                    framework=rule.framework,
                                    violation_type=rule.title,
                                    severity=rule.severity,
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    description=f"Potential {rule.framework.value} violation: {rule.description}",
                                    evidence=[line.strip()],
                                    remediation_steps=rule.remediation_guidance.split('. '),
                                    stakeholders=self._get_rule_stakeholders(rule),
                                    deadline=self._calculate_remediation_deadline(rule.severity),
                                    business_impact=self._assess_violation_business_impact(rule)
                                )
                                violations.append(violation)
                                
            except Exception as e:
                logger.warning(f"Error checking {file_path} for rule {rule.rule_id}: {e}")
                
        return violations
        
    async def generate_executive_compliance_dashboard(self, project_path: str) -> Dict[str, Any]:
        """Generate executive compliance dashboard."""
        # Assess all frameworks
        all_frameworks = list(ComplianceFramework)
        reports = await self.assess_compliance(project_path, all_frameworks)
        
        # Calculate overall metrics
        total_violations = sum(len(report.violations) for report in reports.values())
        avg_compliance_score = sum(report.compliance_score for report in reports.values()) / len(reports) if reports else 0
        
        critical_violations = []
        for report in reports.values():
            critical_violations.extend([v for v in report.violations if v.severity == 'critical'])
            
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'executive_summary': {
                'overall_compliance_score': avg_compliance_score,
                'total_violations': total_violations,
                'critical_violations': len(critical_violations),
                'frameworks_assessed': len(reports),
                'compliance_trend': self._calculate_compliance_trend(),
                'risk_level': self._calculate_overall_risk_level(avg_compliance_score, critical_violations)
            },
            'framework_status': {
                framework.value: {
                    'status': report.overall_status.value,
                    'score': report.compliance_score,
                    'violations': len(report.violations),
                    'critical_issues': len([v for v in report.violations if v.severity == 'critical'])
                }
                for framework, report in reports.items()
            },
            'top_priorities': self._identify_top_compliance_priorities(reports),
            'recommendations': self._generate_executive_recommendations(reports),
            'certification_readiness': self._assess_certification_readiness(reports)
        }
        
        return dashboard
        
    async def automated_remediation_workflow(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Execute automated remediation workflows."""
        remediation_results = {
            'timestamp': datetime.now().isoformat(),
            'violations_processed': len(violations),
            'auto_fixed': 0,
            'requires_manual_review': 0,
            'stakeholder_notifications_sent': 0,
            'remediation_actions': []
        }
        
        for violation in violations:
            if self.auto_remediation_enabled and self._can_auto_remediate(violation):
                # Attempt automated remediation
                success = await self._execute_auto_remediation(violation)
                if success:
                    remediation_results['auto_fixed'] += 1
                    violation.resolved = True
                else:
                    remediation_results['requires_manual_review'] += 1
            else:
                # Create manual review task
                await self._create_manual_review_task(violation)
                remediation_results['requires_manual_review'] += 1
                
            # Send stakeholder notifications
            if self.stakeholder_notifications:
                await self._notify_stakeholders(violation)
                remediation_results['stakeholder_notifications_sent'] += 1
                
        return remediation_results
        
    # Helper methods (simplified implementations)
    def _get_rule_stakeholders(self, rule: ComplianceRule) -> List[str]:
        """Get stakeholders for a compliance rule."""
        stakeholder_map = {
            ComplianceFramework.GDPR: ['privacy_officer', 'legal_team'],
            ComplianceFramework.PCI_DSS: ['security_team', 'payments_team'],
            ComplianceFramework.HIPAA: ['privacy_officer', 'healthcare_compliance']
        }
        return stakeholder_map.get(rule.framework, ['compliance_team'])
        
    def _calculate_remediation_deadline(self, severity: str) -> Optional[datetime]:
        """Calculate remediation deadline based on severity."""
        deadline_map = {
            'critical': timedelta(days=1),
            'high': timedelta(days=7),
            'medium': timedelta(days=30),
            'low': timedelta(days=90)
        }
        days = deadline_map.get(severity, timedelta(days=30))
        return datetime.now() + days
        
    def _assess_violation_business_impact(self, rule: ComplianceRule) -> str:
        """Assess business impact of compliance violation."""
        impact_map = {
            'critical': 'high',
            'high': 'medium',
            'medium': 'low',
            'low': 'minimal'
        }
        return impact_map.get(rule.severity, 'low')
        
    def _determine_certification_status(self, compliance_score: float) -> str:
        """Determine certification readiness status."""
        if compliance_score >= 95:
            return "certification_ready"
        elif compliance_score >= 85:
            return "approaching_compliance"
        elif compliance_score >= 70:
            return "improvement_needed"
        else:
            return "significant_gaps"