"""
Enterprise Compliance Validation Framework for TestMaster
Comprehensive compliance testing for SOC2, GDPR, HIPAA, PCI-DSS standards
"""

import asyncio
import json
import time
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sqlite3

class ComplianceStandard(Enum):
    """Supported compliance standards"""
    SOC2_TYPE1 = "SOC2_TYPE1"
    SOC2_TYPE2 = "SOC2_TYPE2"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"
    ISO27001 = "ISO27001"
    NIST_CSF = "NIST_CSF"
    CCPA = "CCPA"

class ComplianceCategory(Enum):
    """Compliance control categories"""
    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    AUDIT_LOGGING = "audit_logging"
    ENCRYPTION = "encryption"
    INCIDENT_RESPONSE = "incident_response"
    BACKUP_RECOVERY = "backup_recovery"
    NETWORK_SECURITY = "network_security"
    VENDOR_MANAGEMENT = "vendor_management"
    PRIVACY_RIGHTS = "privacy_rights"
    DATA_RETENTION = "data_retention"

class ComplianceLevel(Enum):
    """Compliance assessment levels"""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"

class RiskLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class ComplianceControl:
    """Individual compliance control definition"""
    control_id: str
    standard: ComplianceStandard
    category: ComplianceCategory
    title: str
    description: str
    requirements: List[str]
    test_procedures: List[str]
    evidence_required: List[str]
    risk_level: RiskLevel

@dataclass
class ComplianceEvidence:
    """Evidence collected for compliance validation"""
    control_id: str
    evidence_type: str
    evidence_data: Any
    source: str
    timestamp: float
    confidence_score: float
    automated: bool = True

@dataclass
class ComplianceTestResult:
    """Result of compliance control testing"""
    control_id: str
    standard: ComplianceStandard
    category: ComplianceCategory
    status: ComplianceLevel
    score: float
    findings: List[str]
    evidence: List[ComplianceEvidence]
    recommendations: List[str]
    risk_level: RiskLevel
    timestamp: float

@dataclass
class ComplianceReport:
    """Comprehensive compliance assessment report"""
    organization: str
    standards_assessed: List[ComplianceStandard]
    overall_compliance_score: float
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    partially_compliant_controls: int
    critical_findings: int
    high_risk_findings: int
    control_results: List[ComplianceTestResult]
    executive_summary: str
    remediation_plan: List[str]
    next_assessment_date: float
    assessment_duration: float
    timestamp: float

class ComplianceValidator:
    """Enterprise Compliance Validation Framework"""
    
    def __init__(self, organization: str = "TestMaster Enterprise"):
        self.organization = organization
        self.controls = self._initialize_compliance_controls()
        self.evidence_store: List[ComplianceEvidence] = []
        self.test_results: List[ComplianceTestResult] = []
        
    def _initialize_compliance_controls(self) -> Dict[ComplianceStandard, List[ComplianceControl]]:
        """Initialize comprehensive compliance control framework"""
        return {
            ComplianceStandard.SOC2_TYPE2: [
                ComplianceControl(
                    "CC6.1", ComplianceStandard.SOC2_TYPE2, ComplianceCategory.ACCESS_CONTROL,
                    "Logical Access Controls",
                    "System restricts logical access through authentication and authorization",
                    [
                        "Multi-factor authentication required for privileged access",
                        "User access reviews performed quarterly",
                        "Terminated users removed within 24 hours"
                    ],
                    [
                        "Review authentication logs",
                        "Test MFA enforcement",
                        "Validate user provisioning/deprovisioning"
                    ],
                    ["Authentication logs", "User access reports", "MFA configuration"],
                    RiskLevel.HIGH
                ),
                ComplianceControl(
                    "A1.2", ComplianceStandard.SOC2_TYPE2, ComplianceCategory.DATA_PROTECTION,
                    "Data Classification and Handling",
                    "Sensitive data is classified and handled according to policy",
                    [
                        "Data classification scheme implemented",
                        "Encryption in transit and at rest",
                        "Data loss prevention controls"
                    ],
                    [
                        "Review data classification policies",
                        "Test encryption implementation",
                        "Validate DLP controls"
                    ],
                    ["Data classification policy", "Encryption certificates", "DLP logs"],
                    RiskLevel.HIGH
                )
            ],
            
            ComplianceStandard.GDPR: [
                ComplianceControl(
                    "ART7", ComplianceStandard.GDPR, ComplianceCategory.PRIVACY_RIGHTS,
                    "Right to Consent",
                    "Clear and explicit consent mechanisms for data processing",
                    [
                        "Explicit consent collection for each purpose",
                        "Consent withdrawal mechanisms available",
                        "Consent records maintained"
                    ],
                    [
                        "Review consent collection mechanisms",
                        "Test consent withdrawal process",
                        "Validate consent record keeping"
                    ],
                    ["Consent forms", "Privacy notices", "Consent management system logs"],
                    RiskLevel.CRITICAL
                ),
                ComplianceControl(
                    "ART17", ComplianceStandard.GDPR, ComplianceCategory.PRIVACY_RIGHTS,
                    "Right to Erasure",
                    "Individuals can request deletion of personal data",
                    [
                        "Data deletion procedures implemented",
                        "Response to erasure requests within 30 days",
                        "Third-party deletion coordination"
                    ],
                    [
                        "Test data deletion process",
                        "Validate erasure request handling",
                        "Check third-party deletion coordination"
                    ],
                    ["Deletion logs", "Erasure request records", "Third-party agreements"],
                    RiskLevel.HIGH
                ),
                ComplianceControl(
                    "ART25", ComplianceStandard.GDPR, ComplianceCategory.DATA_PROTECTION,
                    "Data Protection by Design",
                    "Privacy protection built into system design",
                    [
                        "Privacy impact assessments conducted",
                        "Data minimization principles applied",
                        "Privacy by default settings"
                    ],
                    [
                        "Review privacy impact assessments",
                        "Test data minimization controls",
                        "Validate default privacy settings"
                    ],
                    ["Privacy impact assessments", "System design docs", "Default configuration"],
                    RiskLevel.MEDIUM
                )
            ],
            
            ComplianceStandard.HIPAA: [
                ComplianceControl(
                    "164.312(a)(1)", ComplianceStandard.HIPAA, ComplianceCategory.ACCESS_CONTROL,
                    "Access Control Standard",
                    "Unique user identification, emergency access, automatic logoff",
                    [
                        "Unique user IDs for each person accessing ePHI",
                        "Emergency access procedures documented",
                        "Automatic logoff after period of inactivity"
                    ],
                    [
                        "Test unique user identification",
                        "Review emergency access procedures",
                        "Validate automatic logoff functionality"
                    ],
                    ["User access logs", "Emergency access procedures", "System configuration"],
                    RiskLevel.HIGH
                ),
                ComplianceControl(
                    "164.312(e)(1)", ComplianceStandard.HIPAA, ComplianceCategory.ENCRYPTION,
                    "Transmission Security",
                    "ePHI transmission security measures",
                    [
                        "ePHI encrypted during transmission",
                        "End-to-end encryption for external communications",
                        "Secure communication protocols used"
                    ],
                    [
                        "Test transmission encryption",
                        "Validate secure protocols",
                        "Review encryption key management"
                    ],
                    ["Encryption logs", "Network configuration", "Certificate management"],
                    RiskLevel.CRITICAL
                )
            ],
            
            ComplianceStandard.PCI_DSS: [
                ComplianceControl(
                    "REQ3", ComplianceStandard.PCI_DSS, ComplianceCategory.DATA_PROTECTION,
                    "Protect Stored Cardholder Data",
                    "Cardholder data protection requirements",
                    [
                        "Strong cryptography for cardholder data",
                        "Encryption key management",
                        "Data retention and disposal"
                    ],
                    [
                        "Test cardholder data encryption",
                        "Review key management procedures",
                        "Validate data retention policies"
                    ],
                    ["Encryption implementation", "Key management logs", "Data retention policy"],
                    RiskLevel.CRITICAL
                ),
                ComplianceControl(
                    "REQ8", ComplianceStandard.PCI_DSS, ComplianceCategory.ACCESS_CONTROL,
                    "Identify and Authenticate Access",
                    "User identification and authentication requirements",
                    [
                        "Unique user IDs for all users",
                        "Strong authentication for system components",
                        "Multi-factor authentication for remote access"
                    ],
                    [
                        "Test user identification systems",
                        "Validate authentication strength",
                        "Review MFA implementation"
                    ],
                    ["User management system", "Authentication logs", "MFA configuration"],
                    RiskLevel.HIGH
                )
            ]
        }
    
    def add_evidence(self, evidence: ComplianceEvidence) -> None:
        """Add compliance evidence to store"""
        self.evidence_store.append(evidence)
    
    def collect_system_evidence(self, system_path: str) -> List[ComplianceEvidence]:
        """Automatically collect compliance evidence from system"""
        evidence = []
        
        # File system evidence collection
        evidence.extend(self._collect_file_evidence(system_path))
        
        # Configuration evidence collection
        evidence.extend(self._collect_config_evidence(system_path))
        
        # Log evidence collection
        evidence.extend(self._collect_log_evidence(system_path))
        
        # Code evidence collection
        evidence.extend(self._collect_code_evidence(system_path))
        
        return evidence
    
    def _collect_file_evidence(self, system_path: str) -> List[ComplianceEvidence]:
        """Collect evidence from file system"""
        evidence = []
        system_dir = Path(system_path)
        
        if not system_dir.exists():
            return evidence
        
        # Check for encryption configuration
        for config_file in system_dir.rglob("*.conf"):
            try:
                content = config_file.read_text()
                if any(keyword in content.lower() for keyword in ["ssl", "tls", "encrypt", "cipher"]):
                    evidence.append(ComplianceEvidence(
                        control_id="ENCRYPTION_CONFIG",
                        evidence_type="configuration",
                        evidence_data={"file": str(config_file), "contains_encryption": True},
                        source="file_system",
                        timestamp=time.time(),
                        confidence_score=0.8
                    ))
            except:
                pass
        
        # Check for access control files
        access_files = ["users.txt", "groups.txt", "permissions.conf", "auth.conf"]
        for access_file in access_files:
            file_path = system_dir / access_file
            if file_path.exists():
                evidence.append(ComplianceEvidence(
                    control_id="ACCESS_CONTROL_CONFIG",
                    evidence_type="access_control",
                    evidence_data={"file": str(file_path), "exists": True},
                    source="file_system",
                    timestamp=time.time(),
                    confidence_score=0.7
                ))
        
        return evidence
    
    def _collect_config_evidence(self, system_path: str) -> List[ComplianceEvidence]:
        """Collect evidence from configuration files"""
        evidence = []
        system_dir = Path(system_path)
        
        # Database configurations
        for db_config in system_dir.rglob("database.yml"):
            try:
                content = db_config.read_text()
                if "encrypted" in content.lower() or "ssl" in content.lower():
                    evidence.append(ComplianceEvidence(
                        control_id="DATABASE_ENCRYPTION",
                        evidence_type="database_config",
                        evidence_data={"encryption_enabled": True},
                        source="configuration",
                        timestamp=time.time(),
                        confidence_score=0.9
                    ))
            except:
                pass
        
        return evidence
    
    def _collect_log_evidence(self, system_path: str) -> List[ComplianceEvidence]:
        """Collect evidence from log files"""
        evidence = []
        system_dir = Path(system_path)
        
        # Check for audit logs
        log_patterns = ["audit.log", "access.log", "auth.log", "security.log"]
        for pattern in log_patterns:
            for log_file in system_dir.rglob(pattern):
                if log_file.exists():
                    evidence.append(ComplianceEvidence(
                        control_id="AUDIT_LOGGING",
                        evidence_type="audit_log",
                        evidence_data={"log_file": str(log_file), "size": log_file.stat().st_size},
                        source="log_files",
                        timestamp=time.time(),
                        confidence_score=0.8
                    ))
        
        return evidence
    
    def _collect_code_evidence(self, system_path: str) -> List[ComplianceEvidence]:
        """Collect evidence from source code"""
        evidence = []
        system_dir = Path(system_path)
        
        # Check for security-related code patterns
        security_patterns = {
            "encryption": [r"encrypt\(", r"decrypt\(", r"AES", r"RSA", r"hash\("],
            "authentication": [r"authenticate\(", r"login\(", r"verify_token", r"check_password"],
            "authorization": [r"authorize\(", r"check_permission", r"has_role", r"access_control"],
            "audit": [r"audit_log", r"log_event", r"track_action", r"security_log"]
        }
        
        for py_file in system_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                for category, patterns in security_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            evidence.append(ComplianceEvidence(
                                control_id=f"CODE_{category.upper()}",
                                evidence_type="source_code",
                                evidence_data={
                                    "file": str(py_file),
                                    "pattern": pattern,
                                    "category": category
                                },
                                source="source_code",
                                timestamp=time.time(),
                                confidence_score=0.6
                            ))
                            break
            except:
                pass
        
        return evidence
    
    async def assess_compliance(self, standards: List[ComplianceStandard], 
                               system_path: str = None) -> ComplianceReport:
        """Perform comprehensive compliance assessment"""
        start_time = time.time()
        
        # Collect evidence if system path provided
        if system_path:
            system_evidence = self.collect_system_evidence(system_path)
            self.evidence_store.extend(system_evidence)
        
        all_results = []
        
        for standard in standards:
            controls = self.controls.get(standard, [])
            for control in controls:
                result = await self._assess_control(control)
                all_results.append(result)
        
        # Generate comprehensive report
        return self._generate_compliance_report(standards, all_results, time.time() - start_time)
    
    async def _assess_control(self, control: ComplianceControl) -> ComplianceTestResult:
        """Assess individual compliance control"""
        # Gather relevant evidence
        relevant_evidence = self._get_relevant_evidence(control)
        
        # Evaluate control based on evidence
        status, score, findings = self._evaluate_control_compliance(control, relevant_evidence)
        
        # Generate recommendations
        recommendations = self._generate_control_recommendations(control, status, findings)
        
        return ComplianceTestResult(
            control_id=control.control_id,
            standard=control.standard,
            category=control.category,
            status=status,
            score=score,
            findings=findings,
            evidence=relevant_evidence,
            recommendations=recommendations,
            risk_level=control.risk_level,
            timestamp=time.time()
        )
    
    def _get_relevant_evidence(self, control: ComplianceControl) -> List[ComplianceEvidence]:
        """Get evidence relevant to specific control"""
        relevant = []
        
        # Match evidence by control category
        category_keywords = {
            ComplianceCategory.ACCESS_CONTROL: ["access", "auth", "login", "user", "permission"],
            ComplianceCategory.DATA_PROTECTION: ["encrypt", "data", "protect", "secure"],
            ComplianceCategory.AUDIT_LOGGING: ["audit", "log", "track", "monitor"],
            ComplianceCategory.ENCRYPTION: ["encrypt", "ssl", "tls", "cipher", "key"]
        }
        
        keywords = category_keywords.get(control.category, [])
        
        for evidence in self.evidence_store:
            # Direct control ID match
            if evidence.control_id == control.control_id:
                relevant.append(evidence)
                continue
            
            # Category-based matching
            evidence_text = str(evidence.evidence_data).lower()
            if any(keyword in evidence_text for keyword in keywords):
                relevant.append(evidence)
        
        return relevant
    
    def _evaluate_control_compliance(self, control: ComplianceControl, 
                                   evidence: List[ComplianceEvidence]) -> Tuple[ComplianceLevel, float, List[str]]:
        """Evaluate control compliance based on evidence"""
        findings = []
        score = 0.0
        
        if not evidence:
            findings.append("No evidence found for this control")
            return ComplianceLevel.REQUIRES_REVIEW, 0.0, findings
        
        # Evidence-based scoring
        total_confidence = sum(e.confidence_score for e in evidence)
        evidence_count = len(evidence)
        
        if evidence_count >= 3 and total_confidence >= 2.0:
            score = 85.0
            status = ComplianceLevel.COMPLIANT
            findings.append("Sufficient evidence found supporting compliance")
        elif evidence_count >= 2 and total_confidence >= 1.5:
            score = 65.0
            status = ComplianceLevel.PARTIALLY_COMPLIANT
            findings.append("Partial evidence found, may require additional controls")
        elif evidence_count >= 1:
            score = 40.0
            status = ComplianceLevel.NON_COMPLIANT
            findings.append("Insufficient evidence for compliance demonstration")
        else:
            score = 0.0
            status = ComplianceLevel.REQUIRES_REVIEW
            findings.append("No evidence available for assessment")
        
        # Adjust for control risk level
        if control.risk_level == RiskLevel.CRITICAL and score < 90:
            status = ComplianceLevel.NON_COMPLIANT
            findings.append("Critical control requires higher evidence threshold")
        
        return status, score, findings
    
    def _generate_control_recommendations(self, control: ComplianceControl, 
                                        status: ComplianceLevel, findings: List[str]) -> List[str]:
        """Generate recommendations for control improvement"""
        recommendations = []
        
        if status == ComplianceLevel.NON_COMPLIANT:
            recommendations.append(f"Implement missing controls for {control.title}")
            recommendations.append("Establish documentation and evidence collection procedures")
            
        if status == ComplianceLevel.PARTIALLY_COMPLIANT:
            recommendations.append("Strengthen existing controls and documentation")
            recommendations.append("Implement additional monitoring and validation")
            
        if status == ComplianceLevel.REQUIRES_REVIEW:
            recommendations.append("Conduct detailed control assessment with subject matter experts")
            recommendations.append("Implement evidence collection mechanisms")
        
        # Risk-based recommendations
        if control.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("Prioritize this control for immediate attention")
            recommendations.append("Consider additional compensating controls")
        
        return recommendations
    
    def _generate_compliance_report(self, standards: List[ComplianceStandard], 
                                  results: List[ComplianceTestResult], 
                                  duration: float) -> ComplianceReport:
        """Generate comprehensive compliance assessment report"""
        compliant = len([r for r in results if r.status == ComplianceLevel.COMPLIANT])
        non_compliant = len([r for r in results if r.status == ComplianceLevel.NON_COMPLIANT])
        partially_compliant = len([r for r in results if r.status == ComplianceLevel.PARTIALLY_COMPLIANT])
        
        critical_findings = len([r for r in results if r.risk_level == RiskLevel.CRITICAL and 
                               r.status != ComplianceLevel.COMPLIANT])
        high_risk_findings = len([r for r in results if r.risk_level == RiskLevel.HIGH and 
                                r.status != ComplianceLevel.COMPLIANT])
        
        overall_score = sum(r.score for r in results) / len(results) if results else 0.0
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            standards, overall_score, compliant, non_compliant, critical_findings
        )
        
        # Generate remediation plan
        remediation_plan = self._generate_remediation_plan(results)
        
        return ComplianceReport(
            organization=self.organization,
            standards_assessed=standards,
            overall_compliance_score=overall_score,
            total_controls=len(results),
            compliant_controls=compliant,
            non_compliant_controls=non_compliant,
            partially_compliant_controls=partially_compliant,
            critical_findings=critical_findings,
            high_risk_findings=high_risk_findings,
            control_results=results,
            executive_summary=executive_summary,
            remediation_plan=remediation_plan,
            next_assessment_date=time.time() + (365 * 24 * 3600),  # Next year
            assessment_duration=duration,
            timestamp=time.time()
        )
    
    def _generate_executive_summary(self, standards: List[ComplianceStandard], 
                                  score: float, compliant: int, non_compliant: int, 
                                  critical: int) -> str:
        """Generate executive summary for compliance report"""
        standards_str = ", ".join([s.value for s in standards])
        
        summary = f"""
        EXECUTIVE SUMMARY - COMPLIANCE ASSESSMENT
        
        Standards Assessed: {standards_str}
        Overall Compliance Score: {score:.1f}%
        
        Key Findings:
        - {compliant} controls are compliant
        - {non_compliant} controls require attention
        - {critical} critical findings require immediate action
        
        Recommendation: {"Immediate remediation required" if critical > 0 else "Continue monitoring and improvement"}
        """
        
        return summary.strip()
    
    def _generate_remediation_plan(self, results: List[ComplianceTestResult]) -> List[str]:
        """Generate prioritized remediation plan"""
        plan = []
        
        # Critical issues first
        critical_controls = [r for r in results if r.risk_level == RiskLevel.CRITICAL and 
                           r.status != ComplianceLevel.COMPLIANT]
        if critical_controls:
            plan.append("IMMEDIATE ACTIONS (0-30 days):")
            for control in critical_controls[:3]:  # Top 3 critical
                plan.append(f"- Address {control.control_id}: {control.findings[0] if control.findings else 'Implement control'}")
        
        # High priority issues
        high_controls = [r for r in results if r.risk_level == RiskLevel.HIGH and 
                        r.status != ComplianceLevel.COMPLIANT]
        if high_controls:
            plan.append("\nSHORT-TERM ACTIONS (30-90 days):")
            for control in high_controls[:5]:  # Top 5 high priority
                plan.append(f"- {control.control_id}: {control.recommendations[0] if control.recommendations else 'Strengthen control'}")
        
        # Medium priority
        medium_controls = [r for r in results if r.risk_level == RiskLevel.MEDIUM and 
                          r.status != ComplianceLevel.COMPLIANT]
        if medium_controls:
            plan.append("\nMEDIUM-TERM ACTIONS (90-180 days):")
            plan.append(f"- Review and improve {len(medium_controls)} medium-priority controls")
            plan.append("- Implement continuous monitoring processes")
        
        plan.append("\nONGOING ACTIVITIES:")
        plan.append("- Regular compliance assessments (quarterly)")
        plan.append("- Evidence collection automation")
        plan.append("- Staff training and awareness programs")
        
        return plan