"""
Compliance Framework - PHASE 3.4

Enterprise-grade compliance framework supporting multiple regulatory standards
including SOX, GDPR, PCI-DSS, HIPAA, ISO 27001, NIST, and custom policies.
"""

import sqlite3
import json
import hashlib
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from threading import RLock
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from enum import Enum
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLite Database Schema
COMPLIANCE_FRAMEWORK_SCHEMA = '''
CREATE TABLE IF NOT EXISTS compliance_standards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_id TEXT NOT NULL UNIQUE,
    standard_name TEXT NOT NULL,
    version TEXT NOT NULL,
    authority TEXT NOT NULL,
    description TEXT,
    scope TEXT,
    effective_date DATE,
    status TEXT DEFAULT 'active',
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS compliance_requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requirement_id TEXT NOT NULL UNIQUE,
    standard_id TEXT NOT NULL,
    section TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    requirement_text TEXT NOT NULL,
    control_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    applicability TEXT,
    testing_procedures TEXT,
    evidence_requirements TEXT,
    implementation_guidance TEXT,
    related_frameworks TEXT,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (standard_id) REFERENCES compliance_standards (standard_id)
);

CREATE TABLE IF NOT EXISTS compliance_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id TEXT NOT NULL UNIQUE,
    standard_id TEXT NOT NULL,
    target_system TEXT NOT NULL,
    assessment_type TEXT NOT NULL,
    overall_status TEXT NOT NULL,
    overall_score REAL NOT NULL,
    total_requirements INTEGER DEFAULT 0,
    compliant_count INTEGER DEFAULT 0,
    non_compliant_count INTEGER DEFAULT 0,
    not_applicable_count INTEGER DEFAULT 0,
    evidence_collected TEXT,
    gaps_identified TEXT,
    remediation_plan TEXT,
    assessor TEXT,
    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    due_date TIMESTAMP,
    status TEXT DEFAULT 'active',
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS compliance_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id TEXT NOT NULL UNIQUE,
    assessment_id TEXT NOT NULL,
    requirement_id TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL,
    evidence TEXT,
    findings TEXT,
    gaps TEXT,
    recommendations TEXT,
    responsible_party TEXT,
    target_date TIMESTAMP,
    verified_by TEXT,
    verification_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (assessment_id) REFERENCES compliance_assessments (assessment_id),
    FOREIGN KEY (requirement_id) REFERENCES compliance_requirements (requirement_id)
);

CREATE TABLE IF NOT EXISTS remediation_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id TEXT NOT NULL UNIQUE,
    result_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    priority TEXT NOT NULL,
    effort_estimate TEXT,
    cost_estimate REAL,
    assigned_to TEXT,
    due_date TIMESTAMP,
    status TEXT DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    completion_evidence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (result_id) REFERENCES compliance_results (result_id)
);
'''

class ComplianceStandard(Enum):
    """Supported compliance standards."""
    SOX = "sox"  # Sarbanes-Oxley Act
    GDPR = "gdpr"  # General Data Protection Regulation
    PCI_DSS = "pci_dss"  # Payment Card Industry
    HIPAA = "hipaa"  # Health Insurance Portability
    ISO_27001 = "iso_27001"  # Information Security Management
    NIST_CSF = "nist_csf"  # NIST Cybersecurity Framework
    CIS_CONTROLS = "cis_controls"  # CIS Critical Security Controls
    OWASP_ASVS = "owasp_asvs"  # OWASP Application Security
    CCPA = "ccpa"  # California Consumer Privacy Act
    FISMA = "fisma"  # Federal Information Security
    FedRAMP = "fedramp"  # Federal Risk and Authorization
    COBIT = "cobit"  # Control Objectives for IT

class ComplianceStatus(Enum):
    """Compliance assessment status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNDER_REVIEW = "under_review"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"

class ControlType(Enum):
    """Control types."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    ADMINISTRATIVE = "administrative"
    TECHNICAL = "technical"
    PHYSICAL = "physical"

class SeverityLevel(Enum):
    """Severity levels for requirements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AssessmentType(Enum):
    """Assessment types."""
    SELF_ASSESSMENT = "self_assessment"
    THIRD_PARTY_AUDIT = "third_party_audit"
    REGULATORY_EXAMINATION = "regulatory_examination"
    CONTINUOUS_MONITORING = "continuous_monitoring"
    PENETRATION_TEST = "penetration_test"

@dataclass
class ComplianceRequirement:
    """Compliance requirement definition."""
    requirement_id: str
    standard_id: str
    section: str
    title: str
    description: str
    requirement_text: str
    control_type: ControlType
    severity: SeverityLevel
    applicability: Optional[str] = None
    testing_procedures: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    implementation_guidance: str = ""
    related_frameworks: List[str] = field(default_factory=list)
    status: str = "active"
    id: Optional[int] = None

@dataclass
class ComplianceResult:
    """Result of a compliance requirement assessment."""
    result_id: str
    assessment_id: str
    requirement_id: str
    status: ComplianceStatus
    score: float  # 0.0 to 100.0
    evidence: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    responsible_party: Optional[str] = None
    target_date: Optional[datetime] = None
    verified_by: Optional[str] = None
    verification_date: Optional[datetime] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None

@dataclass
class ComplianceAssessment:
    """Complete compliance assessment."""
    assessment_id: str
    standard_id: str
    target_system: str
    assessment_type: AssessmentType
    overall_status: ComplianceStatus
    overall_score: float
    total_requirements: int = 0
    compliant_count: int = 0
    non_compliant_count: int = 0
    not_applicable_count: int = 0
    evidence_collected: List[str] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
    remediation_plan: List[Dict[str, Any]] = field(default_factory=list)
    assessor: Optional[str] = None
    assessment_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: List[ComplianceResult] = field(default_factory=list)
    id: Optional[int] = None

@dataclass
class RemediationAction:
    """Remediation action for compliance gaps."""
    action_id: str
    result_id: str
    action_type: str
    title: str
    description: str
    priority: str
    effort_estimate: Optional[str] = None
    cost_estimate: Optional[float] = None
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    status: str = "pending"
    progress: int = 0
    completion_evidence: List[str] = field(default_factory=list)
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class ComplianceEngine:
    """Core compliance assessment engine."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = RLock()
        self.requirement_checkers: Dict[str, Callable] = {}
        self._register_built_in_checkers()
    
    def assess_requirement(self, requirement: ComplianceRequirement,
                          assessment_context: Dict[str, Any]) -> ComplianceResult:
        """Assess a single compliance requirement."""
        result_id = hashlib.sha256(
            f"{requirement.requirement_id}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        result = ComplianceResult(
            result_id=result_id,
            assessment_id=assessment_context.get('assessment_id', ''),
            requirement_id=requirement.requirement_id,
            status=ComplianceStatus.UNDER_REVIEW,
            score=0.0,
            created_at=datetime.now()
        )
        
        # Check if we have a specific checker for this requirement
        checker_key = f"{requirement.standard_id}_{requirement.section}"
        if checker_key in self.requirement_checkers:
            try:
                self.requirement_checkers[checker_key](requirement, assessment_context, result)
            except Exception as e:
                logger.error(f"Error in requirement checker {checker_key}: {e}")
                result.findings.append(f"Assessment error: {str(e)}")
                result.status = ComplianceStatus.UNDER_REVIEW
        else:
            # Perform generic assessment
            self._generic_requirement_assessment(requirement, assessment_context, result)
        
        return result
    
    def _generic_requirement_assessment(self, requirement: ComplianceRequirement,
                                      context: Dict[str, Any], result: ComplianceResult):
        """Generic requirement assessment logic."""
        # Analyze based on control type and available evidence
        system_info = context.get('system_info', {})
        code_analysis = context.get('code_analysis', {})
        documentation = context.get('documentation', [])
        
        score = 50.0  # Neutral starting point
        evidence = []
        findings = []
        gaps = []
        recommendations = []
        
        # Check for documentation evidence
        relevant_docs = [doc for doc in documentation 
                        if any(keyword in doc.lower() for keyword in 
                              requirement.title.lower().split())]
        if relevant_docs:
            evidence.extend(relevant_docs)
            score += 20.0
        else:
            gaps.append(f"No documentation found for {requirement.title}")
            recommendations.append(f"Create documentation for {requirement.title}")
        
        # Check for code implementation (if technical control)
        if requirement.control_type == ControlType.TECHNICAL:
            security_features = code_analysis.get('security_features', [])
            if security_features:
                evidence.append(f"Security features implemented: {', '.join(security_features)}")
                score += 15.0
            else:
                gaps.append("No security features detected in code analysis")
                recommendations.append("Implement required security controls in code")
        
        # Adjust score based on severity
        if requirement.severity == SeverityLevel.CRITICAL:
            if score < 80.0:
                score = max(score - 10.0, 0.0)  # Penalty for critical requirements
        
        # Determine status based on score
        if score >= 90.0:
            status = ComplianceStatus.COMPLIANT
        elif score >= 70.0:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        elif score >= 30.0:
            status = ComplianceStatus.NON_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        # Update result
        result.status = status
        result.score = min(100.0, score)
        result.evidence = evidence
        result.findings = findings
        result.gaps = gaps
        result.recommendations = recommendations
    
    def _register_built_in_checkers(self):
        """Register built-in requirement checkers."""
        # SOX checkers
        self.requirement_checkers["sox_302"] = self._check_sox_302
        self.requirement_checkers["sox_404"] = self._check_sox_404
        
        # GDPR checkers
        self.requirement_checkers["gdpr_25"] = self._check_gdpr_25
        self.requirement_checkers["gdpr_32"] = self._check_gdpr_32
        
        # PCI-DSS checkers
        self.requirement_checkers["pci_dss_3"] = self._check_pci_3
        self.requirement_checkers["pci_dss_4"] = self._check_pci_4
        
        # OWASP ASVS checkers
        self.requirement_checkers["owasp_asvs_2"] = self._check_owasp_auth
        self.requirement_checkers["owasp_asvs_5"] = self._check_owasp_input
    
    def _check_sox_302(self, req: ComplianceRequirement, 
                      context: Dict[str, Any], result: ComplianceResult):
        """Check SOX Section 302 - Management Assessment."""
        # Look for management review controls
        documentation = context.get('documentation', [])
        has_mgmt_review = any('management review' in doc.lower() or 
                             'executive sign-off' in doc.lower() 
                             for doc in documentation)
        
        if has_mgmt_review:
            result.status = ComplianceStatus.COMPLIANT
            result.score = 95.0
            result.evidence.append("Management review documentation found")
        else:
            result.status = ComplianceStatus.NON_COMPLIANT
            result.score = 20.0
            result.gaps.append("No management review process documented")
            result.recommendations.append("Implement management review and sign-off process")
    
    def _check_sox_404(self, req: ComplianceRequirement, 
                      context: Dict[str, Any], result: ComplianceResult):
        """Check SOX Section 404 - Internal Controls."""
        code_analysis = context.get('code_analysis', {})
        has_controls = code_analysis.get('access_controls', False)
        has_audit_logs = code_analysis.get('audit_logging', False)
        
        score = 30.0
        evidence = []
        gaps = []
        recommendations = []
        
        if has_controls:
            score += 35.0
            evidence.append("Access controls implemented")
        else:
            gaps.append("No access controls detected")
            recommendations.append("Implement role-based access controls")
        
        if has_audit_logs:
            score += 35.0
            evidence.append("Audit logging implemented")
        else:
            gaps.append("No audit logging detected")
            recommendations.append("Implement comprehensive audit logging")
        
        result.score = score
        result.evidence = evidence
        result.gaps = gaps
        result.recommendations = recommendations
        result.status = (ComplianceStatus.COMPLIANT if score >= 85 else
                        ComplianceStatus.PARTIALLY_COMPLIANT if score >= 60 else
                        ComplianceStatus.NON_COMPLIANT)
    
    def _check_gdpr_25(self, req: ComplianceRequirement, 
                      context: Dict[str, Any], result: ComplianceResult):
        """Check GDPR Article 25 - Data Protection by Design."""
        code_analysis = context.get('code_analysis', {})
        has_encryption = code_analysis.get('encryption', False)
        has_data_minimization = code_analysis.get('data_minimization', False)
        has_privacy_controls = code_analysis.get('privacy_controls', False)
        
        score = 20.0
        if has_encryption:
            score += 30.0
            result.evidence.append("Encryption implemented")
        if has_data_minimization:
            score += 25.0
            result.evidence.append("Data minimization practices detected")
        if has_privacy_controls:
            score += 25.0
            result.evidence.append("Privacy controls implemented")
        
        result.score = score
        result.status = (ComplianceStatus.COMPLIANT if score >= 85 else
                        ComplianceStatus.PARTIALLY_COMPLIANT if score >= 60 else
                        ComplianceStatus.NON_COMPLIANT)
        
        if score < 60:
            result.gaps.append("Insufficient privacy by design implementation")
            result.recommendations.append("Implement comprehensive privacy controls")
    
    def _check_gdpr_32(self, req: ComplianceRequirement, 
                      context: Dict[str, Any], result: ComplianceResult):
        """Check GDPR Article 32 - Security of Processing."""
        code_analysis = context.get('code_analysis', {})
        security_score = code_analysis.get('security_score', 0)
        
        if security_score >= 80:
            result.status = ComplianceStatus.COMPLIANT
            result.score = 90.0
            result.evidence.append(f"High security score: {security_score}")
        elif security_score >= 60:
            result.status = ComplianceStatus.PARTIALLY_COMPLIANT
            result.score = 70.0
            result.evidence.append(f"Moderate security score: {security_score}")
            result.recommendations.append("Enhance security controls")
        else:
            result.status = ComplianceStatus.NON_COMPLIANT
            result.score = 30.0
            result.gaps.append(f"Low security score: {security_score}")
            result.recommendations.append("Implement comprehensive security controls")
    
    def _check_pci_3(self, req: ComplianceRequirement, 
                    context: Dict[str, Any], result: ComplianceResult):
        """Check PCI-DSS Requirement 3 - Protect Cardholder Data."""
        code_analysis = context.get('code_analysis', {})
        has_encryption = code_analysis.get('strong_encryption', False)
        has_key_management = code_analysis.get('key_management', False)
        
        score = 25.0
        if has_encryption:
            score += 40.0
            result.evidence.append("Strong encryption detected")
        if has_key_management:
            score += 35.0
            result.evidence.append("Key management system detected")
        
        result.score = score
        result.status = (ComplianceStatus.COMPLIANT if score >= 85 else
                        ComplianceStatus.NON_COMPLIANT)  # PCI is strict
        
        if not has_encryption:
            result.gaps.append("No strong encryption for cardholder data")
            result.recommendations.append("Implement AES encryption for cardholder data")
    
    def _check_pci_4(self, req: ComplianceRequirement, 
                    context: Dict[str, Any], result: ComplianceResult):
        """Check PCI-DSS Requirement 4 - Encrypt Transmission."""
        code_analysis = context.get('code_analysis', {})
        has_tls = code_analysis.get('tls_encryption', False)
        tls_version = code_analysis.get('tls_version', '')
        
        score = 20.0
        if has_tls:
            score += 50.0
            result.evidence.append("TLS encryption detected")
            if 'tls1.2' in tls_version.lower() or 'tls1.3' in tls_version.lower():
                score += 30.0
                result.evidence.append(f"Secure TLS version: {tls_version}")
            else:
                result.gaps.append("Outdated TLS version")
                result.recommendations.append("Upgrade to TLS 1.2 or higher")
        else:
            result.gaps.append("No TLS encryption detected")
            result.recommendations.append("Implement TLS encryption for data transmission")
        
        result.score = score
        result.status = (ComplianceStatus.COMPLIANT if score >= 85 else
                        ComplianceStatus.NON_COMPLIANT)
    
    def _check_owasp_auth(self, req: ComplianceRequirement, 
                         context: Dict[str, Any], result: ComplianceResult):
        """Check OWASP ASVS Authentication requirements."""
        code_analysis = context.get('code_analysis', {})
        has_strong_auth = code_analysis.get('strong_authentication', False)
        has_session_mgmt = code_analysis.get('session_management', False)
        has_mfa = code_analysis.get('multi_factor_auth', False)
        
        score = 20.0
        if has_strong_auth:
            score += 30.0
            result.evidence.append("Strong authentication implemented")
        if has_session_mgmt:
            score += 25.0
            result.evidence.append("Session management implemented")
        if has_mfa:
            score += 25.0
            result.evidence.append("Multi-factor authentication detected")
        
        result.score = score
        result.status = (ComplianceStatus.COMPLIANT if score >= 80 else
                        ComplianceStatus.PARTIALLY_COMPLIANT if score >= 60 else
                        ComplianceStatus.NON_COMPLIANT)
    
    def _check_owasp_input(self, req: ComplianceRequirement, 
                          context: Dict[str, Any], result: ComplianceResult):
        """Check OWASP ASVS Input Validation requirements."""
        code_analysis = context.get('code_analysis', {})
        has_input_validation = code_analysis.get('input_validation', False)
        has_output_encoding = code_analysis.get('output_encoding', False)
        has_sql_protection = code_analysis.get('sql_injection_protection', False)
        
        score = 15.0
        if has_input_validation:
            score += 30.0
            result.evidence.append("Input validation implemented")
        if has_output_encoding:
            score += 25.0
            result.evidence.append("Output encoding implemented")
        if has_sql_protection:
            score += 30.0
            result.evidence.append("SQL injection protection implemented")
        
        result.score = score
        result.status = (ComplianceStatus.COMPLIANT if score >= 85 else
                        ComplianceStatus.PARTIALLY_COMPLIANT if score >= 65 else
                        ComplianceStatus.NON_COMPLIANT)

class ComplianceFramework:
    """Enterprise compliance framework."""
    
    def __init__(self, db_path: str = "compliance.db"):
        self.db_path = db_path
        self.lock = RLock()
        self.engine = ComplianceEngine(db_path)
        
        self._initialize_db()
        self._load_standard_requirements()
    
    def _initialize_db(self):
        """Initialize database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(COMPLIANCE_FRAMEWORK_SCHEMA)
    
    def _load_standard_requirements(self):
        """Load standard compliance requirements."""
        standards = [
            {
                'standard_id': 'sox',
                'standard_name': 'Sarbanes-Oxley Act',
                'version': '2002',
                'authority': 'SEC',
                'description': 'Financial reporting and corporate governance',
                'scope': 'Public companies'
            },
            {
                'standard_id': 'gdpr',
                'standard_name': 'General Data Protection Regulation',
                'version': '2018',
                'authority': 'EU',
                'description': 'Data protection and privacy',
                'scope': 'EU personal data processing'
            },
            {
                'standard_id': 'pci_dss',
                'standard_name': 'Payment Card Industry Data Security Standard',
                'version': '4.0',
                'authority': 'PCI SSC',
                'description': 'Payment card data security',
                'scope': 'Organizations handling card data'
            },
            {
                'standard_id': 'owasp_asvs',
                'standard_name': 'OWASP Application Security Verification Standard',
                'version': '4.0',
                'authority': 'OWASP',
                'description': 'Application security requirements',
                'scope': 'Web applications and APIs'
            }
        ]
        
        for standard_data in standards:
            self._save_standard(standard_data)
            
        # Load requirements for each standard
        self._load_sox_requirements()
        self._load_gdpr_requirements()
        self._load_pci_requirements()
        self._load_owasp_requirements()
    
    def _save_standard(self, standard_data: Dict[str, Any]):
        """Save compliance standard to database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO compliance_standards
                    (standard_id, standard_name, version, authority, 
                     description, scope, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    standard_data['standard_id'],
                    standard_data['standard_name'],
                    standard_data['version'],
                    standard_data['authority'],
                    standard_data['description'],
                    standard_data['scope'],
                    json.dumps(standard_data)
                ))
    
    def _load_sox_requirements(self):
        """Load SOX requirements."""
        requirements = [
            {
                'requirement_id': 'sox_302',
                'section': '302',
                'title': 'Management Assessment of Internal Controls',
                'description': 'Management must assess and report on internal controls',
                'requirement_text': 'Principal officers must certify the accuracy of financial reports',
                'control_type': ControlType.ADMINISTRATIVE,
                'severity': SeverityLevel.CRITICAL
            },
            {
                'requirement_id': 'sox_404',
                'section': '404',
                'title': 'Internal Control Over Financial Reporting',
                'description': 'Establish and maintain internal controls',
                'requirement_text': 'Management must establish internal controls over financial reporting',
                'control_type': ControlType.TECHNICAL,
                'severity': SeverityLevel.CRITICAL
            }
        ]
        
        for req_data in requirements:
            req = ComplianceRequirement(
                requirement_id=req_data['requirement_id'],
                standard_id='sox',
                section=req_data['section'],
                title=req_data['title'],
                description=req_data['description'],
                requirement_text=req_data['requirement_text'],
                control_type=req_data['control_type'],
                severity=req_data['severity']
            )
            self._save_requirement(req)
    
    def _load_gdpr_requirements(self):
        """Load GDPR requirements."""
        requirements = [
            {
                'requirement_id': 'gdpr_25',
                'section': '25',
                'title': 'Data Protection by Design and by Default',
                'description': 'Implement data protection measures from the outset',
                'requirement_text': 'Technical and organizational measures to implement data protection principles',
                'control_type': ControlType.TECHNICAL,
                'severity': SeverityLevel.HIGH
            },
            {
                'requirement_id': 'gdpr_32',
                'section': '32',
                'title': 'Security of Processing',
                'description': 'Implement appropriate technical and organizational measures',
                'requirement_text': 'Ensure security of personal data processing',
                'control_type': ControlType.TECHNICAL,
                'severity': SeverityLevel.HIGH
            }
        ]
        
        for req_data in requirements:
            req = ComplianceRequirement(
                requirement_id=req_data['requirement_id'],
                standard_id='gdpr',
                section=req_data['section'],
                title=req_data['title'],
                description=req_data['description'],
                requirement_text=req_data['requirement_text'],
                control_type=req_data['control_type'],
                severity=req_data['severity']
            )
            self._save_requirement(req)
    
    def _load_pci_requirements(self):
        """Load PCI-DSS requirements."""
        requirements = [
            {
                'requirement_id': 'pci_3',
                'section': '3',
                'title': 'Protect Stored Cardholder Data',
                'description': 'Protect stored cardholder data',
                'requirement_text': 'Cardholder data must be protected with strong cryptography',
                'control_type': ControlType.TECHNICAL,
                'severity': SeverityLevel.CRITICAL
            },
            {
                'requirement_id': 'pci_4',
                'section': '4',
                'title': 'Encrypt Transmission of Cardholder Data',
                'description': 'Encrypt transmission across open, public networks',
                'requirement_text': 'Cardholder data must be encrypted during transmission',
                'control_type': ControlType.TECHNICAL,
                'severity': SeverityLevel.CRITICAL
            }
        ]
        
        for req_data in requirements:
            req = ComplianceRequirement(
                requirement_id=req_data['requirement_id'],
                standard_id='pci_dss',
                section=req_data['section'],
                title=req_data['title'],
                description=req_data['description'],
                requirement_text=req_data['requirement_text'],
                control_type=req_data['control_type'],
                severity=req_data['severity']
            )
            self._save_requirement(req)
    
    def _load_owasp_requirements(self):
        """Load OWASP ASVS requirements."""
        requirements = [
            {
                'requirement_id': 'owasp_asvs_2',
                'section': '2',
                'title': 'Authentication Verification Requirements',
                'description': 'Verify authentication mechanisms',
                'requirement_text': 'Authentication controls must be implemented',
                'control_type': ControlType.TECHNICAL,
                'severity': SeverityLevel.HIGH
            },
            {
                'requirement_id': 'owasp_asvs_5',
                'section': '5',
                'title': 'Input Validation Verification Requirements',
                'description': 'Verify input validation controls',
                'requirement_text': 'All input must be validated and sanitized',
                'control_type': ControlType.TECHNICAL,
                'severity': SeverityLevel.HIGH
            }
        ]
        
        for req_data in requirements:
            req = ComplianceRequirement(
                requirement_id=req_data['requirement_id'],
                standard_id='owasp_asvs',
                section=req_data['section'],
                title=req_data['title'],
                description=req_data['description'],
                requirement_text=req_data['requirement_text'],
                control_type=req_data['control_type'],
                severity=req_data['severity']
            )
            self._save_requirement(req)
    
    def _save_requirement(self, requirement: ComplianceRequirement):
        """Save compliance requirement to database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO compliance_requirements
                    (requirement_id, standard_id, section, title, description,
                     requirement_text, control_type, severity, applicability,
                     testing_procedures, evidence_requirements, implementation_guidance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    requirement.requirement_id,
                    requirement.standard_id,
                    requirement.section,
                    requirement.title,
                    requirement.description,
                    requirement.requirement_text,
                    requirement.control_type.value,
                    requirement.severity.value,
                    requirement.applicability,
                    json.dumps(requirement.testing_procedures),
                    json.dumps(requirement.evidence_requirements),
                    requirement.implementation_guidance
                ))
    
    def start_assessment(self, standard_id: str, target_system: str,
                        assessment_type: AssessmentType = AssessmentType.SELF_ASSESSMENT,
                        assessor: str = None) -> str:
        """Start a compliance assessment."""
        assessment_id = hashlib.sha256(
            f"{standard_id}_{target_system}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            standard_id=standard_id,
            target_system=target_system,
            assessment_type=assessment_type,
            overall_status=ComplianceStatus.UNDER_REVIEW,
            overall_score=0.0,
            assessor=assessor,
            assessment_date=datetime.now()
        )
        
        self._save_assessment(assessment)
        return assessment_id
    
    def conduct_assessment(self, assessment_id: str, 
                          assessment_context: Dict[str, Any]) -> ComplianceAssessment:
        """Conduct compliance assessment."""
        assessment = self._get_assessment(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        # Get requirements for this standard
        requirements = self._get_requirements(assessment.standard_id)
        assessment_context['assessment_id'] = assessment_id
        
        results = []
        for requirement in requirements:
            result = self.engine.assess_requirement(requirement, assessment_context)
            results.append(result)
            self._save_result(result)
        
        # Calculate overall assessment scores
        assessment.results = results
        assessment.total_requirements = len(results)
        assessment.compliant_count = len([r for r in results if r.status == ComplianceStatus.COMPLIANT])
        assessment.non_compliant_count = len([r for r in results if r.status == ComplianceStatus.NON_COMPLIANT])
        assessment.not_applicable_count = len([r for r in results if r.status == ComplianceStatus.NOT_APPLICABLE])
        
        # Calculate overall score
        if results:
            assessment.overall_score = sum(r.score for r in results) / len(results)
        
        # Determine overall status
        compliance_rate = assessment.compliant_count / max(1, len(results) - assessment.not_applicable_count)
        if compliance_rate >= 0.95:
            assessment.overall_status = ComplianceStatus.COMPLIANT
        elif compliance_rate >= 0.70:
            assessment.overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            assessment.overall_status = ComplianceStatus.NON_COMPLIANT
        
        # Generate remediation plan
        assessment.remediation_plan = self._generate_remediation_plan(assessment)
        
        # Update assessment in database
        self._update_assessment(assessment)
        
        return assessment
    
    def _save_assessment(self, assessment: ComplianceAssessment):
        """Save compliance assessment to database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO compliance_assessments
                    (assessment_id, standard_id, target_system, assessment_type,
                     overall_status, overall_score, assessor, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    assessment.assessment_id,
                    assessment.standard_id,
                    assessment.target_system,
                    assessment.assessment_type.value,
                    assessment.overall_status.value,
                    assessment.overall_score,
                    assessment.assessor,
                    json.dumps(assessment.metadata)
                ))
    
    def _get_assessment(self, assessment_id: str) -> Optional[ComplianceAssessment]:
        """Get assessment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM compliance_assessments WHERE assessment_id = ?
            ''', (assessment_id,))
            row = cursor.fetchone()
            
            if row:
                return ComplianceAssessment(
                    id=row['id'],
                    assessment_id=row['assessment_id'],
                    standard_id=row['standard_id'],
                    target_system=row['target_system'],
                    assessment_type=AssessmentType(row['assessment_type']),
                    overall_status=ComplianceStatus(row['overall_status']),
                    overall_score=row['overall_score'],
                    assessor=row['assessor'],
                    assessment_date=datetime.fromisoformat(row['assessment_date']),
                    metadata=json.loads(row['metadata'] or '{}')
                )
        return None
    
    def _get_requirements(self, standard_id: str) -> List[ComplianceRequirement]:
        """Get requirements for a standard."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM compliance_requirements 
                WHERE standard_id = ? AND status = 'active'
            ''', (standard_id,))
            rows = cursor.fetchall()
            
            requirements = []
            for row in rows:
                req = ComplianceRequirement(
                    id=row['id'],
                    requirement_id=row['requirement_id'],
                    standard_id=row['standard_id'],
                    section=row['section'],
                    title=row['title'],
                    description=row['description'],
                    requirement_text=row['requirement_text'],
                    control_type=ControlType(row['control_type']),
                    severity=SeverityLevel(row['severity']),
                    applicability=row['applicability'],
                    testing_procedures=json.loads(row['testing_procedures'] or '[]'),
                    evidence_requirements=json.loads(row['evidence_requirements'] or '[]'),
                    implementation_guidance=row['implementation_guidance'] or '',
                    status=row['status']
                )
                requirements.append(req)
            
            return requirements
    
    def _save_result(self, result: ComplianceResult):
        """Save compliance result to database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO compliance_results
                    (result_id, assessment_id, requirement_id, status, score,
                     evidence, findings, gaps, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.result_id,
                    result.assessment_id,
                    result.requirement_id,
                    result.status.value,
                    result.score,
                    json.dumps(result.evidence),
                    json.dumps(result.findings),
                    json.dumps(result.gaps),
                    json.dumps(result.recommendations)
                ))
    
    def _update_assessment(self, assessment: ComplianceAssessment):
        """Update assessment in database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE compliance_assessments 
                    SET overall_status = ?, overall_score = ?, 
                        total_requirements = ?, compliant_count = ?,
                        non_compliant_count = ?, not_applicable_count = ?,
                        remediation_plan = ?
                    WHERE assessment_id = ?
                ''', (
                    assessment.overall_status.value,
                    assessment.overall_score,
                    assessment.total_requirements,
                    assessment.compliant_count,
                    assessment.non_compliant_count,
                    assessment.not_applicable_count,
                    json.dumps(assessment.remediation_plan),
                    assessment.assessment_id
                ))
    
    def _generate_remediation_plan(self, assessment: ComplianceAssessment) -> List[Dict[str, Any]]:
        """Generate remediation plan for assessment gaps."""
        plan = []
        
        # Focus on non-compliant critical and high severity items
        critical_results = [r for r in assessment.results 
                           if r.status == ComplianceStatus.NON_COMPLIANT]
        
        for result in critical_results:
            req = next((r for r in self._get_requirements(assessment.standard_id)
                       if r.requirement_id == result.requirement_id), None)
            
            if req:
                action = {
                    'requirement_id': result.requirement_id,
                    'title': f"Address {req.title}",
                    'description': req.implementation_guidance or "Implement compliance controls",
                    'priority': req.severity.value,
                    'estimated_effort': self._estimate_effort(req, result),
                    'recommendations': result.recommendations,
                    'gaps': result.gaps,
                    'target_date': (datetime.now() + timedelta(days=30)).isoformat()
                }
                plan.append(action)
        
        return plan
    
    def _estimate_effort(self, requirement: ComplianceRequirement, 
                        result: ComplianceResult) -> str:
        """Estimate effort required for remediation."""
        if requirement.severity == SeverityLevel.CRITICAL:
            return "High"
        elif requirement.control_type == ControlType.TECHNICAL:
            return "Medium"
        else:
            return "Low"
    
    def get_assessment_status(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """Get assessment status."""
        assessment = self._get_assessment(assessment_id)
        if assessment:
            return {
                'assessment_id': assessment.assessment_id,
                'standard': assessment.standard_id,
                'target_system': assessment.target_system,
                'status': assessment.overall_status.value,
                'score': assessment.overall_score,
                'total_requirements': assessment.total_requirements,
                'compliant': assessment.compliant_count,
                'non_compliant': assessment.non_compliant_count,
                'assessment_date': assessment.assessment_date.isoformat() if assessment.assessment_date else None
            }
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compliance framework statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total standards and requirements
            cursor.execute('SELECT COUNT(*) FROM compliance_standards WHERE status = "active"')
            stats['total_standards'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM compliance_requirements WHERE status = "active"')
            stats['total_requirements'] = cursor.fetchone()[0]
            
            # Assessments
            cursor.execute('SELECT COUNT(*) FROM compliance_assessments')
            stats['total_assessments'] = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT overall_status, COUNT(*) FROM compliance_assessments 
                GROUP BY overall_status
            ''')
            stats['assessments_by_status'] = dict(cursor.fetchall())
            
            # Recent assessments
            cursor.execute('''
                SELECT COUNT(*) FROM compliance_assessments 
                WHERE assessment_date > datetime('now', '-30 days')
            ''')
            stats['recent_assessments'] = cursor.fetchone()[0]
            
            return stats

# Global instance
compliance_framework = ComplianceFramework()

# Convenience functions
def start_compliance_assessment(standard_id: str, target_system: str,
                               assessment_type: AssessmentType = AssessmentType.SELF_ASSESSMENT) -> str:
    """Start compliance assessment."""
    return compliance_framework.start_assessment(standard_id, target_system, assessment_type)

def conduct_compliance_assessment(assessment_id: str, context: Dict[str, Any]) -> ComplianceAssessment:
    """Conduct compliance assessment."""
    return compliance_framework.conduct_assessment(assessment_id, context)

def get_compliance_statistics() -> Dict[str, Any]:
    """Get compliance statistics."""
    return compliance_framework.get_statistics()