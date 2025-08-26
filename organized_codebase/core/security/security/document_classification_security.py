"""
MetaGPT Derived Document Classification Security
Extracted from MetaGPT document handling patterns and content moderation
Enhanced for comprehensive document security and sensitive information protection
"""

import logging
import re
import hashlib
import time
from typing import Dict, Any, Optional, List, Set, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from .error_handler import SecurityError, ValidationError, security_error_handler


class DocumentClassification(Enum):
    """Document security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class SensitivityLevel(Enum):
    """Document sensitivity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentType(Enum):
    """Types of content in documents"""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    CONFIGURATION = "configuration"
    CREDENTIALS = "credentials"
    PERSONAL_INFO = "personal_info"
    FINANCIAL = "financial"
    MEDICAL = "medical"


class ScanResult(Enum):
    """Document scanning results"""
    CLEAN = "clean"
    WARNING = "warning"
    SENSITIVE = "sensitive"
    BLOCKED = "blocked"
    QUARANTINE = "quarantine"


@dataclass
class SensitivePattern:
    """Pattern for detecting sensitive information"""
    pattern_id: str
    name: str
    regex_pattern: str
    content_type: ContentType
    sensitivity_level: SensitivityLevel
    description: str
    action: ScanResult = ScanResult.WARNING
    confidence_threshold: float = 0.7
    
    def matches(self, text: str) -> List[Dict[str, Any]]:
        """Find matches of this pattern in text"""
        matches = []
        try:
            for match in re.finditer(self.regex_pattern, text, re.IGNORECASE | re.MULTILINE):
                matches.append({
                    'match': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'line_number': text[:match.start()].count('\n') + 1,
                    'pattern_id': self.pattern_id,
                    'content_type': self.content_type.value,
                    'sensitivity': self.sensitivity_level.value,
                    'confidence': 1.0  # Regex matches are always confident
                })
        except Exception as e:
            logging.getLogger(__name__).error(f"Pattern matching error: {e}")
        
        return matches


@dataclass
class DocumentMetadata:
    """Comprehensive document metadata"""
    document_id: str
    filename: str
    file_size: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    classification: DocumentClassification = DocumentClassification.INTERNAL
    sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
    content_hash: Optional[str] = None
    author: Optional[str] = None
    department: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_access_log(self, user_id: str, action: str, details: Dict[str, Any] = None):
        """Add access log entry"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'details': details or {}
        }
        self.access_log.append(log_entry)
        
        # Limit log size
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]


@dataclass
class ScanReport:
    """Document security scan report"""
    document_id: str
    scan_id: str
    scan_timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_result: ScanResult = ScanResult.CLEAN
    classification: DocumentClassification = DocumentClassification.PUBLIC
    sensitivity_level: SensitivityLevel = SensitivityLevel.LOW
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    scan_duration_ms: float = 0.0
    
    @property
    def has_sensitive_content(self) -> bool:
        """Check if document contains sensitive content"""
        return self.overall_result in [ScanResult.SENSITIVE, ScanResult.BLOCKED, ScanResult.QUARANTINE]
    
    @property
    def critical_findings_count(self) -> int:
        """Count critical findings"""
        return sum(1 for finding in self.findings 
                  if finding.get('sensitivity') == SensitivityLevel.CRITICAL.value)


class PatternLibrary:
    """Library of sensitive information detection patterns"""
    
    def __init__(self):
        self.patterns: Dict[str, SensitivePattern] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_default_patterns()
    
    def add_pattern(self, pattern: SensitivePattern):
        """Add new detection pattern"""
        self.patterns[pattern.pattern_id] = pattern
        self.logger.info(f"Added detection pattern: {pattern.name}")
    
    def get_patterns_by_type(self, content_type: ContentType) -> List[SensitivePattern]:
        """Get patterns for specific content type"""
        return [pattern for pattern in self.patterns.values() 
                if pattern.content_type == content_type]
    
    def get_all_patterns(self) -> List[SensitivePattern]:
        """Get all detection patterns"""
        return list(self.patterns.values())
    
    def _initialize_default_patterns(self):
        """Initialize default sensitive information patterns"""
        
        # Personal Information Patterns
        self.patterns['ssn'] = SensitivePattern(
            pattern_id='ssn',
            name='Social Security Number',
            regex_pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
            content_type=ContentType.PERSONAL_INFO,
            sensitivity_level=SensitivityLevel.CRITICAL,
            description='US Social Security Number',
            action=ScanResult.BLOCKED
        )
        
        self.patterns['credit_card'] = SensitivePattern(
            pattern_id='credit_card',
            name='Credit Card Number',
            regex_pattern=r'\b(?:\d{4}[\s-]?){3}\d{4}\b',
            content_type=ContentType.FINANCIAL,
            sensitivity_level=SensitivityLevel.CRITICAL,
            description='Credit card number',
            action=ScanResult.BLOCKED
        )
        
        self.patterns['email'] = SensitivePattern(
            pattern_id='email',
            name='Email Address',
            regex_pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            content_type=ContentType.PERSONAL_INFO,
            sensitivity_level=SensitivityLevel.MEDIUM,
            description='Email address',
            action=ScanResult.WARNING
        )
        
        self.patterns['phone'] = SensitivePattern(
            pattern_id='phone',
            name='Phone Number',
            regex_pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            content_type=ContentType.PERSONAL_INFO,
            sensitivity_level=SensitivityLevel.MEDIUM,
            description='Phone number',
            action=ScanResult.WARNING
        )
        
        # Credential Patterns
        self.patterns['api_key'] = SensitivePattern(
            pattern_id='api_key',
            name='API Key',
            regex_pattern=r'(?i)(api[_-]?key|apikey)[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9]{20,})',
            content_type=ContentType.CREDENTIALS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            description='API key or token',
            action=ScanResult.BLOCKED
        )
        
        self.patterns['password'] = SensitivePattern(
            pattern_id='password',
            name='Password',
            regex_pattern=r'(?i)(password|pwd|pass)[\'"\s]*[:=][\'"\s]*([^\s\'"]{8,})',
            content_type=ContentType.CREDENTIALS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            description='Password field',
            action=ScanResult.BLOCKED
        )
        
        self.patterns['private_key'] = SensitivePattern(
            pattern_id='private_key',
            name='Private Key',
            regex_pattern=r'-----BEGIN (RSA )?PRIVATE KEY-----',
            content_type=ContentType.CREDENTIALS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            description='Private key file',
            action=ScanResult.BLOCKED
        )
        
        # Financial Patterns
        self.patterns['bank_account'] = SensitivePattern(
            pattern_id='bank_account',
            name='Bank Account Number',
            regex_pattern=r'\b\d{8,17}\b',
            content_type=ContentType.FINANCIAL,
            sensitivity_level=SensitivityLevel.HIGH,
            description='Potential bank account number',
            action=ScanResult.SENSITIVE
        )
        
        # Medical Patterns
        self.patterns['medical_record'] = SensitivePattern(
            pattern_id='medical_record',
            name='Medical Record Number',
            regex_pattern=r'(?i)(medical|patient|mrn)[_\s-]?(record|number|id)[\'"\s]*[:=][\'"\s]*([A-Z0-9]{6,})',
            content_type=ContentType.MEDICAL,
            sensitivity_level=SensitivityLevel.CRITICAL,
            description='Medical record identifier',
            action=ScanResult.BLOCKED
        )
        
        # Code Patterns
        self.patterns['secret_key'] = SensitivePattern(
            pattern_id='secret_key',
            name='Secret Key',
            regex_pattern=r'(?i)(secret[_-]?key|secretkey)[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9+/]{16,})',
            content_type=ContentType.CREDENTIALS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            description='Secret key or token',
            action=ScanResult.BLOCKED
        )
        
        self.patterns['connection_string'] = SensitivePattern(
            pattern_id='connection_string',
            name='Database Connection String',
            regex_pattern=r'(?i)(connection[_-]?string|connectionstring|server=|database=)[\'"\s]*[:=][\'"\s]*([^\'";]{10,})',
            content_type=ContentType.CONFIGURATION,
            sensitivity_level=SensitivityLevel.HIGH,
            description='Database connection string',
            action=ScanResult.SENSITIVE
        )


class DocumentScanner:
    """Advanced document security scanner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pattern_library = PatternLibrary()
        self.scan_history: List[ScanReport] = []
        self.max_scan_history = 10000
    
    def scan_document(self, content: str, document_id: str = None,
                     document_type: str = "text") -> ScanReport:
        """Perform comprehensive security scan of document content"""
        scan_start = time.time()
        
        try:
            document_id = document_id or hashlib.sha256(content.encode()).hexdigest()[:12]
            scan_id = f"scan_{document_id}_{int(time.time())}"
            
            report = ScanReport(
                document_id=document_id,
                scan_id=scan_id
            )
            
            # Scan with all patterns
            all_findings = []
            patterns = self.pattern_library.get_all_patterns()
            
            for pattern in patterns:
                matches = pattern.matches(content)
                for match in matches:
                    finding = {
                        **match,
                        'pattern_name': pattern.name,
                        'description': pattern.description,
                        'action': pattern.action.value
                    }
                    all_findings.append(finding)
            
            report.findings = all_findings
            
            # Determine overall classification and sensitivity
            if all_findings:
                # Get highest sensitivity level found
                sensitivity_levels = [finding['sensitivity'] for finding in all_findings]
                
                if SensitivityLevel.CRITICAL.value in sensitivity_levels:
                    report.sensitivity_level = SensitivityLevel.CRITICAL
                    report.classification = DocumentClassification.RESTRICTED
                elif SensitivityLevel.HIGH.value in sensitivity_levels:
                    report.sensitivity_level = SensitivityLevel.HIGH
                    report.classification = DocumentClassification.CONFIDENTIAL
                elif SensitivityLevel.MEDIUM.value in sensitivity_levels:
                    report.sensitivity_level = SensitivityLevel.MEDIUM
                    report.classification = DocumentClassification.INTERNAL
                else:
                    report.sensitivity_level = SensitivityLevel.LOW
                    report.classification = DocumentClassification.INTERNAL
                
                # Determine overall scan result
                actions = [finding['action'] for finding in all_findings]
                
                if ScanResult.BLOCKED.value in actions:
                    report.overall_result = ScanResult.BLOCKED
                elif ScanResult.QUARANTINE.value in actions:
                    report.overall_result = ScanResult.QUARANTINE
                elif ScanResult.SENSITIVE.value in actions:
                    report.overall_result = ScanResult.SENSITIVE
                elif ScanResult.WARNING.value in actions:
                    report.overall_result = ScanResult.WARNING
                else:
                    report.overall_result = ScanResult.CLEAN
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            # Calculate scan duration
            report.scan_duration_ms = (time.time() - scan_start) * 1000
            
            # Add to scan history
            self._add_to_scan_history(report)
            
            self.logger.info(f"Document scan completed: {scan_id} - {report.overall_result.value}")
            return report
            
        except Exception as e:
            scan_duration = (time.time() - scan_start) * 1000
            
            error_report = ScanReport(
                document_id=document_id or "error",
                scan_id=f"error_{int(time.time())}",
                overall_result=ScanResult.BLOCKED,
                scan_duration_ms=scan_duration,
                findings=[{
                    'pattern_id': 'scan_error',
                    'content_type': ContentType.TEXT.value,
                    'sensitivity': SensitivityLevel.CRITICAL.value,
                    'description': f"Scan error: {str(e)}",
                    'action': ScanResult.BLOCKED.value,
                    'match': 'SCAN_ERROR',
                    'line_number': 1
                }]
            )
            
            self._add_to_scan_history(error_report)
            return error_report
    
    def _generate_recommendations(self, report: ScanReport) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []
        
        try:
            if report.overall_result == ScanResult.BLOCKED:
                recommendations.append("Document contains critical sensitive information and should not be shared")
                recommendations.append("Remove or redact all sensitive content before distribution")
            
            elif report.overall_result == ScanResult.SENSITIVE:
                recommendations.append("Document contains sensitive information requiring careful handling")
                recommendations.append("Ensure appropriate access controls are in place")
            
            elif report.overall_result == ScanResult.WARNING:
                recommendations.append("Review identified content for potential sensitivity")
            
            # Type-specific recommendations
            content_types = set(finding.get('content_type', '') for finding in report.findings)
            
            if ContentType.CREDENTIALS.value in content_types:
                recommendations.append("Remove all hardcoded credentials and use secure storage")
                recommendations.append("Rotate any exposed credentials immediately")
            
            if ContentType.PERSONAL_INFO.value in content_types:
                recommendations.append("Ensure compliance with privacy regulations (GDPR, CCPA, etc.)")
                recommendations.append("Consider data minimization and anonymization")
            
            if ContentType.FINANCIAL.value in content_types:
                recommendations.append("Apply financial data protection controls")
                recommendations.append("Ensure PCI DSS compliance if applicable")
            
            if ContentType.MEDICAL.value in content_types:
                recommendations.append("Ensure HIPAA compliance for medical information")
                recommendations.append("Apply appropriate safeguards for health data")
            
            # General recommendations
            if report.findings:
                recommendations.append("Implement data loss prevention (DLP) controls")
                recommendations.append("Regular security awareness training for document handling")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Manual review recommended due to analysis error")
        
        return recommendations
    
    def _add_to_scan_history(self, report: ScanReport):
        """Add scan report to history"""
        self.scan_history.append(report)
        
        # Limit history size
        if len(self.scan_history) > self.max_scan_history:
            self.scan_history = self.scan_history[-self.max_scan_history // 2:]


class DocumentClassificationManager:
    """Comprehensive document classification and security management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.documents: Dict[str, DocumentMetadata] = {}
        self.scanner = DocumentScanner()
        self.access_policies: Dict[DocumentClassification, Set[str]] = {}
        self.redaction_rules: Dict[str, str] = {}
        
        # Initialize default access policies
        self._initialize_access_policies()
        self._initialize_redaction_rules()
    
    def register_document(self, content: str, filename: str, 
                         author: str = None, department: str = None) -> DocumentMetadata:
        """Register and classify new document"""
        try:
            # Generate document ID
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            document_id = f"doc_{content_hash[:12]}"
            
            # Scan document for sensitive content
            scan_report = self.scanner.scan_document(content, document_id)
            
            # Create document metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_size=len(content.encode()),
                classification=scan_report.classification,
                sensitivity_level=scan_report.sensitivity_level,
                content_hash=content_hash,
                author=author,
                department=department
            )
            
            # Add security tags based on scan results
            if scan_report.has_sensitive_content:
                metadata.tags.add('sensitive_content')
            
            if scan_report.critical_findings_count > 0:
                metadata.tags.add('critical_findings')
            
            # Store document
            self.documents[document_id] = metadata
            
            self.logger.info(f"Registered document: {filename} (classification: {scan_report.classification.value})")
            return metadata
            
        except Exception as e:
            error = SecurityError(f"Failed to register document: {str(e)}", "DOC_REG_001")
            security_error_handler.handle_error(error)
            raise error
    
    def check_access_permission(self, document_id: str, user_id: str, 
                               user_clearance: DocumentClassification = DocumentClassification.PUBLIC) -> bool:
        """Check if user has permission to access document"""
        try:
            if document_id not in self.documents:
                return False
            
            document = self.documents[document_id]
            
            # Check clearance level
            clearance_hierarchy = {
                DocumentClassification.PUBLIC: 0,
                DocumentClassification.INTERNAL: 1,
                DocumentClassification.CONFIDENTIAL: 2,
                DocumentClassification.RESTRICTED: 3,
                DocumentClassification.TOP_SECRET: 4
            }
            
            user_level = clearance_hierarchy.get(user_clearance, 0)
            doc_level = clearance_hierarchy.get(document.classification, 0)
            
            has_access = user_level >= doc_level
            
            # Log access attempt
            document.add_access_log(
                user_id=user_id,
                action='access_check',
                details={
                    'user_clearance': user_clearance.value,
                    'document_classification': document.classification.value,
                    'access_granted': has_access
                }
            )
            
            return has_access
            
        except Exception as e:
            self.logger.error(f"Access check failed: {e}")
            return False
    
    def redact_sensitive_content(self, content: str, redaction_level: SensitivityLevel = SensitivityLevel.HIGH) -> Dict[str, Any]:
        """Redact sensitive content from document"""
        try:
            # Scan for sensitive content
            scan_report = self.scanner.scan_document(content)
            
            redacted_content = content
            redaction_log = []
            
            # Sort findings by position (reverse order to maintain positions)
            findings_by_position = sorted(
                [f for f in scan_report.findings 
                 if SensitivityLevel(f['sensitivity']).value >= redaction_level.value],
                key=lambda x: x['start'],
                reverse=True
            )
            
            for finding in findings_by_position:
                start = finding['start']
                end = finding['end']
                content_type = finding['content_type']
                
                # Get redaction replacement
                replacement = self.redaction_rules.get(content_type, '[REDACTED]')
                
                # Apply redaction
                redacted_content = redacted_content[:start] + replacement + redacted_content[end:]
                
                redaction_log.append({
                    'pattern': finding.get('pattern_name', 'Unknown'),
                    'content_type': content_type,
                    'original_length': end - start,
                    'redacted_with': replacement,
                    'line_number': finding.get('line_number', 0)
                })
            
            return {
                'original_content': content,
                'redacted_content': redacted_content,
                'redaction_count': len(redaction_log),
                'redaction_log': redaction_log,
                'redaction_level': redaction_level.value,
                'content_safe': len(redaction_log) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Content redaction failed: {e}")
            return {
                'original_content': content,
                'redacted_content': content,
                'redaction_count': 0,
                'redaction_log': [],
                'error': str(e),
                'content_safe': False
            }
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive document classification statistics"""
        try:
            if not self.documents:
                return {'total_documents': 0}
            
            total_documents = len(self.documents)
            
            # Classification distribution
            classification_counts = {}
            for doc in self.documents.values():
                classification = doc.classification.value
                classification_counts[classification] = classification_counts.get(classification, 0) + 1
            
            # Sensitivity distribution
            sensitivity_counts = {}
            for doc in self.documents.values():
                sensitivity = doc.sensitivity_level.value
                sensitivity_counts[sensitivity] = sensitivity_counts.get(sensitivity, 0) + 1
            
            # Scan statistics
            total_scans = len(self.scanner.scan_history)
            recent_scans = [s for s in self.scanner.scan_history 
                           if (datetime.utcnow() - s.scan_timestamp).total_seconds() < 3600]
            
            # Security findings
            total_findings = sum(len(scan.findings) for scan in self.scanner.scan_history)
            critical_scans = sum(1 for scan in self.scanner.scan_history if scan.critical_findings_count > 0)
            
            return {
                'document_stats': {
                    'total_documents': total_documents,
                    'classification_distribution': classification_counts,
                    'sensitivity_distribution': sensitivity_counts
                },
                'scan_stats': {
                    'total_scans': total_scans,
                    'recent_scans_1h': len(recent_scans),
                    'total_findings': total_findings,
                    'critical_scans': critical_scans
                },
                'security_metrics': {
                    'critical_scan_rate_pct': (critical_scans / max(total_scans, 1)) * 100,
                    'average_findings_per_scan': total_findings / max(total_scans, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating classification statistics: {e}")
            return {'error': str(e)}
    
    def _initialize_access_policies(self):
        """Initialize default access control policies"""
        self.access_policies = {
            DocumentClassification.PUBLIC: {'everyone'},
            DocumentClassification.INTERNAL: {'employees', 'contractors'},
            DocumentClassification.CONFIDENTIAL: {'managers', 'leads', 'authorized_personnel'},
            DocumentClassification.RESTRICTED: {'executives', 'security_officers'},
            DocumentClassification.TOP_SECRET: {'c_level', 'security_admin'}
        }
    
    def _initialize_redaction_rules(self):
        """Initialize content redaction rules"""
        self.redaction_rules = {
            ContentType.PERSONAL_INFO.value: '[PERSONAL_INFO_REDACTED]',
            ContentType.CREDENTIALS.value: '[CREDENTIALS_REDACTED]',
            ContentType.FINANCIAL.value: '[FINANCIAL_INFO_REDACTED]',
            ContentType.MEDICAL.value: '[MEDICAL_INFO_REDACTED]',
            ContentType.DATA.value: '[DATA_REDACTED]',
            ContentType.CONFIGURATION.value: '[CONFIG_REDACTED]'
        }


# Global document classification manager
document_classification_security = DocumentClassificationManager()


# Convenience functions
def classify_document(content: str, filename: str, author: str = None) -> DocumentMetadata:
    """Convenience function to classify document"""
    return document_classification_security.register_document(content, filename, author)


def scan_document_content(content: str) -> ScanReport:
    """Convenience function to scan document content"""
    return document_classification_security.scanner.scan_document(content)


def redact_document_content(content: str, 
                          redaction_level: SensitivityLevel = SensitivityLevel.HIGH) -> Dict[str, Any]:
    """Convenience function to redact document content"""
    return document_classification_security.redact_sensitive_content(content, redaction_level)