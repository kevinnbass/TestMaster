"""
AutoGen Derived License Compliance Framework
Extracted from AutoGen LICENSE-CODE patterns and Microsoft compliance standards
Enhanced for comprehensive license and legal compliance validation
"""

import os
import re
import json
import logging
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from .error_handler import SecurityError, ValidationError, security_error_handler


class LicenseType(Enum):
    """Software license types"""
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    GPL_V2 = "GPL-2.0"
    GPL_V3 = "GPL-3.0"
    BSD_2_CLAUSE = "BSD-2-Clause"
    BSD_3_CLAUSE = "BSD-3-Clause"
    ISC = "ISC"
    LGPL = "LGPL"
    MPL_2_0 = "MPL-2.0"
    UNLICENSED = "Unlicensed"
    PROPRIETARY = "Proprietary"
    UNKNOWN = "Unknown"


class ComplianceStatus(Enum):
    """Compliance validation status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"
    REVIEW_REQUIRED = "review_required"


@dataclass
class LicenseInfo:
    """License information structure"""
    license_type: LicenseType
    license_text: Optional[str] = None
    copyright_holder: Optional[str] = None
    copyright_year: Optional[str] = None
    source_file: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class ComplianceRule:
    """Compliance validation rule"""
    name: str
    description: str
    license_types: List[LicenseType]
    required_patterns: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    severity: str = "error"  # error, warning, info


@dataclass
class ComplianceReport:
    """License compliance report"""
    project_name: str
    scan_timestamp: datetime
    total_files_scanned: int
    licenses_detected: Dict[LicenseType, int]
    compliance_status: ComplianceStatus
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'project_name': self.project_name,
            'scan_timestamp': self.scan_timestamp.isoformat(),
            'total_files_scanned': self.total_files_scanned,
            'licenses_detected': {k.value: v for k, v in self.licenses_detected.items()},
            'compliance_status': self.compliance_status.value,
            'violations': self.violations,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }


class LicenseDetector:
    """License detection and analysis engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # License detection patterns based on common license texts
        self.license_patterns = {
            LicenseType.MIT: [
                r'MIT License',
                r'Permission is hereby granted, free of charge',
                r'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND'
            ],
            LicenseType.APACHE_2_0: [
                r'Apache License.*Version 2\.0',
                r'Licensed under the Apache License, Version 2\.0',
                r'http://www\.apache\.org/licenses/LICENSE-2\.0'
            ],
            LicenseType.GPL_V2: [
                r'GNU GENERAL PUBLIC LICENSE.*Version 2',
                r'This program is free software.*GPL.*version 2',
                r'http://www\.gnu\.org/licenses/gpl-2\.0'
            ],
            LicenseType.GPL_V3: [
                r'GNU GENERAL PUBLIC LICENSE.*Version 3',
                r'This program is free software.*GPL.*version 3',
                r'http://www\.gnu\.org/licenses/gpl-3\.0'
            ],
            LicenseType.BSD_2_CLAUSE: [
                r'BSD 2-Clause',
                r'Redistribution and use in source and binary forms.*2 conditions',
                r'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS'
            ],
            LicenseType.BSD_3_CLAUSE: [
                r'BSD 3-Clause',
                r'Redistribution and use in source and binary forms.*3 conditions',
                r'Neither the name of.*may be used to endorse'
            ]
        }
        
        # Copyright patterns
        self.copyright_patterns = [
            r'Copyright \(c\) (\d{4}(?:-\d{4})?)\s+(.+)',
            r'Copyright (\d{4}(?:-\d{4})?)\s+(.+)',
            r'© (\d{4}(?:-\d{4})?)\s+(.+)',
        ]
    
    def detect_license(self, text: str, source_file: str = None) -> LicenseInfo:
        """Detect license type from text"""
        try:
            best_match = None
            best_confidence = 0.0
            
            # Normalize text for pattern matching
            normalized_text = re.sub(r'\s+', ' ', text)
            
            for license_type, patterns in self.license_patterns.items():
                confidence = 0.0
                matches = 0
                
                for pattern in patterns:
                    if re.search(pattern, normalized_text, re.IGNORECASE):
                        matches += 1
                
                if matches > 0:
                    confidence = (matches / len(patterns)) * 100
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = license_type
            
            # Extract copyright information
            copyright_holder = None
            copyright_year = None
            
            for pattern in self.copyright_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    copyright_year = match.group(1)
                    copyright_holder = match.group(2).strip()
                    break
            
            # Determine final license type
            if best_match and best_confidence >= 50.0:
                license_type = best_match
            else:
                license_type = LicenseType.UNKNOWN
                if best_confidence < 50.0 and best_confidence > 0:
                    license_type = LicenseType.UNKNOWN
            
            return LicenseInfo(
                license_type=license_type,
                license_text=text if len(text) < 10000 else text[:10000],
                copyright_holder=copyright_holder,
                copyright_year=copyright_year,
                source_file=source_file,
                confidence_score=best_confidence
            )
            
        except Exception as e:
            self.logger.error(f"License detection error: {e}")
            return LicenseInfo(
                license_type=LicenseType.UNKNOWN,
                source_file=source_file,
                validation_errors=[f"Detection error: {str(e)}"]
            )
    
    def scan_file(self, file_path: str) -> Optional[LicenseInfo]:
        """Scan a single file for license information"""
        try:
            if not os.path.exists(file_path):
                return None
            
            # Skip binary files and very large files
            if os.path.getsize(file_path) > 1024 * 1024:  # 1MB limit
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 50 lines or 10KB, whichever is smaller
                content_lines = []
                bytes_read = 0
                for line in f:
                    if len(content_lines) >= 50 or bytes_read >= 10240:
                        break
                    content_lines.append(line)
                    bytes_read += len(line.encode('utf-8'))
                
                content = ''.join(content_lines)
            
            # Only process if content contains license-related keywords
            license_keywords = ['license', 'copyright', 'permission', 'warranty', 'distribute']
            content_lower = content.lower()
            
            if any(keyword in content_lower for keyword in license_keywords):
                return self.detect_license(content, file_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
            return LicenseInfo(
                license_type=LicenseType.UNKNOWN,
                source_file=file_path,
                validation_errors=[f"Scan error: {str(e)}"]
            )


class ComplianceValidator:
    """License compliance validation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default compliance rules based on common enterprise policies
        self.compliance_rules = {
            'microsoft_compatible': ComplianceRule(
                name='microsoft_compatible',
                description='Microsoft compatible licenses',
                license_types=[LicenseType.MIT, LicenseType.APACHE_2_0, LicenseType.BSD_2_CLAUSE, LicenseType.BSD_3_CLAUSE],
                forbidden_patterns=[r'GPL', r'LGPL', r'AGPL'],
                severity='error'
            ),
            'copyleft_warning': ComplianceRule(
                name='copyleft_warning',
                description='Copyleft license warning',
                license_types=[LicenseType.GPL_V2, LicenseType.GPL_V3, LicenseType.LGPL],
                required_patterns=[r'source.*available', r'distribute.*source'],
                severity='warning'
            ),
            'copyright_required': ComplianceRule(
                name='copyright_required',
                description='Copyright notice required',
                license_types=list(LicenseType),
                required_patterns=[r'Copyright.*\d{4}', r'©.*\d{4}'],
                severity='error'
            )
        }
    
    def validate_license_info(self, license_info: LicenseInfo, 
                             rules: List[str] = None) -> Tuple[ComplianceStatus, List[Dict[str, Any]]]:
        """Validate license information against compliance rules"""
        try:
            issues = []
            worst_status = ComplianceStatus.COMPLIANT
            
            rules_to_check = rules or list(self.compliance_rules.keys())
            
            for rule_name in rules_to_check:
                if rule_name not in self.compliance_rules:
                    continue
                
                rule = self.compliance_rules[rule_name]
                rule_issues = self._check_rule(license_info, rule)
                
                for issue in rule_issues:
                    issues.append({
                        'rule': rule_name,
                        'severity': rule.severity,
                        'description': issue['description'],
                        'file': license_info.source_file,
                        'license_type': license_info.license_type.value
                    })
                    
                    # Update worst status
                    if rule.severity == 'error':
                        worst_status = ComplianceStatus.NON_COMPLIANT
                    elif rule.severity == 'warning' and worst_status == ComplianceStatus.COMPLIANT:
                        worst_status = ComplianceStatus.WARNING
            
            return worst_status, issues
            
        except Exception as e:
            self.logger.error(f"License validation error: {e}")
            return ComplianceStatus.UNKNOWN, [{
                'rule': 'validation_error',
                'severity': 'error',
                'description': f"Validation error: {str(e)}",
                'file': license_info.source_file
            }]
    
    def _check_rule(self, license_info: LicenseInfo, rule: ComplianceRule) -> List[Dict[str, Any]]:
        """Check a specific compliance rule"""
        issues = []
        
        try:
            # Check if license type is relevant for this rule
            if rule.license_types and license_info.license_type not in rule.license_types:
                return issues
            
            license_text = license_info.license_text or ""
            
            # Check required patterns
            for pattern in rule.required_patterns:
                if not re.search(pattern, license_text, re.IGNORECASE):
                    issues.append({
                        'description': f"Missing required pattern: {pattern}",
                        'pattern': pattern,
                        'type': 'missing_requirement'
                    })
            
            # Check forbidden patterns
            for pattern in rule.forbidden_patterns:
                if re.search(pattern, license_text, re.IGNORECASE):
                    issues.append({
                        'description': f"Contains forbidden pattern: {pattern}",
                        'pattern': pattern,
                        'type': 'forbidden_content'
                    })
            
            return issues
            
        except Exception as e:
            return [{
                'description': f"Rule check error: {str(e)}",
                'type': 'check_error'
            }]


class LicenseComplianceManager:
    """Central license compliance management system"""
    
    def __init__(self):
        self.detector = LicenseDetector()
        self.validator = ComplianceValidator()
        self.scan_history: List[ComplianceReport] = []
        self.logger = logging.getLogger(__name__)
    
    def scan_project(self, project_path: str, project_name: str = None,
                    file_patterns: List[str] = None) -> ComplianceReport:
        """Scan entire project for license compliance"""
        try:
            if not project_name:
                project_name = os.path.basename(os.path.abspath(project_path))
            
            # Default file patterns to scan
            if not file_patterns:
                file_patterns = [
                    'LICENSE*', 'COPYING*', 'COPYRIGHT*',
                    '*.py', '*.js', '*.java', '*.cs', '*.cpp', '*.h',
                    'README*', '*.md', '*.txt'
                ]
            
            report = ComplianceReport(
                project_name=project_name,
                scan_timestamp=datetime.utcnow(),
                total_files_scanned=0,
                licenses_detected={},
                compliance_status=ComplianceStatus.COMPLIANT
            )
            
            # Scan files
            scanned_licenses = []
            
            for root, dirs, files in os.walk(project_path):
                # Skip common ignored directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check if file matches patterns
                    if self._matches_patterns(file, file_patterns):
                        license_info = self.detector.scan_file(file_path)
                        if license_info:
                            scanned_licenses.append(license_info)
                            report.total_files_scanned += 1
            
            # Analyze detected licenses
            for license_info in scanned_licenses:
                license_type = license_info.license_type
                report.licenses_detected[license_type] = report.licenses_detected.get(license_type, 0) + 1
                
                # Validate compliance
                status, issues = self.validator.validate_license_info(license_info)
                
                for issue in issues:
                    if issue['severity'] == 'error':
                        report.violations.append(issue)
                        if report.compliance_status == ComplianceStatus.COMPLIANT:
                            report.compliance_status = ComplianceStatus.NON_COMPLIANT
                    elif issue['severity'] == 'warning':
                        report.warnings.append(issue)
                        if report.compliance_status == ComplianceStatus.COMPLIANT:
                            report.compliance_status = ComplianceStatus.WARNING
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            # Store in history
            self.scan_history.append(report)
            if len(self.scan_history) > 100:  # Keep last 100 reports
                self.scan_history = self.scan_history[-100:]
            
            self.logger.info(f"License compliance scan completed: {project_name} - Status: {report.compliance_status.value}")
            return report
            
        except Exception as e:
            error = SecurityError(f"License compliance scan failed: {str(e)}", "LICENSE_SCAN_001")
            security_error_handler.handle_error(error)
            
            # Return error report
            return ComplianceReport(
                project_name=project_name or "unknown",
                scan_timestamp=datetime.utcnow(),
                total_files_scanned=0,
                licenses_detected={},
                compliance_status=ComplianceStatus.UNKNOWN,
                violations=[{
                    'rule': 'scan_error',
                    'severity': 'error',
                    'description': str(e),
                    'file': project_path
                }]
            )
    
    def _matches_patterns(self, filename: str, patterns: List[str]) -> bool:
        """Check if filename matches any of the given patterns"""
        import fnmatch
        for pattern in patterns:
            if fnmatch.fnmatch(filename.upper(), pattern.upper()):
                return True
        return False
    
    def _generate_recommendations(self, report: ComplianceReport) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Check for unknown licenses
        if LicenseType.UNKNOWN in report.licenses_detected:
            recommendations.append("Review files with unknown licenses and add proper license headers")
        
        # Check for GPL licenses
        gpl_licenses = [LicenseType.GPL_V2, LicenseType.GPL_V3, LicenseType.LGPL]
        if any(license_type in report.licenses_detected for license_type in gpl_licenses):
            recommendations.append("Review GPL/LGPL licenses for copyleft compliance requirements")
        
        # Check for missing copyright
        if len(report.violations) > 0:
            copyright_violations = [v for v in report.violations if 'copyright' in v['description'].lower()]
            if copyright_violations:
                recommendations.append("Add copyright notices to all source files")
        
        # General recommendations
        if report.compliance_status != ComplianceStatus.COMPLIANT:
            recommendations.append("Review all compliance violations and address before release")
            recommendations.append("Consider implementing automated license scanning in CI/CD pipeline")
        
        return recommendations
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of all compliance scans"""
        if not self.scan_history:
            return {'message': 'No scans performed yet'}
        
        recent_scans = self.scan_history[-10:]  # Last 10 scans
        
        # Count statuses
        status_counts = {}
        for report in recent_scans:
            status = report.compliance_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Most common licenses
        all_licenses = {}
        for report in recent_scans:
            for license_type, count in report.licenses_detected.items():
                all_licenses[license_type.value] = all_licenses.get(license_type.value, 0) + count
        
        return {
            'total_scans': len(self.scan_history),
            'recent_scans': len(recent_scans),
            'status_distribution': status_counts,
            'common_licenses': dict(sorted(all_licenses.items(), key=lambda x: x[1], reverse=True)),
            'last_scan': recent_scans[-1].scan_timestamp.isoformat() if recent_scans else None
        }


# Global compliance manager
license_compliance_manager = LicenseComplianceManager()


def scan_project_compliance(project_path: str, project_name: str = None) -> ComplianceReport:
    """Convenience function to scan project for license compliance"""
    return license_compliance_manager.scan_project(project_path, project_name)