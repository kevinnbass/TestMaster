"""
Advanced API Security Testing Framework for TestMaster
Comprehensive API security validation with OWASP API Top 10 coverage
"""

import asyncio
import json
import time
import hashlib
import hmac
import base64
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import requests
from urllib.parse import urlparse, parse_qs, urlencode, quote

class APISecurityCategory(Enum):
    """OWASP API Security Top 10 categories"""
    BROKEN_OBJECT_LEVEL_AUTH = "API1_2023"
    BROKEN_AUTHENTICATION = "API2_2023"
    BROKEN_OBJECT_PROPERTY_LEVEL_AUTH = "API3_2023"
    UNRESTRICTED_RESOURCE_CONSUMPTION = "API4_2023"
    BROKEN_FUNCTION_LEVEL_AUTH = "API5_2023"
    UNRESTRICTED_ACCESS_SENSITIVE_BUSINESS_FLOWS = "API6_2023"
    SERVER_SIDE_REQUEST_FORGERY = "API7_2023"
    SECURITY_MISCONFIGURATION = "API8_2023"
    IMPROPER_INVENTORY_MANAGEMENT = "API9_2023"
    UNSAFE_CONSUMPTION_APIS = "API10_2023"

class VulnerabilityLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class TestType(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    RATE_LIMITING = "rate_limiting"
    DATA_EXPOSURE = "data_exposure"
    BUSINESS_LOGIC = "business_logic"
    INJECTION = "injection"
    CONFIGURATION = "configuration"

@dataclass
class APIEndpoint:
    """API endpoint specification"""
    url: str
    method: str
    headers: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    auth_required: bool = True
    rate_limit: Optional[int] = None

@dataclass
class SecurityPayload:
    """Security test payload"""
    name: str
    payload: Any
    test_type: TestType
    category: APISecurityCategory
    expected_behavior: str
    detection_pattern: Optional[str] = None

@dataclass
class SecurityTest:
    """Individual security test"""
    test_id: str
    endpoint: APIEndpoint
    payload: SecurityPayload
    timestamp: float
    duration: float = 0.0
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    headers_received: Dict[str, str] = field(default_factory=dict)
    vulnerability_detected: bool = False
    severity: VulnerabilityLevel = VulnerabilityLevel.INFO

@dataclass
class SecurityTestResult:
    """Comprehensive security test result"""
    test_id: str
    endpoint_url: str
    category: APISecurityCategory
    test_type: TestType
    vulnerability_found: bool
    severity: VulnerabilityLevel
    description: str
    evidence: List[str]
    remediation: str
    confidence_score: float
    timestamp: float

@dataclass
class APISecurityReport:
    """Complete API security assessment report"""
    target_base_url: str
    total_tests: int
    vulnerabilities_found: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    test_results: List[SecurityTestResult]
    coverage_metrics: Dict[str, float]
    recommendations: List[str]
    scan_duration: float
    timestamp: float

class APISecurityTester:
    """Advanced API Security Testing Framework"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.endpoints: List[APIEndpoint] = []
        self.test_results: List[SecurityTest] = []
        self.payloads = self._initialize_security_payloads()
        
    def _initialize_security_payloads(self) -> Dict[APISecurityCategory, List[SecurityPayload]]:
        """Initialize comprehensive API security payload database"""
        return {
            APISecurityCategory.BROKEN_OBJECT_LEVEL_AUTH: [
                SecurityPayload(
                    "IDOR_USER_ID", {"user_id": "../../../etc/passwd"},
                    TestType.AUTHORIZATION, APISecurityCategory.BROKEN_OBJECT_LEVEL_AUTH,
                    "Should return 403/404", r"root:.*:0:0"
                ),
                SecurityPayload(
                    "IDOR_NUMERIC_ID", {"id": 99999},
                    TestType.AUTHORIZATION, APISecurityCategory.BROKEN_OBJECT_LEVEL_AUTH,
                    "Should validate object ownership", None
                ),
                SecurityPayload(
                    "UUID_ENUMERATION", {"uuid": "00000000-0000-0000-0000-000000000001"},
                    TestType.AUTHORIZATION, APISecurityCategory.BROKEN_OBJECT_LEVEL_AUTH,
                    "Should validate UUID ownership", None
                )
            ],
            
            APISecurityCategory.BROKEN_AUTHENTICATION: [
                SecurityPayload(
                    "JWT_NONE_ALG", {"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJ1c2VyIjoiYWRtaW4ifQ."},
                    TestType.AUTHENTICATION, APISecurityCategory.BROKEN_AUTHENTICATION,
                    "Should reject none algorithm", r"admin"
                ),
                SecurityPayload(
                    "WEAK_JWT_SECRET", {"Authorization": "Bearer " + self._generate_weak_jwt()},
                    TestType.AUTHENTICATION, APISecurityCategory.BROKEN_AUTHENTICATION,
                    "Should use strong JWT secrets", None
                ),
                SecurityPayload(
                    "EXPIRED_TOKEN", {"Authorization": "Bearer expired_token_here"},
                    TestType.AUTHENTICATION, APISecurityCategory.BROKEN_AUTHENTICATION,
                    "Should reject expired tokens", None
                )
            ],
            
            APISecurityCategory.BROKEN_OBJECT_PROPERTY_LEVEL_AUTH: [
                SecurityPayload(
                    "MASS_ASSIGNMENT", {"is_admin": True, "role": "admin", "permissions": ["*"]},
                    TestType.INPUT_VALIDATION, APISecurityCategory.BROKEN_OBJECT_PROPERTY_LEVEL_AUTH,
                    "Should filter input properties", None
                ),
                SecurityPayload(
                    "EXCESSIVE_DATA_EXPOSURE", {},
                    TestType.DATA_EXPOSURE, APISecurityCategory.BROKEN_OBJECT_PROPERTY_LEVEL_AUTH,
                    "Should limit response data", r"(password|secret|key|token)"
                )
            ],
            
            APISecurityCategory.UNRESTRICTED_RESOURCE_CONSUMPTION: [
                SecurityPayload(
                    "LARGE_PAYLOAD", {"data": "A" * 10000000},
                    TestType.INPUT_VALIDATION, APISecurityCategory.UNRESTRICTED_RESOURCE_CONSUMPTION,
                    "Should limit payload size", None
                ),
                SecurityPayload(
                    "DEEP_NESTING", self._create_deep_nested_object(100),
                    TestType.INPUT_VALIDATION, APISecurityCategory.UNRESTRICTED_RESOURCE_CONSUMPTION,
                    "Should limit nesting depth", None
                )
            ],
            
            APISecurityCategory.BROKEN_FUNCTION_LEVEL_AUTH: [
                SecurityPayload(
                    "PRIVILEGE_ESCALATION", {},
                    TestType.AUTHORIZATION, APISecurityCategory.BROKEN_FUNCTION_LEVEL_AUTH,
                    "Should enforce function-level authorization", None
                ),
                SecurityPayload(
                    "HTTP_VERB_TAMPERING", {},
                    TestType.AUTHORIZATION, APISecurityCategory.BROKEN_FUNCTION_LEVEL_AUTH,
                    "Should validate HTTP methods", None
                )
            ],
            
            APISecurityCategory.SERVER_SIDE_REQUEST_FORGERY: [
                SecurityPayload(
                    "INTERNAL_IP_ACCESS", {"url": "http://127.0.0.1:22"},
                    TestType.INJECTION, APISecurityCategory.SERVER_SIDE_REQUEST_FORGERY,
                    "Should block internal IP access", r"SSH"
                ),
                SecurityPayload(
                    "CLOUD_METADATA", {"url": "http://169.254.169.254/latest/meta-data/"},
                    TestType.INJECTION, APISecurityCategory.SERVER_SIDE_REQUEST_FORGERY,
                    "Should block cloud metadata access", r"(ami-|instance-id)"
                )
            ]
        }
    
    def _generate_weak_jwt(self) -> str:
        """Generate JWT with weak secret for testing"""
        header = base64.urlsafe_b64encode(b'{"typ":"JWT","alg":"HS256"}').decode().rstrip('=')
        payload = base64.urlsafe_b64encode(b'{"user":"admin","exp":9999999999}').decode().rstrip('=')
        weak_secret = os.getenv('SECRET')
        signature = hmac.new(weak_secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256)
        sig_b64 = base64.urlsafe_b64encode(signature.digest()).decode().rstrip('=')
        return f"{header}.{payload}.{sig_b64}"
    
    def _create_deep_nested_object(self, depth: int) -> Dict[str, Any]:
        """Create deeply nested object for testing"""
        if depth <= 0:
            return {"value": "deep"}
        return {"nested": self._create_deep_nested_object(depth - 1)}
    
    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        """Add endpoint for security testing"""
        self.endpoints.append(endpoint)
    
    def discover_endpoints(self, openapi_spec: Optional[Dict[str, Any]] = None) -> List[APIEndpoint]:
        """Discover API endpoints from OpenAPI spec or common paths"""
        discovered = []
        
        if openapi_spec:
            discovered.extend(self._parse_openapi_endpoints(openapi_spec))
        else:
            # Common API endpoint patterns
            common_paths = [
                "/api/v1/users", "/api/v1/login", "/api/v1/admin",
                "/api/users", "/users", "/login", "/admin",
                "/api/v1/data", "/api/v1/files", "/api/v1/config"
            ]
            
            for path in common_paths:
                for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    endpoint = APIEndpoint(url=f"{self.base_url}{path}", method=method)
                    if self._endpoint_exists(endpoint):
                        discovered.append(endpoint)
        
        return discovered
    
    def _parse_openapi_endpoints(self, spec: Dict[str, Any]) -> List[APIEndpoint]:
        """Parse endpoints from OpenAPI specification"""
        endpoints = []
        paths = spec.get("paths", {})
        
        for path, path_spec in paths.items():
            for method, method_spec in path_spec.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    endpoint = APIEndpoint(
                        url=f"{self.base_url}{path}",
                        method=method.upper(),
                        auth_required=self._requires_auth(method_spec)
                    )
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _endpoint_exists(self, endpoint: APIEndpoint) -> bool:
        """Check if endpoint exists and is accessible"""
        try:
            response = self.session.request(
                endpoint.method, endpoint.url,
                headers=endpoint.headers, timeout=5
            )
            return response.status_code != 404
        except:
            return False
    
    def _requires_auth(self, method_spec: Dict[str, Any]) -> bool:
        """Determine if endpoint requires authentication"""
        security = method_spec.get("security", [])
        return len(security) > 0
    
    async def execute_security_tests(self, endpoints: Optional[List[APIEndpoint]] = None) -> APISecurityReport:
        """Execute comprehensive security tests"""
        start_time = time.time()
        test_endpoints = endpoints or self.endpoints
        
        if not test_endpoints:
            test_endpoints = self.discover_endpoints()
        
        all_results = []
        
        for endpoint in test_endpoints:
            for category, payloads in self.payloads.items():
                for payload in payloads:
                    result = await self._execute_single_test(endpoint, payload)
                    all_results.append(result)
        
        # Analyze results and generate report
        return self._generate_security_report(all_results, time.time() - start_time)
    
    async def _execute_single_test(self, endpoint: APIEndpoint, payload: SecurityPayload) -> SecurityTestResult:
        """Execute single security test"""
        test_id = f"{endpoint.method}_{hash(endpoint.url)}_{payload.name}"
        start_time = time.time()
        
        try:
            # Prepare test request
            test_endpoint = self._prepare_test_endpoint(endpoint, payload)
            
            # Execute request
            response = self.session.request(
                test_endpoint.method,
                test_endpoint.url,
                headers=test_endpoint.headers,
                params=test_endpoint.parameters,
                json=test_endpoint.body,
                timeout=self.timeout
            )
            
            # Analyze response for vulnerabilities
            vulnerability_found, severity, evidence = self._analyze_response(
                response, payload
            )
            
            return SecurityTestResult(
                test_id=test_id,
                endpoint_url=endpoint.url,
                category=payload.category,
                test_type=payload.test_type,
                vulnerability_found=vulnerability_found,
                severity=severity,
                description=self._get_vulnerability_description(payload.category, vulnerability_found),
                evidence=evidence,
                remediation=self._get_remediation_advice(payload.category),
                confidence_score=self._calculate_confidence_score(response, payload),
                timestamp=start_time
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_id=test_id,
                endpoint_url=endpoint.url,
                category=payload.category,
                test_type=payload.test_type,
                vulnerability_found=False,
                severity=VulnerabilityLevel.INFO,
                description=f"Test failed: {str(e)}",
                evidence=[],
                remediation="",
                confidence_score=0.0,
                timestamp=start_time
            )
    
    def _prepare_test_endpoint(self, endpoint: APIEndpoint, payload: SecurityPayload) -> APIEndpoint:
        """Prepare endpoint with security payload"""
        test_endpoint = APIEndpoint(
            url=endpoint.url,
            method=endpoint.method,
            headers=endpoint.headers.copy(),
            parameters=endpoint.parameters.copy(),
            body=endpoint.body.copy() if endpoint.body else None
        )
        
        # Apply payload based on type
        if payload.test_type == TestType.AUTHENTICATION:
            if isinstance(payload.payload, dict) and "Authorization" in payload.payload:
                test_endpoint.headers.update(payload.payload)
        elif payload.test_type in [TestType.INPUT_VALIDATION, TestType.DATA_EXPOSURE]:
            if test_endpoint.method in ["POST", "PUT", "PATCH"]:
                test_endpoint.body = payload.payload
            else:
                test_endpoint.parameters.update(payload.payload)
        elif payload.test_type == TestType.AUTHORIZATION:
            test_endpoint.parameters.update(payload.payload)
        
        return test_endpoint
    
    def _analyze_response(self, response: requests.Response, payload: SecurityPayload) -> Tuple[bool, VulnerabilityLevel, List[str]]:
        """Analyze response for security vulnerabilities"""
        evidence = []
        vulnerability_found = False
        severity = VulnerabilityLevel.INFO
        
        # Check for detection patterns
        if payload.detection_pattern:
            pattern_match = re.search(payload.detection_pattern, response.text, re.IGNORECASE)
            if pattern_match:
                vulnerability_found = True
                evidence.append(f"Pattern matched: {pattern_match.group()}")
                severity = VulnerabilityLevel.HIGH
        
        # Analyze response codes
        if payload.category == APISecurityCategory.BROKEN_OBJECT_LEVEL_AUTH:
            if response.status_code == 200 and "id" in str(payload.payload):
                vulnerability_found = True
                evidence.append(f"Potential IDOR: {response.status_code}")
                severity = VulnerabilityLevel.HIGH
        
        # Check for sensitive data exposure
        sensitive_patterns = [
            r'"password"\s*:\s*"[^"]*"',
            r'"secret"\s*:\s*"[^"]*"',
            r'"token"\s*:\s*"[^"]*"',
            r'"key"\s*:\s*"[^"]*"'
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, response.text, re.IGNORECASE):
                vulnerability_found = True
                evidence.append("Sensitive data exposed in response")
                severity = VulnerabilityLevel.MEDIUM
                break
        
        return vulnerability_found, severity, evidence
    
    def _calculate_confidence_score(self, response: requests.Response, payload: SecurityPayload) -> float:
        """Calculate confidence score for vulnerability detection"""
        score = 0.5  # Base score
        
        # Increase confidence based on response analysis
        if payload.detection_pattern and re.search(payload.detection_pattern, response.text):
            score += 0.4
        
        if response.status_code in [200, 500, 403]:
            score += 0.1
        
        return min(1.0, score)
    
    def _get_vulnerability_description(self, category: APISecurityCategory, found: bool) -> str:
        """Get vulnerability description"""
        descriptions = {
            APISecurityCategory.BROKEN_OBJECT_LEVEL_AUTH: "Insecure Direct Object Reference vulnerability",
            APISecurityCategory.BROKEN_AUTHENTICATION: "Authentication bypass vulnerability",
            APISecurityCategory.BROKEN_OBJECT_PROPERTY_LEVEL_AUTH: "Mass assignment or data exposure vulnerability",
            APISecurityCategory.UNRESTRICTED_RESOURCE_CONSUMPTION: "Resource exhaustion vulnerability",
            APISecurityCategory.BROKEN_FUNCTION_LEVEL_AUTH: "Authorization bypass vulnerability",
            APISecurityCategory.SERVER_SIDE_REQUEST_FORGERY: "Server-Side Request Forgery vulnerability"
        }
        
        if found:
            return descriptions.get(category, "Security vulnerability detected")
        else:
            return f"No {category.value} vulnerability detected"
    
    def _get_remediation_advice(self, category: APISecurityCategory) -> str:
        """Get remediation advice for vulnerability category"""
        advice = {
            APISecurityCategory.BROKEN_OBJECT_LEVEL_AUTH: "Implement proper object-level authorization checks",
            APISecurityCategory.BROKEN_AUTHENTICATION: "Use strong authentication mechanisms and validate tokens",
            APISecurityCategory.BROKEN_OBJECT_PROPERTY_LEVEL_AUTH: "Implement input filtering and output filtering",
            APISecurityCategory.UNRESTRICTED_RESOURCE_CONSUMPTION: "Implement rate limiting and input validation",
            APISecurityCategory.BROKEN_FUNCTION_LEVEL_AUTH: "Implement proper function-level authorization",
            APISecurityCategory.SERVER_SIDE_REQUEST_FORGERY: "Validate and sanitize all external URLs"
        }
        
        return advice.get(category, "Follow OWASP API security guidelines")
    
    def _generate_security_report(self, results: List[SecurityTestResult], duration: float) -> APISecurityReport:
        """Generate comprehensive security assessment report"""
        vulnerabilities = [r for r in results if r.vulnerability_found]
        
        critical_count = len([v for v in vulnerabilities if v.severity == VulnerabilityLevel.CRITICAL])
        high_count = len([v for v in vulnerabilities if v.severity == VulnerabilityLevel.HIGH])
        medium_count = len([v for v in vulnerabilities if v.severity == VulnerabilityLevel.MEDIUM])
        low_count = len([v for v in vulnerabilities if v.severity == VulnerabilityLevel.LOW])
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(vulnerabilities)
        
        return APISecurityReport(
            target_base_url=self.base_url,
            total_tests=len(results),
            vulnerabilities_found=len(vulnerabilities),
            critical_issues=critical_count,
            high_issues=high_count,
            medium_issues=medium_count,
            low_issues=low_count,
            test_results=results,
            coverage_metrics=coverage_metrics,
            recommendations=recommendations,
            scan_duration=duration,
            timestamp=time.time()
        )
    
    def _calculate_coverage_metrics(self, results: List[SecurityTestResult]) -> Dict[str, float]:
        """Calculate security test coverage metrics"""
        total_categories = len(APISecurityCategory)
        tested_categories = len(set(r.category for r in results))
        
        total_test_types = len(TestType)
        tested_types = len(set(r.test_type for r in results))
        
        return {
            "category_coverage": (tested_categories / total_categories) * 100,
            "test_type_coverage": (tested_types / total_test_types) * 100,
            "endpoint_coverage": len(set(r.endpoint_url for r in results)),
            "vulnerability_detection_rate": (len([r for r in results if r.vulnerability_found]) / len(results)) * 100 if results else 0
        }
    
    def _generate_recommendations(self, vulnerabilities: List[SecurityTestResult]) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        categories_found = set(v.category for v in vulnerabilities)
        
        if APISecurityCategory.BROKEN_OBJECT_LEVEL_AUTH in categories_found:
            recommendations.append("Implement object-level authorization checks for all API endpoints")
        
        if APISecurityCategory.BROKEN_AUTHENTICATION in categories_found:
            recommendations.append("Strengthen authentication mechanisms and token validation")
        
        if APISecurityCategory.BROKEN_OBJECT_PROPERTY_LEVEL_AUTH in categories_found:
            recommendations.append("Implement input/output filtering and prevent mass assignment")
        
        if len(vulnerabilities) > 0:
            recommendations.append("Conduct regular security assessments and penetration testing")
            recommendations.append("Implement API security monitoring and logging")
        
        return recommendations