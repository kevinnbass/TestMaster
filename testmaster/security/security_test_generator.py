"""
Security-Aware Test Generator

Generates security-focused tests based on threat models and vulnerabilities.
Adapted from Agency Swarm's security testing patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
import json
from datetime import datetime

from .universal_scanner import VulnerabilityFinding, VulnerabilityType, SeverityLevel
from .compliance_framework import ComplianceStandard, ComplianceReport
from ..core.framework_abstraction import (
    UniversalTest, UniversalTestCase, UniversalTestSuite,
    TestAssertion, AssertionType, TestMetadata
)
from ..core.ast_abstraction import UniversalAST


class ThreatCategory(Enum):
    """Categories of security threats."""
    INJECTION = "injection"
    BROKEN_AUTHENTICATION = "broken_authentication"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    XML_EXTERNAL_ENTITIES = "xml_external_entities"
    BROKEN_ACCESS_CONTROL = "broken_access_control"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    CROSS_SITE_SCRIPTING = "cross_site_scripting"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    BUSINESS_LOGIC = "business_logic"
    CRYPTOGRAPHIC = "cryptographic"


@dataclass
class ThreatModel:
    """Represents a threat model for the application."""
    name: str
    description: str
    threats: List[ThreatCategory] = field(default_factory=list)
    
    # Risk assessment
    risk_level: SeverityLevel = SeverityLevel.MEDIUM
    likelihood: float = 0.5  # 0.0 to 1.0
    impact: float = 0.5  # 0.0 to 1.0
    
    # Context
    applicable_languages: List[str] = field(default_factory=list)
    applicable_frameworks: List[str] = field(default_factory=list)
    
    # Attack vectors
    attack_vectors: List[str] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    
    # Mitigation
    mitigations: List[str] = field(default_factory=list)
    test_scenarios: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'threats': [t.value for t in self.threats],
            'risk_level': self.risk_level.value,
            'likelihood': self.likelihood,
            'impact': self.impact,
            'applicable_languages': self.applicable_languages,
            'applicable_frameworks': self.applicable_frameworks,
            'attack_vectors': self.attack_vectors,
            'entry_points': self.entry_points,
            'mitigations': self.mitigations,
            'test_scenarios': self.test_scenarios
        }


@dataclass
class SecurityTestConfig:
    """Configuration for security test generation."""
    # Test scope
    include_owasp_top10: bool = True
    include_compliance_tests: bool = True
    include_penetration_tests: bool = False
    
    # Test types
    generate_injection_tests: bool = True
    generate_auth_tests: bool = True
    generate_access_control_tests: bool = True
    generate_crypto_tests: bool = True
    generate_business_logic_tests: bool = True
    
    # Test intensity
    test_intensity: str = "medium"  # low, medium, high, comprehensive
    max_tests_per_vulnerability: int = 5
    include_edge_cases: bool = True
    
    # Compliance standards
    target_compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    
    # Output settings
    include_negative_tests: bool = True
    include_boundary_tests: bool = True
    include_fuzzing_tests: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'include_owasp_top10': self.include_owasp_top10,
            'include_compliance_tests': self.include_compliance_tests,
            'include_penetration_tests': self.include_penetration_tests,
            'generate_injection_tests': self.generate_injection_tests,
            'generate_auth_tests': self.generate_auth_tests,
            'generate_access_control_tests': self.generate_access_control_tests,
            'generate_crypto_tests': self.generate_crypto_tests,
            'generate_business_logic_tests': self.generate_business_logic_tests,
            'test_intensity': self.test_intensity,
            'max_tests_per_vulnerability': self.max_tests_per_vulnerability,
            'include_edge_cases': self.include_edge_cases,
            'target_compliance_standards': [s.value for s in self.target_compliance_standards],
            'include_negative_tests': self.include_negative_tests,
            'include_boundary_tests': self.include_boundary_tests,
            'include_fuzzing_tests': self.include_fuzzing_tests
        }


@dataclass
class SecurityTestSuite:
    """Security-focused test suite."""
    name: str
    universal_test_suite: UniversalTestSuite
    
    # Security context
    threat_models: List[ThreatModel] = field(default_factory=list)
    vulnerabilities_tested: List[VulnerabilityType] = field(default_factory=list)
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    
    # Metrics
    security_coverage: float = 0.0
    threat_coverage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'test_suite': self.universal_test_suite.to_dict(),
            'threat_models': [tm.to_dict() for tm in self.threat_models],
            'vulnerabilities_tested': [vt.value for vt in self.vulnerabilities_tested],
            'compliance_standards': [cs.value for cs in self.compliance_standards],
            'security_coverage': self.security_coverage,
            'threat_coverage': self.threat_coverage
        }


class SecurityTestGenerator:
    """Generates security-focused tests."""
    
    def __init__(self, config: SecurityTestConfig = None):
        self.config = config or SecurityTestConfig()
        
        # Load threat models and attack patterns
        self.threat_models = self._load_threat_models()
        self.attack_patterns = self._load_attack_patterns()
        self.test_templates = self._load_test_templates()
        
        print(f"Security Test Generator initialized")
        print(f"   Threat models: {len(self.threat_models)}")
        print(f"   Test intensity: {self.config.test_intensity}")
    
    def generate_security_tests(self, 
                               universal_ast: UniversalAST,
                               vulnerabilities: List[VulnerabilityFinding] = None,
                               compliance_reports: List[ComplianceReport] = None) -> SecurityTestSuite:
        """Generate comprehensive security test suite."""
        
        print(f"\nGenerating security tests...")
        print(f"   AST functions: {universal_ast.total_functions}")
        print(f"   Vulnerabilities: {len(vulnerabilities) if vulnerabilities else 0}")
        
        # Create base test suite
        test_suite = UniversalTestSuite(
            name="SecurityTestSuite",
            metadata=TestMetadata(
                tags=["security", "automated"],
                category="security",
                description="Comprehensive security test suite"
            )
        )
        
        # Generate different types of security tests
        generated_vulnerabilities = set()
        
        # 1. OWASP Top 10 tests
        if self.config.include_owasp_top10:
            owasp_tests = self._generate_owasp_tests(universal_ast)
            test_suite.test_cases.extend(owasp_tests)
            generated_vulnerabilities.update([VulnerabilityType.SQL_INJECTION, VulnerabilityType.XSS])
        
        # 2. Vulnerability-specific tests
        if vulnerabilities:
            vuln_tests = self._generate_vulnerability_tests(vulnerabilities)
            test_suite.test_cases.extend(vuln_tests)
            generated_vulnerabilities.update([v.type for v in vulnerabilities])
        
        # 3. Authentication and authorization tests
        if self.config.generate_auth_tests:
            auth_tests = self._generate_auth_tests(universal_ast)
            test_suite.test_cases.extend(auth_tests)
            generated_vulnerabilities.update([VulnerabilityType.AUTHENTICATION_BYPASS, VulnerabilityType.AUTHORIZATION_FAILURE])
        
        # 4. Cryptographic tests
        if self.config.generate_crypto_tests:
            crypto_tests = self._generate_crypto_tests(universal_ast)
            test_suite.test_cases.extend(crypto_tests)
            generated_vulnerabilities.add(VulnerabilityType.CRYPTO_FAILURE)
        
        # 5. Business logic tests
        if self.config.generate_business_logic_tests:
            business_tests = self._generate_business_logic_tests(universal_ast)
            test_suite.test_cases.extend(business_tests)
        
        # 6. Compliance tests
        compliance_standards = []
        if self.config.include_compliance_tests and compliance_reports:
            compliance_tests = self._generate_compliance_tests(compliance_reports)
            test_suite.test_cases.extend(compliance_tests)
            compliance_standards = [report.standard for report in compliance_reports]
        
        # Calculate metrics
        test_suite.calculate_metrics()
        
        # Create security test suite
        security_suite = SecurityTestSuite(
            name="SecurityTestSuite",
            universal_test_suite=test_suite,
            threat_models=self._get_applicable_threat_models(universal_ast),
            vulnerabilities_tested=list(generated_vulnerabilities),
            compliance_standards=compliance_standards
        )
        
        # Calculate coverage
        security_suite.security_coverage = self._calculate_security_coverage(generated_vulnerabilities)
        security_suite.threat_coverage = self._calculate_threat_coverage(security_suite.threat_models)
        
        print(f"   Security tests generated:")
        print(f"      Test cases: {len(test_suite.test_cases)}")
        print(f"      Total tests: {test_suite.count_tests()}")
        print(f"      Security coverage: {security_suite.security_coverage:.1%}")
        print(f"      Threat coverage: {security_suite.threat_coverage:.1%}")
        
        return security_suite
    
    def _generate_owasp_tests(self, universal_ast: UniversalAST) -> List[UniversalTestCase]:
        """Generate tests for OWASP Top 10 vulnerabilities."""
        test_cases = []
        
        # A01: Broken Access Control
        access_control_case = UniversalTestCase(
            name="BrokenAccessControlTests",
            description="Tests for broken access control vulnerabilities",
            metadata=TestMetadata(tags=["owasp", "access-control"], category="security")
        )
        
        # Generate access control tests
        for module in universal_ast.modules:
            for func in module.functions:
                if self._is_access_sensitive_function(func.name):
                    # Test unauthorized access
                    test = UniversalTest(
                        name=f"test_{func.name}_unauthorized_access",
                        test_function=f"# Test unauthorized access to {func.name}\\nresult = {func.name}(unauthorized_user)",
                        description=f"Test that {func.name} properly denies unauthorized access"
                    )
                    test.add_assertion(TestAssertion(
                        assertion_type=AssertionType.THROWS,
                        actual="result",
                        exception_type="UnauthorizedError",
                        message="Should raise unauthorized error"
                    ))
                    access_control_case.add_test(test)
        
        if access_control_case.tests:
            test_cases.append(access_control_case)
        
        # A03: Injection
        if self.config.generate_injection_tests:
            injection_case = self._generate_injection_tests(universal_ast)
            if injection_case.tests:
                test_cases.append(injection_case)
        
        # A07: Identification and Authentication Failures
        if self.config.generate_auth_tests:
            auth_case = self._generate_authentication_tests(universal_ast)
            if auth_case.tests:
                test_cases.append(auth_case)
        
        return test_cases
    
    def _generate_injection_tests(self, universal_ast: UniversalAST) -> UniversalTestCase:
        """Generate injection attack tests."""
        test_case = UniversalTestCase(
            name="InjectionTests",
            description="Tests for injection vulnerabilities",
            metadata=TestMetadata(tags=["owasp", "injection"], category="security")
        )
        
        # SQL injection payloads
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('admin', 'password'); --"
        ]
        
        # XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "'><script>alert('XSS')</script>"
        ]
        
        # Command injection payloads
        cmd_payloads = [
            "; cat /etc/passwd",
            "| whoami",
            "&& rm -rf /",
            "$(cat /etc/passwd)"
        ]
        
        # Test SQL injection
        for module in universal_ast.modules:
            for func in module.functions:
                if self._is_sql_function(func.name):
                    for payload in sql_payloads:
                        test = UniversalTest(
                            name=f"test_{func.name}_sql_injection_{hash(payload) % 1000}",
                            test_function=f"# Test SQL injection in {func.name}\\ntry:\\n    result = {func.name}('{payload}')\\nexcept Exception as e:\\n    result = str(e)",
                            description=f"Test SQL injection resistance in {func.name}"
                        )
                        test.add_assertion(TestAssertion(
                            assertion_type=AssertionType.NOT_CONTAINS,
                            actual="result",
                            expected="users",
                            message="Should not expose database information"
                        ))
                        test_case.add_test(test)
        
        return test_case
    
    def _generate_authentication_tests(self, universal_ast: UniversalAST) -> UniversalTestCase:
        """Generate authentication and session management tests."""
        test_case = UniversalTestCase(
            name="AuthenticationTests",
            description="Tests for authentication and session management",
            metadata=TestMetadata(tags=["owasp", "authentication"], category="security")
        )
        
        for module in universal_ast.modules:
            for func in module.functions:
                if self._is_auth_function(func.name):
                    # Test weak password
                    test = UniversalTest(
                        name=f"test_{func.name}_weak_password",
                        test_function=f"# Test weak password rejection\\nresult = {func.name}('user', '123')",
                        description=f"Test that {func.name} rejects weak passwords"
                    )
                    test.add_assertion(TestAssertion(
                        assertion_type=AssertionType.FALSE,
                        actual="result",
                        message="Should reject weak passwords"
                    ))
                    test_case.add_test(test)
                    
                    # Test account lockout
                    test = UniversalTest(
                        name=f"test_{func.name}_account_lockout",
                        test_function=f"# Test account lockout after failed attempts\\nfor i in range(6):\\n    {func.name}('user', 'wrong_password')\\nresult = {func.name}('user', 'correct_password')",
                        description=f"Test account lockout in {func.name}"
                    )
                    test.add_assertion(TestAssertion(
                        assertion_type=AssertionType.FALSE,
                        actual="result",
                        message="Should lock account after multiple failed attempts"
                    ))
                    test_case.add_test(test)
        
        return test_case
    
    def _generate_vulnerability_tests(self, vulnerabilities: List[VulnerabilityFinding]) -> List[UniversalTestCase]:
        """Generate tests specific to found vulnerabilities."""
        test_cases = []
        
        # Group vulnerabilities by type
        vuln_groups = {}
        for vuln in vulnerabilities:
            if vuln.type not in vuln_groups:
                vuln_groups[vuln.type] = []
            vuln_groups[vuln.type].append(vuln)
        
        # Generate tests for each vulnerability type
        for vuln_type, vulns in vuln_groups.items():
            test_case = UniversalTestCase(
                name=f"{vuln_type.value.title()}VulnerabilityTests",
                description=f"Tests for {vuln_type.value.replace('_', ' ')} vulnerabilities",
                metadata=TestMetadata(tags=["vulnerability", vuln_type.value], category="security")
            )
            
            for vuln in vulns[:self.config.max_tests_per_vulnerability]:
                # Generate test based on vulnerability
                test = self._create_vulnerability_test(vuln)
                if test:
                    test_case.add_test(test)
            
            if test_case.tests:
                test_cases.append(test_case)
        
        return test_cases
    
    def _create_vulnerability_test(self, vulnerability: VulnerabilityFinding) -> Optional[UniversalTest]:
        """Create a test for a specific vulnerability."""
        
        # Get test template for this vulnerability type
        template = self.test_templates.get(vulnerability.type.value)
        if not template:
            return None
        
        # Create test based on template
        test = UniversalTest(
            name=f"test_vulnerability_{vulnerability.type.value}_{vulnerability.line_number}",
            test_function=template['test_function'].format(
                function_name=vulnerability.function_name or "target_function",
                file_path=vulnerability.file_path,
                line_number=vulnerability.line_number
            ),
            description=f"Test for {vulnerability.title} at {vulnerability.file_path}:{vulnerability.line_number}"
        )
        
        # Add assertions based on template
        for assertion_template in template.get('assertions', []):
            assertion = TestAssertion(
                assertion_type=AssertionType(assertion_template['type']),
                actual=assertion_template['actual'],
                expected=assertion_template.get('expected'),
                message=assertion_template['message']
            )
            test.add_assertion(assertion)
        
        return test
    
    def _generate_auth_tests(self, universal_ast: UniversalAST) -> List[UniversalTestCase]:
        """Generate authentication and authorization tests."""
        test_cases = []
        
        # Find auth-related functions
        auth_functions = []
        for module in universal_ast.modules:
            for func in module.functions:
                if self._is_auth_function(func.name):
                    auth_functions.append((module, func))
        
        if auth_functions:
            test_case = UniversalTestCase(
                name="AuthenticationAuthorizationTests",
                description="Authentication and authorization security tests",
                metadata=TestMetadata(tags=["auth", "security"], category="security")
            )
            
            for module, func in auth_functions:
                # Generate multiple auth tests
                auth_tests = self._create_auth_function_tests(func)
                for test in auth_tests:
                    test_case.add_test(test)
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_crypto_tests(self, universal_ast: UniversalAST) -> List[UniversalTestCase]:
        """Generate cryptographic security tests."""
        test_cases = []
        
        # Find crypto-related functions
        crypto_functions = []
        for module in universal_ast.modules:
            for func in module.functions:
                if self._is_crypto_function(func.name):
                    crypto_functions.append((module, func))
        
        if crypto_functions:
            test_case = UniversalTestCase(
                name="CryptographicTests",
                description="Cryptographic implementation security tests",
                metadata=TestMetadata(tags=["crypto", "security"], category="security")
            )
            
            for module, func in crypto_functions:
                crypto_tests = self._create_crypto_function_tests(func)
                for test in crypto_tests:
                    test_case.add_test(test)
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_business_logic_tests(self, universal_ast: UniversalAST) -> List[UniversalTestCase]:
        """Generate business logic security tests."""
        test_cases = []
        
        # Find business logic functions
        business_functions = []
        for module in universal_ast.modules:
            for func in module.functions:
                if self._is_business_logic_function(func.name):
                    business_functions.append((module, func))
        
        if business_functions:
            test_case = UniversalTestCase(
                name="BusinessLogicTests",
                description="Business logic security tests",
                metadata=TestMetadata(tags=["business-logic", "security"], category="security")
            )
            
            for module, func in business_functions:
                business_tests = self._create_business_logic_tests(func)
                for test in business_tests:
                    test_case.add_test(test)
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_compliance_tests(self, compliance_reports: List[ComplianceReport]) -> List[UniversalTestCase]:
        """Generate tests based on compliance requirements."""
        test_cases = []
        
        for report in compliance_reports:
            test_case = UniversalTestCase(
                name=f"{report.standard.value.upper()}ComplianceTests",
                description=f"Security tests for {report.standard.value.upper()} compliance",
                metadata=TestMetadata(tags=["compliance", report.standard.value], category="security")
            )
            
            # Generate tests for non-compliant rules
            for rule_id, result in report.rule_results.items():
                if result.status.value in ['non_compliant', 'partially_compliant']:
                    compliance_test = self._create_compliance_test(rule_id, result, report.standard)
                    if compliance_test:
                        test_case.add_test(compliance_test)
            
            if test_case.tests:
                test_cases.append(test_case)
        
        return test_cases
    
    def _create_compliance_test(self, rule_id: str, result, standard: ComplianceStandard) -> Optional[UniversalTest]:
        """Create a test for a compliance rule."""
        
        test = UniversalTest(
            name=f"test_compliance_{standard.value}_{rule_id}",
            test_function=f"# Compliance test for {standard.value.upper()} {rule_id}\\n# Check implementation of required controls\\nresult = check_compliance_implementation()",
            description=f"Test compliance with {standard.value.upper()} rule {rule_id}"
        )
        
        # Add appropriate assertions based on rule type
        test.add_assertion(TestAssertion(
            assertion_type=AssertionType.TRUE,
            actual="result",
            message=f"Should comply with {standard.value.upper()} rule {rule_id}"
        ))
        
        return test
    
    # Helper methods for function classification
    
    def _is_access_sensitive_function(self, func_name: str) -> bool:
        """Check if function handles access control."""
        keywords = ['access', 'permission', 'authorize', 'allow', 'deny', 'grant', 'revoke']
        return any(keyword in func_name.lower() for keyword in keywords)
    
    def _is_sql_function(self, func_name: str) -> bool:
        """Check if function handles SQL operations."""
        keywords = ['sql', 'query', 'select', 'insert', 'update', 'delete', 'execute']
        return any(keyword in func_name.lower() for keyword in keywords)
    
    def _is_auth_function(self, func_name: str) -> bool:
        """Check if function handles authentication."""
        keywords = ['auth', 'login', 'signin', 'verify', 'validate', 'session']
        return any(keyword in func_name.lower() for keyword in keywords)
    
    def _is_crypto_function(self, func_name: str) -> bool:
        """Check if function handles cryptography."""
        keywords = ['encrypt', 'decrypt', 'hash', 'sign', 'verify', 'crypto', 'cipher']
        return any(keyword in func_name.lower() for keyword in keywords)
    
    def _is_business_logic_function(self, func_name: str) -> bool:
        """Check if function handles business logic."""
        keywords = ['calculate', 'process', 'validate', 'approve', 'reject', 'transfer', 'payment']
        return any(keyword in func_name.lower() for keyword in keywords)
    
    def _create_auth_function_tests(self, func) -> List[UniversalTest]:
        """Create authentication-specific tests."""
        tests = []
        
        # Test 1: Empty credentials
        test = UniversalTest(
            name=f"test_{func.name}_empty_credentials",
            test_function=f"result = {func.name}('', '')",
            description=f"Test {func.name} with empty credentials"
        )
        test.add_assertion(TestAssertion(
            assertion_type=AssertionType.FALSE,
            actual="result",
            message="Should reject empty credentials"
        ))
        tests.append(test)
        
        # Test 2: SQL injection in username
        test = UniversalTest(
            name=f"test_{func.name}_sql_injection",
            test_function=f"result = {func.name}(\"admin'; --\", 'password')",
            description=f"Test {func.name} against SQL injection"
        )
        test.add_assertion(TestAssertion(
            assertion_type=AssertionType.FALSE,
            actual="result",
            message="Should resist SQL injection attacks"
        ))
        tests.append(test)
        
        return tests
    
    def _create_crypto_function_tests(self, func) -> List[UniversalTest]:
        """Create cryptography-specific tests."""
        tests = []
        
        # Test weak key
        test = UniversalTest(
            name=f"test_{func.name}_weak_key",
            test_function=f"result = {func.name}('weak_key', 'data')",
            description=f"Test {func.name} with weak encryption key"
        )
        test.add_assertion(TestAssertion(
            assertion_type=AssertionType.THROWS,
            actual="result",
            exception_type="WeakKeyError",
            message="Should reject weak encryption keys"
        ))
        tests.append(test)
        
        return tests
    
    def _create_business_logic_tests(self, func) -> List[UniversalTest]:
        """Create business logic security tests."""
        tests = []
        
        # Test negative values
        test = UniversalTest(
            name=f"test_{func.name}_negative_values",
            test_function=f"result = {func.name}(-1000)",
            description=f"Test {func.name} with negative values"
        )
        test.add_assertion(TestAssertion(
            assertion_type=AssertionType.THROWS,
            actual="result",
            exception_type="ValueError",
            message="Should reject negative values where inappropriate"
        ))
        tests.append(test)
        
        return tests
    
    def _load_threat_models(self) -> List[ThreatModel]:
        """Load predefined threat models."""
        return [
            ThreatModel(
                name="Web Application Threats",
                description="Common threats to web applications",
                threats=[ThreatCategory.INJECTION, ThreatCategory.CROSS_SITE_SCRIPTING, ThreatCategory.BROKEN_AUTHENTICATION],
                risk_level=SeverityLevel.HIGH,
                likelihood=0.8,
                impact=0.9,
                applicable_languages=["python", "javascript", "java", "php"],
                attack_vectors=["HTTP requests", "Form inputs", "URL parameters"],
                entry_points=["Web forms", "API endpoints", "File uploads"]
            ),
            ThreatModel(
                name="API Security Threats",
                description="Threats specific to APIs",
                threats=[ThreatCategory.BROKEN_ACCESS_CONTROL, ThreatCategory.INJECTION],
                risk_level=SeverityLevel.HIGH,
                likelihood=0.7,
                impact=0.8,
                applicable_languages=["python", "javascript", "java", "go"],
                attack_vectors=["API calls", "Authentication bypass", "Parameter manipulation"],
                entry_points=["REST endpoints", "GraphQL queries", "API keys"]
            )
        ]
    
    def _load_attack_patterns(self) -> Dict[str, List[str]]:
        """Load attack patterns for different vulnerability types."""
        return {
            'sql_injection': [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "' UNION SELECT * FROM users --"
            ],
            'xss': [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>"
            ],
            'command_injection': [
                "; cat /etc/passwd",
                "| whoami",
                "&& rm -rf /"
            ]
        }
    
    def _load_test_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load test templates for different vulnerability types."""
        return {
            'sql_injection': {
                'test_function': "# Test for SQL injection in {function_name}\\nresult = {function_name}(\"'; DROP TABLE users; --\")",
                'assertions': [
                    {
                        'type': 'not_contains',
                        'actual': 'result',
                        'expected': 'DROP',
                        'message': 'Should not execute SQL commands'
                    }
                ]
            },
            'cross_site_scripting': {
                'test_function': "# Test for XSS in {function_name}\\nresult = {function_name}(\"<script>alert('XSS')</script>\")",
                'assertions': [
                    {
                        'type': 'not_contains',
                        'actual': 'result',
                        'expected': '<script>',
                        'message': 'Should sanitize script tags'
                    }
                ]
            }
        }
    
    def _get_applicable_threat_models(self, universal_ast: UniversalAST) -> List[ThreatModel]:
        """Get threat models applicable to the codebase."""
        applicable = []
        
        # Detect primary language
        primary_language = "python"  # Simplified detection
        
        for threat_model in self.threat_models:
            if primary_language in threat_model.applicable_languages:
                applicable.append(threat_model)
        
        return applicable
    
    def _calculate_security_coverage(self, tested_vulnerabilities: Set[VulnerabilityType]) -> float:
        """Calculate security coverage percentage."""
        total_vulnerability_types = len(VulnerabilityType)
        tested_count = len(tested_vulnerabilities)
        return tested_count / total_vulnerability_types
    
    def _calculate_threat_coverage(self, threat_models: List[ThreatModel]) -> float:
        """Calculate threat coverage percentage."""
        total_threat_categories = len(ThreatCategory)
        covered_threats = set()
        
        for threat_model in threat_models:
            covered_threats.update(threat_model.threats)
        
        return len(covered_threats) / total_threat_categories