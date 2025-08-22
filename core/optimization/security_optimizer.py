"""
Security Optimizer - Advanced security vulnerability detection and remediation

This module provides comprehensive security analysis and optimization capabilities
for identifying vulnerabilities, security anti-patterns, and implementing secure
coding practices with automated remediation suggestions.

Key Capabilities:
- OWASP Top 10 vulnerability detection and remediation
- Input validation and sanitization recommendations
- Cryptographic implementation analysis and improvements
- Authentication and authorization security assessments
- Secure coding pattern enforcement and guidance
- Automated security fix generation with risk assessment
"""

import ast
import re
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import base64

from .optimization_models import (
    OptimizationType, OptimizationPriority, OptimizationStrategy,
    OptimizationRecommendation, SecurityMetrics, RiskAssessment,
    create_optimization_recommendation, create_risk_assessment
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityVulnerability:
    """Represents a detected security vulnerability"""
    vuln_type: str
    severity: str
    confidence: float
    cwe_id: str
    owasp_category: str
    description: str
    location: Tuple[int, int]
    affected_code: str
    remediation: str
    risk_level: str = "medium"
    exploitability: float = 0.5
    impact: float = 0.5
    references: List[str] = field(default_factory=list)


@dataclass
class SecurityPattern:
    """Represents a security pattern or anti-pattern"""
    pattern_id: str
    pattern_type: str
    secure_implementation: str
    insecure_examples: List[str]
    security_impact: str
    implementation_guidance: str
    compliance_standards: List[str] = field(default_factory=list)


class SecurityOptimizer:
    """
    Advanced security analysis and optimization engine
    
    Provides comprehensive security analysis through static code analysis,
    vulnerability detection, and secure coding practice recommendations.
    """
    
    def __init__(self):
        """Initialize security optimizer with vulnerability databases"""
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.secure_patterns = self._load_secure_patterns()
        self.crypto_patterns = self._load_crypto_patterns()
        self.input_validation_patterns = self._load_validation_patterns()
        
        # Security configuration
        self.config = {
            'severity_threshold': 'medium',
            'compliance_standards': ['OWASP', 'NIST', 'ISO27001'],
            'crypto_algorithms': {
                'approved': ['AES', 'RSA', 'ECDSA', 'SHA256', 'SHA384', 'SHA512'],
                'deprecated': ['MD5', 'SHA1', 'DES', '3DES', 'RC4']
            },
            'max_password_age_days': 90,
            'min_password_length': 12
        }
        
        logger.info("Security Optimizer initialized")
    
    async def analyze_security(self, code: str, tree: ast.AST, file_path: str = "") -> List[OptimizationRecommendation]:
        """
        Comprehensive security analysis of code
        
        Args:
            code: Source code as string
            tree: AST representation of the code
            file_path: Path to the file being analyzed
            
        Returns:
            List of security optimization recommendations
        """
        recommendations = []
        
        try:
            # Multi-layer security analysis
            injection_vulns = await self._detect_injection_vulnerabilities(code, tree)
            crypto_issues = await self._analyze_cryptographic_implementations(code, tree)
            auth_issues = await self._analyze_authentication_security(code, tree)
            input_validation = await self._analyze_input_validation(code, tree)
            data_exposure = await self._detect_data_exposure_risks(code, tree)
            access_control = await self._analyze_access_control(code, tree)
            
            # Convert vulnerabilities to recommendations
            recommendations.extend(self._create_injection_recommendations(injection_vulns, file_path))
            recommendations.extend(self._create_crypto_recommendations(crypto_issues, file_path))
            recommendations.extend(self._create_auth_recommendations(auth_issues, file_path))
            recommendations.extend(self._create_validation_recommendations(input_validation, file_path))
            recommendations.extend(self._create_data_exposure_recommendations(data_exposure, file_path))
            recommendations.extend(self._create_access_control_recommendations(access_control, file_path))
            
            # Apply security-specific prioritization
            recommendations = self._prioritize_security_recommendations(recommendations)
            
            logger.info(f"Generated {len(recommendations)} security recommendations for {file_path}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in security analysis: {e}")
            return []
    
    async def _detect_injection_vulnerabilities(self, code: str, tree: ast.AST) -> List[SecurityVulnerability]:
        """Detect injection vulnerabilities (SQL, Command, LDAP, etc.)"""
        vulnerabilities = []
        
        # SQL Injection detection
        sql_patterns = [
            r'cursor\.execute\s*\(\s*["\'].*%.*["\'].*%',
            r'cursor\.execute\s*\(\s*["\'].*\+.*["\']',
            r'cursor\.execute\s*\(\s*f["\'].*\{.*\}.*["\']',
            r'query\s*=\s*["\'].*%.*["\'].*%',
            r'query\s*=\s*["\'].*\+.*["\']'
        ]
        
        for pattern in sql_patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type="sql_injection",
                    severity="high",
                    confidence=0.8,
                    cwe_id="CWE-89",
                    owasp_category="A03:2021 – Injection",
                    description="Potential SQL injection vulnerability detected",
                    location=(line_num, line_num),
                    affected_code=match.group(),
                    remediation="Use parameterized queries or prepared statements",
                    risk_level="high",
                    exploitability=0.8,
                    impact=0.9,
                    references=["https://owasp.org/www-community/attacks/SQL_Injection"]
                ))
        
        # Command Injection detection
        class CommandInjectionVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if hasattr(node.func, 'id'):
                    # subprocess calls with shell=True
                    if node.func.id in ['system', 'popen']:
                        vulnerabilities.append(SecurityVulnerability(
                            vuln_type="command_injection",
                            severity="critical",
                            confidence=0.9,
                            cwe_id="CWE-78",
                            owasp_category="A03:2021 – Injection",
                            description=f"Use of {node.func.id}() can execute arbitrary commands",
                            location=(node.lineno, node.end_lineno or node.lineno),
                            affected_code=f"{node.func.id}(...)",
                            remediation="Use subprocess with shell=False and argument lists",
                            risk_level="critical",
                            exploitability=0.9,
                            impact=1.0
                        ))
                
                elif hasattr(node.func, 'attr'):
                    # subprocess.call, subprocess.run with shell=True
                    if hasattr(node.func.value, 'id') and node.func.value.id == 'subprocess':
                        shell_true = False
                        for keyword in node.keywords:
                            if keyword.arg == 'shell' and hasattr(keyword.value, 'value'):
                                if keyword.value.value is True:
                                    shell_true = True
                                    break
                        
                        if shell_true:
                            vulnerabilities.append(SecurityVulnerability(
                                vuln_type="command_injection",
                                severity="high",
                                confidence=0.8,
                                cwe_id="CWE-78",
                                owasp_category="A03:2021 – Injection",
                                description="subprocess with shell=True is vulnerable to command injection",
                                location=(node.lineno, node.end_lineno or node.lineno),
                                affected_code="subprocess with shell=True",
                                remediation="Use shell=False and pass commands as list arguments",
                                risk_level="high",
                                exploitability=0.7,
                                impact=0.9
                            ))
                
                self.generic_visit(node)
        
        visitor = CommandInjectionVisitor()
        visitor.visit(tree)
        
        return vulnerabilities
    
    async def _analyze_cryptographic_implementations(self, code: str, tree: ast.AST) -> List[SecurityVulnerability]:
        """Analyze cryptographic implementations for security issues"""
        vulnerabilities = []
        
        # Weak cryptographic algorithms
        weak_crypto_patterns = {
            'MD5': ('CWE-327', 'MD5 is cryptographically broken'),
            'SHA1': ('CWE-327', 'SHA-1 is cryptographically weak'),
            'DES': ('CWE-327', 'DES has insufficient key length'),
            '3DES': ('CWE-327', '3DES is deprecated'),
            'RC4': ('CWE-327', 'RC4 has known vulnerabilities')
        }
        
        for algorithm, (cwe, description) in weak_crypto_patterns.items():
            if algorithm.lower() in code.lower():
                line_matches = [i for i, line in enumerate(code.split('\n'), 1) if algorithm.lower() in line.lower()]
                for line_num in line_matches:
                    vulnerabilities.append(SecurityVulnerability(
                        vuln_type="weak_cryptography",
                        severity="high",
                        confidence=0.8,
                        cwe_id=cwe,
                        owasp_category="A02:2021 – Cryptographic Failures",
                        description=f"Weak cryptographic algorithm: {description}",
                        location=(line_num, line_num),
                        affected_code=f"{algorithm} usage",
                        remediation=f"Replace {algorithm} with SHA-256, SHA-384, or SHA-512",
                        risk_level="high",
                        exploitability=0.6,
                        impact=0.8
                    ))
        
        # Hardcoded cryptographic keys/secrets
        secret_patterns = [
            (r'(?i)(password|pwd)\s*=\s*["\'][^"\']{8,}["\']', 'hardcoded_password'),
            (r'(?i)(api_key|apikey)\s*=\s*["\'][^"\']{20,}["\']', 'hardcoded_api_key'),
            (r'(?i)(secret|token)\s*=\s*["\'][^"\']{16,}["\']', 'hardcoded_secret'),
            (r'(?i)(private_key|privatekey)\s*=\s*["\'][^"\']{100,}["\']', 'hardcoded_private_key')
        ]
        
        for pattern, vuln_type in secret_patterns:
            matches = list(re.finditer(pattern, code))
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type=vuln_type,
                    severity="critical",
                    confidence=0.7,
                    cwe_id="CWE-798",
                    owasp_category="A02:2021 – Cryptographic Failures",
                    description=f"Hardcoded {vuln_type.replace('_', ' ')} detected",
                    location=(line_num, line_num),
                    affected_code=match.group()[:50] + "...",
                    remediation="Use environment variables or secure key management systems",
                    risk_level="critical",
                    exploitability=0.9,
                    impact=1.0
                ))
        
        # Insufficient random number generation
        weak_random_patterns = ['random.random', 'random.randint', 'random.choice']
        for pattern in weak_random_patterns:
            if pattern in code:
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type="weak_random",
                    severity="medium",
                    confidence=0.6,
                    cwe_id="CWE-330",
                    owasp_category="A02:2021 – Cryptographic Failures",
                    description="Use of weak random number generator for security purposes",
                    location=(0, 0),
                    affected_code=pattern,
                    remediation="Use secrets module or os.urandom() for cryptographic randomness",
                    risk_level="medium",
                    exploitability=0.4,
                    impact=0.6
                ))
        
        return vulnerabilities
    
    async def _analyze_authentication_security(self, code: str, tree: ast.AST) -> List[SecurityVulnerability]:
        """Analyze authentication and session management security"""
        vulnerabilities = []
        
        # Weak password validation
        if 'password' in code.lower() and 'len(' in code:
            # Look for password length checks
            weak_password_pattern = r'len\(.*password.*\)\s*[<>=]+\s*[1-7]'
            if re.search(weak_password_pattern, code, re.IGNORECASE):
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type="weak_password_policy",
                    severity="medium",
                    confidence=0.7,
                    cwe_id="CWE-521",
                    owasp_category="A07:2021 – Identification and Authentication Failures",
                    description="Weak password length requirement detected",
                    location=(0, 0),
                    affected_code="password length check",
                    remediation="Require minimum 12 characters with complexity requirements",
                    risk_level="medium",
                    exploitability=0.5,
                    impact=0.6
                ))
        
        # Missing authentication checks
        protected_operations = ['delete', 'update', 'admin', 'sensitive']
        for operation in protected_operations:
            if operation in code.lower() and 'auth' not in code.lower():
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type="missing_authentication",
                    severity="high",
                    confidence=0.6,
                    cwe_id="CWE-306",
                    owasp_category="A01:2021 – Broken Access Control",
                    description=f"Potential {operation} operation without authentication check",
                    location=(0, 0),
                    affected_code=f"{operation} operation",
                    remediation="Add authentication and authorization checks",
                    risk_level="high",
                    exploitability=0.8,
                    impact=0.7
                ))
        
        # Session fixation vulnerabilities
        if 'session' in code.lower() and 'regenerate' not in code.lower():
            vulnerabilities.append(SecurityVulnerability(
                vuln_type="session_fixation",
                severity="medium",
                confidence=0.5,
                cwe_id="CWE-384",
                owasp_category="A07:2021 – Identification and Authentication Failures",
                description="Session ID may not be regenerated after authentication",
                location=(0, 0),
                affected_code="session handling",
                remediation="Regenerate session ID after successful authentication",
                risk_level="medium",
                exploitability=0.4,
                impact=0.6
            ))
        
        return vulnerabilities
    
    async def _analyze_input_validation(self, code: str, tree: ast.AST) -> List[SecurityVulnerability]:
        """Analyze input validation and sanitization"""
        vulnerabilities = []
        
        # Missing input validation
        class InputValidationVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check if function parameters are validated
                if node.args.args and len(node.args.args) > 1:  # Skip 'self'
                    has_validation = False
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if hasattr(child.func, 'id') and child.func.id in ['isinstance', 'len', 'validate']:
                                has_validation = True
                                break
                    
                    if not has_validation:
                        vulnerabilities.append(SecurityVulnerability(
                            vuln_type="missing_input_validation",
                            severity="medium",
                            confidence=0.5,
                            cwe_id="CWE-20",
                            owasp_category="A03:2021 – Injection",
                            description=f"Function '{node.name}' may lack input validation",
                            location=(node.lineno, node.end_lineno or node.lineno),
                            affected_code=f"def {node.name}(...)",
                            remediation="Add input validation and sanitization",
                            risk_level="medium",
                            exploitability=0.6,
                            impact=0.5
                        ))
                
                self.generic_visit(node)
        
        visitor = InputValidationVisitor()
        visitor.visit(tree)
        
        # XSS vulnerabilities in web contexts
        xss_patterns = ['render_template_string', 'Markup(', '|safe', 'innerHTML']
        for pattern in xss_patterns:
            if pattern in code:
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type="xss_vulnerability",
                    severity="high",
                    confidence=0.6,
                    cwe_id="CWE-79",
                    owasp_category="A03:2021 – Injection",
                    description="Potential Cross-Site Scripting (XSS) vulnerability",
                    location=(0, 0),
                    affected_code=pattern,
                    remediation="Use proper output encoding and Content Security Policy",
                    risk_level="high",
                    exploitability=0.7,
                    impact=0.6
                ))
        
        return vulnerabilities
    
    async def _detect_data_exposure_risks(self, code: str, tree: ast.AST) -> List[SecurityVulnerability]:
        """Detect data exposure and information disclosure risks"""
        vulnerabilities = []
        
        # Sensitive data in logs
        sensitive_keywords = ['password', 'token', 'key', 'secret', 'ssn', 'credit_card']
        log_functions = ['print(', 'logging.', 'log.', 'logger.', 'console.log']
        
        for log_func in log_functions:
            if log_func in code:
                for keyword in sensitive_keywords:
                    pattern = f'{re.escape(log_func)}.*{keyword}'
                    if re.search(pattern, code, re.IGNORECASE):
                        vulnerabilities.append(SecurityVulnerability(
                            vuln_type="sensitive_data_exposure",
                            severity="medium",
                            confidence=0.7,
                            cwe_id="CWE-532",
                            owasp_category="A09:2021 – Security Logging and Monitoring Failures",
                            description=f"Potential logging of sensitive data: {keyword}",
                            location=(0, 0),
                            affected_code=f"{log_func}...{keyword}",
                            remediation="Remove sensitive data from logs or implement log sanitization",
                            risk_level="medium",
                            exploitability=0.3,
                            impact=0.7
                        ))
        
        # Debug information exposure
        if 'debug=True' in code or 'DEBUG = True' in code:
            vulnerabilities.append(SecurityVulnerability(
                vuln_type="debug_information_exposure",
                severity="medium",
                confidence=0.8,
                cwe_id="CWE-489",
                owasp_category="A05:2021 – Security Misconfiguration",
                description="Debug mode enabled in production",
                location=(0, 0),
                affected_code="debug=True",
                remediation="Disable debug mode in production environments",
                risk_level="medium",
                exploitability=0.4,
                impact=0.5
            ))
        
        return vulnerabilities
    
    async def _analyze_access_control(self, code: str, tree: ast.AST) -> List[SecurityVulnerability]:
        """Analyze access control and authorization mechanisms"""
        vulnerabilities = []
        
        # Missing access control on sensitive operations
        sensitive_operations = ['DELETE', 'UPDATE', 'admin', 'config', 'user_data']
        
        for operation in sensitive_operations:
            if operation.lower() in code.lower():
                # Check if there's any authorization check nearby
                if not re.search(r'(auth|permission|role|access|check)', code, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        vuln_type="missing_access_control",
                        severity="high",
                        confidence=0.6,
                        cwe_id="CWE-862",
                        owasp_category="A01:2021 – Broken Access Control",
                        description=f"Sensitive operation '{operation}' may lack access control",
                        location=(0, 0),
                        affected_code=operation,
                        remediation="Implement proper authorization checks",
                        risk_level="high",
                        exploitability=0.8,
                        impact=0.7
                    ))
        
        # Path traversal vulnerabilities
        file_operations = ['open(', 'file(', 'Path(']
        for file_op in file_operations:
            if file_op in code and 'user' in code.lower():
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type="path_traversal",
                    severity="high",
                    confidence=0.5,
                    cwe_id="CWE-22",
                    owasp_category="A01:2021 – Broken Access Control",
                    description="Potential path traversal vulnerability with user input",
                    location=(0, 0),
                    affected_code=file_op,
                    remediation="Validate and sanitize file paths, use allow-lists",
                    risk_level="high",
                    exploitability=0.6,
                    impact=0.8
                ))
        
        return vulnerabilities
    
    def _create_injection_recommendations(self, vulnerabilities: List[SecurityVulnerability], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for injection vulnerabilities"""
        recommendations = []
        
        for vuln in vulnerabilities:
            if vuln.vuln_type in ["sql_injection", "command_injection"]:
                rec = create_optimization_recommendation(
                    OptimizationType.SECURITY,
                    f"Security Fix: {vuln.vuln_type.replace('_', ' ').title()}",
                    vuln.description,
                    file_path,
                    OptimizationPriority.CRITICAL if vuln.severity == "critical" else OptimizationPriority.HIGH,
                    OptimizationStrategy.IMMEDIATE
                )
                
                rec.target_lines = vuln.location
                rec.original_code = vuln.affected_code
                rec.optimized_code = vuln.remediation
                rec.confidence_score = vuln.confidence
                rec.reasoning = f"CWE-{vuln.cwe_id}: {vuln.owasp_category}"
                
                # High security risk
                rec.risk_assessment = create_risk_assessment(
                    implementation_risk=0.2,
                    business_impact_risk=0.9,
                    technical_debt_risk=0.3
                )
                rec.risk_assessment.security_risk = vuln.impact
                rec.risk_assessment.overall_risk_score = 0.8
                
                rec.tags = [vuln.owasp_category, f"CWE-{vuln.cwe_id}"]
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_crypto_recommendations(self, vulnerabilities: List[SecurityVulnerability], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for cryptographic issues"""
        recommendations = []
        
        for vuln in vulnerabilities:
            if vuln.vuln_type in ["weak_cryptography", "hardcoded_password", "hardcoded_api_key", "hardcoded_secret", "weak_random"]:
                priority = OptimizationPriority.CRITICAL if vuln.severity == "critical" else OptimizationPriority.HIGH
                
                rec = create_optimization_recommendation(
                    OptimizationType.SECURITY,
                    f"Cryptographic Security: {vuln.vuln_type.replace('_', ' ').title()}",
                    vuln.description,
                    file_path,
                    priority,
                    OptimizationStrategy.IMMEDIATE if vuln.severity == "critical" else OptimizationStrategy.GRADUAL
                )
                
                rec.target_lines = vuln.location
                rec.original_code = vuln.affected_code
                rec.optimized_code = vuln.remediation
                rec.confidence_score = vuln.confidence
                rec.reasoning = f"CWE-{vuln.cwe_id}: Critical cryptographic vulnerability"
                
                rec.risk_assessment = create_risk_assessment(
                    implementation_risk=0.3,
                    business_impact_risk=0.9,
                    technical_debt_risk=0.2
                )
                rec.risk_assessment.security_risk = 0.9
                rec.risk_assessment.overall_risk_score = 0.8
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_auth_recommendations(self, vulnerabilities: List[SecurityVulnerability], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for authentication issues"""
        recommendations = []
        
        for vuln in vulnerabilities:
            if vuln.vuln_type in ["weak_password_policy", "missing_authentication", "session_fixation"]:
                rec = create_optimization_recommendation(
                    OptimizationType.SECURITY,
                    f"Authentication Security: {vuln.vuln_type.replace('_', ' ').title()}",
                    vuln.description,
                    file_path,
                    OptimizationPriority.HIGH,
                    OptimizationStrategy.REFACTOR
                )
                
                rec.optimized_code = vuln.remediation
                rec.confidence_score = vuln.confidence
                rec.reasoning = f"CWE-{vuln.cwe_id}: Authentication vulnerability"
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_validation_recommendations(self, vulnerabilities: List[SecurityVulnerability], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for input validation issues"""
        recommendations = []
        
        for vuln in vulnerabilities:
            if vuln.vuln_type in ["missing_input_validation", "xss_vulnerability"]:
                priority = OptimizationPriority.HIGH if vuln.vuln_type == "xss_vulnerability" else OptimizationPriority.MEDIUM
                
                rec = create_optimization_recommendation(
                    OptimizationType.SECURITY,
                    f"Input Security: {vuln.vuln_type.replace('_', ' ').title()}",
                    vuln.description,
                    file_path,
                    priority,
                    OptimizationStrategy.REFACTOR
                )
                
                rec.target_lines = vuln.location
                rec.optimized_code = vuln.remediation
                rec.confidence_score = vuln.confidence
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_data_exposure_recommendations(self, vulnerabilities: List[SecurityVulnerability], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for data exposure issues"""
        recommendations = []
        
        for vuln in vulnerabilities:
            if vuln.vuln_type in ["sensitive_data_exposure", "debug_information_exposure"]:
                rec = create_optimization_recommendation(
                    OptimizationType.SECURITY,
                    f"Data Protection: {vuln.vuln_type.replace('_', ' ').title()}",
                    vuln.description,
                    file_path,
                    OptimizationPriority.MEDIUM,
                    OptimizationStrategy.GRADUAL
                )
                
                rec.optimized_code = vuln.remediation
                rec.confidence_score = vuln.confidence
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_access_control_recommendations(self, vulnerabilities: List[SecurityVulnerability], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for access control issues"""
        recommendations = []
        
        for vuln in vulnerabilities:
            if vuln.vuln_type in ["missing_access_control", "path_traversal"]:
                rec = create_optimization_recommendation(
                    OptimizationType.SECURITY,
                    f"Access Control: {vuln.vuln_type.replace('_', ' ').title()}",
                    vuln.description,
                    file_path,
                    OptimizationPriority.HIGH,
                    OptimizationStrategy.REFACTOR
                )
                
                rec.optimized_code = vuln.remediation
                rec.confidence_score = vuln.confidence
                rec.reasoning = f"CWE-{vuln.cwe_id}: Access control vulnerability"
                
                recommendations.append(rec)
        
        return recommendations
    
    def _prioritize_security_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Apply security-specific prioritization"""
        def security_score(rec: OptimizationRecommendation) -> float:
            base_score = rec.confidence_score
            
            # Critical security issues get highest priority
            if rec.priority == OptimizationPriority.CRITICAL:
                base_score *= 2.0
            
            # High-impact security issues
            if rec.priority == OptimizationPriority.HIGH:
                base_score *= 1.5
            
            # OWASP Top 10 vulnerabilities get priority boost
            if any(tag for tag in rec.tags if 'A0' in tag):
                base_score *= 1.3
            
            return base_score
        
        return sorted(recommendations, key=security_score, reverse=True)
    
    def _load_vulnerability_patterns(self) -> Dict[str, Any]:
        """Load vulnerability pattern database"""
        return {
            'injection_patterns': [
                'sql_injection', 'command_injection', 'ldap_injection',
                'xpath_injection', 'nosql_injection'
            ],
            'crypto_patterns': [
                'weak_algorithms', 'hardcoded_keys', 'insufficient_randomness',
                'improper_certificate_validation'
            ],
            'auth_patterns': [
                'weak_passwords', 'session_fixation', 'missing_authentication',
                'privilege_escalation'
            ]
        }
    
    def _load_secure_patterns(self) -> Dict[str, SecurityPattern]:
        """Load secure coding patterns"""
        return {
            'parameterized_queries': SecurityPattern(
                pattern_id="param_queries",
                pattern_type="database_security",
                secure_implementation="cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                insecure_examples=["f'SELECT * FROM users WHERE id = {user_id}'"],
                security_impact="Prevents SQL injection attacks",
                implementation_guidance="Always use parameterized queries or prepared statements"
            )
        }
    
    def _load_crypto_patterns(self) -> Dict[str, Any]:
        """Load cryptographic security patterns"""
        return {
            'approved_algorithms': ['AES-256', 'RSA-2048', 'ECDSA', 'SHA-256'],
            'deprecated_algorithms': ['MD5', 'SHA1', 'DES', '3DES', 'RC4'],
            'key_sizes': {
                'RSA': 2048,
                'AES': 256,
                'ECDSA': 256
            }
        }
    
    def _load_validation_patterns(self) -> Dict[str, Any]:
        """Load input validation patterns"""
        return {
            'validation_functions': ['isinstance', 'len', 'validate', 'sanitize'],
            'dangerous_inputs': ['eval', 'exec', 'compile', '__import__'],
            'encoding_functions': ['html.escape', 'urllib.parse.quote', 'base64.b64encode']
        }


# Factory function
def create_security_optimizer() -> SecurityOptimizer:
    """Create and configure security optimizer"""
    return SecurityOptimizer()