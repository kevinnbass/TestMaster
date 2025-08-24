"""
Security Analysis Module
========================

Implements comprehensive security analysis:
- Vulnerability pattern detection
- Input validation analysis
- Authentication and authorization checks
- Cryptography usage analysis
- SQL injection and XSS detection
- Security code smells and hotspots
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class SecurityAnalyzer(BaseAnalyzer):
    """Analyzer for security vulnerabilities and patterns."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self._init_security_patterns()
    
    def _init_security_patterns(self):
        """Initialize security vulnerability patterns including OWASP Top 10 2021."""
        # OWASP Top 10 2021 Vulnerability Categories
        self.owasp_categories = {
            'A01_BROKEN_ACCESS_CONTROL': 'Broken Access Control',
            'A02_CRYPTOGRAPHIC_FAILURES': 'Cryptographic Failures', 
            'A03_INJECTION': 'Injection',
            'A04_INSECURE_DESIGN': 'Insecure Design',
            'A05_SECURITY_MISCONFIGURATION': 'Security Misconfiguration',
            'A06_VULNERABLE_COMPONENTS': 'Vulnerable and Outdated Components',
            'A07_IDENTIFICATION_FAILURES': 'Identification and Authentication Failures',
            'A08_SOFTWARE_INTEGRITY_FAILURES': 'Software and Data Integrity Failures',
            'A09_LOGGING_FAILURES': 'Security Logging and Monitoring Failures',
            'A10_SERVER_SIDE_REQUEST_FORGERY': 'Server-Side Request Forgery'
        }
        
        # A01: Broken Access Control patterns
        self.access_control_patterns = [
            r'@login_required.*\n.*@admin_required',  # Multiple decorators stacking
            r'if.*user\.is_admin.*and.*user\.is_authenticated',  # Complex access checks
            r'bypass.*auth',  # Authentication bypass keywords
            r'skip.*permission',  # Permission bypass keywords
            r'admin.*=.*True.*without',  # Admin bypass
            r'permission.*=.*None',  # Missing permissions
            r'authorize.*=.*False',  # Disabled authorization
            r'allow_all.*=.*True',  # Allow all access
        ]
        
        # A02: Cryptographic Failures patterns (enhanced)
        self.cryptographic_failure_patterns = [
            r'md5\s*\(',  # MD5 usage
            r'sha1\s*\(',  # SHA1 usage  
            r'DES\s*\(',  # DES encryption
            r'3DES\s*\(',  # 3DES encryption
            r'RC4\s*\(',  # RC4 encryption
            r'ECB\s*mode',  # ECB mode usage
            r'random\.random\(',  # Weak randomness for crypto
            r'password.*=.*["\'][^"\']{1,8}["\']',  # Short passwords
            r'key.*=.*["\'][^"\']{1,16}["\']',  # Short encryption keys
            r'ssl.*verify.*=.*False',  # SSL verification disabled
            r'ssl.*check_hostname.*=.*False',  # SSL hostname check disabled
            r'TLSv1[^.2]',  # Old TLS versions
            r'SSLv[23]',  # Old SSL versions
            r'verify_mode.*=.*ssl\.CERT_NONE',  # SSL certificate verification disabled
        ]
        
        # A03: Injection patterns (comprehensive)
        self.sql_injection_patterns = [
            r'execute\s*\(\s*["\'].*%.*["\']',  # String formatting in SQL
            r'executemany\s*\(\s*["\'].*%.*["\']',
            r'query\s*\(\s*["\'].*\+.*["\']',  # String concatenation in SQL
            r'cursor\.execute\s*\(\s*["\'].*\+',
            r'SELECT.*\+.*FROM',  # SQL concatenation
            r'INSERT.*\+.*VALUES',
            r'UPDATE.*\+.*SET',
            r'DELETE.*\+.*WHERE',
            r'raw\s*\(\s*["\'].*%.*["\']',  # Django raw SQL with formatting
            r'extra\s*\(\s*where=.*%',  # Django extra() with string formatting
            r'\.filter\s*\(\s*["\'].*%.*["\']',  # ORM filter with string formatting
        ]
        
        # A03: XSS patterns (enhanced)
        self.xss_patterns = [
            r'innerHTML\s*=.*\+',  # Unsafe innerHTML usage
            r'document\.write\s*\(',  # document.write usage
            r'eval\s*\(',  # eval usage
            r'render_template_string.*\+',  # Unsafe template rendering
            r'Markup\s*\(',  # Flask Markup without escaping
            r'html\s*\+.*user',  # HTML concatenation with user input
            r'safe\s*=.*False',  # Django template safe=False
            r'escape\s*=.*False',  # Template escape disabled
            r'autoescape.*off',  # Autoescape disabled
            r'|safe.*}}',  # Django template safe filter
        ]
        
        # A03: Command injection patterns (enhanced)
        self.command_injection_patterns = [
            r'os\.system\s*\(',
            r'subprocess\.(call|check_call|check_output|run)\s*\(',
            r'popen\s*\(',
            r'exec\s*\(',
            r'eval\s*\(',  # Code injection
            r'compile\s*\(',  # Code compilation
            r'shell=True',  # Dangerous subprocess usage
            r'os\.popen\s*\(',  # OS popen
            r'commands\.getoutput\s*\(',  # Deprecated commands module
            r'__import__\s*\(',  # Dynamic imports
        ]
        
        # A04: Insecure Design patterns
        self.insecure_design_patterns = [
            r'TODO.*security',  # Security TODOs
            r'FIXME.*security',  # Security FIXMEs
            r'password.*in.*url',  # Password in URL
            r'secret.*in.*log',  # Secrets in logs
            r'debug.*=.*True.*production',  # Debug mode in production
            r'allow.*origins.*\*',  # Allow all origins CORS
            r'csrf.*exempt',  # CSRF exemption
            r'permission.*classes.*=.*\[\]',  # Empty permission classes
        ]
        
        # A05: Security Misconfiguration patterns
        self.security_misconfiguration_patterns = [
            r'DEBUG.*=.*True',  # Debug mode enabled
            r'ALLOWED_HOSTS.*=.*\[\]',  # Empty allowed hosts
            r'SECRET_KEY.*=.*["\'][^"\']{1,20}["\']',  # Short secret key
            r'USE_TLS.*=.*False',  # TLS disabled
            r'SECURE_SSL_REDIRECT.*=.*False',  # SSL redirect disabled
            r'SESSION_COOKIE_SECURE.*=.*False',  # Insecure session cookies
            r'CSRF_COOKIE_SECURE.*=.*False',  # Insecure CSRF cookies
            r'X_FRAME_OPTIONS.*=.*None',  # X-Frame-Options disabled
            r'SECURE_BROWSER_XSS_FILTER.*=.*False',  # XSS filter disabled
            r'default.*password',  # Default passwords
        ]
        
        # A06: Vulnerable Components patterns
        self.vulnerable_components_patterns = [
            r'requests.*==.*2\.[0-9]\.',  # Old requests version
            r'flask.*==.*0\.',  # Very old Flask
            r'django.*==.*1\.',  # Old Django
            r'urllib3.*==.*1\.[0-9]\.',  # Old urllib3
            r'pycryptodome.*<.*3\.9',  # Outdated crypto library
            r'lxml.*<.*4\.6',  # Vulnerable lxml
            r'pillow.*<.*8\.3',  # Vulnerable Pillow
        ]
        
        # A07: Identification and Authentication Failures
        self.identification_failures_patterns = [
            r'password.*=.*["\'][^"\']{1,6}["\']',  # Very short passwords
            r'login.*attempt.*unlimited',  # Unlimited login attempts
            r'session.*timeout.*=.*None',  # No session timeout
            r'remember.*me.*forever',  # Permanent sessions
            r'password.*reset.*no.*verification',  # Insecure password reset
            r'multi.*factor.*disabled',  # MFA disabled
            r'captcha.*disabled',  # CAPTCHA disabled
        ]
        
        # A08: Software and Data Integrity Failures
        self.integrity_failures_patterns = [
            r'pickle\.load\s*\(',  # Unsafe deserialization
            r'pickle\.loads\s*\(',  # Unsafe deserialization
            r'yaml\.load\s*\(',  # Unsafe YAML loading
            r'eval\s*\(',  # Code execution
            r'exec\s*\(',  # Code execution
            r'auto.*update.*no.*signature',  # Unsigned auto-updates
            r'download.*no.*checksum',  # No integrity check
        ]
        
        # A09: Security Logging and Monitoring Failures
        self.logging_failures_patterns = [
            r'logging.*disabled',  # Logging disabled
            r'log.*level.*=.*DEBUG.*production',  # Debug logging in prod
            r'password.*in.*log',  # Passwords in logs
            r'secret.*in.*log',  # Secrets in logs
            r'exception.*pass',  # Silent exception handling
            r'try:.*except:.*pass',  # Silent exception catching
            r'audit.*disabled',  # Audit logging disabled
        ]
        
        # A10: Server-Side Request Forgery patterns
        self.ssrf_patterns = [
            r'requests\.get\s*\(\s*user',  # User-controlled URL
            r'urllib\.request\.urlopen\s*\(\s*user',  # User-controlled URL
            r'httpx\.(get|post)\s*\(\s*user',  # User-controlled URL with httpx
            r'fetch.*url.*from.*user',  # User-controlled fetch
            r'proxy.*=.*user',  # User-controlled proxy
            r'redirect.*url.*from.*request',  # Open redirect
        ]
        
        # Authentication bypass patterns
        self.auth_bypass_patterns = [
            r'if.*password.*==.*["\'].*["\']',  # Hardcoded password checks
            r'auth.*=.*True',  # Bypassing authentication
            r'login.*without.*password',
            r'admin.*=.*True.*without',
        ]
        
        # File inclusion patterns
        self.file_inclusion_patterns = [
            r'open\s*\(\s*.*\+',  # Path traversal in file operations
            r'include\s*\(\s*.*\+',
            r'require\s*\(\s*.*\+',
            r'file_get_contents\s*\(\s*.*\+',
        ]
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive security analysis."""
        print("[INFO] Analyzing Security Vulnerabilities...")
        
        results = {
            "vulnerability_patterns": self._detect_vulnerability_patterns(),
            "input_validation": self._analyze_input_validation(),
            "authentication_security": self._analyze_authentication(),
            "authorization_security": self._analyze_authorization(), 
            "cryptography_usage": self._analyze_cryptography_usage(),
            "injection_vulnerabilities": self._analyze_injection_vulnerabilities(),
            "security_hotspots": self._identify_security_hotspots(),
            "security_metrics": self._calculate_security_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} security categories")
        return results
    
    def _detect_vulnerability_patterns(self) -> List[Dict[str, Any]]:
        """Detect known vulnerability patterns in code."""
        vulnerabilities = []
        vuln_id = 1
        
        # OWASP Top 10 2021 integrated pattern categories
        pattern_categories = {
            # A01: Broken Access Control
            'broken_access_control': self.access_control_patterns,
            # A02: Cryptographic Failures  
            'cryptographic_failures': self.cryptographic_failure_patterns,
            # A03: Injection
            'sql_injection': self.sql_injection_patterns,
            'xss': self.xss_patterns,
            'command_injection': self.command_injection_patterns,
            # A04: Insecure Design
            'insecure_design': self.insecure_design_patterns,
            # A05: Security Misconfiguration
            'security_misconfiguration': self.security_misconfiguration_patterns,
            # A06: Vulnerable Components
            'vulnerable_components': self.vulnerable_components_patterns,
            # A07: Identification and Authentication Failures
            'identification_failures': self.identification_failures_patterns,
            # A08: Software and Data Integrity Failures
            'integrity_failures': self.integrity_failures_patterns,
            # A09: Security Logging and Monitoring Failures
            'logging_failures': self.logging_failures_patterns,
            # A10: Server-Side Request Forgery
            'ssrf': self.ssrf_patterns,
            # Legacy patterns (maintained for backward compatibility)
            'auth_bypass': self.auth_bypass_patterns,
            'file_inclusion': self.file_inclusion_patterns
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for category, patterns in pattern_categories.items():
                    for pattern in patterns:
                        for line_num, line in enumerate(lines, 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                vulnerabilities.append({
                                    'vuln_id': vuln_id,
                                    'type': category,
                                    'severity': self._get_severity(category),
                                    'file': file_key,
                                    'line': line_num,
                                    'code': line.strip(),
                                    'pattern': pattern,
                                    'description': self._get_vulnerability_description(category),
                                    'recommendation': self._get_vulnerability_recommendation(category)
                                })
                                vuln_id += 1
                                
            except Exception:
                continue
        
        return vulnerabilities
    
    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type based on OWASP Top 10 2021."""
        critical_severity = [
            'broken_access_control',  # A01: Most common and severe
            'sql_injection',          # A03: High impact injection
            'command_injection'       # A03: High impact injection
        ]
        
        high_severity = [
            'cryptographic_failures', # A02: Crypto issues
            'insecure_design',        # A04: Design flaws
            'identification_failures', # A07: Auth failures
            'integrity_failures',     # A08: Data integrity
            'ssrf',                   # A10: Network security
            'auth_bypass'             # Legacy: Authentication bypass
        ]
        
        medium_severity = [
            'security_misconfiguration', # A05: Config issues
            'vulnerable_components',      # A06: Supply chain
            'logging_failures',          # A09: Monitoring
            'xss',                       # A03: Cross-site scripting
            'file_inclusion'             # Legacy: File inclusion
        ]
        
        if vuln_type in critical_severity:
            return 'CRITICAL'
        elif vuln_type in high_severity:
            return 'HIGH'
        elif vuln_type in medium_severity:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type based on OWASP Top 10 2021."""
        descriptions = {
            # OWASP Top 10 2021 Categories
            'broken_access_control': 'A01: Broken Access Control - Failures in access control restrictions',
            'cryptographic_failures': 'A02: Cryptographic Failures - Weak or missing cryptographic protection',
            'sql_injection': 'A03: Injection - SQL injection vulnerability from unsafe query construction',
            'xss': 'A03: Injection - Cross-site scripting vulnerability from unsafe output rendering',
            'command_injection': 'A03: Injection - Command injection risk from unsafe system execution',
            'insecure_design': 'A04: Insecure Design - Missing or ineffective security design controls',
            'security_misconfiguration': 'A05: Security Misconfiguration - Insecure configuration settings',
            'vulnerable_components': 'A06: Vulnerable Components - Use of outdated or vulnerable dependencies',
            'identification_failures': 'A07: Identification Failures - Weak authentication or session management',
            'integrity_failures': 'A08: Integrity Failures - Software and data integrity violations',
            'logging_failures': 'A09: Logging Failures - Insufficient security logging and monitoring',
            'ssrf': 'A10: Server-Side Request Forgery - Unvalidated server-side URL fetching',
            # Legacy patterns (backward compatibility)
            'weak_cryptography': 'Use of weak or deprecated cryptographic algorithms',
            'auth_bypass': 'Authentication bypass vulnerability or hardcoded credentials',
            'file_inclusion': 'File inclusion vulnerability from unsafe file operations'
        }
        return descriptions.get(vuln_type, f'Unknown vulnerability type: {vuln_type}')
    
    def _get_vulnerability_recommendation(self, vuln_type: str) -> str:
        """Get recommendation for vulnerability type based on OWASP Top 10 2021."""
        recommendations = {
            # OWASP Top 10 2021 Recommendations
            'broken_access_control': 'Implement proper access control checks, deny by default, use centralized authorization',
            'cryptographic_failures': 'Use strong algorithms (AES-256, SHA-256), secure key management, and TLS encryption',
            'sql_injection': 'Use parameterized queries, ORM methods, and input validation instead of string concatenation',
            'xss': 'Use proper output encoding/escaping, Content Security Policy, and input sanitization',
            'command_injection': 'Use subprocess with shell=False, input validation, and avoid system calls',
            'insecure_design': 'Implement security by design, threat modeling, and secure development lifecycle',
            'security_misconfiguration': 'Implement secure defaults, disable debug mode, keep systems updated',
            'vulnerable_components': 'Keep dependencies updated, use vulnerability scanning, and dependency management',
            'identification_failures': 'Implement strong authentication, session management, and multi-factor authentication',
            'integrity_failures': 'Use digital signatures, secure CI/CD, and integrity verification for data/software',
            'logging_failures': 'Implement comprehensive logging, monitoring, and incident response procedures',
            'ssrf': 'Validate and sanitize URLs, use allowlists, and implement network segmentation',
            # Legacy patterns (backward compatibility)
            'weak_cryptography': 'Use strong algorithms like AES-256, SHA-256, or bcrypt',
            'auth_bypass': 'Remove hardcoded credentials and implement proper authentication',
            'file_inclusion': 'Validate and sanitize file paths, use whitelist approach'
        }
        return recommendations.get(vuln_type, f'Review code for security best practices related to {vuln_type}')
    
    def _analyze_input_validation(self) -> Dict[str, Any]:
        """Analyze input validation patterns."""
        validation_data = {
            'functions_with_validation': [],
            'functions_without_validation': [],
            'validation_patterns': defaultdict(int),
            'user_input_sources': []
        }
        
        # Common validation patterns
        validation_patterns = [
            r'if.*len\(',  # Length validation
            r'if.*isinstance\(',  # Type validation
            r'if.*re\.match\(',  # Regex validation
            r'if.*in\s+\[',  # Whitelist validation
            r'raise.*ValueError',  # Input validation errors
            r'assert\s+',  # Assertion-based validation
        ]
        
        # User input sources
        input_sources = [
            'request.form', 'request.args', 'request.json',
            'input(', 'sys.argv', 'os.environ',
            'request.files', 'request.data'
        ]
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_content = content.split('\n')[node.lineno-1:node.end_lineno or node.lineno+10]
                        func_text = '\n'.join(func_content)
                        
                        # Check for validation patterns
                        has_validation = False
                        for pattern in validation_patterns:
                            if re.search(pattern, func_text, re.IGNORECASE):
                                has_validation = True
                                validation_data['validation_patterns'][pattern] += 1
                        
                        # Check for user input usage
                        has_user_input = False
                        for input_source in input_sources:
                            if input_source in func_text:
                                has_user_input = True
                                validation_data['user_input_sources'].append({
                                    'file': file_key,
                                    'function': node.name,
                                    'line': node.lineno,
                                    'input_source': input_source
                                })
                        
                        # Categorize function
                        if has_user_input:
                            if has_validation:
                                validation_data['functions_with_validation'].append({
                                    'file': file_key,
                                    'function': node.name,
                                    'line': node.lineno
                                })
                            else:
                                validation_data['functions_without_validation'].append({
                                    'file': file_key,
                                    'function': node.name,
                                    'line': node.lineno,
                                    'risk': 'HIGH'
                                })
                
            except Exception:
                continue
        
        # Calculate metrics
        total_input_functions = len(validation_data['functions_with_validation']) + len(validation_data['functions_without_validation'])
        validation_ratio = len(validation_data['functions_with_validation']) / max(total_input_functions, 1)
        
        validation_data['summary'] = {
            'total_input_functions': total_input_functions,
            'validated_functions': len(validation_data['functions_with_validation']),
            'unvalidated_functions': len(validation_data['functions_without_validation']),
            'validation_ratio': validation_ratio,
            'validation_coverage': 'GOOD' if validation_ratio > 0.8 else 'MEDIUM' if validation_ratio > 0.5 else 'POOR'
        }
        
        return validation_data
    
    def _analyze_authentication(self) -> Dict[str, Any]:
        """Analyze authentication security patterns."""
        auth_data = {
            'auth_mechanisms': [],
            'password_handling': [],
            'session_management': [],
            'weak_auth_patterns': []
        }
        
        # Authentication patterns
        auth_patterns = [
            r'@login_required',  # Flask login decorator
            r'@require_auth',
            r'authenticate\(',
            r'check_password\(',
            r'verify_password\(',
            r'bcrypt\.check',  # Bcrypt password checking
            r'pbkdf2\(',  # PBKDF2 key derivation
        ]
        
        # Weak authentication patterns
        weak_patterns = [
            r'password.*==.*["\']',  # Plain text password comparison
            r'md5\(.*password',  # MD5 for passwords
            r'sha1\(.*password',  # SHA1 for passwords
            r'base64\.encode.*password',  # Base64 encoding passwords
        ]
        
        # Session patterns
        session_patterns = [
            r'session\[',  # Session usage
            r'cookie\[',  # Cookie usage
            r'jwt\.encode\(',  # JWT tokens
            r'secure=True',  # Secure cookies
            r'httponly=True',  # HTTP-only cookies
        ]
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                # Check for authentication mechanisms
                for line_num, line in enumerate(lines, 1):
                    for pattern in auth_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            auth_data['auth_mechanisms'].append({
                                'file': file_key,
                                'line': line_num,
                                'pattern': pattern,
                                'code': line.strip()
                            })
                    
                    for pattern in weak_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            auth_data['weak_auth_patterns'].append({
                                'file': file_key,
                                'line': line_num,
                                'pattern': pattern,
                                'code': line.strip(),
                                'severity': 'HIGH'
                            })
                    
                    for pattern in session_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            auth_data['session_management'].append({
                                'file': file_key,
                                'line': line_num,
                                'pattern': pattern,
                                'code': line.strip()
                            })
                
            except Exception:
                continue
        
        auth_data['summary'] = {
            'auth_mechanisms_found': len(auth_data['auth_mechanisms']),
            'weak_auth_patterns': len(auth_data['weak_auth_patterns']),
            'session_management_usage': len(auth_data['session_management']),
            'auth_security_score': max(0, 100 - (len(auth_data['weak_auth_patterns']) * 20))
        }
        
        return auth_data
    
    def _analyze_authorization(self) -> Dict[str, Any]:
        """Analyze authorization security patterns."""
        authz_data = {
            'permission_checks': [],
            'role_based_access': [],
            'missing_authz': [],
            'privilege_escalation_risks': []
        }
        
        # Authorization patterns
        authz_patterns = [
            r'@require_permission',
            r'@require_role',
            r'check_permission\(',
            r'has_permission\(',
            r'user\.can\(',
            r'authorize\(',
            r'if.*user\.role',
            r'if.*current_user\.is_admin'
        ]
        
        # Privileged operations (should have authorization)
        privileged_operations = [
            r'delete\(',
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'admin',
            r'sudo',
            r'execute\(',
            r'system\('
        ]
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Analyze functions for authorization
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_content = content.split('\n')[node.lineno-1:node.end_lineno or node.lineno+10]
                        func_text = '\n'.join(func_content)
                        
                        has_authz_check = False
                        has_privileged_op = False
                        
                        # Check for authorization patterns
                        for pattern in authz_patterns:
                            if re.search(pattern, func_text, re.IGNORECASE):
                                has_authz_check = True
                                authz_data['permission_checks'].append({
                                    'file': file_key,
                                    'function': node.name,
                                    'line': node.lineno,
                                    'pattern': pattern
                                })
                        
                        # Check for privileged operations
                        for pattern in privileged_operations:
                            if re.search(pattern, func_text, re.IGNORECASE):
                                has_privileged_op = True
                        
                        # Flag functions with privileged operations but no authorization
                        if has_privileged_op and not has_authz_check:
                            authz_data['missing_authz'].append({
                                'file': file_key,
                                'function': node.name,
                                'line': node.lineno,
                                'risk': 'HIGH',
                                'reason': 'Privileged operation without authorization check'
                            })
                
            except Exception:
                continue
        
        authz_data['summary'] = {
            'functions_with_authz': len(authz_data['permission_checks']),
            'functions_missing_authz': len(authz_data['missing_authz']),
            'authz_coverage_ratio': len(authz_data['permission_checks']) / max(
                len(authz_data['permission_checks']) + len(authz_data['missing_authz']), 1
            )
        }
        
        return authz_data
    
    def _analyze_cryptography_usage(self) -> Dict[str, Any]:
        """Analyze cryptographic usage patterns."""
        crypto_data = {
            'strong_crypto': [],
            'weak_crypto': [],
            'crypto_libraries': [],
            'key_management': []
        }
        
        # Strong cryptography patterns
        strong_patterns = [
            r'AES\.new\(',  # AES encryption
            r'RSA\.generate\(',  # RSA key generation
            r'bcrypt\.',  # Bcrypt for passwords
            r'scrypt\(',  # Scrypt key derivation
            r'PBKDF2\(',  # PBKDF2 key derivation
            r'sha256\(',  # SHA-256
            r'sha512\(',  # SHA-512
            r'secrets\.',  # Python secrets module
        ]
        
        # Crypto libraries
        crypto_libraries = [
            'cryptography', 'pycrypto', 'pycryptodome',
            'bcrypt', 'passlib', 'hashlib', 'secrets'
        ]
        
        # Key management patterns
        key_patterns = [
            r'private_key',
            r'public_key', 
            r'secret_key',
            r'api_key',
            r'token',
            r'certificate'
        ]
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                # Check for import statements
                for line_num, line in enumerate(lines, 1):
                    if line.strip().startswith(('import ', 'from ')):
                        for lib in crypto_libraries:
                            if lib in line:
                                crypto_data['crypto_libraries'].append({
                                    'file': file_key,
                                    'line': line_num,
                                    'library': lib,
                                    'import_statement': line.strip()
                                })
                    
                    # Check for strong crypto usage
                    for pattern in strong_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            crypto_data['strong_crypto'].append({
                                'file': file_key,
                                'line': line_num,
                                'pattern': pattern,
                                'code': line.strip()
                            })
                    
                    # Check for weak crypto (from cryptographic failure patterns)
                    for pattern in self.cryptographic_failure_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            crypto_data['weak_crypto'].append({
                                'file': file_key,
                                'line': line_num,
                                'pattern': pattern,
                                'code': line.strip(),
                                'severity': 'MEDIUM'
                            })
                    
                    # Check for key management
                    for pattern in key_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Check if it's hardcoded (basic heuristic)
                            if '=' in line and ('"' in line or "'" in line):
                                crypto_data['key_management'].append({
                                    'file': file_key,
                                    'line': line_num,
                                    'pattern': pattern,
                                    'code': line.strip(),
                                    'risk': 'Potential hardcoded secret'
                                })
                
            except Exception:
                continue
        
        crypto_data['summary'] = {
            'crypto_libraries_used': len(set(lib['library'] for lib in crypto_data['crypto_libraries'])),
            'strong_crypto_usage': len(crypto_data['strong_crypto']),
            'weak_crypto_usage': len(crypto_data['weak_crypto']),
            'key_management_issues': len(crypto_data['key_management']),
            'crypto_security_score': max(0, 100 - (len(crypto_data['weak_crypto']) * 15) - (len(crypto_data['key_management']) * 10))
        }
        
        return crypto_data
    
    def _analyze_injection_vulnerabilities(self) -> Dict[str, Any]:
        """Analyze injection vulnerability patterns."""
        injection_data = {
            'sql_injection_risks': [],
            'xss_risks': [],
            'command_injection_risks': [],
            'ldap_injection_risks': [],
            'xpath_injection_risks': []
        }
        
        # Additional injection patterns
        ldap_patterns = [
            r'ldap.*search.*\+',
            r'ldap.*filter.*%'
        ]
        
        xpath_patterns = [
            r'xpath.*\+',
            r'evaluate.*xpath.*\+'
        ]
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for line_num, line in enumerate(lines, 1):
                    # SQL injection
                    for pattern in self.sql_injection_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            injection_data['sql_injection_risks'].append({
                                'file': file_key,
                                'line': line_num,
                                'code': line.strip(),
                                'severity': 'HIGH'
                            })
                    
                    # XSS
                    for pattern in self.xss_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            injection_data['xss_risks'].append({
                                'file': file_key,
                                'line': line_num,
                                'code': line.strip(),
                                'severity': 'MEDIUM'
                            })
                    
                    # Command injection
                    for pattern in self.command_injection_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            injection_data['command_injection_risks'].append({
                                'file': file_key,
                                'line': line_num,
                                'code': line.strip(),
                                'severity': 'HIGH'
                            })
                    
                    # LDAP injection
                    for pattern in ldap_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            injection_data['ldap_injection_risks'].append({
                                'file': file_key,
                                'line': line_num,
                                'code': line.strip(),
                                'severity': 'MEDIUM'
                            })
                    
                    # XPath injection
                    for pattern in xpath_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            injection_data['xpath_injection_risks'].append({
                                'file': file_key,
                                'line': line_num,
                                'code': line.strip(),
                                'severity': 'MEDIUM'
                            })
                
            except Exception:
                continue
        
        total_risks = sum(len(risks) for risks in injection_data.values())
        injection_data['summary'] = {
            'total_injection_risks': total_risks,
            'high_severity_count': len(injection_data['sql_injection_risks']) + len(injection_data['command_injection_risks']),
            'medium_severity_count': len(injection_data['xss_risks']) + len(injection_data['ldap_injection_risks']) + len(injection_data['xpath_injection_risks'])
        }
        
        return injection_data
    
    def _identify_security_hotspots(self) -> List[Dict[str, Any]]:
        """Identify security hotspots - files/functions with multiple security issues."""
        hotspots = []
        file_issue_counts = defaultdict(int)
        
        # Count issues per file from previous analyses
        vulnerabilities = self._detect_vulnerability_patterns()
        for vuln in vulnerabilities:
            file_issue_counts[vuln['file']] += 1
        
        # Identify hotspots (files with multiple issues)
        hotspot_id = 1
        for file_path, issue_count in file_issue_counts.items():
            if issue_count >= 3:  # Threshold for hotspot
                hotspots.append({
                    'hotspot_id': hotspot_id,
                    'file': file_path,
                    'issue_count': issue_count,
                    'risk_level': 'CRITICAL' if issue_count >= 10 else 'HIGH' if issue_count >= 5 else 'MEDIUM',
                    'priority': 'IMMEDIATE' if issue_count >= 10 else 'HIGH' if issue_count >= 5 else 'MEDIUM',
                    'recommendation': 'Comprehensive security review required'
                })
                hotspot_id += 1
        
        return sorted(hotspots, key=lambda x: x['issue_count'], reverse=True)
    
    def _calculate_security_metrics(self) -> Dict[str, Any]:
        """Calculate overall security metrics."""
        vulnerabilities = self._detect_vulnerability_patterns()
        hotspots = self._identify_security_hotspots()
        
        # Count by severity
        severity_counts = Counter(vuln['severity'] for vuln in vulnerabilities)
        
        # Calculate security score (0-100) including CRITICAL severity
        total_files = len(list(self._get_python_files()))
        security_score = max(0, 100 - 
                           (severity_counts['CRITICAL'] * 30) -  # CRITICAL issues have highest impact
                           (severity_counts['HIGH'] * 20) - 
                           (severity_counts['MEDIUM'] * 10) - 
                           (severity_counts['LOW'] * 5) -
                           (len(hotspots) * 15))
        
        return {
            'total_vulnerabilities': len(vulnerabilities),
            'critical_severity': severity_counts['CRITICAL'],
            'high_severity': severity_counts['HIGH'],
            'medium_severity': severity_counts['MEDIUM'],
            'low_severity': severity_counts['LOW'],
            'security_hotspots': len(hotspots),
            'files_analyzed': total_files,
            'security_score': security_score,
            'security_grade': self._get_security_grade(security_score),
            'vulnerabilities_per_file': len(vulnerabilities) / max(total_files, 1),
            'critical_issues': severity_counts['CRITICAL'] + severity_counts['HIGH'] + len(hotspots),
            'owasp_top_10_coverage': {
                'A01_broken_access_control': len([v for v in vulnerabilities if v['type'] == 'broken_access_control']),
                'A02_cryptographic_failures': len([v for v in vulnerabilities if v['type'] == 'cryptographic_failures']),
                'A03_injection': len([v for v in vulnerabilities if v['type'] in ['sql_injection', 'xss', 'command_injection']]),
                'A04_insecure_design': len([v for v in vulnerabilities if v['type'] == 'insecure_design']),
                'A05_security_misconfiguration': len([v for v in vulnerabilities if v['type'] == 'security_misconfiguration']),
                'A06_vulnerable_components': len([v for v in vulnerabilities if v['type'] == 'vulnerable_components']),
                'A07_identification_failures': len([v for v in vulnerabilities if v['type'] == 'identification_failures']),
                'A08_integrity_failures': len([v for v in vulnerabilities if v['type'] == 'integrity_failures']),
                'A09_logging_failures': len([v for v in vulnerabilities if v['type'] == 'logging_failures']),
                'A10_ssrf': len([v for v in vulnerabilities if v['type'] == 'ssrf'])
            }
        }
    
    def _get_security_grade(self, score: float) -> str:
        """Get security grade based on score with OWASP Top 10 considerations."""
        if score >= 95:
            return 'A+'  # Excellent security posture
        elif score >= 90:
            return 'A'   # Very good security
        elif score >= 80:
            return 'B'   # Good security
        elif score >= 70:
            return 'C'   # Acceptable security
        elif score >= 60:
            return 'D'   # Poor security - needs improvement
        elif score >= 40:
            return 'F'   # Failing security - immediate action required
        else:
            return 'F-'  # Critical security failures - emergency response needed