"""
Predictive Security Analyzer
===========================

Revolutionary predictive security vulnerability analysis system.
Extracted from predictive_code_intelligence.py for enterprise modular architecture.

Agent D Implementation - Hour 16-17: Predictive Intelligence Modularization
"""

import ast
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

from .data_models import CodePrediction, PredictionType, PredictionConfidence, SecurityRisk


class SecurityAnalyzer:
    """
    Revolutionary Predictive Security Analyzer
    
    Analyzes code for security vulnerabilities with predictive capabilities,
    identifying potential threats before they become exploitable.
    """
    
    def __init__(self):
        self.vulnerability_patterns = {
            'code_injection': {
                'patterns': [
                    r'SafeCodeExecutor\.safe_eval\s*\(',
                    r'SafeCodeExecutor\.safe_exec\s*\(',
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'compile\s*\(',
                    r'__import__\s*\('
                ],
                'severity': 'critical',
                'description': 'Dynamic code execution vulnerability'
            },
            'sql_injection': {
                'patterns': [
                    r'\.execute\s*\([^)]*%[sd]',
                    r'\.execute\s*\([^)]*\+',
                    r'\.execute\s*\([^)]*\.format\s*\(',
                    r'SELECT\s+.*\+.*FROM',
                    r'INSERT\s+.*\+.*VALUES',
                    r'UPDATE\s+.*\+.*SET',
                    r'DELETE\s+.*\+.*WHERE'
                ],
                'severity': 'high',
                'description': 'SQL injection vulnerability'
            },
            'command_injection': {
                'patterns': [
                    r'subprocess\.call\s*\([^)]*\+',
                    r'subprocess\.run\s*\([^)]*\+',
                    r'os\.system\s*\([^)]*\+',
                    r'os\.popen\s*\([^)]*\+',
                    r'subprocess\.Popen\s*\([^)]*\+',
                    r'shell=True'
                ],
                'severity': 'high',
                'description': 'Command injection vulnerability'
            },
            'path_traversal': {
                'patterns': [
                    r'open\s*\([^)]*\+.*\)',
                    r'file\s*\([^)]*\+.*\)',
                    r'\.\./',
                    r'\.\.\\\\'
                ],
                'severity': 'medium',
                'description': 'Path traversal vulnerability'
            },
            'hardcoded_secrets': {
                'patterns': [
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'secret\s*=\s*["\'][^"\']+["\']',
                    r'api_key\s*=\s*["\'][^"\']+["\']',
                    r'token\s*=\s*["\'][^"\']+["\']',
                    r'key\s*=\s*["\'][A-Za-z0-9]{20,}["\']'
                ],
                'severity': 'medium',
                'description': 'Hardcoded credentials or secrets'
            },
            'weak_crypto': {
                'patterns': [
                    r'hashlib\.md5\s*\(',
                    r'hashlib\.sha1\s*\(',
                    r'Crypto\.Hash\.MD5',
                    r'Crypto\.Hash\.SHA1',
                    r'random\.random\s*\('
                ],
                'severity': 'medium',
                'description': 'Weak cryptographic practices'
            },
            'deserialization': {
                'patterns': [
                    r'pickle\.loads\s*\(',
                    r'pickle\.load\s*\(',
                    r'yaml\.load\s*\(',
                    r'marshal\.loads\s*\('
                ],
                'severity': 'high',
                'description': 'Unsafe deserialization vulnerability'
            },
            'xss_potential': {
                'patterns': [
                    r'\.innerHTML\s*=',
                    r'document\.write\s*\(',
                    r'\.html\s*\([^)]*\+',
                    r'render_template_string\s*\('
                ],
                'severity': 'medium',
                'description': 'Cross-site scripting (XSS) potential'
            },
            'csrf_missing': {
                'patterns': [
                    r'@app\.route.*methods.*POST',
                    r'request\.form',
                    r'request\.json'
                ],
                'severity': 'medium',
                'description': 'Potential CSRF vulnerability (missing protection)'
            },
            'insecure_random': {
                'patterns': [
                    r'random\.randint\s*\(',
                    r'random\.choice\s*\(',
                    r'random\.uniform\s*\(',
                    r'Math\.random\s*\('
                ],
                'severity': 'low',
                'description': 'Use of predictable random number generator'
            },
            'debug_enabled': {
                'patterns': [
                    r'debug\s*=\s*True',
                    r'DEBUG\s*=\s*True',
                    r'app\.debug\s*=\s*True'
                ],
                'severity': 'medium',
                'description': 'Debug mode enabled in production'
            }
        }
        
        self.security_context_patterns = {
            'authentication': ['auth', 'login', 'password', 'credential', 'token'],
            'authorization': ['permission', 'role', 'access', 'privilege', 'authorize'],
            'encryption': ['encrypt', 'decrypt', 'cipher', 'crypto', 'hash'],
            'input_validation': ['validate', 'sanitize', 'escape', 'filter'],
            'session_management': ['session', 'cookie', 'csrf', 'xsrf'],
            'data_protection': ['sensitive', 'personal', 'private', 'confidential'],
            'network_security': ['ssl', 'tls', 'https', 'certificate'],
            'error_handling': ['try', 'except', 'error', 'exception']
        }
        
        self.severity_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'info': 0.2
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_security_vulnerabilities(self, code: str, file_path: str) -> List[CodePrediction]:
        """Analyze code for security vulnerabilities and predict future risks"""
        
        try:
            predictions = []
            
            # Direct vulnerability detection
            direct_vulns = await self._detect_direct_vulnerabilities(code, file_path)
            predictions.extend(direct_vulns)
            
            # Security anti-pattern detection
            antipatterns = await self._detect_security_antipatterns(code, file_path)
            predictions.extend(antipatterns)
            
            # Contextual security analysis
            contextual_risks = await self._analyze_security_context(code, file_path)
            predictions.extend(contextual_risks)
            
            # Predictive security modeling
            predictive_risks = await self._predict_future_vulnerabilities(code, file_path)
            predictions.extend(predictive_risks)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error analyzing security vulnerabilities: {e}")
            return []
    
    async def _detect_direct_vulnerabilities(self, code: str, file_path: str) -> List[CodePrediction]:
        """Detect direct security vulnerabilities in code"""
        
        predictions = []
        
        for vuln_type, config in self.vulnerability_patterns.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    # Calculate line number
                    line_num = code[:match.start()].count('\n') + 1
                    
                    # Extract context around the match
                    lines = code.split('\n')
                    context_start = max(0, line_num - 3)
                    context_end = min(len(lines), line_num + 2)
                    context = '\n'.join(lines[context_start:context_end])
                    
                    # Create security prediction
                    prediction = CodePrediction(
                        prediction_type=PredictionType.SECURITY_VULNERABILITIES,
                        target_file=file_path,
                        target_element=f"line_{line_num}",
                        prediction_summary=f"Security vulnerability detected: {config['description']}",
                        detailed_analysis=f"Detected {vuln_type} vulnerability at line {line_num}. "
                                        f"Pattern matched: {match.group()}. "
                                        f"This could allow attackers to {self._get_attack_description(vuln_type)}.",
                        confidence=self._get_confidence_from_severity(config['severity']),
                        probability_score=self.severity_scores[config['severity']],
                        timeline_estimate="Immediate attention required",
                        impact_assessment={
                            'security_impact': config['severity'],
                            'data_exposure_risk': 'high' if vuln_type in ['sql_injection', 'code_injection'] else 'medium',
                            'system_compromise_risk': 'high' if vuln_type in ['code_injection', 'command_injection'] else 'medium',
                            'compliance_impact': 'violation_likely'
                        },
                        recommended_actions=self._get_remediation_actions(vuln_type),
                        prevention_strategies=self._get_prevention_strategies(vuln_type),
                        evidence_factors=[
                            f"Pattern detected: {pattern}",
                            f"Match found: {match.group()}",
                            f"Line number: {line_num}",
                            f"Context: {context[:100]}..."
                        ],
                        monitoring_indicators=[
                            "Code review required",
                            "Security testing needed",
                            "Penetration testing recommended"
                        ]
                    )
                    
                    predictions.append(prediction)
        
        return predictions
    
    async def _detect_security_antipatterns(self, code: str, file_path: str) -> List[CodePrediction]:
        """Detect security anti-patterns that indicate poor security practices"""
        
        predictions = []
        
        # Parse AST for structural analysis
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return predictions
        
        # Check for missing input validation
        user_input_nodes = self._find_user_input_nodes(tree)
        validation_nodes = self._find_validation_nodes(tree)
        
        if user_input_nodes and not validation_nodes:
            prediction = CodePrediction(
                prediction_type=PredictionType.SECURITY_VULNERABILITIES,
                target_file=file_path,
                target_element="input_validation",
                prediction_summary="Missing input validation detected",
                detailed_analysis="User input is processed without proper validation. "
                                "This creates potential for injection attacks and data corruption.",
                confidence=PredictionConfidence.MEDIUM,
                probability_score=0.7,
                timeline_estimate="Should be addressed soon",
                impact_assessment={
                    'security_impact': 'medium',
                    'data_integrity_risk': 'medium',
                    'injection_attack_risk': 'high'
                },
                recommended_actions=[
                    "Implement input validation for all user inputs",
                    "Use parameterized queries for database operations",
                    "Sanitize and escape user data before processing",
                    "Implement input length and format restrictions"
                ],
                prevention_strategies=[
                    "Adopt secure coding practices",
                    "Use validation libraries and frameworks",
                    "Implement defense in depth"
                ]
            )
            predictions.append(prediction)
        
        # Check for missing error handling in security-critical areas
        security_operations = self._find_security_operations(tree)
        try_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        
        if security_operations and len(try_blocks) < len(security_operations) / 2:
            prediction = CodePrediction(
                prediction_type=PredictionType.SECURITY_VULNERABILITIES,
                target_file=file_path,
                target_element="error_handling",
                prediction_summary="Insufficient error handling in security operations",
                detailed_analysis="Security-critical operations lack proper error handling. "
                                "This can lead to information disclosure and system instability.",
                confidence=PredictionConfidence.MEDIUM,
                probability_score=0.6,
                timeline_estimate="Should be addressed",
                impact_assessment={
                    'information_disclosure_risk': 'medium',
                    'system_stability_risk': 'medium',
                    'debugging_information_leak': 'high'
                },
                recommended_actions=[
                    "Add comprehensive error handling to security operations",
                    "Implement proper logging without exposing sensitive data",
                    "Use generic error messages for user-facing errors",
                    "Implement monitoring for security-related errors"
                ]
            )
            predictions.append(prediction)
        
        # Check for overly permissive access patterns
        overly_permissive = self._detect_permissive_access(tree, code)
        if overly_permissive:
            for issue in overly_permissive:
                prediction = CodePrediction(
                    prediction_type=PredictionType.SECURITY_VULNERABILITIES,
                    target_file=file_path,
                    target_element=issue['element'],
                    prediction_summary=f"Overly permissive access detected: {issue['type']}",
                    detailed_analysis=issue['description'],
                    confidence=PredictionConfidence.MEDIUM,
                    probability_score=0.5,
                    timeline_estimate="Should be reviewed",
                    impact_assessment={
                        'access_control_risk': 'medium',
                        'privilege_escalation_risk': 'medium'
                    },
                    recommended_actions=issue['recommendations']
                )
                predictions.append(prediction)
        
        return predictions
    
    async def _analyze_security_context(self, code: str, file_path: str) -> List[CodePrediction]:
        """Analyze security context and identify contextual risks"""
        
        predictions = []
        
        # Determine security context
        security_contexts = self._identify_security_contexts(code)
        
        for context, confidence in security_contexts.items():
            if confidence > 0.6:
                # Generate context-specific security predictions
                context_predictions = await self._generate_context_predictions(
                    context, code, file_path, confidence
                )
                predictions.extend(context_predictions)
        
        return predictions
    
    async def _predict_future_vulnerabilities(self, code: str, file_path: str) -> List[CodePrediction]:
        """Predict future security vulnerabilities based on code patterns"""
        
        predictions = []
        
        # Analyze code evolution patterns that lead to vulnerabilities
        evolution_risks = self._analyze_vulnerability_evolution_patterns(code)
        
        for risk in evolution_risks:
            prediction = CodePrediction(
                prediction_type=PredictionType.SECURITY_VULNERABILITIES,
                target_file=file_path,
                target_element=risk['element'],
                prediction_summary=f"Future vulnerability risk: {risk['type']}",
                detailed_analysis=risk['description'],
                confidence=PredictionConfidence.MEDIUM,
                probability_score=risk['probability'],
                timeline_estimate=risk['timeline'],
                impact_assessment=risk['impact'],
                recommended_actions=risk['prevention_actions'],
                monitoring_indicators=risk['monitoring_indicators']
            )
            predictions.append(prediction)
        
        return predictions
    
    def _get_attack_description(self, vuln_type: str) -> str:
        """Get description of potential attacks for vulnerability type"""
        
        attack_descriptions = {
            'code_injection': 'execute arbitrary code on the server',
            'sql_injection': 'access, modify, or delete database information',
            'command_injection': 'execute system commands on the server',
            'path_traversal': 'access files outside the intended directory',
            'hardcoded_secrets': 'gain unauthorized access using exposed credentials',
            'weak_crypto': 'break encryption and access sensitive data',
            'deserialization': 'execute arbitrary code through malicious data',
            'xss_potential': 'inject malicious scripts into web pages',
            'csrf_missing': 'perform unauthorized actions on behalf of users',
            'insecure_random': 'predict random values and compromise security',
            'debug_enabled': 'access debugging information and system details'
        }
        
        return attack_descriptions.get(vuln_type, 'compromise system security')
    
    def _get_confidence_from_severity(self, severity: str) -> PredictionConfidence:
        """Convert severity to confidence level"""
        
        severity_confidence_map = {
            'critical': PredictionConfidence.VERY_HIGH,
            'high': PredictionConfidence.HIGH,
            'medium': PredictionConfidence.MEDIUM,
            'low': PredictionConfidence.LOW,
            'info': PredictionConfidence.LOW
        }
        
        return severity_confidence_map.get(severity, PredictionConfidence.MEDIUM)
    
    def _get_remediation_actions(self, vuln_type: str) -> List[str]:
        """Get specific remediation actions for vulnerability type"""
        
        remediation_map = {
            'code_injection': [
                "Remove or replace eval() and exec() calls",
                "Use safer alternatives like ast.literal_eval() for data parsing",
                "Implement input validation and sanitization",
                "Use parameterized queries and prepared statements",
                "Apply principle of least privilege"
            ],
            'sql_injection': [
                "Use parameterized queries or prepared statements",
                "Implement input validation and sanitization",
                "Use ORM frameworks with built-in protections",
                "Apply database access controls and permissions",
                "Regular security testing and code review"
            ],
            'command_injection': [
                "Avoid shell=True in subprocess calls",
                "Use subprocess with argument lists instead of strings",
                "Validate and sanitize all user inputs",
                "Use safer alternatives to os.system()",
                "Implement command whitelisting"
            ],
            'path_traversal': [
                "Validate and sanitize file paths",
                "Use os.path.join() for path construction",
                "Implement path canonicalization",
                "Restrict file access to designated directories",
                "Use absolute paths and avoid user-controlled paths"
            ],
            'hardcoded_secrets': [
                "Move credentials to environment variables",
                "Use secure credential management systems",
                "Implement proper secret rotation",
                "Use configuration files not in version control",
                "Apply encryption for stored credentials"
            ],
            'weak_crypto': [
                "Use strong hashing algorithms (SHA-256 or better)",
                "Implement proper salt usage for password hashing",
                "Use cryptographically secure random number generators",
                "Apply current cryptographic best practices",
                "Regular review of cryptographic implementations"
            ],
            'deserialization': [
                "Avoid deserializing untrusted data",
                "Use safe serialization formats like JSON",
                "Implement input validation before deserialization",
                "Use serialization libraries with security features",
                "Apply digital signatures to serialized data"
            ],
            'xss_potential': [
                "Implement proper output encoding",
                "Use templating engines with auto-escaping",
                "Validate and sanitize user inputs",
                "Implement Content Security Policy (CSP)",
                "Use secure coding practices for web applications"
            ],
            'csrf_missing': [
                "Implement CSRF tokens for state-changing operations",
                "Use SameSite cookie attributes",
                "Verify HTTP Referer headers",
                "Implement proper session management",
                "Use framework-provided CSRF protections"
            ],
            'insecure_random': [
                "Use cryptographically secure random generators",
                "Replace random.random() with secrets module",
                "Use os.urandom() for random bytes",
                "Implement proper seed management",
                "Review all random number usage for security implications"
            ],
            'debug_enabled': [
                "Disable debug mode in production",
                "Use environment-specific configuration",
                "Implement proper logging instead of debug output",
                "Remove debug statements from production code",
                "Use feature flags for debug functionality"
            ]
        }
        
        return remediation_map.get(vuln_type, [
            "Conduct security code review",
            "Implement security best practices",
            "Perform security testing",
            "Apply defense in depth principles"
        ])
    
    def _get_prevention_strategies(self, vuln_type: str) -> List[str]:
        """Get prevention strategies for vulnerability type"""
        
        prevention_map = {
            'code_injection': [
                "Secure coding training for developers",
                "Static code analysis tools",
                "Regular security code reviews",
                "Input validation frameworks"
            ],
            'sql_injection': [
                "Database security training",
                "ORM usage guidelines",
                "Automated security testing",
                "Database access controls"
            ],
            'command_injection': [
                "System interaction guidelines",
                "Security-focused code reviews",
                "Input validation standards",
                "Privilege minimization"
            ],
            'path_traversal': [
                "File handling security guidelines",
                "Path validation frameworks",
                "Directory access controls",
                "Security testing automation"
            ],
            'hardcoded_secrets': [
                "Secret management training",
                "Automated secret scanning",
                "Configuration management",
                "Version control security"
            ],
            'weak_crypto': [
                "Cryptography training",
                "Security library guidelines",
                "Regular cryptographic reviews",
                "Industry standard adoption"
            ]
        }
        
        return prevention_map.get(vuln_type, [
            "Security awareness training",
            "Secure development lifecycle",
            "Regular security assessments",
            "Threat modeling"
        ])
    
    def _find_user_input_nodes(self, tree: ast.AST) -> List[ast.AST]:
        """Find AST nodes that represent user input"""
        
        input_patterns = [
            'input', 'raw_input', 'request.form', 'request.args', 'request.json',
            'request.data', 'request.files', 'sys.argv', 'os.environ'
        ]
        
        input_nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ['input', 'raw_input']:
                    input_nodes.append(node)
                elif isinstance(node.func, ast.Attribute):
                    attr_name = self._get_full_attribute_name(node.func)
                    if any(pattern in attr_name for pattern in input_patterns):
                        input_nodes.append(node)
        
        return input_nodes
    
    def _find_validation_nodes(self, tree: ast.AST) -> List[ast.AST]:
        """Find AST nodes that represent input validation"""
        
        validation_patterns = [
            'validate', 'sanitize', 'escape', 'clean', 'filter', 'check'
        ]
        
        validation_nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if any(pattern in node.func.id.lower() for pattern in validation_patterns):
                        validation_nodes.append(node)
                elif isinstance(node.func, ast.Attribute):
                    if any(pattern in node.func.attr.lower() for pattern in validation_patterns):
                        validation_nodes.append(node)
        
        return validation_nodes
    
    def _find_security_operations(self, tree: ast.AST) -> List[ast.AST]:
        """Find AST nodes that represent security-critical operations"""
        
        security_patterns = [
            'auth', 'login', 'password', 'encrypt', 'decrypt', 'hash', 'token',
            'session', 'permission', 'access', 'credential'
        ]
        
        security_nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Call)):
                node_name = ""
                if isinstance(node, ast.FunctionDef):
                    node_name = node.name.lower()
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        node_name = node.func.id.lower()
                    elif isinstance(node.func, ast.Attribute):
                        node_name = node.func.attr.lower()
                
                if any(pattern in node_name for pattern in security_patterns):
                    security_nodes.append(node)
        
        return security_nodes
    
    def _detect_permissive_access(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Detect overly permissive access patterns"""
        
        issues = []
        
        # Check for overly broad exception handling
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:  # bare except:
                        issues.append({
                            'element': 'exception_handling',
                            'type': 'bare_except',
                            'description': 'Bare except clause catches all exceptions, potentially hiding security errors',
                            'recommendations': [
                                'Use specific exception types',
                                'Log exceptions appropriately',
                                'Handle security exceptions separately'
                            ]
                        })
        
        # Check for overly permissive file permissions
        if 'chmod' in code:
            chmod_matches = re.finditer(r'chmod\s*\(\s*[^,]+,\s*0o?777\s*\)', code)
            for match in chmod_matches:
                issues.append({
                    'element': 'file_permissions',
                    'type': 'overly_permissive_chmod',
                    'description': 'File permissions set to 777 (world-writable)',
                    'recommendations': [
                        'Use more restrictive file permissions',
                        'Follow principle of least privilege',
                        'Review file permission requirements'
                    ]
                })
        
        return issues
    
    def _identify_security_contexts(self, code: str) -> Dict[str, float]:
        """Identify security contexts present in the code"""
        
        context_scores = {}
        code_lower = code.lower()
        
        for context, keywords in self.security_context_patterns.items():
            score = 0.0
            for keyword in keywords:
                # Count occurrences with diminishing returns
                count = code_lower.count(keyword)
                score += min(count * 0.2, 1.0)
            
            # Normalize score
            context_scores[context] = min(score / len(keywords), 1.0)
        
        return context_scores
    
    async def _generate_context_predictions(self, context: str, code: str, 
                                          file_path: str, confidence: float) -> List[CodePrediction]:
        """Generate security predictions based on context"""
        
        predictions = []
        
        context_risks = {
            'authentication': {
                'risks': ['weak_password_policy', 'session_fixation', 'credential_stuffing'],
                'description': 'Authentication mechanisms may be vulnerable to various attacks',
                'timeline': '1-3 months'
            },
            'authorization': {
                'risks': ['privilege_escalation', 'access_control_bypass', 'role_confusion'],
                'description': 'Authorization controls may allow unauthorized access',
                'timeline': '1-2 months'
            },
            'encryption': {
                'risks': ['weak_encryption', 'key_management_issues', 'side_channel_attacks'],
                'description': 'Cryptographic implementations may have weaknesses',
                'timeline': '2-6 months'
            },
            'input_validation': {
                'risks': ['injection_attacks', 'data_corruption', 'business_logic_bypass'],
                'description': 'Input validation gaps may lead to security vulnerabilities',
                'timeline': 'Immediate to 1 month'
            },
            'session_management': {
                'risks': ['session_hijacking', 'csrf_attacks', 'session_fixation'],
                'description': 'Session management may be vulnerable to attacks',
                'timeline': '1-2 months'
            }
        }
        
        if context in context_risks:
            risk_info = context_risks[context]
            
            prediction = CodePrediction(
                prediction_type=PredictionType.SECURITY_VULNERABILITIES,
                target_file=file_path,
                target_element=context,
                prediction_summary=f"Security context risk: {context}",
                detailed_analysis=risk_info['description'],
                confidence=PredictionConfidence.MEDIUM,
                probability_score=confidence * 0.8,
                timeline_estimate=risk_info['timeline'],
                impact_assessment={
                    'context_risk': context,
                    'security_impact': 'medium',
                    'potential_attacks': risk_info['risks']
                },
                recommended_actions=[
                    f"Review {context} implementation",
                    f"Apply {context} security best practices",
                    f"Test {context} security controls",
                    f"Monitor {context} security events"
                ],
                monitoring_indicators=[
                    f"{context.title()} security events",
                    "Failed security operations",
                    "Unusual access patterns"
                ]
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _analyze_vulnerability_evolution_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Analyze patterns that often lead to future vulnerabilities"""
        
        evolution_risks = []
        
        # Rapid development patterns
        if self._detect_rapid_development_patterns(code):
            evolution_risks.append({
                'element': 'development_velocity',
                'type': 'rapid_development_security_debt',
                'description': 'Rapid development patterns detected that often lead to security debt',
                'probability': 0.7,
                'timeline': '2-4 months',
                'impact': {
                    'security_debt_accumulation': 'high',
                    'vulnerability_introduction_risk': 'medium'
                },
                'prevention_actions': [
                    'Implement security code review gates',
                    'Add automated security testing',
                    'Provide security training for developers'
                ],
                'monitoring_indicators': [
                    'Code review coverage',
                    'Security test failure rates',
                    'Time between commits'
                ]
            })
        
        # Complex authentication logic
        if self._detect_complex_auth_patterns(code):
            evolution_risks.append({
                'element': 'authentication_complexity',
                'type': 'authentication_complexity_risk',
                'description': 'Complex authentication logic often leads to security vulnerabilities',
                'probability': 0.6,
                'timeline': '1-3 months',
                'impact': {
                    'authentication_bypass_risk': 'high',
                    'privilege_escalation_risk': 'medium'
                },
                'prevention_actions': [
                    'Simplify authentication logic',
                    'Use well-tested authentication frameworks',
                    'Implement comprehensive authentication testing'
                ],
                'monitoring_indicators': [
                    'Authentication failure rates',
                    'Unusual login patterns',
                    'Authentication bypass attempts'
                ]
            })
        
        # Data processing complexity
        if self._detect_data_processing_complexity(code):
            evolution_risks.append({
                'element': 'data_processing',
                'type': 'data_processing_vulnerability_risk',
                'description': 'Complex data processing often introduces injection vulnerabilities',
                'probability': 0.5,
                'timeline': '1-2 months',
                'impact': {
                    'injection_attack_risk': 'medium',
                    'data_corruption_risk': 'medium'
                },
                'prevention_actions': [
                    'Implement input validation frameworks',
                    'Use parameterized queries consistently',
                    'Add data processing security tests'
                ],
                'monitoring_indicators': [
                    'Data validation failures',
                    'Unusual data patterns',
                    'Database error rates'
                ]
            })
        
        return evolution_risks
    
    def _detect_rapid_development_patterns(self, code: str) -> bool:
        """Detect patterns indicating rapid development that may compromise security"""
        
        # Look for indicators of rushed development
        indicators = [
            'TODO', 'FIXME', 'HACK', 'TEMP', 'QUICK',
            'pass  # TODO', 'raise NotImplementedError'
        ]
        
        indicator_count = sum(code.count(indicator) for indicator in indicators)
        return indicator_count > 3
    
    def _detect_complex_auth_patterns(self, code: str) -> bool:
        """Detect complex authentication patterns"""
        
        auth_keywords = ['auth', 'login', 'password', 'token', 'session']
        auth_complexity = 0
        
        for keyword in auth_keywords:
            auth_complexity += code.lower().count(keyword)
        
        # Also check for nested conditions in auth-related code
        if auth_complexity > 5:
            # Look for complex conditional logic
            nested_conditions = code.count('if') + code.count('elif') + code.count('else')
            return nested_conditions > 10
        
        return False
    
    def _detect_data_processing_complexity(self, code: str) -> bool:
        """Detect complex data processing patterns"""
        
        processing_keywords = ['process', 'parse', 'transform', 'convert', 'sanitize']
        processing_complexity = 0
        
        for keyword in processing_keywords:
            processing_complexity += code.lower().count(keyword)
        
        # Check for string operations that might be vulnerable
        string_ops = code.count('+') + code.count('format') + code.count('%')
        
        return processing_complexity > 3 and string_ops > 10
    
    def _get_full_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name from AST node"""
        
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_full_attribute_name(node.value)}.{node.attr}"
        else:
            return node.attr
    
    def create_security_risk_report(self, predictions: List[CodePrediction]) -> Dict[str, Any]:
        """Create comprehensive security risk report"""
        
        try:
            report = {
                'summary': {
                    'total_vulnerabilities': len(predictions),
                    'critical_count': 0,
                    'high_count': 0,
                    'medium_count': 0,
                    'low_count': 0,
                    'overall_risk_score': 0.0
                },
                'vulnerability_breakdown': {},
                'immediate_actions': [],
                'long_term_strategy': [],
                'compliance_impact': [],
                'risk_timeline': {}
            }
            
            # Analyze predictions by severity and type
            vulnerability_types = {}
            total_risk_score = 0.0
            
            for prediction in predictions:
                # Count by severity
                severity = prediction.impact_assessment.get('security_impact', 'medium')
                if severity == 'critical':
                    report['summary']['critical_count'] += 1
                elif severity == 'high':
                    report['summary']['high_count'] += 1
                elif severity == 'medium':
                    report['summary']['medium_count'] += 1
                else:
                    report['summary']['low_count'] += 1
                
                # Accumulate risk score
                total_risk_score += prediction.probability_score
                
                # Track vulnerability types
                vuln_type = prediction.target_element
                if vuln_type not in vulnerability_types:
                    vulnerability_types[vuln_type] = {
                        'count': 0,
                        'severity_distribution': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                        'examples': []
                    }
                
                vulnerability_types[vuln_type]['count'] += 1
                vulnerability_types[vuln_type]['severity_distribution'][severity] += 1
                vulnerability_types[vuln_type]['examples'].append(prediction.prediction_summary)
                
                # Collect immediate actions
                if prediction.timeline_estimate in ['Immediate attention required', 'Immediate']:
                    report['immediate_actions'].extend(prediction.recommended_actions[:2])
            
            # Calculate overall risk score
            if predictions:
                report['summary']['overall_risk_score'] = total_risk_score / len(predictions)
            
            # Set vulnerability breakdown
            report['vulnerability_breakdown'] = vulnerability_types
            
            # Remove duplicate immediate actions
            report['immediate_actions'] = list(set(report['immediate_actions']))[:10]
            
            # Generate long-term strategy
            report['long_term_strategy'] = [
                'Implement comprehensive security training program',
                'Establish security code review processes',
                'Deploy automated security testing tools',
                'Create incident response procedures',
                'Regular security assessments and penetration testing'
            ]
            
            # Assess compliance impact
            if report['summary']['critical_count'] > 0 or report['summary']['high_count'] > 3:
                report['compliance_impact'] = [
                    'High risk of compliance violations',
                    'Immediate remediation required for audit compliance',
                    'May impact certifications and customer trust'
                ]
            else:
                report['compliance_impact'] = [
                    'Moderate compliance risk',
                    'Address vulnerabilities to maintain compliance',
                    'Regular security reviews recommended'
                ]
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating security risk report: {e}")
            return {'error': str(e)}


def create_security_analyzer() -> SecurityAnalyzer:
    """Factory function to create SecurityAnalyzer instance"""
    
    return SecurityAnalyzer()