"""
Universal Security Scanner

Language-agnostic security vulnerability detection.
Adapted from Agency Swarm's security patterns and SAST tools integration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
import re
import hashlib
import json
from datetime import datetime
from pathlib import Path

from ..core.ast_abstraction import UniversalAST, UniversalFunction, UniversalClass
from ..core.language_detection import UniversalLanguageDetector


class SeverityLevel(Enum):
    """Severity levels for vulnerabilities."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    CSRF = "cross_site_request_forgery"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    CRYPTO_FAILURE = "cryptographic_failure"
    INJECTION = "injection"
    INSECURE_DESIGN = "insecure_design"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    LOGGING_MONITORING_FAILURE = "logging_monitoring_failure"
    SSRF = "server_side_request_forgery"
    BUFFER_OVERFLOW = "buffer_overflow"
    RACE_CONDITION = "race_condition"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    WEAK_RANDOMNESS = "weak_randomness"
    INSECURE_COMMUNICATION = "insecure_communication"


@dataclass
class VulnerabilityFinding:
    """Represents a security vulnerability finding."""
    type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    file_path: str
    line_number: int
    column_number: int = 0
    
    # Context information
    code_snippet: str = ""
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    
    # Remediation
    recommendation: str = ""
    cwe_id: Optional[str] = None  # Common Weakness Enumeration
    owasp_category: Optional[str] = None
    
    # Detection metadata
    rule_id: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    false_positive_likelihood: float = 0.0
    
    # Additional context
    data_flow: List[str] = field(default_factory=list)
    input_sources: List[str] = field(default_factory=list)
    sinks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'column_number': self.column_number,
            'code_snippet': self.code_snippet,
            'function_name': self.function_name,
            'class_name': self.class_name,
            'recommendation': self.recommendation,
            'cwe_id': self.cwe_id,
            'owasp_category': self.owasp_category,
            'rule_id': self.rule_id,
            'confidence': self.confidence,
            'false_positive_likelihood': self.false_positive_likelihood,
            'data_flow': self.data_flow,
            'input_sources': self.input_sources,
            'sinks': self.sinks
        }


@dataclass
class SecurityScanConfig:
    """Configuration for security scanning."""
    # Scan scope
    include_patterns: List[str] = field(default_factory=lambda: ["**/*"])
    exclude_patterns: List[str] = field(default_factory=list)
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Vulnerability types to scan for
    enabled_vulnerability_types: List[VulnerabilityType] = field(default_factory=lambda: list(VulnerabilityType))
    
    # Severity filtering
    min_severity: SeverityLevel = SeverityLevel.LOW
    include_informational: bool = True
    
    # Performance settings
    parallel_scans: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300
    
    # Language-specific settings
    language_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # False positive reduction
    enable_dataflow_analysis: bool = True
    confidence_threshold: float = 0.5
    suppress_false_positives: bool = True
    
    # Output settings
    include_code_snippets: bool = True
    max_snippet_lines: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'include_patterns': self.include_patterns,
            'exclude_patterns': self.exclude_patterns,
            'max_file_size': self.max_file_size,
            'enabled_vulnerability_types': [vt.value for vt in self.enabled_vulnerability_types],
            'min_severity': self.min_severity.value,
            'include_informational': self.include_informational,
            'parallel_scans': self.parallel_scans,
            'max_workers': self.max_workers,
            'timeout_seconds': self.timeout_seconds,
            'language_configs': self.language_configs,
            'enable_dataflow_analysis': self.enable_dataflow_analysis,
            'confidence_threshold': self.confidence_threshold,
            'suppress_false_positives': self.suppress_false_positives,
            'include_code_snippets': self.include_code_snippets,
            'max_snippet_lines': self.max_snippet_lines
        }


@dataclass
class SecurityScanResult:
    """Result of security scanning."""
    findings: List[VulnerabilityFinding] = field(default_factory=list)
    
    # Summary statistics
    total_files_scanned: int = 0
    total_lines_scanned: int = 0
    scan_duration: float = 0.0
    
    # Findings by severity
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    
    # Findings by type
    findings_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    scan_timestamp: datetime = field(default_factory=datetime.now)
    scanner_version: str = "1.0.0"
    config_used: Optional[SecurityScanConfig] = None
    
    def calculate_statistics(self):
        """Calculate summary statistics."""
        # Count by severity
        severity_counts = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 0,
            SeverityLevel.MEDIUM: 0,
            SeverityLevel.LOW: 0,
            SeverityLevel.INFO: 0
        }
        
        # Count by type
        type_counts = {}
        
        for finding in self.findings:
            severity_counts[finding.severity] += 1
            
            vuln_type = finding.type.value
            type_counts[vuln_type] = type_counts.get(vuln_type, 0) + 1
        
        self.critical_count = severity_counts[SeverityLevel.CRITICAL]
        self.high_count = severity_counts[SeverityLevel.HIGH]
        self.medium_count = severity_counts[SeverityLevel.MEDIUM]
        self.low_count = severity_counts[SeverityLevel.LOW]
        self.info_count = severity_counts[SeverityLevel.INFO]
        
        self.findings_by_type = type_counts
    
    def get_risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        # Weighted scoring based on severity
        score = (
            self.critical_count * 10 +
            self.high_count * 7 +
            self.medium_count * 4 +
            self.low_count * 2 +
            self.info_count * 0.5
        )
        
        # Normalize to 0-100 scale (rough approximation)
        return min(score, 100.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'findings': [f.to_dict() for f in self.findings],
            'summary': {
                'total_findings': len(self.findings),
                'total_files_scanned': self.total_files_scanned,
                'total_lines_scanned': self.total_lines_scanned,
                'scan_duration': self.scan_duration,
                'risk_score': self.get_risk_score()
            },
            'severity_breakdown': {
                'critical': self.critical_count,
                'high': self.high_count,
                'medium': self.medium_count,
                'low': self.low_count,
                'info': self.info_count
            },
            'findings_by_type': self.findings_by_type,
            'metadata': {
                'scan_timestamp': self.scan_timestamp.isoformat(),
                'scanner_version': self.scanner_version
            }
        }


class UniversalSecurityScanner:
    """Universal security scanner for any programming language."""
    
    def __init__(self, config: SecurityScanConfig = None):
        self.config = config or SecurityScanConfig()
        self.language_detector = UniversalLanguageDetector()
        
        # Security rule patterns by language
        self.security_patterns = self._load_security_patterns()
        
        # Vulnerability databases
        self.cwe_database = self._load_cwe_database()
        self.owasp_mappings = self._load_owasp_mappings()
        
        print(f"Universal Security Scanner initialized")
        print(f"   Enabled vulnerability types: {len(self.config.enabled_vulnerability_types)}")
        print(f"   Min severity: {self.config.min_severity.value}")
    
    def scan_directory(self, directory_path: str) -> SecurityScanResult:
        """Scan directory for security vulnerabilities."""
        start_time = datetime.now()
        
        print(f"\nStarting security scan of: {directory_path}")
        
        # Detect languages in directory
        codebase_profile = self.language_detector.detect_codebase(directory_path)
        
        print(f"   Languages detected: {', '.join(codebase_profile.languages.keys())}")
        
        # Collect files to scan
        files_to_scan = self._collect_files(directory_path)
        
        print(f"   Files to scan: {len(files_to_scan)}")
        
        # Initialize result
        result = SecurityScanResult(config_used=self.config)
        result.total_files_scanned = len(files_to_scan)
        
        # Scan each file
        total_lines = 0
        for file_path in files_to_scan:
            try:
                file_findings, lines_scanned = self._scan_file(file_path)
                result.findings.extend(file_findings)
                total_lines += lines_scanned
                
            except Exception as e:
                print(f"   Error scanning {file_path}: {str(e)}")
                continue
        
        result.total_lines_scanned = total_lines
        result.scan_duration = (datetime.now() - start_time).total_seconds()
        
        # Calculate statistics
        result.calculate_statistics()
        
        # Print summary
        print(f"\nSecurity scan completed:")
        print(f"   Total findings: {len(result.findings)}")
        print(f"   Critical: {result.critical_count}")
        print(f"   High: {result.high_count}")
        print(f"   Medium: {result.medium_count}")
        print(f"   Low: {result.low_count}")
        print(f"   Risk score: {result.get_risk_score():.1f}/100")
        print(f"   Scan time: {result.scan_duration:.2f}s")
        
        return result
    
    def scan_ast(self, universal_ast: UniversalAST) -> SecurityScanResult:
        """Scan Universal AST for security vulnerabilities."""
        start_time = datetime.now()
        
        print(f"\nScanning AST for security vulnerabilities...")
        print(f"   Modules: {len(universal_ast.modules)}")
        print(f"   Functions: {universal_ast.total_functions}")
        print(f"   Classes: {universal_ast.total_classes}")
        
        result = SecurityScanResult(config_used=self.config)
        
        # Scan each module
        for module in universal_ast.modules:
            module_findings = self._scan_ast_module(module)
            result.findings.extend(module_findings)
        
        result.total_files_scanned = len(universal_ast.modules)
        result.scan_duration = (datetime.now() - start_time).total_seconds()
        result.calculate_statistics()
        
        print(f"   AST scan completed: {len(result.findings)} findings")
        
        return result
    
    def _collect_files(self, directory_path: str) -> List[str]:
        """Collect files to scan based on patterns."""
        directory = Path(directory_path)
        files_to_scan = []
        
        # Supported file extensions by language
        extensions = {
            '.py', '.js', '.ts', '.java', '.cs', '.go', '.rs', '.cpp', '.c', '.hpp', '.h',
            '.php', '.rb', '.scala', '.kt', '.swift', '.dart', '.elm', '.haskell', '.clj',
            '.sql', '.yaml', '.yml', '.json', '.xml', '.html', '.htm'
        }
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # Check size limit
                if file_path.stat().st_size <= self.config.max_file_size:
                    # Check patterns
                    relative_path = str(file_path.relative_to(directory))
                    if self._matches_patterns(relative_path):
                        files_to_scan.append(str(file_path))
        
        return files_to_scan
    
    def _matches_patterns(self, file_path: str) -> bool:
        """Check if file matches include/exclude patterns."""
        import fnmatch
        
        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return False
        
        # Check include patterns
        for pattern in self.config.include_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        
        return False
    
    def _scan_file(self, file_path: str) -> Tuple[List[VulnerabilityFinding], int]:
        """Scan a single file for vulnerabilities."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return [], 0
        
        lines = content.split('\n')
        findings = []
        
        # Detect file language
        language = self._detect_file_language(file_path, content)
        
        # Get language-specific patterns
        patterns = self.security_patterns.get(language, {})
        patterns.update(self.security_patterns.get('universal', {}))
        
        # Scan each line
        for line_num, line in enumerate(lines, 1):
            line_findings = self._scan_line(line, line_num, file_path, language, patterns)
            findings.extend(line_findings)
        
        # Perform multi-line analysis
        if self.config.enable_dataflow_analysis:
            dataflow_findings = self._analyze_dataflow(content, file_path, language)
            findings.extend(dataflow_findings)
        
        return findings, len(lines)
    
    def _scan_ast_module(self, module) -> List[VulnerabilityFinding]:
        """Scan AST module for vulnerabilities."""
        findings = []
        
        # Scan functions
        for function in module.functions:
            function_findings = self._scan_ast_function(function, module.file_path)
            findings.extend(function_findings)
        
        # Scan classes
        for cls in module.classes:
            class_findings = self._scan_ast_class(cls, module.file_path)
            findings.extend(class_findings)
        
        return findings
    
    def _scan_ast_function(self, function: UniversalFunction, file_path: str) -> List[VulnerabilityFinding]:
        """Scan AST function for vulnerabilities."""
        findings = []
        
        # Check function name for security indicators
        if self._is_security_sensitive_function(function.name):
            # Look for common security issues
            
            # Check for authentication/authorization
            if 'auth' in function.name.lower() or 'login' in function.name.lower():
                if not self._has_security_checks(function):
                    findings.append(VulnerabilityFinding(
                        type=VulnerabilityType.AUTHENTICATION_BYPASS,
                        severity=SeverityLevel.HIGH,
                        title="Potential Authentication Bypass",
                        description=f"Function {function.name} handles authentication but may lack proper security checks",
                        file_path=file_path,
                        line_number=function.start_line,
                        function_name=function.name,
                        rule_id="AUTH_001",
                        confidence=0.7,
                        recommendation="Implement proper input validation and security checks"
                    ))
            
            # Check for SQL operations
            if any(keyword in function.name.lower() for keyword in ['sql', 'query', 'select', 'insert', 'update', 'delete']):
                if not self._has_parameterized_queries(function):
                    findings.append(VulnerabilityFinding(
                        type=VulnerabilityType.SQL_INJECTION,
                        severity=SeverityLevel.HIGH,
                        title="Potential SQL Injection",
                        description=f"Function {function.name} may be vulnerable to SQL injection",
                        file_path=file_path,
                        line_number=function.start_line,
                        function_name=function.name,
                        rule_id="SQL_001",
                        confidence=0.6,
                        owasp_category="A03:2021 - Injection",
                        recommendation="Use parameterized queries or prepared statements"
                    ))
        
        return findings
    
    def _scan_ast_class(self, cls: UniversalClass, file_path: str) -> List[VulnerabilityFinding]:
        """Scan AST class for vulnerabilities."""
        findings = []
        
        # Check for security-related classes
        if self._is_security_sensitive_class(cls.name):
            # Check for proper security implementation
            if not self._has_security_methods(cls):
                findings.append(VulnerabilityFinding(
                    type=VulnerabilityType.INSECURE_DESIGN,
                    severity=SeverityLevel.MEDIUM,
                    title="Insecure Class Design",
                    description=f"Security-related class {cls.name} may lack proper security implementation",
                    file_path=file_path,
                    line_number=cls.start_line,
                    class_name=cls.name,
                    rule_id="DESIGN_001",
                    confidence=0.5,
                    recommendation="Implement proper security controls and validation"
                ))
        
        # Scan methods
        for method in cls.methods:
            method_findings = self._scan_ast_function(method, file_path)
            findings.extend(method_findings)
        
        return findings
    
    def _scan_line(self, line: str, line_num: int, file_path: str, language: str, patterns: Dict) -> List[VulnerabilityFinding]:
        """Scan a single line for vulnerabilities."""
        findings = []
        line_stripped = line.strip()
        
        if not line_stripped or line_stripped.startswith('#') or line_stripped.startswith('//'):
            return findings
        
        # Check patterns for this language
        for vuln_type, pattern_list in patterns.items():
            if not isinstance(pattern_list, list):
                pattern_list = [pattern_list]
            
            try:
                vuln_enum = VulnerabilityType(vuln_type)
            except ValueError:
                continue
            
            if vuln_enum not in self.config.enabled_vulnerability_types:
                continue
            
            for pattern_info in pattern_list:
                if isinstance(pattern_info, str):
                    pattern = pattern_info
                    severity = SeverityLevel.MEDIUM
                    confidence = 0.7
                else:
                    pattern = pattern_info.get('pattern', '')
                    severity = SeverityLevel(pattern_info.get('severity', 'medium'))
                    confidence = pattern_info.get('confidence', 0.7)
                
                if confidence < self.config.confidence_threshold:
                    continue
                
                # Check if pattern matches
                if re.search(pattern, line, re.IGNORECASE):
                    # Create finding
                    finding = VulnerabilityFinding(
                        type=vuln_enum,
                        severity=severity,
                        title=self._get_vulnerability_title(vuln_enum),
                        description=self._get_vulnerability_description(vuln_enum, line_stripped),
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line_stripped,
                        rule_id=f"{vuln_type.upper()}_{hash(pattern) % 1000:03d}",
                        confidence=confidence,
                        cwe_id=self.cwe_database.get(vuln_type),
                        owasp_category=self.owasp_mappings.get(vuln_type),
                        recommendation=self._get_vulnerability_recommendation(vuln_enum)
                    )
                    
                    findings.append(finding)
        
        return findings
    
    def _analyze_dataflow(self, content: str, file_path: str, language: str) -> List[VulnerabilityFinding]:
        """Perform dataflow analysis for vulnerability detection."""
        findings = []
        
        # Simplified dataflow analysis
        # In a full implementation, this would use proper AST analysis
        
        # Look for user input -> dangerous function patterns
        user_input_patterns = [
            r'request\.',
            r'input\(',
            r'argv\[',
            r'os\.environ',
            r'sys\.argv'
        ]
        
        dangerous_functions = [
            r'eval\(',
            r'exec\(',
            r'system\(',
            r'shell_exec\(',
            r'passthru\(',
            r'query\(',
            r'execute\('
        ]
        
        lines = content.split('\n')
        
        # Simple taint analysis
        for i, line in enumerate(lines):
            # Check if line has user input
            has_input = any(re.search(pattern, line, re.IGNORECASE) for pattern in user_input_patterns)
            
            if has_input:
                # Look for dangerous functions in nearby lines (within 10 lines)
                start = max(0, i - 10)
                end = min(len(lines), i + 10)
                
                for j in range(start, end):
                    check_line = lines[j]
                    for dangerous_pattern in dangerous_functions:
                        if re.search(dangerous_pattern, check_line, re.IGNORECASE):
                            findings.append(VulnerabilityFinding(
                                type=VulnerabilityType.INJECTION,
                                severity=SeverityLevel.HIGH,
                                title="Potential Code Injection",
                                description="User input may flow to dangerous function",
                                file_path=file_path,
                                line_number=j + 1,
                                code_snippet=check_line.strip(),
                                rule_id="DATAFLOW_001",
                                confidence=0.8,
                                data_flow=[f"Input at line {i+1}", f"Sink at line {j+1}"],
                                input_sources=[line.strip()],
                                sinks=[check_line.strip()],
                                owasp_category="A03:2021 - Injection",
                                recommendation="Validate and sanitize all user input before using in dangerous functions"
                            ))
        
        return findings
    
    def _detect_file_language(self, file_path: str, content: str) -> str:
        """Detect programming language of a file."""
        # Simple detection based on file extension
        ext_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        file_ext = Path(file_path).suffix.lower()
        return ext_mapping.get(file_ext, 'unknown')
    
    def _load_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load security patterns for different languages."""
        return {
            'universal': {
                'hardcoded_credentials': [
                    {
                        'pattern': r'password\s*=\s*["\'][^"\']{3,}["\']',
                        'severity': 'high',
                        'confidence': 0.9
                    },
                    {
                        'pattern': r'api_key\s*=\s*["\'][^"\']{10,}["\']',
                        'severity': 'high',
                        'confidence': 0.9
                    }
                ],
                'weak_randomness': [
                    {
                        'pattern': r'random\(\)',
                        'severity': 'medium',
                        'confidence': 0.6
                    }
                ],
                'sensitive_data_exposure': [
                    {
                        'pattern': r'print\(.*password.*\)',
                        'severity': 'medium',
                        'confidence': 0.8
                    }
                ]
            },
            'python': {
                'command_injection': [
                    {
                        'pattern': r'os\.system\(',
                        'severity': 'high',
                        'confidence': 0.9
                    },
                    {
                        'pattern': r'subprocess\.call\(',
                        'severity': 'medium',
                        'confidence': 0.7
                    }
                ],
                'sql_injection': [
                    {
                        'pattern': r'cursor\.execute\([^)]*%s[^)]*\)',
                        'severity': 'high',
                        'confidence': 0.8
                    }
                ]
            },
            'javascript': {
                'xss': [
                    {
                        'pattern': r'innerHTML\s*=',
                        'severity': 'medium',
                        'confidence': 0.6
                    },
                    {
                        'pattern': r'document\.write\(',
                        'severity': 'high',
                        'confidence': 0.8
                    }
                ]
            },
            'java': {
                'sql_injection': [
                    {
                        'pattern': r'Statement\.execute\(',
                        'severity': 'high',
                        'confidence': 0.8
                    }
                ]
            }
        }
    
    def _load_cwe_database(self) -> Dict[str, str]:
        """Load CWE (Common Weakness Enumeration) mappings."""
        return {
            'sql_injection': 'CWE-89',
            'cross_site_scripting': 'CWE-79',
            'command_injection': 'CWE-78',
            'path_traversal': 'CWE-22',
            'hardcoded_credentials': 'CWE-798',
            'weak_randomness': 'CWE-330',
            'buffer_overflow': 'CWE-120',
            'race_condition': 'CWE-362',
            'authentication_bypass': 'CWE-287',
            'authorization_failure': 'CWE-285'
        }
    
    def _load_owasp_mappings(self) -> Dict[str, str]:
        """Load OWASP Top 10 mappings."""
        return {
            'authentication_bypass': 'A07:2021 - Identification and Authentication Failures',
            'authorization_failure': 'A01:2021 - Broken Access Control',
            'sql_injection': 'A03:2021 - Injection',
            'cross_site_scripting': 'A03:2021 - Injection',
            'command_injection': 'A03:2021 - Injection',
            'insecure_design': 'A04:2021 - Insecure Design',
            'security_misconfiguration': 'A05:2021 - Security Misconfiguration',
            'vulnerable_components': 'A06:2021 - Vulnerable and Outdated Components',
            'sensitive_data_exposure': 'A02:2021 - Cryptographic Failures',
            'logging_monitoring_failure': 'A09:2021 - Security Logging and Monitoring Failures',
            'server_side_request_forgery': 'A10:2021 - Server-Side Request Forgery (SSRF)'
        }
    
    def _is_security_sensitive_function(self, func_name: str) -> bool:
        """Check if function name indicates security sensitivity."""
        security_keywords = [
            'auth', 'login', 'password', 'token', 'session', 'crypto', 'encrypt', 'decrypt',
            'hash', 'sign', 'verify', 'validate', 'sanitize', 'escape', 'sql', 'query',
            'execute', 'eval', 'system', 'shell', 'command', 'file', 'upload', 'download'
        ]
        
        func_lower = func_name.lower()
        return any(keyword in func_lower for keyword in security_keywords)
    
    def _is_security_sensitive_class(self, class_name: str) -> bool:
        """Check if class name indicates security sensitivity."""
        security_keywords = [
            'auth', 'security', 'crypto', 'session', 'user', 'admin', 'permission',
            'role', 'access', 'token', 'validator', 'sanitizer'
        ]
        
        class_lower = class_name.lower()
        return any(keyword in class_lower for keyword in security_keywords)
    
    def _has_security_checks(self, function: UniversalFunction) -> bool:
        """Check if function has security checks (simplified)."""
        # This would need actual code analysis
        # For now, just check if function has reasonable complexity
        return function.cyclomatic_complexity > 3
    
    def _has_parameterized_queries(self, function: UniversalFunction) -> bool:
        """Check if function uses parameterized queries (simplified)."""
        # This would need actual code analysis
        return len(function.parameters) > 0
    
    def _has_security_methods(self, cls: UniversalClass) -> bool:
        """Check if class has security-related methods."""
        security_method_names = ['validate', 'authorize', 'authenticate', 'sanitize', 'encrypt']
        
        for method in cls.methods:
            if any(sec_name in method.name.lower() for sec_name in security_method_names):
                return True
        
        return False
    
    def _get_vulnerability_title(self, vuln_type: VulnerabilityType) -> str:
        """Get human-readable title for vulnerability type."""
        titles = {
            VulnerabilityType.SQL_INJECTION: "SQL Injection Vulnerability",
            VulnerabilityType.XSS: "Cross-Site Scripting (XSS)",
            VulnerabilityType.COMMAND_INJECTION: "Command Injection",
            VulnerabilityType.HARDCODED_CREDENTIALS: "Hardcoded Credentials",
            VulnerabilityType.WEAK_RANDOMNESS: "Weak Random Number Generation",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "Sensitive Data Exposure",
            VulnerabilityType.AUTHENTICATION_BYPASS: "Authentication Bypass",
            VulnerabilityType.AUTHORIZATION_FAILURE: "Authorization Failure"
        }
        
        return titles.get(vuln_type, f"{vuln_type.value.replace('_', ' ').title()} Vulnerability")
    
    def _get_vulnerability_description(self, vuln_type: VulnerabilityType, code: str) -> str:
        """Get detailed description for vulnerability."""
        return f"Potential {vuln_type.value.replace('_', ' ')} detected in code: {code[:100]}..."
    
    def _get_vulnerability_recommendation(self, vuln_type: VulnerabilityType) -> str:
        """Get recommendation for fixing vulnerability."""
        recommendations = {
            VulnerabilityType.SQL_INJECTION: "Use parameterized queries or prepared statements",
            VulnerabilityType.XSS: "Sanitize user input and use output encoding",
            VulnerabilityType.COMMAND_INJECTION: "Validate input and avoid shell execution",
            VulnerabilityType.HARDCODED_CREDENTIALS: "Use environment variables or secure credential storage",
            VulnerabilityType.WEAK_RANDOMNESS: "Use cryptographically secure random number generators",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "Avoid logging sensitive information",
            VulnerabilityType.AUTHENTICATION_BYPASS: "Implement proper authentication checks",
            VulnerabilityType.AUTHORIZATION_FAILURE: "Implement proper authorization controls"
        }
        
        return recommendations.get(vuln_type, "Review code for security best practices")