from SECURITY_PATCHES.fix_command_injection import SafeCommandExecutor
"""
MetaGPT Derived Code Generation Security
Extracted from MetaGPT code sanitization patterns and validation systems
Enhanced for comprehensive code generation security and validation
"""

import logging
import re
import ast
import hashlib
import time
from typing import Dict, Any, Optional, List, Set, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import tempfile
from .error_handler import SecurityError, ValidationError, security_error_handler


class CodeSecurityLevel(Enum):
    """Code security classification levels"""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"
    BLOCKED = "blocked"


class CodeLanguage(Enum):
    """Supported programming languages for analysis"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    SHELL = "shell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities in code"""
    INJECTION = "injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    CODE_INJECTION = "code_injection"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    HARDCODED_SECRETS = "hardcoded_secrets"
    DANGEROUS_FUNCTIONS = "dangerous_functions"
    INSECURE_RANDOMNESS = "insecure_randomness"
    BUFFER_OVERFLOW = "buffer_overflow"
    RACE_CONDITION = "race_condition"


@dataclass
class SecurityViolation:
    """Security violation found in generated code"""
    violation_id: str
    vulnerability_type: VulnerabilityType
    severity: CodeSecurityLevel
    line_number: int
    column: Optional[int]
    code_snippet: str
    description: str
    recommendation: str
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_critical(self) -> bool:
        """Check if violation is critical"""
        return self.severity in [CodeSecurityLevel.CRITICAL_RISK, CodeSecurityLevel.BLOCKED]


@dataclass
class CodeAnalysisResult:
    """Comprehensive code analysis result"""
    analysis_id: str
    language: CodeLanguage
    code_hash: str
    security_level: CodeSecurityLevel
    violations: List[SecurityViolation] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    analysis_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def critical_violations(self) -> List[SecurityViolation]:
        """Get critical security violations"""
        return [v for v in self.violations if v.is_critical]
    
    @property
    def security_score(self) -> float:
        """Calculate security score (0-100)"""
        if not self.violations:
            return 100.0
        
        score = 100.0
        for violation in self.violations:
            if violation.severity == CodeSecurityLevel.CRITICAL_RISK:
                score -= 30
            elif violation.severity == CodeSecurityLevel.HIGH_RISK:
                score -= 20
            elif violation.severity == CodeSecurityLevel.MEDIUM_RISK:
                score -= 10
            elif violation.severity == CodeSecurityLevel.LOW_RISK:
                score -= 5
        
        return max(0.0, score)


class PythonCodeAnalyzer:
    """Python code security analyzer based on MetaGPT patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dangerous_imports = {
            'os': ['system', 'popen', 'spawn*', 'exec*'],
            'subprocess': ['call', 'run', 'Popen', 'check_call', 'check_output'],
            'eval': ['eval', 'exec', 'compile'],
            'pickle': ['loads', 'load', 'dumps', 'dump'],
            'marshal': ['loads', 'load'],
            'imp': ['load_*', 'exec_*'],
            'importlib': ['import_module'],
            '__builtins__': ['eval', 'exec', 'compile', '__import__']
        }
        
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__',
            'getattr', 'setattr', 'delattr', 'hasattr',
            'globals', 'locals', 'vars', 'dir'
        }
    
    def analyze_code(self, code: str) -> CodeAnalysisResult:
        """Analyze Python code for security issues"""
        analysis_start = time.time()
        analysis_id = hashlib.sha256(f"{code}{time.time()}".encode()).hexdigest()[:12]
        
        try:
            # Parse code into AST
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return CodeAnalysisResult(
                    analysis_id=analysis_id,
                    language=CodeLanguage.PYTHON,
                    code_hash=hashlib.sha256(code.encode()).hexdigest(),
                    security_level=CodeSecurityLevel.BLOCKED,
                    violations=[SecurityViolation(
                        violation_id=f"syntax_{analysis_id}",
                        vulnerability_type=VulnerabilityType.CODE_INJECTION,
                        severity=CodeSecurityLevel.BLOCKED,
                        line_number=e.lineno or 1,
                        column=e.offset,
                        code_snippet=str(e.text or ""),
                        description=f"Syntax error: {str(e)}",
                        recommendation="Fix syntax errors before analysis",
                        confidence=1.0
                    )],
                    analysis_time_ms=(time.time() - analysis_start) * 1000
                )
            
            violations = []
            imports = []
            functions = []
            
            # Walk AST nodes
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_violations = self._check_dangerous_imports(node)
                    violations.extend(import_violations)
                    imports.extend(self._extract_imports(node))
                
                # Check function calls
                elif isinstance(node, ast.Call):
                    call_violations = self._check_dangerous_calls(node, code)
                    violations.extend(call_violations)
                
                # Check function definitions
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                
                # Check string operations that could be injections
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    if self._is_string_concatenation(node):
                        injection_violation = self._check_potential_injection(node, code)
                        if injection_violation:
                            violations.append(injection_violation)
            
            # Determine overall security level
            security_level = self._determine_security_level(violations)
            
            analysis_time = (time.time() - analysis_start) * 1000
            
            return CodeAnalysisResult(
                analysis_id=analysis_id,
                language=CodeLanguage.PYTHON,
                code_hash=hashlib.sha256(code.encode()).hexdigest(),
                security_level=security_level,
                violations=violations,
                imports=imports,
                functions=functions,
                analysis_time_ms=analysis_time
            )
            
        except Exception as e:
            analysis_time = (time.time() - analysis_start) * 1000
            error_violation = SecurityViolation(
                violation_id=f"error_{analysis_id}",
                vulnerability_type=VulnerabilityType.CODE_INJECTION,
                severity=CodeSecurityLevel.HIGH_RISK,
                line_number=1,
                column=0,
                code_snippet="",
                description=f"Analysis error: {str(e)}",
                recommendation="Manual code review required",
                confidence=0.8
            )
            
            return CodeAnalysisResult(
                analysis_id=analysis_id,
                language=CodeLanguage.PYTHON,
                code_hash=hashlib.sha256(code.encode()).hexdigest(),
                security_level=CodeSecurityLevel.HIGH_RISK,
                violations=[error_violation],
                analysis_time_ms=analysis_time
            )
    
    def _check_dangerous_imports(self, node: Union[ast.Import, ast.ImportFrom]) -> List[SecurityViolation]:
        """Check for dangerous imports"""
        violations = []
        
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.dangerous_imports:
                        violations.append(SecurityViolation(
                            violation_id=f"import_{alias.name}_{node.lineno}",
                            vulnerability_type=VulnerabilityType.DANGEROUS_FUNCTIONS,
                            severity=CodeSecurityLevel.HIGH_RISK,
                            line_number=node.lineno,
                            column=node.col_offset,
                            code_snippet=f"import {alias.name}",
                            description=f"Import of potentially dangerous module: {alias.name}",
                            recommendation=f"Avoid importing {alias.name} or use specific functions with caution",
                            confidence=0.9
                        ))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in self.dangerous_imports:
                    for alias in node.names:
                        if (alias.name in self.dangerous_imports[node.module] or 
                            any(alias.name.startswith(pattern.replace('*', '')) 
                                for pattern in self.dangerous_imports[node.module] if '*' in pattern)):
                            
                            violations.append(SecurityViolation(
                                violation_id=f"from_import_{node.module}_{alias.name}_{node.lineno}",
                                vulnerability_type=VulnerabilityType.DANGEROUS_FUNCTIONS,
                                severity=CodeSecurityLevel.CRITICAL_RISK,
                                line_number=node.lineno,
                                column=node.col_offset,
                                code_snippet=f"from {node.module} import {alias.name}",
                                description=f"Import of dangerous function: {node.module}.{alias.name}",
                                recommendation=f"Avoid using {alias.name} from {node.module}",
                                confidence=0.95
                            ))
            
        except Exception as e:
            self.logger.error(f"Error checking dangerous imports: {e}")
        
        return violations
    
    def _check_dangerous_calls(self, node: ast.Call, code: str) -> List[SecurityViolation]:
        """Check for dangerous function calls"""
        violations = []
        
        try:
            func_name = None
            
            # Get function name
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            
            if func_name in self.dangerous_functions:
                # Get code snippet
                lines = code.split('\n')
                if node.lineno <= len(lines):
                    code_snippet = lines[node.lineno - 1].strip()
                else:
                    code_snippet = func_name
                
                severity = CodeSecurityLevel.CRITICAL_RISK
                if func_name in ['eval', 'exec']:
                    severity = CodeSecurityLevel.BLOCKED
                
                violations.append(SecurityViolation(
                    violation_id=f"call_{func_name}_{node.lineno}",
                    vulnerability_type=VulnerabilityType.CODE_INJECTION,
                    severity=severity,
                    line_number=node.lineno,
                    column=node.col_offset,
                    code_snippet=code_snippet,
                    description=f"Dangerous function call: {func_name}",
                    recommendation=f"Avoid using {func_name} or sanitize inputs thoroughly",
                    confidence=0.9
                ))
            
            # Check for subprocess calls with shell=False  # WARNING: shell=True is a security risk - changed to shell=False
            if isinstance(node.func, ast.Attribute) and node.func.attr in ['call', 'run', 'Popen']:
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value:
                        violations.append(SecurityViolation(
                            violation_id=f"shell_injection_{node.lineno}",
                            vulnerability_type=VulnerabilityType.INJECTION,
                            severity=CodeSecurityLevel.CRITICAL_RISK,
                            line_number=node.lineno,
                            column=node.col_offset,
                            code_snippet=code.split('\n')[node.lineno - 1].strip() if node.lineno <= len(code.split('\n')) else "",
                            description="Subprocess call with shell=False  # WARNING: shell=True is a security risk - changed to shell=False",
                            recommendation="Use shell=False or validate inputs carefully",
                            confidence=0.95
                        ))
            
        except Exception as e:
            self.logger.error(f"Error checking dangerous calls: {e}")
        
        return violations
    
    def _extract_imports(self, node: Union[ast.Import, ast.ImportFrom]) -> List[str]:
        """Extract import statements"""
        imports = []
        
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        except Exception:
            pass
        
        return imports
    
    def _is_string_concatenation(self, node: ast.BinOp) -> bool:
        """Check if node is string concatenation"""
        return (isinstance(node.left, ast.Str) or isinstance(node.right, ast.Str) or
                isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) or
                isinstance(node.right, ast.Constant) and isinstance(node.right.value, str))
    
    def _check_potential_injection(self, node: ast.BinOp, code: str) -> Optional[SecurityViolation]:
        """Check for potential injection vulnerabilities in string operations"""
        try:
            lines = code.split('\n')
            if node.lineno <= len(lines):
                code_snippet = lines[node.lineno - 1].strip()
                
                # Check for SQL-like patterns
                if any(keyword in code_snippet.lower() for keyword in ['select', 'insert', 'update', 'delete', 'drop']):
                    return SecurityViolation(
                        violation_id=f"sql_injection_{node.lineno}",
                        vulnerability_type=VulnerabilityType.INJECTION,
                        severity=CodeSecurityLevel.HIGH_RISK,
                        line_number=node.lineno,
                        column=node.col_offset,
                        code_snippet=code_snippet,
                        description="Potential SQL injection vulnerability",
                        recommendation="Use parameterized queries instead of string concatenation",
                        confidence=0.7
                    )
        
        except Exception:
            pass
        
        return None
    
    def _determine_security_level(self, violations: List[SecurityViolation]) -> CodeSecurityLevel:
        """Determine overall security level based on violations"""
        if not violations:
            return CodeSecurityLevel.SAFE
        
        max_severity = max(violation.severity for violation in violations)
        return max_severity


class CodeSanitizer:
    """Code sanitization and safe transformation based on MetaGPT patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blocked_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.\w+\s*\([^)]*shell\s*=\s*True',
            r'pickle\.loads?\s*\(',
            r'marshal\.loads?\s*\(',
        ]
        
        self.replacement_suggestions = {
            'eval': 'ast.literal_eval',
            'exec': '# BLOCKED: Use proper function definitions',
            'os.system': 'subprocess.run with shell=False',
            'pickle.loads': 'json.loads for simple data',
            'compile': '# BLOCKED: Dynamic compilation not allowed'
        }
    
    def sanitize_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> Dict[str, Any]:
        """Sanitize generated code by removing dangerous patterns"""
        try:
            sanitization_result = {
                'original_code': code,
                'sanitized_code': code,
                'modifications': [],
                'blocked_patterns': [],
                'safe': True
            }
            
            sanitized_code = code
            
            # Check for blocked patterns
            for pattern in self.blocked_patterns:
                matches = list(re.finditer(pattern, sanitized_code, re.IGNORECASE))
                
                for match in reversed(matches):  # Reverse to maintain positions
                    matched_text = match.group()
                    start, end = match.span()
                    
                    # Record blocked pattern
                    sanitization_result['blocked_patterns'].append({
                        'pattern': pattern,
                        'matched_text': matched_text,
                        'line_number': sanitized_code[:start].count('\n') + 1,
                        'position': (start, end)
                    })
                    
                    # Replace with safe alternative or comment
                    replacement = self._get_safe_replacement(matched_text)
                    sanitized_code = sanitized_code[:start] + replacement + sanitized_code[end:]
                    
                    sanitization_result['modifications'].append({
                        'type': 'replacement',
                        'original': matched_text,
                        'replacement': replacement,
                        'reason': 'dangerous_pattern'
                    })
                    
                    sanitization_result['safe'] = False
            
            # Additional language-specific sanitization
            if language == CodeLanguage.PYTHON:
                sanitized_code = self._sanitize_python_specific(sanitized_code, sanitization_result)
            elif language == CodeLanguage.JAVASCRIPT:
                sanitized_code = self._sanitize_javascript_specific(sanitized_code, sanitization_result)
            
            sanitization_result['sanitized_code'] = sanitized_code
            
            return sanitization_result
            
        except Exception as e:
            self.logger.error(f"Code sanitization failed: {e}")
            return {
                'original_code': code,
                'sanitized_code': f"# SANITIZATION ERROR: {str(e)}\n# {code}",
                'modifications': [],
                'blocked_patterns': [],
                'safe': False,
                'error': str(e)
            }
    
    def _get_safe_replacement(self, dangerous_code: str) -> str:
        """Get safe replacement for dangerous code patterns"""
        dangerous_lower = dangerous_code.lower().strip()
        
        for pattern, replacement in self.replacement_suggestions.items():
            if pattern in dangerous_lower:
                return f"# BLOCKED: {dangerous_code} -> Use {replacement}"
        
        return f"# BLOCKED: {dangerous_code}"
    
    def _sanitize_python_specific(self, code: str, result: Dict[str, Any]) -> str:
        """Apply Python-specific sanitization"""
        sanitized = code
        
        # Remove or comment out dangerous imports
        dangerous_import_patterns = [
            r'^\s*import\s+(os|subprocess|pickle|marshal)\s*$',
            r'^\s*from\s+(os|subprocess|pickle|marshal)\s+import\s+.*$'
        ]
        
        lines = sanitized.split('\n')
        modified_lines = []
        
        for i, line in enumerate(lines):
            modified_line = line
            
            for pattern in dangerous_import_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    modified_line = f"# BLOCKED IMPORT: {line}"
                    result['modifications'].append({
                        'type': 'comment_out',
                        'line_number': i + 1,
                        'original': line,
                        'replacement': modified_line,
                        'reason': 'dangerous_import'
                    })
                    result['safe'] = False
            
            modified_lines.append(modified_line)
        
        return '\n'.join(modified_lines)
    
    def _sanitize_javascript_specific(self, code: str, result: Dict[str, Any]) -> str:
        """Apply JavaScript-specific sanitization"""
        sanitized = code
        
        # Block dangerous JavaScript patterns
        js_dangerous_patterns = [
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(\s*["\']',
            r'setInterval\s*\(\s*["\']',
            r'document\.write\s*\(',
            r'innerHTML\s*='
        ]
        
        for pattern in js_dangerous_patterns:
            matches = list(re.finditer(pattern, sanitized, re.IGNORECASE))
            
            for match in reversed(matches):
                matched_text = match.group()
                start, end = match.span()
                
                replacement = f"/* BLOCKED: {matched_text} */"
                sanitized = sanitized[:start] + replacement + sanitized[end:]
                
                result['modifications'].append({
                    'type': 'replacement',
                    'original': matched_text,
                    'replacement': replacement,
                    'reason': 'dangerous_js_pattern'
                })
                result['safe'] = False
        
        return sanitized


class CodeGenerationSecurityManager:
    """Comprehensive code generation security management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.python_analyzer = PythonCodeAnalyzer()
        self.sanitizer = CodeSanitizer()
        self.analysis_history: List[CodeAnalysisResult] = []
        self.max_history = 10000
    
    def analyze_generated_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON,
                              sanitize: bool = True) -> Dict[str, Any]:
        """Comprehensive analysis of generated code"""
        try:
            analysis_start = time.time()
            
            # Perform security analysis
            if language == CodeLanguage.PYTHON:
                analysis_result = self.python_analyzer.analyze_code(code)
            else:
                # Placeholder for other language analyzers
                analysis_result = CodeAnalysisResult(
                    analysis_id=f"unsupported_{int(time.time())}",
                    language=language,
                    code_hash=hashlib.sha256(code.encode()).hexdigest(),
                    security_level=CodeSecurityLevel.MEDIUM_RISK,
                    analysis_time_ms=0.0
                )
            
            # Add to analysis history
            self._add_to_analysis_history(analysis_result)
            
            # Perform sanitization if requested
            sanitization_result = None
            if sanitize:
                sanitization_result = self.sanitizer.sanitize_code(code, language)
            
            # Compile comprehensive result
            result = {
                'analysis': {
                    'analysis_id': analysis_result.analysis_id,
                    'language': analysis_result.language.value,
                    'security_level': analysis_result.security_level.value,
                    'security_score': analysis_result.security_score,
                    'violations_count': len(analysis_result.violations),
                    'critical_violations_count': len(analysis_result.critical_violations),
                    'analysis_time_ms': analysis_result.analysis_time_ms,
                    'violations': [
                        {
                            'id': v.violation_id,
                            'type': v.vulnerability_type.value,
                            'severity': v.severity.value,
                            'line': v.line_number,
                            'description': v.description,
                            'recommendation': v.recommendation,
                            'confidence': v.confidence
                        }
                        for v in analysis_result.violations
                    ]
                },
                'sanitization': sanitization_result,
                'recommendation': self._get_overall_recommendation(analysis_result, sanitization_result),
                'approved_for_execution': self._is_safe_for_execution(analysis_result),
                'total_processing_time_ms': (time.time() - analysis_start) * 1000
            }
            
            return result
            
        except Exception as e:
            error = SecurityError(f"Code analysis failed: {str(e)}", "CODE_ANALYSIS_001")
            security_error_handler.handle_error(error)
            
            return {
                'analysis': None,
                'sanitization': None,
                'recommendation': 'REJECTED - Analysis failed',
                'approved_for_execution': False,
                'error': str(e)
            }
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive code generation security statistics"""
        try:
            if not self.analysis_history:
                return {'total_analyses': 0}
            
            total_analyses = len(self.analysis_history)
            
            # Security level distribution
            level_counts = {}
            for analysis in self.analysis_history:
                level = analysis.security_level.value
                level_counts[level] = level_counts.get(level, 0) + 1
            
            # Language distribution
            language_counts = {}
            for analysis in self.analysis_history:
                lang = analysis.language.value
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Recent activity
            recent_analyses = [a for a in self.analysis_history 
                             if (datetime.utcnow() - a.timestamp).total_seconds() < 3600]
            
            # Security metrics
            safe_analyses = sum(1 for a in self.analysis_history 
                              if a.security_level == CodeSecurityLevel.SAFE)
            blocked_analyses = sum(1 for a in self.analysis_history 
                                 if a.security_level == CodeSecurityLevel.BLOCKED)
            
            # Average scores
            avg_security_score = sum(a.security_score for a in self.analysis_history) / total_analyses
            avg_analysis_time = sum(a.analysis_time_ms for a in self.analysis_history) / total_analyses
            
            return {
                'total_analyses': total_analyses,
                'recent_analyses_1h': len(recent_analyses),
                'security_level_distribution': level_counts,
                'language_distribution': language_counts,
                'security_metrics': {
                    'safe_analyses': safe_analyses,
                    'blocked_analyses': blocked_analyses,
                    'safety_rate_pct': (safe_analyses / total_analyses) * 100,
                    'average_security_score': avg_security_score
                },
                'performance_metrics': {
                    'average_analysis_time_ms': avg_analysis_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating code security statistics: {e}")
            return {'error': str(e)}
    
    def _get_overall_recommendation(self, analysis: CodeAnalysisResult, 
                                  sanitization: Dict[str, Any] = None) -> str:
        """Get overall recommendation for generated code"""
        if analysis.security_level == CodeSecurityLevel.SAFE:
            return "APPROVED - Code is safe for execution"
        elif analysis.security_level == CodeSecurityLevel.LOW_RISK:
            return "APPROVED WITH CAUTION - Minor security concerns"
        elif analysis.security_level == CodeSecurityLevel.MEDIUM_RISK:
            return "REVIEW REQUIRED - Moderate security risks detected"
        elif analysis.security_level == CodeSecurityLevel.HIGH_RISK:
            return "REJECTED - High security risks detected"
        elif analysis.security_level == CodeSecurityLevel.CRITICAL_RISK:
            return "REJECTED - Critical security vulnerabilities found"
        elif analysis.security_level == CodeSecurityLevel.BLOCKED:
            return "BLOCKED - Contains dangerous patterns"
        else:
            return "UNKNOWN - Unable to determine safety"
    
    def _is_safe_for_execution(self, analysis: CodeAnalysisResult) -> bool:
        """Determine if code is safe for execution"""
        return analysis.security_level in [
            CodeSecurityLevel.SAFE,
            CodeSecurityLevel.LOW_RISK
        ] and len(analysis.critical_violations) == 0
    
    def _add_to_analysis_history(self, analysis: CodeAnalysisResult):
        """Add analysis result to history"""
        self.analysis_history.append(analysis)
        
        # Limit history size
        if len(self.analysis_history) > self.max_history:
            self.analysis_history = self.analysis_history[-self.max_history // 2:]


# Global code generation security manager
code_generation_security = CodeGenerationSecurityManager()


# Convenience functions
def analyze_code_security(code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> Dict[str, Any]:
    """Convenience function to analyze code security"""
    return code_generation_security.analyze_generated_code(code, language)


def sanitize_generated_code(code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> Dict[str, Any]:
    """Convenience function to sanitize generated code"""
    return code_generation_security.sanitizer.sanitize_code(code, language)