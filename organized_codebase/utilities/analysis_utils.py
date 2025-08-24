from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
TestMaster Analysis Utilities

Domain-specific analysis utilities that implement the functionality
that was stubbed in the analysis subdirectory _shared_utils.py files.
"""

import ast
import re
from typing import Dict, Any, List, Optional
from ..core.shared_utils import (
    AnalysisIssue, 
    BUSINESS_ANALYSIS_PATTERNS,
    SEMANTIC_ANALYSIS_PATTERNS,
    DEBT_ANALYSIS_PATTERNS,
    METAPROG_ANALYSIS_PATTERNS,
    ENERGY_ANALYSIS_PATTERNS
)
from ..core.pattern_utilities import extract_common_patterns, calculate_complexity_score


class AnalysisEngine:
    """Base analysis engine for all domain-specific analyzers"""
    
    def __init__(self, patterns: Dict[str, List[str]] = None):
        self.patterns = patterns or {}
        self.issues = []
        self.metrics = {}
    
    def analyze_tree(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Analyze AST tree and content for patterns"""
        results = {
            "issues": [],
            "patterns": [],
            "metrics": {}
        }
        
        # Extract patterns
        patterns = extract_common_patterns(tree, content)
        results["patterns"] = patterns
        
        # Calculate metrics
        complexity = calculate_complexity_score(tree)
        results["metrics"]["complexity"] = complexity
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        return {
            "issues": self.issues,
            "metrics": self.metrics,
            "summary": {
                "total_issues": len(self.issues),
                "high_severity": len([i for i in self.issues if i.severity == "high"]),
                "medium_severity": len([i for i in self.issues if i.severity == "medium"]),
                "low_severity": len([i for i in self.issues if i.severity == "low"])
            }
        }


class BusinessAnalysisUtils(AnalysisEngine):
    """Business logic analysis utilities"""
    
    def __init__(self):
        super().__init__(BUSINESS_ANALYSIS_PATTERNS)
    
    def analyze_business_logic(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze business logic patterns"""
        issues = []
        
        # Look for business logic patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for business rule violations
                if self._has_mixed_concerns(node):
                    issues.append(AnalysisIssue(
                        issue_type="mixed_concerns",
                        severity="medium",
                        location=f"Line {node.lineno}",
                        description=f"Function '{node.name}' mixes business logic with other concerns",
                        recommendation="Separate business logic into dedicated functions",
                        impact="Reduces maintainability and testability"
                    ))
                
                # Check for hardcoded business rules
                if self._has_hardcoded_rules(node, content):
                    issues.append(AnalysisIssue(
                        issue_type="hardcoded_business_rule",
                        severity="high",
                        location=f"Line {node.lineno}",
                        description=f"Function '{node.name}' contains hardcoded business rules",
                        recommendation="Extract business rules to configuration or rule engine",
                        impact="Makes business rule changes difficult"
                    ))
        
        self.issues.extend(issues)
        return issues
    
    def _has_mixed_concerns(self, node: ast.FunctionDef) -> bool:
        """Check if function mixes business logic with other concerns"""
        has_business_logic = False
        has_io_operations = False
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_str = ast.unparse(child) if hasattr(ast, 'unparse') else str(child)
                
                # Business logic indicators
                if any(pattern in call_str.lower() for pattern in ["calculate", "validate", "process", "compute"]):
                    has_business_logic = True
                
                # I/O operation indicators
                if any(pattern in call_str.lower() for pattern in ["print", "open", "write", "read", "request"]):
                    has_io_operations = True
        
        return has_business_logic and has_io_operations
    
    def _has_hardcoded_rules(self, node: ast.FunctionDef, content: str) -> bool:
        """Check for hardcoded business rules"""
        function_content = content.split('\n')[node.lineno-1:node.end_lineno]
        function_text = '\n'.join(function_content)
        
        # Look for hardcoded values that might be business rules
        hardcoded_patterns = [
            r'\d+\.\d+',  # Decimal numbers (rates, percentages)
            r'["\'][\w\s]+["\'].*[<>=]',  # String comparisons
            r'if.*\d+.*:',  # Numeric comparisons
        ]
        
        for pattern in hardcoded_patterns:
            if re.search(pattern, function_text):
                return True
        
        return False


class SemanticAnalysisUtils(AnalysisEngine):
    """Semantic analysis utilities"""
    
    def __init__(self):
        super().__init__(SEMANTIC_ANALYSIS_PATTERNS)
    
    def analyze_naming_semantics(self, tree: ast.AST) -> List[AnalysisIssue]:
        """Analyze semantic meaning in naming"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Check for meaningful names
                if not self._is_meaningful_name(node.name):
                    issues.append(AnalysisIssue(
                        issue_type="meaningless_name",
                        severity="medium",
                        location=f"Line {node.lineno}",
                        description=f"Name '{node.name}' lacks semantic meaning",
                        recommendation="Use descriptive, domain-specific names",
                        impact="Reduces code readability and maintainability"
                    ))
                
                # Check for domain consistency
                if not self._follows_domain_conventions(node.name):
                    issues.append(AnalysisIssue(
                        issue_type="domain_inconsistency", 
                        severity="low",
                        location=f"Line {node.lineno}",
                        description=f"Name '{node.name}' doesn't follow domain conventions",
                        recommendation="Use domain-specific terminology consistently",
                        impact="Reduces domain understanding"
                    ))
        
        self.issues.extend(issues)
        return issues
    
    def _is_meaningful_name(self, name: str) -> bool:
        """Check if name is semantically meaningful"""
        meaningless_patterns = [
            r'^[a-z]+\d+$',  # Like 'data1', 'func2'
            r'^(temp|tmp|var|val|obj|item)\d*$',  # Generic names
            r'^[a-z]{1,2}$',  # Single/double letters
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, name, re.IGNORECASE):
                return False
        
        return len(name) > 2
    
    def _follows_domain_conventions(self, name: str) -> bool:
        """Check if name follows domain conventions"""
        # This is a simplified check - in practice, would use domain vocabulary
        common_domain_terms = [
            "user", "account", "order", "product", "payment", "invoice",
            "customer", "inventory", "shipping", "billing", "report"
        ]
        
        name_lower = name.lower()
        return any(term in name_lower for term in common_domain_terms)


class DebtAnalysisUtils(AnalysisEngine):
    """Technical debt analysis utilities"""
    
    def __init__(self):
        super().__init__(DEBT_ANALYSIS_PATTERNS)
    
    def analyze_technical_debt(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze technical debt indicators"""
        issues = []
        
        # Look for code smells
        code_smells = self._detect_code_smells(tree)
        issues.extend(code_smells)
        
        # Look for TODO/FIXME comments
        todo_debt = self._analyze_todo_comments(content)
        issues.extend(todo_debt)
        
        # Check for outdated patterns
        outdated_patterns = self._detect_outdated_patterns(content)
        issues.extend(outdated_patterns)
        
        self.issues.extend(issues)
        return issues
    
    def _detect_code_smells(self, tree: ast.AST) -> List[AnalysisIssue]:
        """Detect common code smells"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Long method
                if len(node.body) > 25:
                    issues.append(AnalysisIssue(
                        issue_type="long_method",
                        severity="medium",
                        location=f"Line {node.lineno}",
                        description=f"Method '{node.name}' is too long ({len(node.body)} lines)",
                        recommendation="Break down into smaller, focused methods",
                        impact="Reduces readability and maintainability"
                    ))
                
                # Too many parameters
                if len(node.args.args) > 6:
                    issues.append(AnalysisIssue(
                        issue_type="long_parameter_list",
                        severity="medium",
                        location=f"Line {node.lineno}",
                        description=f"Method '{node.name}' has too many parameters ({len(node.args.args)})",
                        recommendation="Use parameter objects or configuration",
                        impact="Makes function calls complex and error-prone"
                    ))
        
        return issues
    
    def _analyze_todo_comments(self, content: str) -> List[AnalysisIssue]:
        """Analyze TODO/FIXME comments as debt indicators"""
        issues = []
        lines = content.split('\n')
        
        debt_patterns = [
            (r'#.*TODO', "todo_comment", "low"),
            (r'#.*FIXME', "fixme_comment", "medium"), 
            (r'#.*HACK', "hack_comment", "high"),
            (r'#.*XXX', "xxx_comment", "medium")
        ]
        
        for i, line in enumerate(lines):
            for pattern, issue_type, severity in debt_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(AnalysisIssue(
                        issue_type=issue_type,
                        severity=severity,
                        location=f"Line {i + 1}",
                        description=f"Technical debt comment: {line.strip()}",
                        recommendation="Address the underlying issue",
                        impact="Indicates incomplete or problematic code"
                    ))
        
        return issues
    
    def _detect_outdated_patterns(self, content: str) -> List[AnalysisIssue]:
        """Detect outdated coding patterns"""
        issues = []
        
        # Python 2 patterns
        if "print " in content and "from __future__" not in content:
            issues.append(AnalysisIssue(
                issue_type="python2_print", 
                severity="medium",
                location="Multiple locations",
                description="Using Python 2 print statements",
                recommendation="Upgrade to Python 3 print function",
                impact="Code not compatible with Python 3"
            ))
        
        return issues


class MetaprogAnalysisUtils(AnalysisEngine):
    """Metaprogramming analysis utilities"""
    
    def __init__(self):
        super().__init__(METAPROG_ANALYSIS_PATTERNS)
    
    def analyze_metaprogramming(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze metaprogramming usage"""
        issues = []
        
        for node in ast.walk(tree):
            # Dynamic attribute access
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'id') and node.func.id in ['getattr', 'setattr', 'hasattr']:
                    issues.append(AnalysisIssue(
                        issue_type="dynamic_attribute_access",
                        severity="low",
                        location=f"Line {node.lineno}",
                        description="Dynamic attribute access detected",
                        recommendation="Consider if static access is possible",
                        impact="May reduce IDE support and static analysis"
                    ))
            
            # eval/exec usage (should be caught by security patches)
            elif isinstance(node, ast.Call):
                if hasattr(node.func, 'id') and node.func.id in ['eval', 'exec']:
                    issues.append(AnalysisIssue(
                        issue_type="eval_exec_usage",
                        severity="high",
                        location=f"Line {node.lineno}",
                        description=f"Use of {node.func.id}() detected",
                        recommendation="Replace with safer alternatives",
                        impact="Security risk and performance impact"
                    ))
        
        self.issues.extend(issues)
        return issues


class EnergyAnalysisUtils(AnalysisEngine):
    """Energy efficiency analysis utilities"""
    
    def __init__(self):
        super().__init__(ENERGY_ANALYSIS_PATTERNS)
    
    def analyze_energy_efficiency(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze code for energy efficiency"""
        issues = []
        
        # Look for inefficient patterns
        for node in ast.walk(tree):
            # Nested loops
            if isinstance(node, (ast.For, ast.While)):
                nested_loops = self._count_nested_loops(node)
                if nested_loops > 2:
                    issues.append(AnalysisIssue(
                        issue_type="nested_loops",
                        severity="medium",
                        location=f"Line {node.lineno}",
                        description=f"Deeply nested loops detected ({nested_loops} levels)",
                        recommendation="Consider algorithm optimization or caching",
                        impact="High CPU usage and energy consumption"
                    ))
            
            # String concatenation in loops
            if isinstance(node, ast.For):
                if self._has_string_concat_in_loop(node):
                    issues.append(AnalysisIssue(
                        issue_type="string_concat_in_loop",
                        severity="medium",
                        location=f"Line {node.lineno}",
                        description="String concatenation in loop detected",
                        recommendation="Use join() or list accumulation instead",
                        impact="Inefficient memory usage and CPU overhead"
                    ))
        
        self.issues.extend(issues)
        return issues
    
    def _count_nested_loops(self, node: ast.AST, depth: int = 1) -> int:
        """Count nested loop depth"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                child_depth = self._count_nested_loops(child, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _has_string_concat_in_loop(self, node: ast.For) -> bool:
        """Check for string concatenation in loop"""
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                # Check if target might be string
                return True  # Simplified check
        return False