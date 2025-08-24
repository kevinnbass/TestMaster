"""
NASA-STD-8719.13 High-Reliability Compliance Rules Engine
========================================================

This module defines the compliance rules and validation logic for the
autonomous compliance harness. Based on NASA-STD-8719.13 standards
for high-reliability software development.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set, Optional, Tuple
from pathlib import Path
import ast
import re
from enum import Enum


class RuleSeverity(Enum):
    """Severity levels for compliance violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleCategory(Enum):
    """Categories of compliance rules"""
    FUNCTION_SIZE = "function_size"
    DYNAMIC_MEMORY = "dynamic_memory"
    LOOP_BOUNDS = "loop_bounds"
    ERROR_HANDLING = "error_handling"
    TYPE_SAFETY = "type_safety"
    CONTROL_FLOW = "control_flow"
    MODULE_SIZE = "module_size"
    EXTERNAL_DEPS = "external_deps"


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    rule_id: str
    rule_name: str
    description: str
    file_path: str
    line_number: int
    current_code: str
    severity: RuleSeverity
    category: RuleCategory
    high_severity: bool = False
    estimated_complexity: int = 1  # 1-10 scale
    suggested_fix: Optional[str] = None

    @property
    def severity_score(self) -> int:
        """Get numeric severity score"""
        return {
            RuleSeverity.LOW: 1,
            RuleSeverity.MEDIUM: 3,
            RuleSeverity.HIGH: 7,
            RuleSeverity.CRITICAL: 10
        }[self.severity]


@dataclass
class ComplianceRule:
    """Represents a compliance rule"""
    rule_id: str
    name: str
    description: str
    category: RuleCategory
    severity: RuleSeverity
    checker_function: callable
    enabled: bool = True
    max_auto_fix_complexity: int = 5  # Max complexity for auto-fixing


@dataclass
class ComplianceReport:
    """Report of compliance analysis"""
    total_files_analyzed: int = 0
    total_violations: int = 0
    violations_by_severity: Dict[RuleSeverity, int] = None
    violations_by_category: Dict[RuleCategory, int] = None
    violations_by_rule: Dict[str, int] = None
    compliance_score: float = 0.0
    critical_violations: List[ComplianceViolation] = None

    def __post_init__(self):
        if self.violations_by_severity is None:
            self.violations_by_severity = {}
        if self.violations_by_category is None:
            self.violations_by_category = {}
        if self.violations_by_rule is None:
            self.violations_by_rule = {}
        if self.critical_violations is None:
            self.critical_violations = []


class CodeAnalyzer:
    """Analyzes Python code for compliance violations"""

    def __init__(self):
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> List[ComplianceRule]:
        """Initialize all compliance rules"""
        return [
            ComplianceRule(
                rule_id="R1",
                name="Function Size Limit",
                description="Functions must not exceed 60 lines (including docstrings and comments)",
                category=RuleCategory.FUNCTION_SIZE,
                severity=RuleSeverity.HIGH,
                checker_function=self._check_function_size,
                max_auto_fix_complexity=8
            ),
            ComplianceRule(
                rule_id="R2",
                name="Dynamic Object Resizing",
                description="Avoid dynamic resizing of lists/dictionaries after initialization",
                category=RuleCategory.DYNAMIC_MEMORY,
                severity=RuleSeverity.HIGH,
                checker_function=self._check_dynamic_resizing,
                max_auto_fix_complexity=6
            ),
            ComplianceRule(
                rule_id="R3",
                name="Fixed Upper Bounds",
                description="All loops must have fixed upper bounds or use bounded iteration",
                category=RuleCategory.LOOP_BOUNDS,
                severity=RuleSeverity.CRITICAL,
                checker_function=self._check_loop_bounds,
                max_auto_fix_complexity=7
            ),
            ComplianceRule(
                rule_id="R4",
                name="Parameter Validation",
                description="Functions must validate parameters and handle edge cases",
                category=RuleCategory.ERROR_HANDLING,
                severity=RuleSeverity.MEDIUM,
                checker_function=self._check_parameter_validation,
                max_auto_fix_complexity=4
            ),
            ComplianceRule(
                rule_id="R5",
                name="Type Hints",
                description="All functions must include proper type hints",
                category=RuleCategory.TYPE_SAFETY,
                severity=RuleSeverity.MEDIUM,
                checker_function=self._check_type_hints,
                max_auto_fix_complexity=3
            ),
            ComplianceRule(
                rule_id="R6",
                name="Complex Control Flow",
                description="Avoid complex comprehensions and nested conditionals",
                category=RuleCategory.CONTROL_FLOW,
                severity=RuleSeverity.MEDIUM,
                checker_function=self._check_control_flow,
                max_auto_fix_complexity=5
            ),
            ComplianceRule(
                rule_id="R7",
                name="Module Size Limit",
                description="Modules must not exceed 300 lines (500 max)",
                category=RuleCategory.MODULE_SIZE,
                severity=RuleSeverity.MEDIUM,
                checker_function=self._check_module_size,
                max_auto_fix_complexity=9
            ),
            ComplianceRule(
                rule_id="R8",
                name="External Dependencies",
                description="Limit external library dependencies to vetted packages",
                category=RuleCategory.EXTERNAL_DEPS,
                severity=RuleSeverity.LOW,
                checker_function=self._check_external_dependencies,
                max_auto_fix_complexity=2
            )
        ]

    def analyze_file(self, file_path: Path) -> List[ComplianceViolation]:
        """Analyze a single file for compliance violations"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return [ComplianceViolation(
                    rule_id="PARSER",
                    rule_name="Syntax Error",
                    description="File contains syntax errors that prevent analysis",
                    file_path=str(file_path),
                    line_number=0,
                    current_code="",
                    severity=RuleSeverity.CRITICAL,
                    category=RuleCategory.ERROR_HANDLING,
                    high_severity=True,
                    estimated_complexity=1
                )]

            violations = []

            # Apply all enabled rules
            for rule in self.rules:
                if rule.enabled:
                    try:
                        rule_violations = rule.checker_function(file_path, content, tree)
                        violations.extend(rule_violations)
                    except Exception as e:
                        print(f"Error applying rule {rule.rule_id}: {e}")

            return violations

        except Exception as e:
            return [ComplianceViolation(
                rule_id="FILE_ERROR",
                rule_name="File Read Error",
                description=f"Could not read or analyze file: {str(e)}",
                file_path=str(file_path),
                line_number=0,
                current_code="",
                severity=RuleSeverity.CRITICAL,
                category=RuleCategory.ERROR_HANDLING,
                high_severity=True,
                estimated_complexity=1
            )]

    def _check_function_size(self, file_path: Path, content: str, tree: ast.Module) -> List[ComplianceViolation]:
        """Check function size limits"""
        violations = []
        lines = content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', start_line + 1) - 1
                function_lines = end_line - start_line + 1

                if function_lines > 60:
                    # Get the function code
                    function_code = '\n'.join(lines[start_line:end_line + 1])

                    violations.append(ComplianceViolation(
                        rule_id="R1",
                        rule_name="Function Size Limit",
                        description=f"Function '{node.name}' exceeds 60 line limit ({function_lines} lines)",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        current_code=function_code,
                        severity=RuleSeverity.HIGH,
                        category=RuleCategory.FUNCTION_SIZE,
                        high_severity=True,
                        estimated_complexity=8,
                        suggested_fix="Break function into smaller helper functions"
                    ))

        return violations

    def _check_dynamic_resizing(self, file_path: Path, content: str, tree: ast.Module) -> List[ComplianceViolation]:
        """Check for dynamic object resizing"""
        violations = []

        class DynamicResizingVisitor(ast.NodeVisitor):
            def __init__(self, lines):
                self.lines = lines
                self.violations = []

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['append', 'extend', 'insert', 'pop', 'remove', 'clear']:
                        if isinstance(node.func.value, ast.Name):
                            var_name = node.func.value.id
                            line_content = self.lines[node.lineno - 1].strip()
                            if 'append' in line_content or 'extend' in line_content:
                                self.violations.append(ComplianceViolation(
                                    rule_id="R2",
                                    rule_name="Dynamic Object Resizing",
                                    description=f"Dynamic resizing detected: {line_content}",
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    current_code=line_content,
                                    severity=RuleSeverity.HIGH,
                                    category=RuleCategory.DYNAMIC_MEMORY,
                                    high_severity=True,
                                    estimated_complexity=6,
                                    suggested_fix="Use pre-allocated lists with indexed assignment"
                                ))
                self.generic_visit(node)

        visitor = DynamicResizingVisitor(content.split('\n'))
        visitor.visit(tree)

        return visitor.violations

    def _check_loop_bounds(self, file_path: Path, content: str, tree: ast.Module) -> List[ComplianceViolation]:
        """Check for fixed upper bounds in loops"""
        violations = []

        class LoopVisitor(ast.NodeVisitor):
            def __init__(self, lines):
                self.lines = lines
                self.violations = []

            def visit_For(self, node):
                # Check if this is a potentially unbounded loop
                if isinstance(node.iter, ast.Name):
                    # Simple variable iteration - check if it could be unbounded
                    line_content = self.lines[node.lineno - 1].strip()
                    if 'for ' in line_content and ' in ' in line_content:
                        # Look for common unbounded patterns
                        if any(pattern in line_content for pattern in [
                            'for item in ', 'for x in ', 'for _ in '
                        ]):
                            self.violations.append(ComplianceViolation(
                                rule_id="R3",
                                rule_name="Fixed Upper Bounds",
                                description=f"Potentially unbounded loop: {line_content}",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                current_code=line_content,
                                severity=RuleSeverity.CRITICAL,
                                category=RuleCategory.LOOP_BOUNDS,
                                high_severity=True,
                                estimated_complexity=7,
                                suggested_fix="Use range() with fixed bounds or enumerate with len()"
                            ))
                self.generic_visit(node)

        visitor = LoopVisitor(content.split('\n'))
        visitor.visit(tree)

        return visitor.violations

    def _check_parameter_validation(self, file_path: Path, content: str, tree: ast.Module) -> List[ComplianceViolation]:
        """Check for parameter validation"""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_validation = False
                has_assert = False

                # Check function body for validation patterns
                for stmt in node.body:
                    if isinstance(stmt, ast.Assert):
                        has_assert = True
                    elif isinstance(stmt, ast.If):
                        # Look for validation patterns
                        if hasattr(stmt.test, 'left') and isinstance(stmt.test.left, ast.Name):
                            if stmt.test.left.id in [arg.arg for arg in node.args.args]:
                                has_validation = True

                if not (has_validation or has_assert) and node.args.args:
                    violations.append(ComplianceViolation(
                        rule_id="R4",
                        rule_name="Parameter Validation",
                        description=f"Function '{node.name}' lacks parameter validation",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        current_code=f"def {node.name}({', '.join(arg.arg for arg in node.args.args)}):",
                        severity=RuleSeverity.MEDIUM,
                        category=RuleCategory.ERROR_HANDLING,
                        estimated_complexity=4,
                        suggested_fix="Add parameter validation with assert statements"
                    ))

        return violations

    def _check_type_hints(self, file_path: Path, content: str, tree: ast.Module) -> List[ComplianceViolation]:
        """Check for type hints"""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check return type annotation
                if node.returns is None:
                    violations.append(ComplianceViolation(
                        rule_id="R5",
                        rule_name="Type Hints",
                        description=f"Function '{node.name}' missing return type annotation",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        current_code=f"def {node.name}(...):",
                        severity=RuleSeverity.MEDIUM,
                        category=RuleCategory.TYPE_SAFETY,
                        estimated_complexity=3,
                        suggested_fix="Add return type annotation"
                    ))

                # Check parameter type annotations
                for arg in node.args.args:
                    if arg.annotation is None:
                        violations.append(ComplianceViolation(
                            rule_id="R5",
                            rule_name="Type Hints",
                            description=f"Parameter '{arg.arg}' in function '{node.name}' missing type annotation",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            current_code=f"def {node.name}({arg.arg}, ...):",
                            severity=RuleSeverity.MEDIUM,
                            category=RuleCategory.TYPE_SAFETY,
                            estimated_complexity=3,
                            suggested_fix="Add parameter type annotation"
                        ))

        return violations

    def _check_control_flow(self, file_path: Path, content: str, tree: ast.Module) -> List[ComplianceViolation]:
        """Check for complex control flow"""
        violations = []

        # Check for complex comprehensions
        comprehensions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                comprehensions.append(node)

        if len(comprehensions) > 3:  # Allow some simple comprehensions
            violations.append(ComplianceViolation(
                rule_id="R6",
                rule_name="Complex Control Flow",
                description=f"Found {len(comprehensions)} comprehensions - consider using explicit loops",
                file_path=str(file_path),
                line_number=comprehensions[0].lineno if comprehensions else 0,
                current_code="",
                severity=RuleSeverity.MEDIUM,
                category=RuleCategory.CONTROL_FLOW,
                estimated_complexity=5,
                suggested_fix="Replace complex comprehensions with explicit loops"
            ))

        return violations

    def _check_module_size(self, file_path: Path, content: str, tree: ast.Module) -> List[ComplianceViolation]:
        """Check module size limits"""
        violations = []
        lines = content.split('\n')

        if len(lines) > 300:
            violations.append(ComplianceViolation(
                rule_id="R7",
                rule_name="Module Size Limit",
                description=f"Module exceeds 300 line limit ({len(lines)} lines)",
                file_path=str(file_path),
                line_number=1,
                current_code="",
                severity=RuleSeverity.MEDIUM,
                category=RuleCategory.MODULE_SIZE,
                estimated_complexity=9,
                suggested_fix="Split module into smaller focused modules"
            ))

        return violations

    def _check_external_dependencies(self, file_path: Path, content: str, tree: ast.Module) -> List[ComplianceViolation]:
        """Check for external dependencies"""
        violations = []

        # Look for import statements
        dangerous_imports = [
            'subprocess', 'pickle', 'eval', 'exec', 'os.system',
            'requests', 'urllib', 'ftplib', 'telnetlib'
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_imports:
                        violations.append(ComplianceViolation(
                            rule_id="R8",
                            rule_name="External Dependencies",
                            description=f"Potentially risky import: {alias.name}",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            current_code=f"import {alias.name}",
                            severity=RuleSeverity.LOW,
                            category=RuleCategory.EXTERNAL_DEPS,
                            estimated_complexity=2,
                            suggested_fix="Review if this import is necessary"
                        ))

        return violations


class ComplianceEngine:
    """Main compliance engine that orchestrates analysis"""

    def __init__(self):
        self.analyzer = CodeAnalyzer()

    def analyze_codebase(self, directory: Path, exclude_patterns: Optional[List[str]] = None) -> ComplianceReport:
        """Analyze entire codebase for compliance violations"""
        if exclude_patterns is None:
            exclude_patterns = ['**/.*', '**/test*/**', '**/node_modules/**']

        # Find all Python files
        python_files = []
        for pattern in ['**/*.py']:
            python_files.extend(directory.glob(pattern))

        # Filter out excluded files
        def should_analyze(file_path: Path) -> bool:
            for pattern in exclude_patterns:
                if pattern in str(file_path):
                    return False
            return True

        files_to_analyze = [f for f in python_files if should_analyze(f)]

        report = ComplianceReport()
        report.total_files_analyzed = len(files_to_analyze)

        all_violations = []

        print(f"🔍 Analyzing {len(files_to_analyze)} Python files...")

        for file_path in files_to_analyze:
            violations = self.analyzer.analyze_file(file_path)
            all_violations.extend(violations)

        report.total_violations = len(all_violations)
        report.critical_violations = [v for v in all_violations if v.severity == RuleSeverity.CRITICAL]

        # Categorize violations
        for violation in all_violations:
            report.violations_by_severity[violation.severity] = \
                report.violations_by_severity.get(violation.severity, 0) + 1
            report.violations_by_category[violation.category] = \
                report.violations_by_category.get(violation.category, 0) + 1
            report.violations_by_rule[violation.rule_id] = \
                report.violations_by_rule.get(violation.rule_id, 0) + 1

        # Calculate compliance score
        if all_violations:
            total_severity = sum(v.severity_score for v in all_violations)
            max_possible = len(all_violations) * 10
            report.compliance_score = 1.0 - (total_severity / max_possible)
        else:
            report.compliance_score = 1.0

        return report

    def generate_fix_suggestions(self, violations: List[ComplianceViolation]) -> Dict[str, List[str]]:
        """Generate fix suggestions for violations"""
        suggestions = {}

        for violation in violations:
            if violation.rule_id not in suggestions:
                suggestions[violation.rule_id] = []

            if violation.suggested_fix:
                suggestions[violation.rule_id].append(violation.suggested_fix)

        return suggestions


# Global instance
compliance_engine = ComplianceEngine()

# NASA-STD-8719.13 Rules (simplified)
NASA_STD_8719_13_RULES = compliance_engine.analyzer.rules

# Export key classes
__all__ = [
    'ComplianceRule',
    'ComplianceViolation',
    'ComplianceReport',
    'RuleSeverity',
    'RuleCategory',
    'CodeAnalyzer',
    'ComplianceEngine',
    'NASA_STD_8719_13_RULES',
    'compliance_engine'
]

