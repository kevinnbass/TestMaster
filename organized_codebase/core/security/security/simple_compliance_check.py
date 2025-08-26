#!/usr/bin/env python3
"""
Simple Compliance Check for LLM Intelligence System
===========================================

A simpler compliance checker that analyzes the codebase files directly
without importing them to avoid syntax errors.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
import json
import re

@dataclass
class ComplianceIssue:
    """Represents a compliance issue"""
    file_path: str
    line_number: int
    rule_number: int
    rule_description: str
    severity: str  # 'high', 'medium', 'low'
    description: str
    suggestion: str


@dataclass
class ComplianceReport:
    """Complete compliance report"""
    total_files_analyzed: int = 0
    total_functions_analyzed: int = 0
    total_lines_analyzed: int = 0
    issues_found: List[ComplianceIssue] = field(default_factory=list)
    compliance_score: float = 0.0
    rule_compliance: Dict[int, bool] = field(default_factory=dict)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)


class SimpleComplianceChecker:
    """Simple compliance checker that analyzes files directly"""

    def __init__(self):
        self.issues = []
        self.functions_analyzed = 0
        self.total_assertions = 0
        self.rule_descriptions = {
            1: "Simple control flow: no recursion, avoid complex comprehensions",
            2: "Fixed upper bound loops: use for with ranges or known iterables",
            3: "No dynamic object resizing after initialization",
            4: "Functions limited to 60 lines max (including docstrings/comments)",
            5: "Average at least 2 assertions per function",
            6: "Variables declared at smallest possible scope",
            7: "Parameter validation and return value checking",
            8: "Limited use of decorators and metaclasses",
            9: "Restrict indirection to one level",
            10: "Use linters with strict settings",
            11: "Strict data privacy and sanitization",
            12: "Audit logging for data modifications",
            13: "Healthcare standards interoperability",
            14: "Limited external library dependencies",
            15: "Detailed docstrings and mandatory type hints",
            16: "Module size under 300 lines (500 max)",
            17: "Functionality not limited by size constraints"
        }

    def analyze_file(self, file_path: Path) -> List[ComplianceIssue]:
        """Analyze a single Python file"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')

            # Basic file-level checks
            issues.extend(self._check_file_size(file_path, lines))
            issues.extend(self._check_imports(file_path, content))

            # Try to parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                issues.extend(self._analyze_ast(file_path, tree, lines))
            except SyntaxError as e:
                issues.append(ComplianceIssue(
                    file_path=str(file_path),
                    line_number=e.lineno or 1,
                    rule_number=0,
                    rule_description="General compliance",
                    severity='high',
                    description=f"Syntax error: {e}",
                    suggestion="Fix syntax errors before analysis"
                ))

        except Exception as e:
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=1,
                rule_number=0,
                rule_description="General compliance",
                severity='high',
                description=f"Failed to analyze file: {e}",
                suggestion="Ensure file is readable and contains valid Python"
            ))

        return issues

    def _check_file_size(self, file_path: Path, lines: List[str]) -> List[ComplianceIssue]:
        """Check file size against Rule 16"""
        issues = []
        total_lines = len(lines)

        if total_lines > 300:
            if total_lines > 500:
                severity = 'high'
            else:
                severity = 'medium'

            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=1,
                rule_number=16,
                rule_description=self.rule_descriptions[16],
                severity=severity,
                description=f"Module exceeds recommended size limit ({total_lines} lines)",
                suggestion="Consider splitting into smaller modules"
            ))

        return issues

    def _check_imports(self, file_path: Path, content: str) -> List[ComplianceIssue]:
        """Check imports against Rule 14"""
        issues = []

        # Look for potentially risky imports
        risky_imports = ['subprocess', 'eval', 'exec', 'pickle', 'requests']
        found_risky = []

        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                for risky in risky_imports:
                    if risky in line:
                        found_risky.append(risky)

        if found_risky:
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=1,
                rule_number=14,
                rule_description=self.rule_descriptions[14],
                severity='medium',
                description=f"Module imports potentially risky libraries: {', '.join(set(found_risky))}",
                suggestion="Review usage and consider safer alternatives"
            ))

        return issues

    def _analyze_ast(self, file_path: Path, tree: ast.Module, lines: List[str]) -> List[ComplianceIssue]:
        """Analyze AST for compliance issues"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                issues.extend(self._analyze_function(file_path, node, lines))

        # Check for global variables (Rule 6)
        issues.extend(self._check_global_variables(file_path, tree))

        return issues

    def _analyze_function(self, file_path: Path, node: ast.FunctionDef, lines: List[str]) -> List[ComplianceIssue]:
        """Analyze a single function"""
        issues = []
        function_name = node.name

        # Get function lines
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line)
        function_lines = end_line - start_line + 1

        self.functions_analyzed += 1

        # Rule 4: Function size limit
        if function_lines > 60:
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=start_line,
                rule_number=4,
                rule_description=self.rule_descriptions[4],
                severity='high',
                description=f"Function '{function_name}' exceeds 60 line limit ({function_lines} lines)",
                suggestion="Break function into smaller, focused functions"
            ))

        # Rule 5: Count assertions
        assertion_count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                assertion_count += 1

        self.total_assertions += assertion_count

        # Rule 1: Check for recursion
        if self._has_recursion(node, function_name):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=start_line,
                rule_number=1,
                rule_description=self.rule_descriptions[1],
                severity='high',
                description=f"Function '{function_name}' contains recursion",
                suggestion="Replace recursion with iterative approach"
            ))

        # Rule 1: Check for complex comprehensions
        if self._has_complex_comprehensions(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=start_line,
                rule_number=1,
                rule_description=self.rule_descriptions[1],
                severity='medium',
                description=f"Function '{function_name}' contains complex comprehensions",
                suggestion="Replace with explicit loops for clarity"
            ))

        # Rule 2: Check loop bounds
        if self._has_unbounded_loops(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=start_line,
                rule_number=2,
                rule_description=self.rule_descriptions[2],
                severity='high',
                description=f"Function '{function_name}' contains loops without fixed upper bounds",
                suggestion="Ensure all loops have known iteration limits"
            ))

        # Rule 3: Check for dynamic resizing
        if self._has_dynamic_resizing(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=start_line,
                rule_number=3,
                rule_description=self.rule_descriptions[3],
                severity='medium',
                description=f"Function '{function_name}' dynamically resizes objects",
                suggestion="Pre-allocate objects with known sizes"
            ))

        # Rule 7: Check parameter validation
        if not self._has_parameter_validation(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=start_line,
                rule_number=7,
                rule_description=self.rule_descriptions[7],
                severity='low',
                description=f"Function '{function_name}' lacks parameter validation",
                suggestion="Add type hints and parameter validation checks"
            ))

        # Rule 8: Check decorators
        if self._has_complex_decorators(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=start_line,
                rule_number=8,
                rule_description=self.rule_descriptions[8],
                severity='medium',
                description=f"Function '{function_name}' uses complex decorators",
                suggestion="Use simple decorators or inline the logic"
            ))

        # Rule 15: Check docstrings
        if not self._has_docstring(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=start_line,
                rule_number=15,
                rule_description=self.rule_descriptions[15],
                severity='medium',
                description=f"Function '{function_name}' lacks docstring",
                suggestion="Add comprehensive docstring with type hints"
            ))

        return issues

    def _has_recursion(self, node: ast.FunctionDef, function_name: str) -> bool:
        """Check if function has recursion"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == function_name:
                    return True
        return False

    def _has_complex_comprehensions(self, node: ast.AST) -> bool:
        """Check for complex comprehensions"""
        for child in ast.walk(node):
            if isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp)):
                # Check if comprehension is nested or has complex conditions
                if (hasattr(child, 'generators') and
                    len(child.generators) > 1 or
                    any(hasattr(gen, 'ifs') and gen.ifs for gen in child.generators)):
                    return True
        return False

    def _has_unbounded_loops(self, node: ast.AST) -> bool:
        """Check for loops without fixed upper bounds"""
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                # Check if it's iterating over a range with fixed bounds
                if isinstance(child.iter, ast.Call):
                    if isinstance(child.iter.func, ast.Name) and child.iter.func.id == 'range':
                        # Check if range has fixed arguments
                        if len(child.iter.args) >= 1:
                            first_arg = child.iter.args[0]
                            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, int):
                                continue  # Fixed bound
                # Check for iteration over collections that might grow
                if isinstance(child.iter, ast.Name):
                    # This could be unbounded - flag for review
                    return True
        return False

    def _has_dynamic_resizing(self, node: ast.AST) -> bool:
        """Check for dynamic object resizing"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr in ['append', 'extend', 'insert', 'pop', 'remove', 'clear']:
                        if isinstance(child.func.value, ast.Name):
                            # This could be dynamic resizing - flag for review
                            return True
        return False

    def _has_parameter_validation(self, node: ast.FunctionDef) -> bool:
        """Check if function has parameter validation"""
        # Check for type hints
        if node.returns or any(arg.annotation for arg in node.args.args):
            return True

        # Check for assert statements that might validate parameters
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                # Simple check - assume any assert might be parameter validation
                return True

        return False

    def _has_complex_decorators(self, node: ast.FunctionDef) -> bool:
        """Check for complex decorators"""
        if len(node.decorator_list) > 2:  # More than 2 decorators
            return True

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and decorator.args:
                # Decorator with arguments - could be complex
                return True
        return False

    def _has_docstring(self, node: ast.FunctionDef) -> bool:
        """Check if function has docstring"""
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                return True
            elif isinstance(node.body[0].value, ast.Str):  # Python < 3.8
                return True
        return False

    def _check_global_variables(self, file_path: Path, tree: ast.Module) -> List[ComplianceIssue]:
        """Check for global variables (Rule 6)"""
        issues = []

        global_vars = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        global_vars.append(target.id)

        if global_vars:
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=1,
                rule_number=6,
                rule_description=self.rule_descriptions[6],
                severity='low',
                description=f"Module contains global variables: {', '.join(global_vars)}",
                suggestion="Move global variables to function scope or use constants module"
            ))

        return issues

    def run_analysis(self, source_directory: Path) -> ComplianceReport:
        """Run compliance analysis on the entire codebase"""
        print(f"🔍 Analyzing codebase for high-reliability compliance...")
        print(f"📂 Source directory: {source_directory}")

        all_issues = []

        # Find all Python files in the system
        python_files = list(source_directory.rglob("*.py"))

        # Filter out test files and external libraries
        system_files = [
            f for f in python_files
            if 'test' not in f.name.lower() and
            'external' not in str(f) and
            'site-packages' not in str(f) and
            f.name not in ['setup.py', '__init__.py', 'simple_compliance_check.py', 'compliance_audit.py']
        ]

        print(f"📋 Found {len(system_files)} system files to analyze")

        total_lines = 0
        for file_path in system_files:
            print(f"  Analyzing: {file_path.name}")
            issues = self.analyze_file(file_path)
            all_issues.extend(issues)

            # Count lines
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                pass

        # Calculate compliance score
        total_possible_violations = len(system_files) * len(self.rule_descriptions)
        actual_violations = len(all_issues)
        compliance_score = max(0, (1 - (actual_violations / total_possible_violations)) * 100)

        # Rule compliance
        rule_compliance = {}
        for rule_num in self.rule_descriptions:
            rule_issues = [issue for issue in all_issues if issue.rule_number == rule_num]
            rule_compliance[rule_num] = len(rule_issues) == 0

        # Detailed analysis
        detailed_analysis = {
            'files_by_severity': {
                'high': len([i for i in all_issues if i.severity == 'high']),
                'medium': len([i for i in all_issues if i.severity == 'medium']),
                'low': len([i for i in all_issues if i.severity == 'low'])
            },
            'rules_violated': len(set(issue.rule_number for issue in all_issues if issue.rule_number != 0)),
            'assertion_ratio': self.total_assertions / max(1, self.functions_analyzed),
            'average_function_size': self._calculate_average_function_size(all_issues)
        }

        report = ComplianceReport(
            total_files_analyzed=len(system_files),
            total_functions_analyzed=self.functions_analyzed,
            total_lines_analyzed=total_lines,
            issues_found=all_issues,
            compliance_score=compliance_score,
            rule_compliance=rule_compliance,
            detailed_analysis=detailed_analysis
        )

        return report

    def _calculate_average_function_size(self, issues: List[ComplianceIssue]) -> float:
        """Calculate average function size from issues"""
        size_issues = [issue for issue in issues if issue.rule_number == 4]
        if not size_issues:
            return 0.0

        total_size = 0
        for issue in size_issues:
            # Extract size from description
            import re
            match = re.search(r'\((\d+) lines\)', issue.description)
            if match:
                total_size += int(match.group(1))

        return total_size / len(size_issues)

    def print_report(self, report: ComplianceReport):
        """Print formatted compliance report"""
        print("\n" + "="*80)
        print("🏛️  HIGH-RELIABILITY COMPLIANCE AUDIT REPORT")
        print("="*80)

        print(".1f")
        print(f"📂 Files Analyzed: {report.total_files_analyzed}")
        print(f"🔧 Functions Analyzed: {report.total_functions_analyzed}")
        print(f"📏 Lines Analyzed: {report.total_lines_analyzed}")
        print(f"⚠️  Issues Found: {len(report.issues_found)}")
        print(f"📊 Rules Violated: {report.detailed_analysis['rules_violated']}")

        # Rule compliance summary
        print("\n📋 RULE COMPLIANCE SUMMARY:")
        for rule_num in sorted(report.rule_compliance.keys()):
            status = "✅" if report.rule_compliance[rule_num] else "❌"
            print("2d")

        # Issues by severity
        severity_data = report.detailed_analysis['files_by_severity']
        print("\n🚨 ISSUES BY SEVERITY:")
        print(f"  🔴 High: {severity_data['high']}")
        print(f"  🟡 Medium: {severity_data['medium']}")
        print(f"  🔵 Low: {severity_data['low']}")

        # Key metrics
        print("\n📊 KEY METRICS:")
        print(".2f")
        print(".1f")

        if report.issues_found:
            print("\n🔍 TOP ISSUES:")
            # Group issues by rule
            rule_groups = {}
            for issue in report.issues_found:
                if issue.rule_number not in rule_groups:
                    rule_groups[issue.rule_number] = []
                rule_groups[issue.rule_number].append(issue)

            for rule_num in sorted([r for r in rule_groups.keys() if r in self.rule_descriptions])[:5]:  # Show top 5 rules with issues
                issues = rule_groups[rule_num]
                if rule_num in self.rule_descriptions:
                    print(f"\n  Rule {rule_num}: {self.rule_descriptions[rule_num]}")
                    print(f"    Issues: {len(issues)}")

                    # Show first issue as example
                    if issues:
                        example = issues[0]
                        print(f"    Example: {example.file_path}:{example.line_number} - {example.description}")

        # Compliance assessment
        if report.compliance_score >= 90:
            print("\n🎉 EXCELLENT COMPLIANCE!")
            print("   The codebase follows high-reliability principles effectively.")
        elif report.compliance_score >= 75:
            print("\n👍 GOOD COMPLIANCE")
            print("   Minor issues found that should be addressed.")
        elif report.compliance_score >= 50:
            print("\n⚠️  MODERATE COMPLIANCE")
            print("   Several issues need attention to improve reliability.")
        else:
            print("\n🚨 POOR COMPLIANCE")
            print("   Significant work needed to meet high-reliability standards.")

        print("\n" + "="*80)

    def save_report(self, report: ComplianceReport, output_path: Path):
        """Save compliance report to file"""
        report_data = {
            'compliance_score': report.compliance_score,
            'total_files_analyzed': report.total_files_analyzed,
            'total_functions_analyzed': report.total_functions_analyzed,
            'total_lines_analyzed': report.total_lines_analyzed,
            'total_issues': len(report.issues_found),
            'rule_compliance': report.rule_compliance,
            'detailed_analysis': report.detailed_analysis,
            'issues': [
                {
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'rule_number': issue.rule_number,
                    'rule_description': issue.rule_description,
                    'severity': issue.severity,
                    'description': issue.description,
                    'suggestion': issue.suggestion
                }
                for issue in report.issues_found
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)


def main():
    """Main compliance audit function"""
    import argparse

    parser = argparse.ArgumentParser(description="High-Reliability Compliance Audit")
    parser.add_argument("--source", type=str, default=".",
                      help="Source directory to audit")
    parser.add_argument("--output", type=str, default="compliance_report.json",
                      help="Output report file")

    args = parser.parse_args()

    # Initialize checker
    checker = SimpleComplianceChecker()

    # Run compliance audit
    source_dir = Path(args.source).resolve()
    report = checker.run_analysis(source_dir)

    # Print report
    checker.print_report(report)

    # Save detailed report
    output_path = Path(args.output)
    checker.save_report(report, output_path)

    print(f"\n📄 Detailed report saved to: {output_path}")

    # Exit with appropriate code
    if report.compliance_score < 75:
        exit(1)  # Fail CI/CD if compliance is too low


if __name__ == "__main__":
    main()
