#!/usr/bin/env python3
"""
Compliance Audit Coordinator
===========================

Audits the codebase against high-reliability software development rules.
"""

import ast
from pathlib import Path
from typing import Dict, List, Any, Final

# Import specialized modules
from compliance_audit_data import ComplianceIssue, ComplianceReport
from compliance_audit_rules import ComplianceRuleChecker
from compliance_audit_reporting import ComplianceReportGenerator

# Import our system modules for analysis
try:
    from llm_intelligence_system import *
    from intelligence_integration_engine import *
    from reorganization_planner import *
    from run_intelligence_system import *
    from test_intelligence_system import *
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available for analysis: {e}")
    MODULES_AVAILABLE = False

# Constants
MAX_AST_NODES: Final[int] = 5000  # Safety bound for AST processing


class HighReliabilityAuditor:
    """Audits codebase against high-reliability software development rules"""

    def __init__(self):
        """Initialize the auditor with rule descriptions and checker"""
        self.rule_descriptions = {
            1: "Simple control flow - no recursion, complex comprehensions",
            2: "Fixed upper bounds for all loops",
            3: "No dynamic object resizing after initialization",
            4: "Functions must not exceed 60 lines",
            5: "Parameter validation and return value checking",
            6: "No complex decorators or metaclasses",
            7: "Detailed docstrings for all modules and functions",
            8: "Type hints using typing module",
            9: "Limited external library dependencies",
            10: "Data privacy and sanitization",
            11: "Audit logging for data modifications",
            12: "Interoperability with healthcare standards",
            13: "Module size limits (300/500 lines)",
            14: "Linter compliance without warnings",
            15: "Detailed type hints and docstrings",
            16: "Module size under 300 lines (500 max)"
        }

        self.rule_checker = ComplianceRuleChecker()
        self.report_generator = ComplianceReportGenerator(self.rule_descriptions)

    def audit_file(self, file_path: Path) -> List[ComplianceIssue]:
        """Audit a single file for compliance issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            issues = []

            # Audit module-level compliance
            module_issues = self._audit_module(tree, file_path, content)
            issues.extend(module_issues)

            # Audit function-level compliance
            function_issues = self._audit_functions_in_file(tree, file_path, content)
            issues.extend(function_issues)

            return issues

        except Exception as e:
            return [ComplianceIssue(
                file_path=str(file_path),
                line_number=1,
                rule_number=0,
                rule_description="File parsing error",
                severity="medium",
                description=f"Could not parse file: {e}",
                suggestion="Check file syntax and encoding"
            )]

    def _audit_module(self, tree: ast.Module, file_path: Path, content: str) -> List[ComplianceIssue]:
        """Audit module-level compliance"""
        issues = []

        # Check for module docstring
        if not ast.get_docstring(tree):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=1,
                rule_number=7,
                rule_description=self.rule_descriptions[7],
                severity="high",
                description="Module missing docstring",
                suggestion="Add module-level docstring explaining the module's purpose"
            ))

        return issues

    def _audit_functions_in_file(self, tree: ast.Module, file_path: Path, content: str) -> List[ComplianceIssue]:
        """Audit all functions in a file for compliance"""
        issues = []

        # Bounded loop for AST node processing
        nodes_list = list(ast.walk(tree))
        for i in range(min(len(nodes_list), MAX_AST_NODES)):
            node = nodes_list[i]
            if isinstance(node, ast.FunctionDef):
                function_issues = self._audit_function_compliance(node, file_path, content)
                issues.extend(function_issues)

        return issues

    def _audit_function_compliance(self, node: ast.FunctionDef, file_path: Path, content: str) -> List[ComplianceIssue]:
        """Audit a single function for compliance issues"""
        issues = []
        function_name = f"{file_path.name}:{node.name}"
        line_number = node.lineno

        # Rule 1: Check for recursion
        if self.rule_checker.has_recursion(node, node.name):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=line_number,
                rule_number=1,
                rule_description=self.rule_descriptions[1],
                severity="high",
                description=f"Function '{node.name}' contains recursion",
                suggestion="Replace recursion with iteration to ensure bounded execution"
            ))

        # Rule 1: Check for complex comprehensions
        if self.rule_checker.has_complex_comprehensions(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=line_number,
                rule_number=1,
                rule_description=self.rule_descriptions[1],
                severity="medium",
                description=f"Function '{node.name}' contains complex comprehensions",
                suggestion="Replace complex comprehensions with explicit loops for better readability"
            ))

        # Rule 2: Check for unbounded loops
        if self.rule_checker.has_unbounded_loops(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=line_number,
                rule_number=2,
                rule_description=self.rule_descriptions[2],
                severity="critical",
                description=f"Function '{node.name}' contains loops without fixed upper bounds",
                suggestion="Ensure all loops have fixed upper bounds or use bounded iteration patterns"
            ))

        # Rule 3: Check for dynamic resizing
        if self.rule_checker.has_dynamic_resizing(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=line_number,
                rule_number=3,
                rule_description=self.rule_descriptions[3],
                severity="high",
                description=f"Function '{node.name}' contains dynamic object resizing",
                suggestion="Pre-allocate lists/dictionaries with known capacity to avoid dynamic resizing"
            ))

        # Rule 4: Check function size limit
        if self.rule_checker.exceeds_function_size_limit(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=line_number,
                rule_number=4,
                rule_description=self.rule_descriptions[4],
                severity="medium",
                description=f"Function '{node.name}' exceeds 60 line limit ({len(node.body)} lines)",
                suggestion="Break down large function into smaller, focused helper functions"
            ))

        # Rule 5: Check parameter validation
        if not self.rule_checker.has_parameter_validation(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=line_number,
                rule_number=5,
                rule_description=self.rule_descriptions[5],
                severity="medium",
                description=f"Function '{node.name}' lacks parameter validation",
                suggestion="Add type hints and parameter validation to improve reliability"
            ))

        # Rule 6: Check for complex decorators
        if self.rule_checker.has_complex_decorators(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=line_number,
                rule_number=6,
                rule_description=self.rule_descriptions[6],
                severity="medium",
                description=f"Function '{node.name}' has complex decorators",
                suggestion="Simplify decorators or avoid complex decorator patterns"
            ))

        # Rule 7: Check for docstring
        if not self.rule_checker.has_docstring(node):
            issues.append(ComplianceIssue(
                file_path=str(file_path),
                line_number=line_number,
                rule_number=7,
                rule_description=self.rule_descriptions[7],
                severity="high",
                description=f"Function '{node.name}' missing docstring",
                suggestion="Add detailed docstring explaining function purpose, parameters, and return values"
            ))

        return issues

    def generate_compliance_report(self, source_directory: Path) -> ComplianceReport:
        """Generate complete compliance report"""
        print(f"🔍 Auditing codebase for high-reliability compliance...")
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
            f.name not in ['setup.py', '__init__.py']
        ]

        # Audit each file with bounded processing
        total_files = len(system_files)
        total_functions = 0
        total_lines = 0

        for file_path in system_files:
            try:
                # Count lines
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)

                # Audit file
                issues = self.audit_file(file_path)
                all_issues.extend(issues)

                # Count functions from AST
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ast.parse(content)
                function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                total_functions += function_count

            except Exception as e:
                all_issues.append(ComplianceIssue(
                    file_path=str(file_path),
                    line_number=1,
                    rule_number=0,
                    rule_description="File processing error",
                    severity="low",
                    description=f"Could not process file: {e}",
                    suggestion="Check file permissions and syntax"
                ))

        # Calculate compliance score
        total_possible_issues = total_functions * 8  # 8 rules per function
        compliance_score = max(0.0, 100.0 - (len(all_issues) / max(total_possible_issues, 1) * 100))

        # Generate rule compliance status
        rule_compliance = {}
        for rule_num in self.rule_descriptions.keys():
            rule_issues = [issue for issue in all_issues if issue.rule_number == rule_num]
            rule_compliance[rule_num] = len(rule_issues) == 0

        # Generate detailed analysis
        detailed_analysis = {
            'rules_violated': len([r for r in rule_compliance.values() if not r]),
            'files_by_severity': {
                'high': len([i for i in all_issues if i.severity == 'high']),
                'medium': len([i for i in all_issues if i.severity == 'medium']),
                'low': len([i for i in all_issues if i.severity == 'low'])
            }
        }

        return ComplianceReport(
            total_files_analyzed=total_files,
            total_functions_analyzed=total_functions,
            total_lines_analyzed=total_lines,
            issues_found=all_issues,
            compliance_score=compliance_score,
            rule_compliance=rule_compliance,
            detailed_analysis=detailed_analysis
        )

    def print_report(self, report: ComplianceReport):
        """Print formatted compliance report"""
        self.report_generator.print_report(report)

    def save_report(self, report: ComplianceReport, output_path: Path):
        """Save compliance report to file"""
        self.report_generator.save_report(report, output_path)


def main():
    """Main compliance audit function"""
    import argparse

    parser = argparse.ArgumentParser(description="High-Reliability Compliance Audit")
    parser.add_argument("--source", type=str, default=".",
                      help="Source directory to audit")
    parser.add_argument("--output", type=str,
                      help="Output file for compliance report")

    args = parser.parse_args()

    # Initialize auditor and generate report
    auditor = HighReliabilityAuditor()
    source_path = Path(args.source)

    report = auditor.generate_compliance_report(source_path)
    auditor.print_report(report)

    if args.output:
        output_path = Path(args.output)
        auditor.save_report(report, output_path)
        print(f"\n📄 Report saved to: {output_path}")


if __name__ == "__main__":
    main()

