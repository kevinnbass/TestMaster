#!/usr/bin/env python3
"""
Compliance Audit Reporting
=========================

Handles report generation and display for compliance auditing.
"""

import json
from pathlib import Path
from typing import Dict, Any, Final

from compliance_audit_data import ComplianceReport, ComplianceIssue

# Constants
MAX_DISPLAY_ISSUES: Final[int] = 5  # Maximum issues to display per rule


def _print_header(report: ComplianceReport):
    """Print report header"""
    print("\n" + "="*80)
    print("🏛️  HIGH-RELIABILITY COMPLIANCE AUDIT REPORT")
    print("="*80)

    print(".1f")
    print(f"📂 Files Analyzed: {report.total_files_analyzed}")
    print(f"🔧 Functions Analyzed: {report.total_functions_analyzed}")
    print(f"📏 Lines Analyzed: {report.total_lines_analyzed}")
    print(f"⚠️  Issues Found: {len(report.issues_found)}")
    print(f"📊 Rules Violated: {len([r for r in report.rule_compliance.values() if not r])}")


def _print_rule_summary(report: ComplianceReport):
    """Print rule compliance summary"""
    print("\n📋 RULE COMPLIANCE SUMMARY:")
    for rule_num in sorted(report.rule_compliance.keys()):
        status = "✅" if report.rule_compliance[rule_num] else "❌"
        print("2d")


def _print_severity_analysis(report: ComplianceReport):
    """Print issues by severity"""
    if hasattr(report, 'detailed_analysis') and 'files_by_severity' in report.detailed_analysis:
        severity_data = report.detailed_analysis['files_by_severity']
        print("\n🚨 ISSUES BY SEVERITY:")
        print(f"  🔴 High: {severity_data.get('high', 0)}")
        print(f"  🟡 Medium: {severity_data.get('medium', 0)}")
        print(f"  🔵 Low: {severity_data.get('low', 0)}")


def _print_key_metrics(report: ComplianceReport):
    """Print key metrics"""
    print("\n📊 KEY METRICS:")
    print(".2f")
    print(".1f")


def _print_top_issues(report: ComplianceReport, rule_descriptions: Dict[int, str]):
    """Print top issues"""
    if report.issues_found:
        print("\n🔍 TOP ISSUES:")
        # Group issues by rule using explicit loop (Rule 1 compliance)
        rule_groups = {}
        for issue in report.issues_found:
            if issue.rule_number not in rule_groups:
                rule_groups[issue.rule_number] = []
            rule_groups[issue.rule_number].append(issue)

        for rule_num in sorted(rule_groups.keys())[:MAX_DISPLAY_ISSUES]:  # Show top 5 rules with issues
            issues = rule_groups[rule_num]
            print(f"\n  Rule {rule_num}: {rule_descriptions.get(rule_num, 'Unknown Rule')}")
            print(f"    Issues: {len(issues)}")

            # Show first issue as example
            if issues:
                example = issues[0]
                print(f"    Example: {example.file_path}:{example.line_number} - {example.description}")


def _print_compliance_assessment(report: ComplianceReport):
    """Print compliance assessment"""
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


def print_compliance_report(report: ComplianceReport, rule_descriptions: Dict[int, str]):
    """Print formatted compliance report"""
    _print_header(report)
    _print_rule_summary(report)
    _print_severity_analysis(report)
    _print_key_metrics(report)
    _print_top_issues(report, rule_descriptions)
    _print_compliance_assessment(report)


def save_compliance_report(report: ComplianceReport, output_path: Path):
    """Save compliance report to file"""
    report_data = {
        'compliance_score': report.compliance_score,
        'total_files_analyzed': report.total_files_analyzed,
        'total_functions_analyzed': report.total_functions_analyzed,
        'total_lines_analyzed': report.total_lines_analyzed,
        'total_issues': len(report.issues_found),
        'rule_compliance': report.rule_compliance,
        'detailed_analysis': getattr(report, 'detailed_analysis', {}),
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
        json.dump(report_data, f, indent=2, ensure_ascii=False)


def generate_summary_report(report: ComplianceReport) -> Dict[str, Any]:
    """Generate a summary of the compliance report"""
    total_issues = len(report.issues_found)
    rules_violated = len([r for r in report.rule_compliance.values() if not r])

    # Count issues by severity if available
    severity_counts = {'high': 0, 'medium': 0, 'low': 0}
    if hasattr(report, 'detailed_analysis') and 'files_by_severity' in report.detailed_analysis:
        severity_counts = report.detailed_analysis['files_by_severity']

    return {
        'compliance_score': report.compliance_score,
        'total_files': report.total_files_analyzed,
        'total_functions': report.total_functions_analyzed,
        'total_lines': report.total_lines_analyzed,
        'total_issues': total_issues,
        'rules_violated': rules_violated,
        'severity_breakdown': severity_counts,
        'rules_compliant': len([r for r in report.rule_compliance.values() if r]),
        'rules_total': len(report.rule_compliance)
    }


class ComplianceReportGenerator:
    """Handles compliance report generation and display"""

    def __init__(self, rule_descriptions: Dict[int, str]):
        """Initialize report generator with rule descriptions"""
        self.rule_descriptions = rule_descriptions

    def print_report(self, report: ComplianceReport):
        """Print formatted compliance report"""
        print_compliance_report(report, self.rule_descriptions)

    def save_report(self, report: ComplianceReport, output_path: Path):
        """Save compliance report to file"""
        save_compliance_report(report, output_path)

    def generate_summary(self, report: ComplianceReport) -> Dict[str, Any]:
        """Generate a summary of the compliance report"""
        return generate_summary_report(report)
