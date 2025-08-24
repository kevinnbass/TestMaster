#!/usr/bin/env python3
"""
Launcher Reporting Module
=========================

Handles report generation and display for the launcher system.
"""

import time
from typing import Dict, Any, Final

# Import system checks module
from launcher_system_checks import validate_system_requirements

# Constants
MAX_EXECUTION_TIME: Final[int] = 3600  # 1 hour maximum execution time


def generate_final_report(audit_results: Dict, execution_results: Dict,
                         system_status: Dict) -> Dict:
    """Generate comprehensive final report"""
    report = {
        'validation_audit': audit_results,
        'execution_results': execution_results,
        'initial_checks': system_status,
        'timestamp': time.time(),
        'system_compliance': True
    }

    # Overall compliance check
    compliance_checks = [
        validate_system_requirements(system_status),
        audit_results.get('system_compliance', False),
        execution_results.get('system_compliance', True)
    ]

    report['system_compliance'] = all(compliance_checks)

    return report


def print_final_report(final_report: Dict, audit_results: Dict,
                      execution_results: Dict) -> None:
    """Print the final system report"""
    print("\n📊 FINAL SYSTEM REPORT")
    print("=" * 25)

    audit_compliance = audit_results.get('system_compliance', False)
    exec_success = execution_results.get('success', False)
    exec_compliance = execution_results.get('system_compliance', True)

    print(f"   Validation Audit: {'✅ PASSED' if audit_compliance else '❌ FAILED'}")
    print(f"   Execution: {'✅ SUCCESS' if exec_success else '❌ FAILED'}")
    print(f"   System Compliance: {'✅ MAINTAINED' if exec_compliance else '❌ VIOLATED'}")

    if exec_success:
        print(f"   Files Processed: {execution_results.get('files_processed', 0)}")
        print(f"   Operations Executed: {execution_results.get('operations_executed', 0)}")
        execution_time = time.time() - final_report.get('timestamp', time.time())
        print(".1f")

    overall_compliance = final_report['system_compliance']
    compliance_status = "✅ FULLY COMPLIANT" if overall_compliance else "❌ ISSUES DETECTED"
    print(f"\n🎯 Overall Status: {compliance_status}")

    if overall_compliance:
        print("\n🎉 Codebase reorganization completed successfully!")
        print("   All system constraints maintained throughout execution.")
    else:
        print("\n⚠️  Issues detected during execution. Review logs for details.")


def create_execution_summary(audit_results: Dict, execution_results: Dict,
                           system_status: Dict) -> Dict:
    """Create a summary of the entire execution process"""
    execution_time = time.time() - system_status.get('timestamp', time.time())

    return {
        'execution_time_seconds': execution_time,
        'audit_passed': audit_results.get('system_compliance', False),
        'execution_successful': execution_results.get('success', False),
        'files_processed': execution_results.get('files_processed', 0),
        'operations_executed': execution_results.get('operations_executed', 0),
        'system_compliance': validate_system_requirements(system_status),
        'timestamp': time.time()
    }


class ReportGenerator:
    """Report generation handler for the launcher"""

    def __init__(self, system_status: Dict):
        """Initialize report generator with system status"""
        self.system_status = system_status

    def generate_report(self, audit_results: Dict, execution_results: Dict) -> Dict:
        """Generate comprehensive final report"""
        return generate_final_report(audit_results, execution_results, self.system_status)

    def print_report(self, final_report: Dict, audit_results: Dict,
                    execution_results: Dict) -> None:
        """Print the final system report"""
        print_final_report(final_report, audit_results, execution_results)

    def create_summary(self, audit_results: Dict, execution_results: Dict) -> Dict:
        """Create execution summary"""
        return create_execution_summary(audit_results, execution_results, self.system_status)

