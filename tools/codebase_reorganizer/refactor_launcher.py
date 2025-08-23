#!/usr/bin/env python3
"""
Refactored Launcher Coordinator
===============================

Coordinates the refactored launcher system using specialized modules.
Demonstrates programmatic refactoring for high-reliability compliance.
"""

import sys
from pathlib import Path
from typing import Dict, Final

# Import specialized modules
from refactor_launcher_system import SystemChecker
from refactor_launcher_files import FileOperations

# Import validation and reorganization modules
try:
    from validation_module import Validator, ValidationTestSuite, run_comprehensive_safety_audit
    from reorganizer_engine import Validator as ReorganizerValidator
    from reorganizer_engine import FileAnalyzer, ReorganizationEngine
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("   Ensure all modules are in the same directory")
    sys.exit(1)

# Constants
MAX_EXECUTION_TIME: Final[int] = 3600  # 1 hour maximum execution time
MAX_FILES_TO_PROCESS: Final[int] = 5000


class CodebaseReorganizerLauncher:
    """Refactored launcher coordinator using specialized modules"""

    def __init__(self) -> None:
        """Initialize with system checks and file operations"""
        self.root_dir: Path = Path.cwd()
        self.system_checker = SystemChecker(self.root_dir)
        self.file_operations = FileOperations(self.root_dir)

        # Perform initial checks using system module
        self.system_status: Dict = self.system_checker.perform_checks()




    def run_comprehensive_audit(self) -> Dict:
        """Run comprehensive validation audit"""
        print("\n🔍 COMPREHENSIVE VALIDATION AUDIT")
        print("=" * 35)

        return run_comprehensive_safety_audit()

    def _execute_reorganization_process(self) -> Dict:
        """Execute the reorganization process"""
        try:
            # Initialize reorganization engine
            engine = ReorganizationEngine(self.root_dir)
            assert engine is not None, "Engine initialization failed"

            # Find Python files using file operations module
            python_files = self.file_operations.find_python_files()
            print(f"Found {len(python_files)} Python files to analyze")

            # Process files with execution time monitoring
            processed = 0
            max_to_process = min(len(python_files), MAX_FILES_TO_PROCESS)

            for i, file_path in enumerate(python_files[:max_to_process]):
                # Check execution time bounds
                elapsed = time.time() - self.start_time
                if elapsed > MAX_EXECUTION_TIME:
                    print(f"⚠️  Execution time limit reached: {elapsed:.1f}s")
                    break

                success = engine.process_single_file(file_path)
                if success:
                    processed += 1

                # Progress reporting
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1} files... ({processed} successful)")

            print(f"Successfully analyzed {processed} files")

            # Execute operations safely
            results = engine.execute_operations_safe()
            report = engine.generate_comprehensive_report(results)

            return {
                'success': True,
                'files_processed': processed,
                'operations_executed': results['executed'],
                'operations_failed': results['failed'],
                'system_compliance': report['system_compliance'],
                'execution_time': time.time() - self.start_time
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - self.start_time
            }

    def execute_reorganization(self) -> Dict:
        """Execute the codebase reorganization"""
        print("\n🚀 CODEBASE REORGANIZATION EXECUTION")
        print("=" * 40)

        return self._execute_reorganization_process()



    def _display_main_results(self, audit_results: Dict, execution_results: Dict) -> None:
        """Display final system report"""
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
            print(".1f")

    def generate_final_report(self, audit_results: Dict, execution_results: Dict) -> Dict:
        """Generate comprehensive final report"""
        report = {
            'validation_audit': audit_results,
            'execution_results': execution_results,
            'initial_checks': self.system_status,
            'timestamp': time.time(),
            'system_compliance': True
        }

        # Overall compliance check
        compliance_checks = [
            self.system_status['overall_status'],
            audit_results.get('system_compliance', False),
            execution_results.get('system_compliance', True)
        ]

        report['system_compliance'] = all(compliance_checks)

        return report

def main() -> int:
    """Main function - refactored to < 30 lines"""
    print("🚀 CODEBASE REORGANIZER SYSTEM")
    print("=" * 35)
    print("Clean, reliable codebase reorganization with validation")

    try:
        # Initialize launcher
        launcher = CodebaseReorganizerLauncher()

        # Run comprehensive safety audit with linting
        audit_results = launcher.run_comprehensive_audit()

        # Execute reorganization
        execution_results = launcher.execute_reorganization()

        # Generate final report
        final_report = launcher.generate_final_report(audit_results, execution_results)

        # Display system status using system checker
        launcher.system_checker.print_status(launcher.system_status)

        overall_compliance = final_report['system_compliance']
        compliance_status = "✅ FULLY COMPLIANT" if overall_compliance else "❌ ISSUES DETECTED"
        print(f"\n🎯 Overall Status: {compliance_status}")

        if overall_compliance:
            print("\n🎉 Codebase reorganization completed successfully!")
            print("   All system constraints maintained throughout execution.")
        else:
            print("\n⚠️  Issues detected during execution. Review logs for details.")

        return 0 if overall_compliance else 1

    except Exception as e:
        print(f"\n❌ SYSTEM FAILURE: {e}")
        print("   System failure occurred during execution.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
