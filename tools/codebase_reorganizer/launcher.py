#!/usr/bin/env python3
"""
Codebase Reorganizer Launcher
=============================

Comprehensive launcher for the codebase reorganization system.
This script provides a clean, reliable way to reorganize your codebase
with proper validation and error handling.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Final, Any
import time

# Import specialized modules
from launcher_system_checks import SystemChecker, print_system_status
from launcher_file_operations import FileOperations
from launcher_reporting import ReportGenerator

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
MIN_MEMORY_AVAILABLE: Final[int] = 100 * 1024 * 1024  # 100MB minimum
MAX_FILES_TO_PROCESS: Final[int] = 5000
MAX_PATH_LENGTH: Final[int] = 260

class CodebaseReorganizerLauncher:
    """Reliable launcher system for codebase reorganization"""

    def __init__(self) -> None:
        """Initialize with proper validation checks"""
        self.root_dir = Path.cwd()
        self.start_time = time.time()
        self.system_checker = SystemChecker(self.root_dir)
        self.system_status = self.system_checker.perform_initial_checks()
        self.file_operations = FileOperations(self.root_dir)
        self.report_generator = ReportGenerator(self.system_status)





    def run_comprehensive_audit(self) -> Dict:
        """Run comprehensive validation audit"""
        print("\n🔍 COMPREHENSIVE VALIDATION AUDIT")
        print("=" * 35)

        return run_comprehensive_safety_audit()

    def execute_reorganization(self) -> Dict:
        """Execute the codebase reorganization"""
        print("\n🚀 CODEBASE REORGANIZATION EXECUTION")
        print("=" * 40)

        try:
            # Initialize reorganization engine
            engine = ReorganizationEngine(self.root_dir)
            assert engine is not None, "Engine initialization failed"

            # Find Python files with bounds checking
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





def main() -> int:
    """Main function"""
    print("🚀 CODEBASE REORGANIZER SYSTEM")
    print("=" * 35)
    print("Clean, reliable codebase reorganization with validation")

    try:
        # Initialize launcher
        launcher = CodebaseReorganizerLauncher()

        # Print system status
        print_system_status(launcher.system_status)

        # Run comprehensive safety audit with linting
        audit_results = launcher.run_comprehensive_audit()

        # Execute reorganization
        execution_results = launcher.execute_reorganization()

        # Generate final report
        final_report = launcher.report_generator.generate_report(audit_results, execution_results)

        # Display results
        launcher.report_generator.print_report(final_report, audit_results, execution_results)

        return 0 if final_report.get('system_compliance', False) else 1

    except Exception as e:
        print(f"\n❌ SYSTEM FAILURE: {e}")
        print("   System failure occurred during execution.")
        return 1

if __name__ == "__main__":
    sys.exit(main())