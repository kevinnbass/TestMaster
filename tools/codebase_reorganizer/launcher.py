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
        self.system_status = self._perform_initial_checks()

    def _perform_initial_checks(self) -> Dict:
        """Perform initial system checks"""
        print("🔍 INITIAL SYSTEM CHECKS")
        print("=" * 25)

        checks: Dict[str, Any] = {
            'python_version': self._check_python_version(),
            'directory_access': self._check_directory_access(),
            'memory_available': self._check_memory_available(),
            'disk_space': self._check_disk_space(),
            'modules_available': self._check_modules_available(),
            'overall_status': True
        }

        # Print results
        for check_name, result in checks.items():
            if check_name != 'overall_status':
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                print(f"   {check_name}: {status}")
                if not result['passed']:
                    print(f"      Error: {result.get('error', 'Unknown error')}")

        overall_status = all(check['passed'] for check in checks.values() if isinstance(check, dict) and 'passed' in check)
        checks['overall_status'] = overall_status

        if not overall_status:
            print("\n❌ System checks failed. Cannot proceed.")
            sys.exit(1)

        print("\n✅ All system checks passed!")
        return checks

    def _check_python_version(self) -> Dict:
        """Check Python version compatibility"""
        version = sys.version_info
        required_version = (3, 8)

        passed = version >= required_version
        error = None if passed else f"Python {required_version[0]}.{required_version[1]}+ required"

        return {
            'passed': passed,
            'error': error,
            'current_version': f"{version.major}.{version.minor}.{version.micro}",
            'required_version': f"{required_version[0]}.{required_version[1]}"
        }

    def _check_directory_access(self) -> Dict:
        """Check directory access permissions"""
        try:
            assert self.root_dir.exists(), "Root directory does not exist"
            assert self.root_dir.is_dir(), "Root path is not a directory"

            # Test write access
            test_file = self.root_dir / ".test_access.tmp"
            test_file.write_text("test")
            test_file.unlink()

            return {'passed': True, 'error': None}

        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _check_memory_available(self) -> Dict:
        """Check available memory"""
        try:
            import psutil  # type: ignore[import-untyped]
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)

            passed = available_mb >= (MIN_MEMORY_AVAILABLE / (1024 * 1024))
            error = None if passed else ".1f"

            return {
                'passed': passed,
                'error': error,
                'available_mb': available_mb
            }

        except ImportError:
            return {'passed': True, 'error': None, 'note': 'Memory check not available'}

    def _check_disk_space(self) -> Dict:
        """Check available disk space"""
        try:
            # Cross-platform disk space check
            if hasattr(os, 'statvfs'):  # Unix-like systems
                stat = os.statvfs(str(self.root_dir))
                available_bytes = stat.f_bavail * stat.f_frsize
            else:  # Windows fallback
                import shutil
                usage = shutil.disk_usage(str(self.root_dir))
                available_bytes = usage.free

            available_gb = available_bytes / (1024**3)

            passed = available_gb >= 1.0  # Require at least 1GB
            error = None if passed else ".2f"

            return {
                'passed': passed,
                'error': error,
                'available_gb': available_gb
            }

        except AttributeError:
            return {'passed': True, 'error': None, 'note': 'Disk space check not available'}

    def _check_modules_available(self) -> Dict:
        """Check that all required modules are available"""
        required_modules = [
            'validation_module',
            'reorganizer_engine'
        ]

        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        passed = len(missing_modules) == 0
        error = None if passed else f"Missing modules: {', '.join(missing_modules)}"

        return {
            'passed': passed,
            'error': error,
            'missing_modules': missing_modules
        }

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
            python_files = self._find_python_files_safely()
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

    def _find_python_files_safely(self) -> List[Path]:
        """Find Python files with validation bounds"""
        python_files: List[Path] = []

        exclusion_patterns = {
            '**/node_modules/**', '**/.*', '**/test*/**', '**/archive*/**',
            '**/__pycache__/**', '**/.*'
        }

        for root, dirs, files in os.walk(self.root_dir):
            # Remove excluded directories with explicit bounds checking
            filtered_dirs: List[str] = []
            for d in dirs:
                if len(filtered_dirs) >= 100:  # Fixed upper bound
                    break
                should_exclude = False
                for pattern in exclusion_patterns:
                    if len(str(Path(root) / d)) <= MAX_PATH_LENGTH and pattern in str(Path(root) / d):
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered_dirs.append(d)
            dirs[:] = filtered_dirs

            for file in files:
                if len(python_files) >= MAX_FILES_TO_PROCESS:
                    break

                file_path = Path(root) / file
                # Check file with explicit bounds checking
                is_excluded = False
                for pattern in exclusion_patterns:
                    if len(str(file_path)) <= MAX_PATH_LENGTH and pattern in str(file_path):
                        is_excluded = True
                        break

                if (file.endswith('.py') and
                    not is_excluded and
                    file_path.stat().st_size <= 10 * 1024 * 1024):  # 10MB max

                    python_files.append(file_path)

        return python_files

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
    """Main function"""
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

        # Display results
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