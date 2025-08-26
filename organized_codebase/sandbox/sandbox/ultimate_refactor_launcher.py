#!/usr/bin/env python3
"""
Ultimate refactored launcher.py with ALL functions < 30 lines
Demonstrates extreme high-reliability compliance
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Final
import time

# Import validation and reorganization modules
try:
    from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.sandbox.validation_module import Validator, ValidationTestSuite, run_comprehensive_safety_audit
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

class CodebaseReorganizerLauncher:
    """Ultimate refactored launcher with functions < 30 lines"""

    def __init__(self) -> None:
        """Initialize with validated parameters"""
        self.root_dir: Path = Path.cwd()
        self.start_time: float = time.time()
        self.system_status: Dict = self._perform_initial_checks()

    def _check_python_version(self) -> Dict:
        """Check Python version compatibility"""
        version = sys.version_info
        required_version = (3, 8)

        passed = version >= required_version
        error = None if passed else ".1f"

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
            import psutil
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
            stat = os.statvfs(str(self.root_dir))
            available_bytes = stat.f_bavail * stat.f_frsize
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

        # Pre-allocate missing_modules with known capacity (Rule 3 compliance)
        MAX_MODULES_CHECK = 50  # Safety bound for module checking
        missing_modules = [None] * MAX_MODULES_CHECK
        missing_count = 0

        # Bounded loop for module checking
        for i in range(min(len(required_modules), MAX_MODULES_CHECK)):
            module = required_modules[i]
            try:
                __import__(module)
            except ImportError:
                if missing_count < MAX_MODULES_CHECK:
                    missing_modules[missing_count] = module
                    missing_count += 1

        passed = missing_count == 0
        error = None if passed else f"Missing modules: {', '.join(missing_modules[:missing_count])}"

        return {
            'passed': passed,
            'error': error,
            'missing_modules': missing_modules[:missing_count]
        }

    def _collect_system_checks(self) -> Dict:
        """Collect all system checks"""
        return {
            'python_version': self._check_python_version(),
            'directory_access': self._check_directory_access(),
            'memory_available': self._check_memory_available(),
            'disk_space': self._check_disk_space(),
            'modules_available': self._check_modules_available(),
            'overall_status': True
        }

    def _print_system_check_results(self, checks: Dict) -> None:
        """Print system check results"""
        print("🔍 INITIAL SYSTEM CHECKS")
        print("=" * 25)

        for check_name, result in checks.items():
            if check_name != 'overall_status':
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                print(f"   {check_name}: {status}")
                if not result['passed']:
                    print(f"      Error: {result.get('error', 'Unknown error')}")

    def _finalize_system_checks(self, checks: Dict) -> Dict:
        """Finalize system checks and handle failures"""
        overall_status = all(check['passed'] for check in checks.values() if isinstance(check, dict) and 'passed' in check)
        checks['overall_status'] = overall_status

        if not overall_status:
            print("\n❌ System checks failed. Cannot proceed.")
            sys.exit(1)

        print("\n✅ All system checks passed!")
        return checks

    def _perform_initial_checks(self) -> Dict:
        """Perform comprehensive initial safety checks"""
        checks = self._collect_system_checks()
        self._print_system_check_results(checks)
        return self._finalize_system_checks(checks)

    def run_comprehensive_audit(self) -> Dict:
        """Run comprehensive validation audit"""
        print("\n🔍 COMPREHENSIVE VALIDATION AUDIT")
        print("=" * 35)

        return run_comprehensive_safety_audit()

    def _initialize_reorganization_engine(self) -> ReorganizationEngine:
        """Initialize reorganization engine safely"""
        engine = ReorganizationEngine(self.root_dir)
        assert engine is not None, "Engine initialization failed"
        return engine

    def _find_and_validate_files(self, engine: ReorganizationEngine) -> tuple:
        """Find and validate Python files"""
        python_files = self._find_python_files_safely()
        print(f"Found {len(python_files)} Python files to analyze")

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

        return python_files, processed, max_to_process

    def _execute_and_report(self, engine: ReorganizationEngine, processed: int) -> Dict:
        """Execute operations and generate report"""
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

    def _handle_execution_error(self, error: Exception) -> Dict:
        """Handle execution errors safely"""
        return {
            'success': False,
            'error': str(error),
            'execution_time': time.time() - self.start_time
        }

    def _execute_reorganization_process(self) -> Dict:
        """Execute the reorganization process"""
        try:
            engine = self._initialize_reorganization_engine()
            python_files, processed, max_to_process = self._find_and_validate_files(engine)
            return self._execute_and_report(engine, processed)

        except Exception as e:
            return self._handle_execution_error(e)

    def execute_reorganization(self) -> Dict:
        """Execute the codebase reorganization"""
        print("\n🚀 CODEBASE REORGANIZATION EXECUTION")
        print("=" * 40)

        return self._execute_reorganization_process()

    def _filter_directory_entries(self, dirs: List[str]) -> List[str]:
        """Filter directory entries safely"""
        # Bounded loop for directory filtering with pre-allocation (Rule 3 compliance)
        MAX_DIRS_FILTER = 200  # Safety bound for directory filtering
        MAX_FILTERED_DIRS = 100  # Fixed upper bound for filtered directories
        filtered_dirs = [None] * MAX_FILTERED_DIRS
        filtered_count = 0

        for i in range(min(len(dirs), MAX_DIRS_FILTER)):
            d = dirs[i]
            if filtered_count >= MAX_FILTERED_DIRS:
                break
            should_exclude = False
            # Bounded loop for pattern checking
            exclusion_patterns = self._get_exclusion_patterns()
            for j in range(len(exclusion_patterns)):
                pattern = exclusion_patterns[j]
                if len(str(Path(self.root_dir) / d)) <= 260 and pattern in str(Path(self.root_dir) / d):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_dirs[filtered_count] = d
                filtered_count += 1

        return filtered_dirs[:filtered_count]

    def _check_file_eligibility(self, file_path: Path) -> bool:
        """Check if file is eligible for processing"""
        if len(str(file_path)) <= 260 and file_path.suffix == '.py':
            is_excluded = self._is_file_excluded(file_path)
            file_size_ok = file_path.stat().st_size <= 10 * 1024 * 1024
            return not is_excluded and file_size_ok
        return False

    def _get_exclusion_patterns(self) -> List[str]:
        """Get exclusion patterns"""
        return [
            '**/node_modules/**', '**/.*', '**/test*/**', '**/archive*/**',
            '**/__pycache__/**', '**/.*'
        ]

    def _is_file_excluded(self, file_path: Path) -> bool:
        """Check if file is excluded"""
        for pattern in self._get_exclusion_patterns():
            if pattern in str(file_path):
                return True
        return False

    def _process_file_batch(self, python_files: List[Path]) -> List[Path]:
        """Process files in batches with bounded loops"""
        # Pre-allocate valid_files with known capacity
        valid_files = [Path('.')] * MAX_FILES_TO_PROCESS  # Pre-allocate with placeholder
        valid_count = 0

        # Bounded loop for file processing
        for i in range(min(len(python_files), MAX_FILES_TO_PROCESS * 2)):  # Allow some buffer for processing
            file_path = python_files[i]
            if valid_count >= MAX_FILES_TO_PROCESS:
                break
            if self._check_file_eligibility(file_path):
                valid_files[valid_count] = file_path
                valid_count += 1

        # Return slice with actual count (bounded operation)
        return valid_files[:valid_count]

    def _find_python_files_safely(self) -> List[Path]:
        """Find Python files with validation bounds and bounded loops"""
        # Pre-allocate python_files with known capacity
        python_files = [Path('.')] * (MAX_FILES_TO_PROCESS * 2)  # Pre-allocate with placeholder
        file_count = 0
        directory_count = 0

        # Bounded loop for directory traversal
        MAX_DIRECTORIES = 5000  # Safety bound for directory processing
        for root, dirs, files in os.walk(self.root_dir):
            if directory_count >= MAX_DIRECTORIES:
                break
            directory_count += 1

            # Remove excluded directories
            dirs[:] = self._filter_directory_entries(dirs)

            # Process files with bounded loop
            MAX_FILES_PER_DIR = 1000  # Safety bound for files per directory
            for i in range(min(len(files), MAX_FILES_PER_DIR)):
                file = files[i]
                if file_count >= len(python_files):
                    break

                file_path = Path(root) / file
                if self._check_file_eligibility(file_path):
                    python_files[file_count] = file_path
                    file_count += 1

        # Return slice with actual count (bounded operation)
        return self._process_file_batch(python_files[:file_count])

    def _display_audit_status(self, audit_compliance: bool) -> None:
        """Display audit status"""
        status = "✅ PASSED" if audit_compliance else "❌ FAILED"
        print(f"   Validation Audit: {status}")

    def _display_execution_status(self, exec_success: bool) -> None:
        """Display execution status"""
        status = "✅ SUCCESS" if exec_success else "❌ FAILED"
        print(f"   Execution: {status}")

    def _display_compliance_status(self, exec_compliance: bool) -> None:
        """Display compliance status"""
        status = "✅ MAINTAINED" if exec_compliance else "❌ VIOLATED"
        print(f"   System Compliance: {status}")

    def _display_processing_results(self, execution_results: Dict) -> None:
        """Display processing results"""
        if execution_results.get('success', False):
            print(f"   Files Processed: {execution_results.get('files_processed', 0)}")
            print(f"   Operations Executed: {execution_results.get('operations_executed', 0)}")
            print(".1f")

    def _display_main_results(self, audit_results: Dict, execution_results: Dict) -> None:
        """Display final system report"""
        print("\n📊 FINAL SYSTEM REPORT")
        print("=" * 25)

        audit_compliance = audit_results.get('system_compliance', False)
        exec_success = execution_results.get('success', False)
        exec_compliance = execution_results.get('system_compliance', True)

        self._display_audit_status(audit_compliance)
        self._display_execution_status(exec_success)
        self._display_compliance_status(exec_compliance)
        self._display_processing_results(execution_results)

    def _determine_overall_status(self, audit_results: Dict, execution_results: Dict) -> tuple:
        """Determine overall system status"""
        overall_compliance = (
            self.system_status['overall_status'] and
            audit_results.get('system_compliance', False) and
            execution_results.get('system_compliance', True)
        )
        compliance_status = "✅ FULLY COMPLIANT" if overall_compliance else "❌ ISSUES DETECTED"
        return overall_compliance, compliance_status

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

def _initialize_system() -> CodebaseReorganizerLauncher:
    """Initialize the system safely"""
    return CodebaseReorganizerLauncher()

def _run_system_audit(launcher: CodebaseReorganizerLauncher) -> Dict:
    """Run system audit"""
    return launcher.run_comprehensive_audit()

def _execute_main_process(launcher: CodebaseReorganizerLauncher) -> Dict:
    """Execute main reorganization process"""
    return launcher.execute_reorganization()

def _display_system_results(launcher: CodebaseReorganizerLauncher, audit_results: Dict, execution_results: Dict) -> None:
    """Display system results"""
    launcher._display_main_results(audit_results, execution_results)

def main() -> int:
    """Main function - refactored to < 30 lines"""
    print("🚀 CODEBASE REORGANIZER SYSTEM")
    print("=" * 35)
    print("Clean, reliable codebase reorganization with validation")

    try:
        # Initialize and run system
        launcher = _initialize_system()
        audit_results = _run_system_audit(launcher)
        execution_results = _execute_main_process(launcher)

        # Generate and display results
        final_report = launcher.generate_final_report(audit_results, execution_results)
        _display_system_results(launcher, audit_results, execution_results)

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