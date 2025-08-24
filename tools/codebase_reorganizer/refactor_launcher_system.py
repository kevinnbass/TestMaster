#!/usr/bin/env python3
"""
Refactor Launcher System Checks Module
====================================

Handles system validation and prerequisite checks for the refactored launcher.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Final

# Constants
MAX_EXECUTION_TIME: Final[int] = 3600  # 1 hour maximum execution time
MIN_MEMORY_AVAILABLE: Final[int] = 100 * 1024 * 1024  # 100MB minimum
MAX_FILES_TO_PROCESS: Final[int] = 5000
MAX_MODULES_CHECK: Final[int] = 50  # Safety bound for module checking


def check_python_version() -> Dict:
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


def check_directory_access(root_dir: Path) -> Dict:
    """Check directory access permissions"""
    try:
        assert root_dir.exists(), "Root directory does not exist"
        assert root_dir.is_dir(), "Root path is not a directory"

        # Test write access
        test_file = root_dir / ".test_access.tmp"
        test_file.write_text("test")
        test_file.unlink()

        return {'passed': True, 'error': None}

    except Exception as e:
        return {'passed': False, 'error': str(e)}


def check_memory_available() -> Dict:
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


def check_disk_space(root_dir: Path) -> Dict:
    """Check available disk space"""
    try:
        stat = os.statvfs(str(root_dir))
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


def check_modules_available() -> Dict:
    """Check that all required modules are available"""
    required_modules = [
        'validation_module',
        'reorganizer_engine'
    ]

    # Pre-allocate missing_modules with known capacity (Rule 3 compliance)
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


def perform_initial_checks(root_dir: Path) -> Dict:
    """Perform initial system checks"""
    checks = {
        'python_version': check_python_version(),
        'directory_access': check_directory_access(root_dir),
        'memory_available': check_memory_available(),
        'disk_space': check_disk_space(root_dir),
        'modules_available': check_modules_available()
    }

    # Determine overall status
    overall_status = all(check['passed'] for check in checks.values())

    return {
        **checks,
        'overall_status': overall_status
    }


def print_system_status(system_status: Dict) -> None:
    """Print formatted system status"""
    print("🔍 INITIAL SYSTEM CHECKS")
    print("=" * 25)

    for check_name, check_result in system_status.items():
        if check_name == 'overall_status':
            continue

        if check_name == 'python_version':
            status = "✅" if check_result['passed'] else "❌"
            print(f"🐍 Python Version: {status} {check_result['current_version']} (req: {check_result['required_version']})")
        elif check_name == 'directory_access':
            status = "✅" if check_result['passed'] else "❌"
            print(f"📁 Directory Access: {status} {check_result.get('error', 'OK')}")
        elif check_name == 'memory_available':
            status = "✅" if check_result['passed'] else "❌"
            if 'available_mb' in check_result:
                print(f"🧠 Memory Available: {status} {check_result['available_mb']:.1f}MB")
            else:
                print(f"🧠 Memory Available: {status} {check_result.get('note', 'Unknown')}")
        elif check_name == 'disk_space':
            status = "✅" if check_result['passed'] else "❌"
            if 'available_gb' in check_result:
                print(f"💽 Disk Space: {status} {check_result['available_gb']:.1f}GB free")
            else:
                print(f"💽 Disk Space: {status} {check_result.get('note', 'Unknown')}")
        elif check_name == 'modules_available':
            status = "✅" if check_result['passed'] else "❌"
            print(f"📦 Modules Available: {status} {check_result.get('error', 'OK')}")

    print()


class SystemChecker:
    """Handles system validation checks for the refactored launcher"""

    def __init__(self, root_dir: Path):
        """Initialize system checker with root directory"""
        self.root_dir = root_dir

    def perform_checks(self) -> Dict:
        """Perform all system checks"""
        return perform_initial_checks(self.root_dir)

    def print_status(self, system_status: Dict) -> None:
        """Print system status"""
        print_system_status(system_status)

