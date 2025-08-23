#!/usr/bin/env python3
"""
System Checks Module for Codebase Reorganizer Launcher
====================================================

This module handles all system validation and prerequisite checks
for the codebase reorganization process.
"""

import sys
import os
import psutil
import shutil
from pathlib import Path
from typing import Dict, Any, Final
import time

# Constants
MAX_EXECUTION_TIME: Final[int] = 3600  # 1 hour maximum execution time
MIN_MEMORY_AVAILABLE: Final[int] = 100 * 1024 * 1024  # 100MB minimum
MAX_FILES_TO_PROCESS: Final[int] = 5000
MAX_PATH_LENGTH: Final[int] = 260


class SystemChecker:
    """Handles all system validation checks"""

    def __init__(self, root_dir: Path):
        """Initialize system checker"""
        self.root_dir = root_dir

    def perform_initial_checks(self) -> Dict[str, Any]:
        """Perform initial system checks"""
        print("🔍 INITIAL SYSTEM CHECKS")
        print("=" * 25)

        checks: Dict[str, Any] = {
            'python_version': self._check_python_version(),
            'directory_access': self._check_directory_access(),
            'memory_available': self._check_memory_available(),
            'disk_space': self._check_disk_space(),
            'modules_available': self._check_modules_available()
        }

        return checks

    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility"""
        version_info = sys.version_info
        current_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

        min_version = (3, 8, 0)
        is_compatible = version_info >= min_version

        return {
            'current': current_version,
            'required': '.'.join(map(str, min_version)),
            'compatible': is_compatible,
            'status': '✅' if is_compatible else '❌'
        }

    def _check_directory_access(self) -> Dict[str, Any]:
        """Check directory read/write access"""
        try:
            # Test read access
            files = list(self.root_dir.glob("*.py"))
            can_read = True
        except Exception as e:
            can_read = False
            read_error = str(e)

        try:
            # Test write access by creating a temporary file
            test_file = self.root_dir / ".temp_access_test"
            test_file.write_text("test")
            test_file.unlink()
            can_write = True
        except Exception as e:
            can_write = False
            write_error = str(e)

        has_access = can_read and can_write

        return {
            'path': str(self.root_dir),
            'can_read': can_read,
            'can_write': can_write,
            'has_access': has_access,
            'status': '✅' if has_access else '❌'
        }

    def _check_memory_available(self) -> Dict[str, Any]:
        """Check available system memory"""
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            total_mb = memory.total / (1024 * 1024)

            has_enough = memory.available >= MIN_MEMORY_AVAILABLE

            return {
                'available_mb': round(available_mb, 1),
                'total_mb': round(total_mb, 1),
                'required_mb': MIN_MEMORY_AVAILABLE / (1024 * 1024),
                'sufficient': has_enough,
                'status': '✅' if has_enough else '❌'
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'available_mb': 'unknown',
                'total_mb': 'unknown',
                'required_mb': MIN_MEMORY_AVAILABLE / (1024 * 1024),
                'sufficient': True,  # Assume sufficient if can't check
                'status': '⚠️'
            }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            usage = shutil.disk_usage(self.root_dir)
            free_gb = usage.free / (1024 * 1024 * 1024)
            total_gb = usage.total / (1024 * 1024 * 1024)

            min_required_gb = 1.0  # 1GB minimum
            has_enough = free_gb >= min_required_gb

            return {
                'free_gb': round(free_gb, 1),
                'total_gb': round(total_gb, 1),
                'required_gb': min_required_gb,
                'sufficient': has_enough,
                'status': '✅' if has_enough else '❌'
            }
        except Exception as e:
            return {
                'free_gb': 'unknown',
                'total_gb': 'unknown',
                'required_gb': 1.0,
                'sufficient': True,  # Assume sufficient if can't check
                'status': '⚠️'
            }

    def _check_modules_available(self) -> Dict[str, Any]:
        """Check if required modules are available"""
        required_modules = [
            'validation_module',
            'reorganizer_engine',
            'pathlib',
            'typing',
            'os',
            'sys',
            'time'
        ]

        # Pre-allocate with known maximum (Rule 3 compliance)
        MAX_MODULES_CHECK = 20  # Safety bound for module checking
        available_modules = [None] * MAX_MODULES_CHECK
        missing_modules = [None] * MAX_MODULES_CHECK
        available_count = 0
        missing_count = 0

        # Bounded loop for module checking (Rule 2 compliance)
        for i in range(min(len(required_modules), MAX_MODULES_CHECK)):
            module = required_modules[i]
            try:
                if module == 'validation_module':
                    from validation_module import Validator
                elif module == 'reorganizer_engine':
                    from reorganizer_engine import ReorganizationEngine
                elif module in ['pathlib', 'typing', 'os', 'sys', 'time']:
                    __import__(module)
                if available_count < MAX_MODULES_CHECK:
                    available_modules[available_count] = module
                    available_count += 1
            except ImportError:
                if missing_count < MAX_MODULES_CHECK:
                    missing_modules[missing_count] = module
                    missing_count += 1

        all_available = missing_count == 0

        # Return actual lists (Rule 3 compliance)
        actual_available = available_modules[:available_count] if available_count > 0 else []
        actual_missing = missing_modules[:missing_count] if missing_count > 0 else []

        return {
            'available': actual_available,
            'missing': actual_missing,
            'all_available': all_available,
            'status': '✅' if all_available else '❌'
        }


def print_system_status(checks: Dict[str, Any]) -> None:
    """Print formatted system status"""
    print("\n📊 SYSTEM STATUS")
    print("=" * 25)

    for check_name, check_result in checks.items():
        if check_name == 'python_version':
            print(f"🐍 Python Version: {check_result['status']} {check_result['current']} (req: {check_result['required']})")
        elif check_name == 'directory_access':
            print(f"📁 Directory Access: {check_result['status']} {check_result['path']}")
        elif check_name == 'memory_available':
            if check_result['available_mb'] != 'unknown':
                print(f"🧠 Memory Available: {check_result['status']} {check_result['available_mb']}MB (req: {check_result['required_mb']}MB)")
            else:
                print(f"🧠 Memory Available: {check_result['status']} Unknown (req: {check_result['required_mb']}MB)")
        elif check_name == 'disk_space':
            if check_result['free_gb'] != 'unknown':
                print(f"💽 Disk Space: {check_result['status']} {check_result['free_gb']}GB free (req: {check_result['required_gb']}GB)")
            else:
                print(f"💽 Disk Space: {check_result['status']} Unknown (req: {check_result['required_gb']}GB)")
        elif check_name == 'modules_available':
            print(f"📦 Modules Available: {check_result['status']} {len(check_result['available'])}/{len(check_result['available']) + len(check_result['missing'])} modules")
            if check_result['missing']:
                print(f"   Missing: {', '.join(check_result['missing'])}")

    print()


def validate_system_requirements(checks: Dict[str, Any]) -> bool:
    """Validate that all system requirements are met"""
    # Pre-allocate with known maximum (Rule 3 compliance)
    MAX_CRITICAL_CHECKS = 5  # Safety bound for critical checks
    critical_failures = [None] * MAX_CRITICAL_CHECKS
    failure_count = 0

    # Bounded loop for checking requirements (Rule 2 compliance)
    check_items = list(checks.items())
    for i in range(min(len(check_items), MAX_CRITICAL_CHECKS)):
        check_name, check_result = check_items[i]
        if check_name in ['python_version', 'directory_access', 'modules_available']:
            if not check_result.get('compatible', check_result.get('has_access', check_result.get('all_available', False))):
                if failure_count < MAX_CRITICAL_CHECKS:
                    critical_failures[failure_count] = check_name
                    failure_count += 1

    if failure_count > 0:
        # Create list of actual failures with pre-allocation (Rule 3 compliance)
        actual_failures = [None] * failure_count
        for i in range(failure_count):
            if critical_failures[i] is not None:
                actual_failures[i] = critical_failures[i]

        print(f"\n❌ CRITICAL SYSTEM ISSUES: {', '.join(actual_failures)}")
        print("   Please resolve these issues before running the reorganizer.")
        return False

    return True
