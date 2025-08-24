#!/usr/bin/env python3
"""
Launcher File Operations Module
===============================

Handles safe file discovery and processing operations for the launcher.
"""

import os
from pathlib import Path
from typing import List, Set, Final

# Constants
MAX_FILES_TO_PROCESS: Final[int] = 5000
MAX_PATH_LENGTH: Final[int] = 260
MAX_DIRS_FILTER: Final[int] = 200
MAX_FILES_PROCESS: Final[int] = 1000


def get_exclusion_patterns() -> Set[str]:
    """Get exclusion patterns for file filtering"""
    return {
        '**/node_modules/**', '**/.*', '**/test*/**', '**/archive*/**',
        '**/__pycache__/**', '**/.*'
    }


def filter_directory(dirs: List[str], root: str, exclusion_patterns: Set[str]) -> List[str]:
    """Filter excluded directories with bounds checking"""
    filtered_dirs: List[str] = [None] * 100  # Fixed upper bound
    filtered_count = 0

    # Bounded loop for directory filtering
    for i in range(min(len(dirs), MAX_DIRS_FILTER)):
        d = dirs[i]
        if filtered_count >= 100:  # Fixed upper bound
            break
        should_exclude = False
        # Bounded loop for pattern checking
        pattern_list = list(exclusion_patterns)
        for j in range(len(pattern_list)):
            pattern = pattern_list[j]
            if len(str(Path(root) / d)) <= MAX_PATH_LENGTH and pattern in str(Path(root) / d):
                should_exclude = True
                break
        if not should_exclude and filtered_count < len(filtered_dirs):
            filtered_dirs[filtered_count] = d
            filtered_count += 1

    # Create final list with pre-allocation (Rule 3 compliance)
    final_filtered_dirs = [None] * filtered_count  # Pre-allocate with exact size
    for i in range(filtered_count):
        if i < len(filtered_dirs) and filtered_dirs[i] is not None:
            final_filtered_dirs[i] = filtered_dirs[i]
    return final_filtered_dirs


def process_python_files(files: List[str], root: str, exclusion_patterns: Set[str],
                        python_files: List[Path], python_file_count: int) -> int:
    """Process Python files with validation bounds"""
    # Bounded loop for file processing
    for i in range(min(len(files), MAX_FILES_PROCESS)):
        file = files[i]
        if python_file_count >= MAX_FILES_TO_PROCESS:
            break

        file_path = Path(root) / file
        # Check file with explicit bounds checking
        is_excluded = False
        # Bounded loop for pattern checking
        pattern_list = list(exclusion_patterns)
        for j in range(len(pattern_list)):
            pattern = pattern_list[j]
            if len(str(file_path)) <= MAX_PATH_LENGTH and pattern in str(file_path):
                is_excluded = True
                break

        if (file.endswith('.py') and
            not is_excluded and
            python_file_count < len(python_files) and
            file_path.stat().st_size <= 10 * 1024 * 1024):  # 10MB max
            python_files[python_file_count] = file_path
            python_file_count += 1

    return python_file_count


def find_python_files_safely(root_dir: Path) -> List[Path]:
    """Find Python files with validation bounds"""
    # Pre-allocate with known maximum to avoid dynamic resizing
    python_files: List[Path] = [None] * MAX_FILES_TO_PROCESS
    python_file_count = 0

    exclusion_patterns = get_exclusion_patterns()

    # Add bounds checking to os.walk to prevent unbounded directory traversal
    max_directories = 1000  # Safety bound for directory traversal
    directory_count = 0

    for root, dirs, files in os.walk(root_dir):
        if directory_count >= max_directories:
            break  # Safety bound reached
        directory_count += 1

        # Filter excluded directories
        filtered_dirs = filter_directory(dirs, root, exclusion_patterns)
        dirs[:] = filtered_dirs

        # Process Python files
        python_file_count = process_python_files(
            files, root, exclusion_patterns, python_files, python_file_count
        )

    # Trim to actual size
    return python_files[:python_file_count]


class FileOperations:
    """File operations handler for the launcher"""

    def __init__(self, root_dir: Path):
        """Initialize file operations with root directory"""
        self.root_dir = root_dir

    def find_python_files(self) -> List[Path]:
        """Find Python files using the safe file discovery function"""
        return find_python_files_safely(self.root_dir)

