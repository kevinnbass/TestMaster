#!/usr/bin/env python3
"""
Refactor Launcher File Operations Module
=======================================

Handles safe file discovery and processing operations for the refactored launcher.
"""

import os
from pathlib import Path
from typing import List, Set, Final

# Constants
MAX_FILES: Final[int] = 1000  # Safety bound for file processing
MAX_DIRECTORIES: Final[int] = 5000  # Safety bound for directory processing
MAX_PATH_LENGTH: Final[int] = 260
MAX_DIRS_FILTER: Final[int] = 200  # Safety bound for directory filtering
MAX_FILES_PROCESS: Final[int] = 1000  # Safety bound for file processing
MAX_PATTERNS: Final[int] = 10  # Safety bound for exclusion patterns


def get_exclusion_patterns() -> Set[str]:
    """Get exclusion patterns for file filtering"""
    return {
        '**/node_modules/**', '**/.*', '**/test*/**', '**/archive*/**',
        '**/__pycache__/**', '**/.*'
    }


def filter_directories(dirs: List[str], root: str, exclusion_patterns: Set[str]) -> List[str]:
    """Filter directories based on exclusion patterns"""
    # Remove excluded directories with pre-allocation (Rule 3 compliance)
    filtered_dirs = [None] * 100  # Pre-allocate with fixed upper bound
    filtered_count = 0

    for i in range(min(len(dirs), MAX_DIRS_FILTER)):
        d = dirs[i]
        if filtered_count >= 100:  # Fixed upper bound
            break
        should_exclude = False
        # Bounded loop for pattern checking (Rule 2 compliance)
        patterns_list = list(exclusion_patterns)
        for j in range(min(len(patterns_list), MAX_PATTERNS)):
            pattern = patterns_list[j]
            if len(str(Path(root) / d)) <= MAX_PATH_LENGTH and pattern in str(Path(root) / d):
                should_exclude = True
                break
        if not should_exclude:
            filtered_dirs[filtered_count] = d
            filtered_count += 1

    return filtered_dirs[:filtered_count]  # Use actual count (bounded operation)


def is_file_eligible(file_path: Path, exclusion_patterns: Set[str]) -> bool:
    """Check if file is eligible for processing"""
    # Check file exclusion with bounded loop (Rule 2 compliance)
    patterns_list = list(exclusion_patterns)
    for j in range(min(len(patterns_list), MAX_PATTERNS)):
        pattern = patterns_list[j]
        if len(str(file_path)) <= MAX_PATH_LENGTH and pattern in str(file_path):
            return False

    return (file_path.suffix == '.py' and
            file_path.stat().st_size <= 10 * 1024 * 1024)


def process_directory_files(files: List[str], root: str,
                          exclusion_patterns: Set[str], python_files: List[Path],
                          file_count: int, MAX_FILES: int) -> int:
    """Process files in a directory"""
    current_count = file_count

    for i in range(min(len(files), MAX_FILES_PROCESS)):
        if current_count >= MAX_FILES:
            break  # Safety bound reached

        file_path = Path(root) / files[i]
        if is_file_eligible(file_path, exclusion_patterns):
            python_files[current_count] = file_path
            current_count += 1

    return current_count


def find_python_files_safely(root_dir: Path) -> List[Path]:
    """Find Python files with validation bounds (coordinator function)"""
    # Pre-allocate python_files with known capacity (Rule 3 compliance)
    python_files = [Path('.')] * MAX_FILES
    file_count = 0

    exclusion_patterns = get_exclusion_patterns()

    # Bounded loop for directory traversal
    directory_count = 0

    for root, dirs, files in os.walk(root_dir):
        if directory_count >= MAX_DIRECTORIES:
            break
        directory_count += 1

        # Filter directories using helper
        dirs[:] = filter_directories(dirs, root, exclusion_patterns)

        # Process files using helper
        file_count = process_directory_files(files, root, exclusion_patterns,
                                           python_files, file_count, MAX_FILES)

    return python_files[:file_count]  # Return actual data (bounded operation)


class FileOperations:
    """Handles file operations for the refactored launcher"""

    def __init__(self, root_dir: Path):
        """Initialize file operations with root directory"""
        self.root_dir = root_dir

    def find_python_files(self) -> List[Path]:
        """Find Python files using safe file discovery"""
        return find_python_files_safely(self.root_dir)
