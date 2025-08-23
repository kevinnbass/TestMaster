#!/usr/bin/env python3
"""
Compliance File Discovery Module
===============================

Handles safe and bounded file discovery for compliance analysis.
"""

from pathlib import Path
from typing import List, Set, Final

# Constants
MAX_FILES: Final[int] = 1000  # Safety bound for file processing
MAX_CORE_FILES: Final[int] = 100  # Safety bound for core files
MAX_PATTERN_CHECKS: Final[int] = 20  # Safety bound for pattern checking

# Excluded patterns for compliance check (exclude demo and utility files)
EXCLUDED_PATTERNS: Final[Set[str]] = {
    'demo_', 'test_', 'refactor_', 'analyze_', 'run_', 'intelligence_',
    'meta_', 'pattern_', 'relationship_', 'semantic_', 'quality_'
}


def get_core_python_files() -> List[Path]:
    """Get core Python files for compliance analysis with pre-allocation"""
    # Add bounds checking to prevent unbounded file discovery
    python_files_iter = Path('.').rglob('*.py')

    # Pre-allocate list with known capacity (Rule 3 compliance)
    python_files = [Path('.')] * MAX_FILES  # Pre-allocate with placeholder
    file_count = 0

    # Bounded iteration with safety limit
    for i, file_path in enumerate(python_files_iter):
        if i >= MAX_FILES:
            break  # Safety bound reached
        if file_count < MAX_FILES:
            python_files[file_count] = file_path
            file_count += 1

    # Return slice with actual count (bounded operation)
    python_files = python_files[:file_count]

    # Focus on core files for compliance check (exclude demo and utility files)
    # Pre-allocate core_files with estimated capacity
    core_files = [Path('.')] * MAX_CORE_FILES  # Pre-allocate with placeholder
    core_count = 0

    # Bounded loop with safety check
    for i in range(len(python_files)):
        f = python_files[i]
        exclude = False
        # Bounded loop for pattern checking
        pattern_list = list(EXCLUDED_PATTERNS)  # Convert to list for bounded iteration
        for j in range(min(len(pattern_list), MAX_PATTERN_CHECKS)):
            if pattern_list[j] in f.name:
                exclude = True
                break

        if not exclude and core_count < MAX_CORE_FILES:
            core_files[core_count] = f
            core_count += 1

    # Return slice with actual count
    return core_files[:core_count]


def filter_excluded_files(python_files: List[Path]) -> List[Path]:
    """Filter out excluded files from the file list"""
    # Pre-allocate filtered list
    MAX_FILTERED_FILES = len(python_files)
    filtered_files = [Path('.')] * MAX_FILTERED_FILES
    filtered_count = 0

    # Bounded loop for filtering
    for i in range(len(python_files)):
        f = python_files[i]
        exclude = False

        # Check against excluded patterns with bounded iteration
        pattern_list = list(EXCLUDED_PATTERNS)
        for j in range(min(len(pattern_list), MAX_PATTERN_CHECKS)):
            if pattern_list[j] in f.name:
                exclude = True
                break

        if not exclude and filtered_count < MAX_FILTERED_FILES:
            filtered_files[filtered_count] = f
            filtered_count += 1

    return filtered_files[:filtered_count]


def validate_file_for_analysis(file_path: Path) -> bool:
    """Validate if a file should be included in compliance analysis"""
    if not file_path.exists() or not file_path.is_file():
        return False

    # Check file size (avoid very large files)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    if file_path.stat().st_size > MAX_FILE_SIZE:
        return False

    # Check against excluded patterns
    pattern_list = list(EXCLUDED_PATTERNS)
    for i in range(min(len(pattern_list), MAX_PATTERN_CHECKS)):
        if pattern_list[i] in file_path.name:
            return False

    return True
