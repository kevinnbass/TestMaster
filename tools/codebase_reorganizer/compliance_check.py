#!/usr/bin/env python3
"""
High-Reliability Compliance Check Coordinator
=============================================

Coordinates the compliance checking process using specialized modules.
"""

from pathlib import Path
from typing import Final

# Import specialized modules
from compliance_file_discovery import get_core_python_files
from compliance_analysis import analyze_compliance_batch
from compliance_reporting import (
    print_compliance_results,
    display_compliance_status,
    print_compliance_summary
)

# Constants
MAX_ANALYSIS_FILES: Final[int] = 100  # Safety bound for analysis


def _print_compliance_header() -> None:
    """Print compliance check header"""
    print('🎯 HIGH-RELIABILITY COMPLIANCE CHECK')
    print('=' * 35)
    print('Analyzing code for NASA-STD-8719.13 compliance')
    print('Rules: Function size limits, type hints, docstrings')
    print('=' * 35)


def _get_and_validate_files() -> list[Path]:
    """Get and validate Python files for analysis"""
    python_files = get_core_python_files()
    print(f'📁 Found {len(python_files)} core Python files to analyze')

    if not python_files:
        print('❌ No Python files found for analysis')
        return []

    # Limit analysis to prevent unbounded processing
    files_to_analyze = python_files[:MAX_ANALYSIS_FILES]

    if len(python_files) > MAX_ANALYSIS_FILES:
        print(f'⚠️  Limiting analysis to first {MAX_ANALYSIS_FILES} files for performance')

    return files_to_analyze


def _collect_missing_items(analysis_results: dict) -> tuple[list, list]:
    """Collect missing items from analysis results"""
    # Pre-allocate with known capacity (Rule 3 compliance)
    MAX_MISSING_ITEMS = 500  # Safety bound for missing items
    all_missing_docstrings = [None] * MAX_MISSING_ITEMS
    all_missing_type_hints = [None] * MAX_MISSING_ITEMS
    docstring_count = 0
    typehint_count = 0

    # Bounded loop for collecting missing items
    for i in range(min(len(analysis_results['results']), MAX_MISSING_ITEMS)):
        result = analysis_results['results'][i]
        if result and result.get('missing_docstrings'):
            missing_docs = result['missing_docstrings']
            for j in range(min(len(missing_docs), MAX_MISSING_ITEMS - docstring_count)):
                if docstring_count < MAX_MISSING_ITEMS:
                    all_missing_docstrings[docstring_count] = missing_docs[j]
                    docstring_count += 1

        if result and result.get('missing_type_hints'):
            missing_hints = result['missing_type_hints']
            for j in range(min(len(missing_hints), MAX_MISSING_ITEMS - typehint_count)):
                if typehint_count < MAX_MISSING_ITEMS:
                    all_missing_type_hints[typehint_count] = missing_hints[j]
                    typehint_count += 1

    # Use actual counts (Rule 3 compliance)
    return (all_missing_docstrings[:docstring_count],
            all_missing_type_hints[:typehint_count])


def check_high_reliability_compliance() -> None:
    """Check high-reliability compliance for docstrings and type hints (coordinator function)"""
    _print_compliance_header()

    try:
        # Get and validate files
        files_to_analyze = _get_and_validate_files()
        if not files_to_analyze:
            return

        # Analyze files using batch processing
        analysis_results = analyze_compliance_batch(files_to_analyze)

        # Extract results for display
        total_modules = analysis_results['total_modules']
        modules_with_docstrings = analysis_results['modules_with_docstrings']
        total_functions = analysis_results['total_functions']
        functions_with_docstrings = analysis_results['functions_with_docstrings']
        functions_with_type_hints = analysis_results['functions_with_type_hints']

        # Collect missing items
        all_missing_docstrings, all_missing_type_hints = _collect_missing_items(analysis_results)

        # Display comprehensive results
        print_compliance_results(
            total_modules, modules_with_docstrings, total_functions,
            functions_with_docstrings, functions_with_type_hints,
            all_missing_docstrings, all_missing_type_hints
        )

        display_compliance_status(
            total_functions, functions_with_docstrings, functions_with_type_hints,
            total_modules, modules_with_docstrings
        )

        # Print summary
        print_compliance_summary(analysis_results)

    except Exception as e:
        print(f'❌ Error during compliance check: {e}')
        print('   Please check file permissions and syntax')