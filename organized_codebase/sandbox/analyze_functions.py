#!/usr/bin/env python3
"""
High-Reliability Function Length Compliance Analysis
"""

import ast
from pathlib import Path

def _discover_python_files() -> List[Path]:
    """Discover Python files with safety bounds and pre-allocation"""
    MAX_FILES = 1000  # Safety bound for file processing
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
    return python_files[:file_count]


def _analyze_single_file(file_path: Path) -> Tuple[List[str], List[str]]:
    """Analyze a single file for function length compliance"""
    # Pre-allocate with estimated capacity to avoid dynamic resizing
    MAX_FUNCTIONS = 50  # Safety bound for functions per file
    violations = [None] * MAX_FUNCTIONS
    warnings = [None] * MAX_FUNCTIONS
    violation_count = 0
    warning_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        tree = ast.parse(content)

        # Bounded line splitting with pre-allocation (Rule 3 compliance)
        MAX_LINES = 10000  # Safety bound for file lines
        raw_lines = content.split('\n')
        # Pre-allocate lines with known capacity
        lines = [''] * MAX_LINES  # Pre-allocate with placeholder
        actual_lines = min(len(raw_lines), MAX_LINES)
        for i in range(actual_lines):
            lines[i] = raw_lines[i]

        # Bounded loop with safety check
        function_index = 0
        for node in ast.walk(tree):
            if function_index >= MAX_FUNCTIONS:
                break  # Safety bound reached

            if isinstance(node, ast.FunctionDef):
                func_length = _calculate_function_length(node, lines)
                func_name = node.name

                if func_length > 60 and violation_count < MAX_FUNCTIONS:
                    violations[violation_count] = f'{file_path.name}:{func_name} - {func_length} lines'
                    violation_count += 1
                elif func_length > 30 and warning_count < MAX_FUNCTIONS:
                    warnings[warning_count] = f'{file_path.name}:{func_name} - {func_length} lines'
                    warning_count += 1

                function_index += 1

    except Exception as e:
        print(f'Error parsing {file_path}: {e}')

    # Create final lists with pre-allocation (Rule 3 compliance)
    final_violations = [None] * violation_count  # Pre-allocate with exact size
    for i in range(violation_count):
        if i < MAX_FUNCTIONS:
            final_violations[i] = violations[i]
    violations = final_violations

    final_warnings = [None] * warning_count  # Pre-allocate with exact size
    for i in range(warning_count):
        if i < MAX_FUNCTIONS:
            final_warnings[i] = warnings[i]
    warnings = final_warnings

    return violations, warnings


def _calculate_function_length(node: ast.FunctionDef, lines: List[str]) -> int:
    """Calculate function length with bounds checking"""
    start_line = node.lineno
    if hasattr(node, 'end_lineno') and node.end_lineno:
        return node.end_lineno - start_line + 1

    # Fallback: count lines until next function or end
    end_line = start_line
    max_search_lines = min(1000, len(lines) - start_line)  # Safety bound

    for j in range(max_search_lines):
        current_line = start_line + j - 1
        if current_line >= len(lines):
            break
        end_line = current_line
        if (current_line + 1 < len(lines) and
            lines[current_line + 1].strip().startswith('def ')):
            break

    return end_line - start_line + 1


def _initialize_analysis_storage() -> Tuple[List[Optional[str]], List[Optional[str]]]:
    """Initialize pre-allocated storage for analysis results"""
    MAX_VIOLATIONS = 100  # Safety bound for violations
    MAX_WARNINGS = 200    # Safety bound for warnings
    violations = [None] * MAX_VIOLATIONS
    warnings = [None] * MAX_WARNINGS
    return violations, warnings


def _process_file_violations(file_violations: List[str], file_warnings: List[str],
                           violations: List[Optional[str]], warnings: List[Optional[str]],
                           violation_count: int, warning_count: int) -> Tuple[int, int]:
    """Process violations from a single file with bounds checking"""
    MAX_ITEMS_PER_FILE = 50  # Safety bound for items per file

    # Process violations with bounded loops
    for j in range(min(len(file_violations), MAX_ITEMS_PER_FILE)):
        if violation_count < len(violations):
            violations[violation_count] = file_violations[j]
            violation_count += 1

    # Process warnings with bounded loops
    for j in range(min(len(file_warnings), MAX_ITEMS_PER_FILE)):
        if warning_count < len(warnings):
            warnings[warning_count] = file_warnings[j]
            warning_count += 1

    return violation_count, warning_count


def _finalize_analysis_results(violations: List[Optional[str]], warnings: List[Optional[str]],
                             violation_count: int, warning_count: int) -> Tuple[List[str], List[str]]:
    """Create final lists with pre-allocation (Rule 3 compliance)"""
    # Create final violations list with pre-allocation
    final_violations = [None] * violation_count  # Pre-allocate with exact size
    violation_index = 0
    for i in range(violation_count):
        if i < len(violations) and violations[i] is not None:
            final_violations[violation_index] = violations[i]
            violation_index += 1

    # Create final warnings list with pre-allocation
    final_warnings = [None] * warning_count  # Pre-allocate with exact size
    warning_index = 0
    for i in range(warning_count):
        if i < len(warnings) and warnings[i] is not None:
            final_warnings[warning_index] = warnings[i]
            warning_index += 1

    return final_violations, final_warnings


def _print_analysis_results(python_files: List[Path], violations: List[str], warnings: List[str]) -> None:
    """Print analysis results with compliance summary"""
    print(f'\n=== HIGH-RELIABILITY FUNCTION LENGTH ANALYSIS ===')
    print(f'Total files analyzed: {len(python_files)}')

    if violations:
        print(f'\n🚨 VIOLATIONS (functions > 60 lines):')
        # Bounded loop for printing violations
        MAX_PRINT_ITEMS = 100  # Safety bound for printing
        for i in range(min(len(violations), MAX_PRINT_ITEMS)):
            print(f'   {violations[i]}')
    else:
        print('\n✅ NO VIOLATIONS - All functions ≤ 60 lines')

    if warnings:
        print(f'\n⚠️  WARNINGS (functions > 30 lines):')
        # Bounded loop for printing warnings
        for i in range(min(len(warnings), MAX_PRINT_ITEMS)):
            print(f'   {warnings[i]}')

    print(f'\n=== COMPLIANCE SUMMARY ===')
    print(f'Functions > 60 lines: {len(violations)}')
    print(f'Functions > 30 lines: {len(warnings)}')
    compliance_status = "✅ FULLY COMPLIANT" if not violations else "❌ VIOLATIONS FOUND"
    print(f'HIGH-RELIABILITY COMPLIANCE: {compliance_status}')


def check_function_lengths() -> bool:
    """Analyze all Python files for high-reliability function length compliance"""
    python_files = _discover_python_files()
    violations, warnings = _initialize_analysis_storage()
    violation_count = 0
    warning_count = 0

    # Bounded loop with safety check
    MAX_FILES = 1000  # Safety bound for file processing
    for i in range(min(len(python_files), MAX_FILES)):
        file_path = python_files[i]
        file_violations, file_warnings = _analyze_single_file(file_path)
        violation_count, warning_count = _process_file_violations(
            file_violations, file_warnings, violations, warnings,
            violation_count, warning_count
        )

    # Create final results
    final_violations, final_warnings = _finalize_analysis_results(
        violations, warnings, violation_count, warning_count
    )

    # Print results
    _print_analysis_results(python_files, final_violations, final_warnings)

    return len(final_violations) == 0

if __name__ == "__main__":
    check_function_lengths()
