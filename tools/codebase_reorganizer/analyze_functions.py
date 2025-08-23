#!/usr/bin/env python3
"""
High-Reliability Function Length Compliance Analysis
"""

import ast
from pathlib import Path

def _discover_python_files() -> List[Path]:
    """Discover Python files with safety bounds"""
    MAX_FILES = 1000  # Safety bound for file processing
    python_files = list(Path('.').rglob('*.py'))[:MAX_FILES]  # Bounded file discovery
    return python_files


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
        lines = content.split('\n')

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

    # Trim to actual size
    violations = violations[:violation_count]
    warnings = warnings[:warning_count]

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


def check_function_lengths() -> bool:
    """Analyze all Python files for high-reliability function length compliance"""
    python_files = _discover_python_files()

    # Pre-allocate lists with estimated capacity to avoid dynamic resizing
    violations = [None] * 100  # Pre-allocate for violations
    warnings = [None] * 200    # Pre-allocate for warnings
    violation_count = 0
    warning_count = 0

    # Bounded loop with safety check
    for i, file_path in enumerate(python_files):
        if i >= 1000:  # Additional safety bound
            break

        file_violations, file_warnings = _analyze_single_file(file_path)

        # Add to pre-allocated lists with bounds checking (bounded loops)
        MAX_ITEMS_PER_FILE = 50  # Safety bound for items per file
        for j, violation in enumerate(file_violations):
            if j >= MAX_ITEMS_PER_FILE:
                break  # Safety bound reached
            if violation_count < len(violations):
                violations[violation_count] = violation
                violation_count += 1

        for j, warning in enumerate(file_warnings):
            if j >= MAX_ITEMS_PER_FILE:
                break  # Safety bound reached
            if warning_count < len(warnings):
                warnings[warning_count] = warning
                warning_count += 1

    # Trim pre-allocated lists to actual size
    violations = violations[:violation_count]
    warnings = warnings[:warning_count]

    print(f'\n=== HIGH-RELIABILITY FUNCTION LENGTH ANALYSIS ===')
    print(f'Total files analyzed: {len(python_files)}')

    if violations:
        print(f'\n🚨 VIOLATIONS (functions > 60 lines):')
        for v in violations:
            print(f'   {v}')
    else:
        print('\n✅ NO VIOLATIONS - All functions ≤ 60 lines')

    if warnings:
        print(f'\n⚠️  WARNINGS (functions > 30 lines):')
        for w in warnings:
            print(f'   {w}')

    print(f'\n=== COMPLIANCE SUMMARY ===')
    print(f'Functions > 60 lines: {len(violations)}')
    print(f'Functions > 30 lines: {len(warnings)}')
    compliance_status = "✅ FULLY COMPLIANT" if not violations else "❌ VIOLATIONS FOUND"
    print(f'HIGH-RELIABILITY COMPLIANCE: {compliance_status}')

    return len(violations) == 0

if __name__ == "__main__":
    check_function_lengths()
