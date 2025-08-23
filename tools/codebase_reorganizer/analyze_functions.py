#!/usr/bin/env python3
"""
High-Reliability Function Length Compliance Analysis
"""

import ast
from pathlib import Path

def check_function_lengths() -> bool:
    """Analyze all Python files for high-reliability function length compliance"""
    python_files = list(Path('.').rglob('*.py'))
    violations = []
    warnings = []

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count lines in function
                    start_line = node.lineno
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        end_line = node.end_lineno
                    else:
                        # Fallback: count lines until next function or end
                        lines = content.split('\n')
                        end_line = start_line
                        for i in range(start_line, len(lines)):
                            end_line = i
                            if i + 1 < len(lines) and lines[i + 1].strip().startswith('def '):
                                break

                    func_length = end_line - start_line + 1

                    if func_length > 60:
                        violations.append(f'{file_path.name}:{node.name} - {func_length} lines')
                    elif func_length > 30:
                        warnings.append(f'{file_path.name}:{node.name} - {func_length} lines')

        except Exception as e:
            print(f'Error parsing {file_path}: {e}')

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
