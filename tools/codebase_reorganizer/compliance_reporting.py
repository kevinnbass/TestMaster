#!/usr/bin/env python3
"""
Compliance Reporting Module
==========================

Handles display and reporting of compliance analysis results.
"""

from typing import List, Tuple, Dict, Any, Final

# Constants
MAX_DISPLAY_ITEMS: Final[int] = 10  # Safety bound for display
MAX_MISSING_ITEMS: Final[int] = 1000  # Safety bound for missing items


def print_compliance_results(total_modules: int, modules_with_docstrings: int,
                            total_functions: int, functions_with_docstrings: int,
                            functions_with_type_hints: int, missing_docstrings: List[str],
                            missing_type_hints: List[str]) -> None:
    """Print compliance analysis results"""
    print(f'Total modules analyzed: {total_modules}')
    print(f'Modules with docstrings: {modules_with_docstrings} ({modules_with_docstrings/total_modules*100:.1f}%)')
    print(f'Total functions analyzed: {total_functions}')
    print(f'Functions with docstrings: {functions_with_docstrings} ({functions_with_docstrings/total_functions*100:.1f}%)')
    print(f'Functions with type hints: {functions_with_type_hints} ({functions_with_type_hints/total_functions*100:.1f}%)')

    # Show missing items with bounded loops
    if missing_docstrings:
        print(f'\n📝 FUNCTIONS MISSING DOCSTRINGS ({len(missing_docstrings)}):')
        # Bounded loop for displaying missing docstrings
        for i in range(min(len(missing_docstrings), MAX_DISPLAY_ITEMS)):
            print(f'   • {missing_docstrings[i]}')
        if len(missing_docstrings) > MAX_DISPLAY_ITEMS:
            print(f'   ... and {len(missing_docstrings) - MAX_DISPLAY_ITEMS} more')

    if missing_type_hints:
        print(f'\n🏷️  FUNCTIONS MISSING TYPE HINTS ({len(missing_type_hints)}):')
        # Bounded loop for displaying missing type hints
        for i in range(min(len(missing_type_hints), MAX_DISPLAY_ITEMS)):
            print(f'   • {missing_type_hints[i]}')
        if len(missing_type_hints) > MAX_DISPLAY_ITEMS:
            print(f'   ... and {len(missing_type_hints) - MAX_DISPLAY_ITEMS} more')


def calculate_compliance_status(total_functions: int, functions_with_docstrings: int,
                               functions_with_type_hints: int, total_modules: int,
                               modules_with_docstrings: int) -> Tuple[bool, bool]:
    """Calculate compliance status"""
    docstring_compliance = functions_with_docstrings == total_functions and modules_with_docstrings == total_modules
    type_hint_compliance = functions_with_type_hints / total_functions >= 0.95  # 95% compliance
    return docstring_compliance, type_hint_compliance


def display_compliance_status(total_functions: int, functions_with_docstrings: int,
                              functions_with_type_hints: int, total_modules: int,
                              modules_with_docstrings: int) -> None:
    """Display compliance status (helper function)"""
    # Calculate and display compliance status
    docstring_compliance, type_hint_compliance = calculate_compliance_status(
        total_functions, functions_with_docstrings,
        functions_with_type_hints, total_modules, modules_with_docstrings
    )

    print('\n📋 COMPLIANCE STATUS:')
    print(f'   Docstring compliance: {"✅ FULLY COMPLIANT" if docstring_compliance else "❌ NEEDS WORK"}')
    print(f'   Type hint compliance: {"✅ FULLY COMPLIANT" if type_hint_compliance else "⚠️  GOOD PROGRESS"}')

    if docstring_compliance and type_hint_compliance:
        print('\n🎉 HIGH-RELIABILITY COMPLIANCE ACHIEVED!')
        print('   All modules have detailed docstrings')
        print('   All functions have mandatory type hints')
        print('   Ready for mypy strict type checking')
    else:
        print('\n⚠️  Additional work needed for full compliance')


def print_compliance_summary(analysis_results: Dict[str, Any]) -> None:
    """Print a comprehensive compliance summary"""
    total_modules = analysis_results.get('total_modules', 0)
    modules_with_docstrings = analysis_results.get('modules_with_docstrings', 0)
    total_functions = analysis_results.get('total_functions', 0)
    functions_with_docstrings = analysis_results.get('functions_with_docstrings', 0)
    functions_with_type_hints = analysis_results.get('functions_with_type_hints', 0)

    print('\n' + '=' * 60)
    print('📊 HIGH-RELIABILITY COMPLIANCE SUMMARY')
    print('=' * 60)

    print(f'📁 Modules Analyzed: {total_modules}')
    if total_modules > 0:
        module_compliance_rate = (modules_with_docstrings / total_modules) * 100
        print(f'📝 Module Docstrings: {modules_with_docstrings}/{total_modules} ({module_compliance_rate:.1f}%)')

    print(f'🔧 Functions Analyzed: {total_functions}')
    if total_functions > 0:
        docstring_rate = (functions_with_docstrings / total_functions) * 100
        typehint_rate = (functions_with_type_hints / total_functions) * 100
        print(f'📝 Function Docstrings: {functions_with_docstrings}/{total_functions} ({docstring_rate:.1f}%)')
        print(f'🏷️  Function Type Hints: {functions_with_type_hints}/{total_functions} ({typehint_rate:.1f}%)')

    # Overall compliance assessment
    docstring_compliance, type_hint_compliance = calculate_compliance_status(
        total_functions, functions_with_docstrings, functions_with_type_hints,
        total_modules, modules_with_docstrings
    )

    print('\n🎯 COMPLIANCE ASSESSMENT:')
    if docstring_compliance and type_hint_compliance:
        print('   ✅ FULLY COMPLIANT - Ready for high-reliability deployment!')
    elif docstring_compliance:
        print('   📝 DOCSTRINGS: ✅ Complete')
        print('   🏷️  TYPE HINTS: ⚠️  Needs more work')
    elif type_hint_compliance:
        print('   📝 DOCSTRINGS: ❌ Needs completion')
        print('   🏷️  TYPE HINTS: ✅ Good progress')
    else:
        print('   📝 DOCSTRINGS: ❌ Needs completion')
        print('   🏷️  TYPE HINTS: ⚠️  Needs more work')

    print('=' * 60)


def generate_compliance_report(analysis_results: Dict[str, Any]) -> str:
    """Generate a detailed compliance report as a string"""
    total_modules = analysis_results.get('total_modules', 0)
    modules_with_docstrings = analysis_results.get('modules_with_docstrings', 0)
    total_functions = analysis_results.get('total_functions', 0)
    functions_with_docstrings = analysis_results.get('functions_with_docstrings', 0)
    functions_with_type_hints = analysis_results.get('functions_with_type_hints', 0)

    # Pre-allocate with known capacity (Rule 3 compliance)
    MAX_REPORT_LINES = 10  # Safety bound for report lines
    report_lines = [None] * MAX_REPORT_LINES
    line_count = 0

    # Add lines with bounds checking
    report_lines[0] = "HIGH-RELIABILITY COMPLIANCE REPORT"
    report_lines[1] = "=" * 40
    report_lines[2] = f"Modules Analyzed: {total_modules}"
    report_lines[3] = f"Modules with Docstrings: {modules_with_docstrings}"
    report_lines[4] = f"Functions Analyzed: {total_functions}"
    report_lines[5] = f"Functions with Docstrings: {functions_with_docstrings}"
    report_lines[6] = f"Functions with Type Hints: {functions_with_type_hints}"
    line_count = 7

    if total_modules > 0:
        module_rate = (modules_with_docstrings / total_modules) * 100
        if line_count < MAX_REPORT_LINES:
            report_lines[line_count] = f"Module Docstring Compliance: {module_rate:.1f}%"
            line_count += 1

    if total_functions > 0:
        docstring_rate = (functions_with_docstrings / total_functions) * 100
        typehint_rate = (functions_with_type_hints / total_functions) * 100
        if line_count < MAX_REPORT_LINES:
            report_lines[line_count] = f"Function Docstring Compliance: {docstring_rate:.1f}%"
            line_count += 1
        if line_count < MAX_REPORT_LINES:
            report_lines[line_count] = f"Function Type Hint Compliance: {typehint_rate:.1f}%"
            line_count += 1

    # Return joined lines (bounded operation)
    return '\n'.join(report_lines[:line_count])
