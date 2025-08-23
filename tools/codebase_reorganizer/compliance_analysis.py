#!/usr/bin/env python3
"""
Compliance Analysis Module
=========================

Handles AST analysis and compliance checking for docstrings and type hints.
"""

import ast
from pathlib import Path
from typing import Dict, Any, List, Final

# Constants
MAX_FUNCTIONS_PER_FILE: Final[int] = 100  # Safety bound for functions per file
MAX_AST_NODES: Final[int] = 2000  # Safety bound for AST nodes


def check_module_docstring(tree: ast.Module) -> bool:
    """Check if module has a docstring (helper function)"""
    module_docstring = ast.get_docstring(tree)
    return module_docstring is not None


def analyze_function_compliance(node: ast.FunctionDef, file_path: Path) -> Dict[str, Any]:
    """Analyze a single function for compliance metrics (helper function)"""
    func_name = f"{file_path.name}:{node.name}"

    # Check docstring
    has_docstring = ast.get_docstring(node) is not None

    # Check type hints
    has_return_hint = node.returns is not None
    has_arg_hints = all(arg.annotation is not None for arg in node.args.args if arg.arg != 'self')

    return {
        'func_name': func_name,
        'has_docstring': has_docstring,
        'has_type_hints': has_return_hint and has_arg_hints
    }


def analyze_functions_in_file(tree: ast.Module, file_path: Path) -> Dict[str, Any]:
    """Analyze all functions in a file for compliance (helper function)"""
    # Pre-allocate with estimated capacity (Rule 3 compliance)
    missing_docstrings = [None] * MAX_FUNCTIONS_PER_FILE
    missing_type_hints = [None] * MAX_FUNCTIONS_PER_FILE
    docstring_count = 0
    typehint_count = 0

    total_functions = 0
    functions_with_docstrings = 0
    functions_with_type_hints = 0

    # Bounded loop for AST node processing
    nodes_list = list(ast.walk(tree))
    for i in range(min(len(nodes_list), MAX_AST_NODES)):
        node = nodes_list[i]
        if isinstance(node, ast.FunctionDef):
            total_functions += 1
            func_result = analyze_function_compliance(node, file_path)

            if func_result['has_docstring']:
                functions_with_docstrings += 1
            else:
                # Use pre-allocated list with bounds checking
                if docstring_count < MAX_FUNCTIONS_PER_FILE:
                    missing_docstrings[docstring_count] = func_result['func_name']
                    docstring_count += 1

            if func_result['has_type_hints']:
                functions_with_type_hints += 1
            else:
                # Use pre-allocated list with bounds checking
                if typehint_count < MAX_FUNCTIONS_PER_FILE:
                    missing_type_hints[typehint_count] = func_result['func_name']
                    typehint_count += 1

    return {
        'total_functions': total_functions,
        'functions_with_docstrings': functions_with_docstrings,
        'functions_with_type_hints': functions_with_type_hints,
        'missing_docstrings': missing_docstrings[:docstring_count],
        'missing_type_hints': missing_type_hints[:typehint_count]
    }


def analyze_single_file_compliance(file_path: Path) -> Dict[str, Any]:
    """Analyze a single file for compliance metrics"""
    try:
        # Read file content with safety bounds
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit for compliance analysis
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return {
                'file_path': str(file_path),
                'error': f"File too large: {file_path.stat().st_size} bytes",
                'total_functions': 0,
                'functions_with_docstrings': 0,
                'functions_with_type_hints': 0,
                'has_module_docstring': False
            }

        # Parse AST with error handling
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                'file_path': str(file_path),
                'error': f"Syntax error: {e}",
                'total_functions': 0,
                'functions_with_docstrings': 0,
                'functions_with_type_hints': 0,
                'has_module_docstring': False
            }

        # Check module docstring
        has_module_docstring = check_module_docstring(tree)

        # Analyze functions
        function_analysis = analyze_functions_in_file(tree, file_path)

        return {
            'file_path': str(file_path),
            'total_functions': function_analysis['total_functions'],
            'functions_with_docstrings': function_analysis['functions_with_docstrings'],
            'functions_with_type_hints': function_analysis['functions_with_type_hints'],
            'has_module_docstring': has_module_docstring,
            'missing_docstrings': function_analysis['missing_docstrings'],
            'missing_type_hints': function_analysis['missing_type_hints'],
            'error': None
        }

    except Exception as e:
        return {
            'file_path': str(file_path),
            'error': f"Analysis error: {e}",
            'total_functions': 0,
            'functions_with_docstrings': 0,
            'functions_with_type_hints': 0,
            'has_module_docstring': False
        }


def analyze_compliance_batch(files: List[Path]) -> Dict[str, Any]:
    """Analyze a batch of files for compliance"""
    # Pre-allocate results with known capacity
    MAX_BATCH_SIZE = 100  # Safety bound for batch processing
    results = [None] * min(len(files), MAX_BATCH_SIZE)
    results_count = 0

    total_functions = 0
    functions_with_docstrings = 0
    functions_with_type_hints = 0
    modules_with_docstrings = 0
    total_modules = 0

    # Bounded loop for batch processing
    for i in range(min(len(files), MAX_BATCH_SIZE)):
        file_path = files[i]
        result = analyze_single_file_compliance(file_path)

        if result['error'] is None:
            total_modules += 1
            if result['has_module_docstring']:
                modules_with_docstrings += 1

            total_functions += result['total_functions']
            functions_with_docstrings += result['functions_with_docstrings']
            functions_with_type_hints += result['functions_with_type_hints']

        if results_count < len(results):
            results[results_count] = result
            results_count += 1

    return {
        'results': results[:results_count],
        'total_modules': total_modules,
        'modules_with_docstrings': modules_with_docstrings,
        'total_functions': total_functions,
        'functions_with_docstrings': functions_with_docstrings,
        'functions_with_type_hints': functions_with_type_hints
    }
