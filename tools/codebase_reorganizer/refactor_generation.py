#!/usr/bin/env python3
"""
Refactor Generation Module
==========================

Handles generation of refactored functions and files.
"""

from pathlib import Path
from typing import Dict, List, Final
import ast

# Constants
MAX_REFACTORED_LINES: Final[int] = 1000  # Safety bound for refactored lines
MAX_EXTRACTED_FUNCTIONS: Final[int] = 50  # Safety bound for extracted functions
MAX_REMAINING_LINES: Final[int] = 1000  # Safety bound for remaining lines
MAX_EXTRACTED_LINES: Final[int] = 50  # Safety bound for extracted function lines


def extract_function_signature(lines: List[str]) -> tuple[List[str], int, int]:
    """Extract function signature from lines (helper function)"""
    refactored_lines = [None] * 20  # Function signature should be small
    refactored_count = 0
    signature_end = 0

    # Bounded loop for signature copying
    for i in range(min(len(lines), 10)):  # Function signature should be within first 10 lines
        line = lines[i]
        if refactored_count < len(refactored_lines):
            refactored_lines[refactored_count] = line
            refactored_count += 1
        if line.strip().endswith(':'):
            signature_end = i
            break

    return refactored_lines[:refactored_count], refactored_count, signature_end


def create_extracted_function(opportunity: Dict, lines: List[str]) -> List[str]:
    """Create an extracted function from refactoring opportunity (helper function)"""
    lines_range = opportunity['lines'].split('-')
    start_line = int(lines_range[0]) - 1
    end_line = int(lines_range[1]) - 1

    # Extract the block with bounded operation
    block_length = min(end_line - start_line + 1, 200)  # Safety bound for block size
    block_lines = [''] * block_length
    for j in range(block_length):
        if start_line + j < len(lines):
            block_lines[j] = lines[start_line + j]

    # Create extracted function with pre-allocation
    extracted_func = [None] * MAX_EXTRACTED_LINES
    extracted_func[0] = f"def {opportunity['suggested_function']}():"
    extracted_func[1] = f'    """{opportunity["purpose"]}"""'

    # Add indented block lines
    func_line_count = 2
    for j in range(min(len(block_lines), MAX_EXTRACTED_LINES - 3)):
        if func_line_count < MAX_EXTRACTED_LINES:
            extracted_func[func_line_count] = f'    {block_lines[j]}'
            func_line_count += 1

    if func_line_count < MAX_EXTRACTED_LINES:
        extracted_func[func_line_count] = ""
        func_line_count += 1

    return extracted_func[:func_line_count]


def process_refactoring_opportunities(analysis: Dict, lines: List[str]) -> tuple[List[str], int, List[List[str]]]:
    """Process refactoring opportunities and build refactored function (helper function)"""
    refactored_lines = [None] * MAX_REFACTORED_LINES
    refactored_count = 0
    extracted_functions = [None] * MAX_EXTRACTED_FUNCTIONS
    extracted_count = 0

    opportunities = analysis.get('refactoring_opportunities', [])

    # Bounded loop for processing opportunities
    for i in range(min(len(opportunities), MAX_EXTRACTED_FUNCTIONS)):
        opportunity = opportunities[i]

        # Create extracted function using helper
        extracted_func = create_extracted_function(opportunity, lines)

        # Add extracted function to collection
        if extracted_count < MAX_EXTRACTED_FUNCTIONS:
            extracted_functions[extracted_count] = extracted_func
            extracted_count += 1

        # Add function call to main function
        if refactored_count < MAX_REFACTORED_LINES:
            refactored_lines[refactored_count] = f"    {opportunity['suggested_function']}()"
            refactored_count += 1

    return refactored_lines[:refactored_count], refactored_count, extracted_functions[:extracted_count]


def generate_refactored_function(original_content: str, analysis: Dict) -> str:
    """Generate a refactored version of a function with bounded operations (coordinator function)"""
    lines = original_content.split('\n')

    # Extract function signature using helper
    signature_lines, refactored_count, signature_end = extract_function_signature(lines)

    # Process refactoring opportunities using helper
    opportunity_lines, opportunity_count, extracted_functions = process_refactoring_opportunities(
        analysis, lines
    )

    # Combine all parts
    final_lines = signature_lines + opportunity_lines

    # Add extracted functions at the end
    for extracted_func in extracted_functions:
        final_lines.extend(extracted_func)

    return '\n'.join(final_lines)


def generate_refactored_file(original_content: str, refactored_functions: Dict) -> str:
    """Generate a refactored version of the entire file (coordinator function)"""
    # For now, return the original content as the full file refactoring
    # would require more complex AST manipulation
    return original_content


def create_refactored_version(original_file: Path, functions_to_refactor: List) -> str:
    """Create a refactored version of the file"""
    with open(original_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Parse the AST to understand the structure
    tree = ast.parse(content)

    refactored_functions = {}

    # Find and analyze each function that needs refactoring
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            if func_name in [f['function_name'] for f in functions_to_refactor]:
                # Get the function content
                lines = content.split('\n')
                start_line = node.lineno - 1  # AST uses 1-based, list uses 0-based
                end_line = getattr(node, 'end_lineno', len(lines)) - 1

                func_content = '\n'.join(lines[start_line:end_line+1])

                # Analyze for refactoring
                from refactor_analysis import analyze_function_for_refactoring
                analysis = analyze_function_for_refactoring(func_content, func_name)

                if analysis['refactoring_opportunities']:
                    refactored_functions[func_name] = {
                        'analysis': analysis,
                        'original_content': func_content,
                        'refactored_content': generate_refactored_function(func_content, analysis)
                    }

    return generate_refactored_file(content, refactored_functions)
