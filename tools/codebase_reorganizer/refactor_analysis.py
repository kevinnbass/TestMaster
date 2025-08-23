#!/usr/bin/env python3
"""
Refactor Analysis Module
========================

Handles analysis and suggestion functions for function refactoring.
"""

from typing import Dict, List, Tuple, Final
import re

# Constants
MAX_OPPORTUNITIES: Final[int] = 50  # Safety bound for refactoring opportunities
MAX_BLOCKS: Final[int] = 100  # Safety bound for blocks
MAX_BLOCK_SIZE: Final[int] = 200  # Safety bound for block size
MAX_LINES_ANALYZE: Final[int] = 1000  # Safety bound for line analysis


def initialize_refactoring_analysis(func_name: str, total_lines: int) -> Dict:
    """Initialize refactoring analysis structure (helper function)"""
    return {
        'function_name': func_name,
        'total_lines': total_lines,
        'refactoring_opportunities': [None] * MAX_OPPORTUNITIES,
        'suggested_functions': [None] * MAX_OPPORTUNITIES
    }


def extract_logical_blocks(lines: List[str]) -> List[Tuple[int, int, List[str]]]:
    """Extract logical blocks from function content (helper function)"""
    blocks = [None] * MAX_BLOCKS  # Pre-allocate with placeholder
    block_count = 0
    current_block = [None] * MAX_BLOCK_SIZE  # Pre-allocate current block
    current_block_size = 0
    block_start = 0

    # Bounded loop for line analysis
    for i in range(min(len(lines), MAX_LINES_ANALYZE)):
        line = lines[i]
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Detect logical blocks
        if any(keyword in line.lower() for keyword in ['try:', 'if ', 'for ', 'while ', 'def ', 'class ']):
            if current_block_size > 0 and block_count < MAX_BLOCKS:
                # Create block with actual size
                actual_block = current_block[:current_block_size]
                blocks[block_count] = (block_start, i-1, actual_block)
                block_count += 1
                current_block_size = 0  # Reset for new block
            if current_block_size < MAX_BLOCK_SIZE:
                current_block[current_block_size] = line
                current_block_size += 1
            block_start = i
        else:
            if current_block_size < MAX_BLOCK_SIZE:
                current_block[current_block_size] = line
                current_block_size += 1

    # Add final block if it exists
    if current_block_size > 0 and block_count < MAX_BLOCKS:
        actual_block = current_block[:current_block_size]
        blocks[block_count] = (block_start, len(lines)-1, actual_block)
        block_count += 1

    return blocks[:block_count]  # Return actual blocks


def analyze_blocks_for_refactoring(blocks: List[Tuple[int, int, List[str]]], func_name: str) -> Tuple[List[Dict], List[Dict]]:
    """Analyze blocks to identify refactoring opportunities (helper function)"""
    opportunities = [None] * MAX_OPPORTUNITIES
    suggested_functions = [None] * MAX_OPPORTUNITIES
    opportunity_count = 0
    function_count = 0

    # Bounded loop for block analysis
    for i in range(min(len(blocks), MAX_OPPORTUNITIES)):
        if blocks[i] is not None:
            start, end, block_lines = blocks[i]
            if len(block_lines) > 8 and opportunity_count < MAX_OPPORTUNITIES:
                block_content = '\n'.join(block_lines)
                suggested_name = suggest_function_name(block_content, func_name)

                opportunities[opportunity_count] = {
                    'lines': f"{start+1}-{end+1}",
                    'size': len(block_lines),
                    'content_preview': block_content[:100] + "..." if len(block_content) > 100 else block_content,
                    'suggested_function': suggested_name
                }
                opportunity_count += 1

                if function_count < MAX_OPPORTUNITIES:
                    suggested_functions[function_count] = {
                        'name': suggested_name,
                        'purpose': f"Extracted from {func_name}",
                        'lines': len(block_lines),
                        'content': block_content
                    }
                    function_count += 1

    return opportunities[:opportunity_count], suggested_functions[:function_count]


def suggest_function_name(content: str, parent_func: str) -> str:
    """Suggest a meaningful function name based on content (helper function)"""
    # Extract keywords from content
    keywords = []

    # Look for common patterns
    patterns = [
        (r'if\s+(\w+)', 'check'),
        (r'for\s+(\w+)', 'process'),
        (r'while\s+(\w+)', 'handle'),
        (r'try:', 'attempt'),
        (r'except', 'handle_error'),
        (r'validate', 'validate'),
        (r'process', 'process'),
        (r'calculate', 'calculate'),
        (r'generate', 'generate'),
        (r'parse', 'parse'),
        (r'convert', 'convert'),
        (r'format', 'format'),
        (r'analyze', 'analyze')
    ]

    for pattern, action in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            keywords.extend([action] + matches[:2])  # Limit to 2 additional keywords

    # Create function name from keywords
    if keywords:
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword.lower())

        # Create camelCase function name
        if len(unique_keywords) > 1:
            function_name = unique_keywords[0] + ''.join(word.capitalize() for word in unique_keywords[1:])
        else:
            function_name = unique_keywords[0]
    else:
        # Fallback name based on parent function
        function_name = f"extracted_from_{parent_func}"

    return function_name


def analyze_function_for_refactoring(func_content: str, func_name: str) -> Dict:
    """Analyze a function to identify refactoring opportunities (coordinator function)"""
    lines = func_content.split('\n')

    # Initialize analysis structure using helper
    analysis = initialize_refactoring_analysis(func_name, len(lines))

    # Extract logical blocks
    blocks = extract_logical_blocks(lines)

    # Analyze blocks for refactoring opportunities
    opportunities, suggested_functions = analyze_blocks_for_refactoring(blocks, func_name)

    # Update analysis structure
    analysis['refactoring_opportunities'] = opportunities
    analysis['suggested_functions'] = suggested_functions
    analysis['total_opportunities'] = len(opportunities)

    return analysis
