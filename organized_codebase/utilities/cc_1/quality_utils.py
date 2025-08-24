#!/usr/bin/env python3
"""
Quality Utilities Module
=======================

Utility functions for code quality analysis.
Contains helper functions for AST processing, metrics calculation, and analysis.
"""

import ast
import re
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class QualityUtils:
    """Utility functions for quality analysis"""

    @staticmethod
    def calculate_cyclomatic_complexity(tree: ast.Module) -> int:
        """Calculate cyclomatic complexity of AST"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    @staticmethod
    def calculate_nesting_depth(tree: ast.Module) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        MAX_NODES = 10000  # Safety bound for AST nodes
        nodes_list = list(ast.walk(tree))

        for i in range(min(len(nodes_list), MAX_NODES)):
            node = nodes_list[i]
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif hasattr(node, 'body') and hasattr(node, 'orelse'):
                # Handle if-else structures
                if_body_depth = QualityUtils._calculate_block_depth(node.body)
                else_depth = QualityUtils._calculate_block_depth(node.orelse)
                max_depth = max(max_depth, current_depth + if_body_depth, current_depth + else_depth)

        return max_depth

    @staticmethod
    def _calculate_block_depth(block: List[ast.stmt]) -> int:
        """Calculate depth of a block of statements"""
        if not block:
            return 0

        max_nested = 0
        # Bounded loop for block processing (Rule 2 compliance)
        MAX_BLOCK_SIZE = 1000  # Safety bound for block size
        for i in range(min(len(block), MAX_BLOCK_SIZE)):
            stmt = block[i]
            if isinstance(stmt, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                nested_depth = QualityUtils._calculate_block_depth(getattr(stmt, 'body', []))
                max_nested = max(max_nested, nested_depth + 1)

        return max_nested

    @staticmethod
    def count_functions_and_classes(tree: ast.Module) -> Tuple[int, int]:
        """Count functions and classes in AST"""
        functions = 0
        classes = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1

        return functions, classes

    @staticmethod
    def analyze_identifier_lengths(tree: ast.Module) -> Dict[str, List[int]]:
        """Analyze identifier lengths in the code"""
        lengths = {'functions': [], 'classes': [], 'variables': []}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lengths['functions'].append(len(node.name))
            elif isinstance(node, ast.ClassDef):
                lengths['classes'].append(len(node.name))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                lengths['variables'].append(len(node.id))

        return lengths

    @staticmethod
    def check_naming_conventions(content: str) -> Dict[str, int]:
        """Check naming convention compliance"""
        results = {
            'snake_case_functions': 0,
            'PascalCase_classes': 0,
            'UPPER_CASE_constants': 0,
            'violations': 0
        }

        # Function names should be snake_case
        func_pattern = r'def\s+([a-z_][a-z0-9_]*)\s*\('
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            results['snake_case_functions'] += 1

        # Class names should be PascalCase
        class_pattern = r'class\s+([A-Z][a-zA-Z0-9]*)\s*:'
        for match in re.finditer(class_pattern, content):
            results['PascalCase_classes'] += 1

        # Constants should be UPPER_CASE
        const_pattern = r'^([A-Z_][A-Z0-9_]*)\s*='
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            results['UPPER_CASE_constants'] += 1

        return results

    @staticmethod
    def calculate_readability_score(content: str) -> float:
        """Calculate a basic readability score"""
        lines = content.split('\n')
        total_lines = len(lines)

        if total_lines == 0:
            return 0.0

        # Count good patterns
        good_patterns = [
            r'# .*',  # Comments
            r'"""[\s\S]*?"""',  # Docstrings
            r"'''[\s\S]*?'''",  # Docstrings
            r'\n\s*\n',  # Proper spacing
        ]

        good_matches = 0
        # Bounded loop for pattern matching (Rule 2 compliance)
        MAX_PATTERNS = 10  # Safety bound for patterns
        for i in range(min(len(good_patterns), MAX_PATTERNS)):
            pattern = good_patterns[i]
            good_matches += len(re.findall(pattern, content))

        # Count bad patterns
        bad_patterns = [
            r'[A-Z]{3,}',  # ALL CAPS
            r'\w{30,}',  # Very long identifiers
            r'[^\w\s]{3,}',  # Multiple consecutive symbols
        ]

        bad_matches = 0
        for i in range(min(len(bad_patterns), MAX_PATTERNS)):
            pattern = bad_patterns[i]
            bad_matches += len(re.findall(pattern, content))

        # Calculate score
        good_score = min(good_matches / total_lines, 1.0)
        bad_penalty = min(bad_matches / total_lines, 0.5)

        return max(0.0, good_score - bad_penalty)

    @staticmethod
    def extract_code_metrics(content: str) -> Dict[str, Any]:
        """Extract basic code metrics"""
        lines = content.split('\n')

        metrics = {
            'total_lines': len(lines),
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'docstring_lines': 0
        }

        in_docstring = False

        # Bounded loop for line processing (Rule 2 compliance)
        MAX_LINES = 10000  # Safety bound for lines
        for i in range(min(len(lines), MAX_LINES)):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                metrics['blank_lines'] += 1
            elif stripped.startswith('#'):
                metrics['comment_lines'] += 1
            elif '"""' in line or "'''" in line:
                metrics['docstring_lines'] += 1
                in_docstring = not in_docstring
            elif in_docstring:
                metrics['docstring_lines'] += 1
            else:
                metrics['code_lines'] += 1

        return metrics
