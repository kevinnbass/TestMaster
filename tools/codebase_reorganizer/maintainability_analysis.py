#!/usr/bin/env python3
"""
Maintainability Analysis Module
==============================

Core analysis methods for maintainability metrics.
"""

import ast
import re
from typing import Dict, Any, Final

# Constants
MAX_FUNCTIONS: Final[int] = 200  # Safety bound for functions
MAX_CLASSES: Final[int] = 100  # Safety bound for classes
MAX_NODES: Final[int] = 5000  # Safety bound for AST nodes


class MaintainabilityAnalysis:
    """Core analysis methods for maintainability metrics"""

    def __init__(self, config):
        """Initialize with quality configuration"""
        self.config = config

    def analyze_function_lengths(self, tree: ast.Module) -> float:
        """Analyze function lengths"""
        # Pre-allocate function_lengths with known capacity (Rule 3 compliance)
        function_lengths = [None] * MAX_FUNCTIONS
        func_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        nodes_list = list(ast.walk(tree))
        for i in range(min(len(nodes_list), MAX_NODES)):
            node = nodes_list[i]
            if isinstance(node, ast.FunctionDef) and func_count < MAX_FUNCTIONS:
                length = len(node.body) if hasattr(node, 'body') else 0
                function_lengths[func_count] = length
                func_count += 1

        if func_count == 0:
            return 1.0

        thresholds = self.config.COMPLEXITY_THRESHOLDS['function_length']
        long_functions = sum(1 for i in range(func_count) if function_lengths[i] and function_lengths[i] > thresholds['warning'])

        if long_functions == 0:
            return 1.0
        elif long_functions <= func_count * 0.2:
            return 0.7
        else:
            return 0.4

    def analyze_class_complexity(self, tree: ast.Module) -> float:
        """Analyze class complexity"""
        # Pre-allocate class_sizes with known capacity (Rule 3 compliance)
        class_sizes = [None] * MAX_CLASSES
        class_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        nodes_list = list(ast.walk(tree))
        for i in range(min(len(nodes_list), MAX_NODES)):
            node = nodes_list[i]
            if isinstance(node, ast.ClassDef) and class_count < MAX_CLASSES:
                size = len(node.body) if hasattr(node, 'body') else 0
                class_sizes[class_count] = size
                class_count += 1

        if class_count == 0:
            return 1.0

        thresholds = self.config.COMPLEXITY_THRESHOLDS['class_length']
        large_classes = sum(1 for i in range(class_count) if class_sizes[i] and class_sizes[i] > thresholds['warning'])

        if large_classes == 0:
            return 1.0
        elif large_classes <= class_count * 0.2:
            return 0.7
        else:
            return 0.4

    def analyze_coupling(self, content: str) -> float:
        """Analyze coupling metrics"""
        # Count import statements
        import_count = len(re.findall(r'^\s*(import|from)', content, re.MULTILINE))

        # Analyze external dependencies
        external_patterns = ['requests', 'urllib', 'http', 'api', 'network']
        external_refs = sum(1 for pattern in external_patterns if pattern in content.lower())

        # Calculate coupling score
        if import_count <= 5 and external_refs <= 2:
            return 1.0
        elif import_count <= 10 and external_refs <= 5:
            return 0.7
        else:
            return 0.4

    def analyze_testability(self, tree: ast.Module, content: str) -> float:
        """Analyze testability metrics"""
        # Count functions and classes
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])

        # Check for side effects (global variables, file operations)
        global_vars = len([n for n in ast.walk(tree) if isinstance(n, ast.Global)])
        file_ops = len(re.findall(r'\b(open|write|read)\b', content))

        # Calculate testability score
        side_effects = global_vars + file_ops

        if side_effects == 0 and function_count <= 20:
            return 1.0
        elif side_effects <= 3 and function_count <= 30:
            return 0.7
        else:
            return 0.4

    def get_function_count(self, tree: ast.Module) -> int:
        """Get total function count"""
        return len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])

    def get_class_count(self, tree: ast.Module) -> int:
        """Get total class count"""
        return len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])

    def get_long_function_count(self, tree: ast.Module) -> int:
        """Get count of functions exceeding length threshold"""
        if not hasattr(self.config, 'COMPLEXITY_THRESHOLDS') or 'function_length' not in self.config.COMPLEXITY_THRESHOLDS:
            return 0

        threshold = self.config.COMPLEXITY_THRESHOLDS['function_length']['warning']
        return len([n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and len(n.body) > threshold])

    def get_large_class_count(self, tree: ast.Module) -> int:
        """Get count of classes exceeding size threshold"""
        if not hasattr(self.config, 'COMPLEXITY_THRESHOLDS') or 'class_length' not in self.config.COMPLEXITY_THRESHOLDS:
            return 0

        threshold = self.config.COMPLEXITY_THRESHOLDS['class_length']['warning']
        return len([n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and len(n.body) > threshold])

