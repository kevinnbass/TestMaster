#!/usr/bin/env python3
"""
Maintainability Descriptions Module
=================================

Description methods for maintainability metrics.
"""

import ast
import re
from typing import Final

# Constants
MAX_FUNCTIONS: Final[int] = 200  # Safety bound for functions
MAX_CLASSES: Final[int] = 100  # Safety bound for classes
MAX_NODES: Final[int] = 5000  # Safety bound for AST nodes


class MaintainabilityDescriptions:
    """Description methods for maintainability metrics"""

    def __init__(self, config):
        """Initialize with quality configuration"""
        self.config = config

    def get_function_length_description(self, tree: ast.Module) -> str:
        """Get function length description"""
        # Pre-allocate function_lengths with known capacity (Rule 3 compliance)
        function_lengths = [None] * MAX_FUNCTIONS
        func_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        nodes_list = list(ast.walk(tree))
        for i in range(min(len(nodes_list), MAX_NODES)):
            node = nodes_list[i]
            if isinstance(node, ast.FunctionDef) and func_count < MAX_FUNCTIONS:
                function_lengths[func_count] = len(node.body) if hasattr(node, 'body') else 0
                func_count += 1

        if func_count == 0:
            return "No functions found"

        avg_length = sum(function_lengths[:func_count]) / func_count
        max_length = max(function_lengths[:func_count])
        long_functions = sum(1 for i in range(func_count) if function_lengths[i] and function_lengths[i] > 30)

        return ".1f"

    def get_class_complexity_description(self, tree: ast.Module) -> str:
        """Get class complexity description"""
        # Pre-allocate class_sizes with known capacity (Rule 3 compliance)
        class_sizes = [None] * MAX_CLASSES
        class_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        nodes_list = list(ast.walk(tree))
        for i in range(min(len(nodes_list), MAX_NODES)):
            node = nodes_list[i]
            if isinstance(node, ast.ClassDef) and class_count < MAX_CLASSES:
                class_sizes[class_count] = len(node.body) if hasattr(node, 'body') else 0
                class_count += 1

        if class_count == 0:
            return "No classes found"

        avg_size = sum(class_sizes[:class_count]) / class_count
        max_size = max(class_sizes[:class_count])
        large_classes = sum(1 for i in range(class_count) if class_sizes[i] and class_sizes[i] > 200)

        return ".1f"

    def get_coupling_description(self, content: str) -> str:
        """Get coupling description"""
        global_vars = len(re.findall(r'\bglobal\s+\w+', content))
        function_calls = len(re.findall(r'\w+\([^)]*\)', content))

        return f"Global variables: {global_vars}, Function calls: {function_calls}"

    def get_testability_description(self, tree: ast.Module, content: str) -> str:
        """Get testability description"""
        # Replace complex comprehensions with simple loops (Rule 1 compliance)
        function_count = 0
        small_functions = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                if len(node.body) < 20:
                    small_functions += 1

        if function_count == 0:
            return "No functions to test"

        testability_ratio = small_functions / function_count
        return ".1%"

