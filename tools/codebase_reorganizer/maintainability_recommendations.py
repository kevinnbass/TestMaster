#!/usr/bin/env python3
"""
Maintainability Recommendations Module
====================================

Recommendation methods for maintainability improvements.
"""

import ast
import re
from typing import List, Final

# Constants
MAX_RECOMMENDATIONS: Final[int] = 10  # Expected number of recommendations
MAX_LONG_FUNCTIONS: Final[int] = 50  # Safety bound for long functions
MAX_LARGE_CLASSES: Final[int] = 20  # Safety bound for large classes
MAX_NODES: Final[int] = 5000  # Safety bound for AST nodes


class MaintainabilityRecommendations:
    """Recommendation methods for maintainability improvements"""

    def __init__(self, config):
        """Initialize with quality configuration"""
        self.config = config

    def get_function_length_recommendations(self, tree: ast.Module) -> List[str]:
        """Get function length recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        thresholds = self.config.COMPLEXITY_THRESHOLDS['function_length']

        # Pre-allocate long_functions with known capacity (Rule 3 compliance)
        long_functions = [None] * MAX_LONG_FUNCTIONS
        long_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        nodes_list = list(ast.walk(tree))
        for i in range(min(len(nodes_list), MAX_NODES)):
            node = nodes_list[i]
            if isinstance(node, ast.FunctionDef) and long_count < MAX_LONG_FUNCTIONS:
                length = len(node.body) if hasattr(node, 'body') else 0
                if length > thresholds['warning']:
                    long_functions[long_count] = (node.name, length)
                    long_count += 1

        if long_count > 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Break down {long_count} long functions into smaller functions"
            rec_count += 1

            # Find worst function using bounded approach
            worst_length = 0
            worst_name = ""
            for i in range(long_count):
                if long_functions[i] and long_functions[i][1] > worst_length:
                    worst_length = long_functions[i][1]
                    worst_name = long_functions[i][0]

            if rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = f"Function '{worst_name}' ({worst_length} lines) should be refactored first"
                rec_count += 1

        return recommendations[:rec_count]

    def get_class_complexity_recommendations(self, tree: ast.Module) -> List[str]:
        """Get class complexity recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        thresholds = self.config.COMPLEXITY_THRESHOLDS['class_length']

        # Pre-allocate large_classes with known capacity (Rule 3 compliance)
        large_classes = [None] * MAX_LARGE_CLASSES
        large_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        nodes_list = list(ast.walk(tree))
        for i in range(min(len(nodes_list), MAX_NODES)):
            node = nodes_list[i]
            if isinstance(node, ast.ClassDef) and large_count < MAX_LARGE_CLASSES:
                size = len(node.body) if hasattr(node, 'body') else 0
                if size > thresholds['warning']:
                    large_classes[large_count] = (node.name, size)
                    large_count += 1

        if large_count > 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Split {large_count} large classes into smaller classes"
            rec_count += 1

            # Find worst class using bounded approach
            worst_size = 0
            worst_name = ""
            for i in range(large_count):
                if large_classes[i] and large_classes[i][1] > worst_size:
                    worst_size = large_classes[i][1]
                    worst_name = large_classes[i][0]

            if rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = f"Class '{worst_name}' ({worst_size} lines) should be refactored first"
                rec_count += 1

        return recommendations[:rec_count]

    def get_coupling_recommendations(self, content: str) -> List[str]:
        """Get coupling recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        global_vars = len(re.findall(r'\bglobal\s+\w+', content))

        if global_vars > 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Reduce {global_vars} global variables - pass parameters instead"
            rec_count += 1

        return recommendations[:rec_count]

    def get_testability_recommendations(self, tree: ast.Module, content: str) -> List[str]:
        """Get testability recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        # Check for side effects
        global_vars = len([n for n in ast.walk(tree) if isinstance(n, ast.Global)])
        file_ops = len(re.findall(r'\b(open|write|read)\b', content))

        if global_vars > 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Remove {global_vars} global variables to improve testability"
            rec_count += 1

        if file_ops > 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Abstract {file_ops} file operations to improve testability"
            rec_count += 1

        return recommendations[:rec_count]

    def get_general_recommendations(self, tree: ast.Module, content: str) -> List[str]:
        """Get general maintainability recommendations"""
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        # Check function count
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        if function_count > 30 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Consider splitting {function_count} functions across multiple modules"
            rec_count += 1

        # Check import count
        import_count = len(re.findall(r'^\s*(import|from)', content, re.MULTILINE))
        if import_count > 15 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Reduce {import_count} imports - consider consolidating dependencies"
            rec_count += 1

        return recommendations[:rec_count]
