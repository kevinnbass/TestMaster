#!/usr/bin/env python3
"""
Compliance Audit Rule Checking
==============================

Rule checking methods for compliance auditing.
"""

import ast
from typing import List, Final

# Constants
MAX_FUNCTION_SIZE: Final[int] = 60  # NASA-STD-8719.13 Rule 4


class ComplianceRuleChecker:
    """Handles individual rule checking operations"""

    def has_recursion(self, node: ast.FunctionDef, function_name: str) -> bool:
        """Check if function has recursion"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == function_name:
                    return True
        return False

    def has_complex_comprehensions(self, node: ast.AST) -> bool:
        """Check for complex comprehensions"""
        for child in ast.walk(node):
            if isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp)):
                # Check if comprehension is nested or has complex conditions
                if (hasattr(child, 'generators') and
                    len(child.generators) > 1 or
                    any(hasattr(gen, 'ifs') and gen.ifs for gen in child.generators)):
                    return True
        return False

    def has_unbounded_loops(self, node: ast.AST) -> bool:
        """Check for loops without fixed upper bounds"""
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                # Check if it's iterating over a range with fixed bounds
                if isinstance(child.iter, ast.Call):
                    if isinstance(child.iter.func, ast.Name) and child.iter.func.id == 'range':
                        # Check if range has fixed arguments
                        if len(child.iter.args) >= 1:
                            first_arg = child.iter.args[0]
                            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, int):
                                continue  # Fixed bound
                # Check for iteration over collections that might grow
                if isinstance(child.iter, ast.Name):
                    # This could be unbounded - flag for review
                    return True
        return False

    def has_dynamic_resizing(self, node: ast.AST) -> bool:
        """Check for dynamic object resizing"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr in ['append', 'extend', 'insert', 'pop', 'remove', 'clear']:
                        if isinstance(child.func.value, ast.Name):
                            # This could be dynamic resizing - flag for review
                            return True
        return False

    def has_parameter_validation(self, node: ast.FunctionDef) -> bool:
        """Check if function has parameter validation"""
        # Check for type hints
        if node.returns or any(arg.annotation for arg in node.args.args):
            return True

        # Check for assert statements that might validate parameters
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                # Simple check - assume any assert might be parameter validation
                return True

        return False

    def has_complex_decorators(self, node: ast.FunctionDef) -> bool:
        """Check for complex decorators"""
        if len(node.decorator_list) > 2:  # More than 2 decorators
            return True

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and decorator.args:
                # Decorator with arguments - could be complex
                return True
        return False

    def has_docstring(self, node: ast.FunctionDef) -> bool:
        """Check if function has docstring"""
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                return True
            elif isinstance(node.body[0].value, ast.Str):  # Python < 3.8
                return True
        return False

    def exceeds_function_size_limit(self, node: ast.FunctionDef) -> bool:
        """Check if function exceeds size limit"""
        return len(node.body) > MAX_FUNCTION_SIZE

    def has_function_size_limit_compliance(self, node: ast.FunctionDef) -> bool:
        """Check function size compliance"""
        return not self.exceeds_function_size_limit(node)

