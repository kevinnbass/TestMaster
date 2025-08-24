#!/usr/bin/env python3
"""
Maintainability Analyzer Module
==============================

Analyzes code maintainability including function length,
class complexity, coupling, and other maintainability metrics.
"""

import ast
from typing import Dict, List, Any
from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.sandbox.quality_config import QualityMetric, QualityConfig
from .quality_utils import QualityUtils


class MaintainabilityAnalyzer:
    """Analyzes code maintainability metrics"""

    def __init__(self):
        """Initialize maintainability analyzer"""
        assert QualityConfig is not None, "QualityConfig must be available"
        self.config = QualityConfig()
        assert self.config is not None, "Config initialization failed"

    def analyze_maintainability(self, tree: ast.Module, content: str) -> List[QualityMetric]:
        """Analyze maintainability metrics"""
        # Pre-allocate metrics with known capacity (Rule 3 compliance)
        MAX_METRICS = 10  # Expected number of metrics
        metrics = [None] * MAX_METRICS
        metrics_count = 0

        # Function length analysis
        function_length_score = self._analyze_function_lengths(tree)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Function Length",
                score=function_length_score,
                category="maintainability",
                description=self._get_function_length_description(tree),
                recommendations=self._get_function_length_recommendations(tree)
            )
            metrics_count += 1

        # Class complexity analysis
        class_complexity_score = self._analyze_class_complexity(tree)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Class Complexity",
                score=class_complexity_score,
                category="maintainability",
                description=self._get_class_complexity_description(tree),
                recommendations=self._get_class_complexity_recommendations(tree)
            )
            metrics_count += 1

        # Coupling analysis
        coupling_score = self._analyze_coupling(content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Code Coupling",
                score=coupling_score,
                category="maintainability",
                description=self._get_coupling_description(content),
                recommendations=self._get_coupling_recommendations(content)
            )
            metrics_count += 1

        # Testability analysis
        testability_score = self._analyze_testability(tree, content)
        if metrics_count < MAX_METRICS:
            metrics[metrics_count] = QualityMetric(
                name="Testability",
                score=testability_score,
                category="maintainability",
                description=self._get_testability_description(tree, content),
                recommendations=self._get_testability_recommendations(tree, content)
            )
            metrics_count += 1

        return metrics[:metrics_count]

    def _analyze_function_lengths(self, tree: ast.Module) -> float:
        """Analyze function lengths"""
        # Pre-allocate function_lengths with known capacity (Rule 3 compliance)
        MAX_FUNCTIONS = 200  # Safety bound for functions
        function_lengths = [None] * MAX_FUNCTIONS
        func_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        MAX_NODES = 5000  # Safety bound for AST nodes
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

    def _analyze_class_complexity(self, tree: ast.Module) -> float:
        """Analyze class complexity"""
        # Pre-allocate class_sizes with known capacity (Rule 3 compliance)
        MAX_CLASSES = 100  # Safety bound for classes
        class_sizes = [None] * MAX_CLASSES
        class_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        MAX_NODES = 5000  # Safety bound for AST nodes
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

    def _analyze_coupling(self, content: str) -> float:
        """Analyze code coupling"""
        import re

        # Count global variable usage (potential coupling)
        global_vars = len(re.findall(r'\bglobal\s+\w+', content))

        # Count direct function calls (potential coupling)
        function_calls = len(re.findall(r'\w+\([^)]*\)', content))

        # Simple coupling score - lower coupling is better
        coupling_indicators = global_vars + (function_calls / 100)  # Scale function calls
        coupling_score = max(0.0, 1.0 - (coupling_indicators / 10))

        return coupling_score

    def _analyze_testability(self, tree: ast.Module, content: str) -> float:
        """Analyze code testability"""
        testability_factors = 0
        total_factors = 4

        # 1. Small functions are easier to test
        # Pre-allocate function_lengths with known capacity (Rule 3 compliance)
        MAX_FUNCTIONS = 100  # Safety bound for functions
        function_lengths = [None] * MAX_FUNCTIONS
        func_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and func_count < MAX_FUNCTIONS:
                function_lengths[func_count] = len(node.body) if hasattr(node, 'body') else 0
                func_count += 1

        if func_count > 0:
            avg_length = sum(function_lengths[:func_count]) / func_count
            if avg_length < 20:
                testability_factors += 1

        # 2. Functions with few parameters are easier to test
        # Pre-allocate param_counts with known capacity (Rule 3 compliance)
        param_counts = [None] * MAX_FUNCTIONS
        param_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and param_count < MAX_FUNCTIONS:
                param_counts[param_count] = len(node.args.args)
                param_count += 1

        if param_count > 0:
            avg_params = sum(param_counts[:param_count]) / param_count
            if avg_params < 4:
                testability_factors += 1

        # 3. Pure functions are easier to test
        # This is hard to detect statically, so we'll give a basic score
        testability_factors += 1

        # 4. Clear function names
        # Replace complex comprehension with simple loop (Rule 1 compliance)
        clear_names = 0
        total_functions = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if len(node.name) > 3:
                    clear_names += 1
        # Count total functions using simple loop (Rule 1 compliance)
        # Note: total_functions is already counted in the loop above

        if total_functions > 0 and clear_names / total_functions > 0.8:
            testability_factors += 1

        return testability_factors / total_factors

    def _get_function_length_description(self, tree: ast.Module) -> str:
        """Get function length description"""
        # Pre-allocate function_lengths with known capacity (Rule 3 compliance)
        MAX_FUNCTIONS = 200  # Safety bound for functions
        function_lengths = [None] * MAX_FUNCTIONS
        func_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        MAX_NODES = 5000  # Safety bound for AST nodes
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

    def _get_class_complexity_description(self, tree: ast.Module) -> str:
        """Get class complexity description"""
        # Pre-allocate class_sizes with known capacity (Rule 3 compliance)
        MAX_CLASSES = 100  # Safety bound for classes
        class_sizes = [None] * MAX_CLASSES
        class_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        MAX_NODES = 5000  # Safety bound for AST nodes
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

    def _get_coupling_description(self, content: str) -> str:
        """Get coupling description"""
        import re

        global_vars = len(re.findall(r'\bglobal\s+\w+', content))
        function_calls = len(re.findall(r'\w+\([^)]*\)', content))

        return f"Global variables: {global_vars}, Function calls: {function_calls}"

    def _get_testability_description(self, tree: ast.Module, content: str) -> str:
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

    def _get_function_length_recommendations(self, tree: ast.Module) -> List[str]:
        """Get function length recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 10  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        thresholds = self.config.COMPLEXITY_THRESHOLDS['function_length']

        # Pre-allocate long_functions with known capacity (Rule 3 compliance)
        MAX_LONG_FUNCTIONS = 50  # Safety bound for long functions
        long_functions = [None] * MAX_LONG_FUNCTIONS
        long_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        MAX_NODES = 5000  # Safety bound for AST nodes
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

    def _get_class_complexity_recommendations(self, tree: ast.Module) -> List[str]:
        """Get class complexity recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 10  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        thresholds = self.config.COMPLEXITY_THRESHOLDS['class_length']

        # Pre-allocate large_classes with known capacity (Rule 3 compliance)
        MAX_LARGE_CLASSES = 20  # Safety bound for large classes
        large_classes = [None] * MAX_LARGE_CLASSES
        large_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        MAX_NODES = 5000  # Safety bound for AST nodes
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

    def _get_coupling_recommendations(self, content: str) -> List[str]:
        """Get coupling recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 5  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        import re

        global_vars = len(re.findall(r'\bglobal\s+\w+', content))

        if global_vars > 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Reduce {global_vars} global variables - pass parameters instead"
            rec_count += 1

        return recommendations[:rec_count]

    def _get_testability_recommendations(self, tree: ast.Module, content: str) -> List[str]:
        """Get testability recommendations"""
        # Pre-allocate recommendations with known capacity (Rule 3 compliance)
        MAX_RECOMMENDATIONS = 10  # Expected number of recommendations
        recommendations = [None] * MAX_RECOMMENDATIONS
        rec_count = 0

        # Check for functions with many parameters
        # Pre-allocate high_param_functions with known capacity (Rule 3 compliance)
        MAX_HIGH_PARAM_FUNCTIONS = 20  # Safety bound for high parameter functions
        high_param_functions = [None] * MAX_HIGH_PARAM_FUNCTIONS
        high_param_count = 0

        # Bounded loop for AST traversal (Rule 2 compliance)
        MAX_NODES = 5000  # Safety bound for AST nodes
        nodes_list = list(ast.walk(tree))
        for i in range(min(len(nodes_list), MAX_NODES)):
            node = nodes_list[i]
            if isinstance(node, ast.FunctionDef) and high_param_count < MAX_HIGH_PARAM_FUNCTIONS:
                param_count = len(node.args.args)
                if param_count > 6:
                    high_param_functions[high_param_count] = (node.name, param_count)
                    high_param_count += 1

        if high_param_count > 0 and rec_count < MAX_RECOMMENDATIONS:
            recommendations[rec_count] = f"Reduce parameters in {high_param_count} functions"
            rec_count += 1

            # Find worst function using bounded approach
            worst_params = 0
            worst_name = ""
            for i in range(high_param_count):
                if high_param_functions[i] and high_param_functions[i][1] > worst_params:
                    worst_params = high_param_functions[i][1]
                    worst_name = high_param_functions[i][0]

            if rec_count < MAX_RECOMMENDATIONS:
                recommendations[rec_count] = f"Function '{worst_name}' has {worst_params} parameters - consider using a config object"
                rec_count += 1

        return recommendations[:rec_count]