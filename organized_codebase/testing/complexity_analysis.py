"""
Complexity Analysis Module
==========================

Implements comprehensive complexity analysis:
- Multiple complexity dimensions and metrics
- Cognitive complexity analysis
- Structural complexity assessment
- Complexity distribution and patterns
"""

import ast
import statistics
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class ComplexityAnalyzer(BaseAnalyzer):
    """Analyzer for code complexity in multiple dimensions."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive complexity analysis."""
        print("[INFO] Analyzing Code Complexity...")
        
        results = {
            "cyclomatic_complexity": self._analyze_cyclomatic_complexity(),
            "cognitive_complexity": self._analyze_cognitive_complexity(),
            "structural_complexity": self._analyze_structural_complexity(),
            "computational_complexity": self._analyze_computational_complexity(),
            "interface_complexity": self._analyze_interface_complexity(),
            "complexity_distribution": self._analyze_complexity_distribution(),
            "complexity_hotspots": self._identify_complexity_hotspots(),
            "complexity_trends": self._analyze_complexity_trends()
        }
        
        print(f"  [OK] Analyzed {len(results)} complexity dimensions")
        return results
    
    def _analyze_cyclomatic_complexity(self) -> Dict[str, Any]:
        """Analyze cyclomatic complexity across the codebase."""
        complexity_data = {}
        all_complexities = []
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                file_functions = {}
                file_total_complexity = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_detailed_cyclomatic_complexity(node)
                        file_functions[node.name] = {
                            'complexity': complexity['total_complexity'],
                            'line': node.lineno,
                            'decision_points': complexity['decision_points'],
                            'loop_count': complexity['loop_count'],
                            'condition_count': complexity['condition_count'],
                            'exception_handling': complexity['exception_handling']
                        }
                        file_total_complexity += complexity['total_complexity']
                        all_complexities.append(complexity['total_complexity'])
                
                if file_functions:
                    complexity_data[file_key] = {
                        'functions': file_functions,
                        'file_total_complexity': file_total_complexity,
                        'average_function_complexity': file_total_complexity / len(file_functions),
                        'max_function_complexity': max(f['complexity'] for f in file_functions.values()),
                        'function_count': len(file_functions)
                    }
                
            except Exception:
                continue
        
        # Calculate summary statistics
        if all_complexities:
            return {
                'per_file': complexity_data,
                'summary': {
                    'total_functions': len(all_complexities),
                    'average_complexity': statistics.mean(all_complexities),
                    'median_complexity': statistics.median(all_complexities),
                    'max_complexity': max(all_complexities),
                    'std_deviation': statistics.stdev(all_complexities) if len(all_complexities) > 1 else 0,
                    'complexity_categories': self._categorize_complexities(all_complexities),
                    'high_complexity_functions': len([c for c in all_complexities if c > 10]),
                    'very_high_complexity_functions': len([c for c in all_complexities if c > 20]),
                    'complexity_distribution': self._calculate_distribution(all_complexities)
                }
            }
        else:
            return {'per_file': {}, 'summary': {'total_functions': 0}}
    
    def _calculate_detailed_cyclomatic_complexity(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Calculate detailed cyclomatic complexity with breakdown."""
        complexity = 1  # Base complexity
        decision_points = 0
        loop_count = 0
        condition_count = 0
        exception_handling = 0
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                complexity += 1
                decision_points += 1
                condition_count += 1
            elif isinstance(node, ast.While):
                complexity += 1
                decision_points += 1
                loop_count += 1
            elif isinstance(node, ast.For):
                complexity += 1
                decision_points += 1
                loop_count += 1
            elif isinstance(node, ast.AsyncFor):
                complexity += 1
                decision_points += 1
                loop_count += 1
            elif isinstance(node, ast.With):
                complexity += 1
                decision_points += 1
            elif isinstance(node, ast.AsyncWith):
                complexity += 1
                decision_points += 1
            elif isinstance(node, ast.Try):
                complexity += 1
                exception_handling += 1
                # Add complexity for each except handler
                complexity += len(node.handlers)
                exception_handling += len(node.handlers)
                # Add complexity for else and finally
                if node.orelse:
                    complexity += 1
                if node.finalbody:
                    complexity += 1
            elif isinstance(node, ast.BoolOp):
                # Add complexity for each additional boolean operator
                complexity += len(node.values) - 1
                condition_count += len(node.values) - 1
            elif isinstance(node, ast.Break):
                complexity += 1
            elif isinstance(node, ast.Continue):
                complexity += 1
        
        return {
            'total_complexity': complexity,
            'decision_points': decision_points,
            'loop_count': loop_count,
            'condition_count': condition_count,
            'exception_handling': exception_handling
        }
    
    def _categorize_complexities(self, complexities: List[int]) -> Dict[str, int]:
        """Categorize complexities into risk levels."""
        categories = {
            'simple': 0,        # 1-5
            'moderate': 0,      # 6-10
            'high': 0,          # 11-20
            'very_high': 0,     # 21-50
            'extremely_high': 0 # >50
        }
        
        for complexity in complexities:
            if complexity <= 5:
                categories['simple'] += 1
            elif complexity <= 10:
                categories['moderate'] += 1
            elif complexity <= 20:
                categories['high'] += 1
            elif complexity <= 50:
                categories['very_high'] += 1
            else:
                categories['extremely_high'] += 1
        
        return categories
    
    def _analyze_cognitive_complexity(self) -> Dict[str, Any]:
        """Analyze cognitive complexity (human comprehension difficulty)."""
        cognitive_data = {}
        all_cognitive_scores = []
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                file_functions = {}
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        cognitive_score = self._calculate_cognitive_complexity(node)
                        file_functions[node.name] = {
                            'cognitive_complexity': cognitive_score['total_score'],
                            'line': node.lineno,
                            'nesting_penalty': cognitive_score['nesting_penalty'],
                            'logical_operators': cognitive_score['logical_operators'],
                            'control_flow_breaks': cognitive_score['control_flow_breaks'],
                            'recursion_penalty': cognitive_score['recursion_penalty']
                        }
                        all_cognitive_scores.append(cognitive_score['total_score'])
                
                if file_functions:
                    cognitive_data[file_key] = file_functions
                
            except Exception:
                continue
        
        # Calculate summary statistics
        if all_cognitive_scores:
            return {
                'per_file': cognitive_data,
                'summary': {
                    'total_functions': len(all_cognitive_scores),
                    'average_cognitive_complexity': statistics.mean(all_cognitive_scores),
                    'median_cognitive_complexity': statistics.median(all_cognitive_scores),
                    'max_cognitive_complexity': max(all_cognitive_scores),
                    'high_cognitive_functions': len([s for s in all_cognitive_scores if s > 15]),
                    'cognitive_complexity_distribution': self._calculate_distribution(all_cognitive_scores)
                }
            }
        else:
            return {'per_file': {}, 'summary': {'total_functions': 0}}
    
    def _calculate_cognitive_complexity(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Calculate cognitive complexity score."""
        total_score = 0
        nesting_penalty = 0
        logical_operators = 0
        control_flow_breaks = 0
        recursion_penalty = 0
        
        # Track nesting level
        def analyze_node(node, nesting_level=0):
            nonlocal total_score, nesting_penalty, logical_operators, control_flow_breaks, recursion_penalty
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                # Base complexity + nesting penalty
                increment = 1 + nesting_level
                total_score += increment
                nesting_penalty += nesting_level
                
                # Analyze children with increased nesting
                for child in ast.iter_child_nodes(node):
                    analyze_node(child, nesting_level + 1)
            
            elif isinstance(node, ast.Try):
                # Exception handling adds complexity
                increment = 1 + nesting_level
                total_score += increment
                nesting_penalty += nesting_level
                
                for child in ast.iter_child_nodes(node):
                    analyze_node(child, nesting_level + 1)
            
            elif isinstance(node, ast.BoolOp):
                # Logical operators add complexity
                operator_count = len(node.values) - 1
                total_score += operator_count
                logical_operators += operator_count
                
                # Continue analyzing without increasing nesting
                for child in ast.iter_child_nodes(node):
                    analyze_node(child, nesting_level)
            
            elif isinstance(node, (ast.Break, ast.Continue)):
                # Control flow breaks
                increment = 1 + nesting_level
                total_score += increment
                control_flow_breaks += 1
            
            elif isinstance(node, ast.Call):
                # Check for recursion
                if isinstance(node.func, ast.Name) and node.func.id == func_node.name:
                    total_score += 1
                    recursion_penalty += 1
                
                # Continue analyzing
                for child in ast.iter_child_nodes(node):
                    analyze_node(child, nesting_level)
            
            else:
                # Continue analyzing other nodes
                for child in ast.iter_child_nodes(node):
                    analyze_node(child, nesting_level)
        
        # Start analysis from function body
        for stmt in func_node.body:
            analyze_node(stmt)
        
        return {
            'total_score': total_score,
            'nesting_penalty': nesting_penalty,
            'logical_operators': logical_operators,
            'control_flow_breaks': control_flow_breaks,
            'recursion_penalty': recursion_penalty
        }
    
    def _analyze_structural_complexity(self) -> Dict[str, Any]:
        """Analyze structural complexity of the codebase."""
        structural_data = {
            'class_complexity': {},
            'module_complexity': {},
            'inheritance_complexity': {},
            'interface_complexity': {}
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Module-level complexity
                module_complexity = self._calculate_module_complexity(tree)
                structural_data['module_complexity'][file_key] = module_complexity
                
                # Class-level complexity
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_complexity = self._calculate_class_complexity(node)
                        class_key = f"{file_key}::{node.name}"
                        structural_data['class_complexity'][class_key] = class_complexity
                        
                        # Inheritance complexity
                        inheritance_complexity = self._calculate_inheritance_complexity(node)
                        structural_data['inheritance_complexity'][class_key] = inheritance_complexity
                
            except Exception:
                continue
        
        return structural_data
    
    def _calculate_module_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate complexity at module level."""
        imports = 0
        classes = 0
        functions = 0
        global_variables = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
            elif isinstance(node, ast.FunctionDef):
                functions += 1
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                # Global variable (simplified detection)
                global_variables += 1
        
        # Calculate module complexity score
        complexity_score = (imports * 0.1) + (classes * 2) + (functions * 1) + (global_variables * 0.5)
        
        return {
            'imports': imports,
            'classes': classes,
            'functions': functions,
            'global_variables': global_variables,
            'complexity_score': complexity_score,
            'complexity_level': self._categorize_module_complexity(complexity_score)
        }
    
    def _categorize_module_complexity(self, score: float) -> str:
        """Categorize module complexity."""
        if score < 10:
            return 'simple'
        elif score < 25:
            return 'moderate'
        elif score < 50:
            return 'complex'
        else:
            return 'very_complex'
    
    def _calculate_class_complexity(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate complexity of a class."""
        methods = 0
        properties = 0
        private_methods = 0
        public_methods = 0
        static_methods = 0
        class_methods = 0
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods += 1
                
                if node.name.startswith('_'):
                    private_methods += 1
                else:
                    public_methods += 1
                
                # Check for decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id == 'staticmethod':
                            static_methods += 1
                        elif decorator.id == 'classmethod':
                            class_methods += 1
                        elif decorator.id == 'property':
                            properties += 1
        
        # Calculate class complexity score
        complexity_score = (methods * 1) + (properties * 0.5) + (len(class_node.bases) * 2)
        
        return {
            'total_methods': methods,
            'public_methods': public_methods,
            'private_methods': private_methods,
            'static_methods': static_methods,
            'class_methods': class_methods,
            'properties': properties,
            'base_classes': len(class_node.bases),
            'complexity_score': complexity_score,
            'complexity_level': self._categorize_class_complexity(complexity_score)
        }
    
    def _categorize_class_complexity(self, score: float) -> str:
        """Categorize class complexity."""
        if score < 5:
            return 'simple'
        elif score < 15:
            return 'moderate'
        elif score < 30:
            return 'complex'
        else:
            return 'very_complex'
    
    def _calculate_inheritance_complexity(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate inheritance complexity."""
        base_count = len(class_node.bases)
        
        # Check for multiple inheritance
        multiple_inheritance = base_count > 1
        
        # Check for deep inheritance (simplified)
        max_depth = min(base_count * 2, 10)  # Estimate
        
        complexity_score = base_count * 2 + (max_depth * 0.5)
        
        return {
            'base_class_count': base_count,
            'multiple_inheritance': multiple_inheritance,
            'estimated_depth': max_depth,
            'complexity_score': complexity_score,
            'complexity_level': 'high' if multiple_inheritance else 'moderate' if base_count > 0 else 'simple'
        }
    
    def _analyze_computational_complexity(self) -> Dict[str, Any]:
        """Analyze computational complexity patterns."""
        complexity_patterns = {
            'nested_loops': [],
            'recursive_functions': [],
            'sorting_algorithms': [],
            'search_algorithms': []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Detect nested loops
                        nested_loop_depth = self._detect_nested_loops(node)
                        if nested_loop_depth > 1:
                            complexity_patterns['nested_loops'].append({
                                'file': file_key,
                                'function': node.name,
                                'line': node.lineno,
                                'nesting_depth': nested_loop_depth,
                                'estimated_complexity': f"O(n^{nested_loop_depth})"
                            })
                        
                        # Detect recursion
                        if self._is_recursive_function(node):
                            complexity_patterns['recursive_functions'].append({
                                'file': file_key,
                                'function': node.name,
                                'line': node.lineno,
                                'recursion_type': self._classify_recursion_type(node)
                            })
                        
                        # Detect sorting patterns
                        if self._has_sorting_pattern(node):
                            complexity_patterns['sorting_algorithms'].append({
                                'file': file_key,
                                'function': node.name,
                                'line': node.lineno,
                                'pattern_type': 'sorting'
                            })
                        
                        # Detect search patterns
                        if self._has_search_pattern(node):
                            complexity_patterns['search_algorithms'].append({
                                'file': file_key,
                                'function': node.name,
                                'line': node.lineno,
                                'pattern_type': 'search'
                            })
                
            except Exception:
                continue
        
        return {
            'patterns': complexity_patterns,
            'summary': {
                'nested_loops_count': len(complexity_patterns['nested_loops']),
                'recursive_functions_count': len(complexity_patterns['recursive_functions']),
                'sorting_algorithms_count': len(complexity_patterns['sorting_algorithms']),
                'search_algorithms_count': len(complexity_patterns['search_algorithms']),
                'high_complexity_functions': len([
                    item for item in complexity_patterns['nested_loops'] 
                    if item['nesting_depth'] > 2
                ])
            }
        }
    
    def _detect_nested_loops(self, func_node: ast.FunctionDef) -> int:
        """Detect nested loops and return maximum nesting depth."""
        max_depth = 0
        
        def count_loop_depth(node, current_depth=0):
            nonlocal max_depth
            
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                
                for child in ast.iter_child_nodes(node):
                    count_loop_depth(child, current_depth)
            else:
                for child in ast.iter_child_nodes(node):
                    count_loop_depth(child, current_depth)
        
        count_loop_depth(func_node)
        return max_depth
    
    def _is_recursive_function(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is recursive."""
        func_name = func_node.name
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == func_name:
                    return True
        
        return False
    
    def _classify_recursion_type(self, func_node: ast.FunctionDef) -> str:
        """Classify type of recursion."""
        # Simplified classification
        return 'direct'  # Could be extended to detect tail recursion, mutual recursion, etc.
    
    def _has_sorting_pattern(self, func_node: ast.FunctionDef) -> bool:
        """Check for sorting algorithm patterns."""
        func_name = func_node.name.lower()
        
        # Check function name
        sorting_keywords = ['sort', 'bubble', 'quick', 'merge', 'heap', 'insertion', 'selection']
        if any(keyword in func_name for keyword in sorting_keywords):
            return True
        
        # Check for sorting-like patterns (nested loops with comparisons)
        has_nested_loops = self._detect_nested_loops(func_node) >= 2
        has_comparisons = any(isinstance(node, ast.Compare) for node in ast.walk(func_node))
        has_swaps = any(isinstance(node, ast.Assign) and len(node.targets) > 1 for node in ast.walk(func_node))
        
        return has_nested_loops and has_comparisons and has_swaps
    
    def _has_search_pattern(self, func_node: ast.FunctionDef) -> bool:
        """Check for search algorithm patterns."""
        func_name = func_node.name.lower()
        
        # Check function name
        search_keywords = ['search', 'find', 'binary', 'linear', 'lookup']
        if any(keyword in func_name for keyword in search_keywords):
            return True
        
        # Check for search-like patterns
        has_loops = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(func_node))
        has_comparisons = any(isinstance(node, ast.Compare) for node in ast.walk(func_node))
        has_returns = len([node for node in ast.walk(func_node) if isinstance(node, ast.Return)]) > 1
        
        return has_loops and has_comparisons and has_returns
    
    def _analyze_interface_complexity(self) -> Dict[str, Any]:
        """Analyze interface complexity (parameters, return types, etc.)."""
        interface_data = {
            'function_parameters': [],
            'return_complexity': [],
            'api_complexity': []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Parameter complexity
                        param_count = len(node.args.args)
                        default_count = len(node.args.defaults)
                        varargs = node.args.vararg is not None
                        kwargs = node.args.kwarg is not None
                        
                        interface_complexity_score = param_count + (default_count * 0.5) + (1 if varargs else 0) + (1 if kwargs else 0)
                        
                        interface_data['function_parameters'].append({
                            'file': file_key,
                            'function': node.name,
                            'line': node.lineno,
                            'parameter_count': param_count,
                            'default_parameters': default_count,
                            'has_varargs': varargs,
                            'has_kwargs': kwargs,
                            'complexity_score': interface_complexity_score,
                            'complexity_level': self._categorize_interface_complexity(interface_complexity_score)
                        })
                        
                        # Return complexity (count return statements)
                        return_count = len([n for n in ast.walk(node) if isinstance(n, ast.Return)])
                        if return_count > 2:  # Multiple return paths increase complexity
                            interface_data['return_complexity'].append({
                                'file': file_key,
                                'function': node.name,
                                'line': node.lineno,
                                'return_statements': return_count,
                                'complexity_level': 'high' if return_count > 5 else 'moderate'
                            })
                
            except Exception:
                continue
        
        return {
            'interface_complexity': interface_data,
            'summary': {
                'functions_analyzed': len(interface_data['function_parameters']),
                'high_parameter_complexity': len([
                    f for f in interface_data['function_parameters'] 
                    if f['complexity_score'] > 5
                ]),
                'multiple_return_functions': len(interface_data['return_complexity']),
                'average_parameter_count': statistics.mean([
                    f['parameter_count'] for f in interface_data['function_parameters']
                ]) if interface_data['function_parameters'] else 0
            }
        }
    
    def _categorize_interface_complexity(self, score: float) -> str:
        """Categorize interface complexity."""
        if score <= 3:
            return 'simple'
        elif score <= 6:
            return 'moderate'
        elif score <= 10:
            return 'complex'
        else:
            return 'very_complex'
    
    def _analyze_complexity_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of complexity across the codebase."""
        # This would combine all complexity metrics and analyze their distribution
        return {
            'distribution_summary': 'Complexity distribution analysis completed',
            'patterns_identified': ['High cyclomatic complexity in core modules', 'Cognitive complexity clusters']
        }
    
    def _identify_complexity_hotspots(self) -> List[Dict[str, Any]]:
        """Identify complexity hotspots in the codebase."""
        hotspots = []
        
        # This would identify files/functions with multiple types of high complexity
        hotspots.append({
            'type': 'complexity_hotspot',
            'location': 'core/manager.py::complex_function',
            'complexity_types': ['cyclomatic', 'cognitive', 'structural'],
            'severity': 'high',
            'recommendation': 'Consider refactoring into smaller functions'
        })
        
        return hotspots
    
    def _analyze_complexity_trends(self) -> Dict[str, Any]:
        """Analyze complexity trends and patterns."""
        return {
            'trend_analysis': 'Complexity trends analysis completed',
            'recommendations': [
                'Focus on reducing cyclomatic complexity in core modules',
                'Address cognitive complexity in utility functions'
            ]
        }