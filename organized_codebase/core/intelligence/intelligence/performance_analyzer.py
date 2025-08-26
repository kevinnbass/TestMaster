"""
Performance Analyzer Module
============================

Analyzes code performance and identifies bottlenecks for optimization.
"""

import ast
import logging
import re
import numpy as np
from typing import Dict, List, Any

from .data_models import PerformanceAnalysis

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzes code performance and identifies bottlenecks"""
    
    def __init__(self):
        self.complexity_thresholds = {
            'low': 5,
            'medium': 10,
            'high': 20,
            'very_high': 50
        }
        
        self.performance_patterns = {
            'inefficient_loops': [
                r'for\s+\w+\s+in\s+range\(len\(',
                r'while.*len\(',
            ],
            'string_concatenation': [
                r'\+\s*=\s*["\']',
                r'["\'].*\+.*["\']'
            ],
            'repeated_calculations': [
                r'for.*in.*:.*\w+\(.*\)',
                r'while.*:.*\w+\(.*\)'
            ],
            'inefficient_data_structures': [
                r'\.append\(.*\)\s*in\s+for',
                r'list\(.*\)\s*\+\s*list\('
            ]
        }
    
    def analyze_performance(self, code: str, file_path: str = "") -> PerformanceAnalysis:
        """Comprehensive performance analysis of code"""
        try:
            analysis = PerformanceAnalysis()
            
            # Parse code
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return analysis
            
            # Analyze complexity
            analysis.complexity_analysis = self._analyze_complexity(tree)
            
            # Identify bottlenecks
            analysis.bottlenecks = self._identify_bottlenecks(tree, code)
            
            # Analyze memory usage patterns
            analysis.memory_analysis = self._analyze_memory_patterns(tree, code)
            
            # Analyze algorithmic efficiency
            analysis.algorithmic_efficiency = self._analyze_algorithmic_efficiency(tree)
            
            # Calculate performance score
            analysis.performance_score = self._calculate_performance_score(analysis)
            
            # Estimate optimization potential
            analysis.optimization_potential = self._estimate_optimization_potential(analysis)
            
            # Identify critical paths
            analysis.critical_paths = self._identify_critical_paths(tree)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return PerformanceAnalysis()
    
    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze computational complexity of code"""
        try:
            complexity_info = {
                'cyclomatic_complexity': 0,
                'cognitive_complexity': 0,
                'nesting_depth': 0,
                'function_complexities': {},
                'class_complexities': {}
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_complexity = self._calculate_function_complexity(node)
                    complexity_info['function_complexities'][node.name] = func_complexity
                    complexity_info['cyclomatic_complexity'] += func_complexity['cyclomatic']
                    complexity_info['cognitive_complexity'] += func_complexity['cognitive']
                    complexity_info['nesting_depth'] = max(
                        complexity_info['nesting_depth'], 
                        func_complexity['nesting_depth']
                    )
                elif isinstance(node, ast.ClassDef):
                    class_complexity = self._calculate_class_complexity(node)
                    complexity_info['class_complexities'][node.name] = class_complexity
            
            return complexity_info
            
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            return {}
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Calculate complexity metrics for a function"""
        try:
            cyclomatic = 1  # Base complexity
            cognitive = 0
            max_nesting = 0
            
            def analyze_node(n, nesting_level=0):
                nonlocal cyclomatic, cognitive, max_nesting
                max_nesting = max(max_nesting, nesting_level)
                
                if isinstance(n, (ast.If, ast.For, ast.While, ast.Try)):
                    cyclomatic += 1
                    cognitive += 1 + nesting_level  # Cognitive complexity increases with nesting
                    
                    for child in ast.iter_child_nodes(n):
                        analyze_node(child, nesting_level + 1)
                elif isinstance(n, ast.BoolOp):
                    cyclomatic += len(n.values) - 1
                    cognitive += len(n.values) - 1
                else:
                    for child in ast.iter_child_nodes(n):
                        analyze_node(child, nesting_level)
            
            analyze_node(node)
            
            return {
                'cyclomatic': cyclomatic,
                'cognitive': cognitive,
                'nesting_depth': max_nesting,
                'line_count': (node.end_lineno or node.lineno) - node.lineno,
                'parameter_count': len(node.args.args)
            }
            
        except Exception as e:
            logger.error(f"Error calculating function complexity: {e}")
            return {'cyclomatic': 1, 'cognitive': 1}
    
    def _calculate_class_complexity(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate complexity metrics for a class"""
        try:
            method_complexities = []
            total_methods = 0
            
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    total_methods += 1
                    method_complexity = self._calculate_function_complexity(child)
                    method_complexities.append(method_complexity['cyclomatic'])
            
            avg_complexity = np.mean(method_complexities) if method_complexities else 0
            max_complexity = max(method_complexities) if method_complexities else 0
            
            return {
                'method_count': total_methods,
                'average_method_complexity': avg_complexity,
                'max_method_complexity': max_complexity,
                'total_complexity': sum(method_complexities),
                'inheritance_depth': len(node.bases)
            }
            
        except Exception as e:
            logger.error(f"Error calculating class complexity: {e}")
            return {'method_count': 0}
    
    def _identify_bottlenecks(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in code"""
        try:
            bottlenecks = []
            
            # Check for performance anti-patterns
            for pattern_type, patterns in self.performance_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, code, re.MULTILINE)
                    for match in matches:
                        line_num = code[:match.start()].count('\n') + 1
                        bottlenecks.append({
                            'type': pattern_type,
                            'description': f"{pattern_type.replace('_', ' ').title()} detected",
                            'line': line_num,
                            'code_snippet': match.group(),
                            'severity': self._get_bottleneck_severity(pattern_type),
                            'optimization_suggestion': self._get_optimization_suggestion(pattern_type)
                        })
            
            # Check for nested loops
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    nesting_level = self._calculate_loop_nesting(node)
                    if nesting_level > 2:
                        bottlenecks.append({
                            'type': 'nested_loops',
                            'description': f"Deeply nested loops (level {nesting_level})",
                            'line': node.lineno,
                            'severity': 'high',
                            'optimization_suggestion': 'Consider algorithmic improvements or vectorization'
                        })
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {e}")
            return []
    
    def _calculate_loop_nesting(self, node: ast.For) -> int:
        """Calculate nesting level of loops"""
        try:
            nesting = 1
            for child in ast.walk(node):
                if isinstance(child, (ast.For, ast.While)) and child != node:
                    child_nesting = 1 + self._calculate_loop_nesting(child)
                    nesting = max(nesting, child_nesting)
            return nesting
        except:
            return 1
    
    def _get_bottleneck_severity(self, pattern_type: str) -> str:
        """Get severity level for bottleneck type"""
        severity_map = {
            'inefficient_loops': 'high',
            'string_concatenation': 'medium',
            'repeated_calculations': 'high',
            'inefficient_data_structures': 'medium'
        }
        return severity_map.get(pattern_type, 'low')
    
    def _get_optimization_suggestion(self, pattern_type: str) -> str:
        """Get optimization suggestion for bottleneck type"""
        suggestions = {
            'inefficient_loops': 'Use enumerate() or list comprehensions',
            'string_concatenation': 'Use join() or f-strings for better performance',
            'repeated_calculations': 'Cache results or move calculations outside loops',
            'inefficient_data_structures': 'Use appropriate data structures (sets, deques, etc.)'
        }
        return suggestions.get(pattern_type, 'Consider refactoring for better performance')
    
    def _analyze_memory_patterns(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        try:
            memory_info = {
                'large_data_structures': [],
                'memory_leaks_potential': [],
                'inefficient_copying': [],
                'memory_score': 0.8
            }
            
            # Check for large list operations
            for node in ast.walk(tree):
                if isinstance(node, ast.ListComp):
                    # List comprehensions can use significant memory
                    memory_info['large_data_structures'].append({
                        'type': 'list_comprehension',
                        'line': node.lineno,
                        'suggestion': 'Consider generator expressions for large datasets'
                    })
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ['list', 'dict', 'set']:
                        memory_info['large_data_structures'].append({
                            'type': 'explicit_construction',
                            'line': node.lineno,
                            'suggestion': 'Consider lazy evaluation or streaming'
                        })
            
            # Check for potential memory leaks (simplified)
            if 'global ' in code or 'class ' in code:
                memory_info['memory_leaks_potential'].append({
                    'type': 'global_state',
                    'suggestion': 'Review global variables and class instances for proper cleanup'
                })
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error analyzing memory patterns: {e}")
            return {}
    
    def _analyze_algorithmic_efficiency(self, tree: ast.AST) -> Dict[str, str]:
        """Analyze algorithmic efficiency of code"""
        try:
            efficiency_analysis = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    efficiency = self._estimate_time_complexity(node)
                    efficiency_analysis[node.name] = efficiency
            
            return efficiency_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing algorithmic efficiency: {e}")
            return {}
    
    def _estimate_time_complexity(self, node: ast.FunctionDef) -> str:
        """Estimate time complexity of function"""
        try:
            # Simple heuristic-based complexity estimation
            loop_count = 0
            nested_loops = 0
            
            for child in ast.walk(node):
                if isinstance(child, (ast.For, ast.While)):
                    loop_count += 1
                    # Check for nested loops
                    for nested in ast.walk(child):
                        if isinstance(nested, (ast.For, ast.While)) and nested != child:
                            nested_loops += 1
            
            if nested_loops > 1:
                return "O(n³) or higher"
            elif nested_loops == 1:
                return "O(n²)"
            elif loop_count > 0:
                return "O(n)"
            else:
                return "O(1)"
                
        except Exception as e:
            logger.error(f"Error estimating time complexity: {e}")
            return "O(?)"
    
    def _calculate_performance_score(self, analysis: PerformanceAnalysis) -> float:
        """Calculate overall performance score"""
        try:
            score = 1.0
            
            # Penalize high complexity
            avg_complexity = analysis.complexity_analysis.get('cyclomatic_complexity', 0) / max(1, len(analysis.complexity_analysis.get('function_complexities', {})))
            if avg_complexity > 10:
                score -= 0.3
            elif avg_complexity > 5:
                score -= 0.1
            
            # Penalize bottlenecks
            high_severity_bottlenecks = len([b for b in analysis.bottlenecks if b.get('severity') == 'high'])
            score -= high_severity_bottlenecks * 0.2
            
            medium_severity_bottlenecks = len([b for b in analysis.bottlenecks if b.get('severity') == 'medium'])
            score -= medium_severity_bottlenecks * 0.1
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    def _estimate_optimization_potential(self, analysis: PerformanceAnalysis) -> float:
        """Estimate potential for performance optimization"""
        try:
            potential = 0.0
            
            # Higher potential with more bottlenecks
            potential += len(analysis.bottlenecks) * 0.1
            
            # Higher potential with high complexity
            avg_complexity = analysis.complexity_analysis.get('cyclomatic_complexity', 0)
            if avg_complexity > 20:
                potential += 0.4
            elif avg_complexity > 10:
                potential += 0.2
            
            return min(1.0, potential)
            
        except Exception as e:
            logger.error(f"Error estimating optimization potential: {e}")
            return 0.3
    
    def _identify_critical_paths(self, tree: ast.AST) -> List[str]:
        """Identify critical execution paths"""
        try:
            critical_paths = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Functions with high complexity are critical
                    complexity = self._calculate_function_complexity(node)
                    if complexity['cyclomatic'] > 10:
                        critical_paths.append(f"Function: {node.name}")
                        
                elif isinstance(node, ast.For):
                    # Nested loops are critical
                    nesting = self._calculate_loop_nesting(node)
                    if nesting > 2:
                        critical_paths.append(f"Nested loop at line {node.lineno}")
            
            return critical_paths
            
        except Exception as e:
            logger.error(f"Error identifying critical paths: {e}")
            return []


__all__ = ['PerformanceAnalyzer']