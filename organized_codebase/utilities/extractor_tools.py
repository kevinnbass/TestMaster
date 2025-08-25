from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
TestMaster Utility Extraction Tools

Tools for extracting and analyzing utility functions across the codebase.
Based on the utility_function_extractor.py functionality.
"""

import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, Counter


class UtilityExtractor:
    """Extracts utility functions and patterns from code"""
    
    def __init__(self):
        self.functions = {}
        self.function_signatures = defaultdict(list)
        self.utility_patterns = defaultdict(int)
        self.helper_functions = []
        self.common_utilities = []
        self.function_calls = defaultdict(set)
        self.function_metrics = {}
    
    def extract_utilities_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract utilities from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract functions
            functions = self._extract_functions(tree, str(file_path))
            
            # Identify utility patterns
            patterns = self._identify_utility_patterns(tree, content)
            
            # Calculate metrics
            metrics = self._calculate_file_metrics(tree, content)
            
            return {
                'file': str(file_path),
                'functions': functions,
                'patterns': patterns,
                'metrics': metrics,
                'analyzed': True
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'analyzed': False
            }
    
    def _extract_functions(self, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Extract all functions from AST"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node),
                    'decorators': [ast.unparse(d) for d in node.decorator_list],
                    'complexity': self._calculate_complexity(node),
                    'is_utility': self._is_utility_function(node)
                }
                functions.append(func_info)
        
        return functions
    
    def _identify_utility_patterns(self, tree: ast.AST, content: str) -> Dict[str, List[str]]:
        """Identify utility patterns in code"""
        patterns = {
            'helper_functions': [],
            'utility_classes': [],
            'common_patterns': [],
            'data_processors': []
        }
        
        # Helper function patterns
        helper_patterns = [
            'get_', 'set_', 'create_', 'build_', 'make_', 'generate_',
            'format_', 'parse_', 'validate_', 'clean_', 'normalize_',
            'transform_', 'convert_', 'extract_', 'calculate_', 'process_'
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for helper function patterns
                if any(node.name.startswith(pattern) for pattern in helper_patterns):
                    patterns['helper_functions'].append(node.name)
                
                # Check for utility function characteristics
                if self._is_utility_function(node):
                    patterns['common_patterns'].append(node.name)
            
            elif isinstance(node, ast.ClassDef):
                # Check for utility classes
                if any(term in node.name.lower() for term in ['util', 'helper', 'tool', 'manager']):
                    patterns['utility_classes'].append(node.name)
        
        return patterns
    
    def _is_utility_function(self, node: ast.FunctionDef) -> bool:
        """Determine if a function is likely a utility function"""
        # Utility function indicators
        utility_indicators = [
            # Name patterns
            lambda: any(pattern in node.name.lower() for pattern in [
                'util', 'helper', 'tool', 'get_', 'set_', 'create_', 
                'build_', 'make_', 'format_', 'parse_', 'validate_'
            ]),
            
            # Small, focused functions
            lambda: len(node.body) < 20,
            
            # Pure functions (no class methods)
            lambda: not any(isinstance(arg, ast.arg) and arg.arg == 'self' 
                          for arg in node.args.args),
            
            # Functions with docstrings (well-documented utilities)
            lambda: ast.get_docstring(node) is not None,
        ]
        
        # Function is utility if it matches multiple indicators
        matches = sum(1 for indicator in utility_indicators if indicator())
        return matches >= 2
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_file_metrics(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Calculate various metrics for the file"""
        metrics = {
            'lines_of_code': len(content.split('\n')),
            'function_count': 0,
            'class_count': 0,
            'import_count': 0,
            'utility_function_count': 0,
            'complexity_score': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics['function_count'] += 1
                if self._is_utility_function(node):
                    metrics['utility_function_count'] += 1
                metrics['complexity_score'] += self._calculate_complexity(node)
            
            elif isinstance(node, ast.ClassDef):
                metrics['class_count'] += 1
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['import_count'] += 1
        
        return metrics


class FunctionAnalyzer:
    """Analyzes functions for patterns and characteristics"""
    
    def __init__(self):
        self.function_patterns = {}
        self.similarity_matrix = {}
        self.duplication_candidates = []
    
    def analyze_function_patterns(self, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across multiple functions"""
        analysis = {
            'total_functions': len(functions),
            'utility_functions': 0,
            'helper_functions': 0,
            'complex_functions': 0,
            'similar_functions': [],
            'pattern_summary': {}
        }
        
        # Categorize functions
        for func in functions:
            if func.get('is_utility', False):
                analysis['utility_functions'] += 1
            
            if any(func['name'].startswith(pattern) for pattern in 
                  ['get_', 'set_', 'create_', 'build_', 'make_', 'format_']):
                analysis['helper_functions'] += 1
            
            if func.get('complexity', 0) > 10:
                analysis['complex_functions'] += 1
        
        # Find similar functions
        similar_functions = self._find_similar_functions(functions)
        analysis['similar_functions'] = similar_functions
        
        # Pattern summary
        analysis['pattern_summary'] = self._summarize_patterns(functions)
        
        return analysis
    
    def _find_similar_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find functions with similar signatures or names"""
        similar_groups = []
        processed = set()
        
        for i, func1 in enumerate(functions):
            if i in processed:
                continue
            
            similar_group = [func1]
            for j, func2 in enumerate(functions[i+1:], i+1):
                if j in processed:
                    continue
                
                # Check similarity
                if self._functions_similar(func1, func2):
                    similar_group.append(func2)
                    processed.add(j)
            
            if len(similar_group) > 1:
                similar_groups.append({
                    'group_id': len(similar_groups) + 1,
                    'functions': similar_group,
                    'similarity_reason': self._get_similarity_reason(similar_group)
                })
                processed.add(i)
        
        return similar_groups
    
    def _functions_similar(self, func1: Dict[str, Any], func2: Dict[str, Any]) -> bool:
        """Check if two functions are similar"""
        # Name similarity
        name_similarity = self._calculate_name_similarity(func1['name'], func2['name'])
        
        # Argument similarity
        args1 = set(func1.get('args', []))
        args2 = set(func2.get('args', []))
        arg_similarity = len(args1 & args2) / max(len(args1 | args2), 1)
        
        # Overall similarity threshold
        return name_similarity > 0.6 or arg_similarity > 0.8
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between function names"""
        # Simple similarity based on common substrings
        if name1 == name2:
            return 1.0
        
        # Check for common prefixes/suffixes
        common_prefixes = ['get_', 'set_', 'create_', 'build_', 'make_', 'format_']
        
        for prefix in common_prefixes:
            if name1.startswith(prefix) and name2.startswith(prefix):
                suffix1 = name1[len(prefix):]
                suffix2 = name2[len(prefix):]
                if suffix1 == suffix2:
                    return 0.9
        
        # Basic string similarity
        common_chars = set(name1) & set(name2)
        return len(common_chars) / max(len(set(name1) | set(name2)), 1)
    
    def _get_similarity_reason(self, similar_group: List[Dict[str, Any]]) -> str:
        """Get reason why functions are considered similar"""
        names = [func['name'] for func in similar_group]
        
        # Check for common patterns
        common_prefixes = ['get_', 'set_', 'create_', 'build_', 'make_', 'format_']
        
        for prefix in common_prefixes:
            if all(name.startswith(prefix) for name in names):
                return f"Common prefix: {prefix}"
        
        return "Similar naming pattern"
    
    def _summarize_patterns(self, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize patterns found in functions"""
        patterns = {
            'naming_patterns': Counter(),
            'complexity_distribution': Counter(),
            'async_functions': 0,
            'documented_functions': 0,
            'decorated_functions': 0
        }
        
        for func in functions:
            # Naming patterns
            name = func['name']
            if '_' in name:
                parts = name.split('_')
                if parts:
                    patterns['naming_patterns'][parts[0] + '_'] += 1
            
            # Complexity distribution
            complexity = func.get('complexity', 0)
            if complexity <= 5:
                patterns['complexity_distribution']['low'] += 1
            elif complexity <= 10:
                patterns['complexity_distribution']['medium'] += 1
            else:
                patterns['complexity_distribution']['high'] += 1
            
            # Other characteristics
            if func.get('is_async', False):
                patterns['async_functions'] += 1
            
            if func.get('docstring'):
                patterns['documented_functions'] += 1
            
            if func.get('decorators'):
                patterns['decorated_functions'] += 1
        
        return patterns