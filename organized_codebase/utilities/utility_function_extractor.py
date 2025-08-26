#!/usr/bin/env python3
"""
Utility Function Extraction Tool - Agent C Hours 38-40
Identifies and extracts reusable utility functions across the codebase.
Detects helper functions, common patterns, and consolidation opportunities.
"""

import ast
import json
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Optional, Tuple
import sys

class UtilityFunctionExtractor(ast.NodeVisitor):
    """Extracts and analyzes utility functions across the codebase."""
    
    def __init__(self):
        self.functions = {}
        self.function_signatures = defaultdict(list)
        self.utility_patterns = defaultdict(int)
        self.helper_functions = []
        self.common_utilities = []
        self.current_file = None
        self.current_class = None
        self.function_calls = defaultdict(set)
        self.function_metrics = {}
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for utility functions."""
        self.current_file = str(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            self.visit(tree)
            
            return {
                'file': str(file_path),
                'functions_found': len([f for f in self.functions.values() if f['file'] == str(file_path)]),
                'analyzed': True
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'analyzed': False
            }
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to track context."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and analyze for utility patterns."""
        func_info = self._analyze_function(node)
        
        # Generate unique function ID
        func_id = f"{self.current_file}::{self.current_class or 'module'}::{node.name}::{node.lineno}"
        self.functions[func_id] = func_info
        
        # Track function signature for similarity detection
        signature = self._generate_signature(node)
        self.function_signatures[signature].append(func_id)
        
        # Calculate function metrics
        metrics = self._calculate_function_metrics(node)
        self.function_metrics[func_id] = metrics
        
        # Check for utility patterns
        self._detect_utility_patterns(node, func_info)
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        func_info = self._analyze_function(node)
        func_info['is_async'] = True
        
        func_id = f"{self.current_file}::{self.current_class or 'module'}::{node.name}::{node.lineno}"
        self.functions[func_id] = func_info
        
        signature = self._generate_signature(node)
        self.function_signatures[signature].append(func_id)
        
        metrics = self._calculate_function_metrics(node)
        self.function_metrics[func_id] = metrics
        
        self._detect_utility_patterns(node, func_info)
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Track function calls for usage analysis."""
        if isinstance(node.func, ast.Name):
            if hasattr(self, '_current_function'):
                self.function_calls[self._current_function].add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if hasattr(self, '_current_function'):
                attr_name = node.func.attr
                self.function_calls[self._current_function].add(attr_name)
        
        self.generic_visit(node)
    
    def _analyze_function(self, node) -> Dict[str, Any]:
        """Analyze a function node for utility characteristics."""
        return {
            'name': node.name,
            'file': self.current_file,
            'line': node.lineno,
            'class': self.current_class,
            'args': [arg.arg for arg in node.args.args],
            'arg_count': len(node.args.args),
            'has_defaults': len(node.args.defaults) > 0,
            'has_varargs': node.args.vararg is not None,
            'has_kwargs': node.args.kwarg is not None,
            'returns': self._has_return_statement(node),
            'docstring': ast.get_docstring(node),
            'is_private': node.name.startswith('_'),
            'is_dunder': node.name.startswith('__') and node.name.endswith('__'),
            'is_async': False,  # Will be overridden for async functions
            'body_length': len(node.body),
            'complexity': self._calculate_complexity(node)
        }
    
    def _generate_signature(self, node) -> str:
        """Generate a signature hash for function similarity detection."""
        # Create signature based on argument patterns and structure
        args_pattern = f"args:{len(node.args.args)}"
        defaults_pattern = f"defaults:{len(node.args.defaults)}"
        varargs_pattern = f"varargs:{node.args.vararg is not None}"
        kwargs_pattern = f"kwargs:{node.args.kwarg is not None}"
        body_pattern = f"body:{len(node.body)}"
        
        signature_str = f"{args_pattern}|{defaults_pattern}|{varargs_pattern}|{kwargs_pattern}|{body_pattern}"
        return hashlib.md5(signature_str.encode()).hexdigest()[:8]
    
    def _has_return_statement(self, node) -> bool:
        """Check if function has return statements."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
        return False
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_function_metrics(self, node) -> Dict[str, Any]:
        """Calculate detailed metrics for a function."""
        # Count different types of statements
        statements = {'assignments': 0, 'calls': 0, 'returns': 0, 'conditionals': 0}
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                statements['assignments'] += 1
            elif isinstance(child, ast.Call):
                statements['calls'] += 1
            elif isinstance(child, ast.Return):
                statements['returns'] += 1
            elif isinstance(child, ast.If):
                statements['conditionals'] += 1
        
        return {
            'line_count': len(node.body),
            'complexity': self._calculate_complexity(node),
            'statements': statements,
            'parameter_count': len(node.args.args),
            'local_variables': self._count_local_variables(node)
        }
    
    def _count_local_variables(self, node) -> int:
        """Count local variable assignments in function."""
        local_vars = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        local_vars.add(target.id)
        return len(local_vars)
    
    def _detect_utility_patterns(self, node, func_info: Dict[str, Any]):
        """Detect common utility function patterns."""
        # Helper function patterns
        if self._is_helper_function(node, func_info):
            self.helper_functions.append(func_info['name'])
            self.utility_patterns['helper_functions'] += 1
        
        # Validator patterns
        if self._is_validator_function(node, func_info):
            self.utility_patterns['validator_functions'] += 1
        
        # Converter patterns
        if self._is_converter_function(node, func_info):
            self.utility_patterns['converter_functions'] += 1
        
        # Formatter patterns
        if self._is_formatter_function(node, func_info):
            self.utility_patterns['formatter_functions'] += 1
        
        # Configuration patterns
        if self._is_config_function(node, func_info):
            self.utility_patterns['config_functions'] += 1
        
        # Logging patterns
        if self._is_logging_function(node, func_info):
            self.utility_patterns['logging_functions'] += 1
    
    def _is_helper_function(self, node, func_info: Dict[str, Any]) -> bool:
        """Detect if function is a helper function."""
        helper_indicators = [
            func_info['name'].startswith('_') and not func_info['is_dunder'],
            'helper' in func_info['name'].lower(),
            'util' in func_info['name'].lower(),
            'assist' in func_info['name'].lower(),
            func_info['complexity'] <= 3 and func_info['arg_count'] <= 3
        ]
        return any(helper_indicators)
    
    def _is_validator_function(self, node, func_info: Dict[str, Any]) -> bool:
        """Detect validator functions."""
        validator_patterns = [
            'validate' in func_info['name'].lower(),
            'check' in func_info['name'].lower(),
            'verify' in func_info['name'].lower(),
            'is_valid' in func_info['name'].lower(),
            func_info['returns'] and func_info['complexity'] >= 2
        ]
        return any(validator_patterns)
    
    def _is_converter_function(self, node, func_info: Dict[str, Any]) -> bool:
        """Detect converter functions."""
        converter_patterns = [
            'convert' in func_info['name'].lower(),
            'transform' in func_info['name'].lower(),
            'to_' in func_info['name'].lower(),
            'from_' in func_info['name'].lower(),
            'parse' in func_info['name'].lower(),
            func_info['returns'] and func_info['arg_count'] >= 1
        ]
        return any(converter_patterns)
    
    def _is_formatter_function(self, node, func_info: Dict[str, Any]) -> bool:
        """Detect formatter functions."""
        formatter_patterns = [
            'format' in func_info['name'].lower(),
            'render' in func_info['name'].lower(),
            'display' in func_info['name'].lower(),
            'print' in func_info['name'].lower(),
            'show' in func_info['name'].lower()
        ]
        return any(formatter_patterns)
    
    def _is_config_function(self, node, func_info: Dict[str, Any]) -> bool:
        """Detect configuration functions."""
        config_patterns = [
            'config' in func_info['name'].lower(),
            'setting' in func_info['name'].lower(),
            'setup' in func_info['name'].lower(),
            'init' in func_info['name'].lower() and not func_info['is_dunder'],
            'load' in func_info['name'].lower()
        ]
        return any(config_patterns)
    
    def _is_logging_function(self, node, func_info: Dict[str, Any]) -> bool:
        """Detect logging functions."""
        logging_patterns = [
            'log' in func_info['name'].lower(),
            'debug' in func_info['name'].lower(),
            'trace' in func_info['name'].lower(),
            'report' in func_info['name'].lower()
        ]
        return any(logging_patterns)
    
    def analyze_directory(self, root_path: Path) -> Dict[str, Any]:
        """Analyze all Python files in directory."""
        results = {
            'files_analyzed': [],
            'total_files': 0,
            'successful_analyses': 0,
            'errors': []
        }
        
        python_files = list(root_path.rglob('*.py'))
        results['total_files'] = len(python_files)
        
        print(f"Analyzing {len(python_files)} Python files for utility functions...")
        
        for i, file_path in enumerate(python_files):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(python_files)} files analyzed")
            
            analysis = self.analyze_file(file_path)
            results['files_analyzed'].append(analysis)
            
            if analysis.get('analyzed', False):
                results['successful_analyses'] += 1
            else:
                results['errors'].append(analysis)
        
        print(f"Analysis complete: {results['successful_analyses']}/{results['total_files']} files analyzed successfully")
        return results
    
    def detect_extraction_opportunities(self) -> Dict[str, Any]:
        """Detect opportunities for utility function extraction."""
        opportunities = {
            'similar_functions': self._find_similar_functions(),
            'reusable_utilities': self._find_reusable_utilities(),
            'consolidation_candidates': self._find_consolidation_candidates(),
            'common_patterns': self._analyze_common_patterns(),
            'extraction_recommendations': self._generate_extraction_recommendations()
        }
        
        return opportunities
    
    def _find_similar_functions(self) -> List[Dict[str, Any]]:
        """Find functions with similar signatures that could be consolidated."""
        similar_groups = []
        
        for signature, func_ids in self.function_signatures.items():
            if len(func_ids) > 1:
                functions = [self.functions[fid] for fid in func_ids]
                # Check if functions are actually similar (not just same signature)
                if self._are_functions_similar(functions):
                    similar_groups.append({
                        'signature': signature,
                        'function_count': len(func_ids),
                        'functions': func_ids,
                        'similarity_score': self._calculate_similarity_score(functions)
                    })
        
        return sorted(similar_groups, key=lambda x: x['similarity_score'], reverse=True)[:20]
    
    def _find_reusable_utilities(self) -> List[Dict[str, Any]]:
        """Find utility functions that could be extracted to shared modules."""
        reusable = []
        
        for func_id, func_info in self.functions.items():
            metrics = self.function_metrics.get(func_id, {})
            
            # Criteria for reusable utility
            is_reusable = (
                func_info['complexity'] <= 5 and
                func_info['arg_count'] <= 4 and
                func_info['returns'] and
                not func_info['is_private'] and
                metrics.get('line_count', 0) <= 15
            )
            
            if is_reusable:
                reusable.append({
                    'function_id': func_id,
                    'name': func_info['name'],
                    'file': func_info['file'],
                    'complexity': func_info['complexity'],
                    'reusability_score': self._calculate_reusability_score(func_info, metrics)
                })
        
        return sorted(reusable, key=lambda x: x['reusability_score'], reverse=True)[:30]
    
    def _find_consolidation_candidates(self) -> List[Dict[str, Any]]:
        """Find functions that could be consolidated into utility modules."""
        candidates = []
        
        # Group functions by utility patterns
        pattern_groups = defaultdict(list)
        for func_id, func_info in self.functions.items():
            for pattern_type in ['helper', 'validator', 'converter', 'formatter', 'config', 'logging']:
                if getattr(self, f'_is_{pattern_type}_function', lambda n, f: False)(None, func_info):
                    pattern_groups[pattern_type].append(func_id)
        
        for pattern_type, func_ids in pattern_groups.items():
            if len(func_ids) >= 3:  # At least 3 functions to justify consolidation
                candidates.append({
                    'pattern_type': pattern_type,
                    'function_count': len(func_ids),
                    'functions': func_ids[:10],  # Limit for readability
                    'consolidation_benefit': len(func_ids) * 0.1  # Simple benefit score
                })
        
        return sorted(candidates, key=lambda x: x['consolidation_benefit'], reverse=True)
    
    def _analyze_common_patterns(self) -> Dict[str, Any]:
        """Analyze common patterns across utility functions."""
        patterns = {
            'function_lengths': Counter(),
            'complexity_distribution': Counter(),
            'parameter_patterns': Counter(),
            'naming_patterns': Counter()
        }
        
        for func_info in self.functions.values():
            metrics = self.function_metrics.get(f"{func_info['file']}::{func_info.get('class', 'module')}::{func_info['name']}::{func_info['line']}", {})
            
            # Function length distribution
            length = metrics.get('line_count', 0)
            if length <= 5:
                patterns['function_lengths']['very_short'] += 1
            elif length <= 15:
                patterns['function_lengths']['short'] += 1
            elif length <= 30:
                patterns['function_lengths']['medium'] += 1
            else:
                patterns['function_lengths']['long'] += 1
            
            # Complexity distribution
            complexity = func_info['complexity']
            if complexity <= 2:
                patterns['complexity_distribution']['simple'] += 1
            elif complexity <= 5:
                patterns['complexity_distribution']['moderate'] += 1
            else:
                patterns['complexity_distribution']['complex'] += 1
            
            # Parameter patterns
            param_count = func_info['arg_count']
            if param_count == 0:
                patterns['parameter_patterns']['no_params'] += 1
            elif param_count <= 2:
                patterns['parameter_patterns']['few_params'] += 1
            elif param_count <= 4:
                patterns['parameter_patterns']['moderate_params'] += 1
            else:
                patterns['parameter_patterns']['many_params'] += 1
            
            # Naming patterns
            name = func_info['name']
            if name.startswith('_'):
                patterns['naming_patterns']['private'] += 1
            elif any(word in name.lower() for word in ['get', 'set', 'create', 'make']):
                patterns['naming_patterns']['action_verbs'] += 1
            elif any(word in name.lower() for word in ['is', 'has', 'can', 'should']):
                patterns['naming_patterns']['boolean_verbs'] += 1
            else:
                patterns['naming_patterns']['other'] += 1
        
        return {k: dict(v) for k, v in patterns.items()}
    
    def _generate_extraction_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific recommendations for utility extraction."""
        recommendations = []
        
        # Recommend creating utility modules for common patterns
        if self.utility_patterns['helper_functions'] > 10:
            recommendations.append({
                'type': 'create_module',
                'module_name': 'utils.helpers',
                'reason': f"{self.utility_patterns['helper_functions']} helper functions could be consolidated",
                'priority': 'high'
            })
        
        if self.utility_patterns['validator_functions'] > 5:
            recommendations.append({
                'type': 'create_module',
                'module_name': 'utils.validators',
                'reason': f"{self.utility_patterns['validator_functions']} validator functions could be consolidated",
                'priority': 'medium'
            })
        
        if self.utility_patterns['converter_functions'] > 5:
            recommendations.append({
                'type': 'create_module',
                'module_name': 'utils.converters',
                'reason': f"{self.utility_patterns['converter_functions']} converter functions could be consolidated",
                'priority': 'medium'
            })
        
        # Recommend function refactoring for complex utilities
        complex_functions = [f for f in self.functions.values() if f['complexity'] > 10]
        if len(complex_functions) > 0:
            recommendations.append({
                'type': 'refactor_complex',
                'affected_functions': len(complex_functions),
                'reason': f"{len(complex_functions)} functions have high complexity and could be refactored",
                'priority': 'low'
            })
        
        return recommendations
    
    def _are_functions_similar(self, functions: List[Dict[str, Any]]) -> bool:
        """Check if functions are actually similar beyond signature."""
        if len(functions) < 2:
            return False
        
        # Simple similarity check based on naming and characteristics
        names = [f['name'] for f in functions]
        complexities = [f['complexity'] for f in functions]
        
        # Check name similarity
        name_similarity = len(set(names)) < len(names) * 0.8
        
        # Check complexity similarity
        complexity_similarity = max(complexities) - min(complexities) <= 2
        
        return name_similarity or complexity_similarity
    
    def _calculate_similarity_score(self, functions: List[Dict[str, Any]]) -> float:
        """Calculate similarity score for a group of functions."""
        if len(functions) < 2:
            return 0.0
        
        # Base score from number of functions
        score = len(functions) * 0.1
        
        # Bonus for similar names
        names = [f['name'] for f in functions]
        unique_names = len(set(names))
        if unique_names < len(names):
            score += 0.3
        
        # Bonus for similar complexity
        complexities = [f['complexity'] for f in functions]
        if max(complexities) - min(complexities) <= 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_reusability_score(self, func_info: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Calculate reusability score for a function."""
        score = 0.0
        
        # Bonus for simple functions
        if func_info['complexity'] <= 3:
            score += 0.3
        
        # Bonus for few parameters
        if func_info['arg_count'] <= 2:
            score += 0.2
        
        # Bonus for short functions
        if metrics.get('line_count', 0) <= 10:
            score += 0.2
        
        # Bonus for functions with returns
        if func_info['returns']:
            score += 0.1
        
        # Bonus for good naming
        if any(word in func_info['name'].lower() for word in ['get', 'create', 'format', 'convert']):
            score += 0.1
        
        # Penalty for private functions
        if func_info['is_private']:
            score -= 0.1
        
        return max(score, 0.0)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        extraction_opportunities = self.detect_extraction_opportunities()
        
        summary = {
            'function_statistics': {
                'total_functions': len(self.functions),
                'utility_patterns': dict(self.utility_patterns),
                'helper_functions': len(self.helper_functions),
                'average_complexity': sum(f['complexity'] for f in self.functions.values()) / len(self.functions) if self.functions else 0,
                'functions_by_type': self._categorize_functions_by_type()
            },
            'extraction_analysis': {
                'similar_function_groups': len(extraction_opportunities['similar_functions']),
                'reusable_utilities': len(extraction_opportunities['reusable_utilities']),
                'consolidation_opportunities': len(extraction_opportunities['consolidation_candidates']),
                'extraction_recommendations': len(extraction_opportunities['extraction_recommendations'])
            },
            'extraction_opportunities': extraction_opportunities,
            'common_patterns': extraction_opportunities['common_patterns'],
            'utility_health_score': self._calculate_utility_health_score()
        }
        
        return summary
    
    def _categorize_functions_by_type(self) -> Dict[str, int]:
        """Categorize functions by their characteristics."""
        categories = {
            'public_methods': 0,
            'private_methods': 0,
            'class_methods': 0,
            'module_functions': 0,
            'async_functions': 0,
            'simple_functions': 0,
            'complex_functions': 0
        }
        
        for func_info in self.functions.values():
            if func_info['class']:
                categories['class_methods'] += 1
            else:
                categories['module_functions'] += 1
            
            if func_info['is_private']:
                categories['private_methods'] += 1
            else:
                categories['public_methods'] += 1
            
            if func_info.get('is_async', False):
                categories['async_functions'] += 1
            
            if func_info['complexity'] <= 3:
                categories['simple_functions'] += 1
            else:
                categories['complex_functions'] += 1
        
        return categories
    
    def _calculate_utility_health_score(self) -> float:
        """Calculate overall utility health score (0-100)."""
        if not self.functions:
            return 0.0
        
        score = 100.0
        
        # Penalize high complexity
        avg_complexity = sum(f['complexity'] for f in self.functions.values()) / len(self.functions)
        if avg_complexity > 5:
            score -= min((avg_complexity - 5) * 5, 20)
        
        # Bonus for good utility patterns
        total_utility_patterns = sum(self.utility_patterns.values())
        utility_ratio = total_utility_patterns / len(self.functions)
        if utility_ratio > 0.2:
            score += min(utility_ratio * 20, 15)
        
        # Penalize too many similar functions
        similar_groups = len([sig for sig, funcs in self.function_signatures.items() if len(funcs) > 1])
        if similar_groups > 10:
            score -= min(similar_groups * 2, 25)
        
        return max(score, 0)


def main():
    parser = argparse.ArgumentParser(description='Utility Function Extraction Tool')
    parser.add_argument('--root', type=str, required=True, help='Root directory to analyze')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    print("=== Agent C Hours 38-40: Utility Function Extraction ===")
    print(f"Analyzing directory: {args.root}")
    
    extractor = UtilityFunctionExtractor()
    root_path = Path(args.root)
    
    # Analyze directory
    analysis_results = extractor.analyze_directory(root_path)
    
    # Generate summary
    summary = extractor.generate_summary()
    
    # Combine results
    final_results = {
        'analysis_metadata': {
            'tool': 'utility_function_extractor',
            'version': '1.0',
            'agent': 'Agent_C',
            'hours': '38-40',
            'phase': 'Utility_Component_Extraction'
        },
        'analysis_results': analysis_results,
        'summary': summary,
        'raw_data': {
            'functions': dict(extractor.functions),
            'function_signatures': {k: list(v) for k, v in extractor.function_signatures.items()},
            'utility_patterns': dict(extractor.utility_patterns),
            'function_metrics': dict(extractor.function_metrics)
        }
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== UTILITY FUNCTION EXTRACTION COMPLETE ===")
    print(f"Files analyzed: {analysis_results['successful_analyses']}/{analysis_results['total_files']}")
    print(f"Functions found: {summary['function_statistics']['total_functions']}")
    print(f"Helper functions: {summary['function_statistics']['helper_functions']}")
    print(f"Reusable utilities: {summary['extraction_analysis']['reusable_utilities']}")
    print(f"Consolidation opportunities: {summary['extraction_analysis']['consolidation_opportunities']}")
    print(f"Utility health score: {summary['utility_health_score']:.1f}/100")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()