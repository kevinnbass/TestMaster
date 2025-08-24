#!/usr/bin/env python3
"""
AST Code Processing Engine for Enhanced Linkage Dashboard
=========================================================

Extracted from enhanced_linkage_dashboard.py for STEELCLAD modularization.
Provides Abstract Syntax Tree analysis and code structure processing.

Author: Agent Y (STEELCLAD Protocol)
"""

import ast
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any


class ASTCodeProcessor:
    """Advanced AST processing system for code analysis."""
    
    def __init__(self):
        self.analysis_cache = {}
        self.supported_node_types = {
            'FunctionDef': 'Function definitions',
            'ClassDef': 'Class definitions',
            'Import': 'Import statements',
            'ImportFrom': 'From import statements',
            'Assign': 'Assignment statements',
            'If': 'Conditional statements',
            'For': 'For loops',
            'While': 'While loops',
            'Try': 'Try-except blocks',
            'With': 'Context managers'
        }
        
    def analyze_file_quick(self, file_path):
        """Quick AST analysis of a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            imports = []
            exports = []
            
            # Walk the AST and extract key information
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not node.name.startswith('_'):
                        exports.append(node.name)
            
            return imports, exports
            
        except Exception as e:
            return [], []
    
    def analyze_file_comprehensive(self, file_path):
        """Comprehensive AST analysis of a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            analysis_result = {
                'file_path': str(file_path),
                'analysis_timestamp': datetime.now().isoformat(),
                'ast_structure': self._analyze_ast_structure(tree),
                'imports': self._extract_imports(tree),
                'exports': self._extract_exports(tree),
                'complexity_metrics': self._calculate_complexity_metrics(tree),
                'code_patterns': self._identify_code_patterns(tree),
                'dependencies': self._analyze_dependencies(tree),
                'quality_indicators': self._assess_code_quality(tree, content)
            }
            
            return analysis_result
            
        except Exception as e:
            return {
                'file_path': str(file_path),
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _analyze_ast_structure(self, tree):
        """Analyze the overall AST structure."""
        node_counts = defaultdict(int)
        depth_levels = defaultdict(int)
        
        def count_nodes(node, depth=0):
            node_type = type(node).__name__
            node_counts[node_type] += 1
            depth_levels[depth] += 1
            
            for child in ast.iter_child_nodes(node):
                count_nodes(child, depth + 1)
        
        count_nodes(tree)
        
        return {
            'node_counts': dict(node_counts),
            'max_depth': max(depth_levels.keys()) if depth_levels else 0,
            'total_nodes': sum(node_counts.values()),
            'depth_distribution': dict(depth_levels)
        }
    
    def _extract_imports(self, tree):
        """Extract detailed import information."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level
                
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'level': level,
                        'line': node.lineno
                    })
        
        return imports
    
    def _extract_exports(self, tree):
        """Extract exportable symbols (functions, classes)."""
        exports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    exports.append({
                        'type': 'function',
                        'name': node.name,
                        'line': node.lineno,
                        'args_count': len(node.args.args),
                        'has_decorators': bool(node.decorator_list),
                        'is_async': False
                    })
            elif isinstance(node, ast.AsyncFunctionDef):
                if not node.name.startswith('_'):
                    exports.append({
                        'type': 'function',
                        'name': node.name,
                        'line': node.lineno,
                        'args_count': len(node.args.args),
                        'has_decorators': bool(node.decorator_list),
                        'is_async': True
                    })
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    
                    exports.append({
                        'type': 'class',
                        'name': node.name,
                        'line': node.lineno,
                        'bases_count': len(node.bases),
                        'methods': methods,
                        'has_decorators': bool(node.decorator_list)
                    })
        
        return exports
    
    def _calculate_complexity_metrics(self, tree):
        """Calculate various complexity metrics."""
        complexity_metrics = {
            'cyclomatic_complexity': 1,  # Start at 1
            'nesting_depth': 0,
            'function_count': 0,
            'class_count': 0,
            'condition_count': 0,
            'loop_count': 0
        }
        
        def calculate_complexity(node, depth=0):
            complexity_metrics['nesting_depth'] = max(
                complexity_metrics['nesting_depth'], depth
            )
            
            node_type = type(node).__name__
            
            # Count different types of nodes
            if node_type in ['FunctionDef', 'AsyncFunctionDef']:
                complexity_metrics['function_count'] += 1
            elif node_type == 'ClassDef':
                complexity_metrics['class_count'] += 1
            elif node_type in ['If', 'While', 'For', 'Try', 'With']:
                complexity_metrics['cyclomatic_complexity'] += 1
                if node_type in ['If']:
                    complexity_metrics['condition_count'] += 1
                elif node_type in ['While', 'For']:
                    complexity_metrics['loop_count'] += 1
                    
            # Recursively process child nodes
            for child in ast.iter_child_nodes(node):
                calculate_complexity(child, depth + 1)
        
        calculate_complexity(tree)
        return complexity_metrics
    
    def _identify_code_patterns(self, tree):
        """Identify common code patterns."""
        patterns = {
            'design_patterns': [],
            'antipatterns': [],
            'best_practices': [],
            'code_smells': []
        }
        
        # Check for common patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for long parameter lists (code smell)
                if len(node.args.args) > 5:
                    patterns['code_smells'].append(f"Long parameter list in {node.name}")
                
                # Check for docstrings (best practice)
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    patterns['best_practices'].append(f"Documented function: {node.name}")
            
            elif isinstance(node, ast.ClassDef):
                # Check for singleton pattern (design pattern)
                method_names = [item.name for item in node.body 
                              if isinstance(item, ast.FunctionDef)]
                if '__new__' in method_names:
                    patterns['design_patterns'].append(f"Possible singleton: {node.name}")
        
        return patterns
    
    def _analyze_dependencies(self, tree):
        """Analyze code dependencies and coupling."""
        dependencies = {
            'internal_imports': [],
            'external_imports': [],
            'stdlib_imports': [],
            'dependency_depth': 0
        }
        
        # Python standard library modules (partial list)
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'random', 'collections',
            'pathlib', 'ast', 'threading', 'asyncio', 're', 'math', 'logging'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in stdlib_modules:
                        dependencies['stdlib_imports'].append(alias.name)
                    elif module.startswith('.'):
                        dependencies['internal_imports'].append(alias.name)
                    else:
                        dependencies['external_imports'].append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module in stdlib_modules:
                        dependencies['stdlib_imports'].append(node.module)
                    elif node.level > 0:  # Relative import
                        dependencies['internal_imports'].append(node.module or 'relative')
                    else:
                        dependencies['external_imports'].append(node.module)
        
        # Calculate dependency depth
        all_imports = (dependencies['internal_imports'] + 
                      dependencies['external_imports'] + 
                      dependencies['stdlib_imports'])
        dependencies['dependency_depth'] = len(all_imports)
        
        return dependencies
    
    def _assess_code_quality(self, tree, content):
        """Assess various code quality indicators."""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        
        quality_indicators = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'comment_lines': len(comment_lines),
            'comment_ratio': len(comment_lines) / max(len(non_empty_lines), 1),
            'avg_line_length': sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1),
            'docstring_count': 0,
            'has_main_guard': 'if __name__ == "__main__"' in content,
            'has_type_hints': 'typing' in content or '->' in content,
            'has_error_handling': any('try:' in line or 'except' in line for line in lines)
        }
        
        # Count docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    quality_indicators['docstring_count'] += 1
        
        return quality_indicators


class CodeStructureAnalyzer:
    """Advanced code structure analysis using AST."""
    
    def __init__(self):
        self.processor = ASTCodeProcessor()
        
    def analyze_codebase_structure(self, base_dir, max_files=None):
        """Analyze the structure of an entire codebase."""
        results = {
            'codebase_structure': {},
            'file_analyses': [],
            'global_metrics': {},
            'patterns_summary': {},
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Find Python files
        python_files = []
        base_path = Path(base_dir)
        
        if not base_path.exists():
            return results
        
        for root, dirs, files in os.walk(base_path):
            # Skip problematic directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'QUARANTINE', 'archive']]
            
            for file in files:
                if file.endswith('.py') and (max_files is None or len(python_files) < max_files):
                    if not any(skip in file for skip in ['original_', '_original', 'ARCHIVED', 'backup']):
                        python_files.append(Path(root) / file)
        
        # Analyze each file
        all_imports = []
        all_exports = []
        total_complexity = 0
        
        for py_file in python_files:
            try:
                analysis = self.processor.analyze_file_comprehensive(py_file)
                results['file_analyses'].append(analysis)
                
                if 'imports' in analysis:
                    all_imports.extend(analysis['imports'])
                if 'exports' in analysis:
                    all_exports.extend(analysis['exports'])
                if 'complexity_metrics' in analysis:
                    total_complexity += analysis['complexity_metrics'].get('cyclomatic_complexity', 0)
                    
            except Exception as e:
                continue
        
        # Calculate global metrics
        results['global_metrics'] = {
            'total_files_analyzed': len(results['file_analyses']),
            'total_imports': len(all_imports),
            'total_exports': len(all_exports),
            'average_complexity': total_complexity / max(len(results['file_analyses']), 1),
            'import_diversity': len(set(imp.get('module', '') for imp in all_imports)),
            'export_diversity': len(set(exp.get('name', '') for exp in all_exports))
        }
        
        return results


# Factory functions for integration
def create_ast_processor():
    """Factory function to create AST code processor."""
    return ASTCodeProcessor()

def create_structure_analyzer():
    """Factory function to create code structure analyzer."""
    return CodeStructureAnalyzer()

# Global instances for Flask integration
ast_processor = ASTCodeProcessor()
structure_analyzer = CodeStructureAnalyzer()

# Integration functions for dashboard
def analyze_file_quick_endpoint(file_path):
    """Quick file analysis endpoint for dashboard integration."""
    return ast_processor.analyze_file_quick(file_path)

def analyze_file_comprehensive_endpoint(file_path):
    """Comprehensive file analysis endpoint for dashboard integration."""
    return ast_processor.analyze_file_comprehensive(file_path)

def get_ast_processing_status():
    """Get AST processing system status."""
    return {
        'processor_status': 'active',
        'supported_node_types': len(ast_processor.supported_node_types),
        'analysis_capabilities': [
            'Quick import/export extraction',
            'Comprehensive AST structure analysis',
            'Complexity metrics calculation',
            'Code pattern identification',
            'Dependency analysis',
            'Code quality assessment'
        ],
        'cache_size': len(ast_processor.analysis_cache)
    }

def get_codebase_structure_summary(base_dir="../TestMaster", max_files=100):
    """Get summary of codebase structure for dashboard."""
    analysis = structure_analyzer.analyze_codebase_structure(base_dir, max_files)
    return {
        'files_analyzed': analysis['global_metrics'].get('total_files_analyzed', 0),
        'total_imports': analysis['global_metrics'].get('total_imports', 0),
        'total_exports': analysis['global_metrics'].get('total_exports', 0),
        'average_complexity': round(analysis['global_metrics'].get('average_complexity', 0), 1),
        'analysis_timestamp': analysis['analysis_timestamp']
    }