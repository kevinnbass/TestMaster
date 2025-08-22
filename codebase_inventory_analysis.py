#!/usr/bin/env python3
"""
TestMaster Codebase Inventory and Documentation Analysis
========================================================

Comprehensive analysis script for Agent B to begin documentation work.
Provides detailed inventory of Python files, functions, classes, exports,
and documentation coverage statistics.

Agent: Claude (Agent B Analysis Support)
Date: 2025-08-21
"""

import os
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import traceback

class CodebaseAnalyzer:
    """Comprehensive codebase analysis for documentation assessment."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.results = {
            'summary': {},
            'files': {},
            'documentation_coverage': {},
            'test_files': {},
            'exports': {},
            'functions': {},
            'classes': {},
            'errors': []
        }
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for structure and documentation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return {
                    'error': f'Syntax error: {e}',
                    'lines': len(content.splitlines()),
                    'functions': 0,
                    'classes': 0,
                    'exports': [],
                    'has_module_docstring': False,
                    'documentation_score': 0
                }
            
            # Analyze structure
            functions = []
            classes = []
            imports = []
            exports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'is_private': node.name.startswith('_'),
                        'has_docstring': ast.get_docstring(node) is not None,
                        'args_count': len(node.args.args),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    })
                elif isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                'name': item.name,
                                'has_docstring': ast.get_docstring(item) is not None,
                                'is_private': item.name.startswith('_')
                            })
                    
                    classes.append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'has_docstring': ast.get_docstring(node) is not None,
                        'methods': methods,
                        'method_count': len(methods)
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    else:
                        module = node.module or ''
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}")
                            
            # Check for __all__ exports
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            if isinstance(node.value, ast.List):
                                for item in node.value.elts:
                                    if isinstance(item, ast.Str):
                                        exports.append(item.s)
                                    elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                                        exports.append(item.value)
            
            # Module docstring
            module_docstring = ast.get_docstring(tree)
            has_module_docstring = module_docstring is not None
            
            # Calculate documentation score
            total_documentable = len(functions) + len(classes) + 1  # +1 for module
            documented = (1 if has_module_docstring else 0)
            documented += sum(1 for f in functions if f['has_docstring'])
            documented += sum(1 for c in classes if c['has_docstring'])
            
            # Add method documentation
            for cls in classes:
                total_documentable += len(cls['methods'])
                documented += sum(1 for m in cls['methods'] if m['has_docstring'])
            
            documentation_score = (documented / total_documentable * 100) if total_documentable > 0 else 0
            
            return {
                'lines': len(content.splitlines()),
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'exports': exports,
                'function_count': len(functions),
                'class_count': len(classes),
                'import_count': len(imports),
                'export_count': len(exports),
                'has_module_docstring': has_module_docstring,
                'module_docstring': module_docstring[:200] + '...' if module_docstring and len(module_docstring) > 200 else module_docstring,
                'documentation_score': round(documentation_score, 2),
                'total_documentable': total_documentable,
                'documented_items': documented
            }
            
        except Exception as e:
            return {
                'error': f'Analysis error: {e}',
                'traceback': traceback.format_exc()
            }
    
    def is_test_file(self, file_path: Path) -> bool:
        """Determine if a file is a test file."""
        name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        test_indicators = [
            name.startswith('test_'),
            name.endswith('_test.py'),
            'test' in file_path.parts,
            'tests' in file_path.parts,
            '/test/' in path_str,
            '/tests/' in path_str
        ]
        
        return any(test_indicators)
    
    def should_exclude_path(self, file_path: Path) -> bool:
        """Check if path should be excluded from analysis."""
        path_str = str(file_path).lower()
        exclude_patterns = [
            'archive',
            'backup',
            '__pycache__',
            '.git',
            'htmlcov',
            'coverage',
            '.backup',
            'deprecated',
            'legacy_scripts',
            'original_backup'
        ]
        
        return any(pattern in path_str for pattern in exclude_patterns)
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive codebase analysis."""
        print("Starting comprehensive codebase analysis...")
        
        # Find all Python files
        python_files = []
        test_files = []
        
        for py_file in self.root_path.rglob("*.py"):
            if self.should_exclude_path(py_file):
                continue
                
            if self.is_test_file(py_file):
                test_files.append(py_file)
            else:
                python_files.append(py_file)
        
        print(f"Found {len(python_files)} source files and {len(test_files)} test files")
        
        # Analyze each file
        total_lines = 0
        total_functions = 0
        total_classes = 0
        total_exports = 0
        documentation_scores = []
        
        for file_path in python_files:
            print(f"Analyzing: {file_path.relative_to(self.root_path)}")
            analysis = self.analyze_file(file_path)
            
            rel_path = str(file_path.relative_to(self.root_path))
            self.results['files'][rel_path] = analysis
            
            if 'error' not in analysis:
                total_lines += analysis['lines']
                total_functions += analysis['function_count']
                total_classes += analysis['class_count']
                total_exports += analysis['export_count']
                documentation_scores.append(analysis['documentation_score'])
        
        # Analyze test files
        test_coverage_info = {}
        for test_file in test_files:
            print(f"Analyzing test file: {test_file.relative_to(self.root_path)}")
            analysis = self.analyze_file(test_file)
            rel_path = str(test_file.relative_to(self.root_path))
            test_coverage_info[rel_path] = analysis
        
        # Calculate summary statistics
        avg_doc_score = sum(documentation_scores) / len(documentation_scores) if documentation_scores else 0
        
        self.results['summary'] = {
            'total_source_files': len(python_files),
            'total_test_files': len(test_files),
            'total_lines_of_code': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'total_exports': total_exports,
            'average_documentation_score': round(avg_doc_score, 2),
            'files_with_no_documentation': len([f for f in self.results['files'].values() 
                                               if f.get('documentation_score', 0) == 0]),
            'files_with_full_documentation': len([f for f in self.results['files'].values() 
                                                  if f.get('documentation_score', 0) == 100]),
            'files_needing_documentation': len([f for f in self.results['files'].values() 
                                                if f.get('documentation_score', 0) < 80])
        }
        
        self.results['test_files'] = test_coverage_info
        
        # Generate documentation recommendations
        self.generate_documentation_recommendations()
        
        return self.results
    
    def generate_documentation_recommendations(self):
        """Generate specific documentation recommendations."""
        recommendations = []
        
        for file_path, analysis in self.results['files'].items():
            if 'error' in analysis:
                continue
                
            file_recommendations = []
            
            # Module documentation
            if not analysis['has_module_docstring']:
                file_recommendations.append("Add module docstring")
            
            # Function documentation
            undocumented_functions = [f for f in analysis['functions'] if not f['has_docstring'] and not f['is_private']]
            if undocumented_functions:
                file_recommendations.append(f"Document {len(undocumented_functions)} public functions")
            
            # Class documentation
            undocumented_classes = [c for c in analysis['classes'] if not c['has_docstring']]
            if undocumented_classes:
                file_recommendations.append(f"Document {len(undocumented_classes)} classes")
            
            # Method documentation
            for cls in analysis['classes']:
                undocumented_methods = [m for m in cls['methods'] if not m['has_docstring'] and not m['is_private']]
                if undocumented_methods:
                    file_recommendations.append(f"Document {len(undocumented_methods)} methods in class {cls['name']}")
            
            if file_recommendations:
                recommendations.append({
                    'file': file_path,
                    'score': analysis['documentation_score'],
                    'recommendations': file_recommendations
                })
        
        # Sort by priority (lowest score first)
        recommendations.sort(key=lambda x: x['score'])
        
        self.results['documentation_recommendations'] = recommendations[:20]  # Top 20 priorities
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        summary = self.results['summary']
        
        report = f"""
TestMaster Codebase Analysis Report
==================================
Generated: 2025-08-21
Analyzer: Agent B Documentation Support

EXECUTIVE SUMMARY
================
Total Source Files: {summary['total_source_files']}
Total Test Files: {summary['total_test_files']}
Total Lines of Code: {summary['total_lines_of_code']:,}
Total Functions: {summary['total_functions']}
Total Classes: {summary['total_classes']}
Total Exports: {summary['total_exports']}

DOCUMENTATION COVERAGE
======================
Average Documentation Score: {summary['average_documentation_score']:.1f}%
Files with No Documentation: {summary['files_with_no_documentation']}
Files with Full Documentation: {summary['files_with_full_documentation']}
Files Needing Documentation: {summary['files_needing_documentation']}

TOP PRIORITY FILES FOR DOCUMENTATION
===================================
"""
        
        for i, rec in enumerate(self.results.get('documentation_recommendations', [])[:10], 1):
            report += f"{i:2d}. {rec['file']} (Score: {rec['score']:.1f}%)\n"
            for suggestion in rec['recommendations']:
                report += f"    - {suggestion}\n"
            report += "\n"
        
        report += """
DETAILED FILE BREAKDOWN
======================
"""
        
        # Group files by directory
        by_directory = defaultdict(list)
        for file_path, analysis in self.results['files'].items():
            directory = str(Path(file_path).parent)
            by_directory[directory].append((file_path, analysis))
        
        for directory in sorted(by_directory.keys()):
            report += f"\nDirectory: {directory}\n"
            report += "-" * (len(directory) + 11) + "\n"
            
            for file_path, analysis in sorted(by_directory[directory]):
                if 'error' in analysis:
                    report += f"  {Path(file_path).name}: ERROR - {analysis['error']}\n"
                    continue
                    
                filename = Path(file_path).name
                report += f"  {filename}:\n"
                report += f"    Lines: {analysis['lines']}\n"
                report += f"    Functions: {analysis['function_count']}\n"
                report += f"    Classes: {analysis['class_count']}\n"
                report += f"    Exports: {analysis['export_count']}\n"
                report += f"    Documentation: {analysis['documentation_score']:.1f}%\n"
                
                if analysis['has_module_docstring']:
                    report += f"    Module Doc: YES\n"
                else:
                    report += f"    Module Doc: MISSING\n"
        
        return report

def main():
    """Main analysis execution."""
    testmaster_path = r"C:\Users\kbass\OneDrive\Documents\testmaster\TestMaster"
    
    analyzer = CodebaseAnalyzer(testmaster_path)
    results = analyzer.analyze_codebase()
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save results
    with open('codebase_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open('codebase_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(report)
    print("\nDetailed results saved to:")
    print("- codebase_analysis_results.json")
    print("- codebase_analysis_report.txt")

if __name__ == "__main__":
    main()