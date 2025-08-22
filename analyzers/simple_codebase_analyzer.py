#!/usr/bin/env python3
"""
Simple Codebase Analyzer
========================

A straightforward tool to help you understand your codebase better.
Analyzes file structure, dependencies, and provides useful insights.
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

class CodebaseAnalyzer:
    """Simple analyzer to understand your codebase structure"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.files = []
        self.dependencies = defaultdict(set)
        self.file_stats = {}
        self.analysis_results = {}
    
    def analyze(self):
        """Run complete codebase analysis"""
        print(f"Analyzing codebase: {self.root_path}")
        print("=" * 50)
        
        # Find all Python files
        self.find_python_files()
        
        # Analyze each file
        for file_path in self.files:
            self.analyze_file(file_path)
        
        # Generate summary
        self.generate_summary()
        
        return self.analysis_results
    
    def find_python_files(self):
        """Find all Python files in the codebase"""
        for file_path in self.root_path.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):
                self.files.append(file_path)
        
        print(f"Found {len(self.files)} Python files")
    
    def analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic stats
            lines = content.split('\n')
            self.file_stats[str(file_path)] = {
                'lines': len(lines),
                'size_bytes': len(content),
                'imports': [],
                'functions': [],
                'classes': []
            }
            
            # Parse AST for more details
            try:
                tree = ast.parse(content)
                self.extract_imports_and_structure(tree, file_path)
            except SyntaxError:
                print(f"Syntax error in {file_path}")
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def extract_imports_and_structure(self, tree: ast.AST, file_path: Path):
        """Extract imports, functions, and classes from AST"""
        stats = self.file_stats[str(file_path)]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    stats['imports'].append(alias.name)
                    self.dependencies[str(file_path)].add(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    stats['imports'].append(node.module)
                    self.dependencies[str(file_path)].add(node.module)
            
            elif isinstance(node, ast.FunctionDef):
                stats['functions'].append(node.name)
            
            elif isinstance(node, ast.ClassDef):
                stats['classes'].append(node.name)
    
    def generate_summary(self):
        """Generate analysis summary"""
        total_lines = sum(stats['lines'] for stats in self.file_stats.values())
        total_files = len(self.file_stats)
        
        # Largest files
        largest_files = sorted(
            self.file_stats.items(), 
            key=lambda x: x[1]['lines'], 
            reverse=True
        )[:10]
        
        # Most common imports
        all_imports = []
        for stats in self.file_stats.values():
            all_imports.extend(stats['imports'])
        
        import_counts = defaultdict(int)
        for imp in all_imports:
            import_counts[imp] += 1
        
        common_imports = sorted(
            import_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Files with most functions/classes
        most_functions = sorted(
            self.file_stats.items(),
            key=lambda x: len(x[1]['functions']),
            reverse=True
        )[:5]
        
        most_classes = sorted(
            self.file_stats.items(),
            key=lambda x: len(x[1]['classes']),
            reverse=True
        )[:5]
        
        self.analysis_results = {
            'summary': {
                'total_files': total_files,
                'total_lines': total_lines,
                'average_lines_per_file': total_lines / total_files if total_files > 0 else 0
            },
            'largest_files': largest_files,
            'common_imports': common_imports,
            'most_functions': most_functions,
            'most_classes': most_classes
        }
        
        self.print_summary()
    
    def print_summary(self):
        """Print analysis summary"""
        results = self.analysis_results
        
        print(f"\nCodebase Summary:")
        print(f"  Total files: {results['summary']['total_files']}")
        print(f"  Total lines: {results['summary']['total_lines']:,}")
        print(f"  Average lines per file: {results['summary']['average_lines_per_file']:.1f}")
        
        print(f"\nLargest files:")
        for file_path, stats in results['largest_files']:
            rel_path = Path(file_path).relative_to(self.root_path)
            print(f"  {rel_path}: {stats['lines']} lines")
        
        print(f"\nMost common imports:")
        for import_name, count in results['common_imports']:
            print(f"  {import_name}: {count} files")
        
        print(f"\nFiles with most functions:")
        for file_path, stats in results['most_functions']:
            rel_path = Path(file_path).relative_to(self.root_path)
            print(f"  {rel_path}: {len(stats['functions'])} functions")
        
        print(f"\nFiles with most classes:")
        for file_path, stats in results['most_classes']:
            rel_path = Path(file_path).relative_to(self.root_path)
            print(f"  {rel_path}: {len(stats['classes'])} classes")
    
    def save_results(self, output_file: str = "codebase_analysis.json"):
        """Save analysis results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"\nAnalysis saved to {output_file}")
    
    def find_potential_issues(self):
        """Find potential issues in the codebase"""
        issues = []
        
        # Very large files
        for file_path, stats in self.file_stats.items():
            if stats['lines'] > 500:
                rel_path = Path(file_path).relative_to(self.root_path)
                issues.append(f"Large file: {rel_path} ({stats['lines']} lines)")
        
        # Files with many functions
        for file_path, stats in self.file_stats.items():
            if len(stats['functions']) > 20:
                rel_path = Path(file_path).relative_to(self.root_path)
                issues.append(f"Many functions: {rel_path} ({len(stats['functions'])} functions)")
        
        if issues:
            print(f"\nPotential issues to consider:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\nNo obvious issues found!")
    
    def suggest_improvements(self):
        """Suggest potential improvements"""
        suggestions = []
        
        # Check for very large files
        large_files = [path for path, stats in self.file_stats.items() if stats['lines'] > 300]
        if large_files:
            suggestions.append(f"Consider breaking down {len(large_files)} large files (>300 lines)")
        
        # Check for files with many functions
        function_heavy = [path for path, stats in self.file_stats.items() if len(stats['functions']) > 15]
        if function_heavy:
            suggestions.append(f"Consider organizing {len(function_heavy)} files with many functions")
        
        if suggestions:
            print(f"\nSuggestions:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")


def main():
    """Main analysis function"""
    # Analyze current directory by default
    current_dir = Path.cwd()
    
    print("Simple Codebase Analyzer")
    print("For understanding your Python codebase")
    print()
    
    analyzer = CodebaseAnalyzer(current_dir)
    analyzer.analyze()
    analyzer.find_potential_issues()
    analyzer.suggest_improvements()
    analyzer.save_results()


if __name__ == "__main__":
    main()