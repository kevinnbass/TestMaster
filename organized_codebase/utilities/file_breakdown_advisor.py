#!/usr/bin/env python3
"""
File Breakdown Advisor
=====================

Analyzes large Python files and suggests specific ways to break them down
into smaller, more manageable pieces.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

class FileBreakdownAdvisor:
    """Analyzes files and suggests breakdown strategies"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_large_files(self, min_lines: int = 200):
        """Analyze files that are larger than min_lines"""
        print(f"File Breakdown Analysis")
        print(f"Analyzing files with {min_lines}+ lines")
        print("=" * 40)
        
        current_dir = Path.cwd()
        python_files = [f for f in current_dir.glob("*.py") if f.is_file()]
        
        large_files = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                if len(lines) >= min_lines:
                    analysis = self.analyze_file_structure(file_path, content)
                    if analysis:
                        large_files.append((file_path, len(lines), analysis))
            
            except Exception as e:
                print(f"Error analyzing {file_path.name}: {e}")
        
        # Sort by line count
        large_files.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(large_files)} files with {min_lines}+ lines\n")
        
        for file_path, line_count, analysis in large_files:
            self.print_breakdown_suggestions(file_path, line_count, analysis)
            self.analysis_results[str(file_path)] = analysis
        
        return self.analysis_results
    
    def analyze_file_structure(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze the internal structure of a file"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None
        
        analysis = {
            'classes': [],
            'functions': [],
            'imports': [],
            'constants': [],
            'breakdown_suggestions': []
        }
        
        # Analyze top-level nodes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self.analyze_class(node, content)
                analysis['classes'].append(class_info)
                
            elif isinstance(node, ast.FunctionDef):
                func_info = self.analyze_function(node, content)
                analysis['functions'].append(func_info)
                
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                analysis['imports'].append(self.get_import_info(node))
                
            elif isinstance(node, ast.Assign):
                # Look for constants (uppercase variables)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        analysis['constants'].append(target.id)
        
        # Generate breakdown suggestions
        analysis['breakdown_suggestions'] = self.generate_suggestions(analysis)
        
        return analysis
    
    def analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze a class definition"""
        lines = content.split('\n')
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
        
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_lines = (item.end_lineno or item.lineno) - item.lineno + 1
                methods.append({
                    'name': item.name,
                    'lines': method_lines,
                    'is_private': item.name.startswith('_')
                })
        
        return {
            'name': node.name,
            'lines': end_line - start_line + 1,
            'methods': methods,
            'method_count': len(methods),
            'start_line': start_line,
            'end_line': end_line
        }
    
    def analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze a function definition"""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
        
        return {
            'name': node.name,
            'lines': end_line - start_line + 1,
            'start_line': start_line,
            'end_line': end_line,
            'is_private': node.name.startswith('_')
        }
    
    def get_import_info(self, node) -> str:
        """Get import information"""
        if isinstance(node, ast.Import):
            return f"import {', '.join([alias.name for alias in node.names])}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ', '.join([alias.name for alias in node.names])
            return f"from {module} import {names}"
        return ""
    
    def generate_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific breakdown suggestions"""
        suggestions = []
        
        # Large classes
        large_classes = [cls for cls in analysis['classes'] if cls['lines'] > 100]
        for cls in large_classes:
            if cls['method_count'] > 10:
                suggestions.append({
                    'type': 'split_class',
                    'target': cls['name'],
                    'reason': f"Large class with {cls['method_count']} methods ({cls['lines']} lines)",
                    'suggestion': f"Consider splitting {cls['name']} into multiple classes based on responsibility"
                })
        
        # Many top-level functions
        if len(analysis['functions']) > 8:
            related_functions = self.group_related_functions(analysis['functions'])
            if related_functions:
                suggestions.append({
                    'type': 'group_functions',
                    'target': 'top_level_functions',
                    'reason': f"{len(analysis['functions'])} top-level functions",
                    'suggestion': "Consider grouping related functions into classes or separate modules",
                    'groups': related_functions
                })
        
        # Long functions
        long_functions = [func for func in analysis['functions'] if func['lines'] > 50]
        for func in long_functions:
            suggestions.append({
                'type': 'split_function',
                'target': func['name'],
                'reason': f"Long function ({func['lines']} lines)",
                'suggestion': f"Consider breaking {func['name']} into smaller functions"
            })
        
        # Many imports suggest potential for splitting
        if len(analysis['imports']) > 15:
            suggestions.append({
                'type': 'review_imports',
                'target': 'imports',
                'reason': f"Many imports ({len(analysis['imports'])})",
                'suggestion': "High import count suggests file might have multiple responsibilities"
            })
        
        return suggestions
    
    def group_related_functions(self, functions: List[Dict[str, Any]]) -> List[List[str]]:
        """Try to group functions by naming patterns"""
        groups = defaultdict(list)
        
        for func in functions:
            name = func['name']
            # Group by common prefixes
            if '_' in name:
                prefix = name.split('_')[0]
                if prefix and len(prefix) > 2:
                    groups[prefix].append(name)
        
        # Only return groups with 2+ functions
        return [group for group in groups.values() if len(group) >= 2]
    
    def print_breakdown_suggestions(self, file_path: Path, line_count: int, analysis: Dict[str, Any]):
        """Print breakdown suggestions for a file"""
        print(f"{file_path.name} ({line_count} lines)")
        print("-" * (len(file_path.name) + 15))
        
        print(f"Structure:")
        print(f"  Classes: {len(analysis['classes'])}")
        print(f"  Functions: {len(analysis['functions'])}")
        print(f"  Imports: {len(analysis['imports'])}")
        print(f"  Constants: {len(analysis['constants'])}")
        print()
        
        # Show largest components
        if analysis['classes']:
            print("Largest classes:")
            sorted_classes = sorted(analysis['classes'], key=lambda x: x['lines'], reverse=True)
            for cls in sorted_classes[:3]:
                print(f"  {cls['name']}: {cls['lines']} lines, {cls['method_count']} methods")
            print()
        
        if analysis['functions']:
            large_funcs = [f for f in analysis['functions'] if f['lines'] > 20]
            if large_funcs:
                print("Largest functions:")
                sorted_funcs = sorted(large_funcs, key=lambda x: x['lines'], reverse=True)
                for func in sorted_funcs[:3]:
                    print(f"  {func['name']}: {func['lines']} lines")
                print()
        
        # Show breakdown suggestions
        if analysis['breakdown_suggestions']:
            print("Breakdown suggestions:")
            for suggestion in analysis['breakdown_suggestions']:
                print(f"  • {suggestion['suggestion']}")
                print(f"    Reason: {suggestion['reason']}")
                if 'groups' in suggestion:
                    for group in suggestion['groups']:
                        print(f"    Related functions: {', '.join(group)}")
                print()
        else:
            print("No specific breakdown suggestions - file structure looks reasonable.")
            print()
        
        print("-" * 50)
        print()
    
    def save_results(self, filename: str = "breakdown_analysis.json"):
        """Save analysis results to file"""
        with open(filename, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"Analysis saved to {filename}")

def main():
    """Main analysis function"""
    print("Personal File Breakdown Advisor")
    print("Suggestions for breaking down large files")
    print()
    
    advisor = FileBreakdownAdvisor()
    advisor.analyze_large_files(min_lines=200)
    advisor.save_results()

if __name__ == "__main__":
    main()