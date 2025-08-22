#!/usr/bin/env python3
"""
Focused Codebase Analyzer
=========================

Analyzes just the main files in the testmaster directory
to help you understand your core codebase structure.
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

def analyze_core_files():
    """Analyze the core files in the current directory"""
    
    print("Focused Codebase Analysis")
    print("=" * 40)
    
    # Look for Python files in the current directory only
    current_dir = Path.cwd()
    python_files = [f for f in current_dir.glob("*.py") if f.is_file()]
    
    print(f"Found {len(python_files)} Python files in current directory")
    print()
    
    file_stats = {}
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            stats = {
                'lines': len(lines),
                'non_empty_lines': len(non_empty_lines),
                'size_kb': len(content) / 1024,
                'functions': [],
                'classes': [],
                'imports': []
            }
            
            # Try to parse the file
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        stats['functions'].append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        stats['classes'].append(node.name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            stats['imports'].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            stats['imports'].append(node.module)
                            
            except SyntaxError:
                print(f"  Syntax error in {file_path.name}")
            
            file_stats[file_path.name] = stats
            
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")
    
    # Display results
    print("File Analysis:")
    print("-" * 40)
    
    # Sort by line count
    sorted_files = sorted(file_stats.items(), key=lambda x: x[1]['lines'], reverse=True)
    
    for filename, stats in sorted_files:
        print(f"{filename}:")
        print(f"  Lines: {stats['lines']} ({stats['non_empty_lines']} non-empty)")
        print(f"  Size: {stats['size_kb']:.1f} KB")
        print(f"  Functions: {len(stats['functions'])}")
        print(f"  Classes: {len(stats['classes'])}")
        print(f"  Imports: {len(set(stats['imports']))}")
        
        # Show main classes and functions if not too many
        if len(stats['classes']) <= 5 and stats['classes']:
            print(f"  Main classes: {', '.join(stats['classes'])}")
        if len(stats['functions']) <= 8 and stats['functions']:
            print(f"  Main functions: {', '.join(stats['functions'][:8])}")
        
        print()
    
    # Summary
    total_lines = sum(stats['lines'] for stats in file_stats.values())
    total_functions = sum(len(stats['functions']) for stats in file_stats.values())
    total_classes = sum(len(stats['classes']) for stats in file_stats.values())
    
    print("Summary:")
    print("-" * 40)
    print(f"Total files: {len(file_stats)}")
    print(f"Total lines: {total_lines:,}")
    print(f"Total functions: {total_functions}")
    print(f"Total classes: {total_classes}")
    print(f"Average lines per file: {total_lines / len(file_stats) if file_stats else 0:.1f}")
    
    # Identify largest files
    large_files = [name for name, stats in file_stats.items() if stats['lines'] > 300]
    if large_files:
        print(f"\nLarger files (>300 lines):")
        for filename in large_files:
            lines = file_stats[filename]['lines']
            print(f"  {filename}: {lines} lines")
    
    # Most common imports
    all_imports = []
    for stats in file_stats.values():
        all_imports.extend(stats['imports'])
    
    if all_imports:
        import_counts = defaultdict(int)
        for imp in all_imports:
            import_counts[imp] += 1
        
        common_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nMost common imports:")
        for imp, count in common_imports:
            print(f"  {imp}: {count} files")
    
    return file_stats

if __name__ == "__main__":
    analyze_core_files()