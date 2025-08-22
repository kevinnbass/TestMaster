#!/usr/bin/env python3
"""
Dependency Analyzer
==================

Analyzes dependencies between Python files in your codebase
to help you understand how files connect to each other.
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

class DependencyAnalyzer:
    """Analyzes dependencies between Python files"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.dependencies = defaultdict(set)
        self.reverse_dependencies = defaultdict(set)
        self.local_imports = defaultdict(set)
        self.external_imports = defaultdict(set)
    
    def analyze_dependencies(self):
        """Analyze all dependencies in the codebase"""
        print("Dependency Analysis")
        print("=" * 30)
        
        python_files = list(self.root_path.glob("*.py"))
        print(f"Analyzing {len(python_files)} Python files...")
        print()
        
        for file_path in python_files:
            self.analyze_file(file_path)
        
        self.print_results()
        return self.get_results()
    
    def analyze_file(self, file_path: Path):
        """Analyze dependencies for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                self.extract_dependencies(tree, file_path)
            except SyntaxError:
                print(f"Syntax error in {file_path.name}")
                
        except Exception as e:
            print(f"Error analyzing {file_path.name}: {e}")
    
    def extract_dependencies(self, tree: ast.AST, file_path: Path):
        """Extract import dependencies from AST"""
        file_name = file_path.name
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    if self.is_local_import(import_name):
                        self.local_imports[file_name].add(import_name)
                        self.dependencies[file_name].add(import_name)
                    else:
                        self.external_imports[file_name].add(import_name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_name = node.module
                    if self.is_local_import(import_name):
                        self.local_imports[file_name].add(import_name)
                        self.dependencies[file_name].add(import_name)
                        
                        # Build reverse dependencies
                        target_file = f"{import_name}.py"
                        if (self.root_path / target_file).exists():
                            self.reverse_dependencies[target_file].add(file_name)
                    else:
                        self.external_imports[file_name].add(import_name)
    
    def is_local_import(self, import_name: str) -> bool:
        """Check if an import refers to a local file"""
        # Simple heuristic: if a .py file with that name exists locally
        potential_file = self.root_path / f"{import_name}.py"
        return potential_file.exists()
    
    def print_results(self):
        """Print dependency analysis results"""
        # Files with most dependencies
        print("Files with most local dependencies:")
        print("-" * 35)
        
        local_dep_counts = {
            file: len(deps) for file, deps in self.local_imports.items()
        }
        
        sorted_deps = sorted(local_dep_counts.items(), key=lambda x: x[1], reverse=True)
        
        for file, count in sorted_deps[:10]:
            if count > 0:
                deps = sorted(self.local_imports[file])
                print(f"{file}: {count} dependencies")
                print(f"  -> {', '.join(deps)}")
                print()
        
        # Files that are most depended upon
        print("Files that other files depend on most:")
        print("-" * 38)
        
        reverse_dep_counts = {
            file: len(deps) for file, deps in self.reverse_dependencies.items()
        }
        
        sorted_reverse = sorted(reverse_dep_counts.items(), key=lambda x: x[1], reverse=True)
        
        for file, count in sorted_reverse[:10]:
            if count > 0:
                dependents = sorted(self.reverse_dependencies[file])
                print(f"{file}: used by {count} files")
                print(f"  <- {', '.join(dependents)}")
                print()
        
        # Most common external dependencies
        print("Most common external dependencies:")
        print("-" * 33)
        
        all_external = []
        for deps in self.external_imports.values():
            all_external.extend(deps)
        
        external_counts = defaultdict(int)
        for dep in all_external:
            external_counts[dep] += 1
        
        sorted_external = sorted(external_counts.items(), key=lambda x: x[1], reverse=True)
        
        for dep, count in sorted_external[:15]:
            print(f"  {dep}: {count} files")
        
        print()
        
        # Circular dependencies
        self.find_circular_dependencies()
    
    def find_circular_dependencies(self):
        """Find potential circular dependencies"""
        print("Checking for circular dependencies:")
        print("-" * 32)
        
        circular_found = False
        
        for file_a in self.local_imports:
            for dep in self.local_imports[file_a]:
                dep_file = f"{dep}.py"
                if dep_file in self.local_imports:
                    # Check if dep_file imports back to file_a
                    file_a_module = file_a.replace('.py', '')
                    if file_a_module in self.local_imports[dep_file]:
                        print(f"  Circular: {file_a} <-> {dep_file}")
                        circular_found = True
        
        if not circular_found:
            print("  No circular dependencies found")
        
        print()
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results as dictionary"""
        return {
            'local_dependencies': dict(self.local_imports),
            'external_dependencies': dict(self.external_imports),
            'reverse_dependencies': dict(self.reverse_dependencies),
            'dependency_counts': {
                file: len(deps) for file, deps in self.local_imports.items()
            }
        }
    
    def suggest_refactoring(self):
        """Suggest potential refactoring opportunities"""
        print("Refactoring suggestions:")
        print("-" * 22)
        
        # Files with too many dependencies
        high_dep_files = [
            (file, len(deps)) for file, deps in self.local_imports.items()
            if len(deps) > 5
        ]
        
        if high_dep_files:
            print("Files with many dependencies (consider breaking down):")
            for file, count in sorted(high_dep_files, key=lambda x: x[1], reverse=True):
                print(f"  {file}: {count} dependencies")
        
        # Central files that many others depend on
        central_files = [
            (file, len(deps)) for file, deps in self.reverse_dependencies.items()
            if len(deps) > 3
        ]
        
        if central_files:
            print("\nCentral files (many files depend on these):")
            for file, count in sorted(central_files, key=lambda x: x[1], reverse=True):
                print(f"  {file}: used by {count} files")
                print(f"    Consider: ensure this file has clear, stable interface")
        
        # Isolated files
        isolated_files = []
        for file in self.root_path.glob("*.py"):
            file_name = file.name
            if (len(self.local_imports.get(file_name, [])) == 0 and 
                len(self.reverse_dependencies.get(file_name, [])) == 0):
                isolated_files.append(file_name)
        
        if isolated_files:
            print(f"\nIsolated files (no local dependencies):")
            for file in isolated_files[:10]:
                print(f"  {file}")
            print("  Consider: these might be standalone scripts or utilities")

def main():
    """Main analysis function"""
    print("Personal Codebase Dependency Analyzer")
    print("Understand how your files connect to each other")
    print()
    
    analyzer = DependencyAnalyzer()
    results = analyzer.analyze_dependencies()
    analyzer.suggest_refactoring()
    
    # Save results
    output_file = "dependency_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()