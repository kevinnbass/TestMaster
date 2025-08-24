"""
Agent A - Export Inventory Scanner
Phase 1: Hours 11-15 - Complete Export Inventory Creation
"""

import ast
import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import traceback

class ExportScanner:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.exports = defaultdict(list)
        self.statistics = {
            "total_files": 0,
            "scanned_files": 0,
            "failed_files": 0,
            "total_functions": 0,
            "total_classes": 0,
            "total_constants": 0,
            "total_variables": 0
        }
        
    def scan_directory(self, directory: str, max_files: int = 100) -> Dict:
        """Scan directory for Python files and extract exports"""
        dir_path = self.root_path / directory
        python_files = list(dir_path.rglob("*.py"))[:max_files]
        
        print(f"Scanning {len(python_files)} Python files in {directory}...")
        
        for file_path in python_files:
            self.statistics["total_files"] += 1
            try:
                self.extract_exports(file_path)
                self.statistics["scanned_files"] += 1
            except Exception as e:
                self.statistics["failed_files"] += 1
                print(f"Failed to parse {file_path}: {e}")
                
        return self.generate_report()
    
    def extract_exports(self, file_path: Path) -> None:
        """Extract all exports from a Python file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                content = f.read()
                tree = ast.parse(content)
            except:
                return
                
        relative_path = str(file_path.relative_to(self.root_path))
        
        for node in ast.walk(tree):
            # Extract functions
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    export_data = {
                        'type': 'function',
                        'name': node.name,
                        'file': relative_path,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': self._get_decorators(node),
                        'docstring': ast.get_docstring(node) or ""
                    }
                    self.exports['functions'].append(export_data)
                    self.statistics["total_functions"] += 1
                    
            # Extract classes
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                            
                    export_data = {
                        'type': 'class',
                        'name': node.name,
                        'file': relative_path,
                        'line': node.lineno,
                        'bases': self._get_bases(node),
                        'methods': methods,
                        'docstring': ast.get_docstring(node) or ""
                    }
                    self.exports['classes'].append(export_data)
                    self.statistics["total_classes"] += 1
                    
            # Extract module-level constants
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():
                            export_data = {
                                'type': 'constant',
                                'name': target.id,
                                'file': relative_path,
                                'line': node.lineno
                            }
                            self.exports['constants'].append(export_data)
                            self.statistics["total_constants"] += 1
                        elif not target.id.startswith('_'):
                            export_data = {
                                'type': 'variable',
                                'name': target.id,
                                'file': relative_path,
                                'line': node.lineno
                            }
                            self.exports['variables'].append(export_data)
                            self.statistics["total_variables"] += 1
    
    def _get_decorators(self, node) -> List[str]:
        """Extract decorator names from a node"""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)
        return decorators
    
    def _get_bases(self, node) -> List[str]:
        """Extract base class names"""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id if hasattr(base.value, 'id') else '?'}.{base.attr}")
        return bases
    
    def generate_report(self) -> Dict:
        """Generate export inventory report"""
        return {
            "statistics": self.statistics,
            "exports": {
                "functions": self.exports['functions'][:20],  # Sample first 20
                "classes": self.exports['classes'][:20],
                "constants": self.exports['constants'][:20],
                "variables": self.exports['variables'][:20]
            },
            "summary": {
                "total_exports": sum([
                    self.statistics["total_functions"],
                    self.statistics["total_classes"],
                    self.statistics["total_constants"],
                    self.statistics["total_variables"]
                ]),
                "export_distribution": {
                    "functions": self.statistics["total_functions"],
                    "classes": self.statistics["total_classes"],
                    "constants": self.statistics["total_constants"],
                    "variables": self.statistics["total_variables"]
                }
            }
        }

# Execute export scan for TestMaster directory
if __name__ == "__main__":
    scanner = ExportScanner()
    report = scanner.scan_directory("TestMaster", max_files=100)
    
    print("\n=== EXPORT INVENTORY REPORT ===")
    print(f"Files Scanned: {report['statistics']['scanned_files']}/{report['statistics']['total_files']}")
    print(f"Total Exports Found: {report['summary']['total_exports']}")
    print(f"\nExport Distribution:")
    for export_type, count in report['summary']['export_distribution'].items():
        print(f"  {export_type}: {count}")
    
    # Save sample exports to file
    with open("export_inventory_sample.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n=== SAMPLE EXPORTS ===")
    print("\nFunctions (first 5):")
    for func in report['exports']['functions'][:5]:
        print(f"  - {func['name']} in {func['file']}:{func['line']}")
        
    print("\nClasses (first 5):")  
    for cls in report['exports']['classes'][:5]:
        print(f"  - {cls['name']} in {cls['file']}:{cls['line']} ({len(cls['methods'])} methods)")