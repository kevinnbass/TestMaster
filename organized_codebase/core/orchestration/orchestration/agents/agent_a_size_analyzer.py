"""
Agent A - Size Violation Analyzer
Phase 1: Hours 21-25 - Analyze Size Violations and Modularization Candidates
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import ast

class SizeViolationAnalyzer:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.violations = {
            'critical': [],  # >1000 lines
            'high': [],      # >500 lines
            'medium': [],    # >300 lines
            'large_functions': [],  # Functions >50 lines
            'large_classes': []     # Classes >200 lines
        }
        self.statistics = {
            'total_files': 0,
            'critical_violations': 0,
            'high_violations': 0,
            'medium_violations': 0,
            'compliant_files': 0,
            'total_lines': 0
        }
        
    def analyze_directory(self, directory: str, max_files: int = 200) -> Dict:
        """Analyze Python files for size violations"""
        dir_path = self.root_path / directory
        python_files = list(dir_path.rglob("*.py"))[:max_files]
        
        print(f"Analyzing {len(python_files)} Python files for size violations...")
        
        for file_path in python_files:
            self.statistics['total_files'] += 1
            self.analyze_file(file_path)
            
        return self.generate_report()
    
    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for size violations"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                line_count = len(lines)
                
            self.statistics['total_lines'] += line_count
            relative_path = str(file_path.relative_to(self.root_path))
            
            # Check file size violations
            if line_count > 1000:
                self.violations['critical'].append((relative_path, line_count))
                self.statistics['critical_violations'] += 1
            elif line_count > 500:
                self.violations['high'].append((relative_path, line_count))
                self.statistics['high_violations'] += 1
            elif line_count > 300:
                self.violations['medium'].append((relative_path, line_count))
                self.statistics['medium_violations'] += 1
            else:
                self.statistics['compliant_files'] += 1
                
            # Analyze function and class sizes
            self.analyze_internal_sizes(file_path, relative_path)
            
        except Exception as e:
            pass
    
    def analyze_internal_sizes(self, file_path: Path, relative_path: str) -> None:
        """Analyze function and class sizes within a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tree = ast.parse(content)
                
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = self._get_node_lines(node)
                    if func_lines > 50:
                        self.violations['large_functions'].append({
                            'file': relative_path,
                            'name': node.name,
                            'lines': func_lines,
                            'start_line': node.lineno
                        })
                        
                elif isinstance(node, ast.ClassDef):
                    class_lines = self._get_node_lines(node)
                    if class_lines > 200:
                        self.violations['large_classes'].append({
                            'file': relative_path,
                            'name': node.name,
                            'lines': class_lines,
                            'start_line': node.lineno
                        })
        except:
            pass
    
    def _get_node_lines(self, node) -> int:
        """Calculate number of lines for a node"""
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            return node.end_lineno - node.lineno + 1
        return 0
    
    def generate_report(self) -> Dict:
        """Generate size violation report"""
        # Sort violations by size
        for key in ['critical', 'high', 'medium']:
            self.violations[key].sort(key=lambda x: x[1], reverse=True)
            
        return {
            'statistics': self.statistics,
            'violations': {
                'critical': self.violations['critical'][:10],
                'high': self.violations['high'][:10],
                'medium': self.violations['medium'][:10],
                'large_functions': self.violations['large_functions'][:10],
                'large_classes': self.violations['large_classes'][:10]
            },
            'summary': {
                'total_violations': (
                    self.statistics['critical_violations'] +
                    self.statistics['high_violations'] +
                    self.statistics['medium_violations']
                ),
                'modularization_candidates': len(self.violations['critical']) + len(self.violations['high']),
                'average_file_size': self.statistics['total_lines'] // max(1, self.statistics['total_files']),
                'compliance_rate': (self.statistics['compliant_files'] / max(1, self.statistics['total_files'])) * 100
            }
        }

# Execute size violation analysis
if __name__ == "__main__":
    analyzer = SizeViolationAnalyzer()
    report = analyzer.analyze_directory("TestMaster", max_files=200)
    
    print("\n=== SIZE VIOLATION REPORT ===")
    print(f"Total Files Analyzed: {report['statistics']['total_files']}")
    print(f"Total Lines of Code: {report['statistics']['total_lines']:,}")
    print(f"Average File Size: {report['summary']['average_file_size']} lines")
    print(f"Compliance Rate: {report['summary']['compliance_rate']:.1f}%")
    
    print(f"\n=== VIOLATIONS SUMMARY ===")
    print(f"Critical (>1000 lines): {report['statistics']['critical_violations']}")
    print(f"High (>500 lines): {report['statistics']['high_violations']}")
    print(f"Medium (>300 lines): {report['statistics']['medium_violations']}")
    print(f"Compliant (<300 lines): {report['statistics']['compliant_files']}")
    
    print("\n=== TOP CRITICAL VIOLATIONS ===")
    for file, lines in report['violations']['critical'][:5]:
        print(f"  {file}: {lines:,} lines")
        
    print("\n=== LARGE FUNCTIONS (>50 lines) ===")
    for func in report['violations']['large_functions'][:5]:
        print(f"  {func['name']} in {func['file']}: {func['lines']} lines")
        
    print("\n=== LARGE CLASSES (>200 lines) ===")
    for cls in report['violations']['large_classes'][:5]:
        print(f"  {cls['name']} in {cls['file']}: {cls['lines']} lines")