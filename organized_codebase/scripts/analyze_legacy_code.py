#!/usr/bin/env python3
"""
Legacy Code Analysis Tool
Extracts function signatures, dependencies, and functionality from legacy scripts.
"""

import ast
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    signature: str
    docstring: Optional[str]
    line_number: int
    is_public: bool
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    calls: List[str] = field(default_factory=list)  # Functions it calls
    complexity: int = 0

@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    line_number: int
    methods: List[FunctionInfo] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    docstring: Optional[str] = None

@dataclass
class ScriptAnalysis:
    """Complete analysis of a script."""
    file_path: str
    file_hash: str
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    globals: List[str] = field(default_factory=list)
    main_block: bool = False
    total_lines: int = 0
    complexity: int = 0

class LegacyCodeAnalyzer:
    """Analyzes legacy code to extract comprehensive information."""
    
    def __init__(self):
        self.analyses: Dict[str, ScriptAnalysis] = {}
        
    def analyze_file(self, file_path: Path) -> ScriptAnalysis:
        """Analyze a single Python file."""
        print(f"Analyzing {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate file hash
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Parse AST
            tree = ast.parse(content)
            
            analysis = ScriptAnalysis(
                file_path=str(file_path),
                file_hash=file_hash,
                total_lines=len(content.splitlines())
            )
            
            # Extract top-level elements
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node)
                    analysis.functions.append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    analysis.classes.append(class_info)
                    
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.add(alias.name)
                        analysis.dependencies.add(alias.name.split('.')[0])
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis.imports.add(node.module)
                        analysis.dependencies.add(node.module.split('.')[0])
                        
                elif isinstance(node, ast.Assign):
                    # Global variables
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis.globals.append(target.id)
                            
                elif isinstance(node, ast.If):
                    # Check for if __name__ == "__main__"
                    if self._is_main_block(node):
                        analysis.main_block = True
            
            # Calculate overall complexity
            analysis.complexity = sum(f.complexity for f in analysis.functions)
            
            self.analyses[str(file_path)] = analysis
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return ScriptAnalysis(
                file_path=str(file_path),
                file_hash="",
                total_lines=0
            )
    
    def _analyze_function(self, node: ast.FunctionDef) -> FunctionInfo:
        """Analyze a function definition."""
        # Extract signature
        signature = self._extract_function_signature(node)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        # Find function calls
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        return FunctionInfo(
            name=node.name,
            signature=signature,
            docstring=docstring,
            line_number=node.lineno,
            is_public=not node.name.startswith('_'),
            parameters=parameters,
            return_type=return_type,
            calls=list(set(calls)),  # Remove duplicates
            complexity=complexity
        )
    
    def _analyze_class(self, node: ast.ClassDef) -> ClassInfo:
        """Analyze a class definition."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item)
                methods.append(method_info)
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            else:
                bases.append(ast.unparse(base))
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        return ClassInfo(
            name=node.name,
            line_number=node.lineno,
            methods=methods,
            bases=bases,
            docstring=docstring
        )
    
    def _extract_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature as string."""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Default arguments
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                arg_index = len(args) - num_defaults + i
                if arg_index >= 0:
                    args[arg_index] += f" = {ast.unparse(default)}"
        
        # *args
        if node.args.vararg:
            vararg = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(vararg)
        
        # **kwargs
        if node.args.kwarg:
            kwarg = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(kwarg)
        
        # Return type
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"
        
        return f"def {node.name}({', '.join(args)}){return_type}"
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _is_main_block(self, node: ast.If) -> bool:
        """Check if this is the if __name__ == "__main__" block."""
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                for comparator in node.test.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
                        return True
        return False
    
    def analyze_directory(self, directory: Path) -> Dict[str, ScriptAnalysis]:
        """Analyze all Python files in a directory."""
        for py_file in directory.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                self.analyze_file(py_file)
        return self.analyses
    
    def generate_consolidation_report(self) -> str:
        """Generate a detailed consolidation report."""
        report_lines = [
            "=" * 80,
            "LEGACY CODE ANALYSIS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Files analyzed: {len(self.analyses)}",
            ""
        ]
        
        # Summary statistics
        total_functions = sum(len(analysis.functions) for analysis in self.analyses.values())
        total_classes = sum(len(analysis.classes) for analysis in self.analyses.values())
        total_lines = sum(analysis.total_lines for analysis in self.analyses.values())
        
        report_lines.extend([
            "SUMMARY STATISTICS:",
            f"  Total functions: {total_functions}",
            f"  Total classes: {total_classes}",
            f"  Total lines of code: {total_lines}",
            f"  Average file size: {total_lines / len(self.analyses):.1f} lines",
            ""
        ])
        
        # Function analysis by file
        report_lines.append("FUNCTION ANALYSIS BY FILE:")
        for file_path, analysis in sorted(self.analyses.items()):
            file_name = Path(file_path).name
            report_lines.append(f"\n{file_name} ({len(analysis.functions)} functions, {analysis.total_lines} lines):")
            
            for func in analysis.functions:
                visibility = "PUBLIC" if func.is_public else "PRIVATE"
                report_lines.append(f"  - {func.name}() [{visibility}] (complexity: {func.complexity})")
                if func.calls:
                    calls_str = ", ".join(func.calls[:5])
                    if len(func.calls) > 5:
                        calls_str += f" ... and {len(func.calls) - 5} more"
                    report_lines.append(f"    Calls: {calls_str}")
        
        # Dependency analysis
        all_dependencies = set()
        for analysis in self.analyses.values():
            all_dependencies.update(analysis.dependencies)
        
        report_lines.extend([
            "",
            "DEPENDENCY ANALYSIS:",
            f"  Unique dependencies: {len(all_dependencies)}",
            f"  Dependencies: {', '.join(sorted(all_dependencies))}",
            ""
        ])
        
        # Consolidation candidates
        report_lines.extend([
            "CONSOLIDATION CANDIDATES:",
            "  Generator scripts: " + str(len([f for f in self.analyses.keys() if 'generator' in f.lower()])),
            "  Converter scripts: " + str(len([f for f in self.analyses.keys() if 'convert' in f.lower()])),
            "  Verification scripts: " + str(len([f for f in self.analyses.keys() if 'verif' in f.lower()])),
            "  Coverage scripts: " + str(len([f for f in self.analyses.keys() if 'coverage' in f.lower()])),
            "  Fix/maintenance scripts: " + str(len([f for f in self.analyses.keys() if 'fix' in f.lower()])),
            ""
        ])
        
        # Function name analysis (looking for duplicates/similar)
        all_function_names = []
        for analysis in self.analyses.values():
            for func in analysis.functions:
                all_function_names.append((func.name, Path(analysis.file_path).name))
        
        # Find duplicate function names
        from collections import Counter
        function_counts = Counter(name for name, _ in all_function_names)
        duplicates = {name: count for name, count in function_counts.items() if count > 1}
        
        if duplicates:
            report_lines.append("DUPLICATE FUNCTION NAMES (potential consolidation targets):")
            for func_name, count in sorted(duplicates.items()):
                files_with_func = [file for name, file in all_function_names if name == func_name]
                report_lines.append(f"  {func_name}: {count} occurrences in {', '.join(files_with_func)}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        return "\n".join(report_lines)
    
    def save_analysis(self, output_file: Path):
        """Save analysis results to JSON file."""
        # Convert to JSON-serializable format
        serializable_data = {}
        for file_path, analysis in self.analyses.items():
            serializable_data[file_path] = asdict(analysis)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        print(f"Analysis saved to {output_file}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze legacy code for consolidation")
    parser.add_argument("--directory", default="archive/legacy_scripts", help="Directory to analyze")
    parser.add_argument("--output", default="tools/migration/legacy_analysis.json", help="Output file")
    parser.add_argument("--report", default="tools/migration/legacy_analysis_report.txt", help="Report file")
    
    args = parser.parse_args()
    
    analyzer = LegacyCodeAnalyzer()
    
    # Analyze legacy scripts
    legacy_dir = Path(args.directory)
    if not legacy_dir.exists():
        print(f"Directory {legacy_dir} not found!")
        return
    
    print(f"Analyzing legacy scripts in {legacy_dir}...")
    analyzer.analyze_directory(legacy_dir)
    
    # Generate and save report
    report = analyzer.generate_consolidation_report()
    
    # Save analysis data
    analyzer.save_analysis(Path(args.output))
    
    # Save report
    with open(args.report, 'w') as f:
        f.write(report)
    
    print(f"Analysis complete. Report saved to {args.report}")
    print(f"Analysis data saved to {args.output}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"  Files analyzed: {len(analyzer.analyses)}")
    total_functions = sum(len(analysis.functions) for analysis in analyzer.analyses.values())
    print(f"  Total functions: {total_functions}")
    print(f"  Report location: {args.report}")

if __name__ == "__main__":
    main()