#!/usr/bin/env python3
"""
Test Coverage Analyzer
Comprehensive tool for analyzing test coverage and identifying gaps.
"""

import os
import sys
import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FunctionCoverage:
    """Coverage data for a single function."""
    name: str
    file_path: str
    line_start: int
    line_end: int
    is_covered: bool
    coverage_percentage: float
    test_count: int
    complexity: int
    missing_lines: List[int] = field(default_factory=list)


@dataclass
class ClassCoverage:
    """Coverage data for a class."""
    name: str
    file_path: str
    methods: List[FunctionCoverage]
    coverage_percentage: float
    total_methods: int
    covered_methods: int
    test_count: int


@dataclass
class ModuleCoverage:
    """Coverage data for a module."""
    name: str
    path: str
    functions: List[FunctionCoverage]
    classes: List[ClassCoverage]
    coverage_percentage: float
    lines_covered: int
    total_lines: int
    test_files: List[str]
    missing_tests: List[str]


class CoverageAnalyzer:
    """Analyzes test coverage across the codebase."""
    
    def __init__(self, source_dir: Path, test_dir: Path):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.coverage_data = {}
        self.test_mapping = defaultdict(list)
        
    def analyze_coverage(self) -> Dict[str, ModuleCoverage]:
        """Perform complete coverage analysis."""
        logger.info("Starting coverage analysis...")
        
        # Run pytest with coverage
        coverage_data = self._run_coverage_analysis()
        
        # Analyze source files
        modules = self._analyze_source_files()
        
        # Map tests to source
        self._map_tests_to_source()
        
        # Calculate coverage metrics
        self._calculate_coverage_metrics(modules, coverage_data)
        
        # Identify gaps
        self._identify_coverage_gaps(modules)
        
        return modules
    
    def _run_coverage_analysis(self) -> Dict[str, Any]:
        """Run pytest with coverage and get results."""
        try:
            # Run pytest with coverage
            cmd = [
                "pytest",
                f"--cov={self.source_dir}",
                "--cov-report=json",
                "--cov-report=term-missing",
                str(self.test_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.source_dir.parent
            )
            
            # Parse coverage JSON
            coverage_file = self.source_dir.parent / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    return json.load(f)
            
            return {}
            
        except Exception as e:
            logger.warning(f"Could not run coverage analysis: {e}")
            return {}
    
    def _analyze_source_files(self) -> Dict[str, ModuleCoverage]:
        """Analyze all source files for coverage targets."""
        modules = {}
        
        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            module_name = py_file.stem
            module_coverage = self._analyze_module(py_file)
            modules[module_name] = module_coverage
        
        return modules
    
    def _analyze_module(self, file_path: Path) -> ModuleCoverage:
        """Analyze a single module for coverage targets."""
        content = file_path.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ModuleCoverage(
                name=file_path.stem,
                path=str(file_path),
                functions=[],
                classes=[],
                coverage_percentage=0.0,
                lines_covered=0,
                total_lines=len(content.split('\n')),
                test_files=[],
                missing_tests=[]
            )
        
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip if it's a method inside a class
                parent_is_class = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        if node in ast.walk(parent):
                            parent_is_class = True
                            break
                
                if not parent_is_class:
                    func_coverage = self._analyze_function(node, file_path)
                    functions.append(func_coverage)
            
            elif isinstance(node, ast.ClassDef):
                class_coverage = self._analyze_class(node, file_path)
                classes.append(class_coverage)
        
        total_lines = len(content.split('\n'))
        
        return ModuleCoverage(
            name=file_path.stem,
            path=str(file_path),
            functions=functions,
            classes=classes,
            coverage_percentage=0.0,  # Will be calculated later
            lines_covered=0,
            total_lines=total_lines,
            test_files=[],
            missing_tests=[]
        )
    
    def _analyze_function(self, node: ast.FunctionDef, file_path: Path) -> FunctionCoverage:
        """Analyze a function for coverage."""
        complexity = self._calculate_complexity(node)
        
        return FunctionCoverage(
            name=node.name,
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_covered=False,  # Will be determined later
            coverage_percentage=0.0,
            test_count=0,
            complexity=complexity,
            missing_lines=[]
        )
    
    def _analyze_class(self, node: ast.ClassDef, file_path: Path) -> ClassCoverage:
        """Analyze a class for coverage."""
        methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_coverage = self._analyze_function(item, file_path)
                methods.append(method_coverage)
        
        return ClassCoverage(
            name=node.name,
            file_path=str(file_path),
            methods=methods,
            coverage_percentage=0.0,
            total_methods=len(methods),
            covered_methods=0,
            test_count=0
        )
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.Raise):
                complexity += 1
        
        return complexity
    
    def _map_tests_to_source(self):
        """Map test files to their source counterparts."""
        for test_file in self.test_dir.rglob("test_*.py"):
            if "__pycache__" in str(test_file):
                continue
            
            # Extract module name from test file
            test_name = test_file.stem
            if test_name.startswith("test_"):
                module_name = test_name[5:]  # Remove "test_" prefix
                self.test_mapping[module_name].append(str(test_file))
    
    def _calculate_coverage_metrics(self, modules: Dict[str, ModuleCoverage], 
                                   coverage_data: Dict[str, Any]):
        """Calculate coverage metrics for all modules."""
        if not coverage_data:
            return
        
        files_data = coverage_data.get("files", {})
        
        for module_name, module in modules.items():
            file_coverage = files_data.get(module.path, {})
            
            if file_coverage:
                summary = file_coverage.get("summary", {})
                module.coverage_percentage = summary.get("percent_covered", 0.0)
                module.lines_covered = summary.get("covered_lines", 0)
                
                # Update function coverage
                executed_lines = set(file_coverage.get("executed_lines", []))
                missing_lines = set(file_coverage.get("missing_lines", []))
                
                for func in module.functions:
                    func_lines = set(range(func.line_start, func.line_end + 1))
                    covered_lines = func_lines & executed_lines
                    func.missing_lines = sorted(func_lines & missing_lines)
                    
                    if func_lines:
                        func.coverage_percentage = len(covered_lines) / len(func_lines) * 100
                        func.is_covered = func.coverage_percentage > 0
                
                # Update class coverage
                for cls in module.classes:
                    covered_methods = sum(1 for m in cls.methods if m.is_covered)
                    cls.covered_methods = covered_methods
                    
                    if cls.total_methods > 0:
                        cls.coverage_percentage = covered_methods / cls.total_methods * 100
    
    def _identify_coverage_gaps(self, modules: Dict[str, ModuleCoverage]):
        """Identify functions and classes missing test coverage."""
        for module_name, module in modules.items():
            missing_tests = []
            
            # Check functions
            for func in module.functions:
                if not func.is_covered:
                    missing_tests.append(f"Function: {func.name}")
            
            # Check classes
            for cls in module.classes:
                if cls.coverage_percentage < 100:
                    for method in cls.methods:
                        if not method.is_covered:
                            missing_tests.append(f"Method: {cls.name}.{method.name}")
            
            module.missing_tests = missing_tests
            module.test_files = self.test_mapping.get(module_name, [])
    
    def generate_report(self, modules: Dict[str, ModuleCoverage]) -> str:
        """Generate a coverage report."""
        report = []
        report.append("=" * 80)
        report.append("TEST COVERAGE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_modules = len(modules)
        covered_modules = sum(1 for m in modules.values() if m.coverage_percentage > 0)
        avg_coverage = sum(m.coverage_percentage for m in modules.values()) / total_modules if total_modules else 0
        
        report.append("OVERALL STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Modules: {total_modules}")
        report.append(f"Covered Modules: {covered_modules}")
        report.append(f"Average Coverage: {avg_coverage:.1f}%")
        report.append("")
        
        # Modules with low coverage
        report.append("MODULES NEEDING ATTENTION (Coverage < 50%)")
        report.append("-" * 40)
        
        low_coverage = sorted(
            [(name, m) for name, m in modules.items() if m.coverage_percentage < 50],
            key=lambda x: x[1].coverage_percentage
        )
        
        for name, module in low_coverage[:10]:
            report.append(f"{name}: {module.coverage_percentage:.1f}% coverage")
            if module.missing_tests:
                report.append(f"  Missing tests for: {', '.join(module.missing_tests[:3])}")
        
        if not low_coverage:
            report.append("All modules have adequate coverage!")
        
        report.append("")
        
        # Complex functions without tests
        report.append("COMPLEX FUNCTIONS WITHOUT TESTS (Complexity > 5)")
        report.append("-" * 40)
        
        complex_untested = []
        for module in modules.values():
            for func in module.functions:
                if func.complexity > 5 and not func.is_covered:
                    complex_untested.append((func.name, func.complexity, module.name))
        
        complex_untested.sort(key=lambda x: x[1], reverse=True)
        
        for func_name, complexity, module_name in complex_untested[:10]:
            report.append(f"{module_name}.{func_name}: Complexity {complexity}")
        
        if not complex_untested:
            report.append("All complex functions have test coverage!")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_metrics(self, modules: Dict[str, ModuleCoverage], output_file: Path):
        """Export coverage metrics to JSON."""
        data = {
            "modules": {
                name: {
                    "coverage": module.coverage_percentage,
                    "lines_covered": module.lines_covered,
                    "total_lines": module.total_lines,
                    "missing_tests": module.missing_tests,
                    "test_files": module.test_files
                }
                for name, module in modules.items()
            },
            "summary": {
                "total_modules": len(modules),
                "average_coverage": sum(m.coverage_percentage for m in modules.values()) / len(modules) if modules else 0,
                "modules_with_tests": sum(1 for m in modules.values() if m.test_files),
                "modules_fully_covered": sum(1 for m in modules.values() if m.coverage_percentage >= 100)
            }
        }
        
        output_file.write_text(json.dumps(data, indent=2))
        logger.info(f"Metrics exported to {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze test coverage")
    parser.add_argument("source", help="Source directory to analyze")
    parser.add_argument("tests", help="Test directory")
    parser.add_argument("--output", help="Output file for metrics (JSON)")
    parser.add_argument("--report", help="Output file for report (text)")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    test_dir = Path(args.tests)
    
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        sys.exit(1)
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)
    
    analyzer = CoverageAnalyzer(source_dir, test_dir)
    modules = analyzer.analyze_coverage()
    
    # Generate report
    report = analyzer.generate_report(modules)
    print(report)
    
    if args.report:
        Path(args.report).write_text(report)
        print(f"\nReport saved to: {args.report}")
    
    # Export metrics
    if args.output:
        analyzer.export_metrics(modules, Path(args.output))
        print(f"Metrics exported to: {args.output}")


if __name__ == "__main__":
    main()