"""
Unified Coverage Analysis System

Consolidates functionality from:
- measure_final_coverage.py
- coverage_analysis.py  
- branch_coverage_analyzer.py
- coverage_baseline.py

Provides comprehensive coverage measurement, analysis, and reporting.
"""

import ast
import coverage
import subprocess
import sys
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

from ..core.layer_manager import requires_layer


@dataclass
class FunctionCoverage:
    """Coverage information for a function."""
    name: str
    line_start: int
    line_end: int
    covered_lines: Set[int]
    total_lines: int
    coverage_percentage: float
    complexity: int
    missing_lines: Set[int]
    is_tested: bool = False
    test_quality_score: float = 0.0


@dataclass  
class ModuleCoverage:
    """Coverage information for a module."""
    name: str
    file_path: str
    total_lines: int
    covered_lines: int
    coverage_percentage: float
    functions: List[FunctionCoverage]
    missing_lines: Set[int]
    branch_coverage: float = 0.0
    has_tests: bool = False
    test_files: List[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Complete coverage analysis results."""
    overall_percentage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    modules: List[ModuleCoverage]
    total_lines: int
    covered_lines: int
    missing_lines: int
    test_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Categorized test counts (from measure_final_coverage.py)
    test_categories: Dict[str, int] = field(default_factory=dict)
    

class CoverageAnalyzer:
    """
    Unified coverage analyzer that consolidates multiple analysis approaches.
    
    Features:
    - Line and branch coverage measurement
    - Function-level coverage analysis
    - Test categorization and tracking
    - Missing coverage identification
    - Quality scoring integration
    """
    
    @requires_layer("layer1_test_foundation", "test_mapping")
    def __init__(self, source_dir: Union[str, Path], test_dir: Union[str, Path]):
        """
        Initialize the coverage analyzer.
        
        Args:
            source_dir: Directory containing source code
            test_dir: Directory containing tests
        """
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        
        # Coverage tracking
        self.cov = coverage.Coverage(
            source=[str(self.source_dir)],
            branch=True
        )
        
        # Analysis cache
        self._module_ast_cache = {}
        self._function_cache = {}
        
    def run_full_analysis(self) -> CoverageReport:
        """
        Run comprehensive coverage analysis.
        
        Returns:
            Complete coverage report with all metrics
        """
        print("🔍 Running comprehensive coverage analysis...")
        
        # Start coverage measurement
        self.cov.start()
        
        # Run all tests with coverage
        test_result = self._run_tests_with_coverage()
        
        # Stop coverage measurement
        self.cov.stop()
        self.cov.save()
        
        # Analyze coverage data
        modules = self._analyze_modules()
        test_categories = self._categorize_tests()
        
        # Calculate overall metrics
        total_lines = sum(m.total_lines for m in modules)
        covered_lines = sum(m.covered_lines for m in modules)
        line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        # Calculate branch coverage
        branch_coverage = self._calculate_branch_coverage()
        
        # Calculate function coverage
        function_coverage = self._calculate_function_coverage(modules)
        
        report = CoverageReport(
            overall_percentage=line_coverage,
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            modules=modules,
            total_lines=total_lines,
            covered_lines=covered_lines,
            missing_lines=total_lines - covered_lines,
            test_count=sum(test_categories.values()),
            test_categories=test_categories
        )
        
        self._print_summary(report)
        return report
        
    def _run_tests_with_coverage(self) -> bool:
        """Run tests with coverage measurement."""
        try:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'pytest', 
                    str(self.test_dir),
                    '--cov=' + str(self.source_dir),
                    '--cov-branch',
                    '--cov-report=json:coverage.json',
                    '--cov-report=xml:coverage.xml',
                    '-q'  # Quiet mode
                ],
                capture_output=True,
                text=True,
                cwd=self.source_dir.parent
            )
            
            if result.returncode != 0:
                print(f"⚠️ Test execution had issues: {result.stderr}")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def _analyze_modules(self) -> List[ModuleCoverage]:
        """Analyze coverage for all source modules."""
        modules = []
        
        for py_file in self.source_dir.rglob("*.py"):
            if self._should_analyze_file(py_file):
                module_cov = self._analyze_module(py_file)
                if module_cov:
                    modules.append(module_cov)
                    
        return sorted(modules, key=lambda m: m.coverage_percentage)
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed for coverage."""
        if "__pycache__" in str(file_path):
            return False
        if file_path.name.startswith("_") and file_path.stem != "__init__":
            return False
        if file_path.suffix != ".py":
            return False
        if "test" in file_path.name.lower():
            return False
        return True
    
    def _analyze_module(self, file_path: Path) -> Optional[ModuleCoverage]:
        """Analyze coverage for a single module."""
        try:
            # Get coverage data for this file
            analysis = self.cov.analysis2(str(file_path))
            
            if not analysis:
                return None
                
            _, executable_lines, missing_lines, _ = analysis
            
            if not executable_lines:
                return None
            
            covered_lines = len(executable_lines) - len(missing_lines)
            coverage_percentage = (covered_lines / len(executable_lines)) * 100
            
            # Analyze functions in this module
            functions = self._analyze_functions(file_path, executable_lines, missing_lines)
            
            # Find related test files
            test_files = self._find_test_files(file_path)
            
            return ModuleCoverage(
                name=file_path.stem,
                file_path=str(file_path),
                total_lines=len(executable_lines),
                covered_lines=covered_lines,
                coverage_percentage=coverage_percentage,
                functions=functions,
                missing_lines=set(missing_lines),
                has_tests=len(test_files) > 0,
                test_files=test_files
            )
            
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
            return None
    
    def _analyze_functions(self, file_path: Path, executable_lines: Set[int], 
                          missing_lines: Set[int]) -> List[FunctionCoverage]:
        """Analyze coverage for functions in a module."""
        functions = []
        
        try:
            # Parse AST to find functions
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_cov = self._analyze_function(
                        node, executable_lines, missing_lines
                    )
                    if func_cov:
                        functions.append(func_cov)
                        
        except Exception as e:
            print(f"⚠️ Error parsing functions in {file_path}: {e}")
            
        return functions
    
    def _analyze_function(self, func_node: ast.FunctionDef, 
                         executable_lines: Set[int], 
                         missing_lines: Set[int]) -> Optional[FunctionCoverage]:
        """Analyze coverage for a single function."""
        func_start = func_node.lineno
        func_end = getattr(func_node, 'end_lineno', func_start)
        
        # Find executable lines in this function
        func_executable = {
            line for line in executable_lines 
            if func_start <= line <= func_end
        }
        
        if not func_executable:
            return None
            
        # Find missing lines in this function
        func_missing = {
            line for line in missing_lines
            if func_start <= line <= func_end
        }
        
        covered_lines = func_executable - func_missing
        coverage_percentage = (
            len(covered_lines) / len(func_executable) * 100
            if func_executable else 0
        )
        
        # Calculate complexity (simple metric)
        complexity = self._calculate_complexity(func_node)
        
        return FunctionCoverage(
            name=func_node.name,
            line_start=func_start,
            line_end=func_end,
            covered_lines=covered_lines,
            total_lines=len(func_executable),
            coverage_percentage=coverage_percentage,
            complexity=complexity,
            missing_lines=func_missing,
            is_tested=coverage_percentage > 0
        )
    
    def _calculate_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                               ast.With, ast.AsyncWith, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Break, ast.Continue)):
                complexity += 1
                
        return complexity
    
    def _find_test_files(self, source_file: Path) -> List[str]:
        """Find test files related to a source module."""
        test_files = []
        module_name = source_file.stem
        
        # Common test file patterns
        patterns = [
            f"test_{module_name}.py",
            f"test_{module_name}_*.py",
            f"{module_name}_test.py",
            f"*{module_name}*.py"
        ]
        
        for pattern in patterns:
            matches = list(self.test_dir.glob(f"**/{pattern}"))
            test_files.extend([str(f.relative_to(self.test_dir)) for f in matches])
        
        return test_files
    
    def _categorize_tests(self) -> Dict[str, int]:
        """Categorize tests by type (from measure_final_coverage.py logic)."""
        categories = {
            "ai_generated": 0,
            "gemini_generated": 0, 
            "integration": 0,
            "intelligent_converted": 0,
            "manual": 0,
            "healed": 0,
            "other": 0
        }
        
        if not self.test_dir.exists():
            return categories
        
        # Count by directory structure
        for category in categories:
            category_dir = self.test_dir / category
            if category_dir.exists():
                categories[category] = len(list(category_dir.glob("test_*.py")))
        
        # Count manual and healed tests in main directory
        manual_tests = len(list(self.test_dir.glob("test_*_manual.py")))
        healed_tests = len(list(self.test_dir.glob("test_*_healed.py")))
        
        categories["manual"] += manual_tests
        categories["healed"] += healed_tests
        
        # Count other tests
        all_tests = len(list(self.test_dir.glob("test_*.py")))
        categorized_tests = sum(categories.values()) - categories["other"]
        categories["other"] = max(0, all_tests - categorized_tests)
        
        return categories
    
    def _calculate_branch_coverage(self) -> float:
        """Calculate branch coverage from coverage data."""
        try:
            # Load coverage JSON data if available
            coverage_json = Path("coverage.json")
            if coverage_json.exists():
                with open(coverage_json) as f:
                    data = json.load(f)
                    totals = data.get("totals", {})
                    return totals.get("percent_covered_display", 0.0)
            
            # Fallback to basic branch analysis
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_function_coverage(self, modules: List[ModuleCoverage]) -> float:
        """Calculate function-level coverage percentage."""
        total_functions = 0
        tested_functions = 0
        
        for module in modules:
            for func in module.functions:
                total_functions += 1
                if func.is_tested:
                    tested_functions += 1
        
        return (tested_functions / total_functions * 100) if total_functions > 0 else 0
    
    def _print_summary(self, report: CoverageReport):
        """Print coverage summary."""
        print("\n" + "="*60)
        print("🎯 TESTMASTER COVERAGE ANALYSIS REPORT")
        print("="*60)
        
        print(f"📊 Overall Coverage: {report.overall_percentage:.1f}%")
        print(f"📈 Line Coverage: {report.line_coverage:.1f}%")
        print(f"🌿 Branch Coverage: {report.branch_coverage:.1f}%") 
        print(f"⚡ Function Coverage: {report.function_coverage:.1f}%")
        
        print(f"\n📝 Coverage Details:")
        print(f"   Total Lines: {report.total_lines:,}")
        print(f"   Covered Lines: {report.covered_lines:,}")
        print(f"   Missing Lines: {report.missing_lines:,}")
        
        print(f"\n🧪 Test Summary:")
        print(f"   Total Tests: {report.test_count}")
        for category, count in report.test_categories.items():
            if count > 0:
                print(f"   {category.replace('_', ' ').title()}: {count}")
        
        # Show modules with lowest coverage
        print(f"\n⚠️ Modules Needing Attention:")
        low_coverage = [m for m in report.modules if m.coverage_percentage < 80][:5]
        for module in low_coverage:
            print(f"   {module.name}: {module.coverage_percentage:.1f}% coverage")
        
        print("="*60)
    
    def get_uncovered_functions(self, report: CoverageReport) -> List[Tuple[str, str]]:
        """Get list of uncovered functions for test generation."""
        uncovered = []
        
        for module in report.modules:
            for func in module.functions:
                if not func.is_tested:
                    uncovered.append((module.name, func.name))
        
        return uncovered
    
    def save_report(self, report: CoverageReport, output_path: str = "coverage_report.json"):
        """Save coverage report to JSON file."""
        # Convert report to serializable format
        data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_percentage": report.overall_percentage,
            "line_coverage": report.line_coverage,
            "branch_coverage": report.branch_coverage,
            "function_coverage": report.function_coverage,
            "total_lines": report.total_lines,
            "covered_lines": report.covered_lines,
            "missing_lines": report.missing_lines,
            "test_count": report.test_count,
            "test_categories": report.test_categories,
            "modules": [
                {
                    "name": m.name,
                    "file_path": m.file_path,
                    "coverage_percentage": m.coverage_percentage,
                    "total_lines": m.total_lines,
                    "covered_lines": m.covered_lines,
                    "missing_lines": list(m.missing_lines),
                    "has_tests": m.has_tests,
                    "test_files": m.test_files,
                    "functions": [
                        {
                            "name": f.name,
                            "coverage_percentage": f.coverage_percentage,
                            "complexity": f.complexity,
                            "is_tested": f.is_tested,
                            "line_start": f.line_start,
                            "line_end": f.line_end
                        }
                        for f in m.functions
                    ]
                }
                for m in report.modules
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"📄 Coverage report saved to {output_path}")