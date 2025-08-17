"""
Advanced Test Coverage Analysis and Reporting System.
Provides detailed coverage metrics and automated improvement suggestions.
"""

import ast
import coverage
import subprocess
import sys
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


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


@dataclass
class CoverageAnalysis:
    """Complete coverage analysis results."""
    overall_coverage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    modules: List[ModuleCoverage]
    total_lines: int
    covered_lines: int
    total_functions: int
    covered_functions: int
    generated_at: datetime = field(default_factory=datetime.now)


class CoverageAnalyzer:
    """
    Advanced coverage analysis with detailed reporting and recommendations.
    """
    
    def __init__(self, src_dir: Path = Path("src"), test_dir: Path = Path("tests")):
        self.src_dir = src_dir
        self.test_dir = test_dir
        self.coverage_data = None
        self.analysis_results = None
        
    def run_comprehensive_coverage(self) -> CoverageAnalysis:
        """Run comprehensive coverage analysis."""
        print("[COVERAGE] Starting comprehensive coverage analysis...")
        
        # Initialize coverage
        cov = coverage.Coverage(branch=True, source=[str(self.src_dir)])
        cov.start()
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(self.test_dir),
                "--quiet",
                "--tb=short"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            print(f"[COVERAGE] Tests execution result: {result.returncode}")
            
        finally:
            cov.stop()
            cov.save()
            
        self.coverage_data = cov
        
        # Analyze coverage
        analysis = self._analyze_coverage_data(cov)
        self.analysis_results = analysis
        
        return analysis
        
    def _analyze_coverage_data(self, cov: coverage.Coverage) -> CoverageAnalysis:
        """Analyze coverage data and generate detailed results."""
        modules = []
        total_lines = 0
        covered_lines = 0
        total_functions = 0
        covered_functions = 0
        
        # Analyze each Python file
        for py_file in self.src_dir.rglob("*.py"):
            if self._should_analyze_file(py_file):
                module_coverage = self._analyze_module_coverage(cov, py_file)
                if module_coverage:
                    modules.append(module_coverage)
                    total_lines += module_coverage.total_lines
                    covered_lines += module_coverage.covered_lines
                    total_functions += len(module_coverage.functions)
                    covered_functions += sum(1 for f in module_coverage.functions if f.is_tested)
                    
        # Calculate overall metrics
        overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        function_coverage = (covered_functions / total_functions * 100) if total_functions > 0 else 0
        
        # Calculate branch coverage (if available)
        try:
            branch_data = cov.get_data().branch_lines()
            branch_coverage = self._calculate_branch_coverage(cov)
        except:
            branch_coverage = 0.0
            
        return CoverageAnalysis(
            overall_coverage=overall_coverage,
            line_coverage=overall_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            modules=modules,
            total_lines=total_lines,
            covered_lines=covered_lines,
            total_functions=total_functions,
            covered_functions=covered_functions
        )
        
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be included in coverage analysis."""
        file_str = str(file_path)
        
        # Skip test files, __pycache__, and __init__.py
        if any(skip in file_str for skip in ["__pycache__", "test_", "__init__.py"]):
            return False
            
        # Skip migration and script files
        if any(skip in file_str for skip in ["migration", "scripts"]):
            return False
            
        return True
        
    def _analyze_module_coverage(self, cov: coverage.Coverage, 
                                file_path: Path) -> Optional[ModuleCoverage]:
        """Analyze coverage for a specific module."""
        try:
            # Get coverage data
            analysis = cov.analysis2(str(file_path))
            filename, executed, excluded, missing, missing_formatted = analysis
            
            total_lines = len(executed) + len(missing)
            covered_lines_count = len(executed)
            coverage_percentage = (covered_lines_count / total_lines * 100) if total_lines > 0 else 0
            
            # Analyze functions in the file
            functions = self._analyze_function_coverage(file_path, set(executed), set(missing))
            
            # Check if module has tests
            has_tests = self._module_has_tests(file_path)
            
            # Calculate branch coverage for this module
            branch_coverage = self._calculate_module_branch_coverage(cov, str(file_path))
            
            return ModuleCoverage(
                name=self._path_to_module_name(file_path),
                file_path=str(file_path),
                total_lines=total_lines,
                covered_lines=covered_lines_count,
                coverage_percentage=coverage_percentage,
                functions=functions,
                missing_lines=set(missing),
                branch_coverage=branch_coverage,
                has_tests=has_tests
            )
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
            
    def _analyze_function_coverage(self, file_path: Path, executed: Set[int], 
                                 missing: Set[int]) -> List[FunctionCoverage]:
        """Analyze coverage for functions in a module."""
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                tree = ast.parse(source)
                
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_coverage = self._analyze_single_function_coverage(
                        node, executed, missing, source
                    )
                    functions.append(func_coverage)
                    
        except Exception as e:
            print(f"Error analyzing functions in {file_path}: {e}")
            
        return functions
        
    def _analyze_single_function_coverage(self, node: ast.FunctionDef, 
                                        executed: Set[int], missing: Set[int],
                                        source: str) -> FunctionCoverage:
        """Analyze coverage for a single function."""
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        
        # Get lines for this function
        function_lines = set(range(line_start, line_end + 1))
        covered_lines = function_lines.intersection(executed)
        missing_lines = function_lines.intersection(missing)
        
        total_lines = len(function_lines)
        coverage_percentage = (len(covered_lines) / total_lines * 100) if total_lines > 0 else 0
        
        # Calculate complexity
        complexity = self._calculate_function_complexity(node)
        
        # Check if function is tested (heuristic: >50% coverage)
        is_tested = coverage_percentage > 50
        
        return FunctionCoverage(
            name=node.name,
            line_start=line_start,
            line_end=line_end,
            covered_lines=covered_lines,
            total_lines=total_lines,
            coverage_percentage=coverage_percentage,
            complexity=complexity,
            missing_lines=missing_lines,
            is_tested=is_tested
        )
        
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
        
    def _calculate_branch_coverage(self, cov: coverage.Coverage) -> float:
        """Calculate overall branch coverage."""
        try:
            # Get branch coverage data
            data = cov.get_data()
            branch_stats = cov._analyze_branch_coverage()  # This might not exist
            return 0.0  # Placeholder - would need more sophisticated branch analysis
        except:
            return 0.0
            
    def _calculate_module_branch_coverage(self, cov: coverage.Coverage, 
                                        file_path: str) -> float:
        """Calculate branch coverage for a specific module."""
        # Simplified branch coverage calculation
        try:
            return 0.0  # Would need more sophisticated implementation
        except:
            return 0.0
            
    def _module_has_tests(self, file_path: Path) -> bool:
        """Check if a module has corresponding tests."""
        module_name = file_path.stem
        
        # Look for test files
        test_patterns = [
            f"test_{module_name}.py",
            f"test_{module_name}_*.py",
            f"{module_name}_test.py"
        ]
        
        for pattern in test_patterns:
            test_files = list(self.test_dir.rglob(pattern))
            if test_files:
                return True
                
        return False
        
    def _path_to_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        parts = file_path.parts
        if 'src' in parts:
            src_index = parts.index('src')
            module_parts = parts[src_index + 1:]  # Skip 'src'
            module_path = '.'.join(module_parts)
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            return module_path
        return str(file_path.stem)
        
    def generate_coverage_gaps_report(self) -> List[Dict[str, Any]]:
        """Generate detailed report of coverage gaps."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analysis first.")
            
        gaps = []
        
        for module in self.analysis_results.modules:
            if module.coverage_percentage < 95:  # Target coverage
                # Identify specific gaps
                uncovered_functions = [
                    f for f in module.functions 
                    if not f.is_tested and f.coverage_percentage < 80
                ]
                
                complex_uncovered = [
                    f for f in uncovered_functions 
                    if f.complexity > 5
                ]
                
                gap_info = {
                    "module": module.name,
                    "current_coverage": module.coverage_percentage,
                    "target_coverage": 95.0,
                    "gap": 95.0 - module.coverage_percentage,
                    "total_functions": len(module.functions),
                    "uncovered_functions": len(uncovered_functions),
                    "complex_uncovered_functions": len(complex_uncovered),
                    "missing_lines_count": len(module.missing_lines),
                    "has_tests": module.has_tests,
                    "priority": self._calculate_priority(module),
                    "recommendations": self._generate_module_recommendations(module)
                }
                
                gaps.append(gap_info)
                
        # Sort by priority
        gaps.sort(key=lambda x: (
            x["priority"] == "high",
            x["complex_uncovered_functions"],
            x["gap"]
        ), reverse=True)
        
        return gaps
        
    def _calculate_priority(self, module: ModuleCoverage) -> str:
        """Calculate priority level for improving module coverage."""
        complex_uncovered = sum(1 for f in module.functions if f.complexity > 5 and not f.is_tested)
        
        if module.coverage_percentage < 50:
            return "critical"
        elif module.coverage_percentage < 70 or complex_uncovered > 0:
            return "high"
        elif module.coverage_percentage < 85:
            return "medium"
        else:
            return "low"
            
    def _generate_module_recommendations(self, module: ModuleCoverage) -> List[str]:
        """Generate specific recommendations for improving module coverage."""
        recommendations = []
        
        if not module.has_tests:
            recommendations.append("Create basic test file for this module")
            
        uncovered_functions = [f for f in module.functions if not f.is_tested]
        if uncovered_functions:
            recommendations.append(f"Add tests for {len(uncovered_functions)} uncovered functions")
            
        complex_functions = [f for f in module.functions if f.complexity > 5 and not f.is_tested]
        if complex_functions:
            recommendations.append(f"Priority: Test {len(complex_functions)} complex functions")
            
        if module.coverage_percentage < 50:
            recommendations.append("Focus on basic functionality tests first")
        elif module.coverage_percentage < 80:
            recommendations.append("Add edge case and error handling tests")
        else:
            recommendations.append("Add comprehensive integration tests")
            
        return recommendations
        
    def generate_coverage_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for coverage dashboard visualization."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analysis first.")
            
        # Overall statistics
        dashboard_data = {
            "overall_stats": {
                "line_coverage": round(self.analysis_results.line_coverage, 1),
                "function_coverage": round(self.analysis_results.function_coverage, 1),
                "branch_coverage": round(self.analysis_results.branch_coverage, 1),
                "total_lines": self.analysis_results.total_lines,
                "covered_lines": self.analysis_results.covered_lines,
                "total_functions": self.analysis_results.total_functions,
                "covered_functions": self.analysis_results.covered_functions,
                "modules_analyzed": len(self.analysis_results.modules)
            },
            
            # Coverage by module
            "module_coverage": [
                {
                    "name": module.name,
                    "coverage": round(module.coverage_percentage, 1),
                    "lines": module.total_lines,
                    "functions": len(module.functions),
                    "has_tests": module.has_tests,
                    "priority": self._calculate_priority(module)
                }
                for module in self.analysis_results.modules
            ],
            
            # Coverage distribution
            "coverage_distribution": self._calculate_coverage_distribution(),
            
            # Top coverage gaps
            "top_gaps": self.generate_coverage_gaps_report()[:10],
            
            # Complexity analysis
            "complexity_analysis": self._analyze_complexity_coverage(),
            
            # Trends (would need historical data)
            "trends": {
                "coverage_trend": [self.analysis_results.line_coverage],  # Single point for now
                "function_trend": [self.analysis_results.function_coverage]
            }
        }
        
        return dashboard_data
        
    def _calculate_coverage_distribution(self) -> Dict[str, int]:
        """Calculate distribution of coverage levels."""
        distribution = {
            "excellent": 0,    # 95-100%
            "good": 0,         # 80-95%
            "fair": 0,         # 60-80%
            "poor": 0,         # 40-60%
            "critical": 0      # 0-40%
        }
        
        for module in self.analysis_results.modules:
            coverage = module.coverage_percentage
            
            if coverage >= 95:
                distribution["excellent"] += 1
            elif coverage >= 80:
                distribution["good"] += 1
            elif coverage >= 60:
                distribution["fair"] += 1
            elif coverage >= 40:
                distribution["poor"] += 1
            else:
                distribution["critical"] += 1
                
        return distribution
        
    def _analyze_complexity_coverage(self) -> Dict[str, Any]:
        """Analyze relationship between complexity and coverage."""
        complexity_data = []
        
        for module in self.analysis_results.modules:
            for func in module.functions:
                complexity_data.append({
                    "complexity": func.complexity,
                    "coverage": func.coverage_percentage,
                    "is_tested": func.is_tested,
                    "name": f"{module.name}.{func.name}"
                })
                
        # Calculate averages by complexity level
        complexity_analysis = {
            "low_complexity": [],    # 1-3
            "medium_complexity": [], # 4-7
            "high_complexity": []    # 8+
        }
        
        for data in complexity_data:
            if data["complexity"] <= 3:
                complexity_analysis["low_complexity"].append(data)
            elif data["complexity"] <= 7:
                complexity_analysis["medium_complexity"].append(data)
            else:
                complexity_analysis["high_complexity"].append(data)
                
        # Calculate coverage rates by complexity
        result = {}
        for level, functions in complexity_analysis.items():
            if functions:
                avg_coverage = sum(f["coverage"] for f in functions) / len(functions)
                tested_count = sum(1 for f in functions if f["is_tested"])
                result[level] = {
                    "average_coverage": round(avg_coverage, 1),
                    "total_functions": len(functions),
                    "tested_functions": tested_count,
                    "test_rate": round(tested_count / len(functions) * 100, 1)
                }
            else:
                result[level] = {
                    "average_coverage": 0,
                    "total_functions": 0,
                    "tested_functions": 0,
                    "test_rate": 0
                }
                
        return result
        
    def export_coverage_report(self, output_path: Path) -> None:
        """Export comprehensive coverage report to JSON."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analysis first.")
            
        report = {
            "summary": {
                "overall_coverage": self.analysis_results.overall_coverage,
                "line_coverage": self.analysis_results.line_coverage,
                "function_coverage": self.analysis_results.function_coverage,
                "branch_coverage": self.analysis_results.branch_coverage,
                "target_met": self.analysis_results.overall_coverage >= 95,
                "total_lines": self.analysis_results.total_lines,
                "covered_lines": self.analysis_results.covered_lines,
                "total_functions": self.analysis_results.total_functions,
                "covered_functions": self.analysis_results.covered_functions
            },
            "modules": [
                {
                    "name": module.name,
                    "file_path": module.file_path,
                    "coverage": module.coverage_percentage,
                    "total_lines": module.total_lines,
                    "covered_lines": module.covered_lines,
                    "missing_lines": list(module.missing_lines),
                    "has_tests": module.has_tests,
                    "functions": [
                        {
                            "name": func.name,
                            "coverage": func.coverage_percentage,
                            "complexity": func.complexity,
                            "is_tested": func.is_tested,
                            "lines": f"{func.line_start}-{func.line_end}"
                        }
                        for func in module.functions
                    ]
                }
                for module in self.analysis_results.modules
            ],
            "coverage_gaps": self.generate_coverage_gaps_report(),
            "dashboard_data": self.generate_coverage_dashboard_data(),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"Coverage report exported to {output_path}")
        
    def generate_visual_reports(self, output_dir: Path) -> None:
        """Generate visual coverage reports."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analysis first.")
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Coverage by module bar chart
        self._create_module_coverage_chart(output_dir)
        
        # Coverage distribution pie chart
        self._create_coverage_distribution_chart(output_dir)
        
        # Complexity vs Coverage scatter plot
        self._create_complexity_coverage_chart(output_dir)
        
    def _create_module_coverage_chart(self, output_dir: Path) -> None:
        """Create module coverage bar chart."""
        modules = [m.name for m in self.analysis_results.modules]
        coverages = [m.coverage_percentage for m in self.analysis_results.modules]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(modules)), coverages)
        
        # Color bars based on coverage level
        for i, bar in enumerate(bars):
            coverage = coverages[i]
            if coverage >= 95:
                bar.set_color('green')
            elif coverage >= 80:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
                
        plt.xlabel('Modules')
        plt.ylabel('Coverage Percentage')
        plt.title('Test Coverage by Module')
        plt.xticks(range(len(modules)), modules, rotation=45, ha='right')
        plt.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='Target (95%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'module_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_coverage_distribution_chart(self, output_dir: Path) -> None:
        """Create coverage distribution pie chart."""
        distribution = self._calculate_coverage_distribution()
        
        labels = []
        sizes = []
        colors = []
        
        color_map = {
            "excellent": "green",
            "good": "lightgreen", 
            "fair": "yellow",
            "poor": "orange",
            "critical": "red"
        }
        
        for level, count in distribution.items():
            if count > 0:
                labels.append(f"{level.title()} ({count})")
                sizes.append(count)
                colors.append(color_map[level])
                
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Coverage Distribution Across Modules')
        plt.axis('equal')
        plt.savefig(output_dir / 'coverage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_complexity_coverage_chart(self, output_dir: Path) -> None:
        """Create complexity vs coverage scatter plot."""
        complexities = []
        coverages = []
        colors = []
        
        for module in self.analysis_results.modules:
            for func in module.functions:
                complexities.append(func.complexity)
                coverages.append(func.coverage_percentage)
                colors.append('green' if func.is_tested else 'red')
                
        plt.figure(figsize=(12, 8))
        plt.scatter(complexities, coverages, c=colors, alpha=0.6)
        plt.xlabel('Function Complexity')
        plt.ylabel('Coverage Percentage')
        plt.title('Function Complexity vs Test Coverage')
        plt.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Good Coverage (80%)')
        plt.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='Target Coverage (95%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'complexity_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()