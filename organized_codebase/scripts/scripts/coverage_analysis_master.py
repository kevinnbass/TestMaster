#!/usr/bin/env python3
"""
Unified Coverage Analysis Master
================================

Consolidates ALL coverage analysis functionality into one powerful tool.

Consolidated scripts:
- branch_coverage_analyzer.py
- coverage_baseline.py
- coverage_improver.py
- check_what_needs_tests.py
- diagnose_final_five.py
- find_truly_missing.py
- measure_final_coverage.py
- generate_coverage_sequential.py
- quick_coverage_boost.py
- systematic_coverage.py
... and more

Author: Agent E - Infrastructure Consolidation
"""

import os
import sys
import json
import subprocess
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoverageType(Enum):
    """Types of coverage analysis."""
    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    PATH = "path"


class AnalysisMode(Enum):
    """Coverage analysis modes."""
    QUICK = "quick"  # Fast analysis
    DETAILED = "detailed"  # Comprehensive analysis
    INCREMENTAL = "incremental"  # Compare with baseline
    TARGETED = "targeted"  # Focus on specific areas
    CONTINUOUS = "continuous"  # Continuous monitoring


@dataclass
class CoverageReport:
    """Coverage analysis report."""
    file_path: Path
    coverage_type: CoverageType
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    class_coverage: float = 0.0
    
    # Detailed metrics
    lines_covered: int = 0
    lines_total: int = 0
    branches_covered: int = 0
    branches_total: int = 0
    functions_covered: int = 0
    functions_total: int = 0
    
    # Missing coverage
    missing_lines: List[int] = field(default_factory=list)
    missing_branches: List[str] = field(default_factory=list)
    missing_functions: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    analysis_time: float = 0.0


@dataclass
class CoverageAnalysisConfig:
    """Configuration for coverage analysis."""
    mode: AnalysisMode = AnalysisMode.DETAILED
    coverage_types: Set[CoverageType] = field(default_factory=lambda: {CoverageType.LINE, CoverageType.BRANCH})
    target_coverage: float = 100.0
    output_format: str = "html"  # html, json, xml, term
    baseline_file: Optional[Path] = None
    ignore_patterns: List[str] = field(default_factory=lambda: ["*/test_*", "*/tests/*"])
    verbose: bool = True


class CoverageAnalysisMaster:
    """
    Master coverage analysis system consolidating all coverage capabilities.
    """
    
    def __init__(self, config: Optional[CoverageAnalysisConfig] = None):
        self.config = config or CoverageAnalysisConfig()
        self.reports: Dict[Path, CoverageReport] = {}
        self.baseline: Optional[Dict[str, Any]] = None
        
        # Load baseline if specified
        if self.config.baseline_file and self.config.baseline_file.exists():
            self.load_baseline(self.config.baseline_file)
        
        # Statistics
        self.stats = {
            "files_analyzed": 0,
            "total_coverage": 0.0,
            "files_meeting_target": 0,
            "files_below_target": 0,
            "critical_gaps": []
        }
    
    def analyze(self, target: str) -> Dict[Path, CoverageReport]:
        """Analyze coverage for target."""
        logger.info(f"Starting coverage analysis for: {target}")
        start_time = time.time()
        
        # Find target files
        target_files = self._find_target_files(target)
        if not target_files:
            logger.error(f"No Python files found for: {target}")
            return {}
        
        logger.info(f"Analyzing {len(target_files)} files")
        
        # Analyze based on mode
        if self.config.mode == AnalysisMode.QUICK:
            self._analyze_quick(target_files)
        elif self.config.mode == AnalysisMode.INCREMENTAL:
            self._analyze_incremental(target_files)
        elif self.config.mode == AnalysisMode.TARGETED:
            self._analyze_targeted(target_files)
        else:
            self._analyze_detailed(target_files)
        
        # Update statistics
        self._update_statistics()
        
        # Generate report
        self._generate_report()
        
        # Print summary
        self._print_summary(time.time() - start_time)
        
        return self.reports
    
    def _find_target_files(self, target: str) -> List[Path]:
        """Find Python files to analyze."""
        target_path = Path(target)
        
        if target_path.is_file() and target_path.suffix == ".py":
            return [target_path]
        elif target_path.is_dir():
            files = list(target_path.rglob("*.py"))
            # Apply ignore patterns
            return [f for f in files if not any(f.match(p) for p in self.config.ignore_patterns)]
        else:
            # Treat as pattern
            return list(Path(".").rglob(target))
    
    def _analyze_quick(self, files: List[Path]):
        """Quick coverage analysis."""
        # Run coverage with minimal processing
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "run", "-m", "pytest"],
            capture_output=True,
            text=True
        )
        
        # Get coverage data
        coverage_data = self._get_coverage_data()
        
        for file_path in files:
            if str(file_path) in coverage_data.get("files", {}):
                file_data = coverage_data["files"][str(file_path)]
                report = CoverageReport(
                    file_path=file_path,
                    coverage_type=CoverageType.LINE,
                    line_coverage=file_data.get("summary", {}).get("percent_covered", 0.0)
                )
                self.reports[file_path] = report
    
    def _analyze_detailed(self, files: List[Path]):
        """Detailed coverage analysis."""
        for file_path in files:
            report = self._analyze_file(file_path)
            self.reports[file_path] = report
    
    def _analyze_incremental(self, files: List[Path]):
        """Incremental coverage analysis comparing with baseline."""
        current_coverage = {}
        
        for file_path in files:
            report = self._analyze_file(file_path)
            self.reports[file_path] = report
            current_coverage[str(file_path)] = report.line_coverage
        
        # Compare with baseline
        if self.baseline:
            for file_str, baseline_cov in self.baseline.items():
                file_path = Path(file_str)
                if file_path in self.reports:
                    current_cov = self.reports[file_path].line_coverage
                    improvement = current_cov - baseline_cov
                    
                    if improvement > 0:
                        logger.info(f"{file_path}: +{improvement:.1f}% coverage improvement")
                    elif improvement < 0:
                        logger.warning(f"{file_path}: {improvement:.1f}% coverage regression")
    
    def _analyze_targeted(self, files: List[Path]):
        """Targeted analysis focusing on low coverage areas."""
        # First pass: identify low coverage files
        low_coverage_files = []
        
        for file_path in files:
            quick_report = self._get_quick_coverage(file_path)
            if quick_report.line_coverage < self.config.target_coverage:
                low_coverage_files.append(file_path)
        
        # Detailed analysis of low coverage files
        logger.info(f"Performing detailed analysis on {len(low_coverage_files)} low coverage files")
        
        for file_path in low_coverage_files:
            report = self._analyze_file(file_path, detailed=True)
            self.reports[file_path] = report
    
    def _analyze_file(self, file_path: Path, detailed: bool = True) -> CoverageReport:
        """Analyze coverage for a single file."""
        logger.debug(f"Analyzing: {file_path}")
        start_time = time.time()
        
        report = CoverageReport(
            file_path=file_path,
            coverage_type=CoverageType.LINE
        )
        
        try:
            # Run coverage for this file
            result = subprocess.run(
                [sys.executable, "-m", "coverage", "run", "--source", str(file_path), "-m", "pytest"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Get coverage data
            coverage_json = self._get_coverage_json(file_path)
            
            if coverage_json:
                self._parse_coverage_data(report, coverage_json)
            
            if detailed:
                # Analyze missing coverage
                self._analyze_missing_coverage(report, file_path)
                
                # Analyze branches if requested
                if CoverageType.BRANCH in self.config.coverage_types:
                    self._analyze_branch_coverage(report, file_path)
                
                # Analyze functions if requested
                if CoverageType.FUNCTION in self.config.coverage_types:
                    self._analyze_function_coverage(report, file_path)
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
        
        report.analysis_time = time.time() - start_time
        return report
    
    def _get_quick_coverage(self, file_path: Path) -> CoverageReport:
        """Get quick coverage metrics for a file."""
        report = CoverageReport(
            file_path=file_path,
            coverage_type=CoverageType.LINE
        )
        
        try:
            # Use coverage.py API if available
            coverage_data = self._get_coverage_data()
            if str(file_path) in coverage_data.get("files", {}):
                file_data = coverage_data["files"][str(file_path)]
                report.line_coverage = file_data.get("summary", {}).get("percent_covered", 0.0)
        except:
            pass
        
        return report
    
    def _get_coverage_data(self) -> Dict[str, Any]:
        """Get coverage data from coverage.py."""
        try:
            subprocess.run(
                [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
                capture_output=True
            )
            
            with open("coverage.json") as f:
                return json.load(f)
        except:
            return {}
    
    def _get_coverage_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get coverage JSON for specific file."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "coverage", "json", "--include", str(file_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except:
            pass
        
        return None
    
    def _parse_coverage_data(self, report: CoverageReport, coverage_data: Dict[str, Any]):
        """Parse coverage data into report."""
        files_data = coverage_data.get("files", {})
        
        for file_path, file_data in files_data.items():
            if Path(file_path) == report.file_path:
                summary = file_data.get("summary", {})
                
                report.line_coverage = summary.get("percent_covered", 0.0)
                report.lines_covered = summary.get("covered_lines", 0)
                report.lines_total = summary.get("num_statements", 0)
                
                # Missing lines
                report.missing_lines = file_data.get("missing_lines", [])
                
                # Branch coverage if available
                if "branch_coverage" in file_data:
                    branch_data = file_data["branch_coverage"]
                    report.branch_coverage = branch_data.get("percent_covered", 0.0)
                    report.branches_covered = branch_data.get("covered_branches", 0)
                    report.branches_total = branch_data.get("num_branches", 0)
    
    def _analyze_missing_coverage(self, report: CoverageReport, file_path: Path):
        """Analyze what's missing coverage."""
        if not file_path.exists():
            return
        
        with open(file_path) as f:
            source_lines = f.readlines()
        
        # Group missing lines into ranges
        if report.missing_lines:
            ranges = []
            start = report.missing_lines[0]
            end = start
            
            for line in report.missing_lines[1:]:
                if line == end + 1:
                    end = line
                else:
                    ranges.append((start, end))
                    start = end = line
            
            ranges.append((start, end))
            
            # Log missing coverage ranges
            for start, end in ranges[:5]:  # Show first 5 ranges
                if start == end:
                    logger.debug(f"  Missing line {start}: {source_lines[start-1].strip()[:50]}")
                else:
                    logger.debug(f"  Missing lines {start}-{end}")
    
    def _analyze_branch_coverage(self, report: CoverageReport, file_path: Path):
        """Analyze branch coverage."""
        # Would implement detailed branch analysis here
        pass
    
    def _analyze_function_coverage(self, report: CoverageReport, file_path: Path):
        """Analyze function coverage."""
        if not file_path.exists():
            return
        
        with open(file_path) as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
            
            all_functions = []
            covered_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    all_functions.append(node.name)
                    
                    # Check if function is covered (simplified)
                    if node.lineno not in report.missing_lines:
                        covered_functions.append(node.name)
                    else:
                        report.missing_functions.append(node.name)
            
            report.functions_total = len(all_functions)
            report.functions_covered = len(covered_functions)
            
            if report.functions_total > 0:
                report.function_coverage = (report.functions_covered / report.functions_total) * 100
        
        except:
            pass
    
    def load_baseline(self, baseline_file: Path):
        """Load coverage baseline."""
        try:
            with open(baseline_file) as f:
                self.baseline = json.load(f)
            logger.info(f"Loaded baseline from {baseline_file}")
        except Exception as e:
            logger.error(f"Could not load baseline: {e}")
    
    def save_baseline(self, output_file: Path):
        """Save current coverage as baseline."""
        baseline = {}
        
        for file_path, report in self.reports.items():
            baseline[str(file_path)] = report.line_coverage
        
        with open(output_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logger.info(f"Saved baseline to {output_file}")
    
    def _update_statistics(self):
        """Update coverage statistics."""
        if not self.reports:
            return
        
        self.stats["files_analyzed"] = len(self.reports)
        
        total_coverage = sum(r.line_coverage for r in self.reports.values())
        self.stats["total_coverage"] = total_coverage / len(self.reports)
        
        for file_path, report in self.reports.items():
            if report.line_coverage >= self.config.target_coverage:
                self.stats["files_meeting_target"] += 1
            else:
                self.stats["files_below_target"] += 1
                
                if report.line_coverage < 50:  # Critical gap
                    self.stats["critical_gaps"].append({
                        "file": str(file_path),
                        "coverage": report.line_coverage
                    })
    
    def _generate_report(self):
        """Generate coverage report in specified format."""
        if self.config.output_format == "json":
            self._generate_json_report()
        elif self.config.output_format == "html":
            self._generate_html_report()
        elif self.config.output_format == "xml":
            self._generate_xml_report()
    
    def _generate_json_report(self):
        """Generate JSON coverage report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "files": {}
        }
        
        for file_path, report in self.reports.items():
            report_data["files"][str(file_path)] = {
                "line_coverage": report.line_coverage,
                "branch_coverage": report.branch_coverage,
                "function_coverage": report.function_coverage,
                "missing_lines": report.missing_lines,
                "missing_functions": report.missing_functions
            }
        
        with open("coverage_analysis.json", 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def _generate_html_report(self):
        """Generate HTML coverage report."""
        subprocess.run([sys.executable, "-m", "coverage", "html"], capture_output=True)
    
    def _generate_xml_report(self):
        """Generate XML coverage report."""
        subprocess.run([sys.executable, "-m", "coverage", "xml"], capture_output=True)
    
    def _print_summary(self, total_time: float):
        """Print coverage analysis summary."""
        print("\n" + "="*60)
        print("COVERAGE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Files analyzed: {self.stats['files_analyzed']}")
        print(f"Overall coverage: {self.stats['total_coverage']:.1f}%")
        print(f"Files meeting target ({self.config.target_coverage}%): {self.stats['files_meeting_target']}")
        print(f"Files below target: {self.stats['files_below_target']}")
        
        if self.stats["critical_gaps"]:
            print("\nCritical coverage gaps (<50%):")
            for gap in self.stats["critical_gaps"][:5]:
                print(f"  - {gap['file']}: {gap['coverage']:.1f}%")
        
        print(f"\nAnalysis time: {total_time:.2f}s")
        print("="*60)
    
    def find_uncovered_code(self) -> Dict[Path, List[str]]:
        """Find all uncovered code sections."""
        uncovered = {}
        
        for file_path, report in self.reports.items():
            if report.missing_lines or report.missing_functions:
                uncovered[file_path] = []
                
                if report.missing_lines:
                    uncovered[file_path].append(f"Missing lines: {report.missing_lines[:10]}")
                
                if report.missing_functions:
                    uncovered[file_path].append(f"Missing functions: {report.missing_functions[:5]}")
        
        return uncovered


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Coverage Analysis Master")
    parser.add_argument("target", help="Target file, directory, or pattern")
    parser.add_argument("--mode", choices=[m.value for m in AnalysisMode],
                       default=AnalysisMode.DETAILED.value,
                       help="Analysis mode")
    parser.add_argument("--target-coverage", type=float, default=100.0,
                       help="Target coverage percentage")
    parser.add_argument("--format", choices=["html", "json", "xml", "term"],
                       default="html",
                       help="Output format")
    parser.add_argument("--baseline", help="Baseline coverage file")
    parser.add_argument("--save-baseline", help="Save current coverage as baseline")
    
    args = parser.parse_args()
    
    config = CoverageAnalysisConfig(
        mode=AnalysisMode(args.mode),
        target_coverage=args.target_coverage,
        output_format=args.format,
        baseline_file=Path(args.baseline) if args.baseline else None
    )
    
    analyzer = CoverageAnalysisMaster(config)
    reports = analyzer.analyze(args.target)
    
    if args.save_baseline:
        analyzer.save_baseline(Path(args.save_baseline))


if __name__ == "__main__":
    main()