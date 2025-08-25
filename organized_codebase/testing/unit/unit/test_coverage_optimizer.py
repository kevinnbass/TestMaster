"""
Test Coverage Optimizer Module (Part 2 of advanced_testing_intelligence split)
Module size: <300 lines
Optimizes test coverage and identifies coverage gaps.
"""

import coverage
import subprocess
import sys
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class CoverageReport:
    """Coverage analysis report."""
    file_path: str
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    coverage_percentage: float
    uncovered_functions: List[str]
    uncovered_classes: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass 
class CoverageGap:
    """Identified coverage gap."""
    module: str
    gap_type: str  # 'function', 'class', 'branch', 'exception'
    location: str
    priority: int  # 1-10, higher is more important
    suggested_test: str

class TestCoverageOptimizer:
    """Optimizes test coverage and identifies gaps."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.coverage_data = None
        self.coverage_reports = {}
        self.coverage_gaps = []
        self.target_coverage = 0.95  # 95% target
        
    def measure_coverage(self, test_path: Path = None) -> Dict[str, Any]:
        """Measure test coverage for project."""
        try:
            # Initialize coverage
            cov = coverage.Coverage()
            cov.start()
            
            # Run tests
            if test_path:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_path), "-v"],
                    capture_output=True,
                    text=True
                )
            else:
                # Run all tests
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-v"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
            
            cov.stop()
            cov.save()
            
            # Analyze coverage
            coverage_data = {}
            for filename in cov.get_data().measured_files():
                file_path = Path(filename)
                if self._should_analyze_file(file_path):
                    analysis = cov.analysis2(filename)
                    executed = analysis[1]
                    missing = analysis[3]
                    
                    total_lines = len(executed) + len(missing)
                    coverage_pct = len(executed) / total_lines if total_lines > 0 else 0
                    
                    report = CoverageReport(
                        file_path=str(file_path),
                        total_lines=total_lines,
                        covered_lines=len(executed),
                        missing_lines=missing,
                        coverage_percentage=coverage_pct,
                        uncovered_functions=self._find_uncovered_functions(file_path, missing),
                        uncovered_classes=self._find_uncovered_classes(file_path, missing)
                    )
                    
                    coverage_data[str(file_path)] = report
                    self.coverage_reports[str(file_path)] = report
            
            # Calculate overall coverage
            total_lines = sum(r.total_lines for r in coverage_data.values())
            covered_lines = sum(r.covered_lines for r in coverage_data.values())
            overall_coverage = covered_lines / total_lines if total_lines > 0 else 0
            
            return {
                'overall_coverage': overall_coverage,
                'target_coverage': self.target_coverage,
                'meets_target': overall_coverage >= self.target_coverage,
                'file_coverage': {k: v.coverage_percentage for k, v in coverage_data.items()},
                'total_files': len(coverage_data),
                'files_below_target': sum(1 for r in coverage_data.values() 
                                         if r.coverage_percentage < self.target_coverage)
            }
            
        except Exception as e:
            logger.error(f"Error measuring coverage: {e}")
            return {'error': str(e)}
    
    def identify_coverage_gaps(self) -> List[CoverageGap]:
        """Identify and prioritize coverage gaps."""
        gaps = []
        
        for file_path, report in self.coverage_reports.items():
            # Check for uncovered functions
            for func in report.uncovered_functions:
                gap = CoverageGap(
                    module=file_path,
                    gap_type='function',
                    location=func,
                    priority=self._calculate_priority('function', report.coverage_percentage),
                    suggested_test=self._suggest_test_for_function(func)
                )
                gaps.append(gap)
            
            # Check for uncovered classes
            for cls in report.uncovered_classes:
                gap = CoverageGap(
                    module=file_path,
                    gap_type='class',
                    location=cls,
                    priority=self._calculate_priority('class', report.coverage_percentage),
                    suggested_test=self._suggest_test_for_class(cls)
                )
                gaps.append(gap)
            
            # Check for branch coverage gaps
            if report.missing_lines:
                gap = CoverageGap(
                    module=file_path,
                    gap_type='branch',
                    location=f"Lines {report.missing_lines[:5]}...",  # First 5 missing lines
                    priority=self._calculate_priority('branch', report.coverage_percentage),
                    suggested_test=f"Add branch tests for lines {report.missing_lines[:3]}"
                )
                gaps.append(gap)
        
        # Sort by priority
        gaps.sort(key=lambda x: x.priority, reverse=True)
        self.coverage_gaps = gaps
        
        return gaps
    
    def generate_coverage_improvement_plan(self) -> Dict[str, Any]:
        """Generate plan to improve coverage."""
        if not self.coverage_gaps:
            self.identify_coverage_gaps()
        
        # Group gaps by module
        gaps_by_module = {}
        for gap in self.coverage_gaps:
            if gap.module not in gaps_by_module:
                gaps_by_module[gap.module] = []
            gaps_by_module[gap.module].append(gap)
        
        # Create improvement plan
        plan = {
            'total_gaps': len(self.coverage_gaps),
            'critical_gaps': len([g for g in self.coverage_gaps if g.priority >= 8]),
            'modules_needing_work': len(gaps_by_module),
            'improvement_tasks': []
        }
        
        # Generate tasks
        for module, gaps in gaps_by_module.items():
            report = self.coverage_reports.get(module)
            current_coverage = report.coverage_percentage if report else 0
            
            task = {
                'module': module,
                'current_coverage': f"{current_coverage:.1%}",
                'target_coverage': f"{self.target_coverage:.1%}",
                'gaps_count': len(gaps),
                'priority': max(g.priority for g in gaps),
                'suggested_tests': [g.suggested_test for g in gaps[:3]]  # Top 3 suggestions
            }
            plan['improvement_tasks'].append(task)
        
        # Sort tasks by priority
        plan['improvement_tasks'].sort(key=lambda x: x['priority'], reverse=True)
        
        # Add time estimate
        plan['estimated_effort'] = self._estimate_effort(plan['total_gaps'])
        
        return plan
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed for coverage."""
        # Skip test files, __pycache__, etc.
        exclude_patterns = ['test_', '__pycache__', '.pyc', 'tests/', 'testing/']
        return not any(pattern in str(file_path) for pattern in exclude_patterns)
    
    def _find_uncovered_functions(self, file_path: Path, missing_lines: List[int]) -> List[str]:
        """Find functions that are not covered."""
        # Simplified - would need AST analysis for accurate results
        return [f"function_at_line_{line}" for line in missing_lines[:3]]
    
    def _find_uncovered_classes(self, file_path: Path, missing_lines: List[int]) -> List[str]:
        """Find classes that are not covered."""
        # Simplified - would need AST analysis for accurate results
        return []
    
    def _calculate_priority(self, gap_type: str, current_coverage: float) -> int:
        """Calculate priority for a coverage gap."""
        base_priority = {
            'function': 7,
            'class': 8,
            'branch': 5,
            'exception': 6
        }.get(gap_type, 5)
        
        # Adjust based on current coverage
        if current_coverage < 0.3:
            return min(10, base_priority + 3)
        elif current_coverage < 0.5:
            return min(10, base_priority + 2)
        elif current_coverage < 0.7:
            return min(10, base_priority + 1)
        return base_priority
    
    def _suggest_test_for_function(self, func_name: str) -> str:
        """Suggest a test for an uncovered function."""
        return f"def test_{func_name}(): # Test {func_name} with various inputs"
    
    def _suggest_test_for_class(self, class_name: str) -> str:
        """Suggest a test for an uncovered class."""
        return f"class Test{class_name}: # Test {class_name} initialization and methods"
    
    def _estimate_effort(self, gap_count: int) -> str:
        """Estimate effort to close coverage gaps."""
        hours = gap_count * 0.5  # Rough estimate: 30 min per gap
        if hours < 8:
            return f"{hours:.1f} hours"
        else:
            days = hours / 8
            return f"{days:.1f} days"
    
    def export_coverage_report(self, output_path: Path) -> bool:
        """Export coverage report to file."""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'coverage_reports': {k: {
                    'coverage': v.coverage_percentage,
                    'missing_lines': v.missing_lines[:10],  # First 10
                    'uncovered_functions': v.uncovered_functions
                } for k, v in self.coverage_reports.items()},
                'gaps': [{
                    'module': g.module,
                    'type': g.gap_type,
                    'location': g.location,
                    'priority': g.priority
                } for g in self.coverage_gaps[:20]]  # Top 20 gaps
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return False