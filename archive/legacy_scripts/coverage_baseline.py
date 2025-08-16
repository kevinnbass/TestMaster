#!/usr/bin/env python3
"""
Coverage Baseline Assessment
============================

Generate comprehensive baseline coverage report for all modules.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class CoverageBaseline:
    """Assess current test coverage and create baseline."""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.results = {}
        
    def run_coverage(self) -> Dict:
        """Run coverage and get detailed results."""
        print("Running comprehensive coverage analysis...")
        print("=" * 70)
        
        # Run pytest with coverage
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new', 
             '--cov=src_new', '--cov-report=json', 
             '--cov-report=html', '--cov-report=term-missing',
             '--tb=no', '-q', '--disable-warnings'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Load coverage data
        coverage_file = Path('coverage.json')
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
                
            return coverage_data
        
        return {}
    
    def analyze_uncovered(self, coverage_data: Dict) -> Dict[str, List[int]]:
        """Analyze which lines are uncovered."""
        uncovered = {}
        
        for file_path, file_data in coverage_data.get('files', {}).items():
            if 'src_new' in file_path:
                missing_lines = file_data.get('missing_lines', [])
                if missing_lines:
                    uncovered[file_path] = missing_lines
        
        return uncovered
    
    def categorize_failures(self) -> Dict[str, List[str]]:
        """Categorize test failures by type."""
        print("\nAnalyzing test failures...")
        
        # Run tests verbosely to capture failures
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        failures = {
            'import_errors': [],
            'api_mismatches': [],
            'async_issues': [],
            'mock_problems': [],
            'assertion_failures': [],
            'other': []
        }
        
        lines = result.stdout.split('\n')
        for line in lines:
            if 'ImportError' in line or 'ModuleNotFoundError' in line:
                failures['import_errors'].append(line)
            elif 'TypeError' in line or 'unexpected keyword' in line:
                failures['api_mismatches'].append(line)
            elif 'coroutine' in line or 'await' in line:
                failures['async_issues'].append(line)
            elif 'Mock' in line or 'mock' in line:
                failures['mock_problems'].append(line)
            elif 'AssertionError' in line:
                failures['assertion_failures'].append(line)
            elif 'FAILED' in line:
                failures['other'].append(line)
        
        return failures
    
    def generate_report(self):
        """Generate comprehensive baseline report."""
        print("\n" + "=" * 70)
        print("COVERAGE BASELINE ASSESSMENT")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}\n")
        
        # Get coverage data
        coverage_data = self.run_coverage()
        
        if coverage_data:
            totals = coverage_data.get('totals', {})
            percent = totals.get('percent_covered', 0)
            
            print(f"OVERALL COVERAGE: {percent:.2f}%")
            print(f"Lines Covered: {totals.get('covered_lines', 0)}")
            print(f"Lines Missing: {totals.get('missing_lines', 0)}")
            print(f"Total Lines: {totals.get('num_statements', 0)}")
            
            # Analyze uncovered code
            uncovered = self.analyze_uncovered(coverage_data)
            
            print(f"\nFILES WITH UNCOVERED CODE: {len(uncovered)}")
            
            # Show worst covered files
            worst_files = sorted(
                [(f, len(lines)) for f, lines in uncovered.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            print("\nTOP 10 LEAST COVERED FILES:")
            for file_path, missing_count in worst_files:
                file_name = Path(file_path).name
                print(f"  {file_name}: {missing_count} uncovered lines")
            
            # Save uncovered lines report
            with open('uncovered_lines.json', 'w') as f:
                json.dump(uncovered, f, indent=2)
            print("\nDetailed uncovered lines saved to: uncovered_lines.json")
            
            # Analyze test failures
            failures = self.categorize_failures()
            
            print("\nTEST FAILURE CATEGORIES:")
            for category, items in failures.items():
                if items:
                    print(f"  {category}: {len(items)} failures")
            
            # Save baseline
            baseline = {
                'timestamp': self.timestamp,
                'overall_coverage': percent,
                'totals': totals,
                'uncovered_files': len(uncovered),
                'failure_categories': {k: len(v) for k, v in failures.items()}
            }
            
            with open('coverage_baseline.json', 'w') as f:
                json.dump(baseline, f, indent=2)
            
            print("\nBaseline saved to: coverage_baseline.json")
            print("HTML report available at: htmlcov/index.html")
            
            # Generate priority matrix
            self.generate_priority_matrix(uncovered, failures)
            
            return percent
        else:
            print("ERROR: Could not generate coverage data")
            return 0
    
    def generate_priority_matrix(self, uncovered: Dict, failures: Dict):
        """Generate priority matrix for fixes."""
        print("\n" + "=" * 70)
        print("PRIORITY MATRIX")
        print("=" * 70)
        
        priorities = []
        
        # Priority 1: Core modules with low coverage
        core_modules = ['application', 'domain', 'container', 'bootstrap']
        for module in core_modules:
            for file_path in uncovered:
                if module in file_path:
                    priorities.append({
                        'priority': 1,
                        'module': module,
                        'file': file_path,
                        'uncovered_lines': len(uncovered[file_path]),
                        'reason': 'Core infrastructure'
                    })
        
        # Priority 2: Import errors (blocking other tests)
        if failures['import_errors']:
            priorities.append({
                'priority': 1,
                'category': 'import_errors',
                'count': len(failures['import_errors']),
                'reason': 'Blocking test execution'
            })
        
        # Priority 3: API mismatches
        if failures['api_mismatches']:
            priorities.append({
                'priority': 2,
                'category': 'api_mismatches',
                'count': len(failures['api_mismatches']),
                'reason': 'Test correctness'
            })
        
        # Save priority matrix
        with open('priority_matrix.json', 'w') as f:
            json.dump(priorities, f, indent=2)
        
        print("\nTOP PRIORITIES:")
        for item in sorted(priorities, key=lambda x: x['priority'])[:5]:
            print(f"  P{item['priority']}: {item.get('module', item.get('category', 'Unknown'))}")
            print(f"       Reason: {item['reason']}")
        
        print("\nPriority matrix saved to: priority_matrix.json")


def main():
    """Run baseline assessment."""
    baseline = CoverageBaseline()
    coverage = baseline.generate_report()
    
    print("\n" + "=" * 70)
    print("BASELINE ASSESSMENT COMPLETE")
    print("=" * 70)
    print(f"Current Coverage: {coverage:.2f}%")
    print(f"Target Coverage: 100%")
    print(f"Gap to Close: {100 - coverage:.2f}%")
    
    print("\nNEXT STEPS:")
    print("1. Review priority_matrix.json for fix order")
    print("2. Fix import errors first (Phase 2)")
    print("3. Fix API mismatches (Phase 2)")
    print("4. Generate tests for uncovered lines (Phase 3-4)")
    
    return 0 if coverage > 0 else 1


if __name__ == "__main__":
    sys.exit(main())