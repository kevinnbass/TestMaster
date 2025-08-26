#!/usr/bin/env python3
"""
Test Orchestrator
Intelligent test execution and orchestration system.
"""

import os
import sys
import json
import time
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPriority(Enum):
    """Test execution priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case."""
    name: str
    file_path: str
    priority: TestPriority
    estimated_duration: float
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    duration: float = 0.0


@dataclass
class TestSuite:
    """Collection of related test cases."""
    name: str
    tests: List[TestCase]
    parallel: bool = True
    timeout: float = 300.0
    setup_script: Optional[str] = None
    teardown_script: Optional[str] = None


class TestOrchestrator:
    """Orchestrates test execution across the codebase."""
    
    def __init__(self, test_dir: Path, max_workers: int = None):
        self.test_dir = Path(test_dir)
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.test_suites = {}
        self.test_results = {}
        self.execution_order = []
        
    def discover_tests(self) -> Dict[str, TestSuite]:
        """Discover all tests in the test directory."""
        logger.info(f"Discovering tests in {self.test_dir}")
        
        test_suites = {}
        
        # Scan for test files
        for test_file in self.test_dir.rglob("test_*.py"):
            if "__pycache__" in str(test_file):
                continue
            
            suite_name = self._get_suite_name(test_file)
            
            if suite_name not in test_suites:
                test_suites[suite_name] = TestSuite(
                    name=suite_name,
                    tests=[],
                    parallel=self._can_run_parallel(suite_name)
                )
            
            # Discover tests in file
            tests = self._discover_tests_in_file(test_file)
            test_suites[suite_name].tests.extend(tests)
        
        self.test_suites = test_suites
        logger.info(f"Discovered {len(test_suites)} test suites")
        
        return test_suites
    
    def _get_suite_name(self, test_file: Path) -> str:
        """Get suite name from test file path."""
        # Group by directory
        relative_path = test_file.relative_to(self.test_dir)
        if len(relative_path.parts) > 1:
            return relative_path.parts[0]
        return "default"
    
    def _can_run_parallel(self, suite_name: str) -> bool:
        """Determine if suite can run tests in parallel."""
        # Integration tests should run sequentially
        if "integration" in suite_name.lower():
            return False
        return True
    
    def _discover_tests_in_file(self, test_file: Path) -> List[TestCase]:
        """Discover individual tests in a file."""
        tests = []
        
        try:
            # Use pytest collection
            result = subprocess.run(
                ["pytest", "--collect-only", "-q", str(test_file)],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if "::" in line and "test_" in line:
                    test_name = line.strip()
                    tests.append(TestCase(
                        name=test_name,
                        file_path=str(test_file),
                        priority=self._determine_priority(test_name),
                        estimated_duration=self._estimate_duration(test_name),
                        tags=self._extract_tags(test_name)
                    ))
        
        except Exception as e:
            logger.warning(f"Could not discover tests in {test_file}: {e}")
        
        return tests
    
    def _determine_priority(self, test_name: str) -> TestPriority:
        """Determine test priority based on name and tags."""
        lower_name = test_name.lower()
        
        if "critical" in lower_name or "security" in lower_name:
            return TestPriority.CRITICAL
        elif "integration" in lower_name or "api" in lower_name:
            return TestPriority.HIGH
        elif "unit" in lower_name:
            return TestPriority.MEDIUM
        else:
            return TestPriority.LOW
    
    def _estimate_duration(self, test_name: str) -> float:
        """Estimate test duration based on type."""
        lower_name = test_name.lower()
        
        if "integration" in lower_name:
            return 10.0
        elif "performance" in lower_name:
            return 30.0
        elif "unit" in lower_name:
            return 1.0
        else:
            return 5.0
    
    def _extract_tags(self, test_name: str) -> List[str]:
        """Extract tags from test name."""
        tags = []
        lower_name = test_name.lower()
        
        # Common tags
        for tag in ["unit", "integration", "api", "security", "performance"]:
            if tag in lower_name:
                tags.append(tag)
        
        return tags
    
    def plan_execution(self, filter_tags: List[str] = None) -> List[TestCase]:
        """Plan test execution order based on priority and dependencies."""
        all_tests = []
        
        for suite in self.test_suites.values():
            for test in suite.tests:
                # Apply tag filter if specified
                if filter_tags:
                    if not any(tag in test.tags for tag in filter_tags):
                        continue
                all_tests.append(test)
        
        # Sort by priority and estimated duration
        all_tests.sort(key=lambda t: (t.priority.value, -t.estimated_duration))
        
        self.execution_order = all_tests
        return all_tests
    
    def execute_tests(self, tests: List[TestCase] = None, 
                      parallel: bool = True) -> Dict[str, Any]:
        """Execute tests according to plan."""
        if tests is None:
            tests = self.execution_order
        
        if not tests:
            logger.warning("No tests to execute")
            return {}
        
        logger.info(f"Executing {len(tests)} tests")
        
        start_time = time.time()
        
        if parallel:
            results = self._execute_parallel(tests)
        else:
            results = self._execute_sequential(tests)
        
        end_time = time.time()
        
        # Generate summary
        summary = self._generate_summary(results, end_time - start_time)
        
        self.test_results = {
            "results": results,
            "summary": summary,
            "execution_time": end_time - start_time
        }
        
        return self.test_results
    
    def _execute_parallel(self, tests: List[TestCase]) -> Dict[str, Any]:
        """Execute tests in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for test in tests:
                future = executor.submit(self._run_test, test)
                futures[future] = test
            
            for future in futures:
                test = futures[future]
                try:
                    result = future.result(timeout=test.estimated_duration * 3)
                    results[test.name] = result
                    test.status = TestStatus.PASSED if result["passed"] else TestStatus.FAILED
                    test.result = result
                    test.duration = result.get("duration", 0)
                except Exception as e:
                    results[test.name] = {
                        "passed": False,
                        "error": str(e)
                    }
                    test.status = TestStatus.ERROR
        
        return results
    
    def _execute_sequential(self, tests: List[TestCase]) -> Dict[str, Any]:
        """Execute tests sequentially."""
        results = {}
        
        for test in tests:
            try:
                result = self._run_test(test)
                results[test.name] = result
                test.status = TestStatus.PASSED if result["passed"] else TestStatus.FAILED
                test.result = result
                test.duration = result.get("duration", 0)
            except Exception as e:
                results[test.name] = {
                    "passed": False,
                    "error": str(e)
                }
                test.status = TestStatus.ERROR
        
        return results
    
    def _run_test(self, test: TestCase) -> Dict[str, Any]:
        """Run a single test."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["pytest", "-xvs", f"{test.file_path}::{test.name.split('::')[-1]}"],
                capture_output=True,
                text=True,
                timeout=test.estimated_duration * 3
            )
            
            end_time = time.time()
            
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "returncode": result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": "Test timed out",
                "duration": time.time() - start_time
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _generate_summary(self, results: Dict[str, Any], 
                         total_time: float) -> Dict[str, Any]:
        """Generate test execution summary."""
        passed = sum(1 for r in results.values() if r.get("passed"))
        failed = len(results) - passed
        
        return {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / len(results) * 100) if results else 0,
            "total_time": total_time,
            "average_time": total_time / len(results) if results else 0
        }
    
    def generate_report(self) -> str:
        """Generate execution report."""
        if not self.test_results:
            return "No test results available"
        
        summary = self.test_results["summary"]
        
        report = []
        report.append("=" * 60)
        report.append("TEST EXECUTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed']}")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Pass Rate: {summary['pass_rate']:.1f}%")
        report.append(f"Total Time: {summary['total_time']:.2f}s")
        report.append(f"Average Time: {summary['average_time']:.2f}s")
        report.append("")
        
        # Failed tests
        if summary['failed'] > 0:
            report.append("FAILED TESTS:")
            report.append("-" * 40)
            
            for test_name, result in self.test_results["results"].items():
                if not result.get("passed"):
                    report.append(f"  {test_name}")
                    if "error" in result:
                        report.append(f"    Error: {result['error']}")
            report.append("")
        
        # Slow tests
        report.append("SLOWEST TESTS:")
        report.append("-" * 40)
        
        slow_tests = sorted(
            [(name, r.get("duration", 0)) for name, r in self.test_results["results"].items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for name, duration in slow_tests[:5]:
            report.append(f"  {name}: {duration:.2f}s")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, output_file: Path):
        """Save test results to file."""
        if not self.test_results:
            logger.warning("No results to save")
            return
        
        # Convert TestCase objects to dict for JSON serialization
        serializable_results = {
            "results": self.test_results["results"],
            "summary": self.test_results["summary"],
            "execution_time": self.test_results["execution_time"]
        }
        
        output_file.write_text(json.dumps(serializable_results, indent=2))
        logger.info(f"Results saved to {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Orchestrate test execution")
    parser.add_argument("test_dir", help="Test directory")
    parser.add_argument("--workers", type=int, help="Max parallel workers")
    parser.add_argument("--tags", nargs="+", help="Filter tests by tags")
    parser.add_argument("--sequential", action="store_true", help="Run tests sequentially")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--report", help="Output file for report (text)")
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)
    
    orchestrator = TestOrchestrator(test_dir, max_workers=args.workers)
    
    # Discover tests
    orchestrator.discover_tests()
    
    # Plan execution
    tests = orchestrator.plan_execution(filter_tags=args.tags)
    print(f"Found {len(tests)} tests to execute")
    
    # Execute tests
    results = orchestrator.execute_tests(tests, parallel=not args.sequential)
    
    # Generate report
    report = orchestrator.generate_report()
    print(report)
    
    if args.report:
        Path(args.report).write_text(report)
        print(f"\nReport saved to: {args.report}")
    
    # Save results
    if args.output:
        orchestrator.save_results(Path(args.output))
        print(f"Results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results["summary"]["failed"] == 0 else 1)


if __name__ == "__main__":
    main()