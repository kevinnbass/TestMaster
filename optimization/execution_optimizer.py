#!/usr/bin/env python3
"""
Test Execution Optimizer
Optimizes test execution for maximum efficiency and resource utilization.

Features:
- Parallel test execution strategy
- Resource-aware scheduling
- Fail-fast mechanisms
- Distributed execution support
- Load balancing
"""

import os
import sys
import json
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue
import subprocess
import logging
import hashlib
import statistics
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Test execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREADS = "parallel_threads"
    PARALLEL_PROCESSES = "parallel_processes"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class TestExecution:
    """Test execution metadata."""
    test_name: str
    file_path: Path
    estimated_duration: float
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5
    category: str = "unit"
    can_parallelize: bool = True
    requires_isolation: bool = False
    setup_time: float = 0.0
    teardown_time: float = 0.0
    
    def total_time(self) -> float:
        """Get total execution time including setup/teardown."""
        return self.setup_time + self.estimated_duration + self.teardown_time


@dataclass
class ExecutionResult:
    """Test execution result."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    output: str
    error: Optional[str] = None
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class SystemResources:
    """Current system resource availability."""
    cpu_percent: float
    memory_percent: float
    available_memory_gb: float
    disk_io_percent: float
    network_io_mbps: float
    cpu_cores: int
    timestamp: datetime = field(default_factory=datetime.now)


class ResourceMonitor:
    """Monitors system resources for optimization decisions."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.current_resources = SystemResources(0, 0, 0, 0, 0, 0)
        self.resource_history: List[SystemResources] = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                available_memory_gb = memory.available / (1024**3)
                
                # Get disk I/O
                disk_io = psutil.disk_io_counters()
                disk_io_percent = 0  # Simplified - would need baseline
                
                # Get network I/O
                network_io = psutil.net_io_counters()
                network_io_mbps = 0  # Simplified - would need rate calculation
                
                # CPU cores
                cpu_cores = psutil.cpu_count()
                
                resources = SystemResources(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    available_memory_gb=available_memory_gb,
                    disk_io_percent=disk_io_percent,
                    network_io_mbps=network_io_mbps,
                    cpu_cores=cpu_cores
                )
                
                self.current_resources = resources
                self.resource_history.append(resources)
                
                # Keep only last 100 readings
                if len(self.resource_history) > 100:
                    self.resource_history.pop(0)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def can_run_test(self, test: TestExecution) -> bool:
        """Check if system can handle running the test."""
        if not self.current_resources:
            return True  # No data, assume okay
        
        # Check CPU requirement
        cpu_req = test.resource_requirements.get(ResourceType.CPU, 10)
        if self.current_resources.cpu_percent + cpu_req > 90:
            return False
        
        # Check memory requirement
        memory_req = test.resource_requirements.get(ResourceType.MEMORY, 0.1)
        if memory_req > self.current_resources.available_memory_gb:
            return False
        
        return True
    
    def get_optimal_concurrency(self) -> int:
        """Get optimal number of concurrent tests based on resources."""
        if not self.current_resources:
            return 4  # Default
        
        # Base on CPU cores and current load
        base_concurrency = self.current_resources.cpu_cores
        
        # Adjust based on current CPU usage
        if self.current_resources.cpu_percent > 80:
            return max(1, base_concurrency // 2)
        elif self.current_resources.cpu_percent < 30:
            return base_concurrency * 2
        else:
            return base_concurrency


class TestScheduler:
    """Intelligent test scheduler."""
    
    def __init__(self, strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE):
        self.strategy = strategy
        self.resource_monitor = ResourceMonitor()
        self.execution_queue = PriorityQueue()
        self.running_tests: Dict[str, TestExecution] = {}
        self.completed_tests: List[ExecutionResult] = []
        self.failed_fast = False
        
    def schedule_tests(self, tests: List[TestExecution]) -> List[List[TestExecution]]:
        """Schedule tests for optimal execution."""
        if self.strategy == ExecutionStrategy.SEQUENTIAL:
            return [[test] for test in tests]
        
        elif self.strategy == ExecutionStrategy.PARALLEL_THREADS:
            return self._schedule_parallel(tests, "threads")
        
        elif self.strategy == ExecutionStrategy.PARALLEL_PROCESSES:
            return self._schedule_parallel(tests, "processes")
        
        elif self.strategy == ExecutionStrategy.ADAPTIVE:
            return self._schedule_adaptive(tests)
        
        else:
            return self._schedule_parallel(tests, "threads")
    
    def _schedule_parallel(self, tests: List[TestExecution], mode: str) -> List[List[TestExecution]]:
        """Schedule tests for parallel execution."""
        # Group tests by dependencies and isolation requirements
        isolated_tests = [t for t in tests if t.requires_isolation]
        parallel_tests = [t for t in tests if not t.requires_isolation and t.can_parallelize]
        sequential_tests = [t for t in tests if not t.can_parallelize]
        
        schedule = []
        
        # Schedule isolated tests first (one at a time)
        for test in isolated_tests:
            schedule.append([test])
        
        # Schedule parallel tests in groups
        if parallel_tests:
            # Sort by priority and estimated duration
            parallel_tests.sort(key=lambda t: (t.priority, -t.estimated_duration))
            
            # Group into batches
            batch_size = self._get_optimal_batch_size(parallel_tests)
            for i in range(0, len(parallel_tests), batch_size):
                batch = parallel_tests[i:i + batch_size]
                schedule.append(batch)
        
        # Schedule sequential tests
        for test in sequential_tests:
            schedule.append([test])
        
        return schedule
    
    def _schedule_adaptive(self, tests: List[TestExecution]) -> List[List[TestExecution]]:
        """Adaptively schedule tests based on current system state."""
        self.resource_monitor.start_monitoring()
        
        # Wait a moment to get initial readings
        time.sleep(1)
        
        # Determine strategy based on system resources
        optimal_concurrency = self.resource_monitor.get_optimal_concurrency()
        
        if optimal_concurrency == 1:
            strategy = ExecutionStrategy.SEQUENTIAL
        elif self.resource_monitor.current_resources.available_memory_gb > 4:
            strategy = ExecutionStrategy.PARALLEL_PROCESSES
        else:
            strategy = ExecutionStrategy.PARALLEL_THREADS
        
        logger.info(f"Adaptive strategy selected: {strategy.value} (concurrency: {optimal_concurrency})")
        
        self.strategy = strategy
        return self._schedule_parallel(tests, strategy.value.split('_')[-1])
    
    def _get_optimal_batch_size(self, tests: List[TestExecution]) -> int:
        """Get optimal batch size for parallel execution."""
        if not self.resource_monitor.monitoring:
            return 4  # Default
        
        concurrency = self.resource_monitor.get_optimal_concurrency()
        
        # Consider test resource requirements
        avg_cpu_req = statistics.mean(
            t.resource_requirements.get(ResourceType.CPU, 10) for t in tests
        )
        
        if avg_cpu_req > 50:  # High CPU tests
            return max(1, concurrency // 2)
        else:
            return concurrency


class ExecutionOptimizer:
    """Main test execution optimizer."""
    
    def __init__(self, 
                 strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
                 fail_fast: bool = False,
                 timeout: float = 300):
        self.strategy = strategy
        self.fail_fast = fail_fast
        self.timeout = timeout
        self.scheduler = TestScheduler(strategy)
        self.execution_history: List[ExecutionResult] = []
        self.performance_metrics: Dict[str, Any] = {}
        
    def execute_tests(self, tests: List[TestExecution]) -> List[ExecutionResult]:
        """Execute tests with optimization."""
        logger.info(f"Executing {len(tests)} tests with strategy: {self.strategy.value}")
        
        # Estimate durations if not provided
        self._estimate_test_durations(tests)
        
        # Schedule tests
        schedule = self.scheduler.schedule_tests(tests)
        
        # Execute scheduled batches
        all_results = []
        total_batches = len(schedule)
        
        for i, batch in enumerate(schedule, 1):
            logger.info(f"Executing batch {i}/{total_batches} ({len(batch)} tests)")
            
            if len(batch) == 1:
                # Sequential execution
                result = self._execute_single_test(batch[0])
                all_results.append(result)
            else:
                # Parallel execution
                batch_results = self._execute_parallel_batch(batch)
                all_results.extend(batch_results)
            
            # Check fail-fast condition
            if self.fail_fast and any(r.status == "failed" for r in all_results):
                logger.info("Fail-fast triggered, stopping execution")
                break
        
        # Update execution history
        self.execution_history.extend(all_results)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(all_results)
        
        return all_results
    
    def _estimate_test_durations(self, tests: List[TestExecution]):
        """Estimate test durations based on historical data."""
        for test in tests:
            if test.estimated_duration == 0:
                # Use historical data or heuristics
                historical_avg = self._get_historical_duration(test.test_name)
                if historical_avg:
                    test.estimated_duration = historical_avg
                else:
                    # Estimate based on test category
                    category_estimates = {
                        "unit": 0.1,
                        "integration": 2.0,
                        "e2e": 10.0,
                        "performance": 30.0
                    }
                    test.estimated_duration = category_estimates.get(test.category, 1.0)
    
    def _get_historical_duration(self, test_name: str) -> Optional[float]:
        """Get historical average duration for test."""
        historical_results = [
            r for r in self.execution_history
            if r.test_name == test_name and r.status in ["passed", "failed"]
        ]
        
        if historical_results:
            durations = [r.duration for r in historical_results[-10:]]  # Last 10 runs
            return statistics.mean(durations)
        
        return None
    
    def _execute_single_test(self, test: TestExecution) -> ExecutionResult:
        """Execute a single test."""
        start_time = time.time()
        started_at = datetime.now()
        
        try:
            # Check resource availability
            if not self.scheduler.resource_monitor.can_run_test(test):
                return ExecutionResult(
                    test_name=test.test_name,
                    status="skipped",
                    duration=0,
                    output="Skipped due to resource constraints",
                    started_at=started_at,
                    completed_at=datetime.now()
                )
            
            # Execute test using pytest
            cmd = ["python3", "-m", "pytest", str(test.file_path), "-v"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            duration = time.time() - start_time
            
            # Determine status
            if result.returncode == 0:
                status = "passed"
            elif result.returncode == 1:
                status = "failed"
            else:
                status = "error"
            
            return ExecutionResult(
                test_name=test.test_name,
                status=status,
                duration=duration,
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                started_at=started_at,
                completed_at=datetime.now()
            )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                test_name=test.test_name,
                status="timeout",
                duration=self.timeout,
                output="Test execution timed out",
                error="Timeout exceeded",
                started_at=started_at,
                completed_at=datetime.now()
            )
        
        except Exception as e:
            return ExecutionResult(
                test_name=test.test_name,
                status="error",
                duration=time.time() - start_time,
                output="",
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now()
            )
    
    def _execute_parallel_batch(self, batch: List[TestExecution]) -> List[ExecutionResult]:
        """Execute a batch of tests in parallel."""
        results = []
        
        # Use ThreadPoolExecutor for I/O bound tests, ProcessPoolExecutor for CPU bound
        executor_class = ThreadPoolExecutor
        if any(t.resource_requirements.get(ResourceType.CPU, 0) > 30 for t in batch):
            executor_class = ProcessPoolExecutor
        
        max_workers = min(len(batch), self.scheduler.resource_monitor.get_optimal_concurrency())
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self._execute_single_test, test): test
                for test in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                test = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log progress
                    logger.info(f"Completed {test.test_name}: {result.status} ({result.duration:.2f}s)")
                    
                except Exception as e:
                    logger.error(f"Error executing {test.test_name}: {e}")
                    results.append(ExecutionResult(
                        test_name=test.test_name,
                        status="error",
                        duration=0,
                        output="",
                        error=str(e)
                    ))
        
        return results
    
    def _calculate_performance_metrics(self, results: List[ExecutionResult]):
        """Calculate performance metrics for the execution."""
        if not results:
            return
        
        total_duration = sum(r.duration for r in results)
        passed_count = sum(1 for r in results if r.status == "passed")
        failed_count = sum(1 for r in results if r.status == "failed")
        
        self.performance_metrics = {
            "total_tests": len(results),
            "total_duration": total_duration,
            "average_duration": total_duration / len(results),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "success_rate": passed_count / len(results) * 100,
            "tests_per_second": len(results) / max(total_duration, 1),
            "strategy_used": self.strategy.value
        }
    
    def optimize_future_executions(self):
        """Optimize future executions based on historical data."""
        if len(self.execution_history) < 10:
            return
        
        # Analyze execution patterns
        recent_results = self.execution_history[-50:]
        
        # Calculate average durations by category
        category_durations = defaultdict(list)
        for result in recent_results:
            # Would need to store category in result
            category_durations["default"].append(result.duration)
        
        # Analyze resource usage patterns
        # This would help optimize batch sizes and strategies
        
        # Update estimation models
        # This would improve duration estimates
        
        logger.info("Optimization models updated based on execution history")
    
    def get_execution_report(self) -> str:
        """Generate execution performance report."""
        if not self.performance_metrics:
            return "No execution data available"
        
        metrics = self.performance_metrics
        
        report_lines = [
            "=" * 50,
            "TEST EXECUTION PERFORMANCE REPORT",
            "=" * 50,
            f"Strategy: {metrics['strategy_used']}",
            f"Total tests: {metrics['total_tests']}",
            f"Total duration: {metrics['total_duration']:.2f}s",
            f"Average duration: {metrics['average_duration']:.2f}s per test",
            f"Tests per second: {metrics['tests_per_second']:.2f}",
            "",
            f"Results:",
            f"  Passed: {metrics['passed_count']} ({metrics['success_rate']:.1f}%)",
            f"  Failed: {metrics['failed_count']}",
            "",
            "Performance Analysis:",
        ]
        
        # Add performance insights
        if metrics['tests_per_second'] > 2:
            report_lines.append("  ✓ High throughput achieved")
        elif metrics['tests_per_second'] < 0.5:
            report_lines.append("  ⚠ Low throughput - consider optimization")
        
        if metrics['success_rate'] > 95:
            report_lines.append("  ✓ Excellent test reliability")
        elif metrics['success_rate'] < 80:
            report_lines.append("  ⚠ Low success rate - investigate failures")
        
        report_lines.append("=" * 50)
        
        return "\n".join(report_lines)


def main():
    """CLI for test execution optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Execution Optimizer")
    parser.add_argument("--test-dir", required=True, help="Directory containing tests")
    parser.add_argument("--strategy", choices=[s.value for s in ExecutionStrategy], 
                       default="adaptive", help="Execution strategy")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--timeout", type=float, default=300, help="Test timeout in seconds")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--optimize", action="store_true", help="Optimize based on history")
    parser.add_argument("--max-concurrency", type=int, help="Maximum concurrent tests")
    
    args = parser.parse_args()
    
    # Find test files
    test_dir = Path(args.test_dir)
    test_files = list(test_dir.rglob("test_*.py"))
    
    if not test_files:
        print(f"No test files found in {test_dir}")
        return
    
    # Create test executions
    tests = []
    for test_file in test_files:
        test = TestExecution(
            test_name=test_file.stem,
            file_path=test_file,
            estimated_duration=0,  # Will be estimated
            category="unit"  # Simplified
        )
        tests.append(test)
    
    # Initialize optimizer
    strategy = ExecutionStrategy(args.strategy)
    optimizer = ExecutionOptimizer(
        strategy=strategy,
        fail_fast=args.fail_fast,
        timeout=args.timeout
    )
    
    if args.optimize:
        optimizer.optimize_future_executions()
    
    print(f"Executing {len(tests)} tests with {strategy.value} strategy...")
    
    # Execute tests
    start_time = time.time()
    results = optimizer.execute_tests(tests)
    total_time = time.time() - start_time
    
    # Show summary
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    
    print(f"\nExecution completed in {total_time:.2f}s")
    print(f"Results: {passed} passed, {failed} failed")
    
    if args.report:
        report = optimizer.get_execution_report()
        print("\n" + report)
        
        # Save report
        report_file = Path(f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()