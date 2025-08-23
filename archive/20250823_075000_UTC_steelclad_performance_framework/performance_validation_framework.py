#!/usr/bin/env python3
"""
AGENT BETA - PERFORMANCE VALIDATION FRAMEWORK
Phase 1, Hours 20-25: Initial Performance Validation
===================================================

Comprehensive performance validation system with benchmarking, load testing,
regression detection, and performance measurement reporting.

Created: 2025-08-23 02:55:00 UTC
Agent: Beta (Performance Optimization Specialist)
Phase: 1 (Hours 20-25)
"""

import os
import sys
import time
import json
import asyncio
import aiohttp
import threading
import concurrent.futures
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import logging
import psutil
import sqlite3
from contextlib import contextmanager
import subprocess
import requests
import socket
import random
import uuid

# Import previous optimization systems
try:
    from performance_monitoring_infrastructure import PerformanceMonitoringSystem, MonitoringConfig
    from database_performance_optimizer import DatabasePerformanceOptimizer
    from memory_management_optimizer import MemoryManager
    OPTIMIZATION_SYSTEMS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_SYSTEMS_AVAILABLE = False

@dataclass
class PerformanceTest:
    """Individual performance test definition"""
    test_id: str
    test_name: str
    test_type: str  # 'load', 'stress', 'volume', 'spike', 'endurance'
    target_function: Optional[Callable] = None
    target_url: Optional[str] = None
    expected_response_time_ms: float = 100.0
    max_response_time_ms: float = 1000.0
    concurrent_users: int = 1
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    success_criteria: Dict[str, float] = None
    
    def __post_init__(self):
        if self.success_criteria is None:
            self.success_criteria = {
                'avg_response_time_ms': self.expected_response_time_ms,
                'max_response_time_ms': self.max_response_time_ms,
                'success_rate_percent': 95.0,
                'throughput_requests_per_second': 10.0
            }

@dataclass
class PerformanceResult:
    """Performance test execution result"""
    test_id: str
    execution_timestamp: datetime
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    memory_usage_mb: float
    cpu_usage_percent: float
    errors: List[str] = None
    success: bool = True
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass
class LoadTestScenario:
    """Load test scenario configuration"""
    name: str
    base_url: str
    endpoints: List[Dict[str, Any]]
    user_scenarios: List[Dict[str, Any]]
    ramp_up_pattern: str  # 'linear', 'exponential', 'step'
    max_concurrent_users: int
    test_duration_minutes: int
    think_time_seconds: Tuple[float, float] = (0.5, 2.0)  # min, max

class PerformanceBenchmarker:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self, output_dir: str = "performance_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Results storage
        self.test_results: Dict[str, List[PerformanceResult]] = defaultdict(list)
        self.benchmark_history: List[Dict] = []
        
        # Initialize results database
        self.db_path = self.output_dir / "performance_results.db"
        self._init_results_db()
        
        # Performance test registry
        self.registered_tests: Dict[str, PerformanceTest] = {}
        
        # Load testing components
        self.load_test_executor = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('PerformanceBenchmarker')
        
        # Register default performance tests
        self._register_default_tests()
    
    def _init_results_db(self):
        """Initialize performance results database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    execution_timestamp TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    total_requests INTEGER NOT NULL,
                    successful_requests INTEGER NOT NULL,
                    failed_requests INTEGER NOT NULL,
                    avg_response_time_ms REAL NOT NULL,
                    min_response_time_ms REAL NOT NULL,
                    max_response_time_ms REAL NOT NULL,
                    p50_response_time_ms REAL NOT NULL,
                    p95_response_time_ms REAL NOT NULL,
                    p99_response_time_ms REAL NOT NULL,
                    throughput_rps REAL NOT NULL,
                    error_rate_percent REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    cpu_usage_percent REAL NOT NULL,
                    errors TEXT,
                    success BOOLEAN NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    session_timestamp TEXT NOT NULL,
                    session_name TEXT NOT NULL,
                    total_tests INTEGER NOT NULL,
                    passed_tests INTEGER NOT NULL,
                    failed_tests INTEGER NOT NULL,
                    overall_success BOOLEAN NOT NULL,
                    session_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_results_timestamp 
                ON performance_results(execution_timestamp)
            """)
    
    def _register_default_tests(self):
        """Register default performance tests"""
        # CPU-intensive test
        self.register_test(PerformanceTest(
            test_id="cpu_intensive",
            test_name="CPU Intensive Computation",
            test_type="stress",
            target_function=self._cpu_intensive_operation,
            expected_response_time_ms=50.0,
            max_response_time_ms=200.0,
            concurrent_users=1,
            duration_seconds=30
        ))
        
        # Memory-intensive test
        self.register_test(PerformanceTest(
            test_id="memory_intensive",
            test_name="Memory Allocation Test",
            test_type="stress",
            target_function=self._memory_intensive_operation,
            expected_response_time_ms=100.0,
            max_response_time_ms=500.0,
            concurrent_users=1,
            duration_seconds=30
        ))
        
        # I/O-intensive test
        self.register_test(PerformanceTest(
            test_id="io_intensive",
            test_name="File I/O Operations",
            test_type="stress",
            target_function=self._io_intensive_operation,
            expected_response_time_ms=20.0,
            max_response_time_ms=100.0,
            concurrent_users=5,
            duration_seconds=30
        ))
        
        # Concurrency test
        self.register_test(PerformanceTest(
            test_id="concurrency_test",
            test_name="Concurrent Operations",
            test_type="load",
            target_function=self._concurrent_operation,
            expected_response_time_ms=30.0,
            max_response_time_ms=150.0,
            concurrent_users=10,
            duration_seconds=60
        ))
        
        self.logger.info(f"Registered {len(self.registered_tests)} default performance tests")
    
    def _cpu_intensive_operation(self) -> Dict[str, Any]:
        """CPU-intensive test operation"""
        start_time = time.perf_counter()
        
        # Prime number calculation
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        # Find primes up to 10000
        primes = [n for n in range(2, 10000) if is_prime(n)]
        
        end_time = time.perf_counter()
        
        return {
            'result': len(primes),
            'execution_time': end_time - start_time,
            'operation_type': 'cpu_intensive'
        }
    
    def _memory_intensive_operation(self) -> Dict[str, Any]:
        """Memory-intensive test operation"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        # Create large data structures
        data = []
        for i in range(1000):
            row = [j * random.random() for j in range(1000)]
            data.append(row)
        
        # Process the data
        total = sum(sum(row) for row in data)
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        # Cleanup
        del data
        
        return {
            'result': total,
            'execution_time': end_time - start_time,
            'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
            'operation_type': 'memory_intensive'
        }
    
    def _io_intensive_operation(self) -> Dict[str, Any]:
        """I/O-intensive test operation"""
        start_time = time.perf_counter()
        
        # Create temporary file and perform I/O operations
        temp_file = self.output_dir / f"temp_io_{uuid.uuid4().hex}.txt"
        
        try:
            # Write data
            with open(temp_file, 'w') as f:
                for i in range(1000):
                    f.write(f"Test line {i} with some data {random.random()}\n")
            
            # Read data back
            lines = []
            with open(temp_file, 'r') as f:
                lines = f.readlines()
            
            # Process data
            word_count = sum(len(line.split()) for line in lines)
            
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
        
        end_time = time.perf_counter()
        
        return {
            'result': word_count,
            'execution_time': end_time - start_time,
            'lines_processed': len(lines),
            'operation_type': 'io_intensive'
        }
    
    def _concurrent_operation(self) -> Dict[str, Any]:
        """Concurrent operation test"""
        start_time = time.perf_counter()
        
        # Simulate concurrent work
        import threading
        results = []
        threads = []
        
        def worker(worker_id):
            # Simulate some work
            time.sleep(random.uniform(0.01, 0.05))
            result = sum(i * i for i in range(100))
            results.append(result)
        
        # Start multiple threads
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        
        return {
            'result': sum(results),
            'execution_time': end_time - start_time,
            'threads_used': len(threads),
            'operation_type': 'concurrent'
        }
    
    def register_test(self, test: PerformanceTest):
        """Register a performance test"""
        self.registered_tests[test.test_id] = test
        self.logger.info(f"Registered performance test: {test.test_name} ({test.test_id})")
    
    def execute_single_test(self, test_id: str) -> PerformanceResult:
        """Execute a single performance test"""
        if test_id not in self.registered_tests:
            raise ValueError(f"Test {test_id} not registered")
        
        test = self.registered_tests[test_id]
        self.logger.info(f"Executing performance test: {test.test_name}")
        
        # Collect baseline metrics
        start_time = datetime.now(timezone.utc)
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        start_cpu = psutil.Process().cpu_percent()
        
        # Execute test with multiple iterations for concurrent users
        response_times = []
        errors = []
        successful_requests = 0
        failed_requests = 0
        
        execution_start = time.perf_counter()
        
        # Single user test
        if test.concurrent_users == 1:
            end_time = time.time() + test.duration_seconds
            while time.time() < end_time:
                try:
                    request_start = time.perf_counter()
                    result = test.target_function()
                    request_end = time.perf_counter()
                    
                    response_time_ms = (request_end - request_start) * 1000
                    response_times.append(response_time_ms)
                    successful_requests += 1
                    
                except Exception as e:
                    errors.append(str(e))
                    failed_requests += 1
                    response_times.append(test.max_response_time_ms)  # Penalty time
        
        # Multi-user concurrent test
        else:
            def worker():
                worker_errors = []
                worker_times = []
                worker_success = 0
                worker_failed = 0
                
                end_time = time.time() + test.duration_seconds
                while time.time() < end_time:
                    try:
                        request_start = time.perf_counter()
                        result = test.target_function()
                        request_end = time.perf_counter()
                        
                        response_time_ms = (request_end - request_start) * 1000
                        worker_times.append(response_time_ms)
                        worker_success += 1
                        
                    except Exception as e:
                        worker_errors.append(str(e))
                        worker_failed += 1
                        worker_times.append(test.max_response_time_ms)
                    
                    # Add think time for realistic load
                    time.sleep(random.uniform(0.01, 0.1))
                
                return worker_times, worker_errors, worker_success, worker_failed
            
            # Execute concurrent workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=test.concurrent_users) as executor:
                futures = [executor.submit(worker) for _ in range(test.concurrent_users)]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        worker_times, worker_errors, worker_success, worker_failed = future.result()
                        response_times.extend(worker_times)
                        errors.extend(worker_errors)
                        successful_requests += worker_success
                        failed_requests += worker_failed
                    except Exception as e:
                        errors.append(f"Worker execution error: {str(e)}")
                        failed_requests += 1
        
        execution_end = time.perf_counter()
        
        # Collect final metrics
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        end_cpu = psutil.Process().cpu_percent()
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            sorted_times = sorted(response_times)
            p50_response_time = sorted_times[int(len(sorted_times) * 0.5)]
            p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
            p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        total_requests = successful_requests + failed_requests
        duration = execution_end - execution_start
        throughput = total_requests / duration if duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Check success criteria
        success = (
            avg_response_time <= test.success_criteria['avg_response_time_ms'] and
            max_response_time <= test.success_criteria['max_response_time_ms'] and
            (100 - error_rate) >= test.success_criteria['success_rate_percent'] and
            throughput >= test.success_criteria['throughput_requests_per_second']
        )
        
        result = PerformanceResult(
            test_id=test_id,
            execution_timestamp=start_time,
            duration_seconds=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            throughput_rps=throughput,
            error_rate_percent=error_rate,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=end_cpu,
            errors=errors[:10],  # Limit error list
            success=success
        )
        
        # Store result
        self.test_results[test_id].append(result)
        self._store_result_db(result)
        
        self.logger.info(f"Test {test.test_name} completed: "
                        f"{'PASSED' if success else 'FAILED'} "
                        f"(avg: {avg_response_time:.2f}ms, throughput: {throughput:.1f} rps)")
        
        return result
    
    def _store_result_db(self, result: PerformanceResult):
        """Store performance result in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_results (
                    test_id, execution_timestamp, duration_seconds, total_requests,
                    successful_requests, failed_requests, avg_response_time_ms,
                    min_response_time_ms, max_response_time_ms, p50_response_time_ms,
                    p95_response_time_ms, p99_response_time_ms, throughput_rps,
                    error_rate_percent, memory_usage_mb, cpu_usage_percent,
                    errors, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_id, result.execution_timestamp.isoformat(),
                result.duration_seconds, result.total_requests,
                result.successful_requests, result.failed_requests,
                result.avg_response_time_ms, result.min_response_time_ms,
                result.max_response_time_ms, result.p50_response_time_ms,
                result.p95_response_time_ms, result.p99_response_time_ms,
                result.throughput_rps, result.error_rate_percent,
                result.memory_usage_mb, result.cpu_usage_percent,
                json.dumps(result.errors), result.success
            ))
    
    def execute_benchmark_suite(self, test_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a complete benchmark suite"""
        if test_ids is None:
            test_ids = list(self.registered_tests.keys())
        
        session_id = str(uuid.uuid4())
        session_start = datetime.now(timezone.utc)
        
        self.logger.info(f"Starting benchmark suite: {len(test_ids)} tests")
        
        suite_results = {}
        passed_tests = 0
        failed_tests = 0
        
        # Execute each test
        for test_id in test_ids:
            try:
                result = self.execute_single_test(test_id)
                suite_results[test_id] = result
                
                if result.success:
                    passed_tests += 1
                else:
                    failed_tests += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to execute test {test_id}: {e}")
                failed_tests += 1
                suite_results[test_id] = None
        
        # Calculate overall success
        overall_success = failed_tests == 0
        
        # Store session results
        session_data = {
            'session_id': session_id,
            'start_time': session_start.isoformat(),
            'end_time': datetime.now(timezone.utc).isoformat(),
            'test_ids': test_ids,
            'results_summary': {
                'total_tests': len(test_ids),
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / len(test_ids)) * 100 if test_ids else 0
            }
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO benchmark_sessions (
                    session_id, session_timestamp, session_name, total_tests,
                    passed_tests, failed_tests, overall_success, session_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, session_start.isoformat(), "Default Benchmark Suite",
                len(test_ids), passed_tests, failed_tests, overall_success,
                json.dumps(session_data)
            ))
        
        self.benchmark_history.append(session_data)
        
        self.logger.info(f"Benchmark suite completed: {passed_tests}/{len(test_ids)} tests passed")
        
        return {
            'session_id': session_id,
            'overall_success': overall_success,
            'results': suite_results,
            'summary': session_data['results_summary']
        }

class LoadTestExecutor:
    """Advanced load testing system"""
    
    def __init__(self):
        self.logger = logging.getLogger('LoadTestExecutor')
        self.active_tests: Dict[str, bool] = {}
    
    async def execute_http_load_test(self, scenario: LoadTestScenario) -> Dict[str, Any]:
        """Execute HTTP load test scenario"""
        self.logger.info(f"Starting load test: {scenario.name}")
        
        results = {
            'scenario_name': scenario.name,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'requests': [],
            'summary': {}
        }
        
        # Track active test
        test_id = scenario.name
        self.active_tests[test_id] = True
        
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                
                # Execute load test
                tasks = []
                for user_id in range(scenario.max_concurrent_users):
                    task = asyncio.create_task(
                        self._simulate_user_session(session, scenario, user_id, results)
                    )
                    tasks.append(task)
                    
                    # Gradual ramp-up
                    if scenario.ramp_up_pattern == 'linear':
                        ramp_delay = (scenario.test_duration_minutes * 60) / scenario.max_concurrent_users
                        if user_id < scenario.max_concurrent_users - 1:
                            await asyncio.sleep(ramp_delay)
                
                # Wait for all users to complete
                await asyncio.gather(*tasks, return_exceptions=True)
        
        finally:
            self.active_tests[test_id] = False
        
        # Calculate summary statistics
        requests = results['requests']
        if requests:
            response_times = [r['response_time_ms'] for r in requests if r['success']]
            
            results['summary'] = {
                'total_requests': len(requests),
                'successful_requests': len([r for r in requests if r['success']]),
                'failed_requests': len([r for r in requests if not r['success']]),
                'avg_response_time_ms': statistics.mean(response_times) if response_times else 0,
                'p95_response_time_ms': (
                    statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else
                    max(response_times) if response_times else 0
                ),
                'throughput_rps': len(requests) / (scenario.test_duration_minutes * 60),
                'error_rate_percent': (
                    (len([r for r in requests if not r['success']]) / len(requests)) * 100
                    if requests else 0
                )
            }
        
        results['end_time'] = datetime.now(timezone.utc).isoformat()
        
        self.logger.info(f"Load test completed: {scenario.name}")
        return results
    
    async def _simulate_user_session(self, session: aiohttp.ClientSession, 
                                   scenario: LoadTestScenario, user_id: int, 
                                   results: Dict[str, Any]):
        """Simulate a single user session"""
        end_time = time.time() + (scenario.test_duration_minutes * 60)
        
        while time.time() < end_time and self.active_tests.get(scenario.name, False):
            # Select random endpoint
            if scenario.endpoints:
                endpoint = random.choice(scenario.endpoints)
                url = f"{scenario.base_url}{endpoint['path']}"
                method = endpoint.get('method', 'GET').upper()
                
                # Execute request
                request_start = time.perf_counter()
                success = False
                status_code = 0
                error_message = None
                
                try:
                    async with session.request(method, url) as response:
                        await response.text()  # Read response body
                        status_code = response.status
                        success = 200 <= status_code < 400
                        
                except Exception as e:
                    error_message = str(e)
                    success = False
                
                request_end = time.perf_counter()
                response_time_ms = (request_end - request_start) * 1000
                
                # Record result
                request_result = {
                    'user_id': user_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'url': url,
                    'method': method,
                    'response_time_ms': response_time_ms,
                    'status_code': status_code,
                    'success': success,
                    'error_message': error_message
                }
                
                results['requests'].append(request_result)
            
            # Think time between requests
            think_time = random.uniform(*scenario.think_time_seconds)
            await asyncio.sleep(think_time)

class PerformanceRegressionDetector:
    """Detects performance regressions by comparing results"""
    
    def __init__(self, benchmarker: PerformanceBenchmarker):
        self.benchmarker = benchmarker
        self.logger = logging.getLogger('PerformanceRegressionDetector')
        
        # Regression thresholds
        self.regression_thresholds = {
            'response_time_increase_percent': 10.0,  # 10% increase
            'throughput_decrease_percent': 10.0,     # 10% decrease
            'error_rate_increase_percent': 5.0,      # 5% increase
            'memory_increase_percent': 20.0          # 20% increase
        }
    
    def detect_regressions(self, current_results: Dict[str, PerformanceResult], 
                          baseline_results: Optional[Dict[str, PerformanceResult]] = None) -> Dict[str, Any]:
        """Detect performance regressions"""
        
        if baseline_results is None:
            baseline_results = self._get_latest_baseline()
        
        regression_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regressions_detected': [],
            'improvements_detected': [],
            'summary': {
                'total_tests_compared': 0,
                'regressions_count': 0,
                'improvements_count': 0,
                'stable_count': 0
            }
        }
        
        for test_id, current_result in current_results.items():
            if current_result is None:
                continue
                
            baseline_result = baseline_results.get(test_id)
            if baseline_result is None:
                continue
            
            regression_report['summary']['total_tests_compared'] += 1
            
            # Compare metrics
            comparison = self._compare_results(current_result, baseline_result)
            
            if comparison['has_regression']:
                regression_report['regressions_detected'].append({
                    'test_id': test_id,
                    'test_name': self.benchmarker.registered_tests.get(test_id, {}).test_name,
                    'regression_type': comparison['regression_type'],
                    'current_value': comparison['current_value'],
                    'baseline_value': comparison['baseline_value'],
                    'change_percent': comparison['change_percent'],
                    'severity': comparison['severity']
                })
                regression_report['summary']['regressions_count'] += 1
                
            elif comparison['has_improvement']:
                regression_report['improvements_detected'].append({
                    'test_id': test_id,
                    'improvement_type': comparison['improvement_type'],
                    'current_value': comparison['current_value'],
                    'baseline_value': comparison['baseline_value'],
                    'improvement_percent': comparison['change_percent']
                })
                regression_report['summary']['improvements_count'] += 1
                
            else:
                regression_report['summary']['stable_count'] += 1
        
        return regression_report
    
    def _get_latest_baseline(self) -> Dict[str, PerformanceResult]:
        """Get latest baseline results from database"""
        baseline_results = {}
        
        with sqlite3.connect(self.benchmarker.db_path) as conn:
            # Get latest successful results for each test
            cursor = conn.execute("""
                SELECT DISTINCT test_id FROM performance_results 
                WHERE success = 1 
                ORDER BY execution_timestamp DESC
            """)
            
            test_ids = [row[0] for row in cursor.fetchall()]
            
            for test_id in test_ids:
                cursor = conn.execute("""
                    SELECT * FROM performance_results 
                    WHERE test_id = ? AND success = 1 
                    ORDER BY execution_timestamp DESC 
                    LIMIT 2
                """, (test_id,))
                
                rows = cursor.fetchall()
                if len(rows) >= 2:  # Use second-latest as baseline
                    row = rows[1]
                    baseline_results[test_id] = PerformanceResult(
                        test_id=row[1],
                        execution_timestamp=datetime.fromisoformat(row[2]),
                        duration_seconds=row[3],
                        total_requests=row[4],
                        successful_requests=row[5],
                        failed_requests=row[6],
                        avg_response_time_ms=row[7],
                        min_response_time_ms=row[8],
                        max_response_time_ms=row[9],
                        p50_response_time_ms=row[10],
                        p95_response_time_ms=row[11],
                        p99_response_time_ms=row[12],
                        throughput_rps=row[13],
                        error_rate_percent=row[14],
                        memory_usage_mb=row[15],
                        cpu_usage_percent=row[16],
                        errors=json.loads(row[17]) if row[17] else [],
                        success=bool(row[18])
                    )
        
        return baseline_results
    
    def _compare_results(self, current: PerformanceResult, baseline: PerformanceResult) -> Dict[str, Any]:
        """Compare two performance results"""
        
        # Calculate percentage changes
        response_time_change = ((current.avg_response_time_ms - baseline.avg_response_time_ms) / 
                               baseline.avg_response_time_ms) * 100 if baseline.avg_response_time_ms > 0 else 0
        
        throughput_change = ((current.throughput_rps - baseline.throughput_rps) / 
                           baseline.throughput_rps) * 100 if baseline.throughput_rps > 0 else 0
        
        error_rate_change = current.error_rate_percent - baseline.error_rate_percent
        
        memory_change = ((current.memory_usage_mb - baseline.memory_usage_mb) / 
                        abs(baseline.memory_usage_mb)) * 100 if baseline.memory_usage_mb != 0 else 0
        
        # Check for regressions
        has_regression = (
            response_time_change > self.regression_thresholds['response_time_increase_percent'] or
            throughput_change < -self.regression_thresholds['throughput_decrease_percent'] or
            error_rate_change > self.regression_thresholds['error_rate_increase_percent'] or
            memory_change > self.regression_thresholds['memory_increase_percent']
        )
        
        # Check for improvements
        has_improvement = (
            response_time_change < -5.0 or  # 5% improvement in response time
            throughput_change > 5.0 or      # 5% improvement in throughput
            error_rate_change < -1.0        # 1% improvement in error rate
        )
        
        # Determine regression type and severity
        regression_type = None
        severity = 'low'
        current_value = None
        baseline_value = None
        change_percent = 0
        
        if has_regression:
            if response_time_change > self.regression_thresholds['response_time_increase_percent']:
                regression_type = 'response_time'
                current_value = current.avg_response_time_ms
                baseline_value = baseline.avg_response_time_ms
                change_percent = response_time_change
                severity = 'high' if response_time_change > 25 else 'medium' if response_time_change > 15 else 'low'
                
            elif throughput_change < -self.regression_thresholds['throughput_decrease_percent']:
                regression_type = 'throughput'
                current_value = current.throughput_rps
                baseline_value = baseline.throughput_rps
                change_percent = throughput_change
                severity = 'high' if throughput_change < -25 else 'medium' if throughput_change < -15 else 'low'
        
        improvement_type = None
        if has_improvement:
            if response_time_change < -5.0:
                improvement_type = 'response_time'
            elif throughput_change > 5.0:
                improvement_type = 'throughput'
        
        return {
            'has_regression': has_regression,
            'has_improvement': has_improvement,
            'regression_type': regression_type,
            'improvement_type': improvement_type,
            'current_value': current_value,
            'baseline_value': baseline_value,
            'change_percent': abs(change_percent),
            'severity': severity,
            'metrics': {
                'response_time_change_percent': response_time_change,
                'throughput_change_percent': throughput_change,
                'error_rate_change_percent': error_rate_change,
                'memory_change_percent': memory_change
            }
        }

class PerformanceValidationFramework:
    """Main performance validation orchestrator"""
    
    def __init__(self):
        # Set up logging first
        self.logger = logging.getLogger('PerformanceValidationFramework')
        
        # Initialize components
        self.benchmarker = PerformanceBenchmarker()
        self.load_tester = LoadTestExecutor()
        self.regression_detector = PerformanceRegressionDetector(self.benchmarker)
        
        # Integration with optimization systems
        self.monitoring_system = None
        self.database_optimizer = None
        self.memory_manager = None
        
        if OPTIMIZATION_SYSTEMS_AVAILABLE:
            self._initialize_optimization_integration()
    
    def _initialize_optimization_integration(self):
        """Initialize integration with optimization systems"""
        try:
            # Initialize monitoring
            config = MonitoringConfig(
                collection_interval=10.0,
                alert_channels=['console'],
                enable_prometheus=False,
                enable_alerting=True
            )
            self.monitoring_system = PerformanceMonitoringSystem(config)
            
            # Initialize database optimizer
            self.database_optimizer = DatabasePerformanceOptimizer(enable_monitoring=False)
            
            # Initialize memory manager
            self.memory_manager = MemoryManager(enable_monitoring=False)
            
            self.logger.info("Optimization systems integration initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize optimization integration: {e}")
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive performance validation"""
        
        validation_results = {
            'validation_id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'phases': {}
        }
        
        self.logger.info("Starting comprehensive performance validation")
        
        try:
            # Start optimization systems
            if self.monitoring_system:
                self.monitoring_system.start()
            if self.memory_manager:
                self.memory_manager.start()
            
            # Phase 1: Benchmark Suite
            self.logger.info("Phase 1: Executing benchmark suite...")
            benchmark_results = self.benchmarker.execute_benchmark_suite()
            validation_results['phases']['benchmark_suite'] = benchmark_results
            
            # Phase 2: Regression Detection
            self.logger.info("Phase 2: Detecting performance regressions...")
            regression_results = self.regression_detector.detect_regressions(benchmark_results['results'])
            validation_results['phases']['regression_detection'] = regression_results
            
            # Phase 3: Memory Validation (if available)
            if self.memory_manager:
                self.logger.info("Phase 3: Memory management validation...")
                memory_report = self.memory_manager.generate_memory_report()
                validation_results['phases']['memory_validation'] = {
                    'memory_report': memory_report,
                    'memory_cleanup': self.memory_manager.cleanup_memory()
                }
            
            # Phase 4: Database Performance (if available)
            if self.database_optimizer:
                self.logger.info("Phase 4: Database performance validation...")
                db_benchmarks = self.database_optimizer.benchmark_queries()
                validation_results['phases']['database_validation'] = db_benchmarks
            
            # Calculate overall success
            overall_success = (
                benchmark_results['overall_success'] and
                regression_results['summary']['regressions_count'] == 0
            )
            
            validation_results['overall_success'] = overall_success
            validation_results['summary'] = {
                'benchmark_success': benchmark_results['overall_success'],
                'regressions_detected': regression_results['summary']['regressions_count'],
                'improvements_detected': regression_results['summary']['improvements_count'],
                'total_tests': benchmark_results['summary']['total_tests']
            }
            
        finally:
            # Stop optimization systems
            if self.memory_manager:
                self.memory_manager.stop()
            if self.monitoring_system:
                self.monitoring_system.stop()
        
        self.logger.info(f"Comprehensive validation completed: "
                        f"{'SUCCESS' if validation_results['overall_success'] else 'FAILED'}")
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation report"""
        
        lines = [
            "PERFORMANCE VALIDATION COMPREHENSIVE REPORT",
            "=" * 60,
            f"Validation ID: {validation_results['validation_id']}",
            f"Generated: {validation_results['timestamp']}",
            f"Overall Result: {'✅ SUCCESS' if validation_results['overall_success'] else '❌ FAILED'}",
            "",
            "VALIDATION SUMMARY:",
            f"  Total Tests: {validation_results['summary']['total_tests']}",
            f"  Benchmark Success: {'✅' if validation_results['summary']['benchmark_success'] else '❌'}",
            f"  Regressions Detected: {validation_results['summary']['regressions_detected']}",
            f"  Improvements Detected: {validation_results['summary']['improvements_detected']}",
            ""
        ]
        
        # Benchmark results
        if 'benchmark_suite' in validation_results['phases']:
            benchmark = validation_results['phases']['benchmark_suite']
            lines.extend([
                "BENCHMARK SUITE RESULTS:",
                f"  Tests Passed: {benchmark['summary']['passed_tests']}/{benchmark['summary']['total_tests']}",
                f"  Success Rate: {benchmark['summary']['success_rate']:.1f}%",
                ""
            ])
            
            # Individual test results
            for test_id, result in benchmark['results'].items():
                if result:
                    status = "✅ PASS" if result.success else "❌ FAIL"
                    lines.append(f"  {test_id}: {status} "
                               f"(avg: {result.avg_response_time_ms:.2f}ms, "
                               f"throughput: {result.throughput_rps:.1f} rps)")
            lines.append("")
        
        # Regression results
        if 'regression_detection' in validation_results['phases']:
            regression = validation_results['phases']['regression_detection']
            lines.extend([
                "REGRESSION ANALYSIS:",
                f"  Stable Tests: {regression['summary']['stable_count']}",
                f"  Regressions: {regression['summary']['regressions_count']}",
                f"  Improvements: {regression['summary']['improvements_count']}",
                ""
            ])
            
            # List regressions
            for reg in regression['regressions_detected']:
                lines.append(f"  ⚠️  REGRESSION in {reg['test_id']}: "
                           f"{reg['regression_type']} degraded by {reg['change_percent']:.1f}%")
            
            # List improvements
            for imp in regression['improvements_detected']:
                lines.append(f"  ✅ IMPROVEMENT in {imp['test_id']}: "
                           f"{imp['improvement_type']} improved by {imp['improvement_percent']:.1f}%")
            
            if regression['regressions_detected'] or regression['improvements_detected']:
                lines.append("")
        
        # Memory validation
        if 'memory_validation' in validation_results['phases']:
            lines.extend([
                "MEMORY VALIDATION:",
                "  Memory management validation completed successfully",
                f"  Memory cleanup performed",
                ""
            ])
        
        # Database validation
        if 'database_validation' in validation_results['phases']:
            lines.extend([
                "DATABASE VALIDATION:",
                "  Database performance benchmarks completed",
                ""
            ])
        
        return "\n".join(lines)

def main():
    """Main function to demonstrate performance validation"""
    print("AGENT BETA - Performance Validation Framework")
    print("=" * 55)
    
    # Initialize validation framework
    validator = PerformanceValidationFramework()
    
    try:
        # Run comprehensive validation
        print("\nRunning comprehensive performance validation...")
        results = validator.run_comprehensive_validation()
        
        # Generate and display report
        report = validator.generate_validation_report(results)
        print("\n" + report)
        
        # Save results to file
        output_file = validator.benchmarker.output_dir / f"validation_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nValidation results saved: {output_file}")
        print("Performance validation completed successfully!")
        
    except Exception as e:
        print(f"Performance validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()