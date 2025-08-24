#!/usr/bin/env python3
"""
🏗️ MODULE: Performance Benchmarker - Core Benchmarking Engine
==================================================================

📋 PURPOSE:
    Core performance benchmarking system for executing and managing performance tests.
    Handles test registration, execution, result storage, and benchmark suite management.

🎯 CORE FUNCTIONALITY:
    • Performance test registration and management
    • Test execution with concurrent user simulation
    • Result storage in SQLite database
    • Benchmark suite orchestration

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 07:30:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract benchmarking engine from performance_validation_framework.py via STEELCLAD
   └─ Changes: Created dedicated module for performance benchmarking operations
   └─ Impact: Improved modularity and single responsibility for benchmarking

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: sqlite3, psutil, logging, concurrent.futures
🎯 Integration Points: performance_models.py, load_test_executor.py
⚡ Performance Notes: Thread pool execution for concurrent testing
🔒 Security Notes: Database operations use parameterized queries

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Self-testing via benchmarks | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: performance_models for data structures
📤 Provides: Benchmarking capabilities for performance validation
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import os
import sys
import time
import json
import sqlite3
import logging
import psutil
import threading
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import statistics
import random
import uuid

# Import data models
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_models import PerformanceTest, PerformanceResult


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
        
        # Execute test
        response_times = []
        errors = []
        successful_requests = 0
        failed_requests = 0
        
        execution_start = time.perf_counter()
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
                response_times.append(test.max_response_time_ms)
        
        execution_end = time.perf_counter()
        
        # Calculate metrics
        duration_seconds = execution_end - execution_start
        total_requests = successful_requests + failed_requests
        
        if response_times:
            avg_response = statistics.mean(response_times)
            min_response = min(response_times)
            max_response = max(response_times)
            sorted_times = sorted(response_times)
            p50 = sorted_times[int(len(sorted_times) * 0.50)]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 100 else max_response
        else:
            avg_response = min_response = max_response = p50 = p95 = p99 = 0.0
        
        throughput_rps = total_requests / duration_seconds if duration_seconds > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Collect final metrics
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_usage = end_memory - start_memory
        cpu_usage = psutil.Process().cpu_percent()
        
        # Create result
        result = PerformanceResult(
            test_id=test_id,
            execution_timestamp=start_time,
            duration_seconds=duration_seconds,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response,
            min_response_time_ms=min_response,
            max_response_time_ms=max_response,
            p50_response_time_ms=p50,
            p95_response_time_ms=p95,
            p99_response_time_ms=p99,
            throughput_rps=throughput_rps,
            error_rate_percent=error_rate,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            errors=errors[:10],  # Limit stored errors
            success=(error_rate < 5.0 and avg_response < test.expected_response_time_ms)
        )
        
        # Store result
        self.test_results[test_id].append(result)
        self._save_result_to_db(result)
        
        return result
    
    def _save_result_to_db(self, result: PerformanceResult):
        """Save performance result to database"""
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
        """Execute a suite of benchmark tests"""
        if test_ids is None:
            test_ids = list(self.registered_tests.keys())
        
        session_id = str(uuid.uuid4())
        session_start = datetime.now(timezone.utc)
        
        self.logger.info(f"Starting benchmark suite {session_id} with {len(test_ids)} tests")
        
        suite_results = []
        passed_tests = 0
        failed_tests = 0
        
        for test_id in test_ids:
            try:
                result = self.execute_single_test(test_id)
                suite_results.append(asdict(result))
                
                if result.success:
                    passed_tests += 1
                else:
                    failed_tests += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to execute test {test_id}: {str(e)}")
                failed_tests += 1
        
        overall_success = failed_tests == 0
        
        # Store session data
        session_data = {
            'session_id': session_id,
            'timestamp': session_start.isoformat(),
            'total_tests': len(test_ids),
            'passed': passed_tests,
            'failed': failed_tests,
            'success': overall_success,
            'results_summary': {
                'avg_throughput': statistics.mean([r['throughput_rps'] for r in suite_results]) if suite_results else 0,
                'avg_response_time': statistics.mean([r['avg_response_time_ms'] for r in suite_results]) if suite_results else 0,
                'total_requests': sum([r['total_requests'] for r in suite_results]),
                'total_errors': sum([r['failed_requests'] for r in suite_results])
            }
        }
        
        self.benchmark_history.append(session_data)
        
        self.logger.info(f"Benchmark suite completed: {passed_tests}/{len(test_ids)} tests passed")
        
        return {
            'session_id': session_id,
            'overall_success': overall_success,
            'results': suite_results,
            'summary': session_data['results_summary']
        }