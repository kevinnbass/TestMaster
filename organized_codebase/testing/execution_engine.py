"""
TestMaster Test Execution Engine Component
=========================================

Extracted from consolidated testing hub for better modularization.
Manages test execution, result collection, and performance tracking.

Original location: core/intelligence/testing/__init__.py (lines ~900-1100)
"""

from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import logging
import uuid
import statistics

from ..base import TestExecutionResult
from ...base import UnifiedTest


class TestExecutionEngine:
    """
    Manages test execution with performance monitoring and result collection.
    
    Features:
    - Parallel test execution
    - Real-time result streaming
    - Performance profiling
    - Resource monitoring
    - Failure recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("test_execution_engine")
        
        # Execution state
        self._test_results: Dict[str, TestExecutionResult] = {}
        self._execution_history: List[TestExecutionResult] = []
        self._active_executions: Set[str] = set()
        
        # Performance tracking
        self._performance_metrics: Dict[str, List[float]] = {}
        self._resource_usage: Dict[str, Dict[str, float]] = {}
        
        # Threading
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        self._result_callbacks: List[Callable[[TestExecutionResult], None]] = []
        self._execution_lock = threading.Lock()
        
        # Statistics
        self._execution_statistics = {
            'total_executed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        self.logger.info(f"Test execution engine initialized with {self.config.get('max_workers', 4)} workers")
    
    def execute_test(self, test: UnifiedTest, timeout: Optional[float] = None) -> TestExecutionResult:
        """
        Execute a single test with monitoring.
        
        Args:
            test: Test to execute
            timeout: Optional execution timeout
            
        Returns:
            TestExecutionResult with execution details
        """
        test_id = test.test_id
        start_time = time.time()
        
        # Mark as active
        with self._execution_lock:
            self._active_executions.add(test_id)
        
        try:
            # Create result object
            result = TestExecutionResult(
                test_id=test_id,
                test_name=test.test_name,
                status='running',
                execution_time=0.0,
                timestamp=datetime.now(),
                coverage_data={},
                performance_metrics={},
                dependency_map={},
                failure_analysis={}
            )
            
            # Execute test (simulated - in real implementation would run actual test)
            execution_result = self._run_test_implementation(test, timeout)
            
            # Update result
            result.status = execution_result['status']
            result.execution_time = time.time() - start_time
            result.coverage_data = execution_result.get('coverage', {})
            result.performance_metrics = execution_result.get('performance', {})
            
            # Collect resource usage
            result.memory_usage = execution_result.get('memory_usage', 0.0)
            result.cpu_usage = execution_result.get('cpu_usage', 0.0)
            result.io_operations = execution_result.get('io_operations', 0)
            
            # Handle failures
            if result.status == 'failed':
                result.error_message = execution_result.get('error_message')
                result.error_traceback = execution_result.get('error_traceback')
                result.error_category = self._categorize_error(result.error_message)
                result.failure_analysis = self._analyze_failure(result)
            
            # Update statistics
            self._update_statistics(result)
            
            # Store result
            self._test_results[test_id] = result
            self._execution_history.append(result)
            
            # Notify callbacks
            self._notify_callbacks(result)
            
            self.logger.info(f"Test {test_id} completed: {result.status} in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Test execution failed for {test_id}: {e}")
            return self._create_error_result(test, str(e), time.time() - start_time)
            
        finally:
            # Remove from active
            with self._execution_lock:
                self._active_executions.discard(test_id)
    
    def execute_test_suite(self, 
                          tests: List[UnifiedTest],
                          parallel: bool = True,
                          max_parallel: int = 5) -> List[TestExecutionResult]:
        """
        Execute multiple tests with optional parallelization.
        
        Args:
            tests: List of tests to execute
            parallel: Enable parallel execution
            max_parallel: Maximum parallel tests
            
        Returns:
            List of execution results
        """
        results = []
        
        if parallel:
            # Parallel execution
            futures = []
            with ThreadPoolExecutor(max_workers=min(max_parallel, len(tests))) as executor:
                for test in tests:
                    future = executor.submit(self.execute_test, test)
                    futures.append((test.test_id, future))
                
                # Collect results
                for test_id, future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed to execute test {test_id}: {e}")
                        results.append(self._create_error_result(
                            next(t for t in tests if t.test_id == test_id),
                            str(e), 0.0
                        ))
        else:
            # Sequential execution
            for test in tests:
                result = self.execute_test(test)
                results.append(result)
        
        self.logger.info(f"Test suite execution complete: {len(results)} tests executed")
        return results
    
    def register_result_callback(self, callback: Callable[[TestExecutionResult], None]):
        """Register a callback for test results."""
        self._result_callbacks.append(callback)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        stats = self._execution_statistics.copy()
        
        # Add detailed metrics
        if self._execution_history:
            recent_results = self._execution_history[-100:]  # Last 100 executions
            
            stats['recent_performance'] = {
                'average_execution_time': statistics.mean(r.execution_time for r in recent_results),
                'median_execution_time': statistics.median(r.execution_time for r in recent_results),
                'success_rate': sum(1 for r in recent_results if r.status == 'passed') / len(recent_results)
            }
            
            # Coverage trends
            coverage_values = [r.coverage_data.get('line_coverage', 0) for r in recent_results]
            if coverage_values:
                stats['coverage_trend'] = {
                    'average': statistics.mean(coverage_values),
                    'improving': coverage_values[-10:] > coverage_values[-20:-10] if len(coverage_values) >= 20 else False
                }
        
        # Active executions
        stats['active_executions'] = len(self._active_executions)
        
        return stats
    
    def get_test_result(self, test_id: str) -> Optional[TestExecutionResult]:
        """Get result for a specific test."""
        return self._test_results.get(test_id)
    
    def get_execution_history(self, limit: int = 100) -> List[TestExecutionResult]:
        """Get recent execution history."""
        return self._execution_history[-limit:]
    
    def clear_history(self):
        """Clear execution history to free memory."""
        self._execution_history.clear()
        self._test_results.clear()
        self.logger.info("Execution history cleared")
    
    # === Private Methods ===
    
    def _run_test_implementation(self, test: UnifiedTest, timeout: Optional[float]) -> Dict[str, Any]:
        """
        Run the actual test implementation.
        
        Note: This is a simulation. In real implementation, this would:
        - Execute the actual test code
        - Collect coverage data
        - Monitor resource usage
        - Handle timeouts
        """
        import random
        
        # Simulate test execution
        time.sleep(random.uniform(0.1, 0.5))  # Simulate execution time
        
        # Simulate results
        success_rate = 0.85  # 85% success rate
        is_success = random.random() < success_rate
        
        result = {
            'status': 'passed' if is_success else 'failed',
            'coverage': {
                'line_coverage': random.uniform(60, 95),
                'branch_coverage': random.uniform(50, 90),
                'function_coverage': random.uniform(70, 100)
            },
            'performance': {
                'response_time': random.uniform(0.01, 0.5),
                'throughput': random.uniform(100, 1000)
            },
            'memory_usage': random.uniform(10, 100),
            'cpu_usage': random.uniform(5, 50),
            'io_operations': random.randint(10, 1000)
        }
        
        if not is_success:
            result['error_message'] = "Simulated test failure"
            result['error_traceback'] = "Traceback (simulated)..."
        
        return result
    
    def _categorize_error(self, error_message: Optional[str]) -> Optional[str]:
        """Categorize error based on message."""
        if not error_message:
            return None
        
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'network'
        elif 'permission' in error_lower or 'access' in error_lower:
            return 'permission'
        elif 'memory' in error_lower or 'resource' in error_lower:
            return 'resource'
        elif 'assertion' in error_lower or 'assert' in error_lower:
            return 'assertion'
        elif 'import' in error_lower or 'module' in error_lower:
            return 'import'
        else:
            return 'unknown'
    
    def _analyze_failure(self, result: TestExecutionResult) -> Dict[str, Any]:
        """Analyze test failure for insights."""
        analysis = {
            'failure_category': result.error_category,
            'timestamp': result.timestamp.isoformat(),
            'execution_time': result.execution_time
        }
        
        # Check for patterns in history
        similar_failures = [
            r for r in self._execution_history[-50:]
            if r.test_id == result.test_id and r.status == 'failed'
        ]
        
        if similar_failures:
            analysis['failure_rate'] = len(similar_failures) / min(50, len(self._execution_history))
            analysis['is_flaky'] = len(similar_failures) > 2
            analysis['last_success'] = next(
                (r.timestamp for r in reversed(self._execution_history)
                 if r.test_id == result.test_id and r.status == 'passed'),
                None
            )
        
        return analysis
    
    def _create_error_result(self, test: UnifiedTest, error: str, execution_time: float) -> TestExecutionResult:
        """Create error result for failed test execution."""
        return TestExecutionResult(
            test_id=test.test_id,
            test_name=test.test_name,
            status='error',
            execution_time=execution_time,
            timestamp=datetime.now(),
            error_message=error,
            error_category='execution_error'
        )
    
    def _update_statistics(self, result: TestExecutionResult):
        """Update execution statistics."""
        self._execution_statistics['total_executed'] += 1
        
        if result.status == 'passed':
            self._execution_statistics['successful'] += 1
        elif result.status == 'failed':
            self._execution_statistics['failed'] += 1
        elif result.status == 'skipped':
            self._execution_statistics['skipped'] += 1
        
        # Update timing
        self._execution_statistics['total_execution_time'] += result.execution_time
        self._execution_statistics['average_execution_time'] = (
            self._execution_statistics['total_execution_time'] / 
            self._execution_statistics['total_executed']
        )
    
    def _notify_callbacks(self, result: TestExecutionResult):
        """Notify registered callbacks of test result."""
        for callback in self._result_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the execution engine."""
        self.logger.info("Shutting down execution engine...")
        
        # Wait for active executions
        while self._active_executions:
            self.logger.info(f"Waiting for {len(self._active_executions)} active executions...")
            time.sleep(1)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        self.logger.info("Execution engine shutdown complete")
    
    def stop_all_executions(self):
        """Stop all running test executions."""
        self.graceful_shutdown()


# Public API exports
__all__ = ['TestExecutionEngine']