"""
Test Scheduler with Queue Management

Inspired by Agency-Swarm's Gradio queue management patterns
for prioritized test execution and resource management.

Features:
- Priority-based test scheduling
- Resource-aware execution
- Queue management with backpressure
- Configurable execution intervals
"""

import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from concurrent.futures import ThreadPoolExecutor, Future
import subprocess
import json

from core.layer_manager import requires_layer


class TestPriority(IntEnum):
    """Test execution priorities (higher number = higher priority)."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class TestStatus(Enum):
    """Test execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTest:
    """Test scheduled for execution."""
    test_id: str
    test_path: str
    test_command: List[str]
    priority: TestPriority
    queued_at: datetime
    estimated_duration: Optional[float] = None  # seconds
    timeout: Optional[float] = None  # seconds
    retry_count: int = 0
    max_retries: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TestStatus = TestStatus.QUEUED
    result_code: Optional[int] = None
    output: Optional[str] = None
    error_output: Optional[str] = None
    actual_duration: Optional[float] = None


@dataclass
class SchedulerStatistics:
    """Test scheduler statistics."""
    total_tests_scheduled: int
    tests_completed: int
    tests_failed: int
    tests_in_queue: int
    tests_running: int
    avg_queue_time_seconds: float
    avg_execution_time_seconds: float
    success_rate: float
    queue_utilization: float
    worker_utilization: float
    last_updated: datetime = field(default_factory=datetime.now)


class TestScheduler:
    """
    Priority-based test scheduler with queue management.
    
    Uses Agency-Swarm's Gradio queue patterns for managing
    test execution with priorities and resource limits.
    """
    
    @requires_layer("layer2_monitoring", "test_scheduling")
    def __init__(self, max_workers: int = 2, 
                 max_queue_size: int = 100,
                 default_timeout: float = 300.0):
        """
        Initialize test scheduler.
        
        Args:
            max_workers: Maximum concurrent test executions
            max_queue_size: Maximum tests that can be queued
            default_timeout: Default test timeout in seconds
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        
        # Queue management (Agency-Swarm Gradio pattern)
        self._test_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._running_tests: Dict[str, ScheduledTest] = {}
        self._completed_tests: Dict[str, ScheduledTest] = {}
        
        # Thread pool for test execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Future] = {}
        
        # Scheduler control
        self._is_running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Statistics tracking
        self._stats = {
            'total_scheduled': 0,
            'total_completed': 0,
            'total_failed': 0,
            'queue_times': [],
            'execution_times': [],
            'start_time': datetime.now()
        }
        
        # Callbacks
        self.on_test_started: Optional[Callable[[ScheduledTest], None]] = None
        self.on_test_completed: Optional[Callable[[ScheduledTest], None]] = None
        self.on_test_failed: Optional[Callable[[ScheduledTest], None]] = None
        self.on_queue_full: Optional[Callable[[int], None]] = None
        
        print(f" Test scheduler initialized")
        print(f"    Max workers: {max_workers}")
        print(f"    Max queue size: {max_queue_size}")
        print(f"    Default timeout: {default_timeout}s")
    
    def start(self):
        """Start the test scheduler."""
        if self._is_running:
            print("️ Test scheduler is already running")
            return
        
        print(" Starting test scheduler...")
        
        self._is_running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        print(" Test scheduler started")
    
    def stop(self):
        """Stop the test scheduler."""
        if not self._is_running:
            return
        
        print(" Stopping test scheduler...")
        
        self._is_running = False
        
        # Cancel running tests
        for test_id, future in self._futures.items():
            if not future.done():
                future.cancel()
                if test_id in self._running_tests:
                    self._running_tests[test_id].status = TestStatus.CANCELLED
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Wait for scheduler thread
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        
        print(" Test scheduler stopped")
    
    def schedule_test(self, test_path: str, 
                     command: Optional[List[str]] = None,
                     priority: TestPriority = TestPriority.NORMAL,
                     timeout: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Schedule a test for execution.
        
        Args:
            test_path: Path to the test file
            command: Command to run the test (auto-generated if None)
            priority: Test execution priority
            timeout: Test timeout in seconds
            metadata: Additional test metadata
            
        Returns:
            Test ID for tracking
        """
        if not self._is_running:
            raise RuntimeError("Test scheduler is not running")
        
        # Generate test ID
        test_id = f"test_{int(time.time() * 1000)}_{hash(test_path) % 10000}"
        
        # Auto-generate command if not provided
        if command is None:
            command = self._generate_test_command(test_path)
        
        # Create scheduled test
        scheduled_test = ScheduledTest(
            test_id=test_id,
            test_path=test_path,
            test_command=command,
            priority=priority,
            queued_at=datetime.now(),
            timeout=timeout or self.default_timeout,
            metadata=metadata or {}
        )
        
        try:
            # Add to priority queue (negative priority for max-heap behavior)
            self._test_queue.put((-priority.value, time.time(), scheduled_test), block=False)
            self._stats['total_scheduled'] += 1
            
            print(f" Scheduled test: {Path(test_path).name} (priority: {priority.name})")
            return test_id
            
        except queue.Full:
            if self.on_queue_full:
                try:
                    self.on_queue_full(self.max_queue_size)
                except Exception as e:
                    print(f"️ Error in queue full callback: {e}")
            
            raise RuntimeError(f"Test queue is full (max: {self.max_queue_size})")
    
    def schedule_multiple_tests(self, test_paths: List[str],
                              priority: TestPriority = TestPriority.NORMAL) -> List[str]:
        """
        Schedule multiple tests for execution.
        
        Args:
            test_paths: List of test file paths
            priority: Priority for all tests
            
        Returns:
            List of test IDs
        """
        test_ids = []
        
        for test_path in test_paths:
            try:
                test_id = self.schedule_test(test_path, priority=priority)
                test_ids.append(test_id)
            except Exception as e:
                print(f"️ Failed to schedule {test_path}: {e}")
        
        return test_ids
    
    def cancel_test(self, test_id: str) -> bool:
        """
        Cancel a scheduled or running test.
        
        Args:
            test_id: ID of the test to cancel
            
        Returns:
            True if test was cancelled, False if not found or already completed
        """
        # Check if test is running
        if test_id in self._running_tests:
            if test_id in self._futures:
                future = self._futures[test_id]
                if future.cancel():
                    self._running_tests[test_id].status = TestStatus.CANCELLED
                    self._move_to_completed(test_id)
                    print(f" Cancelled running test: {test_id}")
                    return True
        
        # For queued tests, we can't easily remove from PriorityQueue
        # Mark as cancelled when it's dequeued
        print(f"️ Cannot cancel test {test_id} (not found or already completed)")
        return False
    
    def get_test_status(self, test_id: str) -> Optional[ScheduledTest]:
        """Get status of a specific test."""
        # Check running tests
        if test_id in self._running_tests:
            return self._running_tests[test_id]
        
        # Check completed tests
        if test_id in self._completed_tests:
            return self._completed_tests[test_id]
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_size": self._test_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "running_tests": len(self._running_tests),
            "max_workers": self.max_workers,
            "completed_tests": len(self._completed_tests),
            "queue_utilization": (self._test_queue.qsize() / self.max_queue_size) * 100,
            "worker_utilization": (len(self._running_tests) / self.max_workers) * 100
        }
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._is_running:
            try:
                # Get next test from queue
                try:
                    _, _, scheduled_test = self._test_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if scheduler is still running
                if not self._is_running:
                    break
                
                # Check if test was cancelled while in queue
                if scheduled_test.status == TestStatus.CANCELLED:
                    self._test_queue.task_done()
                    continue
                
                # Submit test for execution
                self._submit_test(scheduled_test)
                self._test_queue.task_done()
                
            except Exception as e:
                print(f"️ Error in scheduler loop: {e}")
                time.sleep(1)
    
    def _submit_test(self, scheduled_test: ScheduledTest):
        """Submit test to thread pool for execution."""
        test_id = scheduled_test.test_id
        
        # Update test status
        scheduled_test.status = TestStatus.RUNNING
        scheduled_test.started_at = datetime.now()
        
        # Track queue time
        queue_time = (scheduled_test.started_at - scheduled_test.queued_at).total_seconds()
        self._stats['queue_times'].append(queue_time)
        
        # Add to running tests
        self._running_tests[test_id] = scheduled_test
        
        # Submit to executor
        future = self._executor.submit(self._execute_test, scheduled_test)
        self._futures[test_id] = future
        
        # Add completion callback
        future.add_done_callback(lambda f: self._on_test_execution_complete(test_id, f))
        
        # Call started callback
        if self.on_test_started:
            try:
                self.on_test_started(scheduled_test)
            except Exception as e:
                print(f"️ Error in test started callback: {e}")
        
        print(f" Started test: {Path(scheduled_test.test_path).name}")
    
    def _execute_test(self, scheduled_test: ScheduledTest) -> ScheduledTest:
        """Execute a single test."""
        start_time = time.time()
        
        try:
            # Run test command
            result = subprocess.run(
                scheduled_test.test_command,
                capture_output=True,
                text=True,
                timeout=scheduled_test.timeout,
                cwd=Path.cwd()
            )
            
            # Update test with results
            scheduled_test.result_code = result.returncode
            scheduled_test.output = result.stdout
            scheduled_test.error_output = result.stderr
            scheduled_test.actual_duration = time.time() - start_time
            scheduled_test.completed_at = datetime.now()
            
            # Set status based on result
            if result.returncode == 0:
                scheduled_test.status = TestStatus.COMPLETED
            else:
                scheduled_test.status = TestStatus.FAILED
                
        except subprocess.TimeoutExpired:
            scheduled_test.status = TestStatus.TIMEOUT
            scheduled_test.actual_duration = time.time() - start_time
            scheduled_test.completed_at = datetime.now()
            scheduled_test.error_output = f"Test timed out after {scheduled_test.timeout}s"
            
        except Exception as e:
            scheduled_test.status = TestStatus.FAILED
            scheduled_test.actual_duration = time.time() - start_time
            scheduled_test.completed_at = datetime.now()
            scheduled_test.error_output = f"Test execution failed: {str(e)}"
        
        return scheduled_test
    
    def _on_test_execution_complete(self, test_id: str, future: Future):
        """Handle test execution completion."""
        try:
            scheduled_test = future.result()
            
            # Update statistics
            if scheduled_test.actual_duration:
                self._stats['execution_times'].append(scheduled_test.actual_duration)
            
            if scheduled_test.status == TestStatus.COMPLETED:
                self._stats['total_completed'] += 1
            elif scheduled_test.status in [TestStatus.FAILED, TestStatus.TIMEOUT]:
                self._stats['total_failed'] += 1
                
                # Check for retry
                if scheduled_test.retry_count < scheduled_test.max_retries:
                    scheduled_test.retry_count += 1
                    scheduled_test.status = TestStatus.QUEUED
                    scheduled_test.queued_at = datetime.now()
                    
                    # Re-queue for retry
                    try:
                        self._test_queue.put(
                            (-scheduled_test.priority.value, time.time(), scheduled_test),
                            block=False
                        )
                        print(f" Retrying test: {Path(scheduled_test.test_path).name} (attempt {scheduled_test.retry_count + 1})")
                        return
                    except queue.Full:
                        print(f"️ Cannot retry test - queue is full")
            
            # Move to completed
            self._move_to_completed(test_id)
            
            # Call appropriate callback
            if scheduled_test.status == TestStatus.COMPLETED:
                if self.on_test_completed:
                    try:
                        self.on_test_completed(scheduled_test)
                    except Exception as e:
                        print(f"️ Error in test completed callback: {e}")
                        
                print(f" Test completed: {Path(scheduled_test.test_path).name} ({scheduled_test.actual_duration:.1f}s)")
            else:
                if self.on_test_failed:
                    try:
                        self.on_test_failed(scheduled_test)
                    except Exception as e:
                        print(f"️ Error in test failed callback: {e}")
                        
                print(f" Test failed: {Path(scheduled_test.test_path).name} ({scheduled_test.status.value})")
                
        except Exception as e:
            print(f"️ Error handling test completion: {e}")
    
    def _move_to_completed(self, test_id: str):
        """Move test from running to completed."""
        if test_id in self._running_tests:
            test = self._running_tests.pop(test_id)
            self._completed_tests[test_id] = test
        
        if test_id in self._futures:
            del self._futures[test_id]
    
    def _generate_test_command(self, test_path: str) -> List[str]:
        """Generate appropriate test command for a test file."""
        test_path = Path(test_path)
        
        # Check if it's a pytest test
        if test_path.name.startswith('test_') or test_path.name.endswith('_test.py'):
            return ["python", "-m", "pytest", str(test_path), "-v"]
        
        # Default to python execution
        return ["python", str(test_path)]
    
    def get_statistics(self) -> SchedulerStatistics:
        """Get scheduler statistics."""
        # Calculate averages
        avg_queue_time = 0.0
        if self._stats['queue_times']:
            avg_queue_time = sum(self._stats['queue_times']) / len(self._stats['queue_times'])
        
        avg_execution_time = 0.0
        if self._stats['execution_times']:
            avg_execution_time = sum(self._stats['execution_times']) / len(self._stats['execution_times'])
        
        # Calculate success rate
        total_finished = self._stats['total_completed'] + self._stats['total_failed']
        success_rate = (self._stats['total_completed'] / max(total_finished, 1)) * 100
        
        # Calculate utilization
        queue_utilization = (self._test_queue.qsize() / self.max_queue_size) * 100
        worker_utilization = (len(self._running_tests) / self.max_workers) * 100
        
        return SchedulerStatistics(
            total_tests_scheduled=self._stats['total_scheduled'],
            tests_completed=self._stats['total_completed'],
            tests_failed=self._stats['total_failed'],
            tests_in_queue=self._test_queue.qsize(),
            tests_running=len(self._running_tests),
            avg_queue_time_seconds=avg_queue_time,
            avg_execution_time_seconds=avg_execution_time,
            success_rate=success_rate,
            queue_utilization=queue_utilization,
            worker_utilization=worker_utilization
        )
    
    def export_results(self, output_path: str = "test_results.json"):
        """Export test results to JSON."""
        results = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_statistics().__dict__,
            "completed_tests": [
                {
                    "test_id": test.test_id,
                    "test_path": test.test_path,
                    "status": test.status.value,
                    "priority": test.priority.name,
                    "queued_at": test.queued_at.isoformat(),
                    "started_at": test.started_at.isoformat() if test.started_at else None,
                    "completed_at": test.completed_at.isoformat() if test.completed_at else None,
                    "actual_duration": test.actual_duration,
                    "result_code": test.result_code,
                    "retry_count": test.retry_count
                }
                for test in self._completed_tests.values()
            ]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f" Test results exported to {output_path}")
        except Exception as e:
            print(f"️ Error exporting results: {e}")
    
    def clear_completed_tests(self, older_than_hours: int = 24):
        """Clear old completed test results."""
        cutoff = datetime.now() - timedelta(hours=older_than_hours)
        
        to_remove = [
            test_id for test_id, test in self._completed_tests.items()
            if test.completed_at and test.completed_at < cutoff
        ]
        
        for test_id in to_remove:
            del self._completed_tests[test_id]
        
        print(f" Cleared {len(to_remove)} old test results")


# Convenience function for quick test execution
def quick_test_run(test_paths: Union[str, List[str]], 
                  max_workers: int = 2) -> List[ScheduledTest]:
    """
    Quick test execution without persistent scheduler.
    
    Args:
        test_paths: Test file path(s) to execute
        max_workers: Maximum concurrent executions
        
    Returns:
        List of completed test results
    """
    scheduler = TestScheduler(max_workers=max_workers)
    scheduler.start()
    
    try:
        # Schedule tests
        if isinstance(test_paths, str):
            test_paths = [test_paths]
        
        test_ids = scheduler.schedule_multiple_tests(test_paths)
        
        # Wait for completion
        while any(scheduler.get_test_status(tid).status in [TestStatus.QUEUED, TestStatus.RUNNING] 
                 for tid in test_ids):
            time.sleep(0.1)
        
        # Get results
        return [scheduler.get_test_status(tid) for tid in test_ids]
        
    finally:
        scheduler.stop()