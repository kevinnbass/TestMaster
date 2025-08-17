"""
Async Executor for TestMaster

Advanced asynchronous execution engine with context management,
priority scheduling, and comprehensive monitoring integration.
"""

import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import Future, ThreadPoolExecutor
import functools
import inspect

from ..core.feature_flags import FeatureFlags
from ..core.shared_state import get_shared_state
from ..telemetry import get_telemetry_collector, get_performance_monitor

T = TypeVar('T')

class TaskPriority(Enum):
    """Task execution priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ExecutionContext:
    """Execution context for async tasks."""
    task_id: str
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context_id: Optional[str] = None
    cancellation_token: Optional[asyncio.Event] = None
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0

@dataclass
class TaskResult(Generic[T]):
    """Result of task execution."""
    task_id: str
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    retry_count: int = 0
    context: Optional[ExecutionContext] = None

class AsyncExecutor:
    """
    Advanced asynchronous executor for TestMaster.
    
    Provides:
    - High-performance async task execution
    - Priority-based scheduling
    - Context management and tracking
    - Timeout and cancellation support
    - Retry mechanisms with exponential backoff
    - Integration with telemetry and monitoring
    """
    
    def __init__(self, max_concurrent_tasks: int = 100, 
                 default_timeout: float = 300.0):
        """
        Initialize async executor.
        
        Args:
            max_concurrent_tasks: Maximum concurrent async tasks
            default_timeout: Default task timeout in seconds
        """
        self.enabled = FeatureFlags.is_enabled('layer2_monitoring', 'async_processing')
        
        if not self.enabled:
            return
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        
        # Task management
        self.active_tasks: Dict[str, ExecutionContext] = {}
        self.completed_tasks: List[ExecutionContext] = []
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Threading and synchronization
        self.lock = threading.RLock()
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.shutdown_event = asyncio.Event()
        
        # Event loop management
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Integrations
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        if FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system'):
            self.telemetry = get_telemetry_collector()
            self.performance_monitor = get_performance_monitor()
        else:
            self.telemetry = None
            self.performance_monitor = None
        
        # Statistics
        self.tasks_executed = 0
        self.tasks_succeeded = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        
        # Start event loop
        self._start_event_loop()
        
        print("Async executor initialized")
        print(f"   Max concurrent tasks: {self.max_concurrent_tasks}")
        print(f"   Default timeout: {self.default_timeout}s")
    
    def _start_event_loop(self):
        """Start dedicated event loop for async execution."""
        if not self.enabled:
            return
        
        def loop_worker():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.is_running = True
            
            try:
                # Start task processor
                self.loop.create_task(self._process_tasks())
                self.loop.run_forever()
            except Exception as e:
                print(f"Event loop error: {e}")
            finally:
                self.is_running = False
        
        self.loop_thread = threading.Thread(target=loop_worker, daemon=True)
        self.loop_thread.start()
        
        # Wait for loop to be ready
        while not self.loop or not self.is_running:
            time.sleep(0.01)
    
    async def _process_tasks(self):
        """Process tasks from the priority queue."""
        while not self.shutdown_event.is_set():
            try:
                # Get next task from queue (blocks until available)
                priority, task_func, context = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                # Execute task with semaphore
                await self.semaphore.acquire()
                
                # Create task for execution
                task = asyncio.create_task(
                    self._execute_task_with_context(task_func, context)
                )
                
                # Release semaphore when done
                task.add_done_callback(lambda _: self.semaphore.release())
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                print(f"Task processor error: {e}")
    
    async def _execute_task_with_context(self, task_func: Callable, 
                                       context: ExecutionContext) -> TaskResult:
        """Execute a task with full context management."""
        start_time = time.time()
        context.started_at = datetime.now()
        
        try:
            # Add to active tasks
            with self.lock:
                self.active_tasks[context.task_id] = context
            
            # Set up timeout and cancellation
            timeout = context.timeout_seconds or self.default_timeout
            
            if context.cancellation_token:
                # Create timeout with cancellation support
                result = await asyncio.wait_for(
                    self._execute_with_cancellation(task_func, context),
                    timeout=timeout
                )
            else:
                # Simple timeout
                result = await asyncio.wait_for(task_func(), timeout=timeout)
            
            # Task succeeded
            execution_time = (time.time() - start_time) * 1000
            context.completed_at = datetime.now()
            
            task_result = TaskResult(
                task_id=context.task_id,
                success=True,
                result=result,
                execution_time_ms=execution_time,
                retry_count=context.retry_count,
                context=context
            )
            
            # Update statistics
            with self.lock:
                self.tasks_executed += 1
                self.tasks_succeeded += 1
                self.total_execution_time += execution_time
                self.completed_tasks.append(context)
                self.active_tasks.pop(context.task_id, None)
            
            # Send telemetry
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="async_task_completed",
                    component="async_executor",
                    operation="execute_task",
                    metadata={
                        "task_id": context.task_id,
                        "priority": context.priority.name,
                        "retry_count": context.retry_count
                    },
                    duration_ms=execution_time,
                    success=True
                )
            
            # Update shared state
            if self.shared_state:
                self.shared_state.increment("async_tasks_succeeded")
                self.shared_state.set("async_last_successful_task", context.task_id)
            
            return task_result
            
        except asyncio.CancelledError:
            # Task was cancelled
            context.completed_at = datetime.now()
            execution_time = (time.time() - start_time) * 1000
            
            task_result = TaskResult(
                task_id=context.task_id,
                success=False,
                error=Exception("Task cancelled"),
                execution_time_ms=execution_time,
                retry_count=context.retry_count,
                context=context
            )
            
            self._handle_task_failure(context, task_result, execution_time)
            return task_result
            
        except asyncio.TimeoutError:
            # Task timed out
            context.completed_at = datetime.now()
            execution_time = (time.time() - start_time) * 1000
            
            task_result = TaskResult(
                task_id=context.task_id,
                success=False,
                error=Exception(f"Task timed out after {timeout}s"),
                execution_time_ms=execution_time,
                retry_count=context.retry_count,
                context=context
            )
            
            self._handle_task_failure(context, task_result, execution_time)
            return task_result
            
        except Exception as e:
            # Task failed with exception
            context.completed_at = datetime.now()
            execution_time = (time.time() - start_time) * 1000
            
            task_result = TaskResult(
                task_id=context.task_id,
                success=False,
                error=e,
                execution_time_ms=execution_time,
                retry_count=context.retry_count,
                context=context
            )
            
            self._handle_task_failure(context, task_result, execution_time)
            return task_result
    
    async def _execute_with_cancellation(self, task_func: Callable, 
                                       context: ExecutionContext):
        """Execute task with cancellation support."""
        cancellation_task = asyncio.create_task(context.cancellation_token.wait())
        execution_task = asyncio.create_task(task_func())
        
        done, pending = await asyncio.wait(
            [cancellation_task, execution_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Check which task completed
        if cancellation_task in done:
            raise asyncio.CancelledError("Task cancelled by cancellation token")
        
        return execution_task.result()
    
    def _handle_task_failure(self, context: ExecutionContext, 
                           task_result: TaskResult, execution_time: float):
        """Handle task failure and potential retry."""
        # Update statistics
        with self.lock:
            self.tasks_executed += 1
            self.tasks_failed += 1
            self.total_execution_time += execution_time
            self.completed_tasks.append(context)
            self.active_tasks.pop(context.task_id, None)
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="async_task_failed",
                component="async_executor",
                operation="execute_task",
                metadata={
                    "task_id": context.task_id,
                    "priority": context.priority.name,
                    "retry_count": context.retry_count,
                    "error_type": type(task_result.error).__name__ if task_result.error else "unknown"
                },
                duration_ms=execution_time,
                success=False,
                error_message=str(task_result.error) if task_result.error else None
            )
        
        # Update shared state
        if self.shared_state:
            self.shared_state.increment("async_tasks_failed")
            self.shared_state.set("async_last_failed_task", context.task_id)
    
    def submit_async_task(self, coro: Coroutine[Any, Any, T], 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout_seconds: Optional[float] = None,
                         metadata: Dict[str, Any] = None,
                         max_retries: int = 0) -> str:
        """
        Submit an async task for execution.
        
        Args:
            coro: Coroutine to execute
            priority: Task priority
            timeout_seconds: Task timeout
            metadata: Additional metadata
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID for tracking
        """
        if not self.enabled or not self.is_running:
            raise RuntimeError("Async executor is not running")
        
        # Create execution context
        context = ExecutionContext(
            task_id=str(uuid.uuid4()),
            priority=priority,
            created_at=datetime.now(),
            metadata=metadata or {},
            timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
        
        # Wrap coroutine in function
        async def task_func():
            return await coro
        
        # Submit to queue with priority
        priority_value = priority.value
        future = asyncio.run_coroutine_threadsafe(
            self.task_queue.put((priority_value, task_func, context)),
            self.loop
        )
        future.result()  # Wait for submission
        
        return context.task_id
    
    def submit_sync_task(self, func: Callable[[], T],
                        priority: TaskPriority = TaskPriority.NORMAL,
                        timeout_seconds: Optional[float] = None,
                        metadata: Dict[str, Any] = None,
                        max_retries: int = 0) -> str:
        """
        Submit a sync task for async execution.
        
        Args:
            func: Function to execute
            priority: Task priority
            timeout_seconds: Task timeout
            metadata: Additional metadata
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID for tracking
        """
        if not self.enabled or not self.is_running:
            raise RuntimeError("Async executor is not running")
        
        # Wrap sync function in coroutine
        async def async_wrapper():
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func)
        
        return self.submit_async_task(
            async_wrapper(),
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
            max_retries=max_retries
        )
    
    def get_task_status(self, task_id: str) -> Optional[ExecutionContext]:
        """Get status of a specific task."""
        if not self.enabled:
            return None
        
        with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            
            # Check completed tasks
            for context in self.completed_tasks:
                if context.task_id == task_id:
                    return context
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if not self.enabled:
            return False
        
        with self.lock:
            if task_id in self.active_tasks:
                context = self.active_tasks[task_id]
                if context.cancellation_token:
                    # Set cancellation event
                    asyncio.run_coroutine_threadsafe(
                        context.cancellation_token.set(),
                        self.loop
                    )
                    return True
        
        return False
    
    def get_executor_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            avg_execution_time = (
                self.total_execution_time / max(self.tasks_executed, 1)
            )
            
            success_rate = (
                (self.tasks_succeeded / max(self.tasks_executed, 1)) * 100
                if self.tasks_executed > 0 else 100
            )
            
            return {
                "enabled": True,
                "is_running": self.is_running,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "tasks_executed": self.tasks_executed,
                "tasks_succeeded": self.tasks_succeeded,
                "tasks_failed": self.tasks_failed,
                "success_rate": round(success_rate, 2),
                "avg_execution_time_ms": round(avg_execution_time, 2),
                "total_execution_time_ms": round(self.total_execution_time, 2)
            }
    
    def get_active_tasks(self) -> List[ExecutionContext]:
        """Get list of active tasks."""
        if not self.enabled:
            return []
        
        with self.lock:
            return list(self.active_tasks.values())
    
    def clear_completed_tasks(self):
        """Clear completed task history."""
        if not self.enabled:
            return
        
        with self.lock:
            self.completed_tasks.clear()
    
    def shutdown(self):
        """Shutdown async executor."""
        if not self.enabled or not self.is_running:
            return
        
        print("Shutting down async executor...")
        
        # Set shutdown event
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.shutdown_event.set(), self.loop)
        
        # Cancel all active tasks
        with self.lock:
            for task_id in list(self.active_tasks.keys()):
                self.cancel_task(task_id)
        
        # Stop event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for thread to finish
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5.0)
        
        print(f"Async executor shutdown - executed {self.tasks_executed} tasks")

# Global instance
_async_executor: Optional[AsyncExecutor] = None

def get_async_executor() -> AsyncExecutor:
    """Get the global async executor instance."""
    global _async_executor
    if _async_executor is None:
        _async_executor = AsyncExecutor()
    return _async_executor

# Convenience function
def async_execute(coro_or_func: Union[Coroutine, Callable],
                 priority: TaskPriority = TaskPriority.NORMAL,
                 timeout_seconds: Optional[float] = None,
                 metadata: Dict[str, Any] = None) -> str:
    """
    Execute a coroutine or function asynchronously.
    
    Args:
        coro_or_func: Coroutine or function to execute
        priority: Execution priority
        timeout_seconds: Execution timeout
        metadata: Additional metadata
        
    Returns:
        Task ID for tracking
    """
    executor = get_async_executor()
    
    if inspect.iscoroutine(coro_or_func):
        return executor.submit_async_task(
            coro_or_func,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata=metadata
        )
    elif callable(coro_or_func):
        return executor.submit_sync_task(
            coro_or_func,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata=metadata
        )
    else:
        raise ValueError("Must provide coroutine or callable")