"""
Concurrent Scheduler for TestMaster

Advanced task scheduling system with cron-like functionality,
async execution, and intelligent resource management.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
try:
    import crontab
except ImportError:
    crontab = None

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from ..telemetry import get_telemetry_collector, get_performance_monitor

class ScheduleType(Enum):
    """Schedule type for tasks."""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    DELAYED = "delayed"

@dataclass
class ScheduleConfig:
    """Configuration for scheduled task."""
    schedule_type: ScheduleType
    interval_seconds: Optional[float] = None
    cron_expression: Optional[str] = None
    delay_seconds: Optional[float] = None
    max_executions: Optional[int] = None
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    retry_delay: float = 5.0

@dataclass
class ScheduledTask:
    """Information about a scheduled task."""
    task_id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    config: ScheduleConfig = field(default_factory=lambda: ScheduleConfig(ScheduleType.ONCE))
    created_at: datetime = field(default_factory=datetime.now)
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConcurrentScheduler:
    """
    Advanced concurrent scheduler for TestMaster.
    
    Features:
    - Multiple scheduling types (interval, cron, delayed, once)
    - Async task execution with proper resource management
    - Task dependencies and priorities
    - Comprehensive monitoring and telemetry
    - Dynamic task management (add/remove/pause/resume)
    """
    
    def __init__(self, max_concurrent_tasks: int = 50):
        """
        Initialize concurrent scheduler.
        
        Args:
            max_concurrent_tasks: Maximum concurrent scheduled tasks
        """
        self.enabled = FeatureFlags.is_enabled('layer2_monitoring', 'async_processing')
        
        if not self.enabled:
            return
        
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task management
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[ScheduledTask] = []
        
        # Scheduling
        self.lock = threading.RLock()
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.scheduler_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.is_running = False
        
        # Event loop management
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        
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
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        print("Concurrent scheduler initialized")
        print(f"   Max concurrent tasks: {self.max_concurrent_tasks}")
    
    def start_scheduler(self):
        """Start the scheduler."""
        if not self.enabled or self.is_running:
            return
        
        def loop_worker():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.is_running = True
            
            try:
                # Start scheduler task
                self.scheduler_task = self.loop.create_task(self._scheduler_loop())
                self.loop.run_until_complete(self.scheduler_task)
            except Exception as e:
                print(f"Scheduler loop error: {e}")
            finally:
                self.is_running = False
        
        self.loop_thread = threading.Thread(target=loop_worker, daemon=True)
        self.loop_thread.start()
        
        # Wait for loop to be ready
        while not self.loop or not self.is_running:
            time.sleep(0.01)
        
        print("Concurrent scheduler started")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check for tasks to execute
                tasks_to_run = []
                
                with self.lock:
                    for task in self.scheduled_tasks.values():
                        if self._should_run_task(task, current_time):
                            tasks_to_run.append(task)
                
                # Execute eligible tasks
                for task in tasks_to_run:
                    if len(self.running_tasks) < self.max_concurrent_tasks:
                        await self._execute_scheduled_task(task)
                
                # Cleanup completed tasks
                self._cleanup_completed_tasks()
                
                # Send monitoring telemetry
                self._send_scheduler_telemetry()
                
                # Wait before next check
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                print(f"Scheduler loop error: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    def _should_run_task(self, task: ScheduledTask, current_time: datetime) -> bool:
        """Check if a task should run."""
        if not task.is_active:
            return False
        
        if task.task_id in self.running_tasks:
            return False  # Already running
        
        if task.config.max_executions and task.execution_count >= task.config.max_executions:
            return False  # Max executions reached
        
        if not task.next_run:
            # Calculate next run time
            task.next_run = self._calculate_next_run(task, current_time)
            return False
        
        return current_time >= task.next_run
    
    def _calculate_next_run(self, task: ScheduledTask, base_time: datetime) -> datetime:
        """Calculate next run time for a task."""
        config = task.config
        
        if config.schedule_type == ScheduleType.ONCE:
            if config.delay_seconds:
                return base_time + timedelta(seconds=config.delay_seconds)
            else:
                return base_time
        
        elif config.schedule_type == ScheduleType.DELAYED:
            return base_time + timedelta(seconds=config.delay_seconds or 0)
        
        elif config.schedule_type == ScheduleType.INTERVAL:
            if task.last_run:
                return task.last_run + timedelta(seconds=config.interval_seconds or 60)
            else:
                return base_time + timedelta(seconds=config.interval_seconds or 60)
        
        elif config.schedule_type == ScheduleType.CRON:
            if config.cron_expression and crontab:
                try:
                    cron = crontab.CronTab(config.cron_expression)
                    next_time = cron.next(default_utc=False)
                    return base_time + timedelta(seconds=next_time)
                except Exception as e:
                    print(f"Invalid cron expression '{config.cron_expression}': {e}")
                    return base_time + timedelta(hours=1)  # Fallback
            else:
                print("Cron scheduling requires 'crontab' package - falling back to hourly")
                return base_time + timedelta(hours=1)
        
        return base_time
    
    async def _execute_scheduled_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        await self.semaphore.acquire()
        
        async def task_executor():
            start_time = time.time()
            task.last_run = datetime.now()
            task.execution_count += 1
            
            try:
                # Track task start
                if self.telemetry:
                    self.telemetry.record_event(
                        event_type="scheduled_task_started",
                        component="concurrent_scheduler",
                        operation="execute_task",
                        metadata={
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "schedule_type": task.config.schedule_type.value,
                            "execution_count": task.execution_count
                        }
                    )
                
                # Execute function
                if self.performance_monitor:
                    with self.performance_monitor.track_operation(
                        "scheduled_task", f"{task.name}_{task.task_id}"
                    ):
                        if asyncio.iscoroutinefunction(task.function):
                            result = await task.function(*task.args, **task.kwargs)
                        else:
                            # Run sync function in executor
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None, task.function, *task.args, **task.kwargs
                            )
                else:
                    if asyncio.iscoroutinefunction(task.function):
                        result = await task.function(*task.args, **task.kwargs)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, task.function, *task.args, **task.kwargs
                        )
                
                # Task succeeded
                execution_time = (time.time() - start_time) * 1000
                task.success_count += 1
                
                # Update statistics
                with self.lock:
                    self.total_executions += 1
                    self.successful_executions += 1
                
                # Calculate next run
                if task.config.schedule_type != ScheduleType.ONCE:
                    task.next_run = self._calculate_next_run(task, datetime.now())
                else:
                    task.is_active = False  # One-time task completed
                
                # Send telemetry
                if self.telemetry:
                    self.telemetry.record_event(
                        event_type="scheduled_task_completed",
                        component="concurrent_scheduler",
                        operation="execute_task",
                        metadata={
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "execution_count": task.execution_count,
                            "success_count": task.success_count
                        },
                        duration_ms=execution_time,
                        success=True
                    )
                
                # Update shared state
                if self.shared_state:
                    self.shared_state.increment("scheduled_tasks_succeeded")
                    self.shared_state.set("last_successful_scheduled_task", task.task_id)
                
                print(f"Scheduled task '{task.name}' completed successfully")
                
            except Exception as e:
                # Task failed
                execution_time = (time.time() - start_time) * 1000
                task.failure_count += 1
                
                # Update statistics
                with self.lock:
                    self.total_executions += 1
                    self.failed_executions += 1
                
                # Handle retry
                if task.config.retry_count > 0:
                    task.config.retry_count -= 1
                    task.next_run = datetime.now() + timedelta(seconds=task.config.retry_delay)
                    print(f"Scheduled task '{task.name}' failed, retrying in {task.config.retry_delay}s")
                else:
                    print(f"Scheduled task '{task.name}' failed: {e}")
                    
                    # For non-repeating tasks, deactivate on failure
                    if task.config.schedule_type == ScheduleType.ONCE:
                        task.is_active = False
                
                # Send telemetry
                if self.telemetry:
                    self.telemetry.record_event(
                        event_type="scheduled_task_failed",
                        component="concurrent_scheduler",
                        operation="execute_task",
                        metadata={
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "execution_count": task.execution_count,
                            "failure_count": task.failure_count,
                            "error_type": type(e).__name__
                        },
                        duration_ms=execution_time,
                        success=False,
                        error_message=str(e)
                    )
                
                # Update shared state
                if self.shared_state:
                    self.shared_state.increment("scheduled_tasks_failed")
            
            finally:
                # Remove from running tasks
                with self.lock:
                    self.running_tasks.pop(task.task_id, None)
                
                # Release semaphore
                self.semaphore.release()
        
        # Start task execution
        execution_task = asyncio.create_task(task_executor())
        
        with self.lock:
            self.running_tasks[task.task_id] = execution_task
    
    def schedule_task(self, name: str, function: Callable, 
                     schedule_config: ScheduleConfig,
                     args: tuple = None, kwargs: Dict[str, Any] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """
        Schedule a task for execution.
        
        Args:
            name: Task name
            function: Function to execute
            schedule_config: Scheduling configuration
            args: Function arguments
            kwargs: Function keyword arguments
            metadata: Additional metadata
            
        Returns:
            Task ID for tracking
        """
        if not self.enabled:
            raise RuntimeError("Concurrent scheduler is disabled")
        
        task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            function=function,
            args=args or (),
            kwargs=kwargs or {},
            config=schedule_config,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.scheduled_tasks[task_id] = task
        
        print(f"Scheduled task '{name}' ({schedule_config.schedule_type.value})")
        
        return task_id
    
    def get_task_info(self, task_id: str) -> Optional[ScheduledTask]:
        """Get information about a scheduled task."""
        if not self.enabled:
            return None
        
        with self.lock:
            return self.scheduled_tasks.get(task_id)
    
    def get_scheduled_tasks(self, active_only: bool = True) -> List[ScheduledTask]:
        """Get list of scheduled tasks."""
        if not self.enabled:
            return []
        
        with self.lock:
            tasks = list(self.scheduled_tasks.values())
            
            if active_only:
                tasks = [task for task in tasks if task.is_active]
            
            return tasks
    
    def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task."""
        if not self.enabled:
            return False
        
        with self.lock:
            if task_id in self.scheduled_tasks:
                self.scheduled_tasks[task_id].is_active = False
                print(f"Paused task: {task_id}")
                return True
        
        return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        if not self.enabled:
            return False
        
        with self.lock:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.is_active = True
                task.next_run = self._calculate_next_run(task, datetime.now())
                print(f"Resumed task: {task_id}")
                return True
        
        return False
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        if not self.enabled:
            return False
        
        with self.lock:
            # Cancel running task
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
                self.running_tasks.pop(task_id)
            
            # Remove from scheduled tasks
            if task_id in self.scheduled_tasks:
                self.scheduled_tasks.pop(task_id)
                print(f"Cancelled task: {task_id}")
                return True
        
        return False
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            active_tasks = sum(1 for task in self.scheduled_tasks.values() if task.is_active)
            paused_tasks = len(self.scheduled_tasks) - active_tasks
            
            success_rate = 0.0
            if self.total_executions > 0:
                success_rate = (self.successful_executions / self.total_executions) * 100
            
            return {
                "enabled": True,
                "is_running": self.is_running,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "scheduled_tasks": len(self.scheduled_tasks),
                "active_tasks": active_tasks,
                "paused_tasks": paused_tasks,
                "running_tasks": len(self.running_tasks),
                "total_executions": self.total_executions,
                "successful_executions": self.successful_executions,
                "failed_executions": self.failed_executions,
                "success_rate": round(success_rate, 2)
            }
    
    def _cleanup_completed_tasks(self):
        """Clean up completed one-time tasks."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self.lock:
            completed_task_ids = []
            
            for task_id, task in self.scheduled_tasks.items():
                if (not task.is_active and 
                    task.config.schedule_type == ScheduleType.ONCE and
                    task.last_run and task.last_run < cutoff_time):
                    completed_task_ids.append(task_id)
            
            for task_id in completed_task_ids:
                self.completed_tasks.append(self.scheduled_tasks.pop(task_id))
    
    def _send_scheduler_telemetry(self):
        """Send scheduler telemetry."""
        if not self.telemetry:
            return
        
        stats = self.get_scheduler_statistics()
        
        self.telemetry.record_event(
            event_type="scheduler_status",
            component="concurrent_scheduler",
            operation="monitoring_cycle",
            metadata={
                "scheduled_tasks": stats.get("scheduled_tasks", 0),
                "active_tasks": stats.get("active_tasks", 0),
                "running_tasks": stats.get("running_tasks", 0),
                "success_rate": stats.get("success_rate", 0)
            }
        )
    
    def shutdown(self):
        """Shutdown the scheduler."""
        if not self.enabled or not self.is_running:
            return
        
        print("Shutting down concurrent scheduler...")
        
        # Set shutdown event
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.shutdown_event.set(), self.loop)
        
        # Cancel all running tasks
        with self.lock:
            for task_id, running_task in list(self.running_tasks.items()):
                running_task.cancel()
            
            total_scheduled = len(self.scheduled_tasks)
            total_executions = self.total_executions
        
        # Wait for scheduler to stop
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5.0)
        
        print(f"Scheduler shutdown - managed {total_scheduled} tasks, {total_executions} executions")

# Global instance
_concurrent_scheduler: Optional[ConcurrentScheduler] = None

def get_concurrent_scheduler() -> ConcurrentScheduler:
    """Get the global concurrent scheduler instance."""
    global _concurrent_scheduler
    if _concurrent_scheduler is None:
        _concurrent_scheduler = ConcurrentScheduler()
    return _concurrent_scheduler

# Convenience function
def schedule_task(name: str, function: Callable, schedule_type: ScheduleType,
                 interval_seconds: float = None, cron_expression: str = None,
                 delay_seconds: float = None, args: tuple = None,
                 kwargs: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> str:
    """
    Schedule a task for execution.
    
    Args:
        name: Task name
        function: Function to execute
        schedule_type: Type of scheduling
        interval_seconds: Interval for INTERVAL type
        cron_expression: Cron expression for CRON type
        delay_seconds: Delay for DELAYED type
        args: Function arguments
        kwargs: Function keyword arguments
        metadata: Additional metadata
        
    Returns:
        Task ID for tracking
    """
    config = ScheduleConfig(
        schedule_type=schedule_type,
        interval_seconds=interval_seconds,
        cron_expression=cron_expression,
        delay_seconds=delay_seconds
    )
    
    scheduler = get_concurrent_scheduler()
    return scheduler.schedule_task(
        name=name,
        function=function,
        schedule_config=config,
        args=args,
        kwargs=kwargs,
        metadata=metadata
    )