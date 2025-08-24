"""
Thread Pool Manager for TestMaster

Advanced thread pool management system with auto-scaling,
metrics collection, and intelligent resource allocation.
"""

import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from enum import Enum
import uuid

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from ..telemetry import get_telemetry_collector, get_performance_monitor

class PoolType(Enum):
    """Thread pool types for different workloads."""
    DEFAULT = "default"
    IO_INTENSIVE = "io_intensive"
    CPU_INTENSIVE = "cpu_intensive"
    BACKGROUND = "background"
    HIGH_PRIORITY = "high_priority"

@dataclass
class PoolConfig:
    """Configuration for a thread pool."""
    pool_type: PoolType
    min_workers: int = 2
    max_workers: int = 10
    auto_scale: bool = True
    idle_timeout: float = 60.0  # seconds
    queue_size: int = 1000
    thread_name_prefix: str = "TestMaster"
    
@dataclass
class ThreadMetrics:
    """Metrics for thread pool performance."""
    pool_name: str
    pool_type: PoolType
    current_workers: int = 0
    active_workers: int = 0
    idle_workers: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    queue_wait_time: float = 0.0
    last_auto_scale: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TaskSubmission:
    """Information about a submitted task."""
    task_id: str
    pool_name: str
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    function_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class ThreadPoolManager:
    """
    Advanced thread pool manager for TestMaster.
    
    Features:
    - Multiple specialized thread pools
    - Auto-scaling based on workload
    - Comprehensive metrics and monitoring
    - Integration with telemetry system
    - Intelligent resource allocation
    """
    
    def __init__(self):
        """Initialize thread pool manager."""
        self.enabled = FeatureFlags.is_enabled('layer2_monitoring', 'async_processing')
        
        if not self.enabled:
            return
        
        # Thread pools
        self.pools: Dict[str, ThreadPoolExecutor] = {}
        self.pool_configs: Dict[str, PoolConfig] = {}
        self.pool_metrics: Dict[str, ThreadMetrics] = {}
        
        # Task tracking
        self.active_tasks: Dict[str, TaskSubmission] = {}
        self.completed_tasks: List[TaskSubmission] = []
        
        # Management
        self.lock = threading.RLock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.is_monitoring = False
        
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
        
        # Initialize default pools
        self._initialize_default_pools()
        
        # Start monitoring
        self._start_monitoring()
        
        print("Thread pool manager initialized")
        print(f"   Default pools: {list(self.pools.keys())}")
    
    def _initialize_default_pools(self):
        """Initialize default thread pools."""
        default_configs = {
            "default": PoolConfig(
                pool_type=PoolType.DEFAULT,
                min_workers=2,
                max_workers=8,
                auto_scale=True
            ),
            "io_intensive": PoolConfig(
                pool_type=PoolType.IO_INTENSIVE,
                min_workers=4,
                max_workers=20,
                auto_scale=True,
                thread_name_prefix="TestMaster-IO"
            ),
            "cpu_intensive": PoolConfig(
                pool_type=PoolType.CPU_INTENSIVE,
                min_workers=1,
                max_workers=4,  # Usually number of CPU cores
                auto_scale=False,
                thread_name_prefix="TestMaster-CPU"
            ),
            "background": PoolConfig(
                pool_type=PoolType.BACKGROUND,
                min_workers=1,
                max_workers=3,
                auto_scale=True,
                idle_timeout=120.0,
                thread_name_prefix="TestMaster-BG"
            )
        }
        
        for name, config in default_configs.items():
            self.create_pool(name, config)
    
    def create_pool(self, name: str, config: PoolConfig) -> bool:
        """
        Create a new thread pool.
        
        Args:
            name: Pool name
            config: Pool configuration
            
        Returns:
            True if pool was created successfully
        """
        if not self.enabled:
            return False
        
        with self.lock:
            if name in self.pools:
                print(f"Pool '{name}' already exists")
                return False
            
            # Create thread pool
            pool = ThreadPoolExecutor(
                max_workers=config.max_workers,
                thread_name_prefix=f"{config.thread_name_prefix}-{name}"
            )
            
            # Store pool and config
            self.pools[name] = pool
            self.pool_configs[name] = config
            
            # Initialize metrics
            self.pool_metrics[name] = ThreadMetrics(
                pool_name=name,
                pool_type=config.pool_type,
                current_workers=config.min_workers
            )
            
            print(f"Created thread pool '{name}' ({config.pool_type.value})")
            
            # Send telemetry
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="thread_pool_created",
                    component="thread_pool_manager",
                    operation="create_pool",
                    metadata={
                        "pool_name": name,
                        "pool_type": config.pool_type.value,
                        "max_workers": config.max_workers
                    }
                )
            
            return True
    
    def configure_pool(self, name: str, **kwargs) -> bool:
        """
        Configure an existing pool.
        
        Args:
            name: Pool name
            **kwargs: Configuration parameters
            
        Returns:
            True if configuration was successful
        """
        if not self.enabled:
            return False
        
        with self.lock:
            if name not in self.pools:
                print(f"Pool '{name}' does not exist")
                return False
            
            config = self.pool_configs[name]
            
            # Update configuration
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print(f"Configured pool '{name}': {kwargs}")
            return True
    
    def submit_task(self, pool_name: str, func: Callable, *args, 
                   metadata: Dict[str, Any] = None, **kwargs) -> str:
        """
        Submit a task to a specific pool.
        
        Args:
            pool_name: Target pool name
            func: Function to execute
            *args: Function arguments
            metadata: Task metadata
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
        """
        if not self.enabled:
            raise RuntimeError("Thread pool manager is disabled")
        
        with self.lock:
            if pool_name not in self.pools:
                raise ValueError(f"Pool '{pool_name}' does not exist")
            
            pool = self.pools[pool_name]
            
            # Create task submission record
            task_id = str(uuid.uuid4())
            submission = TaskSubmission(
                task_id=task_id,
                pool_name=pool_name,
                submitted_at=datetime.now(),
                function_name=getattr(func, '__name__', str(func)),
                metadata=metadata or {}
            )
            
            # Wrap function for monitoring
            def monitored_func():
                return self._execute_with_monitoring(submission, func, *args, **kwargs)
            
            # Submit to pool
            future = pool.submit(monitored_func)
            
            # Store task info
            self.active_tasks[task_id] = submission
            
            # Update metrics
            metrics = self.pool_metrics[pool_name]
            metrics.pending_tasks += 1
            
            # Send telemetry
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="task_submitted",
                    component="thread_pool_manager",
                    operation="submit_task",
                    metadata={
                        "task_id": task_id,
                        "pool_name": pool_name,
                        "function_name": submission.function_name
                    }
                )
            
            return task_id
    
    def _execute_with_monitoring(self, submission: TaskSubmission, 
                               func: Callable, *args, **kwargs):
        """Execute function with comprehensive monitoring."""
        start_time = time.time()
        submission.started_at = datetime.now()
        
        try:
            # Update metrics
            with self.lock:
                metrics = self.pool_metrics[submission.pool_name]
                metrics.pending_tasks -= 1
                metrics.active_workers += 1
            
            # Execute function
            if self.performance_monitor:
                with self.performance_monitor.track_operation(
                    "thread_pool", f"{submission.pool_name}_{submission.function_name}"
                ):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Task succeeded
            execution_time = time.time() - start_time
            submission.completed_at = datetime.now()
            
            # Update metrics
            with self.lock:
                metrics = self.pool_metrics[submission.pool_name]
                metrics.active_workers -= 1
                metrics.completed_tasks += 1
                metrics.total_execution_time += execution_time
                metrics.avg_execution_time = (
                    metrics.total_execution_time / metrics.completed_tasks
                )
                
                # Move to completed tasks
                self.completed_tasks.append(submission)
                self.active_tasks.pop(submission.task_id, None)
            
            # Send telemetry
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="task_completed",
                    component="thread_pool_manager",
                    operation="execute_task",
                    metadata={
                        "task_id": submission.task_id,
                        "pool_name": submission.pool_name,
                        "function_name": submission.function_name
                    },
                    duration_ms=execution_time * 1000,
                    success=True
                )
            
            # Update shared state
            if self.shared_state:
                self.shared_state.increment("thread_pool_tasks_completed")
            
            return result
            
        except Exception as e:
            # Task failed
            execution_time = time.time() - start_time
            submission.completed_at = datetime.now()
            
            # Update metrics
            with self.lock:
                metrics = self.pool_metrics[submission.pool_name]
                metrics.active_workers -= 1
                metrics.failed_tasks += 1
                
                # Move to completed tasks
                self.completed_tasks.append(submission)
                self.active_tasks.pop(submission.task_id, None)
            
            # Send telemetry
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="task_failed",
                    component="thread_pool_manager",
                    operation="execute_task",
                    metadata={
                        "task_id": submission.task_id,
                        "pool_name": submission.pool_name,
                        "function_name": submission.function_name,
                        "error_type": type(e).__name__
                    },
                    duration_ms=execution_time * 1000,
                    success=False,
                    error_message=str(e)
                )
            
            # Update shared state
            if self.shared_state:
                self.shared_state.increment("thread_pool_tasks_failed")
            
            raise
    
    def get_pool_metrics(self, pool_name: str = None) -> Union[ThreadMetrics, Dict[str, ThreadMetrics]]:
        """
        Get metrics for a specific pool or all pools.
        
        Args:
            pool_name: Specific pool name, or None for all
            
        Returns:
            ThreadMetrics or dict of all metrics
        """
        if not self.enabled:
            return {}
        
        with self.lock:
            if pool_name:
                return self.pool_metrics.get(pool_name)
            else:
                return dict(self.pool_metrics)
    
    def get_active_tasks(self, pool_name: str = None) -> List[TaskSubmission]:
        """Get active tasks for a specific pool or all pools."""
        if not self.enabled:
            return []
        
        with self.lock:
            if pool_name:
                return [task for task in self.active_tasks.values() 
                       if task.pool_name == pool_name]
            else:
                return list(self.active_tasks.values())
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get overall pool manager status."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            total_workers = sum(metrics.current_workers for metrics in self.pool_metrics.values())
            total_active = sum(metrics.active_workers for metrics in self.pool_metrics.values())
            total_pending = sum(metrics.pending_tasks for metrics in self.pool_metrics.values())
            total_completed = sum(metrics.completed_tasks for metrics in self.pool_metrics.values())
            total_failed = sum(metrics.failed_tasks for metrics in self.pool_metrics.values())
            
            return {
                "enabled": True,
                "total_pools": len(self.pools),
                "total_workers": total_workers,
                "active_workers": total_active,
                "pending_tasks": total_pending,
                "completed_tasks": total_completed,
                "failed_tasks": total_failed,
                "success_rate": (total_completed / max(total_completed + total_failed, 1)) * 100,
                "pools": {name: {
                    "type": config.pool_type.value,
                    "max_workers": config.max_workers,
                    "auto_scale": config.auto_scale,
                    "current_workers": metrics.current_workers,
                    "active_workers": metrics.active_workers,
                    "pending_tasks": metrics.pending_tasks
                } for name, config in self.pool_configs.items() 
                  for metrics in [self.pool_metrics[name]]}
            }
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        if not self.enabled:
            return
        
        def monitor_worker():
            self.is_monitoring = True
            
            while not self.shutdown_event.is_set():
                try:
                    # Perform auto-scaling
                    self._auto_scale_pools()
                    
                    # Update metrics
                    self._update_pool_metrics()
                    
                    # Cleanup old completed tasks
                    self._cleanup_completed_tasks()
                    
                    # Wait for next cycle
                    if self.shutdown_event.wait(timeout=30):  # Check every 30 seconds
                        break
                        
                except Exception as e:
                    print(f"Pool monitoring error: {e}")
            
            self.is_monitoring = False
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
    
    def _auto_scale_pools(self):
        """Auto-scale pools based on workload."""
        current_time = datetime.now()
        
        with self.lock:
            for name, config in self.pool_configs.items():
                if not config.auto_scale:
                    continue
                
                metrics = self.pool_metrics[name]
                pool = self.pools[name]
                
                # Check if scaling is needed
                scale_up = (metrics.pending_tasks > metrics.current_workers and 
                           metrics.current_workers < config.max_workers)
                
                scale_down = (metrics.active_workers == 0 and 
                             metrics.pending_tasks == 0 and
                             metrics.current_workers > config.min_workers and
                             metrics.last_auto_scale and
                             (current_time - metrics.last_auto_scale).seconds > config.idle_timeout)
                
                if scale_up:
                    # Scale up
                    new_max = min(config.max_workers, metrics.current_workers + 2)
                    pool._max_workers = new_max
                    metrics.current_workers = new_max
                    metrics.last_auto_scale = current_time
                    
                    print(f"Scaled up pool '{name}' to {new_max} workers")
                    
                elif scale_down:
                    # Scale down
                    new_max = max(config.min_workers, metrics.current_workers - 1)
                    pool._max_workers = new_max
                    metrics.current_workers = new_max
                    metrics.last_auto_scale = current_time
                    
                    print(f"Scaled down pool '{name}' to {new_max} workers")
    
    def _update_pool_metrics(self):
        """Update pool metrics."""
        with self.lock:
            for name, pool in self.pools.items():
                metrics = self.pool_metrics[name]
                
                # Update worker counts
                metrics.idle_workers = metrics.current_workers - metrics.active_workers
                
                # Update shared state
                if self.shared_state:
                    self.shared_state.set(f"pool_{name}_workers", metrics.current_workers)
                    self.shared_state.set(f"pool_{name}_active", metrics.active_workers)
                    self.shared_state.set(f"pool_{name}_pending", metrics.pending_tasks)
    
    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self.lock:
            self.completed_tasks = [
                task for task in self.completed_tasks
                if task.completed_at and task.completed_at > cutoff_time
            ]
    
    def shutdown_pool(self, name: str) -> bool:
        """Shutdown a specific pool."""
        if not self.enabled:
            return False
        
        with self.lock:
            if name not in self.pools:
                return False
            
            pool = self.pools[name]
            pool.shutdown(wait=True)
            
            # Remove from tracking
            del self.pools[name]
            del self.pool_configs[name]
            del self.pool_metrics[name]
            
            print(f"Shutdown pool '{name}'")
            return True
    
    def shutdown_all_pools(self):
        """Shutdown all pools."""
        if not self.enabled:
            return
        
        print("Shutting down all thread pools...")
        
        # Stop monitoring
        self.shutdown_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        # Shutdown all pools
        with self.lock:
            for name, pool in list(self.pools.items()):
                pool.shutdown(wait=True)
            
            total_completed = sum(metrics.completed_tasks for metrics in self.pool_metrics.values())
            total_failed = sum(metrics.failed_tasks for metrics in self.pool_metrics.values())
            
            self.pools.clear()
            self.pool_configs.clear()
            self.pool_metrics.clear()
        
        print(f"Thread pools shutdown - completed {total_completed} tasks, failed {total_failed}")

# Global instance
_thread_pool_manager: Optional[ThreadPoolManager] = None

def get_thread_pool_manager() -> ThreadPoolManager:
    """Get the global thread pool manager instance."""
    global _thread_pool_manager
    if _thread_pool_manager is None:
        _thread_pool_manager = ThreadPoolManager()
    return _thread_pool_manager

# Convenience function
def submit_task(func: Callable, *args, pool_name: str = "default",
               metadata: Dict[str, Any] = None, **kwargs) -> str:
    """
    Submit a task to a thread pool.
    
    Args:
        func: Function to execute
        *args: Function arguments
        pool_name: Target pool name
        metadata: Task metadata
        **kwargs: Function keyword arguments
        
    Returns:
        Task ID for tracking
    """
    manager = get_thread_pool_manager()
    return manager.submit_task(pool_name, func, *args, metadata=metadata, **kwargs)