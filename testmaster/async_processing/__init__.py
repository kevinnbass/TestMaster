"""
TestMaster Async Thread Processing

Advanced asynchronous processing system inspired by multi-agent frameworks,
providing high-performance concurrent execution, thread pooling, and
async-aware monitoring capabilities.

Features:
- Advanced thread pool management with auto-scaling
- Async-aware execution contexts and state management
- Priority-based task scheduling and execution
- Comprehensive async monitoring and telemetry
- Integration with existing TestMaster components
"""

from .async_executor import (
    AsyncExecutor, ExecutionContext, TaskPriority,
    async_execute, get_async_executor
)
from .thread_pool_manager import (
    ThreadPoolManager, PoolConfig, ThreadMetrics, PoolType,
    get_thread_pool_manager, submit_task
)
from .async_monitor import (
    AsyncMonitor, AsyncTaskInfo, ExecutionStats,
    get_async_monitor, track_async_execution
)
from .concurrent_scheduler import (
    ConcurrentScheduler, ScheduledTask, ScheduleConfig, ScheduleType,
    get_concurrent_scheduler, schedule_task
)
from .async_state_manager import (
    AsyncStateManager, AsyncContext, StateScope,
    get_async_state_manager, async_context
)

__all__ = [
    # Core async execution
    'AsyncExecutor',
    'ExecutionContext',
    'TaskPriority',
    'async_execute',
    'get_async_executor',
    
    # Thread pool management
    'ThreadPoolManager',
    'PoolConfig',
    'ThreadMetrics',
    'PoolType',
    'get_thread_pool_manager',
    'submit_task',
    
    # Async monitoring
    'AsyncMonitor',
    'AsyncTaskInfo',
    'ExecutionStats',
    'get_async_monitor',
    'track_async_execution',
    
    # Concurrent scheduling
    'ConcurrentScheduler',
    'ScheduledTask',
    'ScheduleConfig',
    'ScheduleType',
    'get_concurrent_scheduler',
    'schedule_task',
    
    # Async state management
    'AsyncStateManager',
    'AsyncContext',
    'StateScope',
    'get_async_state_manager',
    'async_context',
    
    # Utilities
    'is_async_enabled',
    'configure_async_processing',
    'shutdown_async_processing'
]

def is_async_enabled() -> bool:
    """Check if async processing is enabled."""
    from ..core.feature_flags import FeatureFlags
    return FeatureFlags.is_enabled('layer2_monitoring', 'async_processing')

def configure_async_processing(max_workers: int = None, 
                             enable_monitoring: bool = True,
                             enable_scheduling: bool = True):
    """Configure async processing system."""
    if not is_async_enabled():
        print("Async processing is disabled")
        return
    
    # Configure thread pool
    if max_workers:
        pool_manager = get_thread_pool_manager()
        pool_manager.configure_pool("default", max_workers=max_workers)
    
    # Configure monitoring
    if enable_monitoring:
        monitor = get_async_monitor()
        monitor.start_monitoring()
    
    # Configure scheduler
    if enable_scheduling:
        scheduler = get_concurrent_scheduler()
        scheduler.start_scheduler()
    
    print(f"Async processing configured (workers: {max_workers}, monitoring: {enable_monitoring})")

def shutdown_async_processing():
    """Shutdown all async processing components."""
    try:
        # Shutdown in reverse order of dependencies
        scheduler = get_concurrent_scheduler()
        scheduler.shutdown()
        
        monitor = get_async_monitor()
        monitor.shutdown()
        
        pool_manager = get_thread_pool_manager()
        pool_manager.shutdown_all_pools()
        
        executor = get_async_executor()
        executor.shutdown()
        
        state_manager = get_async_state_manager()
        state_manager.cleanup()
        
        print("Async processing shutdown completed")
    except Exception as e:
        print(f"Error during async processing shutdown: {e}")

# Initialize async processing if enabled
if is_async_enabled():
    configure_async_processing()