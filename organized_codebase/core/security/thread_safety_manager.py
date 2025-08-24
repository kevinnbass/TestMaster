"""
CrewAI Derived Thread Safety Manager
Extracted from CrewAI thread safety patterns and concurrent execution handling
Enhanced for comprehensive thread-safe operations and context isolation
"""

import logging
import threading
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from .error_handler import SecurityError, ValidationError, security_error_handler

T = TypeVar('T')


@dataclass
class ThreadSafetyConfig:
    """Thread safety configuration based on CrewAI patterns"""
    max_threads: int = 4
    thread_timeout: int = 300
    context_isolation: bool = True
    rpm_limit: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if self.max_threads <= 0:
            raise ValidationError("max_threads must be positive")
        if self.rpm_limit <= 0:
            raise ValidationError("rpm_limit must be positive")


@dataclass
class ThreadContext:
    """Thread-local context for crew operations"""
    thread_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    crew_id: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    isolation_active: bool = True
    
    @property
    def age_seconds(self) -> float:
        """Get context age in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()


class ThreadLocalContextManager:
    """Thread-local context management based on CrewAI patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._local = threading.local()
        self._contexts: Dict[str, ThreadContext] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def get_current_context(self) -> Optional[ThreadContext]:
        """Get current thread's context"""
        try:
            return getattr(self._local, 'context', None)
        except AttributeError:
            return None
    
    def set_current_context(self, context: ThreadContext):
        """Set current thread's context with isolation"""
        try:
            thread_id = threading.current_thread().ident
            
            with self._lock:
                # Store in thread-local storage
                self._local.context = context
                
                # Store in global registry for management
                self._contexts[str(thread_id)] = context
                
                # Cleanup old contexts periodically
                self._periodic_cleanup()
                
                self.logger.debug(f"Set context for thread {thread_id}")
                
        except Exception as e:
            error = SecurityError(f"Failed to set thread context: {str(e)}", "THREAD_CTX_001")
            security_error_handler.handle_error(error)
    
    def clear_current_context(self):
        """Clear current thread's context"""
        try:
            thread_id = threading.current_thread().ident
            
            with self._lock:
                # Clear thread-local storage
                if hasattr(self._local, 'context'):
                    delattr(self._local, 'context')
                
                # Remove from global registry
                self._contexts.pop(str(thread_id), None)
                
                self.logger.debug(f"Cleared context for thread {thread_id}")
                
        except Exception as e:
            self.logger.error(f"Error clearing thread context: {e}")
    
    def get_all_contexts(self) -> Dict[str, ThreadContext]:
        """Get all active thread contexts"""
        with self._lock:
            return self._contexts.copy()
    
    @contextmanager
    def isolated_context(self, crew_id: str = None, context_data: Dict[str, Any] = None):
        """Context manager for isolated thread execution"""
        thread_id = str(threading.current_thread().ident)
        
        context = ThreadContext(
            thread_id=thread_id,
            crew_id=crew_id,
            context_data=context_data or {},
            isolation_active=True
        )
        
        old_context = self.get_current_context()
        
        try:
            self.set_current_context(context)
            yield context
        finally:
            if old_context:
                self.set_current_context(old_context)
            else:
                self.clear_current_context()
    
    def _periodic_cleanup(self):
        """Cleanup old contexts periodically"""
        current_time = time.time()
        
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._last_cleanup = current_time
            
            # Remove contexts older than 1 hour
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            to_remove = []
            for thread_id, context in self._contexts.items():
                if context.created_at < cutoff_time:
                    to_remove.append(thread_id)
            
            for thread_id in to_remove:
                self._contexts.pop(thread_id, None)
            
            if to_remove:
                self.logger.info(f"Cleaned up {len(to_remove)} old thread contexts")


class RPMController:
    """Request Per Minute controller with thread safety based on CrewAI patterns"""
    
    def __init__(self, rpm_limit: int = 100):
        self.rpm_limit = rpm_limit
        self.logger = logging.getLogger(__name__)
        self._request_times: List[float] = []
        self._lock = threading.Lock()
        self._last_reset = time.time()
    
    def can_make_request(self) -> bool:
        """Check if request can be made within RPM limits"""
        with self._lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            cutoff_time = current_time - 60
            self._request_times = [t for t in self._request_times if t > cutoff_time]
            
            return len(self._request_times) < self.rpm_limit
    
    def record_request(self):
        """Record a new request timestamp"""
        with self._lock:
            self._request_times.append(time.time())
    
    def wait_if_needed(self) -> bool:
        """Wait if RPM limit exceeded, return True if waited"""
        if self.can_make_request():
            self.record_request()
            return False
        
        # Calculate wait time
        with self._lock:
            if self._request_times:
                oldest_request = min(self._request_times)
                wait_time = 60 - (time.time() - oldest_request)
                
                if wait_time > 0:
                    self.logger.info(f"RPM limit reached, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    
                    # Record request after waiting
                    self.record_request()
                    return True
        
        return False
    
    def get_current_rpm(self) -> int:
        """Get current requests per minute"""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - 60
            recent_requests = [t for t in self._request_times if t > cutoff_time]
            return len(recent_requests)


class ThreadSafetyManager:
    """Comprehensive thread safety management system"""
    
    def __init__(self, config: ThreadSafetyConfig = None):
        self.config = config or ThreadSafetyConfig()
        self.logger = logging.getLogger(__name__)
        self.context_manager = ThreadLocalContextManager()
        self.rpm_controller = RPMController(self.config.rpm_limit)
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_threads)
        self._active_futures: Dict[str, Future] = {}
        self._lock = threading.RLock()
    
    def execute_thread_safe(self, func: Callable[[], T], 
                           crew_id: str = None,
                           context_data: Dict[str, Any] = None) -> T:
        """Execute function in thread-safe manner with context isolation"""
        try:
            with self.context_manager.isolated_context(crew_id, context_data):
                # Check RPM limits
                self.rpm_controller.wait_if_needed()
                
                # Execute function
                result = func()
                
                self.logger.debug(f"Thread-safe execution completed for crew {crew_id}")
                return result
                
        except Exception as e:
            error = SecurityError(f"Thread-safe execution failed: {str(e)}", "THREAD_EXEC_001")
            security_error_handler.handle_error(error)
            raise error
    
    def submit_concurrent_task(self, func: Callable[[], T],
                              task_id: str,
                              crew_id: str = None,
                              context_data: Dict[str, Any] = None) -> Future[T]:
        """Submit task for concurrent execution"""
        try:
            with self._lock:
                future = self._executor.submit(
                    self.execute_thread_safe,
                    func, crew_id, context_data
                )
                
                self._active_futures[task_id] = future
                
                self.logger.info(f"Submitted concurrent task: {task_id}")
                return future
                
        except Exception as e:
            error = SecurityError(f"Failed to submit concurrent task: {str(e)}", "THREAD_SUBMIT_001")
            security_error_handler.handle_error(error)
            raise error
    
    def wait_for_completion(self, timeout: int = None) -> Dict[str, Any]:
        """Wait for all active tasks to complete"""
        timeout = timeout or self.config.thread_timeout
        results = {}
        errors = {}
        
        try:
            with self._lock:
                futures_copy = self._active_futures.copy()
            
            for task_id, future in as_completed(futures_copy.values(), timeout=timeout):
                # Find task_id for this future
                task_name = None
                for tid, fut in futures_copy.items():
                    if fut is future:
                        task_name = tid
                        break
                
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    errors[task_name] = str(e)
                    self.logger.error(f"Task {task_name} failed: {e}")
            
            # Clear completed futures
            with self._lock:
                for task_id in futures_copy:
                    self._active_futures.pop(task_id, None)
            
            return {
                'results': results,
                'errors': errors,
                'completed_count': len(results),
                'error_count': len(errors)
            }
            
        except Exception as e:
            error = SecurityError(f"Error waiting for task completion: {str(e)}", "THREAD_WAIT_001")
            security_error_handler.handle_error(error)
            raise error
    
    @asynccontextmanager
    async def async_context_isolation(self, crew_id: str = None,
                                     context_data: Dict[str, Any] = None):
        """Async context manager for isolated execution"""
        thread_id = str(threading.current_thread().ident)
        
        context = ThreadContext(
            thread_id=thread_id,
            crew_id=crew_id,
            context_data=context_data or {},
            isolation_active=True
        )
        
        try:
            self.context_manager.set_current_context(context)
            yield context
        finally:
            self.context_manager.clear_current_context()
    
    def get_thread_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive thread safety status"""
        try:
            with self._lock:
                active_tasks = len(self._active_futures)
            
            contexts = self.context_manager.get_all_contexts()
            current_rpm = self.rpm_controller.get_current_rpm()
            
            return {
                'config': {
                    'max_threads': self.config.max_threads,
                    'rpm_limit': self.config.rpm_limit,
                    'thread_timeout': self.config.thread_timeout
                },
                'status': {
                    'active_tasks': active_tasks,
                    'active_contexts': len(contexts),
                    'current_rpm': current_rpm,
                    'rpm_utilization_pct': (current_rpm / self.config.rpm_limit) * 100
                },
                'contexts': [
                    {
                        'thread_id': ctx.thread_id,
                        'crew_id': ctx.crew_id,
                        'age_seconds': ctx.age_seconds,
                        'isolation_active': ctx.isolation_active
                    }
                    for ctx in contexts.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting thread safety status: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown thread safety manager and cleanup resources"""
        try:
            self.logger.info("Shutting down thread safety manager")
            
            # Cancel active futures
            with self._lock:
                for task_id, future in self._active_futures.items():
                    if not future.done():
                        future.cancel()
                        self.logger.info(f"Cancelled task: {task_id}")
                self._active_futures.clear()
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Clear all contexts
            self.context_manager._contexts.clear()
            
            self.logger.info("Thread safety manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during thread safety manager shutdown: {e}")


# Global thread safety manager
thread_safety_manager = ThreadSafetyManager()


# Convenience functions
def execute_thread_safe(func: Callable[[], T], crew_id: str = None) -> T:
    """Convenience function for thread-safe execution"""
    return thread_safety_manager.execute_thread_safe(func, crew_id)


def submit_concurrent_task(func: Callable[[], T], task_id: str, crew_id: str = None) -> Future[T]:
    """Convenience function for concurrent task submission"""
    return thread_safety_manager.submit_concurrent_task(func, task_id, crew_id)


@contextmanager
def isolated_context(crew_id: str = None, context_data: Dict[str, Any] = None):
    """Convenience context manager for thread isolation"""
    with thread_safety_manager.context_manager.isolated_context(crew_id, context_data) as ctx:
        yield ctx