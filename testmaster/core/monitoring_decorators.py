"""
Performance Monitoring Decorators

Toggleable performance monitoring for TestMaster components.
Based on PraisonAI's monitoring patterns.

These decorators provide detailed performance metrics including:
- Execution time
- Memory usage (optional)
- Call frequency
- Error rates
"""

import time
import tracemalloc
from functools import wraps
from typing import Any, Callable, Optional, Dict
import threading
from datetime import datetime

from .feature_flags import FeatureFlags
from .shared_state import get_shared_state


class PerformanceMonitor:
    """Centralized performance monitoring system."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for performance monitor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize performance monitor."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self._metrics = {}
            self._call_counts = {}
            self._error_counts = {}
            self._timing_history = {}
            self._memory_tracking = False
            self._initialized = True
    
    def record_execution(self, name: str, elapsed: float, memory_delta: Optional[int] = None, error: bool = False):
        """Record execution metrics."""
        if name not in self._metrics:
            self._metrics[name] = {
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'avg_time': 0,
                'call_count': 0,
                'error_count': 0,
                'memory_peak': 0,
                'last_called': None
            }
        
        metrics = self._metrics[name]
        metrics['call_count'] += 1
        metrics['total_time'] += elapsed
        metrics['min_time'] = min(metrics['min_time'], elapsed)
        metrics['max_time'] = max(metrics['max_time'], elapsed)
        metrics['avg_time'] = metrics['total_time'] / metrics['call_count']
        metrics['last_called'] = datetime.now()
        
        if error:
            metrics['error_count'] += 1
        
        if memory_delta:
            metrics['memory_peak'] = max(metrics['memory_peak'], memory_delta)
        
        # Keep timing history (last 100 calls)
        if name not in self._timing_history:
            self._timing_history[name] = []
        
        self._timing_history[name].append(elapsed)
        if len(self._timing_history[name]) > 100:
            self._timing_history[name].pop(0)
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics."""
        if name:
            return self._metrics.get(name, {})
        return self._metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_calls = sum(m['call_count'] for m in self._metrics.values())
        total_time = sum(m['total_time'] for m in self._metrics.values())
        total_errors = sum(m['error_count'] for m in self._metrics.values())
        
        slowest_functions = sorted(
            self._metrics.items(),
            key=lambda x: x[1]['avg_time'],
            reverse=True
        )[:5]
        
        most_called = sorted(
            self._metrics.items(),
            key=lambda x: x[1]['call_count'],
            reverse=True
        )[:5]
        
        return {
            'total_functions_monitored': len(self._metrics),
            'total_calls': total_calls,
            'total_time': total_time,
            'total_errors': total_errors,
            'error_rate': (total_errors / max(1, total_calls)) * 100,
            'slowest_functions': [(name, metrics['avg_time']) for name, metrics in slowest_functions],
            'most_called_functions': [(name, metrics['call_count']) for name, metrics in most_called],
            'memory_tracking_enabled': self._memory_tracking
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._call_counts.clear()
        self._error_counts.clear()
        self._timing_history.clear()


def monitor_performance(name: Optional[str] = None, track_memory: bool = False):
    """
    Toggleable performance monitoring decorator.
    
    Args:
        name: Optional name for the function (defaults to function name)
        track_memory: Whether to track memory usage
    
    Example:
        @monitor_performance(name="critical_function")
        def process_data(data):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if monitoring is enabled
            if not FeatureFlags.is_enabled('layer1_test_foundation', 'performance_monitoring'):
                return func(*args, **kwargs)
            
            func_name = name or func.__name__
            monitor = PerformanceMonitor()
            shared_state = get_shared_state() if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state') else None
            
            # Get configuration
            config = FeatureFlags.get_config('layer1_test_foundation', 'performance_monitoring')
            should_track_memory = track_memory or config.get('include_memory', False)
            
            # Start timing
            start_time = time.perf_counter()
            memory_before = None
            
            # Optional memory tracking
            if should_track_memory:
                tracemalloc.start()
                memory_before = tracemalloc.get_traced_memory()[0]
            
            error_occurred = False
            try:
                # Execute function
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                error_occurred = True
                # Log error but re-raise
                if shared_state:
                    shared_state.append(f"errors_{func_name}", {
                        'error': str(e),
                        'timestamp': time.time(),
                        'args': str(args)[:100],  # Truncate for safety
                        'kwargs': str(kwargs)[:100]
                    })
                raise
                
            finally:
                # Calculate elapsed time
                elapsed = time.perf_counter() - start_time
                
                # Calculate memory delta if tracking
                memory_delta = None
                if should_track_memory and memory_before is not None:
                    memory_after = tracemalloc.get_traced_memory()[0]
                    memory_delta = memory_after - memory_before
                    tracemalloc.stop()
                
                # Record metrics
                monitor.record_execution(func_name, elapsed, memory_delta, error_occurred)
                
                # Update shared state if enabled
                if shared_state:
                    shared_state.set(f"perf_{func_name}_last", {
                        'elapsed': elapsed,
                        'timestamp': time.time(),
                        'memory_delta': memory_delta,
                        'error': error_occurred
                    }, ttl=3600)
                    
                    # Update cumulative stats
                    shared_state.increment(f"perf_{func_name}_calls")
                    shared_state.increment(f"perf_{func_name}_total_ms", int(elapsed * 1000))
                
                # Log performance (only if significant time)
                if elapsed > 0.1:  # Log if > 100ms
                    memory_str = f", Δmem: {memory_delta/1024:.1f}KB" if memory_delta else ""
                    error_str = " [ERROR]" if error_occurred else ""
                    print(f"⚡ {func_name}: {elapsed:.3f}s{memory_str}{error_str}")
        
        # Add monitoring metadata to function
        wrapper._monitored = True
        wrapper._monitor_name = name or func.__name__
        
        return wrapper
    return decorator


def monitor_class(cls: type) -> type:
    """
    Apply performance monitoring to all methods of a class.
    
    Args:
        cls: The class to monitor
    
    Returns:
        The monitored class
    """
    if not FeatureFlags.is_enabled('layer1_test_foundation', 'performance_monitoring'):
        return cls
    
    # Get all methods that should be monitored
    for attr_name in dir(cls):
        # Skip private/magic methods except __init__
        if attr_name.startswith('_') and attr_name != '__init__':
            continue
        
        attr = getattr(cls, attr_name)
        if callable(attr) and not getattr(attr, '_monitored', False):
            # Apply monitoring decorator
            monitored_method = monitor_performance(name=f"{cls.__name__}.{attr_name}")(attr)
            setattr(cls, attr_name, monitored_method)
    
    return cls


def get_performance_summary() -> Dict[str, Any]:
    """
    Get a summary of all performance metrics.
    
    Returns:
        Dictionary containing performance summary
    """
    if not FeatureFlags.is_enabled('layer1_test_foundation', 'performance_monitoring'):
        return {'enabled': False}
    
    monitor = PerformanceMonitor()
    summary = monitor.get_summary()
    
    # Add shared state metrics if available
    if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
        shared_state = get_shared_state()
        
        # Get all performance keys from shared state
        perf_keys = shared_state.keys("perf_*")
        summary['shared_state_metrics'] = len(perf_keys)
    
    return summary


def reset_performance_metrics():
    """Reset all performance metrics."""
    if not FeatureFlags.is_enabled('layer1_test_foundation', 'performance_monitoring'):
        return
    
    monitor = PerformanceMonitor()
    monitor.reset_metrics()
    
    # Clear shared state metrics if enabled
    if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
        shared_state = get_shared_state()
        shared_state.clear("perf_*")
    
    print("🔄 Performance metrics reset")


# Convenience decorators with preset configurations
def monitor_critical(func: Callable) -> Callable:
    """Monitor critical functions with memory tracking."""
    return monitor_performance(track_memory=True)(func)


def monitor_api(func: Callable) -> Callable:
    """Monitor API endpoints."""
    return monitor_performance(name=f"api_{func.__name__}")(func)


def monitor_generator(func: Callable) -> Callable:
    """Monitor test generator functions."""
    return monitor_performance(name=f"gen_{func.__name__}")(func)


def monitor_verifier(func: Callable) -> Callable:
    """Monitor test verifier functions."""
    return monitor_performance(name=f"verify_{func.__name__}")(func)