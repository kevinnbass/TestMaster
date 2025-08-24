"""
Analytics Circuit Breaker System
================================

Implements circuit breaker pattern for analytics components to provide
fault tolerance, graceful degradation, and automatic recovery.

Author: TestMaster Team
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import functools

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failures detected, blocking calls
    HALF_OPEN = "half_open" # Testing if service has recovered

class FailureType(Enum):
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    VALIDATION = "validation"
    RESOURCE = "resource"
    UNAVAILABLE = "unavailable"

@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = 5           # Failures before opening
    timeout_seconds: float = 30.0        # Timeout for operations
    recovery_timeout: int = 60           # Seconds before trying half-open
    success_threshold: int = 3           # Successes needed to close from half-open
    max_concurrent_calls: int = 10       # Max concurrent operations
    
class FailureRecord:
    """Records failure information."""
    def __init__(self, failure_type: FailureType, error: str, timestamp: datetime):
        self.failure_type = failure_type
        self.error = error
        self.timestamp = timestamp

class CircuitBreaker:
    """
    Circuit breaker implementation for individual components.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Component name
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.failure_count = 0
        self.success_count = 0
        
        # Failure tracking
        self.failures = deque(maxlen=100)
        self.call_history = deque(maxlen=1000)
        
        # Concurrency control
        self.active_calls = 0
        self.call_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'blocked_calls': 0,
            'timeouts': 0,
            'state_changes': 0,
            'last_state_change': None
        }
        
        # Callbacks
        self.state_change_callbacks = []
        self.failure_callbacks = []
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def __call__(self, func):
        """Decorator to wrap functions with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpenException: When circuit is open
            Exception: Original function exceptions when circuit is closed
        """
        with self.call_lock:
            self.stats['total_calls'] += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats['blocked_calls'] += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Last failure: {self.last_failure_time}"
                    )
            
            # Check concurrent call limit
            if self.active_calls >= self.config.max_concurrent_calls:
                self.stats['blocked_calls'] += 1
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' has reached max concurrent calls limit"
                )
            
            self.active_calls += 1
        
        call_start = time.time()
        success = False
        error = None
        
        try:
            # Execute with timeout
            result = self._execute_with_timeout(func, *args, **kwargs)
            success = True
            self._record_success(call_start)
            return result
            
        except TimeoutError as e:
            error = str(e)
            self._record_failure(FailureType.TIMEOUT, error, call_start)
            raise
            
        except Exception as e:
            error = str(e)
            self._record_failure(FailureType.EXCEPTION, error, call_start)
            raise
            
        finally:
            with self.call_lock:
                self.active_calls -= 1
            
            # Record call history
            self.call_history.append({
                'timestamp': datetime.now(),
                'duration': time.time() - call_start,
                'success': success,
                'error': error
            })
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout."""
        if self.config.timeout_seconds <= 0:
            return func(*args, **kwargs)
        
        result = None
        exception = None
        finished = threading.Event()
        
        def target():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
            finally:
                finished.set()
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        
        if finished.wait(timeout=self.config.timeout_seconds):
            if exception:
                raise exception
            return result
        else:
            # Timeout occurred
            self.stats['timeouts'] += 1
            raise TimeoutError(f"Function timed out after {self.config.timeout_seconds} seconds")
    
    def _record_success(self, call_start: float):
        """Record successful call."""
        with self.call_lock:
            self.stats['successful_calls'] += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def _record_failure(self, failure_type: FailureType, error: str, call_start: float):
        """Record failed call."""
        with self.call_lock:
            self.stats['failed_calls'] += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            # Record failure details
            failure = FailureRecord(failure_type, error, self.last_failure_time)
            self.failures.append(failure)
            
            # Trigger failure callbacks
            for callback in self.failure_callbacks:
                try:
                    callback(self, failure)
                except Exception as e:
                    logger.error(f"Failure callback error: {e}")
            
            # Check if should open circuit
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset from open state."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _transition_to_open(self):
        """Transition to open state."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.stats['state_changes'] += 1
            self.stats['last_state_change'] = datetime.now()
            logger.warning(f"Circuit breaker '{self.name}' opened due to failures")
            self._notify_state_change()
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self.stats['state_changes'] += 1
            self.stats['last_state_change'] = datetime.now()
            logger.info(f"Circuit breaker '{self.name}' half-opened for testing")
            self._notify_state_change()
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.stats['state_changes'] += 1
            self.stats['last_state_change'] = datetime.now()
            logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
            self._notify_state_change()
    
    def _notify_state_change(self):
        """Notify state change callbacks."""
        for callback in self.state_change_callbacks:
            try:
                callback(self, self.state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def add_state_change_callback(self, callback: Callable[['CircuitBreaker', CircuitState], None]):
        """Add state change callback."""
        self.state_change_callbacks.append(callback)
    
    def add_failure_callback(self, callback: Callable[['CircuitBreaker', FailureRecord], None]):
        """Add failure callback."""
        self.failure_callbacks.append(callback)
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self.call_lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        recent_failures = [
            f for f in self.failures
            if datetime.now() - f.timestamp <= timedelta(minutes=30)
        ]
        
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'active_calls': self.active_calls,
            'recent_failures': len(recent_failures),
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'max_concurrent_calls': self.config.max_concurrent_calls
            },
            'stats': self.stats.copy()
        }

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class AnalyticsCircuitBreakerManager:
    """
    Manages circuit breakers for all analytics components.
    """
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers = {}
        self.global_stats = {
            'total_breakers': 0,
            'open_breakers': 0,
            'half_open_breakers': 0,
            'closed_breakers': 0,
            'total_failures': 0,
            'start_time': datetime.now()
        }
        
        # Default configurations for different component types
        self.default_configs = {
            'data_source': CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=10.0,
                recovery_timeout=30,
                success_threshold=2,
                max_concurrent_calls=5
            ),
            'processor': CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=30.0,
                recovery_timeout=60,
                success_threshold=3,
                max_concurrent_calls=10
            ),
            'storage': CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=15.0,
                recovery_timeout=45,
                success_threshold=2,
                max_concurrent_calls=8
            ),
            'external_service': CircuitBreakerConfig(
                failure_threshold=2,
                timeout_seconds=20.0,
                recovery_timeout=120,
                success_threshold=3,
                max_concurrent_calls=3
            )
        }
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Analytics Circuit Breaker Manager initialized")
    
    def create_circuit_breaker(self, name: str, component_type: str = 'processor',
                              config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """
        Create a new circuit breaker.
        
        Args:
            name: Circuit breaker name
            component_type: Type of component (data_source, processor, storage, external_service)
            config: Custom configuration
        
        Returns:
            Circuit breaker instance
        """
        if name in self.circuit_breakers:
            return self.circuit_breakers[name]
        
        # Use default config for component type if not provided
        if config is None:
            config = self.default_configs.get(component_type, CircuitBreakerConfig())
        
        circuit_breaker = CircuitBreaker(name, config)
        
        # Add callbacks for monitoring
        circuit_breaker.add_state_change_callback(self._on_state_change)
        circuit_breaker.add_failure_callback(self._on_failure)
        
        self.circuit_breakers[name] = circuit_breaker
        self.global_stats['total_breakers'] += 1
        self.global_stats['closed_breakers'] += 1
        
        logger.info(f"Created circuit breaker '{name}' for {component_type}")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker."""
        return self.circuit_breakers.get(name)
    
    def protect(self, name: str, component_type: str = 'processor',
                config: CircuitBreakerConfig = None):
        """
        Decorator to protect a function with circuit breaker.
        
        Args:
            name: Circuit breaker name
            component_type: Component type
            config: Custom configuration
        """
        def decorator(func):
            circuit_breaker = self.create_circuit_breaker(name, component_type, config)
            return circuit_breaker(func)
        return decorator
    
    def start_monitoring(self):
        """Start circuit breaker monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Circuit breaker monitoring started")
    
    def stop_monitoring(self):
        """Stop circuit breaker monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Circuit breaker monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                if not self.monitoring_active:
                    break
                
                self._update_global_stats()
                self._check_system_health()
                
            except Exception as e:
                logger.error(f"Circuit breaker monitoring error: {e}")
    
    def _update_global_stats(self):
        """Update global statistics."""
        self.global_stats['open_breakers'] = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )
        self.global_stats['half_open_breakers'] = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.HALF_OPEN
        )
        self.global_stats['closed_breakers'] = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.CLOSED
        )
        self.global_stats['total_failures'] = sum(
            cb.stats['failed_calls'] for cb in self.circuit_breakers.values()
        )
    
    def _check_system_health(self):
        """Check overall system health."""
        total_breakers = len(self.circuit_breakers)
        if total_breakers == 0:
            return
        
        open_percentage = (self.global_stats['open_breakers'] / total_breakers) * 100
        
        if open_percentage > 50:
            logger.warning(f"High percentage of circuit breakers are open: {open_percentage:.1f}%")
        elif open_percentage > 25:
            logger.info(f"Some circuit breakers are open: {open_percentage:.1f}%")
    
    def _on_state_change(self, circuit_breaker: CircuitBreaker, new_state: CircuitState):
        """Handle circuit breaker state changes."""
        logger.info(f"Circuit breaker '{circuit_breaker.name}' changed to {new_state.value}")
    
    def _on_failure(self, circuit_breaker: CircuitBreaker, failure: FailureRecord):
        """Handle circuit breaker failures."""
        logger.warning(
            f"Circuit breaker '{circuit_breaker.name}' failure: "
            f"{failure.failure_type.value} - {failure.error}"
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        circuit_statuses = {
            name: cb.get_status() for name, cb in self.circuit_breakers.items()
        }
        
        uptime = (datetime.now() - self.global_stats['start_time']).total_seconds()
        
        return {
            'global_stats': self.global_stats.copy(),
            'circuit_breakers': circuit_statuses,
            'system_health': self._calculate_system_health(),
            'uptime_seconds': uptime,
            'monitoring_active': self.monitoring_active
        }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health."""
        total_breakers = len(self.circuit_breakers)
        if total_breakers == 0:
            return {'status': 'healthy', 'score': 100}
        
        open_breakers = self.global_stats['open_breakers']
        half_open_breakers = self.global_stats['half_open_breakers']
        
        # Calculate health score
        health_score = max(0, 100 - (open_breakers * 30) - (half_open_breakers * 10))
        
        if health_score >= 90:
            status = 'healthy'
        elif health_score >= 70:
            status = 'degraded'
        elif health_score >= 40:
            status = 'unhealthy'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': health_score,
            'open_percentage': (open_breakers / total_breakers) * 100,
            'degraded_components': [
                name for name, cb in self.circuit_breakers.items()
                if cb.state != CircuitState.CLOSED
            ]
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.reset()
        logger.info("All circuit breakers reset")
    
    def shutdown(self):
        """Shutdown circuit breaker manager."""
        self.stop_monitoring()
        logger.info("Analytics Circuit Breaker Manager shutdown")

# Global circuit breaker manager instance
circuit_breaker_manager = AnalyticsCircuitBreakerManager()