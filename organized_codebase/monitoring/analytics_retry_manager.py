"""
Analytics Retry Manager with Exponential Backoff
================================================

Provides intelligent retry mechanisms for failed analytics operations
with exponential backoff, circuit breaking, and adaptive strategies.

Author: TestMaster Team
"""

import logging
import time
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"

@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

@dataclass
class RetryAttempt:
    """Represents a retry attempt."""
    attempt_number: int
    timestamp: datetime
    delay_seconds: float
    success: bool
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for retry management."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 2):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Time before attempting recovery
            success_threshold: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def call(self, func: Callable[[], T]) -> T:
        """Call function through circuit breaker."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                # Check if we should try recovery
                if self.last_failure_time:
                    time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                    if time_since_failure >= self.recovery_timeout:
                        self.state = CircuitState.HALF_OPEN
                        logger.info("Circuit breaker entering HALF_OPEN state")
                    else:
                        raise Exception(f"Circuit breaker is OPEN (will retry in {self.recovery_timeout - time_since_failure:.1f}s)")
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logger.info("Circuit breaker CLOSED after recovery")
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning("Circuit breaker reopened after failed recovery")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def reset(self):
        """Reset circuit breaker."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

class AnalyticsRetryManager:
    """
    Manages retry logic for analytics operations.
    """
    
    def __init__(self):
        """Initialize retry manager."""
        self.retry_configs = {}
        self.retry_history = defaultdict(lambda: deque(maxlen=100))
        self.circuit_breakers = {}
        
        # Statistics
        self.retry_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_retries': 0,
            'operations_with_retries': 0
        }
        
        # Adaptive learning
        self.adaptive_configs = {}
        self.learning_enabled = True
        
        logger.info("Analytics Retry Manager initialized")
    
    def configure_operation(self, 
                           operation_name: str,
                           config: Optional[RetryConfig] = None,
                           circuit_breaker: bool = True):
        """
        Configure retry settings for an operation.
        
        Args:
            operation_name: Name of the operation
            config: Retry configuration
            circuit_breaker: Enable circuit breaker
        """
        if config is None:
            config = RetryConfig()
        
        self.retry_configs[operation_name] = config
        
        if circuit_breaker:
            self.circuit_breakers[operation_name] = CircuitBreaker()
        
        logger.info(f"Configured retry for operation: {operation_name}")
    
    def retry_operation(self,
                       operation_name: str,
                       func: Callable[[], T],
                       *args,
                       **kwargs) -> T:
        """
        Execute operation with retry logic.
        
        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        # Get or create configuration
        if operation_name not in self.retry_configs:
            self.configure_operation(operation_name)
        
        config = self.retry_configs[operation_name]
        
        # Apply adaptive configuration if available
        if operation_name in self.adaptive_configs:
            config = self.adaptive_configs[operation_name]
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(operation_name)
        
        # Track operation
        self.retry_stats['total_operations'] += 1
        attempts = []
        last_exception = None
        
        for attempt_num in range(config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Execute through circuit breaker if available
                if circuit_breaker:
                    result = circuit_breaker.call(lambda: func(*args, **kwargs))
                else:
                    result = func(*args, **kwargs)
                
                response_time = (time.time() - start_time) * 1000
                
                # Record successful attempt
                attempts.append(RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.now(),
                    delay_seconds=0 if attempt_num == 0 else delay,
                    success=True,
                    response_time_ms=response_time
                ))
                
                # Update statistics
                self.retry_stats['successful_operations'] += 1
                if attempt_num > 0:
                    self.retry_stats['operations_with_retries'] += 1
                
                # Learn from success
                if self.learning_enabled:
                    self._learn_from_attempts(operation_name, attempts)
                
                # Store history
                self.retry_history[operation_name].append(attempts)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Record failed attempt
                attempts.append(RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.now(),
                    delay_seconds=0 if attempt_num == 0 else delay if 'delay' in locals() else 0,
                    success=False,
                    error_message=str(e)
                ))
                
                # Check if we should retry
                if attempt_num >= config.max_retries:
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt_num, config)
                
                logger.warning(
                    f"Retry {attempt_num + 1}/{config.max_retries} for {operation_name} "
                    f"after {delay:.2f}s delay. Error: {e}"
                )
                
                # Wait before retry
                time.sleep(delay)
                
                # Update statistics
                self.retry_stats['total_retries'] += 1
        
        # All retries failed
        self.retry_stats['failed_operations'] += 1
        
        # Learn from failure
        if self.learning_enabled:
            self._learn_from_attempts(operation_name, attempts)
        
        # Store history
        self.retry_history[operation_name].append(attempts)
        
        raise Exception(
            f"Operation {operation_name} failed after {config.max_retries} retries: {last_exception}"
        )
    
    def _calculate_delay(self, attempt_num: int, config: RetryConfig) -> float:
        """Calculate retry delay based on strategy."""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(
                config.initial_delay * (config.exponential_base ** attempt_num),
                config.max_delay
            )
        
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(
                config.initial_delay * (attempt_num + 1),
                config.max_delay
            )
        
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = min(
                self._fibonacci(attempt_num + 1) * config.initial_delay,
                config.max_delay
            )
        
        elif config.strategy == RetryStrategy.ADAPTIVE:
            # Use learned optimal delay
            delay = self._get_adaptive_delay(attempt_num, config)
        
        else:
            delay = config.initial_delay
        
        # Add jitter if enabled
        if config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _get_adaptive_delay(self, attempt_num: int, config: RetryConfig) -> float:
        """Get adaptive delay based on learned patterns."""
        # Start with exponential backoff
        base_delay = config.initial_delay * (2 ** attempt_num)
        
        # Adjust based on success patterns
        # This is simplified - in production, use ML models
        history = self.retry_history.get(config.strategy.value, [])
        if history:
            successful_delays = []
            for attempts in history[-10:]:  # Last 10 operations
                for i, attempt in enumerate(attempts):
                    if attempt.success and i > 0:
                        successful_delays.append(attempts[i-1].delay_seconds)
            
            if successful_delays:
                # Use average successful delay
                base_delay = sum(successful_delays) / len(successful_delays)
        
        return min(base_delay, config.max_delay)
    
    def _learn_from_attempts(self, operation_name: str, attempts: list):
        """Learn from retry attempts to optimize future retries."""
        if not attempts:
            return
        
        # Calculate metrics
        total_attempts = len(attempts)
        successful = attempts[-1].success
        total_time = sum(a.delay_seconds for a in attempts)
        
        # Simple adaptive learning
        current_config = self.retry_configs.get(operation_name)
        if not current_config:
            return
        
        # Create adaptive config if needed
        if operation_name not in self.adaptive_configs:
            self.adaptive_configs[operation_name] = RetryConfig(
                max_retries=current_config.max_retries,
                initial_delay=current_config.initial_delay,
                max_delay=current_config.max_delay,
                strategy=RetryStrategy.ADAPTIVE
            )
        
        adaptive = self.adaptive_configs[operation_name]
        
        # Adjust based on patterns
        if successful:
            if total_attempts == 1:
                # Success on first try - reduce initial delay
                adaptive.initial_delay = max(0.5, adaptive.initial_delay * 0.9)
            elif total_attempts > current_config.max_retries * 0.7:
                # Success but took many retries - increase delays
                adaptive.initial_delay = min(5.0, adaptive.initial_delay * 1.1)
        else:
            # Failure - consider increasing retries or delays
            adaptive.max_retries = min(10, adaptive.max_retries + 1)
            adaptive.initial_delay = min(5.0, adaptive.initial_delay * 1.2)
    
    def get_retry_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics."""
        if operation_name:
            history = self.retry_history.get(operation_name, [])
            
            if not history:
                return {'message': 'No retry history for operation'}
            
            total_operations = len(history)
            successful = sum(1 for attempts in history if attempts[-1].success)
            total_retries = sum(len(attempts) - 1 for attempts in history)
            
            # Calculate average response time for successful operations
            response_times = []
            for attempts in history:
                for attempt in attempts:
                    if attempt.success and attempt.response_time_ms:
                        response_times.append(attempt.response_time_ms)
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                'operation': operation_name,
                'total_operations': total_operations,
                'successful_operations': successful,
                'success_rate': (successful / total_operations * 100) if total_operations > 0 else 0,
                'total_retries': total_retries,
                'average_retries': total_retries / total_operations if total_operations > 0 else 0,
                'average_response_time_ms': avg_response_time,
                'circuit_breaker_state': self.circuit_breakers[operation_name].state.value 
                                        if operation_name in self.circuit_breakers else None
            }
        else:
            # Global statistics
            return {
                'global_stats': self.retry_stats,
                'operations_configured': list(self.retry_configs.keys()),
                'circuit_breakers': {
                    name: cb.state.value 
                    for name, cb in self.circuit_breakers.items()
                }
            }
    
    def reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker for an operation."""
        if operation_name in self.circuit_breakers:
            self.circuit_breakers[operation_name].reset()
            logger.info(f"Reset circuit breaker for {operation_name}")
    
    def decorator(self, operation_name: str, config: Optional[RetryConfig] = None):
        """
        Decorator for adding retry logic to functions.
        
        Usage:
            @retry_manager.decorator("my_operation")
            def my_function():
                # function code
        """
        def decorator_wrapper(func):
            # Configure operation if not already configured
            if operation_name not in self.retry_configs:
                self.configure_operation(operation_name, config)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.retry_operation(operation_name, func, *args, **kwargs)
            
            return wrapper
        
        return decorator_wrapper
    
    def shutdown(self):
        """Shutdown retry manager."""
        logger.info(f"Retry Manager shutdown - Stats: {self.retry_stats}")

# Global retry manager instance
retry_manager = AnalyticsRetryManager()

# Convenience decorator
def with_retry(operation_name: str, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Convenience decorator for retry logic.
    
    Usage:
        @with_retry("analytics_fetch", max_retries=5)
        def fetch_analytics():
            # function code
    """
    config = RetryConfig(max_retries=max_retries, initial_delay=initial_delay)
    return retry_manager.decorator(operation_name, config)