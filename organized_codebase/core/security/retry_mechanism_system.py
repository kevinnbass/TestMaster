"""
CrewAI Derived Retry Mechanism System
Extracted from CrewAI retry patterns and exponential backoff implementations
Enhanced for comprehensive retry logic and intelligent backoff strategies
"""

import logging
import time
import random
import asyncio
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from .error_handler import SecurityError, ValidationError, security_error_handler

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategies based on CrewAI patterns"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"


class RetryCondition(Enum):
    """Conditions under which retry should be attempted"""
    ALL_ERRORS = "all_errors"
    SPECIFIC_ERRORS = "specific_errors"
    TRANSIENT_ERRORS = "transient_errors"
    NETWORK_ERRORS = "network_errors"
    TIMEOUT_ERRORS = "timeout_errors"
    RATE_LIMIT_ERRORS = "rate_limit_errors"


class RetryStatus(Enum):
    """Retry execution status"""
    SUCCESS = "success"
    FAILED_PERMANENTLY = "failed_permanently"
    FAILED_MAX_ATTEMPTS = "failed_max_attempts"
    FAILED_TIMEOUT = "failed_timeout"
    SKIPPED = "skipped"


@dataclass
class RetryAttempt:
    """Individual retry attempt information"""
    attempt_number: int
    timestamp: datetime
    delay_ms: float
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    success: bool = False
    execution_time_ms: float = 0.0
    
    @property
    def failed(self) -> bool:
        """Check if attempt failed"""
        return not self.success


@dataclass
class RetryConfig:
    """Retry configuration based on CrewAI patterns"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1
    timeout_seconds: Optional[float] = None
    retry_condition: RetryCondition = RetryCondition.TRANSIENT_ERRORS
    retryable_exceptions: List[str] = field(default_factory=lambda: [
        'ConnectionError', 'TimeoutError', 'TemporaryFailure',
        'RateLimitError', 'ServiceUnavailable', 'NetworkError'
    ])
    
    def __post_init__(self):
        if self.max_attempts <= 0:
            raise ValidationError("max_attempts must be positive")
        if self.initial_delay < 0:
            raise ValidationError("initial_delay must be non-negative")
        if self.backoff_multiplier <= 0:
            raise ValidationError("backoff_multiplier must be positive")


@dataclass
class RetryResult:
    """Comprehensive retry execution result"""
    success: bool
    status: RetryStatus
    final_result: Any
    total_attempts: int
    total_execution_time_ms: float
    strategy_used: RetryStrategy
    attempts: List[RetryAttempt] = field(default_factory=list)
    config: Optional[RetryConfig] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def failed(self) -> bool:
        """Check if retry failed"""
        return not self.success
    
    @property
    def successful_attempt(self) -> Optional[RetryAttempt]:
        """Get the successful attempt if any"""
        for attempt in self.attempts:
            if attempt.success:
                return attempt
        return None
    
    @property
    def average_delay_ms(self) -> float:
        """Calculate average delay between attempts"""
        if len(self.attempts) <= 1:
            return 0.0
        return sum(attempt.delay_ms for attempt in self.attempts[1:]) / (len(self.attempts) - 1)


class BaseRetryMechanism(ABC):
    """Base class for retry mechanisms"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def calculate_delay(self, attempt_number: int, previous_delays: List[float]) -> float:
        """Calculate delay for the given attempt number"""
        pass
    
    def should_retry(self, error: Exception, attempt_number: int) -> bool:
        """Determine if retry should be attempted"""
        if attempt_number >= self.config.max_attempts:
            return False
        
        error_type = type(error).__name__
        
        if self.config.retry_condition == RetryCondition.ALL_ERRORS:
            return True
        elif self.config.retry_condition == RetryCondition.SPECIFIC_ERRORS:
            return error_type in self.config.retryable_exceptions
        elif self.config.retry_condition == RetryCondition.TRANSIENT_ERRORS:
            transient_errors = [
                'ConnectionError', 'TimeoutError', 'TemporaryFailure',
                'ServiceUnavailable', 'NetworkError'
            ]
            return any(transient in error_type for transient in transient_errors)
        elif self.config.retry_condition == RetryCondition.NETWORK_ERRORS:
            network_errors = ['ConnectionError', 'NetworkError', 'DNSError']
            return any(network in error_type for network in network_errors)
        elif self.config.retry_condition == RetryCondition.TIMEOUT_ERRORS:
            return 'Timeout' in error_type
        elif self.config.retry_condition == RetryCondition.RATE_LIMIT_ERRORS:
            return 'RateLimit' in error_type
        
        return False


class FixedDelayRetry(BaseRetryMechanism):
    """Fixed delay retry mechanism"""
    
    def calculate_delay(self, attempt_number: int, previous_delays: List[float]) -> float:
        """Calculate fixed delay"""
        return min(self.config.initial_delay, self.config.max_delay)


class ExponentialBackoffRetry(BaseRetryMechanism):
    """Exponential backoff retry mechanism based on CrewAI patterns"""
    
    def calculate_delay(self, attempt_number: int, previous_delays: List[float]) -> float:
        """Calculate exponential backoff delay"""
        delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt_number - 1))
        return min(delay, self.config.max_delay)


class LinearBackoffRetry(BaseRetryMechanism):
    """Linear backoff retry mechanism"""
    
    def calculate_delay(self, attempt_number: int, previous_delays: List[float]) -> float:
        """Calculate linear backoff delay"""
        delay = self.config.initial_delay * attempt_number
        return min(delay, self.config.max_delay)


class JitteredBackoffRetry(BaseRetryMechanism):
    """Jittered exponential backoff to avoid thundering herd"""
    
    def calculate_delay(self, attempt_number: int, previous_delays: List[float]) -> float:
        """Calculate jittered exponential backoff delay"""
        base_delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt_number - 1))
        base_delay = min(base_delay, self.config.max_delay)
        
        # Apply jitter
        jitter_range = base_delay * self.config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        
        return max(0, base_delay + jitter)


class FibonacciBackoffRetry(BaseRetryMechanism):
    """Fibonacci sequence backoff retry mechanism"""
    
    def __init__(self, config: RetryConfig):
        super().__init__(config)
        self._fibonacci_cache = [1, 1]  # Start with F(1)=1, F(2)=1
    
    def calculate_delay(self, attempt_number: int, previous_delays: List[float]) -> float:
        """Calculate Fibonacci backoff delay"""
        # Generate Fibonacci number for attempt
        while len(self._fibonacci_cache) < attempt_number:
            next_fib = self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            self._fibonacci_cache.append(next_fib)
        
        fib_multiplier = self._fibonacci_cache[attempt_number - 1]
        delay = self.config.initial_delay * fib_multiplier
        
        return min(delay, self.config.max_delay)


class AdaptiveRetry(BaseRetryMechanism):
    """Adaptive retry mechanism that learns from success/failure patterns"""
    
    def __init__(self, config: RetryConfig):
        super().__init__(config)
        self.success_history: List[int] = []  # Track which attempt numbers succeed
        self.failure_patterns: Dict[str, List[int]] = {}  # Track failure patterns by error type
    
    def calculate_delay(self, attempt_number: int, previous_delays: List[float]) -> float:
        """Calculate adaptive delay based on historical success patterns"""
        # Start with exponential backoff as base
        base_delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt_number - 1))
        
        # Analyze success history
        if self.success_history:
            avg_success_attempt = sum(self.success_history) / len(self.success_history)
            
            # If current attempt is close to average success attempt, reduce delay
            if abs(attempt_number - avg_success_attempt) <= 1:
                base_delay *= 0.7  # Reduce delay by 30%
            # If we're beyond typical success attempts, increase delay more aggressively
            elif attempt_number > avg_success_attempt + 1:
                base_delay *= 1.5  # Increase delay by 50%
        
        return min(base_delay, self.config.max_delay)
    
    def record_success(self, attempt_number: int):
        """Record successful attempt for adaptive learning"""
        self.success_history.append(attempt_number)
        # Keep only recent history
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-50:]
    
    def record_failure_pattern(self, error_type: str, attempt_number: int):
        """Record failure pattern for adaptive learning"""
        if error_type not in self.failure_patterns:
            self.failure_patterns[error_type] = []
        
        self.failure_patterns[error_type].append(attempt_number)
        # Keep only recent patterns
        if len(self.failure_patterns[error_type]) > 50:
            self.failure_patterns[error_type] = self.failure_patterns[error_type][-25:]


class RetryMechanismSystem:
    """Comprehensive retry mechanism system based on CrewAI patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mechanism_registry: Dict[RetryStrategy, type] = {
            RetryStrategy.FIXED_DELAY: FixedDelayRetry,
            RetryStrategy.EXPONENTIAL_BACKOFF: ExponentialBackoffRetry,
            RetryStrategy.LINEAR_BACKOFF: LinearBackoffRetry,
            RetryStrategy.JITTERED_BACKOFF: JitteredBackoffRetry,
            RetryStrategy.FIBONACCI_BACKOFF: FibonacciBackoffRetry,
            RetryStrategy.ADAPTIVE: AdaptiveRetry
        }
        self.retry_history: List[RetryResult] = []
        self.max_history = 10000
    
    def execute_with_retry(self, func: Callable[[], T], 
                          config: RetryConfig = None,
                          function_name: str = None,
                          context: Dict[str, Any] = None) -> RetryResult:
        """Execute function with retry logic"""
        config = config or RetryConfig()
        function_name = function_name or func.__name__
        context = context or {}
        
        # Create retry mechanism
        mechanism_class = self.mechanism_registry[config.strategy]
        mechanism = mechanism_class(config)
        
        # Track retry execution
        attempts = []
        start_time = time.time()
        last_error = None
        
        for attempt_num in range(1, config.max_attempts + 1):
            attempt_start = time.time()
            
            try:
                # Check timeout
                if config.timeout_seconds:
                    elapsed = time.time() - start_time
                    if elapsed > config.timeout_seconds:
                        result = RetryResult(
                            success=False,
                            status=RetryStatus.FAILED_TIMEOUT,
                            final_result=None,
                            total_attempts=attempt_num - 1,
                            total_execution_time_ms=(time.time() - start_time) * 1000,
                            strategy_used=config.strategy,
                            attempts=attempts,
                            config=config,
                            context=context
                        )
                        self._add_to_history(result)
                        return result
                
                # Calculate delay (except for first attempt)
                delay_ms = 0.0
                if attempt_num > 1:
                    previous_delays = [a.delay_ms for a in attempts[:-1]]  # Exclude current attempt
                    delay_seconds = mechanism.calculate_delay(attempt_num, previous_delays)
                    delay_ms = delay_seconds * 1000
                    
                    self.logger.info(f"Retry attempt {attempt_num} for {function_name} in {delay_seconds:.3f}s")
                    time.sleep(delay_seconds)
                
                # Execute function
                result = func()
                
                execution_time = (time.time() - attempt_start) * 1000
                
                # Record successful attempt
                successful_attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.utcnow(),
                    delay_ms=delay_ms,
                    success=True,
                    execution_time_ms=execution_time
                )
                attempts.append(successful_attempt)
                
                # Update adaptive mechanism if applicable
                if isinstance(mechanism, AdaptiveRetry):
                    mechanism.record_success(attempt_num)
                
                # Create successful result
                retry_result = RetryResult(
                    success=True,
                    status=RetryStatus.SUCCESS,
                    final_result=result,
                    total_attempts=attempt_num,
                    total_execution_time_ms=(time.time() - start_time) * 1000,
                    strategy_used=config.strategy,
                    attempts=attempts,
                    config=config,
                    context=context
                )
                
                self._add_to_history(retry_result)
                self.logger.info(f"Function {function_name} succeeded on attempt {attempt_num}")
                
                return retry_result
                
            except Exception as e:
                last_error = e
                execution_time = (time.time() - attempt_start) * 1000
                
                # Record failed attempt
                failed_attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.utcnow(),
                    delay_ms=delay_ms,
                    error=e,
                    error_message=str(e),
                    success=False,
                    execution_time_ms=execution_time
                )
                attempts.append(failed_attempt)
                
                # Update adaptive mechanism if applicable
                if isinstance(mechanism, AdaptiveRetry):
                    mechanism.record_failure_pattern(type(e).__name__, attempt_num)
                
                self.logger.warning(f"Attempt {attempt_num} failed for {function_name}: {str(e)}")
                
                # Check if we should retry
                if not mechanism.should_retry(e, attempt_num):
                    self.logger.info(f"Not retrying {function_name} after {attempt_num} attempts")
                    break
        
        # All attempts failed
        retry_result = RetryResult(
            success=False,
            status=RetryStatus.FAILED_MAX_ATTEMPTS,
            final_result=None,
            total_attempts=len(attempts),
            total_execution_time_ms=(time.time() - start_time) * 1000,
            strategy_used=config.strategy,
            attempts=attempts,
            config=config,
            context=context
        )
        
        self._add_to_history(retry_result)
        self.logger.error(f"Function {function_name} failed after {len(attempts)} attempts")
        
        return retry_result
    
    async def execute_with_retry_async(self, func: Callable[[], T],
                                     config: RetryConfig = None,
                                     function_name: str = None,
                                     context: Dict[str, Any] = None) -> RetryResult:
        """Execute async function with retry logic"""
        config = config or RetryConfig()
        function_name = function_name or func.__name__
        context = context or {}
        
        # Create retry mechanism
        mechanism_class = self.mechanism_registry[config.strategy]
        mechanism = mechanism_class(config)
        
        # Track retry execution
        attempts = []
        start_time = time.time()
        last_error = None
        
        for attempt_num in range(1, config.max_attempts + 1):
            attempt_start = time.time()
            
            try:
                # Check timeout
                if config.timeout_seconds:
                    elapsed = time.time() - start_time
                    if elapsed > config.timeout_seconds:
                        result = RetryResult(
                            success=False,
                            status=RetryStatus.FAILED_TIMEOUT,
                            final_result=None,
                            total_attempts=attempt_num - 1,
                            total_execution_time_ms=(time.time() - start_time) * 1000,
                            strategy_used=config.strategy,
                            attempts=attempts,
                            config=config,
                            context=context
                        )
                        self._add_to_history(result)
                        return result
                
                # Calculate delay (except for first attempt)
                delay_ms = 0.0
                if attempt_num > 1:
                    previous_delays = [a.delay_ms for a in attempts[:-1]]
                    delay_seconds = mechanism.calculate_delay(attempt_num, previous_delays)
                    delay_ms = delay_seconds * 1000
                    
                    self.logger.info(f"Async retry attempt {attempt_num} for {function_name} in {delay_seconds:.3f}s")
                    await asyncio.sleep(delay_seconds)
                
                # Execute async function
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()
                
                execution_time = (time.time() - attempt_start) * 1000
                
                # Record successful attempt
                successful_attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.utcnow(),
                    delay_ms=delay_ms,
                    success=True,
                    execution_time_ms=execution_time
                )
                attempts.append(successful_attempt)
                
                # Update adaptive mechanism
                if isinstance(mechanism, AdaptiveRetry):
                    mechanism.record_success(attempt_num)
                
                # Create successful result
                retry_result = RetryResult(
                    success=True,
                    status=RetryStatus.SUCCESS,
                    final_result=result,
                    total_attempts=attempt_num,
                    total_execution_time_ms=(time.time() - start_time) * 1000,
                    strategy_used=config.strategy,
                    attempts=attempts,
                    config=config,
                    context=context
                )
                
                self._add_to_history(retry_result)
                return retry_result
                
            except Exception as e:
                last_error = e
                execution_time = (time.time() - attempt_start) * 1000
                
                # Record failed attempt
                failed_attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.utcnow(),
                    delay_ms=delay_ms,
                    error=e,
                    error_message=str(e),
                    success=False,
                    execution_time_ms=execution_time
                )
                attempts.append(failed_attempt)
                
                # Update adaptive mechanism
                if isinstance(mechanism, AdaptiveRetry):
                    mechanism.record_failure_pattern(type(e).__name__, attempt_num)
                
                # Check if we should retry
                if not mechanism.should_retry(e, attempt_num):
                    break
        
        # All attempts failed
        retry_result = RetryResult(
            success=False,
            status=RetryStatus.FAILED_MAX_ATTEMPTS,
            final_result=None,
            total_attempts=len(attempts),
            total_execution_time_ms=(time.time() - start_time) * 1000,
            strategy_used=config.strategy,
            attempts=attempts,
            config=config,
            context=context
        )
        
        self._add_to_history(retry_result)
        return retry_result
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retry statistics"""
        try:
            if not self.retry_history:
                return {'total_retries': 0}
            
            total_retries = len(self.retry_history)
            successful = sum(1 for r in self.retry_history if r.success)
            
            # Strategy distribution
            strategy_counts = {}
            strategy_success_rates = {}
            
            for result in self.retry_history:
                strategy = result.strategy_used.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                
                if strategy not in strategy_success_rates:
                    strategy_success_rates[strategy] = {'total': 0, 'successful': 0}
                
                strategy_success_rates[strategy]['total'] += 1
                if result.success:
                    strategy_success_rates[strategy]['successful'] += 1
            
            # Calculate success rates
            for strategy in strategy_success_rates:
                total = strategy_success_rates[strategy]['total']
                successful_count = strategy_success_rates[strategy]['successful']
                strategy_success_rates[strategy]['success_rate_pct'] = (successful_count / total) * 100
            
            # Attempt distribution
            attempt_distribution = {}
            for result in self.retry_history:
                attempts = result.total_attempts
                attempt_distribution[attempts] = attempt_distribution.get(attempts, 0) + 1
            
            # Average execution times
            avg_exec_time = sum(r.total_execution_time_ms for r in self.retry_history) / total_retries
            
            return {
                'total_retries': total_retries,
                'successful_retries': successful,
                'failed_retries': total_retries - successful,
                'overall_success_rate_pct': (successful / total_retries) * 100,
                'average_execution_time_ms': avg_exec_time,
                'strategy_distribution': strategy_counts,
                'strategy_success_rates': {k: v['success_rate_pct'] for k, v in strategy_success_rates.items()},
                'attempt_distribution': attempt_distribution,
                'available_strategies': [strategy.value for strategy in RetryStrategy]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating retry statistics: {e}")
            return {'error': str(e)}
    
    def _add_to_history(self, result: RetryResult):
        """Add result to retry history with limit"""
        self.retry_history.append(result)
        
        # Limit history size
        if len(self.retry_history) > self.max_history:
            self.retry_history = self.retry_history[-self.max_history // 2:]


# Global retry mechanism system
retry_mechanism_system = RetryMechanismSystem()


# Convenience decorator for automatic retry
def with_retry(config: RetryConfig = None, function_name: str = None):
    """Decorator for automatic retry"""
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        def wrapper(*args, **kwargs):
            bound_func = lambda: func(*args, **kwargs)
            result = retry_mechanism_system.execute_with_retry(
                bound_func, config, function_name or func.__name__
            )
            
            if result.success:
                return result.final_result
            else:
                # Re-raise last error if retry failed
                if result.attempts and result.attempts[-1].error:
                    raise result.attempts[-1].error
                else:
                    raise RuntimeError(f"Retry failed for {func.__name__}")
        
        return wrapper
    return decorator


# Convenience function for quick retry
def retry_on_error(func: Callable[[], T], max_attempts: int = 3, 
                  strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF) -> T:
    """Quick retry function with default configuration"""
    config = RetryConfig(
        strategy=strategy,
        max_attempts=max_attempts
    )
    
    result = retry_mechanism_system.execute_with_retry(func, config)
    
    if result.success:
        return result.final_result
    else:
        if result.attempts and result.attempts[-1].error:
            raise result.attempts[-1].error
        else:
            raise RuntimeError("Retry failed")