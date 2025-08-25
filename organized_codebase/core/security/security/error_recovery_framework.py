"""
CrewAI Derived Error Recovery Framework
Extracted from CrewAI error handling patterns and resilience mechanisms
Enhanced for comprehensive error recovery and graceful degradation
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from .error_handler import SecurityError, ValidationError, security_error_handler

T = TypeVar('T')


class RecoveryStrategy(Enum):
    """Error recovery strategies based on CrewAI patterns"""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    ESCALATION = "escalation"
    IGNORE = "ignore"


class ErrorSeverity(Enum):
    """Error severity levels for recovery decision making"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    FATAL = "fatal"


class RecoveryStatus(Enum):
    """Recovery execution status"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ErrorContext:
    """Error context information for recovery decisions"""
    error: Exception
    error_type: str
    error_message: str
    timestamp: datetime
    stack_trace: str
    function_name: str
    attempt_count: int = 1
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Get error age in seconds"""
        return (datetime.utcnow() - self.timestamp).total_seconds()
    
    @classmethod
    def from_exception(cls, error: Exception, function_name: str = None,
                      context_data: Dict[str, Any] = None) -> 'ErrorContext':
        """Create ErrorContext from exception"""
        return cls(
            error=error,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.utcnow(),
            stack_trace=traceback.format_exc(),
            function_name=function_name or 'unknown',
            context_data=context_data or {}
        )


@dataclass
class RecoveryResult:
    """Recovery execution result"""
    success: bool
    strategy: RecoveryStrategy
    status: RecoveryStatus
    message: str
    execution_time_ms: float
    original_error: ErrorContext
    recovery_value: Any = None
    fallback_used: bool = False
    attempts_made: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def failed(self) -> bool:
        """Check if recovery failed"""
        return not self.success


class BaseRecoveryHandler(ABC):
    """Base class for error recovery handlers"""
    
    def __init__(self, name: str, strategy: RecoveryStrategy,
                 max_attempts: int = 3, timeout_seconds: float = 30.0):
        self.name = name
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this handler can handle the error"""
        pass
    
    @abstractmethod
    def recover(self, error_context: ErrorContext, 
                original_function: Callable[[], T],
                *args, **kwargs) -> RecoveryResult:
        """Execute recovery strategy"""
        pass
    
    def create_result(self, success: bool, status: RecoveryStatus,
                     message: str, execution_time_ms: float,
                     error_context: ErrorContext,
                     recovery_value: Any = None,
                     attempts_made: int = 1) -> RecoveryResult:
        """Create standardized recovery result"""
        return RecoveryResult(
            success=success,
            strategy=self.strategy,
            status=status,
            message=message,
            execution_time_ms=execution_time_ms,
            original_error=error_context,
            recovery_value=recovery_value,
            attempts_made=attempts_made
        )


class RetryRecoveryHandler(BaseRecoveryHandler):
    """Retry recovery handler with exponential backoff"""
    
    def __init__(self, retry_delay: float = 1.0, backoff_multiplier: float = 2.0,
                 max_delay: float = 60.0, **kwargs):
        super().__init__(name="retry_handler", strategy=RecoveryStrategy.RETRY, **kwargs)
        self.retry_delay = retry_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if error is retryable"""
        retryable_errors = [
            'ConnectionError', 'TimeoutError', 'TemporaryFailure',
            'RateLimitError', 'ServiceUnavailable', 'NetworkError'
        ]
        
        return any(error_type in error_context.error_type 
                  for error_type in retryable_errors)
    
    def recover(self, error_context: ErrorContext,
                original_function: Callable[[], T],
                *args, **kwargs) -> RecoveryResult:
        """Execute retry recovery with exponential backoff"""
        start_time = time.time()
        last_error = error_context.error
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Calculate delay with exponential backoff
                if attempt > 1:
                    delay = min(
                        self.retry_delay * (self.backoff_multiplier ** (attempt - 2)),
                        self.max_delay
                    )
                    self.logger.info(f"Retrying in {delay:.1f}s (attempt {attempt})")
                    time.sleep(delay)
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.timeout_seconds:
                    execution_time = elapsed * 1000
                    return self.create_result(
                        False, RecoveryStatus.TIMEOUT,
                        f"Recovery timeout after {elapsed:.1f}s",
                        execution_time, error_context, attempts_made=attempt
                    )
                
                # Attempt execution
                result = original_function(*args, **kwargs)
                
                execution_time = (time.time() - start_time) * 1000
                self.logger.info(f"Recovery successful after {attempt} attempts")
                
                return self.create_result(
                    True, RecoveryStatus.SUCCESS,
                    f"Recovered after {attempt} attempts",
                    execution_time, error_context,
                    recovery_value=result, attempts_made=attempt
                )
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Retry attempt {attempt} failed: {str(e)}")
                
                # Update error context
                error_context.attempt_count = attempt
                error_context.error = e
                error_context.error_message = str(e)
        
        # All retries failed
        execution_time = (time.time() - start_time) * 1000
        return self.create_result(
            False, RecoveryStatus.FAILED,
            f"All {self.max_attempts} retry attempts failed",
            execution_time, error_context, attempts_made=self.max_attempts
        )


class FallbackRecoveryHandler(BaseRecoveryHandler):
    """Fallback recovery handler with alternative implementations"""
    
    def __init__(self, fallback_functions: Dict[str, Callable] = None, **kwargs):
        super().__init__(name="fallback_handler", strategy=RecoveryStrategy.FALLBACK, **kwargs)
        self.fallback_functions = fallback_functions or {}
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if fallback is available for this function"""
        return error_context.function_name in self.fallback_functions
    
    def recover(self, error_context: ErrorContext,
                original_function: Callable[[], T],
                *args, **kwargs) -> RecoveryResult:
        """Execute fallback recovery"""
        start_time = time.time()
        
        try:
            fallback_func = self.fallback_functions.get(error_context.function_name)
            
            if not fallback_func:
                execution_time = (time.time() - start_time) * 1000
                return self.create_result(
                    False, RecoveryStatus.FAILED,
                    f"No fallback available for {error_context.function_name}",
                    execution_time, error_context
                )
            
            # Execute fallback function
            result = fallback_func(*args, **kwargs)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(f"Fallback recovery successful for {error_context.function_name}")
            
            recovery_result = self.create_result(
                True, RecoveryStatus.SUCCESS,
                f"Fallback executed successfully",
                execution_time, error_context,
                recovery_value=result
            )
            recovery_result.fallback_used = True
            
            return recovery_result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Fallback recovery failed: {str(e)}")
            
            return self.create_result(
                False, RecoveryStatus.FAILED,
                f"Fallback execution failed: {str(e)}",
                execution_time, error_context
            )
    
    def register_fallback(self, function_name: str, fallback_function: Callable):
        """Register a fallback function"""
        self.fallback_functions[function_name] = fallback_function
        self.logger.info(f"Registered fallback for {function_name}")


class GracefulDegradationHandler(BaseRecoveryHandler):
    """Graceful degradation handler for reduced functionality"""
    
    def __init__(self, degraded_responses: Dict[str, Any] = None, **kwargs):
        super().__init__(
            name="degradation_handler", 
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION, 
            **kwargs
        )
        self.degraded_responses = degraded_responses or {}
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if degraded response is available"""
        return (error_context.function_name in self.degraded_responses or
                'default' in self.degraded_responses)
    
    def recover(self, error_context: ErrorContext,
                original_function: Callable[[], T],
                *args, **kwargs) -> RecoveryResult:
        """Execute graceful degradation"""
        start_time = time.time()
        
        try:
            # Get degraded response
            degraded_value = self.degraded_responses.get(
                error_context.function_name,
                self.degraded_responses.get('default')
            )
            
            if degraded_value is None:
                execution_time = (time.time() - start_time) * 1000
                return self.create_result(
                    False, RecoveryStatus.FAILED,
                    f"No degraded response for {error_context.function_name}",
                    execution_time, error_context
                )
            
            # Return degraded response
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(f"Graceful degradation applied for {error_context.function_name}")
            
            return self.create_result(
                True, RecoveryStatus.PARTIAL,
                f"Graceful degradation applied",
                execution_time, error_context,
                recovery_value=degraded_value
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Graceful degradation failed: {str(e)}")
            
            return self.create_result(
                False, RecoveryStatus.FAILED,
                f"Degradation failed: {str(e)}",
                execution_time, error_context
            )
    
    def register_degraded_response(self, function_name: str, degraded_value: Any):
        """Register a degraded response for a function"""
        self.degraded_responses[function_name] = degraded_value
        self.logger.info(f"Registered degraded response for {function_name}")


class ErrorRecoveryFramework:
    """Comprehensive error recovery framework based on CrewAI patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.handlers: List[BaseRecoveryHandler] = []
        self.recovery_history: List[RecoveryResult] = []
        self.max_history = 10000
        self.error_patterns: Dict[str, ErrorSeverity] = {}
        
        # Register default handlers
        self._register_default_handlers()
        self._initialize_error_patterns()
    
    def register_handler(self, handler: BaseRecoveryHandler):
        """Register a recovery handler"""
        try:
            self.handlers.append(handler)
            self.logger.info(f"Registered recovery handler: {handler.name}")
            
        except Exception as e:
            error = SecurityError(f"Failed to register recovery handler: {str(e)}", "RECOVERY_REG_001")
            security_error_handler.handle_error(error)
    
    def recover_from_error(self, error: Exception, 
                          original_function: Callable[[], T],
                          function_name: str = None,
                          context_data: Dict[str, Any] = None,
                          *args, **kwargs) -> RecoveryResult:
        """Execute error recovery using appropriate strategy"""
        
        error_context = ErrorContext.from_exception(
            error, function_name or original_function.__name__, context_data
        )
        
        try:
            # Find suitable recovery handler
            handler = self._select_recovery_handler(error_context)
            
            if not handler:
                # No handler available - create failure result
                result = RecoveryResult(
                    success=False,
                    strategy=RecoveryStrategy.ESCALATION,
                    status=RecoveryStatus.FAILED,
                    message=f"No recovery handler available for {error_context.error_type}",
                    execution_time_ms=0.0,
                    original_error=error_context
                )
            else:
                # Execute recovery
                result = handler.recover(error_context, original_function, *args, **kwargs)
            
            # Add to recovery history
            self._add_to_history(result)
            
            # Log recovery attempt
            self.logger.info(
                f"Recovery attempt for {error_context.error_type}: "
                f"{result.strategy.value} -> {result.status.value}"
            )
            
            return result
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery framework error: {str(recovery_error)}")
            
            # Create failure result for recovery error
            result = RecoveryResult(
                success=False,
                strategy=RecoveryStrategy.ESCALATION,
                status=RecoveryStatus.FAILED,
                message=f"Recovery framework error: {str(recovery_error)}",
                execution_time_ms=0.0,
                original_error=error_context
            )
            
            self._add_to_history(result)
            return result
    
    def _select_recovery_handler(self, error_context: ErrorContext) -> Optional[BaseRecoveryHandler]:
        """Select appropriate recovery handler for error"""
        for handler in self.handlers:
            try:
                if handler.can_handle(error_context):
                    return handler
            except Exception as e:
                self.logger.error(f"Error checking handler {handler.name}: {e}")
        
        return None
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics"""
        try:
            if not self.recovery_history:
                return {'total_recoveries': 0}
            
            total_recoveries = len(self.recovery_history)
            successful = sum(1 for r in self.recovery_history if r.success)
            
            # Strategy distribution
            strategy_counts = {}
            for result in self.recovery_history:
                strategy = result.strategy.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Status distribution
            status_counts = {}
            for result in self.recovery_history:
                status = result.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Error type distribution
            error_type_counts = {}
            for result in self.recovery_history:
                error_type = result.original_error.error_type
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            # Average execution times by strategy
            strategy_times = {}
            strategy_attempt_counts = {}
            
            for result in self.recovery_history:
                strategy = result.strategy.value
                if strategy not in strategy_times:
                    strategy_times[strategy] = []
                strategy_times[strategy].append(result.execution_time_ms)
                
                if strategy not in strategy_attempt_counts:
                    strategy_attempt_counts[strategy] = []
                strategy_attempt_counts[strategy].append(result.attempts_made)
            
            avg_times = {
                strategy: sum(times) / len(times)
                for strategy, times in strategy_times.items()
            }
            
            avg_attempts = {
                strategy: sum(attempts) / len(attempts)
                for strategy, attempts in strategy_attempt_counts.items()
            }
            
            return {
                'total_recoveries': total_recoveries,
                'successful_recoveries': successful,
                'failed_recoveries': total_recoveries - successful,
                'success_rate_pct': (successful / total_recoveries) * 100,
                'strategy_distribution': strategy_counts,
                'status_distribution': status_counts,
                'error_type_distribution': error_type_counts,
                'average_execution_time_by_strategy_ms': avg_times,
                'average_attempts_by_strategy': avg_attempts,
                'registered_handlers': [h.name for h in self.handlers],
                'handler_count': len(self.handlers)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recovery statistics: {e}")
            return {'error': str(e)}
    
    def _register_default_handlers(self):
        """Register default recovery handlers"""
        try:
            # Retry handler
            retry_handler = RetryRecoveryHandler(
                retry_delay=1.0,
                backoff_multiplier=2.0,
                max_attempts=3
            )
            self.register_handler(retry_handler)
            
            # Fallback handler
            fallback_handler = FallbackRecoveryHandler()
            self.register_handler(fallback_handler)
            
            # Graceful degradation handler
            degradation_handler = GracefulDegradationHandler()
            # Register some common degraded responses
            degradation_handler.register_degraded_response('default', None)
            degradation_handler.register_degraded_response('get_data', {'status': 'degraded', 'data': []})
            self.register_handler(degradation_handler)
            
            self.logger.info("Default recovery handlers registered")
            
        except Exception as e:
            self.logger.error(f"Error registering default handlers: {e}")
    
    def _initialize_error_patterns(self):
        """Initialize error severity patterns"""
        self.error_patterns.update({
            'ConnectionError': ErrorSeverity.MEDIUM,
            'TimeoutError': ErrorSeverity.MEDIUM,
            'RateLimitError': ErrorSeverity.LOW,
            'ValidationError': ErrorSeverity.HIGH,
            'SecurityError': ErrorSeverity.CRITICAL,
            'AuthenticationError': ErrorSeverity.HIGH,
            'AuthorizationError': ErrorSeverity.HIGH,
            'SystemError': ErrorSeverity.FATAL,
            'MemoryError': ErrorSeverity.FATAL,
            'KeyboardInterrupt': ErrorSeverity.FATAL
        })
    
    def _add_to_history(self, result: RecoveryResult):
        """Add result to recovery history with limit"""
        self.recovery_history.append(result)
        
        # Limit history size
        if len(self.recovery_history) > self.max_history:
            self.recovery_history = self.recovery_history[-self.max_history // 2:]


# Global error recovery framework
error_recovery_framework = ErrorRecoveryFramework()


# Convenience decorator for automatic error recovery
def with_error_recovery(function_name: str = None, context_data: Dict[str, Any] = None):
    """Decorator for automatic error recovery"""
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                result = error_recovery_framework.recover_from_error(
                    e, func, function_name or func.__name__, context_data, *args, **kwargs
                )
                
                if result.success:
                    return result.recovery_value
                else:
                    # Re-raise original error if recovery failed
                    raise e
        
        return wrapper
    return decorator