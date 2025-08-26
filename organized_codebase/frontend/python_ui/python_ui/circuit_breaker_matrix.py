"""
Circuit Breaker Matrix - Archive-Derived Reliability System
==========================================================

Advanced circuit breaker management system with component-specific
configurations, health monitoring, and automatic recovery patterns.

Author: Agent C Security Framework
Created: 2025-08-21
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
import json
import sqlite3
import os

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states with extended monitoring."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failures detected, blocking calls
    HALF_OPEN = "half_open" # Testing if service has recovered
    DEGRADED = "degraded"   # Partial functionality available
    MAINTENANCE = "maintenance"  # Planned maintenance mode

class FailureType(Enum):
    """Enhanced failure classification."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    VALIDATION = "validation"
    RESOURCE = "resource"
    UNAVAILABLE = "unavailable"
    OVERLOAD = "overload"
    SECURITY = "security"
    CONFIGURATION = "configuration"

class ComponentType(Enum):
    """System component types for specialized handling."""
    DATA_SOURCE = "data_source"
    PROCESSOR = "processor"
    STORAGE = "storage"
    EXTERNAL_SERVICE = "external_service"
    AUTHENTICATION = "authentication"
    CACHE = "cache"
    QUEUE = "queue"
    NETWORK = "network"

@dataclass
class CircuitBreakerConfig:
    """Advanced circuit breaker configuration."""
    failure_threshold: int = 5           # Failures before opening
    timeout_seconds: float = 30.0        # Timeout for operations
    recovery_timeout: int = 60           # Seconds before trying half-open
    success_threshold: int = 3           # Successes needed to close from half-open
    max_concurrent_calls: int = 10       # Max concurrent operations
    degraded_threshold: int = 3          # Failures before degraded mode
    maintenance_mode: bool = False       # Allow maintenance mode
    auto_recovery: bool = True           # Enable automatic recovery
    health_check_interval: int = 30      # Health check frequency
    failure_rate_window: int = 300       # Time window for failure rate calculation

@dataclass
class FailureRecord:
    """Enhanced failure information record."""
    failure_id: str
    failure_type: FailureType
    error_message: str
    timestamp: datetime
    component_name: str
    severity: int = 3  # 1=Critical, 2=High, 3=Medium, 4=Low, 5=Info
    context: Dict[str, Any] = None
    recovery_suggestions: List[str] = None

class CircuitBreakerMatrix:
    """
    Advanced circuit breaker implementation with component intelligence.
    """
    
    def __init__(self, name: str, component_type: ComponentType, 
                 config: CircuitBreakerConfig = None, db_path: str = "data/circuit_matrix.db"):
        """
        Initialize circuit breaker matrix.
        
        Args:
            name: Component name
            component_type: Type of component
            config: Circuit breaker configuration
            db_path: Database path for persistence
        """
        self.name = name
        self.component_type = component_type
        self.config = config or CircuitBreakerConfig()
        self.db_path = db_path
        
        # Initialize database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.failure_count = 0
        self.success_count = 0
        self.degraded_mode_count = 0
        
        # Enhanced failure tracking
        self.failures = deque(maxlen=1000)
        self.call_history = deque(maxlen=5000)
        self.health_metrics = deque(maxlen=100)
        
        # Concurrency and load tracking
        self.active_calls = 0
        self.total_calls_today = 0
        self.last_reset_time = datetime.now()
        self.call_lock = threading.Lock()
        
        # Advanced statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'blocked_calls': 0,
            'degraded_calls': 0,
            'timeouts': 0,
            'state_changes': 0,
            'last_state_change': None,
            'failure_rate': 0.0,
            'average_response_time': 0.0,
            'uptime_percentage': 100.0
        }
        
        # Component-specific metrics
        self.component_metrics = {
            'peak_concurrent_calls': 0,
            'peak_response_time': 0.0,
            'resource_utilization': 0.0,
            'error_patterns': defaultdict(int),
            'recovery_patterns': defaultdict(int),
            'maintenance_windows': []
        }
        
        # Callback systems
        self.state_change_callbacks = []
        self.failure_callbacks = []
        self.health_check_callbacks = []
        self.recovery_callbacks = []
        
        # Health monitoring
        self.health_monitoring_active = False
        self.health_thread = None
        
        # Auto-recovery system
        self.auto_recovery_active = self.config.auto_recovery
        self.recovery_thread = None
        
        logger.info(f"Circuit Breaker Matrix '{name}' initialized for {component_type.value}")
    
    def _init_database(self):
        """Initialize circuit breaker database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Circuit states table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS circuit_states (
                        component_name TEXT PRIMARY KEY,
                        component_type TEXT NOT NULL,
                        current_state TEXT NOT NULL,
                        last_state_change TEXT,
                        failure_count INTEGER DEFAULT 0,
                        success_count INTEGER DEFAULT 0,
                        configuration TEXT
                    )
                ''')
                
                # Failure records table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS failure_records (
                        failure_id TEXT PRIMARY KEY,
                        component_name TEXT NOT NULL,
                        failure_type TEXT NOT NULL,
                        error_message TEXT,
                        timestamp TEXT NOT NULL,
                        severity INTEGER DEFAULT 3,
                        context TEXT,
                        recovery_suggestions TEXT
                    )
                ''')
                
                # Health metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS health_metrics (
                        metric_id TEXT PRIMARY KEY,
                        component_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        response_time REAL,
                        success_rate REAL,
                        resource_utilization REAL,
                        concurrent_calls INTEGER
                    )
                ''')
                
                # Recovery events table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS recovery_events (
                        event_id TEXT PRIMARY KEY,
                        component_name TEXT NOT NULL,
                        from_state TEXT NOT NULL,
                        to_state TEXT NOT NULL,
                        trigger_type TEXT,
                        timestamp TEXT NOT NULL,
                        success INTEGER DEFAULT 0,
                        duration_seconds REAL
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Circuit matrix database initialization failed: {e}")
    
    def __call__(self, func):
        """Decorator to wrap functions with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with advanced circuit breaker protection.
        """
        with self.call_lock:
            self.stats['total_calls'] += 1
            self.total_calls_today += 1
            
            # Check if circuit is open or in maintenance
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to_half_open()
                else:
                    self.stats['blocked_calls'] += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Component: {self.component_type.value}, "
                        f"Last failure: {self.last_failure_time}"
                    )
            
            elif self.state == CircuitState.MAINTENANCE:
                self.stats['blocked_calls'] += 1
                raise CircuitBreakerMaintenanceException(
                    f"Circuit breaker '{self.name}' is in MAINTENANCE mode"
                )
            
            # Check concurrent call limit
            if self.active_calls >= self.config.max_concurrent_calls:
                self.stats['blocked_calls'] += 1
                raise CircuitBreakerOverloadException(
                    f"Circuit breaker '{self.name}' has reached max concurrent calls limit"
                )
            
            self.active_calls += 1
            self.component_metrics['peak_concurrent_calls'] = max(
                self.component_metrics['peak_concurrent_calls'], 
                self.active_calls
            )
        
        call_start = time.time()
        success = False
        error = None
        degraded = False
        
        try:
            # Execute with timeout and degraded mode handling
            if self.state == CircuitState.DEGRADED:
                result = self._execute_with_degradation(func, *args, **kwargs)
                degraded = True
            else:
                result = self._execute_with_timeout(func, *args, **kwargs)
            
            success = True
            processing_time = time.time() - call_start
            self._record_success(call_start, processing_time, degraded)
            return result
            
        except TimeoutError as e:
            error = str(e)
            processing_time = time.time() - call_start
            self._record_failure(FailureType.TIMEOUT, error, call_start, processing_time)
            raise
            
        except CircuitBreakerException:
            # Re-raise circuit breaker exceptions
            raise
            
        except Exception as e:
            error = str(e)
            processing_time = time.time() - call_start
            failure_type = self._classify_failure(e)
            self._record_failure(failure_type, error, call_start, processing_time)
            raise
            
        finally:
            with self.call_lock:
                self.active_calls -= 1
            
            # Record call in history
            self.call_history.append({
                'timestamp': datetime.now(),
                'duration': time.time() - call_start,
                'success': success,
                'degraded': degraded,
                'error': error,
                'component': self.component_type.value
            })
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with configurable timeout."""
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
            self.stats['timeouts'] += 1
            raise TimeoutError(
                f"Function timed out after {self.config.timeout_seconds} seconds"
            )
    
    def _execute_with_degradation(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in degraded mode with reduced functionality."""
        # In degraded mode, use reduced timeout
        degraded_timeout = self.config.timeout_seconds * 0.7
        
        result = None
        exception = None
        finished = threading.Event()
        
        def target():
            nonlocal result, exception
            try:
                # Attempt with reduced functionality
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
            finally:
                finished.set()
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        
        if finished.wait(timeout=degraded_timeout):
            if exception:
                raise exception
            
            # Mark as degraded response
            if isinstance(result, dict):
                result['_degraded_mode'] = True
                result['_degraded_at'] = datetime.now().isoformat()
            
            self.stats['degraded_calls'] += 1
            return result
        else:
            raise TimeoutError(f"Degraded function timed out after {degraded_timeout} seconds")
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type based on exception."""
        error_str = str(exception).lower()
        
        if 'timeout' in error_str or 'timed out' in error_str:
            return FailureType.TIMEOUT
        elif 'connection' in error_str or 'network' in error_str:
            return FailureType.UNAVAILABLE
        elif 'resource' in error_str or 'memory' in error_str or 'cpu' in error_str:
            return FailureType.RESOURCE
        elif 'overload' in error_str or 'capacity' in error_str:
            return FailureType.OVERLOAD
        elif 'validation' in error_str or 'invalid' in error_str:
            return FailureType.VALIDATION
        elif 'permission' in error_str or 'unauthorized' in error_str or 'forbidden' in error_str:
            return FailureType.SECURITY
        elif 'config' in error_str or 'setting' in error_str:
            return FailureType.CONFIGURATION
        else:
            return FailureType.EXCEPTION
    
    def _record_success(self, call_start: float, processing_time: float, degraded: bool = False):
        """Record successful call with enhanced metrics."""
        with self.call_lock:
            self.stats['successful_calls'] += 1
            
            # Update response time metrics
            if processing_time > self.component_metrics['peak_response_time']:
                self.component_metrics['peak_response_time'] = processing_time
            
            # Update average response time
            total_response_time = self.stats['average_response_time'] * (self.stats['successful_calls'] - 1)
            self.stats['average_response_time'] = (total_response_time + processing_time) / self.stats['successful_calls']
            
            # State transition logic
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.DEGRADED and not degraded:
                # Successful non-degraded call in degraded mode
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
                self.degraded_mode_count = 0
            
            # Update recovery patterns
            if self.state in [CircuitState.HALF_OPEN, CircuitState.DEGRADED]:
                self.component_metrics['recovery_patterns']['success_after_failure'] += 1
    
    def _record_failure(self, failure_type: FailureType, error: str, 
                       call_start: float, processing_time: float):
        """Record failed call with comprehensive analysis."""
        with self.call_lock:
            self.stats['failed_calls'] += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            # Create detailed failure record
            failure_record = FailureRecord(
                failure_id=f"fail_{self.name}_{int(time.time() * 1000000)}",
                failure_type=failure_type,
                error_message=error,
                timestamp=self.last_failure_time,
                component_name=self.name,
                severity=self._calculate_failure_severity(failure_type, error),
                context={
                    'component_type': self.component_type.value,
                    'processing_time': processing_time,
                    'active_calls': self.active_calls,
                    'current_state': self.state.value
                },
                recovery_suggestions=self._generate_recovery_suggestions(failure_type, error)
            )
            
            self.failures.append(failure_record)
            
            # Update error patterns
            self.component_metrics['error_patterns'][failure_type.value] += 1
            
            # Save to database
            self._save_failure_record(failure_record)
            
            # Trigger failure callbacks
            for callback in self.failure_callbacks:
                try:
                    callback(self, failure_record)
                except Exception as e:
                    logger.error(f"Failure callback error: {e}")
            
            # State transition logic based on failure patterns
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
                elif self.failure_count >= self.config.degraded_threshold:
                    self._transition_to_degraded()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.state == CircuitState.DEGRADED:
                self.degraded_mode_count += 1
                if self.degraded_mode_count >= self.config.failure_threshold:
                    self._transition_to_open()
    
    def _calculate_failure_severity(self, failure_type: FailureType, error: str) -> int:
        """Calculate failure severity (1=Critical, 5=Info)."""
        if failure_type == FailureType.SECURITY:
            return 1  # Critical
        elif failure_type in [FailureType.RESOURCE, FailureType.OVERLOAD]:
            return 2  # High
        elif failure_type in [FailureType.TIMEOUT, FailureType.UNAVAILABLE]:
            return 3  # Medium
        elif failure_type in [FailureType.VALIDATION, FailureType.CONFIGURATION]:
            return 4  # Low
        else:
            return 3  # Medium (default)
    
    def _generate_recovery_suggestions(self, failure_type: FailureType, error: str) -> List[str]:
        """Generate context-aware recovery suggestions."""
        suggestions = []
        
        if failure_type == FailureType.TIMEOUT:
            suggestions.extend([
                "Increase timeout configuration",
                "Check network connectivity",
                "Verify service availability",
                "Consider request optimization"
            ])
        elif failure_type == FailureType.RESOURCE:
            suggestions.extend([
                "Monitor resource usage",
                "Scale resources if needed",
                "Check memory leaks",
                "Optimize resource allocation"
            ])
        elif failure_type == FailureType.OVERLOAD:
            suggestions.extend([
                "Implement load balancing",
                "Increase concurrent call limits",
                "Add rate limiting",
                "Scale horizontally"
            ])
        elif failure_type == FailureType.UNAVAILABLE:
            suggestions.extend([
                "Check service health",
                "Verify network connectivity",
                "Consider fallback mechanisms",
                "Review service dependencies"
            ])
        elif failure_type == FailureType.SECURITY:
            suggestions.extend([
                "Review authentication",
                "Check authorization settings",
                "Verify security certificates",
                "Audit access permissions"
            ])
        
        return suggestions
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status."""
        recent_failures = [
            f for f in self.failures
            if datetime.now() - f.timestamp <= timedelta(minutes=30)
        ]
        
        # Calculate failure rate
        window_start = datetime.now() - timedelta(seconds=self.config.failure_rate_window)
        recent_calls = [
            call for call in self.call_history
            if call['timestamp'] >= window_start
        ]
        
        failure_rate = 0.0
        if recent_calls:
            failed_calls = sum(1 for call in recent_calls if not call['success'])
            failure_rate = (failed_calls / len(recent_calls)) * 100
        
        # Calculate uptime
        uptime_percentage = 100.0
        if self.stats['total_calls'] > 0:
            uptime_percentage = (self.stats['successful_calls'] / self.stats['total_calls']) * 100
        
        return {
            'component_name': self.name,
            'component_type': self.component_type.value,
            'current_state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'active_calls': self.active_calls,
            'recent_failures': len(recent_failures),
            'failure_rate_percentage': failure_rate,
            'uptime_percentage': uptime_percentage,
            'configuration': {
                'failure_threshold': self.config.failure_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'max_concurrent_calls': self.config.max_concurrent_calls,
                'auto_recovery': self.config.auto_recovery
            },
            'statistics': self.stats.copy(),
            'component_metrics': self.component_metrics.copy(),
            'health_monitoring_active': self.health_monitoring_active,
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shutdown circuit breaker matrix."""
        self.health_monitoring_active = False
        self.auto_recovery_active = False
        
        for thread in [self.health_thread, self.recovery_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Circuit Breaker Matrix '{self.name}' shutdown")

class CircuitBreakerException(Exception):
    """Base exception for circuit breaker errors."""
    pass

class CircuitBreakerOpenException(CircuitBreakerException):
    """Exception raised when circuit breaker is open."""
    pass

class CircuitBreakerMaintenanceException(CircuitBreakerException):
    """Exception raised when circuit breaker is in maintenance mode."""
    pass

class CircuitBreakerOverloadException(CircuitBreakerException):
    """Exception raised when circuit breaker is overloaded."""
    pass

# Global circuit breaker registry
circuit_breaker_registry = {}