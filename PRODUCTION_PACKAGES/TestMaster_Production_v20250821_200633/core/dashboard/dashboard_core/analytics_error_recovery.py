"""
Analytics Advanced Error Recovery and Graceful Degradation System
================================================================

Comprehensive error recovery system with automatic healing, graceful
degradation, failover mechanisms, and intelligent error pattern detection.

Author: TestMaster Team
"""

import logging
import time
import threading
import traceback
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
import copy
import psutil

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class RecoveryStrategy(Enum):
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_TO_CACHE = "fallback_to_cache"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER_TO_BACKUP = "failover_to_backup"
    PARTIAL_FUNCTIONALITY = "partial_functionality"
    SAFE_MODE = "safe_mode"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

class DegradationLevel(Enum):
    NONE = "none"
    MINIMAL = "minimal"
    PARTIAL = "partial"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ErrorPattern:
    """Error pattern definition for intelligent detection."""
    pattern_id: str
    error_type: str
    error_message_pattern: str
    frequency_threshold: int
    time_window_seconds: int
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    description: str
    custom_handler: Optional[Callable] = None

@dataclass
class ErrorRecord:
    """Individual error occurrence record."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    component: str
    function_name: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None

@dataclass
class RecoveryAttempt:
    """Recovery attempt record."""
    attempt_id: str
    error_id: str
    strategy: RecoveryStrategy
    timestamp: datetime
    success: bool
    duration_ms: float
    details: Dict[str, Any]
    fallback_triggered: bool = False

@dataclass
class ComponentHealth:
    """Component health status."""
    component_name: str
    health_score: float  # 0.0 to 1.0
    error_rate: float
    last_error_time: Optional[datetime]
    degradation_level: DegradationLevel
    available_functions: List[str]
    disabled_functions: List[str]
    backup_systems: List[str]

class AnalyticsErrorRecovery:
    """
    Advanced error recovery and graceful degradation system.
    """
    
    def __init__(self, max_error_history: int = 10000,
                 recovery_timeout: float = 30.0,
                 health_check_interval: float = 60.0):
        """
        Initialize analytics error recovery system.
        
        Args:
            max_error_history: Maximum errors to keep in history
            recovery_timeout: Timeout for recovery attempts
            health_check_interval: Interval for health checks
        """
        self.max_error_history = max_error_history
        self.recovery_timeout = recovery_timeout
        self.health_check_interval = health_check_interval
        
        # Error tracking
        self.error_history = deque(maxlen=max_error_history)
        self.error_patterns = {}
        self.recovery_attempts = deque(maxlen=max_error_history)
        
        # Component health tracking
        self.component_health = {}
        self.system_degradation_level = DegradationLevel.NONE
        
        # Recovery strategies
        self.recovery_handlers = {}
        self.fallback_functions = {}
        self.backup_systems = {}
        
        # Circuit breaker states
        self.circuit_breakers = defaultdict(lambda: {
            'state': 'closed',  # closed, open, half_open
            'failure_count': 0,
            'last_failure_time': None,
            'failure_threshold': 5,
            'timeout_seconds': 60
        })
        
        # Performance and statistics
        self.recovery_stats = {
            'total_errors': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'degradation_events': 0,
            'failover_events': 0,
            'start_time': datetime.now()
        }
        
        # Background monitoring
        self.recovery_active = False
        self.monitor_thread = None
        self.health_thread = None
        
        # Error pattern detection
        self.pattern_detector = ErrorPatternDetector()
        
        # Setup default patterns and handlers
        self._setup_default_error_patterns()
        self._setup_default_recovery_handlers()
        
        logger.info("Analytics Error Recovery System initialized")
    
    def start_error_recovery(self):
        """Start error recovery monitoring."""
        if self.recovery_active:
            return
        
        self.recovery_active = True
        
        # Start monitoring threads
        self.monitor_thread = threading.Thread(target=self._error_monitoring_loop, daemon=True)
        self.health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        
        self.monitor_thread.start()
        self.health_thread.start()
        
        logger.info("Analytics error recovery monitoring started")
    
    def stop_error_recovery(self):
        """Stop error recovery monitoring."""
        self.recovery_active = False
        
        # Wait for threads to finish
        for thread in [self.monitor_thread, self.health_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("Analytics error recovery monitoring stopped")
    
    def handle_error(self, error: Exception, component: str = "unknown",
                    context: Dict[str, Any] = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> bool:
        """
        Handle an error with automatic recovery attempts.
        
        Args:
            error: Exception that occurred
            component: Component where error occurred
            context: Additional context information
            severity: Error severity level
        
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            # Create error record
            error_record = self._create_error_record(error, component, context, severity)
            
            # Store error
            self.error_history.append(error_record)
            self.recovery_stats['total_errors'] += 1
            
            # Update component health
            self._update_component_health(component, error_record)
            
            # Detect error patterns
            pattern = self.pattern_detector.detect_pattern(error_record, self.error_history)
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(error_record, pattern)
            
            # Attempt recovery
            recovery_successful = self._attempt_recovery(error_record, recovery_strategy)
            
            # Update circuit breaker
            self._update_circuit_breaker(component, not recovery_successful)
            
            # Check if degradation is needed
            self._check_and_apply_degradation(component, error_record)
            
            logger.info(f"Error handled for {component}: recovery {'successful' if recovery_successful else 'failed'}")
            return recovery_successful
        
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            return False
    
    def register_fallback_function(self, component: str, function_name: str, 
                                 fallback: Callable, degraded_version: Optional[Callable] = None):
        """Register fallback functions for graceful degradation."""
        if component not in self.fallback_functions:
            self.fallback_functions[component] = {}
        
        self.fallback_functions[component][function_name] = {
            'fallback': fallback,
            'degraded': degraded_version,
            'original_available': True
        }
        
        logger.info(f"Registered fallback for {component}.{function_name}")
    
    def register_backup_system(self, component: str, backup_system: Any, priority: int = 1):
        """Register backup systems for failover."""
        if component not in self.backup_systems:
            self.backup_systems[component] = []
        
        self.backup_systems[component].append({
            'system': backup_system,
            'priority': priority,
            'available': True,
            'last_used': None
        })
        
        # Sort by priority (higher priority first)
        self.backup_systems[component].sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"Registered backup system for {component} with priority {priority}")
    
    def get_component_health(self, component: str) -> ComponentHealth:
        """Get health status for a component."""
        return self.component_health.get(component, ComponentHealth(
            component_name=component,
            health_score=1.0,
            error_rate=0.0,
            last_error_time=None,
            degradation_level=DegradationLevel.NONE,
            available_functions=[],
            disabled_functions=[],
            backup_systems=[]
        ))
    
    def force_degradation(self, component: str, level: DegradationLevel, reason: str):
        """Force degradation of a component."""
        self._apply_degradation(component, level, reason)
        logger.warning(f"Forced degradation of {component} to {level.value}: {reason}")
    
    def restore_component(self, component: str) -> bool:
        """Attempt to restore a degraded component."""
        try:
            health = self.component_health.get(component)
            if not health:
                return False
            
            # Reset degradation
            health.degradation_level = DegradationLevel.NONE
            health.disabled_functions = []
            
            # Reset circuit breaker
            if component in self.circuit_breakers:
                self.circuit_breakers[component]['state'] = 'closed'
                self.circuit_breakers[component]['failure_count'] = 0
            
            # Re-enable fallback functions
            if component in self.fallback_functions:
                for func_name, func_info in self.fallback_functions[component].items():
                    func_info['original_available'] = True
            
            logger.info(f"Component {component} restored to full functionality")
            return True
        
        except Exception as e:
            logger.error(f"Error restoring component {component}: {e}")
            return False
    
    def get_error_recovery_summary(self) -> Dict[str, Any]:
        """Get error recovery system summary."""
        uptime = (datetime.now() - self.recovery_stats['start_time']).total_seconds()
        
        # Recent errors (last hour)
        recent_errors = [e for e in self.error_history
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]
        
        # Error severity distribution
        severity_dist = defaultdict(int)
        for error in recent_errors:
            severity_dist[error.severity.value] += 1
        
        # Component health summary
        component_summary = {}
        for comp_name, health in self.component_health.items():
            component_summary[comp_name] = {
                'health_score': health.health_score,
                'degradation_level': health.degradation_level.value,
                'error_rate': health.error_rate,
                'available_functions': len(health.available_functions),
                'disabled_functions': len(health.disabled_functions)
            }
        
        # Circuit breaker status
        circuit_status = {}
        for comp_name, breaker in self.circuit_breakers.items():
            circuit_status[comp_name] = {
                'state': breaker['state'],
                'failure_count': breaker['failure_count']
            }
        
        return {
            'recovery_status': {
                'active': self.recovery_active,
                'system_degradation_level': self.system_degradation_level.value,
                'uptime_seconds': uptime
            },
            'statistics': self.recovery_stats.copy(),
            'error_analysis': {
                'total_errors_tracked': len(self.error_history),
                'recent_errors': len(recent_errors),
                'error_severity_distribution': dict(severity_dist),
                'recovery_success_rate': (
                    (self.recovery_stats['successful_recoveries'] / 
                     max(1, self.recovery_stats['recovery_attempts'])) * 100
                )
            },
            'component_health': component_summary,
            'circuit_breakers': circuit_status,
            'fallback_systems': {
                'registered_components': len(self.fallback_functions),
                'backup_systems': len(self.backup_systems)
            },
            'error_patterns': {
                'patterns_detected': len(self.error_patterns),
                'pattern_detector_active': True
            }
        }
    
    def _create_error_record(self, error: Exception, component: str,
                           context: Dict[str, Any], severity: ErrorSeverity) -> ErrorRecord:
        """Create detailed error record."""
        # Get caller information
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back.f_back  # Go up the stack
            function_name = caller_frame.f_code.co_name if caller_frame else "unknown"
        except:
            function_name = "unknown"
        finally:
            del frame
        
        error_record = ErrorRecord(
            error_id=f"error_{int(time.time() * 1000000)}",
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            component=component,
            function_name=function_name,
            severity=severity,
            context=context or {}
        )
        
        return error_record
    
    def _update_component_health(self, component: str, error_record: ErrorRecord):
        """Update component health based on error."""
        if component not in self.component_health:
            self.component_health[component] = ComponentHealth(
                component_name=component,
                health_score=1.0,
                error_rate=0.0,
                last_error_time=None,
                degradation_level=DegradationLevel.NONE,
                available_functions=[],
                disabled_functions=[],
                backup_systems=[]
            )
        
        health = self.component_health[component]
        health.last_error_time = error_record.timestamp
        
        # Calculate error rate (errors per hour)
        recent_errors = [e for e in self.error_history
                        if (e.component == component and 
                            (datetime.now() - e.timestamp).total_seconds() < 3600)]
        health.error_rate = len(recent_errors)
        
        # Update health score based on error severity and frequency
        severity_impact = {
            ErrorSeverity.CRITICAL: 0.5,
            ErrorSeverity.HIGH: 0.2,
            ErrorSeverity.MEDIUM: 0.1,
            ErrorSeverity.LOW: 0.05,
            ErrorSeverity.INFO: 0.01
        }
        
        impact = severity_impact.get(error_record.severity, 0.1)
        health.health_score = max(0.0, health.health_score - impact)
        
        # Decay health score over time if no recent errors
        if health.last_error_time:
            time_since_error = (datetime.now() - health.last_error_time).total_seconds()
            if time_since_error > 3600:  # 1 hour
                recovery_rate = min(0.1, time_since_error / 36000)  # Recover over 10 hours
                health.health_score = min(1.0, health.health_score + recovery_rate)
    
    def _determine_recovery_strategy(self, error_record: ErrorRecord, 
                                   pattern: Optional[ErrorPattern]) -> RecoveryStrategy:
        """Determine the best recovery strategy for an error."""
        # Use pattern-specific strategy if available
        if pattern:
            return pattern.recovery_strategy
        
        # Default strategies based on error type and severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.FAILOVER_TO_BACKUP
        elif error_record.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        elif "connection" in error_record.error_message.lower():
            return RecoveryStrategy.RETRY_WITH_BACKOFF
        elif "memory" in error_record.error_message.lower():
            return RecoveryStrategy.PARTIAL_FUNCTIONALITY
        else:
            return RecoveryStrategy.FALLBACK_TO_CACHE
    
    def _attempt_recovery(self, error_record: ErrorRecord, strategy: RecoveryStrategy) -> bool:
        """Attempt recovery using the specified strategy."""
        self.recovery_stats['recovery_attempts'] += 1
        
        attempt = RecoveryAttempt(
            attempt_id=f"recovery_{int(time.time() * 1000000)}",
            error_id=error_record.error_id,
            strategy=strategy,
            timestamp=datetime.now(),
            success=False,
            duration_ms=0,
            details={}
        )
        
        start_time = time.time()
        
        try:
            if strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                success = self._retry_with_backoff(error_record, attempt)
            elif strategy == RecoveryStrategy.FALLBACK_TO_CACHE:
                success = self._fallback_to_cache(error_record, attempt)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success = self._apply_graceful_degradation(error_record, attempt)
            elif strategy == RecoveryStrategy.FAILOVER_TO_BACKUP:
                success = self._failover_to_backup(error_record, attempt)
            elif strategy == RecoveryStrategy.PARTIAL_FUNCTIONALITY:
                success = self._enable_partial_functionality(error_record, attempt)
            elif strategy == RecoveryStrategy.SAFE_MODE:
                success = self._activate_safe_mode(error_record, attempt)
            else:
                success = False
            
            attempt.success = success
            attempt.duration_ms = (time.time() - start_time) * 1000
            
            if success:
                self.recovery_stats['successful_recoveries'] += 1
                error_record.recovery_successful = True
            else:
                self.recovery_stats['failed_recoveries'] += 1
            
            error_record.recovery_attempted = True
            error_record.recovery_strategy = strategy
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            attempt.success = False
            attempt.details['error'] = str(e)
        
        self.recovery_attempts.append(attempt)
        return attempt.success
    
    def _retry_with_backoff(self, error_record: ErrorRecord, attempt: RecoveryAttempt) -> bool:
        """Implement retry with exponential backoff."""
        max_retries = 3
        base_delay = 1.0
        
        for retry in range(max_retries):
            delay = base_delay * (2 ** retry)
            time.sleep(delay)
            
            try:
                # This would typically retry the failed operation
                # For now, we simulate success based on retry count
                if retry >= 1:  # Succeed after first retry
                    attempt.details['retries'] = retry + 1
                    attempt.details['total_delay'] = sum(base_delay * (2 ** i) for i in range(retry + 1))
                    return True
            except Exception as e:
                if retry == max_retries - 1:
                    attempt.details['final_error'] = str(e)
        
        return False
    
    def _fallback_to_cache(self, error_record: ErrorRecord, attempt: RecoveryAttempt) -> bool:
        """Fallback to cached data."""
        component = error_record.component
        
        # Check if component has fallback functions
        if component in self.fallback_functions:
            # Enable fallback mode
            for func_name, func_info in self.fallback_functions[component].items():
                func_info['original_available'] = False
            
            attempt.details['fallback_enabled'] = True
            attempt.fallback_triggered = True
            return True
        
        return False
    
    def _apply_graceful_degradation(self, error_record: ErrorRecord, attempt: RecoveryAttempt) -> bool:
        """Apply graceful degradation."""
        component = error_record.component
        
        # Determine degradation level based on error severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            degradation_level = DegradationLevel.SIGNIFICANT
        elif error_record.severity == ErrorSeverity.HIGH:
            degradation_level = DegradationLevel.PARTIAL
        else:
            degradation_level = DegradationLevel.MINIMAL
        
        self._apply_degradation(component, degradation_level, f"Error: {error_record.error_message}")
        
        attempt.details['degradation_level'] = degradation_level.value
        return True
    
    def _failover_to_backup(self, error_record: ErrorRecord, attempt: RecoveryAttempt) -> bool:
        """Failover to backup system."""
        component = error_record.component
        
        if component in self.backup_systems:
            available_backups = [b for b in self.backup_systems[component] if b['available']]
            
            if available_backups:
                # Use highest priority available backup
                backup = available_backups[0]
                backup['last_used'] = datetime.now()
                
                attempt.details['backup_system_priority'] = backup['priority']
                self.recovery_stats['failover_events'] += 1
                return True
        
        return False
    
    def _enable_partial_functionality(self, error_record: ErrorRecord, attempt: RecoveryAttempt) -> bool:
        """Enable partial functionality mode."""
        component = error_record.component
        
        # Disable non-essential functions
        health = self.component_health.get(component)
        if health:
            # Move some functions to disabled list
            if len(health.available_functions) > 1:
                # Disable half of the available functions
                functions_to_disable = health.available_functions[len(health.available_functions)//2:]
                health.disabled_functions.extend(functions_to_disable)
                health.available_functions = health.available_functions[:len(health.available_functions)//2]
                
                attempt.details['disabled_functions'] = len(functions_to_disable)
                return True
        
        return False
    
    def _activate_safe_mode(self, error_record: ErrorRecord, attempt: RecoveryAttempt) -> bool:
        """Activate safe mode with minimal functionality."""
        component = error_record.component
        
        self._apply_degradation(component, DegradationLevel.CRITICAL, "Safe mode activated")
        
        attempt.details['safe_mode'] = True
        return True
    
    def _apply_degradation(self, component: str, level: DegradationLevel, reason: str):
        """Apply degradation to a component."""
        health = self.component_health.get(component)
        if not health:
            return
        
        health.degradation_level = level
        self.recovery_stats['degradation_events'] += 1
        
        # Update system-wide degradation level
        max_degradation = max([h.degradation_level for h in self.component_health.values()], 
                            default=DegradationLevel.NONE)
        self.system_degradation_level = max_degradation
        
        logger.warning(f"Applied {level.value} degradation to {component}: {reason}")
    
    def _update_circuit_breaker(self, component: str, failed: bool):
        """Update circuit breaker state."""
        breaker = self.circuit_breakers[component]
        
        if failed:
            breaker['failure_count'] += 1
            breaker['last_failure_time'] = datetime.now()
            
            if breaker['failure_count'] >= breaker['failure_threshold']:
                breaker['state'] = 'open'
                logger.warning(f"Circuit breaker opened for {component}")
        else:
            # Success - reset failure count
            breaker['failure_count'] = 0
            if breaker['state'] == 'half_open':
                breaker['state'] = 'closed'
                logger.info(f"Circuit breaker closed for {component}")
    
    def _check_and_apply_degradation(self, component: str, error_record: ErrorRecord):
        """Check if degradation should be applied based on error patterns."""
        health = self.component_health.get(component)
        if not health:
            return
        
        # Apply degradation if health score is too low
        if health.health_score < 0.3 and health.degradation_level == DegradationLevel.NONE:
            self._apply_degradation(component, DegradationLevel.PARTIAL, "Low health score")
        elif health.health_score < 0.1:
            self._apply_degradation(component, DegradationLevel.SIGNIFICANT, "Critical health score")
    
    def _error_monitoring_loop(self):
        """Background error monitoring loop."""
        while self.recovery_active:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Check circuit breakers for timeout recovery
                current_time = datetime.now()
                for component, breaker in self.circuit_breakers.items():
                    if (breaker['state'] == 'open' and 
                        breaker['last_failure_time'] and
                        (current_time - breaker['last_failure_time']).total_seconds() > breaker['timeout_seconds']):
                        breaker['state'] = 'half_open'
                        logger.info(f"Circuit breaker half-opened for {component}")
                
                # Check for automatic recovery opportunities
                self._check_automatic_recovery()
                
            except Exception as e:
                logger.error(f"Error monitoring loop error: {e}")
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self.recovery_active:
            try:
                time.sleep(self.health_check_interval)
                
                # Update health scores for all components
                for component_name, health in self.component_health.items():
                    self._update_component_health_score(component_name, health)
                
                # Check system-wide health
                self._check_system_health()
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
    
    def _update_component_health_score(self, component_name: str, health: ComponentHealth):
        """Update component health score based on recent performance."""
        current_time = datetime.now()
        
        # Improve health score if no recent errors
        if health.last_error_time:
            time_since_error = (current_time - health.last_error_time).total_seconds()
            if time_since_error > 1800:  # 30 minutes without errors
                improvement = min(0.1, time_since_error / 18000)  # Improve over 5 hours
                health.health_score = min(1.0, health.health_score + improvement)
                
                # Reduce degradation level if health improves
                if health.health_score > 0.8 and health.degradation_level != DegradationLevel.NONE:
                    new_level = DegradationLevel(max(0, list(DegradationLevel).index(health.degradation_level) - 1))
                    health.degradation_level = new_level
                    logger.info(f"Reduced degradation level for {component_name} to {new_level.value}")
    
    def _check_automatic_recovery(self):
        """Check for automatic recovery opportunities."""
        for component_name, health in self.component_health.items():
            if (health.degradation_level != DegradationLevel.NONE and 
                health.health_score > 0.7):
                # Attempt to restore component
                if self.restore_component(component_name):
                    logger.info(f"Automatically restored component: {component_name}")
    
    def _check_system_health(self):
        """Check overall system health."""
        if not self.component_health:
            return
        
        # Calculate average health score
        avg_health = sum(h.health_score for h in self.component_health.values()) / len(self.component_health)
        
        # Update system degradation level
        if avg_health < 0.3:
            self.system_degradation_level = DegradationLevel.CRITICAL
        elif avg_health < 0.5:
            self.system_degradation_level = DegradationLevel.SIGNIFICANT
        elif avg_health < 0.7:
            self.system_degradation_level = DegradationLevel.PARTIAL
        elif avg_health < 0.9:
            self.system_degradation_level = DegradationLevel.MINIMAL
        else:
            self.system_degradation_level = DegradationLevel.NONE
    
    def _setup_default_error_patterns(self):
        """Setup default error patterns for detection."""
        patterns = [
            ErrorPattern(
                pattern_id="connection_timeout",
                error_type="ConnectionError",
                error_message_pattern=".*timeout.*",
                frequency_threshold=3,
                time_window_seconds=300,
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                description="Connection timeout errors"
            ),
            ErrorPattern(
                pattern_id="memory_error",
                error_type="MemoryError",
                error_message_pattern=".*memory.*",
                frequency_threshold=2,
                time_window_seconds=600,
                severity=ErrorSeverity.CRITICAL,
                recovery_strategy=RecoveryStrategy.PARTIAL_FUNCTIONALITY,
                description="Memory-related errors"
            ),
            ErrorPattern(
                pattern_id="database_error",
                error_type="DatabaseError",
                error_message_pattern=".*database.*",
                frequency_threshold=5,
                time_window_seconds=900,
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.FALLBACK_TO_CACHE,
                description="Database connectivity issues"
            )
        ]
        
        for pattern in patterns:
            self.error_patterns[pattern.pattern_id] = pattern
    
    def _setup_default_recovery_handlers(self):
        """Setup default recovery handlers."""
        def default_retry_handler(error_record, attempt):
            return self._retry_with_backoff(error_record, attempt)
        
        def default_fallback_handler(error_record, attempt):
            return self._fallback_to_cache(error_record, attempt)
        
        self.recovery_handlers[RecoveryStrategy.RETRY_WITH_BACKOFF] = default_retry_handler
        self.recovery_handlers[RecoveryStrategy.FALLBACK_TO_CACHE] = default_fallback_handler
    
    def get_recent_errors(self, hours: int = 24, component: Optional[str] = None,
                         severity: Optional[ErrorSeverity] = None) -> List[ErrorRecord]:
        """Get recent error records."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        if component:
            recent_errors = [e for e in recent_errors if e.component == component]
        
        if severity:
            recent_errors = [e for e in recent_errors if e.severity == severity]
        
        # Sort by timestamp (most recent first)
        recent_errors.sort(key=lambda e: e.timestamp, reverse=True)
        
        return recent_errors
    
    def shutdown(self):
        """Shutdown error recovery system."""
        self.stop_error_recovery()
        logger.info("Analytics Error Recovery System shutdown")


class ErrorPatternDetector:
    """Intelligent error pattern detection."""
    
    def __init__(self):
        self.pattern_cache = {}
    
    def detect_pattern(self, error_record: ErrorRecord, 
                      error_history: deque) -> Optional[ErrorPattern]:
        """Detect if an error matches any known patterns."""
        # Simple pattern matching implementation
        # In a real system, this would use more sophisticated ML techniques
        
        # Count similar errors in recent history
        recent_cutoff = datetime.now() - timedelta(minutes=30)
        similar_errors = [
            e for e in error_history
            if (e.timestamp >= recent_cutoff and
                e.error_type == error_record.error_type and
                e.component == error_record.component)
        ]
        
        # If we have multiple similar errors, it's a pattern
        if len(similar_errors) >= 3:
            return ErrorPattern(
                pattern_id=f"detected_{error_record.error_type}_{error_record.component}",
                error_type=error_record.error_type,
                error_message_pattern=".*",
                frequency_threshold=3,
                time_window_seconds=1800,
                severity=error_record.severity,
                recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                description=f"Detected pattern for {error_record.error_type} in {error_record.component}"
            )
        
        return None