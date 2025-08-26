"""
Quantum Retry Orchestrator
==========================

Advanced retry orchestration with context tracking, priority management,
and intelligent coordination. Extracted from 1,191-line archive component.

Manages retry contexts, coordinates retry strategies, and handles state management.
"""

import asyncio
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json

from .quantum_retry_strategies import RetryStrategy, FailurePattern, quantum_retry_strategies

logger = logging.getLogger(__name__)


class RetryPriority(Enum):
    """Retry priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class RetryStatus(Enum):
    """Retry operation status"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetryAttempt:
    """Individual retry attempt record"""
    attempt_id: str = field(default_factory=lambda: f"attempt_{uuid.uuid4().hex[:8]}")
    analytics_id: str = ""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    attempt_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    delay_used: float = 0.0
    priority: RetryPriority = RetryPriority.NORMAL
    success: bool = False
    error_info: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'attempt_id': self.attempt_id,
            'analytics_id': self.analytics_id,
            'strategy': self.strategy.value,
            'attempt_number': self.attempt_number,
            'timestamp': self.timestamp.isoformat(),
            'delay_used': self.delay_used,
            'priority': self.priority.value,
            'success': self.success,
            'error_info': self.error_info,
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }


@dataclass
class RetryContext:
    """Comprehensive retry context for adaptive strategies"""
    analytics_id: str
    original_data: Dict[str, Any]
    failure_history: List[RetryAttempt] = field(default_factory=list)
    current_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    priority: RetryPriority = RetryPriority.NORMAL
    max_attempts: int = 10
    created_at: datetime = field(default_factory=datetime.now)
    last_attempt: Optional[datetime] = None
    failure_pattern: FailurePattern = FailurePattern.UNKNOWN
    success_probability: float = 0.5
    predicted_next_success: Optional[datetime] = None
    status: RetryStatus = RetryStatus.PENDING
    context_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_attempt_count(self) -> int:
        """Get total number of attempts"""
        return len(self.failure_history)
    
    def get_success_rate(self) -> float:
        """Calculate success rate from history"""
        if not self.failure_history:
            return 0.0
        
        successes = sum(1 for attempt in self.failure_history if attempt.success)
        return successes / len(self.failure_history)
    
    def get_recent_failures(self, hours: int = 1) -> int:
        """Get number of recent failures"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return sum(
            1 for attempt in self.failure_history
            if not attempt.success and attempt.timestamp > cutoff_time
        )


class QuantumRetryOrchestrator:
    """Advanced retry orchestrator with intelligent coordination"""
    
    def __init__(self, max_concurrent_retries: int = 100):
        self.max_concurrent_retries = max_concurrent_retries
        self.logger = logging.getLogger(__name__)
        
        # Retry management
        self.active_retries: Dict[str, RetryContext] = {}
        self.completed_retries: deque = deque(maxlen=1000)
        self.retry_queue: Dict[RetryPriority, List[str]] = defaultdict(list)
        
        # Strategy engine
        self.strategy_engine = quantum_retry_strategies
        
        # Performance tracking
        self.stats = {
            'total_retries': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'cancelled_retries': 0,
            'avg_attempts': 0.0,
            'avg_success_time': 0.0
        }
        
        # Background processing
        self.is_running = True
        self.orchestrator_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.orchestrator_thread.start()
        
        # Thread safety
        self.lock = threading.RLock()
    
    def create_retry_context(self, analytics_id: str, data: Dict[str, Any], 
                           priority: RetryPriority = RetryPriority.NORMAL,
                           max_attempts: int = 10,
                           strategy: Optional[RetryStrategy] = None) -> str:
        """
        Create new retry context
        
        Args:
            analytics_id: Unique identifier for analytics operation
            data: Original analytics data
            priority: Retry priority level
            max_attempts: Maximum retry attempts
            strategy: Specific retry strategy (auto-selected if None)
            
        Returns:
            Context ID for tracking
        """
        with self.lock:
            # Auto-select strategy if not specified
            if strategy is None:
                context_info = {
                    'failure_pattern': FailurePattern.UNKNOWN,
                    'urgency': 'high' if priority in [RetryPriority.CRITICAL, RetryPriority.HIGH] else 'normal'
                }
                strategy = self.strategy_engine.recommend_strategy(context_info)
            
            # Create retry context
            context = RetryContext(
                analytics_id=analytics_id,
                original_data=data,
                current_strategy=strategy,
                priority=priority,
                max_attempts=max_attempts
            )
            
            # Store context
            self.active_retries[analytics_id] = context
            
            # Add to priority queue
            self.retry_queue[priority].append(analytics_id)
            
            # Update statistics
            self.stats['total_retries'] += 1
            
            self.logger.info(f"Created retry context for {analytics_id} with strategy {strategy.value}")
            return analytics_id
    
    def execute_retry(self, analytics_id: str, operation: Callable[[Dict[str, Any]], Any]) -> bool:
        """
        Execute retry operation with intelligent delay
        
        Args:
            analytics_id: Analytics ID to retry
            operation: Operation to execute
            
        Returns:
            True if successful, False if should continue retrying
        """
        with self.lock:
            context = self.active_retries.get(analytics_id)
            if not context or context.status != RetryStatus.PENDING:
                return False
            
            # Check max attempts
            if context.get_attempt_count() >= context.max_attempts:
                self._complete_retry(analytics_id, False, "Max attempts exceeded")
                return False
            
            # Update context for retry calculation
            retry_context = {
                'success_rate': context.get_success_rate(),
                'recent_failures': context.get_recent_failures(),
                'failure_pattern': context.failure_pattern,
                'failure_count': context.get_attempt_count(),
                'time_since_last_attempt': (
                    (datetime.now() - context.last_attempt).total_seconds()
                    if context.last_attempt else 0
                ),
                'urgency': 'high' if context.priority in [RetryPriority.CRITICAL, RetryPriority.HIGH] else 'normal'
            }
            
            # Calculate delay
            attempt_number = context.get_attempt_count()
            delay = self.strategy_engine.calculate_delay(
                context.current_strategy, 
                attempt_number, 
                retry_context
            )
            
            # Wait for delay
            if delay > 0:
                time.sleep(delay)
            
            # Create attempt record
            attempt = RetryAttempt(
                analytics_id=analytics_id,
                strategy=context.current_strategy,
                attempt_number=attempt_number,
                delay_used=delay,
                priority=context.priority
            )
            
            # Execute operation
            start_time = time.time()
            success = False
            error_info = None
            
            try:
                context.status = RetryStatus.ACTIVE
                result = operation(context.original_data)
                success = True
                self.logger.debug(f"Retry attempt {attempt_number} for {analytics_id} succeeded")
                
            except Exception as e:
                error_info = str(e)
                self.logger.debug(f"Retry attempt {attempt_number} for {analytics_id} failed: {error_info}")
            
            finally:
                processing_time = time.time() - start_time
                
                # Update attempt record
                attempt.success = success
                attempt.error_info = error_info
                attempt.processing_time = processing_time
                
                # Add to history
                context.failure_history.append(attempt)
                context.last_attempt = datetime.now()
                context.status = RetryStatus.PENDING if not success else RetryStatus.COMPLETED
                
                # Update strategy performance
                self.strategy_engine.update_strategy_performance(
                    context.current_strategy, success, delay, processing_time
                )
                
                # Complete retry if successful
                if success:
                    self._complete_retry(analytics_id, True, "Operation succeeded")
                    return True
                
                # Analyze failure pattern
                self._analyze_failure_pattern(context)
                
                # Consider strategy adaptation
                self._adapt_strategy(context)
            
            return False
    
    def _analyze_failure_pattern(self, context: RetryContext):
        """Analyze failure patterns to optimize strategy"""
        if len(context.failure_history) < 3:
            return
        
        recent_attempts = context.failure_history[-3:]
        time_diffs = []
        
        for i in range(1, len(recent_attempts)):
            time_diff = (recent_attempts[i].timestamp - recent_attempts[i-1].timestamp).total_seconds()
            time_diffs.append(time_diff)
        
        # Detect patterns
        if all(not attempt.success for attempt in recent_attempts):
            if len(set(time_diffs)) == 1:
                context.failure_pattern = FailurePattern.PERIODIC
            elif all(diff > 60 for diff in time_diffs):  # Long intervals
                context.failure_pattern = FailurePattern.PERSISTENT
            else:
                context.failure_pattern = FailurePattern.CASCADING
        else:
            context.failure_pattern = FailurePattern.TRANSIENT
    
    def _adapt_strategy(self, context: RetryContext):
        """Adapt retry strategy based on performance"""
        # Get current strategy performance
        stats = self.strategy_engine.get_strategy_stats(context.current_strategy)
        
        # If strategy is performing poorly, consider switching
        if (stats['success_rate'] < 0.3 and 
            stats['sample_count'] > 3 and 
            context.get_attempt_count() > 2):
            
            # Recommend new strategy
            adaptation_context = {
                'failure_pattern': context.failure_pattern,
                'urgency': 'high' if context.priority in [RetryPriority.CRITICAL, RetryPriority.HIGH] else 'normal'
            }
            
            new_strategy = self.strategy_engine.recommend_strategy(adaptation_context)
            
            if new_strategy != context.current_strategy:
                self.logger.info(f"Adapting strategy for {context.analytics_id}: {context.current_strategy.value} → {new_strategy.value}")
                context.current_strategy = new_strategy
    
    def _complete_retry(self, analytics_id: str, success: bool, reason: str):
        """Complete retry operation"""
        context = self.active_retries.pop(analytics_id, None)
        if not context:
            return
        
        # Update status
        context.status = RetryStatus.COMPLETED if success else RetryStatus.FAILED
        
        # Add to completed retries
        self.completed_retries.append(context)
        
        # Update statistics
        if success:
            self.stats['successful_retries'] += 1
        else:
            self.stats['failed_retries'] += 1
        
        # Update averages
        total_completed = self.stats['successful_retries'] + self.stats['failed_retries']
        if total_completed > 0:
            total_attempts = sum(ctx.get_attempt_count() for ctx in self.completed_retries)
            self.stats['avg_attempts'] = total_attempts / total_completed
        
        self.logger.info(f"Completed retry for {analytics_id}: {reason}")
    
    def _orchestration_loop(self):
        """Background orchestration loop"""
        while self.is_running:
            try:
                # Process priority queues
                for priority in RetryPriority:
                    if len(self.active_retries) >= self.max_concurrent_retries:
                        break
                    
                    queue = self.retry_queue[priority]
                    if queue:
                        analytics_id = queue.pop(0)
                        if analytics_id in self.active_retries:
                            context = self.active_retries[analytics_id]
                            if context.status == RetryStatus.PENDING:
                                # Ready for retry
                                pass
                
                # Cleanup expired contexts
                self._cleanup_expired_contexts()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                time.sleep(1)
    
    def _cleanup_expired_contexts(self):
        """Clean up expired retry contexts"""
        with self.lock:
            expired_contexts = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for analytics_id, context in self.active_retries.items():
                if context.created_at < cutoff_time:
                    expired_contexts.append(analytics_id)
            
            for analytics_id in expired_contexts:
                self._complete_retry(analytics_id, False, "Context expired")
    
    def get_retry_status(self, analytics_id: str) -> Optional[Dict[str, Any]]:
        """Get retry status for analytics ID"""
        with self.lock:
            context = self.active_retries.get(analytics_id)
            if not context:
                return None
            
            return {
                'analytics_id': analytics_id,
                'status': context.status.value,
                'strategy': context.current_strategy.value,
                'priority': context.priority.value,
                'attempt_count': context.get_attempt_count(),
                'max_attempts': context.max_attempts,
                'success_rate': context.get_success_rate(),
                'failure_pattern': context.failure_pattern.value,
                'created_at': context.created_at.isoformat(),
                'last_attempt': context.last_attempt.isoformat() if context.last_attempt else None
            }
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        with self.lock:
            return {
                'active_retries': len(self.active_retries),
                'completed_retries': len(self.completed_retries),
                'queue_sizes': {
                    priority.value: len(self.retry_queue[priority])
                    for priority in RetryPriority
                },
                'performance_stats': self.stats.copy(),
                'strategy_performance': {
                    strategy.value: self.strategy_engine.get_strategy_stats(strategy)
                    for strategy in RetryStrategy
                }
            }
    
    def cancel_retry(self, analytics_id: str) -> bool:
        """Cancel active retry"""
        with self.lock:
            context = self.active_retries.get(analytics_id)
            if context:
                context.status = RetryStatus.CANCELLED
                self._complete_retry(analytics_id, False, "Cancelled by user")
                self.stats['cancelled_retries'] += 1
                return True
            return False
    
    def shutdown(self):
        """Shutdown orchestrator"""
        self.is_running = False
        if self.orchestrator_thread and self.orchestrator_thread.is_alive():
            self.orchestrator_thread.join(timeout=5)
        self.logger.info("Quantum retry orchestrator shutdown")


# Global orchestrator instance
quantum_retry_orchestrator = QuantumRetryOrchestrator()

# Export
__all__ = [
    'RetryPriority', 'RetryStatus', 'RetryAttempt', 'RetryContext',
    'QuantumRetryOrchestrator', 'quantum_retry_orchestrator'
]