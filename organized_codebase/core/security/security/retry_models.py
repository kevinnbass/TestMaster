#!/usr/bin/env python3
"""
Quantum Retry Engine Models and Enums
=====================================

Data models and enumerations for the quantum retry engine system.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class RetryStrategy(Enum):
    """Quantum retry strategy types."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"
    QUANTUM = "quantum"
    PREDICTIVE = "predictive"
    NEURAL = "neural"


class FailurePattern(Enum):
    """Failure pattern types for analysis."""
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    CASCADING = "cascading"
    PERIODIC = "periodic"
    RANDOM = "random"
    UNKNOWN = "unknown"


class RetryPriority(Enum):
    """Retry priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class QuantumRetryAttempt:
    """Individual quantum retry attempt record."""
    attempt_id: str
    operation_id: str
    strategy: RetryStrategy
    attempt_number: int
    timestamp: datetime
    delay_used: float
    priority: RetryPriority
    success: bool
    error_info: Optional[str] = None
    processing_time: float = 0.0
    quantum_state: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'attempt_id': self.attempt_id,
            'operation_id': self.operation_id,
            'strategy': self.strategy.value,
            'attempt_number': self.attempt_number,
            'timestamp': self.timestamp.isoformat(),
            'delay_used': self.delay_used,
            'priority': self.priority.value,
            'success': self.success,
            'error_info': self.error_info,
            'processing_time': self.processing_time,
            'quantum_state': self.quantum_state or {}
        }


@dataclass
class QuantumRetryContext:
    """Comprehensive quantum retry context."""
    operation_id: str
    operation_data: Dict[str, Any]
    failure_history: List[QuantumRetryAttempt]
    current_strategy: RetryStrategy
    priority: RetryPriority
    max_attempts: int
    created_at: datetime
    last_attempt: Optional[datetime] = None
    failure_pattern: FailurePattern = FailurePattern.UNKNOWN
    success_probability: float = 0.5
    predicted_next_success: Optional[datetime] = None
    quantum_entanglement: Dict[str, float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'operation_id': self.operation_id,
            'operation_data': self.operation_data,
            'failure_history': [attempt.to_dict() for attempt in self.failure_history],
            'current_strategy': self.current_strategy.value,
            'priority': self.priority.value,
            'max_attempts': self.max_attempts,
            'created_at': self.created_at.isoformat(),
            'last_attempt': self.last_attempt.isoformat() if self.last_attempt else None,
            'failure_pattern': self.failure_pattern.value,
            'success_probability': self.success_probability,
            'predicted_next_success': self.predicted_next_success.isoformat() if self.predicted_next_success else None,
            'quantum_entanglement': self.quantum_entanglement or {}
        }


@dataclass
class QuantumRetryConfig:
    """Configuration for quantum retry engine."""
    db_path: str = "data/quantum_retry.db"
    quantum_interval: float = 1.0
    max_concurrent_operations: int = 10
    default_max_attempts: int = 5
    default_priority: RetryPriority = RetryPriority.NORMAL
    enable_predictive_analysis: bool = True
    enable_pattern_detection: bool = True
    enable_quantum_entanglement: bool = True
    learning_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'db_path': self.db_path,
            'quantum_interval': self.quantum_interval,
            'max_concurrent_operations': self.max_concurrent_operations,
            'default_max_attempts': self.default_max_attempts,
            'default_priority': self.default_priority.value,
            'enable_predictive_analysis': self.enable_predictive_analysis,
            'enable_pattern_detection': self.enable_pattern_detection,
            'enable_quantum_entanglement': self.enable_quantum_entanglement,
            'learning_enabled': self.learning_enabled
        }


@dataclass
class RetryStatistics:
    """Statistics for retry operations."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_attempts: float = 0.0
    average_processing_time: float = 0.0
    strategy_success_rates: Dict[RetryStrategy, float] = None
    pattern_distribution: Dict[FailurePattern, int] = None
    priority_distribution: Dict[RetryPriority, int] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.strategy_success_rates is None:
            self.strategy_success_rates = {}
        if self.pattern_distribution is None:
            self.pattern_distribution = {}
        if self.priority_distribution is None:
            self.priority_distribution = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'average_attempts': self.average_attempts,
            'average_processing_time': self.average_processing_time,
            'strategy_success_rates': {k.value: v for k, v in self.strategy_success_rates.items()},
            'pattern_distribution': {k.value: v for k, v in self.pattern_distribution.items()},
            'priority_distribution': {k.value: v for k, v in self.priority_distribution.items()}
        }
