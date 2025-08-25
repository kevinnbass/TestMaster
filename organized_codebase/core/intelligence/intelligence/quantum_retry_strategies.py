"""
Quantum Retry Strategies
========================

Advanced retry strategies with machine learning, adaptive algorithms,
and predictive failure detection. Extracted from 1,191-line archive component.

Core retry algorithms including exponential, fibonacci, quantum, and neural strategies.
"""

import math
import random
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Quantum retry strategy types"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"
    QUANTUM = "quantum"
    PREDICTIVE = "predictive"
    NEURAL = "neural"


class FailurePattern(Enum):
    """Failure pattern types for analysis"""
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    CASCADING = "cascading"
    PERIODIC = "periodic"
    RANDOM = "random"
    UNKNOWN = "unknown"


@dataclass
class RetryParams:
    """Retry strategy parameters"""
    base_delay: float = 1.0
    max_delay: float = 300.0
    jitter_factor: float = 0.1
    multiplier: float = 2.0
    max_attempts: int = 10
    quantum_factor: float = 1.618  # Golden ratio
    neural_learning_rate: float = 0.01


class QuantumRetryStrategies:
    """Advanced retry strategy implementation with quantum-level intelligence"""
    
    def __init__(self, params: Optional[RetryParams] = None):
        self.params = params or RetryParams()
        self.logger = logging.getLogger(__name__)
        
        # Strategy implementations
        self.strategies = {
            RetryStrategy.EXPONENTIAL: self._exponential_strategy,
            RetryStrategy.LINEAR: self._linear_strategy,
            RetryStrategy.FIBONACCI: self._fibonacci_strategy,
            RetryStrategy.ADAPTIVE: self._adaptive_strategy,
            RetryStrategy.QUANTUM: self._quantum_strategy,
            RetryStrategy.PREDICTIVE: self._predictive_strategy,
            RetryStrategy.NEURAL: self._neural_strategy
        }
        
        # Performance tracking
        self.strategy_performance = {strategy: [] for strategy in RetryStrategy}
        self.fibonacci_cache = {0: 0, 1: 1}
        
        # Neural network weights (simplified)
        self.neural_weights = {
            'failure_count': 0.3,
            'time_since_last': 0.2,
            'success_rate': 0.4,
            'error_type': 0.1
        }
    
    def calculate_delay(self, strategy: RetryStrategy, attempt: int, 
                       context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate retry delay using specified strategy
        
        Args:
            strategy: Retry strategy to use
            attempt: Current attempt number (0-based)
            context: Additional context for adaptive strategies
            
        Returns:
            Delay in seconds
        """
        try:
            if strategy not in self.strategies:
                strategy = RetryStrategy.EXPONENTIAL
            
            delay = self.strategies[strategy](attempt, context or {})
            
            # Apply jitter to prevent thundering herd
            jitter = delay * self.params.jitter_factor * (random.random() - 0.5)
            delay_with_jitter = max(0, delay + jitter)
            
            # Respect maximum delay
            final_delay = min(delay_with_jitter, self.params.max_delay)
            
            self.logger.debug(f"Strategy {strategy.value}, attempt {attempt}: {final_delay:.2f}s")
            return final_delay
            
        except Exception as e:
            self.logger.error(f"Delay calculation failed: {e}")
            return self.params.base_delay
    
    def _exponential_strategy(self, attempt: int, context: Dict[str, Any]) -> float:
        """Exponential backoff with configurable multiplier"""
        return self.params.base_delay * (self.params.multiplier ** attempt)
    
    def _linear_strategy(self, attempt: int, context: Dict[str, Any]) -> float:
        """Linear backoff strategy"""
        return self.params.base_delay * (1 + attempt)
    
    def _fibonacci_strategy(self, attempt: int, context: Dict[str, Any]) -> float:
        """Fibonacci-based backoff for natural scaling"""
        fib_value = self._get_fibonacci(attempt + 1)
        return self.params.base_delay * fib_value
    
    def _adaptive_strategy(self, attempt: int, context: Dict[str, Any]) -> float:
        """Adaptive strategy based on historical success rates"""
        success_rate = context.get('success_rate', 0.5)
        recent_failures = context.get('recent_failures', 0)
        
        # Adapt based on success rate
        if success_rate > 0.8:
            # High success rate - be more aggressive
            multiplier = self.params.multiplier * 0.8
        elif success_rate < 0.3:
            # Low success rate - be more conservative
            multiplier = self.params.multiplier * 1.5
        else:
            multiplier = self.params.multiplier
        
        # Account for recent failures
        failure_factor = 1 + (recent_failures * 0.1)
        
        return self.params.base_delay * (multiplier ** attempt) * failure_factor
    
    def _quantum_strategy(self, attempt: int, context: Dict[str, Any]) -> float:
        """Quantum-inspired strategy using golden ratio and wave functions"""
        # Use golden ratio for natural scaling
        quantum_factor = self.params.quantum_factor ** attempt
        
        # Apply wave function for oscillating behavior
        wave_factor = 1 + 0.2 * math.sin(attempt * math.pi / 4)
        
        # Quantum tunneling effect - occasional short delays
        if random.random() < 0.1:  # 10% chance
            return self.params.base_delay * 0.1
        
        return self.params.base_delay * quantum_factor * wave_factor
    
    def _predictive_strategy(self, attempt: int, context: Dict[str, Any]) -> float:
        """Predictive strategy based on failure patterns"""
        failure_pattern = context.get('failure_pattern', FailurePattern.UNKNOWN)
        time_of_day = datetime.now().hour
        
        base_delay = self.params.base_delay * (self.params.multiplier ** attempt)
        
        # Adjust based on failure pattern
        pattern_factors = {
            FailurePattern.TRANSIENT: 0.5,  # Quick recovery expected
            FailurePattern.PERSISTENT: 2.0,  # Slower recovery needed
            FailurePattern.CASCADING: 1.5,   # Medium recovery time
            FailurePattern.PERIODIC: 0.8,    # Pattern-based recovery
            FailurePattern.RANDOM: 1.0,      # Standard recovery
            FailurePattern.UNKNOWN: 1.2      # Conservative approach
        }
        
        pattern_factor = pattern_factors.get(failure_pattern, 1.0)
        
        # Time-based adjustments (avoid peak hours)
        if 9 <= time_of_day <= 17:  # Business hours
            time_factor = 1.2
        elif 0 <= time_of_day <= 6:  # Late night
            time_factor = 0.8
        else:
            time_factor = 1.0
        
        return base_delay * pattern_factor * time_factor
    
    def _neural_strategy(self, attempt: int, context: Dict[str, Any]) -> float:
        """Neural network-inspired strategy with learning"""
        # Extract features
        failure_count = context.get('failure_count', 0)
        time_since_last = context.get('time_since_last_attempt', 0)
        success_rate = context.get('success_rate', 0.5)
        error_type_weight = context.get('error_type_weight', 0.5)
        
        # Normalize features
        normalized_features = {
            'failure_count': min(failure_count / 10.0, 1.0),
            'time_since_last': min(time_since_last / 3600.0, 1.0),  # Max 1 hour
            'success_rate': success_rate,
            'error_type': error_type_weight
        }
        
        # Calculate weighted sum
        neural_output = sum(
            self.neural_weights[feature] * value
            for feature, value in normalized_features.items()
        )
        
        # Apply activation function (sigmoid)
        activation = 1 / (1 + math.exp(-neural_output))
        
        # Convert to delay
        base_delay = self.params.base_delay * (self.params.multiplier ** attempt)
        neural_factor = 0.5 + activation  # Range: 0.5 to 1.5
        
        return base_delay * neural_factor
    
    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number with caching"""
        if n in self.fibonacci_cache:
            return self.fibonacci_cache[n]
        
        if n < 2:
            return n
        
        # Calculate iteratively to avoid recursion depth issues
        a, b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b
            self.fibonacci_cache[i] = b
        
        return b
    
    def update_strategy_performance(self, strategy: RetryStrategy, success: bool, 
                                  delay_used: float, processing_time: float):
        """Update performance metrics for strategy optimization"""
        performance_data = {
            'success': success,
            'delay_used': delay_used,
            'processing_time': processing_time,
            'timestamp': datetime.now()
        }
        
        self.strategy_performance[strategy].append(performance_data)
        
        # Keep only recent performance data
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.strategy_performance[strategy] = [
            p for p in self.strategy_performance[strategy]
            if p['timestamp'] > cutoff_time
        ]
    
    def get_strategy_stats(self, strategy: RetryStrategy) -> Dict[str, Any]:
        """Get performance statistics for a strategy"""
        performance_data = self.strategy_performance[strategy]
        
        if not performance_data:
            return {'success_rate': 0.0, 'avg_delay': 0.0, 'sample_count': 0}
        
        successes = [p['success'] for p in performance_data]
        delays = [p['delay_used'] for p in performance_data]
        
        return {
            'success_rate': sum(successes) / len(successes),
            'avg_delay': statistics.mean(delays),
            'median_delay': statistics.median(delays),
            'sample_count': len(performance_data),
            'last_24h_count': len(performance_data)
        }
    
    def recommend_strategy(self, context: Dict[str, Any]) -> RetryStrategy:
        """Recommend optimal strategy based on context and performance"""
        failure_pattern = context.get('failure_pattern', FailurePattern.UNKNOWN)
        urgency = context.get('urgency', 'normal')
        
        # High urgency - use faster strategies
        if urgency == 'high':
            return RetryStrategy.LINEAR
        
        # Pattern-based recommendations
        pattern_strategies = {
            FailurePattern.TRANSIENT: RetryStrategy.EXPONENTIAL,
            FailurePattern.PERSISTENT: RetryStrategy.FIBONACCI,
            FailurePattern.CASCADING: RetryStrategy.ADAPTIVE,
            FailurePattern.PERIODIC: RetryStrategy.PREDICTIVE,
            FailurePattern.RANDOM: RetryStrategy.QUANTUM,
            FailurePattern.UNKNOWN: RetryStrategy.NEURAL
        }
        
        recommended = pattern_strategies.get(failure_pattern, RetryStrategy.ADAPTIVE)
        
        # Check performance and switch if needed
        stats = self.get_strategy_stats(recommended)
        if stats['success_rate'] < 0.3 and stats['sample_count'] > 5:
            # Strategy performing poorly, try neural approach
            return RetryStrategy.NEURAL
        
        return recommended
    
    def update_neural_weights(self, features: Dict[str, float], success: bool):
        """Update neural network weights based on outcome"""
        learning_rate = self.params.neural_learning_rate
        target = 1.0 if success else 0.0
        
        # Simple gradient descent update
        for feature, value in features.items():
            if feature in self.neural_weights:
                prediction = self.neural_weights[feature] * value
                error = target - prediction
                self.neural_weights[feature] += learning_rate * error * value
                
                # Keep weights in reasonable range
                self.neural_weights[feature] = max(-2.0, min(2.0, self.neural_weights[feature]))


# Global instance for easy access
quantum_retry_strategies = QuantumRetryStrategies()

# Export
__all__ = [
    'RetryStrategy', 'FailurePattern', 'RetryParams',
    'QuantumRetryStrategies', 'quantum_retry_strategies'
]