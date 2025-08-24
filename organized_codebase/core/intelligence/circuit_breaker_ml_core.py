"""
Advanced ML Circuit Breaker System
==================================
"""Core Module - Split from circuit_breaker_ml.py"""


import logging
import time
import threading
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import functools

# ML imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """ML-enhanced circuit states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failures detected, blocking calls
    HALF_OPEN = "half_open" # Testing recovery
    ML_MONITORING = "ml_monitoring"  # ML-driven monitoring
    PREDICTIVE_OPEN = "predictive_open"  # ML predicted failure

class FailureType(Enum):
    """ML-classified failure types"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    VALIDATION = "validation"
    RESOURCE = "resource"
    UNAVAILABLE = "unavailable"
    ML_PREDICTED = "ml_predicted"
    ANOMALY_DETECTED = "anomaly_detected"

class MLStrategy(Enum):
    """ML strategies for circuit breaker"""
    FAILURE_PREDICTION = "failure_prediction"
    ADAPTIVE_THRESHOLDS = "adaptive_thresholds"
    RECOVERY_OPTIMIZATION = "recovery_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"

@dataclass
class MLCircuitBreakerConfig:
    """ML-enhanced circuit breaker configuration"""
    failure_threshold: int = 5
    timeout_seconds: float = 30.0
    recovery_timeout: int = 60
    success_threshold: int = 3
    max_concurrent_calls: int = 10
    
    # ML-specific configuration
    ml_enabled: bool = True
    prediction_horizon_minutes: int = 15
    anomaly_threshold: float = 0.8
    adaptive_threshold_enabled: bool = True
    min_training_samples: int = 20
    model_retrain_interval_hours: int = 6
    ml_confidence_threshold: float = 0.7

@dataclass
class MLFailureRecord:
    """ML-enhanced failure record"""
    failure_type: FailureType
    error: str
    timestamp: datetime
    ml_features: List[float] = field(default_factory=list)
    anomaly_score: float = 0.0
    prediction_confidence: float = 0.0
    failure_probability: float = 0.0
    context_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MLCircuitMetrics:
    """ML-driven circuit metrics"""
    circuit_name: str
    timestamp: datetime
    success_rate: float
    failure_rate: float
    average_response_time: float
    concurrent_calls: int
    ml_health_score: float = 0.0
    anomaly_score: float = 0.0
    predicted_failure_probability: float = 0.0
    adaptive_threshold: float = 0.0

class MLCircuitBreaker:
    """
    ML-enhanced circuit breaker with predictive failure detection,
    adaptive thresholds, and intelligent recovery strategies.
    """
    
    def __init__(self, name: str, config: MLCircuitBreakerConfig = None):
        """
        Initialize ML circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: ML-enhanced configuration
        """
        self.name = name
        self.config = config or MLCircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.failure_count = 0
        self.success_count = 0
        
        # ML models
        self.failure_predictor: Optional[Any] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.pattern_classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        
        # ML data tracking
        self.ml_failures: deque = deque(maxlen=1000)
        self.ml_metrics: deque = deque(maxlen=1000)
        self.call_history: deque = deque(maxlen=1000)
        self.feature_history: deque = deque(maxlen=1000)
        
        # Adaptive thresholds
        self.adaptive_failure_threshold = self.config.failure_threshold
        self.adaptive_timeout = self.config.timeout_seconds
        self.adaptive_recovery_timeout = self.config.recovery_timeout
        
        # Concurrency control
        self.active_calls = 0
        self.call_lock = threading.Lock()
        
        # ML statistics
        self.ml_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'blocked_calls': 0,
            'ml_predictions': 0,
            'ml_prevented_failures': 0,
            'anomalies_detected': 0,
            'adaptive_adjustments': 0,
            'model_accuracy': 0.0,
            'last_model_training': None,
            'state_changes': 0
        }
        
        # ML callbacks
        self.ml_prediction_callbacks: List[Callable] = []
        self.state_change_callbacks: List[Callable] = []
        self.failure_callbacks: List[Callable] = []
        
        # Background ML tasks
        self.ml_task: Optional[asyncio.Task] = None
        self.is_ml_active = False
        
        logger.info(f"ML Circuit breaker '{name}' initialized")
    
    def __call__(self, func):
        """Decorator to wrap functions with ML circuit breaker"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with ML-enhanced circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            MLCircuitBreakerOpenException: When circuit is open
        """
        # Pre-execution ML analysis
        ml_prediction = self._predict_failure_probability()
        
        with self.call_lock:
            self.ml_stats['total_calls'] += 1
            
            # ML-driven state checking
            if self._should_block_ml_enhanced():
                self.ml_stats['blocked_calls'] += 1
                raise MLCircuitBreakerOpenException(
                    f"ML Circuit breaker '{self.name}' blocked call. "
                    f"State: {self.state.value}, ML prediction: {ml_prediction:.3f}"
                )
            
            # Check concurrent call limit with ML adjustment
            max_calls = self._get_adaptive_max_calls()
            if self.active_calls >= max_calls:
                self.ml_stats['blocked_calls'] += 1
                raise MLCircuitBreakerOpenException(
                    f"ML Circuit breaker '{self.name}' exceeded adaptive call limit: {max_calls}"
                )
            
            self.active_calls += 1
        
        call_start = time.time()
        success = False
        error = None
        ml_features = self._extract_call_features()
        
        try:
            # Execute with adaptive timeout
            adaptive_timeout = self._get_adaptive_timeout()
            result = self._execute_with_ml_timeout(func, adaptive_timeout, *args, **kwargs)
            success = True
            self._record_ml_success(call_start, ml_features)
            return result
            
        except TimeoutError as e:
            error = str(e)
            self._record_ml_failure(FailureType.TIMEOUT, error, call_start, ml_features)
            raise
            
        except Exception as e:
            error = str(e)
            failure_type = self._classify_failure_ml(e, ml_features)
            self._record_ml_failure(failure_type, error, call_start, ml_features)
            raise
            
        finally:
            with self.call_lock:
                self.active_calls -= 1
            
            # Record comprehensive call history
            self._record_ml_call_history(call_start, success, error, ml_features)
    
    def _should_block_ml_enhanced(self) -> bool:
        """ML-enhanced decision on whether to block calls"""
        # Standard state checking
        if self.state == CircuitState.OPEN:
            if self._should_attempt_ml_recovery():
                self._transition_to_half_open()
                return False
            return True
        
        # ML predictive blocking
        if self.state == CircuitState.PREDICTIVE_OPEN:
            return True
        
        # ML-driven proactive blocking
        if self.config.ml_enabled:
            failure_probability = self._predict_failure_probability()
            if failure_probability > self.config.ml_confidence_threshold:
                self._transition_to_predictive_open()
                return True
            
            # Anomaly-based blocking
            anomaly_score = self._calculate_current_anomaly_score()
            if anomaly_score > self.config.anomaly_threshold:
                self.ml_stats['anomalies_detected'] += 1
                return True
        
        return False
    
    def _predict_failure_probability(self) -> float:
        """Predict failure probability using ML"""
        try:
            if not self.config.ml_enabled or not self.failure_predictor:
                return 0.0
            
            # Extract current features
            features = self._extract_prediction_features()
            if not features or len(features) < 5:
                return 0.0
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Make prediction
            if hasattr(self.failure_predictor, 'predict_proba'):
                prediction = self.failure_predictor.predict_proba(features_scaled)[0]
                failure_probability = prediction[1] if len(prediction) > 1 else prediction[0]
            else:
                prediction = self.failure_predictor.predict(features_scaled)[0]
                failure_probability = max(0, min(1, prediction))
            
            self.ml_stats['ml_predictions'] += 1
            
            # Trigger ML prediction callbacks
            for callback in self.ml_prediction_callbacks:
                try:
                    callback(self, failure_probability, features)
                except Exception as e:
                    logger.error(f"ML prediction callback error: {e}")
            
            return failure_probability
            
        except Exception as e:
            logger.debug(f"ML failure prediction error: {e}")
            return 0.0
    
    def _extract_prediction_features(self) -> List[float]:
        """Extract features for ML prediction"""
        try:
            features = []
            
            # Circuit state features
            features.append(float(self.failure_count))
            features.append(float(self.success_count))
            features.append(float(self.active_calls))
            
            # Time-based features
            current_time = datetime.now()
            if self.last_failure_time:
                time_since_failure = (current_time - self.last_failure_time).total_seconds()
                features.append(time_since_failure)
            else:
                features.append(3600.0)  # 1 hour default
            
            # Recent performance features
            recent_calls = list(self.call_history)[-10:]
            if recent_calls:
                success_rate = sum(1 for call in recent_calls if call.get('success', False)) / len(recent_calls)
                avg_duration = np.mean([call.get('duration', 0) for call in recent_calls])
                features.extend([success_rate, avg_duration])
            else:
                features.extend([1.0, 0.1])  # Default values
            
            # Historical failure pattern features
            recent_failures = [f for f in self.ml_failures if 
                             (current_time - f.timestamp).total_seconds() < 3600]
            features.append(float(len(recent_failures)))
            
            # Adaptive threshold features
            features.append(self.adaptive_failure_threshold)
            features.append(self.adaptive_timeout)
            
            # System context features (time of day, etc.)
            features.append(float(current_time.hour))
            features.append(float(current_time.weekday()))
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return []
    
    def _extract_call_features(self) -> List[float]:
        """Extract features for individual call analysis"""
        try:
            features = []
            
            # Current system state
            features.append(float(self.active_calls))
            features.append(float(self.failure_count))
            features.append(float(self.success_count))
            
            # Time context
            current_time = datetime.now()
            features.append(float(current_time.hour))
            features.append(float(current_time.minute))
            
            return features
            
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _calculate_current_anomaly_score(self) -> float:
        """Calculate current anomaly score"""
        try:
            if not self.anomaly_detector or not self.config.ml_enabled:
                return 0.0
            
            features = self._extract_prediction_features()
            if not features:
                return 0.0
            
            # Calculate anomaly score
            if self.scaler:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            # Convert to 0-1 range (negative scores indicate anomalies)
            normalized_score = max(0, 1 - (anomaly_score + 1) / 2)
            
            return normalized_score
            
        except Exception as e:
            logger.debug(f"Anomaly score calculation error: {e}")
            return 0.0
    
    def _classify_failure_ml(self, exception: Exception, ml_features: List[float]) -> FailureType:
        """Classify failure type using ML"""
        try:
            if not self.pattern_classifier or not self.config.ml_enabled:
                return FailureType.EXCEPTION
            
            # Extract failure features
            failure_features = ml_features + [
                len(str(exception)),
                1.0 if 'timeout' in str(exception).lower() else 0.0,
                1.0 if 'connection' in str(exception).lower() else 0.0,
                1.0 if 'resource' in str(exception).lower() else 0.0
            ]
            
            # Ensure consistent feature count
            while len(failure_features) < 10:
                failure_features.append(0.0)
            
            failure_features = failure_features[:10]
            
            # Classify failure
            if self.scaler:
                features_scaled = self.scaler.transform([failure_features])
            else:
                features_scaled = [failure_features]
            
            failure_class = self.pattern_classifier.predict(features_scaled)[0]
            
            # Map to FailureType
            failure_mapping = {
                0: FailureType.TIMEOUT,
                1: FailureType.EXCEPTION,
                2: FailureType.RESOURCE,
                3: FailureType.UNAVAILABLE,
                4: FailureType.VALIDATION
            }
            
            return failure_mapping.get(failure_class, FailureType.EXCEPTION)
            
        except Exception as e:
            logger.debug(f"ML failure classification error: {e}")