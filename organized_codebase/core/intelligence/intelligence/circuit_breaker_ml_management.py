"""
Advanced ML Circuit Breaker System
==================================
"""Management Module - Split from circuit_breaker_ml.py"""


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


            return FailureType.EXCEPTION
    
    def _get_adaptive_max_calls(self) -> int:
        """Get adaptive maximum concurrent calls using ML"""
        if not self.config.adaptive_threshold_enabled:
            return self.config.max_concurrent_calls
        
        try:
            # Adjust based on recent performance
            recent_calls = list(self.call_history)[-20:]
            if recent_calls:
                success_rate = sum(1 for call in recent_calls if call.get('success', False)) / len(recent_calls)
                
                if success_rate > 0.9:
                    # High success rate, allow more calls
                    return min(self.config.max_concurrent_calls * 2, 20)
                elif success_rate < 0.7:
                    # Low success rate, reduce calls
                    return max(self.config.max_concurrent_calls // 2, 1)
            
            return self.config.max_concurrent_calls
            
        except Exception:
            return self.config.max_concurrent_calls
    
    def _get_adaptive_timeout(self) -> float:
        """Get adaptive timeout using ML analysis"""
        if not self.config.adaptive_threshold_enabled:
            return self.config.timeout_seconds
        
        try:
            # Adjust based on recent response times
            recent_calls = list(self.call_history)[-10:]
            if recent_calls:
                durations = [call.get('duration', 0) for call in recent_calls 
                            if call.get('success', False)]
                
                if durations:
                    avg_duration = np.mean(durations)
                    std_duration = np.std(durations)
                    
                    # Set timeout to mean + 2*std with some bounds
                    adaptive_timeout = avg_duration + 2 * std_duration
                    return max(min(adaptive_timeout, self.config.timeout_seconds * 3), 
                             self.config.timeout_seconds * 0.5)
            
            return self.config.timeout_seconds
            
        except Exception:
            return self.config.timeout_seconds
    
    def _execute_with_ml_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with ML-adaptive timeout"""
        if timeout <= 0:
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
        
        if finished.wait(timeout=timeout):
            if exception:
                raise exception
            return result
        else:
            raise TimeoutError(f"Function timed out after {timeout} seconds (ML adaptive)")
    
    def _record_ml_success(self, call_start: float, ml_features: List[float]):
        """Record successful call with ML analysis"""
        with self.call_lock:
            self.ml_stats['successful_calls'] += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset on success
            elif self.state == CircuitState.PREDICTIVE_OPEN:
                # Success in predictive open state, reassess
                self._reassess_predictive_state()
        
        # Record ML metrics
        self._record_ml_metrics(success=True, duration=time.time() - call_start)
    
    def _record_ml_failure(self, failure_type: FailureType, error: str, 
                          call_start: float, ml_features: List[float]):
        """Record failed call with ML analysis"""
        with self.call_lock:
            self.ml_stats['failed_calls'] += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            # Create ML failure record
            ml_failure = MLFailureRecord(
                failure_type=failure_type,
                error=error,
                timestamp=self.last_failure_time,
                ml_features=ml_features,
                anomaly_score=self._calculate_current_anomaly_score(),
                prediction_confidence=self._predict_failure_probability()
            )
            
            self.ml_failures.append(ml_failure)
            
            # Trigger ML failure callbacks
            for callback in self.failure_callbacks:
                try:
                    callback(self, ml_failure)
                except Exception as e:
                    logger.error(f"ML failure callback error: {e}")
            
            # ML-enhanced state transitions
            if self.state == CircuitState.CLOSED:
                adaptive_threshold = self._get_adaptive_failure_threshold()
                if self.failure_count >= adaptive_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.state == CircuitState.ML_MONITORING:
                # Transition based on ML analysis
                self._handle_ml_monitoring_failure(ml_failure)
        
        # Record ML metrics
        self._record_ml_metrics(success=False, duration=time.time() - call_start, 
                               failure_type=failure_type)
    
    def _get_adaptive_failure_threshold(self) -> int:
        """Get adaptive failure threshold using ML"""
        if not self.config.adaptive_threshold_enabled:
            return self.config.failure_threshold
        
        try:
            # Analyze recent failure patterns
            recent_failures = [f for f in self.ml_failures if 
                             (datetime.now() - f.timestamp).total_seconds() < 3600]
            
            if len(recent_failures) > 10:
                # High failure rate, lower threshold
                return max(self.config.failure_threshold // 2, 2)
            elif len(recent_failures) < 2:
                # Low failure rate, higher threshold
                return min(self.config.failure_threshold * 2, 10)
            
            return self.config.failure_threshold
            
        except Exception:
            return self.config.failure_threshold
    
    def _should_attempt_ml_recovery(self) -> bool:
        """ML-enhanced recovery decision"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        
        # Standard timeout check
        if time_since_failure < self.adaptive_recovery_timeout:
            return False
        
        # ML-enhanced recovery decision
        if self.config.ml_enabled:
            recovery_probability = self._predict_recovery_probability()
            return recovery_probability > self.config.ml_confidence_threshold
        
        return True
    
    def _predict_recovery_probability(self) -> float:
        """Predict probability of successful recovery"""
        try:
            # Simple heuristic based on recent trends
            recent_failures = [f for f in self.ml_failures if 
                             (datetime.now() - f.timestamp).total_seconds() < 1800]  # 30 minutes
            
            if not recent_failures:
                return 1.0
            
            # Check if failure rate is decreasing
            failure_times = [f.timestamp for f in recent_failures]
            failure_times.sort()
            
            if len(failure_times) >= 3:
                # Check trend in failure intervals
                intervals = [(failure_times[i] - failure_times[i-1]).total_seconds() 
                           for i in range(1, len(failure_times))]
                
                if intervals:
                    trend = np.polyfit(range(len(intervals)), intervals, 1)[0]
                    # Positive trend means increasing intervals (fewer failures)
                    return min(1.0, max(0.0, 0.5 + trend / 100))
            
            return 0.7  # Default recovery probability
            
        except Exception:
            return 0.7
    
    def _transition_to_predictive_open(self):
        """Transition to ML predictive open state"""
        if self.state != CircuitState.PREDICTIVE_OPEN:
            old_state = self.state
            self.state = CircuitState.PREDICTIVE_OPEN
            self.ml_stats['state_changes'] += 1
            self.ml_stats['ml_prevented_failures'] += 1
            
            logger.warning(f"ML Circuit breaker '{self.name}' transitioned to PREDICTIVE_OPEN "
                          f"(from {old_state.value}) based on ML prediction")
            self._notify_state_change()
    
    def _transition_to_open(self):
        """Transition to open state"""
        if self.state != CircuitState.OPEN:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.ml_stats['state_changes'] += 1
            
            logger.warning(f"ML Circuit breaker '{self.name}' OPENED "
                          f"(from {old_state.value}) - failures: {self.failure_count}")
            self._notify_state_change()
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        if self.state != CircuitState.HALF_OPEN:
            old_state = self.state
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self.ml_stats['state_changes'] += 1
            
            logger.info(f"ML Circuit breaker '{self.name}' HALF_OPENED "
                       f"(from {old_state.value}) for recovery testing")
            self._notify_state_change()
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        if self.state != CircuitState.CLOSED:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.ml_stats['state_changes'] += 1
            
            logger.info(f"ML Circuit breaker '{self.name}' CLOSED "
                       f"(from {old_state.value}) - service recovered")
            self._notify_state_change()
    
    def _reassess_predictive_state(self):
        """Reassess predictive open state based on new data"""
        try:
            failure_probability = self._predict_failure_probability()
            
            if failure_probability < self.config.ml_confidence_threshold * 0.7:
                # Low failure probability, transition to monitoring
                self._transition_to_ml_monitoring()
            
        except Exception as e:
            logger.debug(f"Predictive state reassessment error: {e}")
    
    def _transition_to_ml_monitoring(self):
        """Transition to ML monitoring state"""
        if self.state != CircuitState.ML_MONITORING:
            old_state = self.state
            self.state = CircuitState.ML_MONITORING
            self.ml_stats['state_changes'] += 1
            
            logger.info(f"ML Circuit breaker '{self.name}' entered ML_MONITORING "
                       f"(from {old_state.value})")
            self._notify_state_change()
    
    def _handle_ml_monitoring_failure(self, ml_failure: MLFailureRecord):
        """Handle failure in ML monitoring state"""
        try:
            # Analyze failure in context of ML monitoring
            if ml_failure.anomaly_score > self.config.anomaly_threshold:
                self._transition_to_open()
            elif self.failure_count >= self._get_adaptive_failure_threshold():
                self._transition_to_open()
            
        except Exception as e:
            logger.debug(f"ML monitoring failure handling error: {e}")
    
    def _record_ml_call_history(self, call_start: float, success: bool, 
                               error: Optional[str], ml_features: List[float]):
        """Record comprehensive call history with ML features"""
        call_record = {
            'timestamp': datetime.now(),
            'duration': time.time() - call_start,
            'success': success,
            'error': error,
            'ml_features': ml_features,
            'state': self.state.value,
            'active_calls': self.active_calls
        }
        
        self.call_history.append(call_record)
    
    def _record_ml_metrics(self, success: bool, duration: float, 
                          failure_type: Optional[FailureType] = None):
        """Record ML circuit metrics"""
        try:
            # Calculate metrics
            recent_calls = list(self.call_history)[-20:]
            if recent_calls:
                success_rate = sum(1 for call in recent_calls if call.get('success', False)) / len(recent_calls)
                failure_rate = 1.0 - success_rate
                avg_response_time = np.mean([call.get('duration', 0) for call in recent_calls])
            else:
                success_rate = 1.0 if success else 0.0
                failure_rate = 0.0 if success else 1.0
                avg_response_time = duration
            
            # Calculate ML health score
            ml_health_score = self._calculate_ml_health_score(success_rate, avg_response_time)
            
            metrics = MLCircuitMetrics(
                circuit_name=self.name,
                timestamp=datetime.now(),
                success_rate=success_rate,
                failure_rate=failure_rate,
                average_response_time=avg_response_time,
                concurrent_calls=self.active_calls,
                ml_health_score=ml_health_score,
                anomaly_score=self._calculate_current_anomaly_score(),
                predicted_failure_probability=self._predict_failure_probability(),
                adaptive_threshold=float(self.adaptive_failure_threshold)
            )
            
            self.ml_metrics.append(metrics)
            
        except Exception as e:
            logger.debug(f"ML metrics recording error: {e}")
    
    def _calculate_ml_health_score(self, success_rate: float, avg_response_time: float) -> float:
        """Calculate ML-driven health score"""
        try:
            # Base score from success rate
            health_score = success_rate * 100
            
            # Adjust for response time (penalize slow responses)
            if avg_response_time > self.config.timeout_seconds * 0.5:
                health_score *= 0.8
            elif avg_response_time > self.config.timeout_seconds * 0.8:
                health_score *= 0.6
            
            # Adjust for anomaly score
            anomaly_score = self._calculate_current_anomaly_score()
            health_score *= (1.0 - anomaly_score)
            
            # Adjust for failure prediction
            failure_probability = self._predict_failure_probability()
            health_score *= (1.0 - failure_probability)
            
            return max(0.0, min(100.0, health_score))
            
        except Exception:
            return 50.0  # Default health score
    
    def _notify_state_change(self):
        """Notify state change callbacks"""
        for callback in self.state_change_callbacks:
            try:
                callback(self, self.state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    # ========================================================================
    # ML TRAINING AND UPDATES
    # ========================================================================
    
    async def start_ml_training(self):
        """Start ML training background task"""
        if self.is_ml_active:
            return
        
        self.is_ml_active = True
        self.ml_task = asyncio.create_task(self._ml_training_loop())
        logger.info(f"Started ML training for circuit breaker '{self.name}'")
    
    async def stop_ml_training(self):
        """Stop ML training background task"""
        self.is_ml_active = False
        if self.ml_task:
            self.ml_task.cancel()
        logger.info(f"Stopped ML training for circuit breaker '{self.name}'")
    
    async def _ml_training_loop(self):
        """ML training background loop"""
        while self.is_ml_active:
            try:
                await asyncio.sleep(3600)  # Train every hour
                
                if len(self.ml_failures) >= self.config.min_training_samples:
                    await self._train_ml_models()
                    
                await self._update_adaptive_thresholds()
                
            except Exception as e:
                logger.error(f"ML training loop error: {e}")
    
    async def _train_ml_models(self):
        """Train ML models for failure prediction and anomaly detection"""
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < self.config.min_training_samples:
                return
            
            # Split data
            split_idx = int(len(X) * 0.8)