"""
Advanced ML Circuit Breaker System
==================================
"""ML Algorithms Module - Split from circuit_breaker_ml.py"""


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


            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train scaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train failure predictor
            self.failure_predictor = RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=10
            )
            self.failure_predictor.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            predictions = self.failure_predictor.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions)
            self.ml_stats['model_accuracy'] = accuracy
            self.ml_stats['last_model_training'] = datetime.now()
            
            # Train anomaly detector
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.anomaly_detector.fit(X_train_scaled)
            
            logger.info(f"Trained ML models for '{self.name}' - accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"ML model training error: {e}")
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[int]]:
        """Prepare training data from historical data"""
        X, y = [], []
        
        try:
            # Use call history for training
            for i, call in enumerate(list(self.call_history)):
                if 'ml_features' in call and call['ml_features']:
                    features = call['ml_features'] + [
                        call.get('duration', 0),
                        float(call.get('active_calls', 0)),
                        call.get('timestamp', datetime.now()).hour
                    ]
                    
                    # Ensure consistent feature count
                    while len(features) < 10:
                        features.append(0.0)
                    
                    X.append(features[:10])
                    y.append(0 if call.get('success', True) else 1)
            
        except Exception as e:
            logger.debug(f"Training data preparation error: {e}")
        
        return X, y
    
    async def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on ML analysis"""
        try:
            if not self.config.adaptive_threshold_enabled:
                return
            
            # Update failure threshold
            new_threshold = self._get_adaptive_failure_threshold()
            if new_threshold != self.adaptive_failure_threshold:
                self.adaptive_failure_threshold = new_threshold
                self.ml_stats['adaptive_adjustments'] += 1
            
            # Update timeout
            new_timeout = self._get_adaptive_timeout()
            if abs(new_timeout - self.adaptive_timeout) > 1.0:
                self.adaptive_timeout = new_timeout
                self.ml_stats['adaptive_adjustments'] += 1
            
            # Update recovery timeout based on ML
            if self.config.ml_enabled:
                recovery_probability = self._predict_recovery_probability()
                if recovery_probability > 0.8:
                    self.adaptive_recovery_timeout = max(30, self.config.recovery_timeout // 2)
                elif recovery_probability < 0.5:
                    self.adaptive_recovery_timeout = min(300, self.config.recovery_timeout * 2)
                else:
                    self.adaptive_recovery_timeout = self.config.recovery_timeout
            
        except Exception as e:
            logger.debug(f"Adaptive threshold update error: {e}")
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def add_ml_prediction_callback(self, callback: Callable):
        """Add ML prediction callback"""
        self.ml_prediction_callbacks.append(callback)
    
    def add_state_change_callback(self, callback: Callable):
        """Add state change callback"""
        self.state_change_callbacks.append(callback)
    
    def add_failure_callback(self, callback: Callable):
        """Add failure callback"""
        self.failure_callbacks.append(callback)
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self.call_lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.adaptive_failure_threshold = self.config.failure_threshold
            self.adaptive_timeout = self.config.timeout_seconds
            self.adaptive_recovery_timeout = self.config.recovery_timeout
            
        logger.info(f"ML Circuit breaker '{self.name}' reset")
    
    def get_ml_status(self) -> Dict[str, Any]:
        """Get comprehensive ML circuit breaker status"""
        recent_failures = [f for f in self.ml_failures 
                          if (datetime.now() - f.timestamp).total_seconds() < 1800]
        
        recent_metrics = list(self.ml_metrics)[-5:] if self.ml_metrics else []
        
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'active_calls': self.active_calls,
            'ml_enabled': self.config.ml_enabled,
            'ml_stats': self.ml_stats.copy(),
            'adaptive_thresholds': {
                'failure_threshold': self.adaptive_failure_threshold,
                'timeout_seconds': self.adaptive_timeout,
                'recovery_timeout': self.adaptive_recovery_timeout
            },
            'ml_predictions': {
                'failure_probability': self._predict_failure_probability(),
                'anomaly_score': self._calculate_current_anomaly_score(),
                'recovery_probability': self._predict_recovery_probability()
            },
            'recent_failures': len(recent_failures),
            'recent_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'success_rate': m.success_rate,
                    'ml_health_score': m.ml_health_score,
                    'anomaly_score': m.anomaly_score
                }
                for m in recent_metrics
            ],
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'recovery_timeout': self.config.recovery_timeout,
                'ml_confidence_threshold': self.config.ml_confidence_threshold,
                'anomaly_threshold': self.config.anomaly_threshold
            }
        }

class MLCircuitBreakerOpenException(Exception):
    """Exception raised when ML circuit breaker is open"""
    pass

class AdvancedMLCircuitBreakerManager:
    """
    Manager for ML-enhanced circuit breakers with centralized
    ML training, monitoring, and coordination.
    """
    
    def __init__(self):
        """Initialize ML circuit breaker manager"""
        self.circuit_breakers: Dict[str, MLCircuitBreaker] = {}
        self.global_ml_stats = {
            'total_breakers': 0,
            'ml_enabled_breakers': 0,
            'open_breakers': 0,
            'predictive_open_breakers': 0,
            'total_ml_predictions': 0,
            'total_prevented_failures': 0,
            'start_time': datetime.now()
        }
        
        # Global ML training
        self.global_ml_task: Optional[asyncio.Task] = None
        self.is_global_ml_active = False
        
        logger.info("Advanced ML Circuit Breaker Manager initialized")
    
    def create_ml_circuit_breaker(self, name: str, config: MLCircuitBreakerConfig = None) -> MLCircuitBreaker:
        """Create ML-enhanced circuit breaker"""
        if name in self.circuit_breakers:
            return self.circuit_breakers[name]
        
        config = config or MLCircuitBreakerConfig()
        circuit_breaker = MLCircuitBreaker(name, config)
        
        # Add callbacks for global monitoring
        circuit_breaker.add_state_change_callback(self._on_state_change)
        circuit_breaker.add_failure_callback(self._on_failure)
        circuit_breaker.add_ml_prediction_callback(self._on_ml_prediction)
        
        self.circuit_breakers[name] = circuit_breaker
        self.global_ml_stats['total_breakers'] += 1
        
        if config.ml_enabled:
            self.global_ml_stats['ml_enabled_breakers'] += 1
        
        logger.info(f"Created ML circuit breaker '{name}'")
        return circuit_breaker
    
    def protect_ml(self, name: str, config: MLCircuitBreakerConfig = None):
        """Decorator for ML circuit breaker protection"""
        def decorator(func):
            circuit_breaker = self.create_ml_circuit_breaker(name, config)
            return circuit_breaker(func)
        return decorator
    
    async def start_global_ml_training(self):
        """Start global ML training and coordination"""
        if self.is_global_ml_active:
            return
        
        self.is_global_ml_active = True
        
        # Start individual circuit breaker ML training
        for cb in self.circuit_breakers.values():
            if cb.config.ml_enabled:
                await cb.start_ml_training()
        
        # Start global coordination task
        self.global_ml_task = asyncio.create_task(self._global_ml_coordination_loop())
        
        logger.info("Started global ML training and coordination")
    
    async def stop_global_ml_training(self):
        """Stop global ML training"""
        self.is_global_ml_active = False
        
        # Stop individual circuit breaker ML training
        for cb in self.circuit_breakers.values():
            await cb.stop_ml_training()
        
        if self.global_ml_task:
            self.global_ml_task.cancel()
        
        logger.info("Stopped global ML training")
    
    async def _global_ml_coordination_loop(self):
        """Global ML coordination and optimization"""
        while self.is_global_ml_active:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Update global statistics
                self._update_global_stats()
                
                # Cross-circuit breaker learning
                await self._cross_circuit_learning()
                
                # Global optimization
                await self._global_optimization()
                
            except Exception as e:
                logger.error(f"Global ML coordination error: {e}")
    
    def _update_global_stats(self):
        """Update global ML statistics"""
        self.global_ml_stats['open_breakers'] = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state in [CircuitState.OPEN, CircuitState.PREDICTIVE_OPEN]
        )
        
        self.global_ml_stats['predictive_open_breakers'] = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.PREDICTIVE_OPEN
        )
        
        self.global_ml_stats['total_ml_predictions'] = sum(
            cb.ml_stats['ml_predictions'] for cb in self.circuit_breakers.values()
        )
        
        self.global_ml_stats['total_prevented_failures'] = sum(
            cb.ml_stats['ml_prevented_failures'] for cb in self.circuit_breakers.values()
        )
    
    async def _cross_circuit_learning(self):
        """Implement cross-circuit breaker learning"""
        try:
            # Share successful ML patterns across circuit breakers
            best_performers = [
                cb for cb in self.circuit_breakers.values()
                if cb.ml_stats['model_accuracy'] > 0.8
            ]
            
            if best_performers:
                # Transfer learning patterns could be implemented here
                logger.debug(f"Cross-circuit learning with {len(best_performers)} high-performing circuits")
                
        except Exception as e:
            logger.error(f"Cross-circuit learning error: {e}")
    
    async def _global_optimization(self):
        """Global system optimization based on ML insights"""
        try:
            # Optimize system-wide thresholds based on collective learning
            all_circuits = list(self.circuit_breakers.values())
            
            if all_circuits:
                # Calculate system health score
                health_scores = []
                for cb in all_circuits:
                    if cb.ml_metrics:
                        latest_metric = cb.ml_metrics[-1]
                        health_scores.append(latest_metric.ml_health_score)
                
                if health_scores:
                    system_health = np.mean(health_scores)
                    logger.debug(f"System health score: {system_health:.2f}")
                    
        except Exception as e:
            logger.error(f"Global optimization error: {e}")
    
    def _on_state_change(self, circuit_breaker: MLCircuitBreaker, new_state: CircuitState):
        """Handle circuit breaker state changes"""
        logger.info(f"ML Circuit breaker '{circuit_breaker.name}' -> {new_state.value}")
    
    def _on_failure(self, circuit_breaker: MLCircuitBreaker, failure: MLFailureRecord):
        """Handle circuit breaker failures"""
        logger.warning(f"ML Circuit breaker '{circuit_breaker.name}' failure: "
                      f"{failure.failure_type.value} (ML score: {failure.anomaly_score:.3f})")
    
    def _on_ml_prediction(self, circuit_breaker: MLCircuitBreaker, 
                         failure_probability: float, features: List[float]):
        """Handle ML predictions"""
        if failure_probability > 0.8:
            logger.warning(f"High failure probability predicted for '{circuit_breaker.name}': "
                          f"{failure_probability:.3f}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        circuit_statuses = {
            name: cb.get_ml_status() for name, cb in self.circuit_breakers.items()
        }
        
        uptime = (datetime.now() - self.global_ml_stats['start_time']).total_seconds()
        
        return {
            'global_ml_stats': self.global_ml_stats.copy(),
            'circuit_breakers': circuit_statuses,
            'system_health': self._calculate_system_health(),
            'uptime_seconds': uptime,
            'ml_training_active': self.is_global_ml_active
        }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate ML-driven system health"""
        total_breakers = len(self.circuit_breakers)
        if total_breakers == 0:
            return {'status': 'healthy', 'score': 100}
        
        open_breakers = self.global_ml_stats['open_breakers']
        predictive_open = self.global_ml_stats['predictive_open_breakers']
        
        # Calculate health score with ML enhancements
        health_score = max(0, 100 - (open_breakers * 40) - (predictive_open * 20))
        
        # Adjust for ML prevention effectiveness
        prevented_failures = self.global_ml_stats['total_prevented_failures']
        total_predictions = self.global_ml_stats['total_ml_predictions']
        
        if total_predictions > 0:
            ml_effectiveness = prevented_failures / total_predictions
            health_score += ml_effectiveness * 10  # Bonus for ML effectiveness
        
        health_score = min(100, health_score)
        
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 80:
            status = 'healthy'
        elif health_score >= 60:
            status = 'degraded'
        elif health_score >= 40:
            status = 'unhealthy'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': health_score,
            'ml_effectiveness': prevented_failures / max(1, total_predictions),
            'degraded_components': [
                name for name, cb in self.circuit_breakers.items()
                if cb.state in [CircuitState.OPEN, CircuitState.PREDICTIVE_OPEN]
            ]
        }
    
    def shutdown(self):
        """Shutdown ML circuit breaker manager"""
        if self.is_global_ml_active:
            asyncio.create_task(self.stop_global_ml_training())
        
        logger.info("Advanced ML Circuit Breaker Manager shutdown")

# Global ML circuit breaker manager instance
ml_circuit_breaker_manager = AdvancedMLCircuitBreakerManager()

# Export for external use
__all__ = [
    'CircuitState',
    'FailureType',
    'MLStrategy',
    'MLCircuitBreakerConfig',
    'MLFailureRecord',
    'MLCircuitMetrics',
    'MLCircuitBreaker',
    'MLCircuitBreakerOpenException',
    'AdvancedMLCircuitBreakerManager',
    'ml_circuit_breaker_manager'
]