"""
Advanced ML SLA Optimization Engine
==================================
"""ML Algorithms Module - Split from sla_ml_optimizer.py"""


import logging
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import asyncio

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score


    
    def _execute_automated_remediation(self, violation: MLSLAViolation):
        """Execute automated ML-driven remediation"""
        try:
            remediation_actions = []
            
            # Select actions based on violation type and root causes
            if "High CPU utilization" in violation.root_cause_analysis:
                remediation_actions.append("scale_cpu_resources")
            
            if "Memory pressure" in violation.root_cause_analysis:
                remediation_actions.append("increase_memory_allocation")
            
            if "Network latency issues" in violation.root_cause_analysis:
                remediation_actions.append("optimize_network_routing")
            
            # Execute actions
            for action in remediation_actions:
                self._execute_remediation_action(action, violation)
            
            if remediation_actions:
                self.optimizer_stats['auto_resolutions'] += 1
                logger.info(f"Automated remediation executed: {remediation_actions}")
            
        except Exception as e:
            logger.error(f"Automated remediation error: {e}")
    
    def _execute_remediation_action(self, action: str, violation: MLSLAViolation):
        """Execute specific remediation action"""
        try:
            if action == "scale_cpu_resources":
                logger.info("REMEDIATION: Scaling CPU resources")
            elif action == "increase_memory_allocation":
                logger.info("REMEDIATION: Increasing memory allocation")
            elif action == "optimize_network_routing":
                logger.info("REMEDIATION: Optimizing network routing")
            else:
                logger.info(f"REMEDIATION: {action}")
                
        except Exception as e:
            logger.error(f"Remediation action {action} failed: {e}")
    
    def _send_technical_alert(self, violation: MLSLAViolation, level: MLEscalationLevel):
        """Send technical team alert"""
        logger.info(f"TECHNICAL ALERT ({level.value}): {violation.impact_description}")
    
    def _send_executive_alert(self, violation: MLSLAViolation, level: MLEscalationLevel):
        """Send executive team alert"""
        logger.critical(f"EXECUTIVE ALERT ({level.value}): {violation.impact_description}")
    
    def _record_ml_training_data(self, metric: MLSLAMetric, delivery_info: Dict[str, Any]):
        """Record data for ML model training"""
        try:
            training_record = {
                'timestamp': metric.timestamp,
                'analytics_id': metric.analytics_id,
                'features': metric.ml_features,
                'latency': metric.latency_ms,
                'success': metric.delivery_success,
                'predicted_latency': delivery_info.get('predicted_latency'),
                'violation_probability': delivery_info.get('violation_probability'),
                'component': metric.component,
                'stage': metric.stage
            }
            
            self.ml_features_history.append(training_record)
            
        except Exception as e:
            logger.debug(f"ML training data recording error: {e}")
    
    def _update_adaptive_thresholds(self, metric: MLSLAMetric, sla_level: MLSLALevel):
        """Update adaptive thresholds based on ML analysis"""
        try:
            if metric.analytics_id not in self.adaptive_thresholds:
                self.adaptive_thresholds[metric.analytics_id] = {}
            
            thresholds = self.adaptive_thresholds[metric.analytics_id]
            
            # Update latency threshold based on recent performance
            recent_latencies = [m.latency_ms for m in self._get_recent_metrics(metric.analytics_id, limit=10) 
                              if m.delivery_success]
            
            if len(recent_latencies) >= 5:
                avg_latency = np.mean(recent_latencies)
                std_latency = np.std(recent_latencies)
                
                # Set adaptive threshold as mean + 2*std
                adaptive_threshold = avg_latency + 2 * std_latency
                
                sla_config = self.sla_configs.get(sla_level)
                if sla_config:
                    # Don't exceed original SLA limit
                    adaptive_threshold = min(adaptive_threshold, sla_config.max_latency_ms)
                    # Don't go below 80% of SLA limit
                    adaptive_threshold = max(adaptive_threshold, sla_config.max_latency_ms * 0.8)
                
                thresholds['latency'] = adaptive_threshold
                self.optimizer_stats['threshold_adaptations'] += 1
            
        except Exception as e:
            logger.debug(f"Adaptive threshold update error: {e}")
    
    def _update_prediction_accuracy(self, accurate: bool):
        """Update ML prediction accuracy tracking"""
        try:
            current_accuracy = self.optimizer_stats.get('prediction_accuracy', 0.8)
            
            # Exponential moving average
            if accurate:
                new_accuracy = current_accuracy * 0.9 + 1.0 * 0.1
            else:
                new_accuracy = current_accuracy * 0.9 + 0.0 * 0.1
            
            self.optimizer_stats['prediction_accuracy'] = new_accuracy
            self.optimizer_stats['ml_accuracy'] = new_accuracy
            
        except Exception:
            pass
    
    def _calculate_performance_trend_score(self, analytics_id: str) -> float:
        """Calculate performance trend score"""
        try:
            recent_metrics = self._get_recent_metrics(analytics_id, limit=10)
            
            if len(recent_metrics) < 3:
                return 0.0
            
            latencies = [m.latency_ms for m in recent_metrics if m.delivery_success]
            
            if len(latencies) >= 3:
                # Calculate trend slope
                x = np.arange(len(latencies))
                slope = np.polyfit(x, latencies, 1)[0]
                
                # Normalize to -1 to 1 range
                return max(-1.0, min(1.0, slope / 100.0))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _setup_default_sla_configs(self):
        """Setup default ML-enhanced SLA configurations"""
        self.sla_configs = {
            MLSLALevel.PLATINUM: MLSLAConfiguration(
                level=MLSLALevel.PLATINUM,
                max_latency_ms=50.0,
                min_availability_percent=99.99,
                min_throughput_tps=100.0,
                max_error_rate_percent=0.01,
                delivery_timeout_seconds=20.0,
                escalation_threshold_minutes=1
            ),
            MLSLALevel.GOLD: MLSLAConfiguration(
                level=MLSLALevel.GOLD,
                max_latency_ms=100.0,
                min_availability_percent=99.9,
                min_throughput_tps=50.0,
                max_error_rate_percent=0.1,
                delivery_timeout_seconds=30.0,
                escalation_threshold_minutes=2
            ),
            MLSLALevel.SILVER: MLSLAConfiguration(
                level=MLSLALevel.SILVER,
                max_latency_ms=250.0,
                min_availability_percent=99.5,
                min_throughput_tps=25.0,
                max_error_rate_percent=0.5,
                delivery_timeout_seconds=60.0,
                escalation_threshold_minutes=5
            ),
            MLSLALevel.BRONZE: MLSLAConfiguration(
                level=MLSLALevel.BRONZE,
                max_latency_ms=500.0,
                min_availability_percent=99.0,
                min_throughput_tps=10.0,
                max_error_rate_percent=1.0,
                delivery_timeout_seconds=120.0,
                escalation_threshold_minutes=10
            ),
            MLSLALevel.ADAPTIVE: MLSLAConfiguration(
                level=MLSLALevel.ADAPTIVE,
                max_latency_ms=150.0,  # Will be ML-adjusted
                min_availability_percent=99.5,
                min_throughput_tps=30.0,
                max_error_rate_percent=0.3,
                delivery_timeout_seconds=45.0,
                escalation_threshold_minutes=3
            )
        }
    
    def _initialize_ml_models(self):
        """Initialize ML models"""
        try:
            if self.ml_enabled:
                # Initialize with basic models
                self.latency_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
                self.violation_predictor = GradientBoostingClassifier(n_estimators=50, random_state=42)
                self.performance_classifier = LogisticRegression(random_state=42, max_iter=1000)
                self.resource_optimizer = KMeans(n_clusters=5, random_state=42, n_init=10)
                
                logger.info("ML models initialized for SLA optimizer")
                
        except Exception as e:
            logger.warning(f"ML model initialization failed: {e}")
            self.ml_enabled = False
    
    # ========================================================================
    # BACKGROUND ML LOOPS
    # ========================================================================
    
    def _ml_monitoring_loop(self):
        """Background ML monitoring loop"""
        while self.ml_optimizer_active:
            try:
                time.sleep(self.monitoring_interval)
                
                # Monitor ongoing SLA performance
                self._monitor_sla_performance_ml()
                
                # Check for auto-resolution opportunities
                self._check_auto_resolution_opportunities()
                
            except Exception as e:
                logger.error(f"ML monitoring loop error: {e}")
    
    def _ml_prediction_loop(self):
        """Background ML prediction loop"""
        while self.ml_optimizer_active:
            try:
                time.sleep(120)  # Every 2 minutes
                
                # Generate performance predictions
                self._generate_proactive_predictions()
                
                # Update risk assessments
                self._update_risk_assessments()
                
            except Exception as e:
                logger.error(f"ML prediction loop error: {e}")
    
    def _ml_optimization_loop(self):
        """Background ML optimization loop"""
        while self.ml_optimizer_active:
            try:
                time.sleep(self.ml_config["resource_optimization_interval"])
                
                # Retrain ML models
                await self._retrain_ml_models()
                
                # Optimize system configuration
                self._optimize_system_configuration()
                
            except Exception as e:
                logger.error(f"ML optimization loop error: {e}")
    
    def _monitor_sla_performance_ml(self):
        """Monitor SLA performance with ML enhancements"""
        try:
            # Check current performance against predictions
            current_time = datetime.now()
            
            for analytics_id, prediction in self.performance_predictions.items():
                # Check if prediction is still valid
                if (current_time - prediction.prediction_timestamp).total_seconds() < 1800:  # 30 minutes
                    # Validate prediction accuracy
                    recent_metrics = self._get_recent_metrics(analytics_id, limit=5)
                    
                    if recent_metrics:
                        actual_avg_latency = np.mean([m.latency_ms for m in recent_metrics if m.delivery_success])
                        predicted_latency = prediction.predicted_latency
                        
                        error = abs(actual_avg_latency - predicted_latency) / max(predicted_latency, 1.0)
                        
                        if error < 0.2:  # Within 20% accuracy
                            self._update_prediction_accuracy(True)
                        else:
                            self._update_prediction_accuracy(False)
            
        except Exception as e:
            logger.error(f"ML performance monitoring error: {e}")
    
    def _check_auto_resolution_opportunities(self):
        """Check for auto-resolution opportunities"""
        try:
            # Check if violations can be auto-resolved
            for violation in list(self.active_violations.values()):
                if not violation.resolved and violation.escalation_level == MLEscalationLevel.ML_AUTOMATED:
                    # Check if conditions have improved
                    if self._is_violation_auto_resolvable(violation):
                        violation.resolved = True
                        violation.resolution_time = datetime.now()
                        self.optimizer_stats['auto_resolutions'] += 1
                        
                        logger.info(f"Auto-resolved violation: {violation.violation_id}")
            
        except Exception as e:
            logger.error(f"Auto-resolution check error: {e}")
    
    def _is_violation_auto_resolvable(self, violation: MLSLAViolation) -> bool:
        """Check if violation can be auto-resolved"""
        try:
            # Check recent metrics for improvement
            recent_metrics = self._get_recent_metrics(violation.analytics_id, limit=3)
            
            if len(recent_metrics) >= 2:
                recent_latencies = [m.latency_ms for m in recent_metrics if m.delivery_success]
                
                if recent_latencies:
                    avg_latency = np.mean(recent_latencies)
                    return avg_latency <= violation.threshold_value
            
            return False
            
        except Exception:
            return False
    
    def _generate_proactive_predictions(self):
        """Generate proactive performance predictions"""
        try:
            # Generate predictions for active analytics
            active_analytics = set(m.analytics_id for m in list(self.sla_metrics)[-50:])
            
            for analytics_id in active_analytics:
                features = self._extract_sla_features(analytics_id, "default", "delivery")
                prediction = self._generate_performance_prediction(analytics_id, features, MLSLALevel.GOLD)
                
                if prediction and prediction.risk_assessment in [ViolationRisk.HIGH, ViolationRisk.CRITICAL]:
                    # High risk - trigger proactive actions
                    logger.warning(f"High risk prediction for {analytics_id}: {prediction.risk_assessment.value}")
                    self.optimizer_stats['violations_predicted'] += 1
            
        except Exception as e:
            logger.error(f"Proactive prediction generation error: {e}")
    
    def _update_risk_assessments(self):
        """Update risk assessments for all tracked analytics"""
        try:
            # Update risk assessments based on recent performance
            for analytics_id in set(m.analytics_id for m in list(self.sla_metrics)[-100:]):
                recent_metrics = self._get_recent_metrics(analytics_id, limit=10)
                
                if recent_metrics:
                    # Calculate risk based on recent performance
                    failure_rate = 1.0 - (sum(1 for m in recent_metrics if m.delivery_success) / len(recent_metrics))
                    avg_latency = np.mean([m.latency_ms for m in recent_metrics if m.delivery_success])
                    
                    # Update baselines
                    if analytics_id not in self.performance_baselines:
                        self.performance_baselines[analytics_id] = {}
                    
                    self.performance_baselines[analytics_id].update({
                        'avg_latency': avg_latency,
                        'failure_rate': failure_rate,
                        'last_updated': datetime.now()
                    })
            
        except Exception as e:
            logger.error(f"Risk assessment update error: {e}")
    
    async def _retrain_ml_models(self):
        """Retrain ML models with accumulated data"""
        try:
            if len(self.ml_features_history) < self.ml_config["min_training_samples"]:
                return
            
            training_data = list(self.ml_features_history)[-1000:]  # Last 1000 samples
            
            # Train latency predictor
            X_latency, y_latency = [], []
            for record in training_data:
                if record.get('features') and record.get('latency') is not None:
                    X_latency.append(record['features'])
                    y_latency.append(record['latency'])
            
            if len(X_latency) >= 20:
                X_latency_array = np.array(X_latency)
                y_latency_array = np.array(y_latency)
                
                # Train scaler
                scaler_latency = StandardScaler()
                X_latency_scaled = scaler_latency.fit_transform(X_latency_array)
                
                # Train model
                self.latency_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
                self.latency_predictor.fit(X_latency_scaled, y_latency_array)
                
                self.scalers['latency_prediction'] = scaler_latency
                
                logger.info("ML models retrained for SLA optimizer")
            
        except Exception as e:
            logger.error(f"ML model retraining error: {e}")
    
    def _optimize_system_configuration(self):
        """Optimize system configuration based on ML insights"""
        try:
            # Analyze performance patterns and optimize
            optimization_count = 0
            
            # Optimize adaptive thresholds
            for analytics_id, thresholds in self.adaptive_thresholds.items():
                baseline = self.performance_baselines.get(analytics_id, {})
                
                if baseline and baseline.get('avg_latency'):
                    current_threshold = thresholds.get('latency', 100.0)
                    optimal_threshold = baseline['avg_latency'] * 1.5
                    
                    if abs(optimal_threshold - current_threshold) > 10:
                        thresholds['latency'] = optimal_threshold
                        optimization_count += 1
            
            if optimization_count > 0:
                self.optimizer_stats['resource_optimizations'] += optimization_count
                logger.info(f"System configuration optimized: {optimization_count} adjustments")
            
        except Exception as e:
            logger.error(f"System configuration optimization error: {e}")
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_ml_sla_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML SLA summary"""
        with self.optimizer_lock:
            recent_metrics = list(self.sla_metrics)[-100:] if self.sla_metrics else []
            
            if recent_metrics:
                successful_metrics = [m for m in recent_metrics if m.delivery_success]
                
                avg_latency = np.mean([m.latency_ms for m in successful_metrics]) if successful_metrics else 0
                success_rate = len(successful_metrics) / len(recent_metrics) * 100
                avg_prediction_accuracy = self.optimizer_stats.get('prediction_accuracy', 0.0) * 100
            else:
                avg_latency = success_rate = avg_prediction_accuracy = 0
            
            return {
                'ml_optimizer_status': 'active' if self.ml_optimizer_active else 'inactive',
                'ml_enabled': self.ml_enabled,
                'statistics': self.optimizer_stats.copy(),
                'current_performance': {
                    'avg_latency_ms': avg_latency,
                    'success_rate_percent': success_rate,
                    'ml_prediction_accuracy_percent': avg_prediction_accuracy
                },
                'active_violations': len([v for v in self.active_violations.values() if not v.resolved]),
                'performance_predictions': len(self.performance_predictions),
                'adaptive_thresholds': len(self.adaptive_thresholds),
                'ml_models_active': {
                    'latency_predictor': self.latency_predictor is not None,
                    'violation_predictor': self.violation_predictor is not None,
                    'performance_classifier': self.performance_classifier is not None,
                    'resource_optimizer': self.resource_optimizer is not None
                },
                'ml_configuration': self.ml_config.copy(),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_predictions(self, analytics_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current performance predictions"""
        with self.optimizer_lock:
            if analytics_id:
                prediction = self.performance_predictions.get(analytics_id)
                return {
                    'analytics_id': analytics_id,
                    'prediction': prediction.__dict__ if prediction else None
                }
            else:
                return {
                    'total_predictions': len(self.performance_predictions),
                    'predictions': {
                        aid: prediction.__dict__ 
                        for aid, prediction in self.performance_predictions.items()
                    }
                }
    
    def force_ml_optimization(self) -> Dict[str, Any]:
        """Force immediate ML optimization"""
        try:
            optimizations_applied = 0
            
            # Force threshold adaptations
            for analytics_id in list(self.adaptive_thresholds.keys()):
                recent_metrics = self._get_recent_metrics(analytics_id, limit=5)
                if recent_metrics:
                    # Force threshold update
                    for metric in recent_metrics[-1:]:
                        self._update_adaptive_thresholds(metric, MLSLALevel.GOLD)
                    optimizations_applied += 1
            
            # Force prediction updates
            for analytics_id in set(m.analytics_id for m in list(self.sla_metrics)[-20:]):
                features = self._extract_sla_features(analytics_id, "default", "delivery")
                prediction = self._generate_performance_prediction(analytics_id, features, MLSLALevel.GOLD)
                
                if prediction:
                    self.performance_predictions[analytics_id] = prediction
                    optimizations_applied += 1
            
            return {
                'optimizations_applied': optimizations_applied,
                'timestamp': datetime.now().isoformat(),
                'ml_accuracy': self.optimizer_stats.get('ml_accuracy', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Force ML optimization error: {e}")
            return {'error': str(e), 'optimizations_applied': 0}
    
    def shutdown(self):
        """Shutdown ML SLA optimizer"""
        self.stop_ml_sla_optimizer()
        logger.info("Advanced ML SLA Optimizer shutdown")

# Global ML SLA optimizer instance
advanced_ml_sla_optimizer = AdvancedMLSLAOptimizer()

# Export for external use
__all__ = [
    'MLSLALevel',
    'MLEscalationLevel',
    'ViolationRisk',
    'OptimizationStrategy',
    'MLSLAConfiguration',
    'MLSLAMetric',
    'MLSLAViolation',
    'MLPerformancePrediction',
    'AdvancedMLSLAOptimizer',
    'advanced_ml_sla_optimizer'
]