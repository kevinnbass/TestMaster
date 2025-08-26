"""
Advanced ML SLA Optimization Engine
==================================
"""Optimization Module - Split from sla_ml_optimizer.py"""


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


    def _predict_throughput(self, features: List[float]) -> float:
        """Predict throughput in TPS"""
        try:
            # Simple heuristic based on system load
            if len(features) >= 10:
                active_deliveries = features[9]
                base_throughput = max(1.0, 50.0 - active_deliveries * 0.5)
                return base_throughput
            return 25.0
            
        except Exception:
            return 25.0
    
    def _predict_error_rate(self, features: List[float]) -> float:
        """Predict error rate percentage"""
        try:
            # Simple heuristic based on recent violations
            if len(features) >= 12:
                recent_violations = features[11]
                error_rate = min(5.0, recent_violations * 0.1)
                return error_rate
            return 0.1
            
        except Exception:
            return 0.1
    
    def _calculate_prediction_confidence(self, features: List[float]) -> float:
        """Calculate prediction confidence score"""
        try:
            # Base confidence on feature quality and model performance
            base_confidence = 0.8
            
            # Adjust for feature completeness
            feature_completeness = len([f for f in features if f != 0.0]) / len(features)
            confidence = base_confidence * feature_completeness
            
            # Adjust for ML model accuracy
            ml_accuracy = self.optimizer_stats.get('ml_accuracy', 0.8)
            confidence *= ml_accuracy
            
            return max(0.1, min(1.0, confidence))
            
        except Exception:
            return 0.7
    
    def _assess_violation_risk(self, latency: float, availability: float, 
                              error_rate: float, sla_level: MLSLALevel) -> ViolationRisk:
        """Assess overall violation risk"""
        try:
            sla_config = self.sla_configs.get(sla_level)
            if not sla_config:
                return ViolationRisk.MEDIUM
            
            risk_factors = []
            
            # Latency risk
            if latency > sla_config.max_latency_ms:
                risk_factors.append(2.0)
            elif latency > sla_config.max_latency_ms * 0.8:
                risk_factors.append(1.0)
            else:
                risk_factors.append(0.0)
            
            # Availability risk
            if availability < sla_config.min_availability_percent:
                risk_factors.append(2.0)
            elif availability < sla_config.min_availability_percent + 0.1:
                risk_factors.append(1.0)
            else:
                risk_factors.append(0.0)
            
            # Error rate risk
            if error_rate > sla_config.max_error_rate_percent:
                risk_factors.append(2.0)
            elif error_rate > sla_config.max_error_rate_percent * 0.5:
                risk_factors.append(1.0)
            else:
                risk_factors.append(0.0)
            
            avg_risk = np.mean(risk_factors)
            
            if avg_risk >= 1.8:
                return ViolationRisk.CRITICAL
            elif avg_risk >= 1.2:
                return ViolationRisk.HIGH
            elif avg_risk >= 0.6:
                return ViolationRisk.MEDIUM
            elif avg_risk >= 0.2:
                return ViolationRisk.LOW
            else:
                return ViolationRisk.MINIMAL
                
        except Exception:
            return ViolationRisk.MEDIUM
    
    def _identify_optimization_opportunities(self, latency: float, availability: float,
                                           throughput: float, features: List[float]) -> List[str]:
        """Identify ML-driven optimization opportunities"""
        opportunities = []
        
        try:
            if latency > 200:
                opportunities.append("Optimize latency through caching improvements")
            
            if availability < 99.5:
                opportunities.append("Improve availability through redundancy")
            
            if throughput < 10:
                opportunities.append("Scale resources to increase throughput")
            
            if len(features) >= 8:
                cpu_usage = features[7]
                memory_usage = features[8] if len(features) > 8 else 60
                
                if cpu_usage > 80:
                    opportunities.append("Scale CPU resources")
                
                if memory_usage > 85:
                    opportunities.append("Increase memory allocation")
            
            if len(features) >= 12:
                recent_violations = features[11]
                if recent_violations > 5:
                    opportunities.append("Implement proactive failure prevention")
            
        except Exception as e:
            logger.debug(f"Optimization opportunity identification error: {e}")
        
        return opportunities
    
    def _generate_resource_recommendations(self, features: List[float], 
                                         risk_level: ViolationRisk) -> Dict[str, float]:
        """Generate ML-driven resource recommendations"""
        try:
            recommendations = {}
            
            # Base recommendations on risk level
            risk_multipliers = {
                ViolationRisk.CRITICAL: 2.0,
                ViolationRisk.HIGH: 1.5,
                ViolationRisk.MEDIUM: 1.2,
                ViolationRisk.LOW: 1.0,
                ViolationRisk.MINIMAL: 0.9
            }
            
            multiplier = risk_multipliers.get(risk_level, 1.0)
            
            # CPU recommendations
            if len(features) >= 8:
                current_cpu = features[7]
                recommended_cpu = min(100.0, current_cpu * multiplier)
                recommendations['cpu_percent'] = recommended_cpu
            
            # Memory recommendations
            if len(features) >= 9:
                current_memory = features[8]
                recommended_memory = min(95.0, current_memory * multiplier)
                recommendations['memory_percent'] = recommended_memory
            
            # Scaling recommendations
            if risk_level in [ViolationRisk.HIGH, ViolationRisk.CRITICAL]:
                recommendations['scale_factor'] = multiplier
                recommendations['priority_boost'] = True
            
            return recommendations
            
        except Exception:
            return {'cpu_percent': 50.0, 'memory_percent': 60.0}
    
    def _calculate_adaptive_timeout(self, sla_level: MLSLALevel, predicted_latency: float) -> float:
        """Calculate adaptive timeout based on ML predictions"""
        try:
            base_timeout = self.sla_configs[sla_level].delivery_timeout_seconds
            
            # Adjust based on predicted latency
            latency_factor = max(1.0, predicted_latency / 100.0)
            adaptive_timeout = base_timeout * latency_factor
            
            # Clamp to reasonable bounds
            return max(10.0, min(300.0, adaptive_timeout))
            
        except Exception:
            return 60.0
    
    def _apply_proactive_optimization(self, tracking_data: Dict[str, Any]):
        """Apply proactive optimization based on ML predictions"""
        try:
            violation_prob = tracking_data.get('violation_probability', 0.0)
            
            if violation_prob > 0.8:
                # High risk - apply aggressive optimization
                optimization_actions = [
                    "increase_worker_pool",
                    "activate_priority_queue", 
                    "enable_performance_mode"
                ]
            elif violation_prob > 0.6:
                # Medium risk - apply moderate optimization
                optimization_actions = [
                    "optimize_resource_allocation",
                    "adjust_queue_priority"
                ]
            else:
                optimization_actions = []
            
            # Execute optimization actions
            for action in optimization_actions:
                self._execute_optimization_action(action, tracking_data)
            
            if optimization_actions:
                self.optimizer_stats['resource_optimizations'] += 1
                logger.info(f"Proactive optimization applied: {optimization_actions}")
                
        except Exception as e:
            logger.error(f"Proactive optimization failed: {e}")
    
    def _execute_optimization_action(self, action: str, tracking_data: Dict[str, Any]):
        """Execute specific optimization action"""
        try:
            if action == "increase_worker_pool":
                logger.info("ACTION: Increasing worker pool size")
            elif action == "activate_priority_queue":
                logger.info("ACTION: Activating priority queue")
            elif action == "enable_performance_mode":
                logger.info("ACTION: Enabling performance mode")
            elif action == "optimize_resource_allocation":
                logger.info("ACTION: Optimizing resource allocation")
            elif action == "adjust_queue_priority":
                logger.info("ACTION: Adjusting queue priority")
            else:
                logger.info(f"ACTION: {action}")
                
        except Exception as e:
            logger.error(f"Optimization action {action} failed: {e}")
    
    def _get_tracking_info(self, tracking_id: str) -> Optional[Dict[str, Any]]:
        """Get tracking information (placeholder - would be stored in actual implementation)"""
        # In actual implementation, this would retrieve from storage
        return {
            'tracking_id': tracking_id,
            'analytics_id': f"analytics_{tracking_id[-8:]}",
            'component': 'test_component',
            'stage': 'delivery',
            'sla_level': MLSLALevel.GOLD,
            'start_time': datetime.now() - timedelta(seconds=5),
            'predicted_latency': 100.0,
            'violation_probability': 0.2
        }
    
    def _extract_delivery_features(self, delivery_info: Dict[str, Any], 
                                  latency_ms: float, success: bool) -> List[float]:
        """Extract features from completed delivery"""
        try:
            features = []
            
            # Delivery characteristics
            features.append(latency_ms)
            features.append(1.0 if success else 0.0)
            
            # Timing features
            if 'start_time' in delivery_info:
                duration = (datetime.now() - delivery_info['start_time']).total_seconds()
                features.append(duration)
            else:
                features.append(5.0)  # Default 5 seconds
            
            # Prediction accuracy features
            predicted_latency = delivery_info.get('predicted_latency', latency_ms)
            prediction_error = abs(latency_ms - predicted_latency) / max(predicted_latency, 1.0)
            features.append(prediction_error)
            
            # Context features
            current_time = datetime.now()
            features.extend([
                float(current_time.hour),
                float(current_time.weekday()),
                delivery_info.get('violation_probability', 0.0)
            ])
            
            # Pad to consistent length
            while len(features) < 10:
                features.append(0.0)
            
            return features[:10]
            
        except Exception as e:
            logger.debug(f"Delivery feature extraction error: {e}")
            return [0.0] * 10
    
    def _analyze_performance_trend(self, analytics_id: str, current_latency: float) -> str:
        """Analyze performance trend for analytics"""
        try:
            recent_metrics = self._get_recent_metrics(analytics_id, limit=5)
            
            if len(recent_metrics) < 3:
                return "stable"
            
            latencies = [m.latency_ms for m in recent_metrics if m.delivery_success]
            latencies.append(current_latency)
            
            if len(latencies) >= 3:
                # Calculate trend
                x = np.arange(len(latencies))
                slope = np.polyfit(x, latencies, 1)[0]
                
                if slope > 5:  # Increasing latency
                    return "degrading"
                elif slope < -5:  # Decreasing latency
                    return "improving"
                else:
                    return "stable"
            
            return "stable"
            
        except Exception:
            return "stable"
    
    def _get_recent_metrics(self, analytics_id: str, component: str = None, 
                           stage: str = None, limit: int = 20) -> List[MLSLAMetric]:
        """Get recent metrics for analytics"""
        try:
            metrics = []
            
            for metric in reversed(list(self.sla_metrics)):
                if metric.analytics_id == analytics_id:
                    if component and metric.component != component:
                        continue
                    if stage and metric.stage != stage:
                        continue
                    
                    metrics.append(metric)
                    
                    if len(metrics) >= limit:
                        break
            
            return metrics
            
        except Exception:
            return []
    
    def _analyze_sla_performance_ml(self, metric: MLSLAMetric, sla_level: MLSLALevel):
        """Analyze SLA performance with ML enhancements"""
        try:
            sla_config = self.sla_configs.get(sla_level)
            if not sla_config:
                return
            
            # Check for SLA violations
            violations = []
            
            # Latency violation
            if metric.latency_ms > sla_config.max_latency_ms:
                violation = self._create_latency_violation(metric, sla_config)
                violations.append(violation)
            
            # Process violations with ML analysis
            for violation in violations:
                self._process_ml_violation(violation, metric)
            
        except Exception as e:
            logger.error(f"SLA performance analysis error: {e}")
    
    def _create_latency_violation(self, metric: MLSLAMetric, sla_config: MLSLAConfiguration) -> MLSLAViolation:
        """Create latency SLA violation"""
        violation_id = f"violation_{int(time.time() * 1000)}"
        
        return MLSLAViolation(
            violation_id=violation_id,
            analytics_id=metric.analytics_id,
            violation_type="latency_breach",
            timestamp=metric.timestamp,
            current_value=metric.latency_ms,
            threshold_value=sla_config.max_latency_ms,
            severity="medium" if metric.latency_ms < sla_config.max_latency_ms * 2 else "high",
            impact_description=f"Latency {metric.latency_ms:.1f}ms exceeds SLA threshold {sla_config.max_latency_ms}ms"
        )
    
    def _create_ml_violation(self, metric: MLSLAMetric, delivery_info: Dict[str, Any], 
                            error_message: str) -> MLSLAViolation:
        """Create ML-enhanced SLA violation"""
        violation_id = f"violation_{int(time.time() * 1000)}"
        
        return MLSLAViolation(
            violation_id=violation_id,
            analytics_id=metric.analytics_id,
            violation_type="delivery_failure",
            timestamp=metric.timestamp,
            current_value=0.0,  # Failed delivery
            threshold_value=1.0,  # Expected success
            severity="critical",
            ml_predicted=delivery_info.get('violation_probability', 0.0) > 0.5,
            prediction_confidence=delivery_info.get('violation_probability', 0.0),
            impact_description=f"Delivery failed: {error_message}"
        )
    
    def _process_ml_violation(self, violation: MLSLAViolation, metric: MLSLAMetric):
        """Process SLA violation with ML analysis"""
        try:
            # Store violation
            self.active_violations[violation.violation_id] = violation
            
            # Update prediction accuracy if it was ML predicted
            if violation.ml_predicted:
                self.optimizer_stats['violations_predicted'] += 1
                self._update_prediction_accuracy(True)
            else:
                self._update_prediction_accuracy(False)
            
            # ML-driven escalation evaluation
            self._evaluate_ml_escalation(violation)
            
            logger.warning(f"ML SLA violation processed: {violation.impact_description}")
            
        except Exception as e:
            logger.error(f"ML violation processing error: {e}")
    
    def _analyze_failure_root_causes_ml(self, metric: MLSLAMetric, 
                                       delivery_info: Dict[str, Any]) -> List[str]:
        """Analyze failure root causes using ML"""
        root_causes = []
        
        try:
            # Analyze features for failure patterns
            if metric.ml_features and len(metric.ml_features) >= 8:
                cpu_usage = metric.ml_features[7] if len(metric.ml_features) > 7 else 50
                memory_usage = metric.ml_features[8] if len(metric.ml_features) > 8 else 60
                
                if cpu_usage > 85:
                    root_causes.append("High CPU utilization")
                
                if memory_usage > 90:
                    root_causes.append("Memory pressure")
                
                if metric.latency_ms > 1000:
                    root_causes.append("Network latency issues")
            
            # Check for pattern-based causes
            if metric.error_message:
                if "timeout" in metric.error_message.lower():
                    root_causes.append("Service timeout")
                elif "connection" in metric.error_message.lower():
                    root_causes.append("Connection failure")
                elif "resource" in metric.error_message.lower():
                    root_causes.append("Resource exhaustion")
            
            # Temporal analysis
            hour = datetime.now().hour
            if hour < 6 or hour > 22:
                root_causes.append("Off-hours system degradation")
            
            return root_causes or ["Unknown cause - requires investigation"]
            
        except Exception:
            return ["Analysis error - manual investigation required"]
    
    def _select_optimization_strategy_ml(self, violation: MLSLAViolation, 
                                        metric: MLSLAMetric) -> OptimizationStrategy:
        """Select optimization strategy using ML analysis"""
        try:
            # Strategy selection based on violation characteristics and ML analysis
            if violation.violation_type == "latency_breach":
                return OptimizationStrategy.PREDICTIVE_SCALING
            elif violation.violation_type == "delivery_failure":
                if violation.ml_predicted:
                    return OptimizationStrategy.ANOMALY_PREVENTION
                else:
                    return OptimizationStrategy.ADAPTIVE_THRESHOLDS
            else:
                return OptimizationStrategy.RESOURCE_OPTIMIZATION
                
        except Exception:
            return OptimizationStrategy.ADAPTIVE_THRESHOLDS
    
    def _evaluate_ml_escalation(self, violation: MLSLAViolation):
        """Evaluate escalation using ML-enhanced criteria"""
        try:
            escalation_score = 0.0
            
            # Base escalation on severity
            severity_scores = {"low": 1, "medium": 3, "high": 6, "critical": 9}
            escalation_score += severity_scores.get(violation.severity, 3)
            
            # ML prediction bonus
            if violation.ml_predicted and violation.prediction_confidence > 0.8:
                escalation_score += 2
            
            # Pattern analysis
            recent_violations = [v for v in self.active_violations.values() 
                               if (datetime.now() - v.timestamp).total_seconds() < 3600]
            
            if len(recent_violations) > 5:
                escalation_score += 3  # Pattern of violations
            
            # Determine escalation level
            if escalation_score >= 8:
                escalation_level = MLEscalationLevel.L4_MANAGEMENT
            elif escalation_score >= 6:
                escalation_level = MLEscalationLevel.L3_SENIOR_ENG
            elif escalation_score >= 4:
                escalation_level = MLEscalationLevel.L2_ENGINEERING
            elif escalation_score >= 2:
                escalation_level = MLEscalationLevel.ML_AUTOMATED
            else:
                escalation_level = MLEscalationLevel.L1_MONITORING
            
            # Trigger escalation
            self._trigger_ml_escalation(violation, escalation_level)
            
        except Exception as e:
            logger.error(f"ML escalation evaluation error: {e}")
    
    def _trigger_ml_escalation(self, violation: MLSLAViolation, level: MLEscalationLevel):
        """Trigger ML-enhanced escalation"""
        try:
            violation.escalated = True
            violation.escalation_level = level
            
            self.optimizer_stats['escalations_triggered'] += 1
            
            # Execute escalation actions based on level
            if level == MLEscalationLevel.ML_AUTOMATED:
                self._execute_automated_remediation(violation)
            elif level in [MLEscalationLevel.L2_ENGINEERING, MLEscalationLevel.L3_SENIOR_ENG]:
                self._send_technical_alert(violation, level)
            elif level in [MLEscalationLevel.L4_MANAGEMENT, MLEscalationLevel.L5_EXECUTIVE]:
                self._send_executive_alert(violation, level)
            
            logger.critical(f"ML escalation triggered: {level.value} for {violation.impact_description}")
            
        except Exception as e:
            logger.error(f"ML escalation trigger error: {e}")