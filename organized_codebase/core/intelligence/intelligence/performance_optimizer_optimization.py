"""
Advanced ML Performance Optimization Engine
==========================================
"""Optimization Module - Split from performance_optimizer.py"""


import logging
import time
import threading
import gc
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import asyncio

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error


                except:
                    pass
            
            # Cache metrics
            if self.cache_systems:
                total_hit_rate = 0
                cache_count = 0
                
                for cache in self.cache_systems:
                    if hasattr(cache, 'get_hit_rate'):
                        hit_rate = cache.get_hit_rate()
                        total_hit_rate += hit_rate
                        cache_count += 1
                
                if cache_count > 0:
                    avg_hit_rate = total_hit_rate / cache_count
                    metrics['cache_hit_rate'] = {
                        'value': avg_hit_rate,
                        'ml_features': [avg_hit_rate, cache_count, cpu_usage],
                        'timestamp': current_time
                    }
            
        except Exception as e:
            logger.warning(f"Error collecting ML metrics: {e}")
        
        return metrics
    
    def _record_ml_metric(self, metric_name: str, metric_data: Dict[str, Any]):
        """Record metric with ML analysis"""
        try:
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=metric_data['value'],
                timestamp=metric_data['timestamp'],
                ml_features=metric_data.get('ml_features', []),
                anomaly_score=self._calculate_anomaly_score(metric_name, metric_data['value']),
                trend_direction=self._calculate_trend_direction(metric_name),
                prediction_confidence=self._get_prediction_confidence(metric_name)
            )
            
            self.performance_metrics[metric_name].append(metric)
            
            # Keep only recent metrics
            cutoff_time = datetime.now() - timedelta(hours=4)
            while (self.performance_metrics[metric_name] and 
                   self.performance_metrics[metric_name][0].timestamp < cutoff_time):
                self.performance_metrics[metric_name].popleft()
            
        except Exception as e:
            logger.debug(f"Error recording ML metric: {e}")
    
    def _predict_metric(self, metric_name: str, current_value: float) -> float:
        """Predict future metric value using ML"""
        try:
            if metric_name not in self.ml_models:
                return current_value
            
            model = self.ml_models[metric_name]
            scaler = self.scalers.get(metric_name)
            
            # Prepare features
            recent_metrics = list(self.performance_metrics[metric_name])[-5:]
            if len(recent_metrics) < 3:
                return current_value
            
            features = []
            for metric in recent_metrics[-3:]:
                features.extend(metric.ml_features[:3])  # Use first 3 ML features
            
            # Pad features if needed
            while len(features) < 9:
                features.append(current_value)
            
            features = features[:9]  # Ensure consistent feature count
            
            # Make prediction
            if scaler:
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
            else:
                prediction = model.predict([features])[0]
            
            self.optimizer_stats['ml_predictions_made'] += 1
            return max(0, prediction)  # Ensure non-negative prediction
            
        except Exception as e:
            logger.debug(f"Error predicting metric {metric_name}: {e}")
            return current_value
    
    def _calculate_anomaly_score(self, metric_name: str, value: float) -> float:
        """Calculate anomaly score for a metric value"""
        try:
            recent_metrics = list(self.performance_metrics[metric_name])[-20:]
            if len(recent_metrics) < 10:
                return 0.0
            
            values = [m.value for m in recent_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return 0.0
            
            z_score = abs((value - mean_val) / std_val)
            anomaly_score = min(z_score / 3.0, 1.0)
            
            if anomaly_score > 0.8:
                self.optimizer_stats['anomalies_detected'] += 1
            
            return anomaly_score
            
        except Exception as e:
            logger.debug(f"Error calculating anomaly score: {e}")
            return 0.0
    
    def _calculate_trend_direction(self, metric_name: str) -> str:
        """Calculate trend direction using ML analysis"""
        try:
            recent_metrics = list(self.performance_metrics[metric_name])[-10:]
            if len(recent_metrics) < 5:
                return "stable"
            
            values = [m.value for m in recent_metrics]
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if abs(slope) < 0.1:
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"
            
        except Exception:
            return "stable"
    
    def _get_prediction_confidence(self, metric_name: str) -> float:
        """Get ML prediction confidence for a metric"""
        if metric_name in self.ml_models:
            # Use model score as confidence proxy
            return self.optimizer_stats.get('ml_accuracy_score', 0.7)
        return 0.0
    
    def _can_apply_ml_rule(self, rule: MLOptimizationRule) -> bool:
        """Check if ML rule can be applied"""
        # Check max applications
        if self.rule_applications[rule.rule_id] >= rule.max_applications:
            return False
        
        # Check cooldown
        last_applied = self.rule_last_applied.get(rule.rule_id)
        if last_applied:
            time_since_last = (datetime.now() - last_applied).total_seconds()
            if time_since_last < rule.cooldown_seconds:
                return False
        
        # Check ML confidence if applicable
        if rule.ml_predictor and hasattr(rule.ml_predictor, 'score'):
            confidence = getattr(rule.ml_predictor, 'confidence', 0.5)
            if confidence < rule.confidence_threshold:
                return False
        
        return True
    
    def _apply_ml_optimization_rule(self, rule: MLOptimizationRule) -> Optional[MLOptimizationResult]:
        """Apply ML optimization rule and measure results"""
        try:
            # Collect before metrics
            before_metrics = self._collect_current_metrics_values()
            
            # Apply the ML optimization
            rule.action()
            
            # Wait for effects to take place
            time.sleep(3)
            
            # Collect after metrics
            after_metrics = self._collect_current_metrics_values()
            
            # Calculate improvement using ML analysis
            improvement_percent = self._calculate_ml_improvement(
                before_metrics, after_metrics, rule.optimization_type
            )
            
            # Calculate ML confidence
            ml_confidence = self._calculate_ml_confidence(rule, before_metrics, after_metrics)
            
            # Record the application
            self.rule_applications[rule.rule_id] += 1
            self.rule_last_applied[rule.rule_id] = datetime.now()
            
            result = MLOptimizationResult(
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                success=True,
                improvement_percent=improvement_percent,
                ml_confidence=ml_confidence,
                optimization_strategy=rule.ml_strategy,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                ml_analysis={
                    "strategy_used": rule.ml_strategy.value,
                    "confidence_score": ml_confidence,
                    "optimization_type": rule.optimization_type.value
                }
            )
            
            self.optimization_history.append(result)
            
            # Update statistics
            if improvement_percent > 0:
                if rule.optimization_type == OptimizationType.CPU:
                    cpu_before = before_metrics.get('cpu_usage_percent', 0)
                    cpu_after = after_metrics.get('cpu_usage_percent', 0)
                    self.optimizer_stats['total_cpu_saved_percent'] += max(0, cpu_before - cpu_after)
                    
                elif rule.optimization_type == OptimizationType.MEMORY:
                    mem_before = before_metrics.get('memory_usage_percent', 0)
                    mem_after = after_metrics.get('memory_usage_percent', 0)
                    memory_saved_percent = max(0, mem_before - mem_after)
                    total_memory = psutil.virtual_memory().total / (1024 * 1024)
                    self.optimizer_stats['total_memory_saved_mb'] += (memory_saved_percent / 100) * total_memory
            
            logger.info(f"Applied ML optimization rule {rule.rule_id}: "
                       f"{improvement_percent:.1f}% improvement "
                       f"(ML confidence: {ml_confidence:.1f})")
            return result
            
        except Exception as e:
            result = MLOptimizationResult(
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                success=False,
                improvement_percent=0,
                ml_confidence=0.0,
                optimization_strategy=rule.ml_strategy,
                before_metrics=before_metrics if 'before_metrics' in locals() else {},
                after_metrics={},
                error=str(e)
            )
            
            self.optimization_history.append(result)
            logger.error(f"ML optimization rule {rule.rule_id} failed: {e}")
            return result
    
    def _collect_current_metrics_values(self) -> Dict[str, float]:
        """Collect current metric values"""
        metrics = {}
        
        try:
            metrics['cpu_usage_percent'] = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            metrics['memory_usage_percent'] = memory.percent
            metrics['memory_available_mb'] = memory.available / (1024 * 1024)
            
            # Add cache metrics if available
            if self.cache_systems:
                hit_rates = []
                for cache in self.cache_systems:
                    if hasattr(cache, 'get_hit_rate'):
                        hit_rates.append(cache.get_hit_rate())
                
                if hit_rates:
                    metrics['cache_hit_rate'] = sum(hit_rates) / len(hit_rates)
            
        except Exception as e:
            logger.warning(f"Error collecting current metrics: {e}")
        
        return metrics
    
    def _calculate_ml_improvement(self, before: Dict[str, float], after: Dict[str, float], 
                                 optimization_type: OptimizationType) -> float:
        """Calculate improvement using ML analysis"""
        try:
            if optimization_type == OptimizationType.MEMORY:
                before_val = before.get('memory_usage_percent', 0)
                after_val = after.get('memory_usage_percent', 0)
                if before_val > 0:
                    return max(0, (before_val - after_val) / before_val * 100)
            
            elif optimization_type == OptimizationType.CPU:
                before_val = before.get('cpu_usage_percent', 0)
                after_val = after.get('cpu_usage_percent', 0)
                if before_val > 0:
                    return max(0, (before_val - after_val) / before_val * 100)
            
            elif optimization_type == OptimizationType.CACHE:
                before_val = before.get('cache_hit_rate', 0)
                after_val = after.get('cache_hit_rate', 0)
                if before_val < after_val:
                    return (after_val - before_val) * 100
            
            # General improvement calculation
            total_improvement = 0
            metric_count = 0
            
            for metric_name in before.keys():
                if metric_name in after:
                    if 'usage' in metric_name or 'time' in metric_name:
                        # Lower is better
                        if before[metric_name] > 0:
                            improvement = (before[metric_name] - after[metric_name]) / before[metric_name] * 100
                            total_improvement += improvement
                            metric_count += 1
                    elif 'hit_rate' in metric_name or 'efficiency' in metric_name:
                        # Higher is better
                        improvement = (after[metric_name] - before[metric_name]) / before[metric_name] * 100
                        total_improvement += improvement
                        metric_count += 1
            
            return total_improvement / metric_count if metric_count > 0 else 0
                
        except Exception as e:
            logger.warning(f"Error calculating ML improvement: {e}")
        
        return 0
    
    def _calculate_ml_confidence(self, rule: MLOptimizationRule, 
                               before: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate ML confidence in the optimization"""
        try:
            # Base confidence from rule configuration
            base_confidence = 0.7
            
            # Adjust based on improvement magnitude
            improvement = self._calculate_ml_improvement(before, after, rule.optimization_type)
            if improvement > 10:
                base_confidence += 0.2
            elif improvement > 5:
                base_confidence += 0.1
            elif improvement < 0:
                base_confidence -= 0.3
            
            # Adjust based on ML strategy
            if rule.ml_strategy == MLOptimizationStrategy.PREDICTIVE_SCALING:
                base_confidence += 0.1
            elif rule.ml_strategy == MLOptimizationStrategy.ANOMALY_OPTIMIZATION:
                base_confidence += 0.05
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception:
            return 0.5
    
    # ========================================================================
    # ML OPTIMIZATION IMPLEMENTATIONS
    # ========================================================================
    
    def _perform_ml_memory_optimization(self):
        """Perform ML-driven memory optimization"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clean up caches using ML predictions
            for cache in self.cache_systems:
                if hasattr(cache, 'cleanup_expired'):
                    cache.cleanup_expired()
                elif hasattr(cache, 'optimize_size'):
                    # Use ML to predict optimal cache size
                    optimal_size = self._predict_optimal_cache_size(cache)
                    if hasattr(cache, 'resize'):
                        cache.resize(optimal_size)
            
            self.optimizer_stats['predictive_optimizations'] += 1
            
        except Exception as e:
            logger.error(f"ML memory optimization failed: {e}")
    
    def _perform_ml_cpu_optimization(self):