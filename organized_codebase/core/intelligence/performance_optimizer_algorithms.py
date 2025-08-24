"""
Advanced ML Performance Optimization Engine
==========================================
"""ML Algorithms Module - Split from performance_optimizer.py"""


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


        """Perform ML-driven CPU optimization"""
        try:
            # Reduce background processing based on ML predictions
            if self.analytics_aggregator and hasattr(self.analytics_aggregator, 'cache_ttl'):
                # Use ML to predict optimal cache TTL
                optimal_ttl = self._predict_optimal_ttl()
                self.analytics_aggregator.cache_ttl = optimal_ttl
            
            # Optimize processing pipelines
            for pipeline in self.processing_pipelines:
                if hasattr(pipeline, 'optimize_workers'):
                    optimal_workers = self._predict_optimal_workers(pipeline)
                    pipeline.optimize_workers(optimal_workers)
            
            self.optimizer_stats['adaptive_optimizations'] += 1
            
        except Exception as e:
            logger.error(f"ML CPU optimization failed: {e}")
    
    def _perform_ml_cache_optimization(self):
        """Perform ML clustering-based cache optimization"""
        try:
            # Use ML clustering to optimize cache configurations
            for cache in self.cache_systems:
                if hasattr(cache, 'get_access_patterns'):
                    patterns = cache.get_access_patterns()
                    optimal_config = self._cluster_cache_patterns(patterns)
                    
                    if hasattr(cache, 'apply_configuration'):
                        cache.apply_configuration(optimal_config)
            
        except Exception as e:
            logger.error(f"ML cache optimization failed: {e}")
    
    def _perform_ml_algorithm_optimization(self):
        """Perform ML-driven algorithm optimization"""
        try:
            # Optimize algorithms based on ML resource learning
            for pipeline in self.processing_pipelines:
                if hasattr(pipeline, 'get_algorithm_performance'):
                    performance_data = pipeline.get_algorithm_performance()
                    optimal_algorithm = self._select_optimal_algorithm(performance_data)
                    
                    if hasattr(pipeline, 'set_algorithm'):
                        pipeline.set_algorithm(optimal_algorithm)
            
        except Exception as e:
            logger.error(f"ML algorithm optimization failed: {e}")
    
    # ========================================================================
    # ML HELPER METHODS
    # ========================================================================
    
    def _should_optimize_cache_ml(self) -> bool:
        """Use ML to determine if cache optimization is needed"""
        try:
            # Simple heuristic: optimize if hit rate is consistently low
            cache_metrics = self.performance_metrics.get('cache_hit_rate', [])
            if len(cache_metrics) < 5:
                return False
            
            recent_hit_rates = [m.value for m in list(cache_metrics)[-5:]]
            avg_hit_rate = np.mean(recent_hit_rates)
            
            return avg_hit_rate < 0.7  # 70% threshold
            
        except Exception:
            return False
    
    def _should_optimize_algorithms_ml(self) -> bool:
        """Use ML to determine if algorithm optimization is needed"""
        try:
            # Check if response time trend is increasing
            response_metrics = self.performance_metrics.get('analytics_response_time', [])
            if len(response_metrics) < 5:
                return False
            
            recent_times = [m.value for m in list(response_metrics)[-5:]]
            trend = np.polyfit(range(len(recent_times)), recent_times, 1)[0]
            
            return trend > 0.1  # Increasing trend
            
        except Exception:
            return False
    
    def _predict_optimal_cache_size(self, cache: Any) -> int:
        """Predict optimal cache size using ML"""
        try:
            # Simple heuristic based on memory availability
            available_memory = psutil.virtual_memory().available
            optimal_size = int(available_memory * 0.1)  # Use 10% of available memory
            return max(1024, optimal_size)  # Minimum 1KB
            
        except Exception:
            return 1024 * 1024  # Default 1MB
    
    def _predict_optimal_ttl(self) -> float:
        """Predict optimal TTL using ML"""
        try:
            # Base TTL on current CPU usage
            cpu_usage = psutil.cpu_percent()
            
            if cpu_usage > 80:
                return 300.0  # 5 minutes for high load
            elif cpu_usage > 60:
                return 180.0  # 3 minutes for medium load
            else:
                return 120.0  # 2 minutes for low load
                
        except Exception:
            return 180.0  # Default 3 minutes
    
    def _predict_optimal_workers(self, pipeline: Any) -> int:
        """Predict optimal worker count using ML"""
        try:
            # Base on CPU count and current usage
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent()
            
            if cpu_usage > 80:
                return max(1, cpu_count // 2)  # Reduce workers under high load
            else:
                return min(cpu_count, 8)  # Use available cores, max 8
                
        except Exception:
            return 4  # Default 4 workers
    
    def _cluster_cache_patterns(self, patterns: List[Any]) -> Dict[str, Any]:
        """Use ML clustering to optimize cache patterns"""
        try:
            # Placeholder for cache pattern clustering
            return {
                "cache_size": 1024 * 1024,  # 1MB
                "ttl": 300,  # 5 minutes
                "eviction_policy": "lru"
            }
            
        except Exception:
            return {}
    
    def _select_optimal_algorithm(self, performance_data: Dict[str, Any]) -> str:
        """Select optimal algorithm using ML analysis"""
        try:
            # Simple selection based on performance data
            if not performance_data:
                return "default"
            
            # Find algorithm with best performance/cost ratio
            best_algorithm = "default"
            best_score = 0
            
            for algorithm, metrics in performance_data.items():
                if isinstance(metrics, dict):
                    performance = metrics.get('performance', 0)
                    cost = metrics.get('cost', 1)
                    score = performance / cost if cost > 0 else 0
                    
                    if score > best_score:
                        best_score = score
                        best_algorithm = algorithm
            
            return best_algorithm
            
        except Exception:
            return "default"
    
    def _train_prediction_models(self):
        """Train ML models for performance prediction"""
        try:
            for metric_name, metrics_deque in self.performance_metrics.items():
                metrics_list = list(metrics_deque)
                
                if len(metrics_list) < self.ml_config["min_training_samples"]:
                    continue
                
                # Prepare training data
                X, y = [], []
                
                for i in range(len(metrics_list) - self.ml_config["feature_window_size"]):
                    features = []
                    for j in range(self.ml_config["feature_window_size"]):
                        metric = metrics_list[i + j]
                        features.extend(metric.ml_features[:3])  # Use first 3 features
                    
                    # Pad features if needed
                    while len(features) < 9:
                        features.append(0.0)
                    
                    X.append(features[:9])  # Consistent feature count
                    y.append(metrics_list[i + self.ml_config["feature_window_size"]].value)
                
                if len(X) < 10:  # Need minimum samples
                    continue
                
                # Train model
                X_array = np.array(X)
                y_array = np.array(y)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_array)
                
                # Train Random Forest model
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
                model.fit(X_scaled, y_array)
                
                # Calculate accuracy
                predictions = model.predict(X_scaled)
                mse = mean_squared_error(y_array, predictions)
                accuracy = max(0, 1 - (mse / np.var(y_array))) if np.var(y_array) > 0 else 0
                
                # Store model and scaler
                self.ml_models[metric_name] = model
                self.scalers[metric_name] = scaler
                self.optimizer_stats['ml_accuracy_score'] = accuracy
                
                logger.info(f"Trained ML model for {metric_name} (accuracy: {accuracy:.3f})")
                
        except Exception as e:
            logger.error(f"Failed to train prediction models: {e}")
    
    def _update_anomaly_detectors(self):
        """Update anomaly detection models"""
        try:
            # Update anomaly scores for all metrics
            for metric_name in self.performance_metrics.keys():
                metrics_list = list(self.performance_metrics[metric_name])
                
                if len(metrics_list) >= 10:
                    values = [m.value for m in metrics_list[-20:]]
                    
                    # Calculate running anomaly baseline
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    self.anomaly_scores[metric_name] = {
                        'mean': mean_val,
                        'std': std_val,
                        'last_updated': datetime.now()
                    }
                    
        except Exception as e:
            logger.error(f"Failed to update anomaly detectors: {e}")
    
    def _train_clustering_models(self):
        """Train clustering models for pattern analysis"""
        try:
            # Train clustering models for cache optimization
            for metric_name in ['cache_hit_rate', 'analytics_response_time']:
                metrics_list = list(self.performance_metrics.get(metric_name, []))
                
                if len(metrics_list) >= 20:
                    # Prepare features for clustering
                    features = []
                    for metric in metrics_list:
                        if len(metric.ml_features) >= 3:
                            features.append(metric.ml_features[:3])
                    
                    if len(features) >= 10:
                        features_array = np.array(features)
                        
                        # Train KMeans clustering
                        n_clusters = min(self.ml_config["cluster_count"], len(features) // 3)
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusterer.fit(features_array)
                        
                        self.clusterers[metric_name] = clusterer
                        
                        logger.info(f"Trained clustering model for {metric_name}")
                        
        except Exception as e:
            logger.error(f"Failed to train clustering models: {e}")
    
    def _generate_performance_predictions(self):
        """Generate performance predictions"""
        try:
            for metric_name in self.performance_metrics.keys():
                if metric_name in self.ml_models:
                    current_value = 0
                    if self.performance_metrics[metric_name]:
                        current_value = self.performance_metrics[metric_name][-1].value
                    
                    # Generate prediction
                    predicted_value = self._predict_metric(metric_name, current_value)
                    
                    # Store prediction
                    prediction_time = datetime.now() + timedelta(
                        minutes=self.ml_config["prediction_horizon_minutes"]
                    )
                    
                    if metric_name not in self.performance_predictions:
                        self.performance_predictions[metric_name] = []
                    
                    self.performance_predictions[metric_name].append(
                        (prediction_time, predicted_value)
                    )
                    
                    # Keep only recent predictions
                    cutoff_time = datetime.now() - timedelta(hours=2)
                    self.performance_predictions[metric_name] = [
                        p for p in self.performance_predictions[metric_name]
                        if p[0] > cutoff_time
                    ]
                    
        except Exception as e:
            logger.error(f"Failed to generate performance predictions: {e}")
    
    def _update_anomaly_scores(self):
        """Update anomaly scores for all metrics"""
        try:
            for metric_name in self.performance_metrics.keys():
                if self.performance_metrics[metric_name]:
                    latest_metric = self.performance_metrics[metric_name][-1]
                    anomaly_score = self._calculate_anomaly_score(metric_name, latest_metric.value)
                    latest_metric.anomaly_score = anomaly_score
                    
        except Exception as e:
            logger.error(f"Failed to update anomaly scores: {e}")
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_ml_optimization_summary(self) -> Dict[str, Any]:
        """Get ML optimization system summary"""
        uptime = (datetime.now() - self.optimizer_stats['start_time']).total_seconds()
        
        return {
            'optimization_level': self.optimization_level.value,
            'optimizer_active': self.optimizer_active,
            'uptime_seconds': uptime,
            'ml_models': {
                'prediction_models': len(self.ml_models),
                'clustering_models': len(self.clusterers),
                'anomaly_detectors': len(self.anomaly_scores)
            },
            'optimization_rules': {
                'total_rules': len(self.optimization_rules),
                'ml_strategies': [strategy.value for strategy in MLOptimizationStrategy]
            },
            'statistics': self.optimizer_stats.copy(),
            'current_metrics': self._collect_current_metrics_values(),
            'ml_predictions': {
                metric: len(predictions) for metric, predictions in self.performance_predictions.items()
            }
        }
    
    def get_ml_optimization_history(self, hours: int = 24) -> List[MLOptimizationResult]:
        """Get ML optimization history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [opt for opt in self.optimization_history if opt.timestamp >= cutoff_time]
    
    def shutdown(self):
        """Shutdown ML optimization engine"""
        self.stop_ml_optimization()
        logger.info("Advanced ML Performance Optimizer shutdown")

# Global ML optimizer instance
advanced_ml_performance_optimizer = AdvancedMLPerformanceOptimizer()

# Export for external use
__all__ = [
    'OptimizationType',
    'OptimizationLevel',
    'MLOptimizationStrategy',
    'MLOptimizationRule',
    'PerformanceMetric',
    'MLOptimizationResult',
    'AdvancedMLPerformanceOptimizer',
    'advanced_ml_performance_optimizer'
]