"""
Advanced ML Delivery Optimization Engine
=======================================
"""ML Algorithms Module - Split from delivery_optimizer.py"""


import logging
import time
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import asyncio

# ML imports
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error


                r for r in list(self.delivery_history)[-50:]
                if r.target == target and r.status in [DeliveryStatus.DELIVERED, DeliveryStatus.FAILED]
            ]
            
            if not recent_deliveries:
                return 0.8  # Default good success rate
            
            successful = sum(1 for r in recent_deliveries if r.status == DeliveryStatus.DELIVERED)
            return successful / len(recent_deliveries)
            
        except Exception:
            return 0.8
    
    def _calculate_recent_average_time(self, target: str) -> float:
        """Calculate recent average delivery time for target"""
        try:
            recent_successful = [
                r for r in list(self.delivery_history)[-30:]
                if (r.target == target and 
                    r.status == DeliveryStatus.DELIVERED and 
                    r.delivered_at and r.created_at)
            ]
            
            if not recent_successful:
                return 10.0  # Default delivery time
            
            times = [
                (r.delivered_at - r.created_at).total_seconds()
                for r in recent_successful
            ]
            
            return sum(times) / len(times)
            
        except Exception:
            return 10.0
    
    async def _retrain_ml_models(self):
        """Retrain ML models with new data"""
        try:
            if len(self.ml_features_history) < self.ml_config["min_training_samples"]:
                return
            
            # Prepare training data
            training_data = list(self.ml_features_history)[-1000:]  # Last 1000 samples
            
            # Train delivery time predictor
            await self._train_delivery_time_model(training_data)
            
            # Train failure predictor
            await self._train_failure_prediction_model(training_data)
            
            # Train route optimizer
            await self._train_route_optimization_model(training_data)
            
            logger.info("ML models retrained successfully")
            
        except Exception as e:
            logger.error(f"ML model retraining failed: {e}")
    
    async def _train_delivery_time_model(self, training_data: List[Dict[str, Any]]):
        """Train delivery time prediction model"""
        try:
            # Prepare data
            X, y = [], []
            
            for data in training_data:
                if data.get('delivery_time') is not None and data.get('features'):
                    X.append(data['features'])
                    y.append(data['delivery_time'])
            
            if len(X) < 20:
                return
            
            X_array = np.array(X)
            y_array = np.array(y)
            
            # Train scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            model.fit(X_scaled, y_array)
            
            # Store model and scaler
            self.delivery_time_predictor = model
            self.scalers['delivery_time'] = scaler
            
        except Exception as e:
            logger.error(f"Delivery time model training failed: {e}")
    
    async def _train_failure_prediction_model(self, training_data: List[Dict[str, Any]]):
        """Train failure prediction model"""
        try:
            # Prepare data
            X, y = [], []
            
            for data in training_data:
                if 'success' in data and data.get('features'):
                    X.append(data['features'])
                    y.append(0 if data['success'] else 1)  # 1 for failure
            
            if len(X) < 20:
                return
            
            X_array = np.array(X)
            y_array = np.array(y)
            
            # Train scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_scaled, y_array)
            
            # Store model and scaler
            self.failure_predictor = model
            self.scalers['failure_prediction'] = scaler
            
        except Exception as e:
            logger.error(f"Failure prediction model training failed: {e}")
    
    async def _train_route_optimization_model(self, training_data: List[Dict[str, Any]]):
        """Train route optimization model"""
        try:
            # Prepare data for clustering
            X = []
            
            for data in training_data:
                if data.get('features') and data.get('success'):
                    # Include successful delivery features for route clustering
                    X.append(data['features'])
            
            if len(X) < 20:
                return
            
            X_array = np.array(X)
            
            # Train clustering model for route optimization
            n_clusters = min(5, len(X) // 10)
            if n_clusters > 1:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusterer.fit(X_array)
                self.route_optimizer = clusterer
            
        except Exception as e:
            logger.error(f"Route optimization model training failed: {e}")
    
    def _optimize_delivery_routes(self):
        """Optimize delivery routes using ML analysis"""
        try:
            # Analyze route performance and optimize
            for route_key, performance in self.route_performance.items():
                if performance < 0.6:
                    # Poor performing route - investigate alternatives
                    target = route_key.split('_')[0]
                    self._explore_alternative_routes(target)
            
            self.delivery_stats['route_optimizations'] += 1
            
        except Exception as e:
            logger.error(f"Route optimization failed: {e}")
    
    def _explore_alternative_routes(self, target: str):
        """Explore alternative routes for target"""
        # Implementation would test different routing strategies
        logger.debug(f"Exploring alternative routes for {target}")
    
    def _optimize_resource_allocation(self):
        """Optimize resource allocation using ML"""
        try:
            # Analyze resource utilization and optimize allocation
            total_deliveries = self.delivery_stats['total_deliveries']
            
            if total_deliveries > 100:
                # Calculate optimal resource allocation
                success_rate = (self.delivery_stats['successful_deliveries'] / 
                              max(total_deliveries, 1))
                
                if success_rate < 0.8:
                    # Increase resource allocation
                    self.ml_config['resource_utilization_target'] = min(0.95, 
                        self.ml_config['resource_utilization_target'] + 0.05)
                elif success_rate > 0.95:
                    # Can reduce resource allocation
                    self.ml_config['resource_utilization_target'] = max(0.7, 
                        self.ml_config['resource_utilization_target'] - 0.02)
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
    
    def _adaptive_ml_tuning(self):
        """Adaptively tune ML configuration based on performance"""
        try:
            # Adjust ML parameters based on performance
            ml_accuracy = self.delivery_stats.get('ml_accuracy', 0.8)
            
            if ml_accuracy < 0.7:
                # Increase model complexity
                self.ml_config['min_training_samples'] = max(30, 
                    self.ml_config['min_training_samples'] - 10)
            elif ml_accuracy > 0.9:
                # Can simplify models
                self.ml_config['min_training_samples'] = min(100, 
                    self.ml_config['min_training_samples'] + 5)
            
        except Exception as e:
            logger.error(f"Adaptive ML tuning failed: {e}")
    
    def _generate_ml_insights(self):
        """Generate ML insights and recommendations"""
        try:
            insights = {
                'timestamp': datetime.now(),
                'ml_accuracy': self.delivery_stats.get('ml_accuracy', 0.8),
                'optimization_effectiveness': self._calculate_optimization_effectiveness(),
                'route_recommendations': self._generate_route_recommendations(),
                'resource_recommendations': self._generate_resource_recommendations()
            }
            
            # Store insights for analysis
            self.optimization_history.append(insights)
            
        except Exception as e:
            logger.error(f"ML insights generation failed: {e}")
    
    def _calculate_optimization_effectiveness(self) -> float:
        """Calculate ML optimization effectiveness"""
        try:
            total_optimizations = self.delivery_stats.get('ml_optimizations', 0)
            
            if total_optimizations == 0:
                return 0.0
            
            success_rate = (self.delivery_stats['successful_deliveries'] / 
                          max(self.delivery_stats['total_deliveries'], 1))
            
            # Effectiveness is based on success rate improvement
            baseline_success_rate = 0.8  # Assumed baseline
            improvement = max(0, success_rate - baseline_success_rate)
            
            return min(1.0, improvement * 5)  # Scale to 0-1
            
        except Exception:
            return 0.5
    
    def _generate_route_recommendations(self) -> List[str]:
        """Generate route optimization recommendations"""
        recommendations = []
        
        try:
            for route_key, performance in self.route_performance.items():
                if performance < 0.5:
                    recommendations.append(f"Optimize route {route_key} (performance: {performance:.2f})")
                elif performance > 0.9:
                    recommendations.append(f"Route {route_key} performing excellently (performance: {performance:.2f})")
            
        except Exception:
            pass
        
        return recommendations
    
    def _generate_resource_recommendations(self) -> List[str]:
        """Generate resource optimization recommendations"""
        recommendations = []
        
        try:
            avg_delivery_time = self.delivery_stats.get('average_delivery_time', 0)
            
            if avg_delivery_time > 30:
                recommendations.append("Consider increasing processing resources to reduce delivery time")
            elif avg_delivery_time < 5:
                recommendations.append("Current resource allocation may be excessive")
            
            success_rate = (self.delivery_stats['successful_deliveries'] / 
                          max(self.delivery_stats['total_deliveries'], 1))
            
            if success_rate < 0.8:
                recommendations.append("Increase reliability through resource allocation optimization")
            
        except Exception:
            pass
        
        return recommendations
    
    def _process_data_for_ml_delivery(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for ML-enhanced delivery"""
        processed_data = data.copy()
        
        # Add ML metadata
        processed_data['_ml_metadata'] = {
            'processed_at': datetime.now().isoformat(),
            'ml_version': '2.0',
            'optimization_enabled': self.ml_enabled
        }
        
        return processed_data
    
    def _emergency_queue_optimization(self):
        """Emergency optimization when queue is full"""
        try:
            # Clear low priority items
            temp_items = []
            
            while not self.delivery_queue.empty():
                try:
                    priority, record = self.delivery_queue.get_nowait()
                    if priority < -5:  # Keep only high priority items
                        temp_items.append((priority, record))
                except queue.Empty:
                    break
            
            # Re-queue high priority items
            for priority, record in temp_items:
                try:
                    self.delivery_queue.put_nowait((priority, record))
                except queue.Full:
                    break
            
            logger.warning(f"Emergency queue optimization completed, kept {len(temp_items)} items")
            
        except Exception as e:
            logger.error(f"Emergency queue optimization failed: {e}")
    
    def _initialize_ml_models(self):
        """Initialize ML models with default parameters"""
        try:
            if self.ml_enabled:
                # Initialize with basic models
                self.delivery_time_predictor = RandomForestRegressor(
                    n_estimators=10, random_state=42, max_depth=5
                )
                self.failure_predictor = LogisticRegression(random_state=42, max_iter=100)
                self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
                
                logger.info("ML models initialized")
                
        except Exception as e:
            logger.warning(f"ML model initialization failed: {e}")
            self.ml_enabled = False
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_delivery_analytics(self) -> Dict[str, Any]:
        """Get comprehensive delivery analytics"""
        uptime = (datetime.now() - self.delivery_stats['start_time']).total_seconds()
        
        return {
            'service_status': 'active' if self.workers_active else 'inactive',
            'uptime_seconds': uptime,
            'ml_enabled': self.ml_enabled,
            'delivery_statistics': self.delivery_stats.copy(),
            'ml_performance': {
                'ml_accuracy': self.delivery_stats.get('ml_accuracy', 0.0),
                'optimization_effectiveness': self._calculate_optimization_effectiveness(),
                'models_active': {
                    'delivery_time_predictor': self.delivery_time_predictor is not None,
                    'failure_predictor': self.failure_predictor is not None,
                    'route_optimizer': self.route_optimizer is not None,
                    'anomaly_detector': self.anomaly_detector is not None
                }
            },
            'queue_status': {
                'current_size': self.delivery_queue.qsize(),
                'max_size': self.delivery_queue.maxsize,
                'active_deliveries': len([r for r in self.delivery_records.values() 
                                        if r.status == DeliveryStatus.DELIVERING])
            },
            'route_performance': dict(self.route_performance),
            'worker_performance': dict(self.worker_performance),
            'ml_configuration': self.ml_config.copy(),
            'recent_insights': list(self.optimization_history)[-5:] if self.optimization_history else []
        }
    
    def get_failed_deliveries(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent failed deliveries"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            {
                'delivery_id': record.delivery_id,
                'target': record.target,
                'attempts': record.attempts,
                'failure_probability': record.failure_probability,
                'optimization_strategy': record.optimization_strategy.value if record.optimization_strategy else None,
                'created_at': record.created_at.isoformat(),
                'last_attempt': record.last_attempt.isoformat() if record.last_attempt else None,
                'error_message': record.error_message
            }
            for record in self.failed_deliveries
            if record.created_at >= cutoff_time
        ]
    
    def shutdown(self):
        """Shutdown ML delivery optimizer"""
        self.stop_delivery_service()
        logger.info("Advanced ML Delivery Optimizer shutdown")

# Global ML delivery optimizer instance
advanced_ml_delivery_optimizer = AdvancedMLDeliveryOptimizer()

# Export for external use
__all__ = [
    'DeliveryStatus',
    'DeliveryPriority',
    'OptimizationStrategy',
    'MLDeliveryRecord',
    'DeliveryMetrics',
    'AdvancedMLDeliveryOptimizer',
    'advanced_ml_delivery_optimizer'
]