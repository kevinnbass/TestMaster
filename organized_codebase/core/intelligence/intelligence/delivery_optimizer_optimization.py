"""
Advanced ML Delivery Optimization Engine
=======================================
"""Optimization Module - Split from delivery_optimizer.py"""


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


            if 'failure_prediction' in self.scalers:
                features_scaled = self.scalers['failure_prediction'].transform([features])
            else:
                features_scaled = [features]
            
            # Make prediction
            if hasattr(self.failure_predictor, 'predict_proba'):
                probability = self.failure_predictor.predict_proba(features_scaled)[0][1]
            else:
                probability = self.failure_predictor.predict(features_scaled)[0]
            
            self.delivery_stats['failure_predictions'] = self.delivery_stats.get('failure_predictions', 0) + 1
            
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            logger.debug(f"Failure prediction error: {e}")
            return 0.1
    
    def _analyze_delivery_complexity(self, data: Dict[str, Any]) -> float:
        """Analyze delivery complexity using ML techniques"""
        try:
            complexity_factors = []
            
            # Data structure complexity
            structure_complexity = self._calculate_data_complexity(data)
            complexity_factors.append(structure_complexity)
            
            # Data size factor
            data_size = len(str(data))
            size_factor = min(1.0, data_size / 10000)  # Normalize to 0-1
            complexity_factors.append(size_factor)
            
            # Nested structure factor
            max_depth = self._calculate_max_depth(data)
            depth_factor = min(1.0, max_depth / 10)
            complexity_factors.append(depth_factor)
            
            # Content variety factor
            unique_types = self._count_unique_types(data)
            type_factor = min(1.0, unique_types / 10)
            complexity_factors.append(type_factor)
            
            # Calculate weighted complexity score
            weights = [0.4, 0.3, 0.2, 0.1]
            complexity = sum(w * f for w, f in zip(weights, complexity_factors))
            
            return min(1.0, complexity)
            
        except Exception:
            return 0.5  # Default medium complexity
    
    def _calculate_max_depth(self, obj, current_depth=0) -> int:
        """Calculate maximum nesting depth of data structure"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_max_depth(value, current_depth + 1) 
                      for value in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_max_depth(item, current_depth + 1) 
                      for item in obj)
        else:
            return current_depth
    
    def _count_unique_types(self, obj, types_seen=None) -> int:
        """Count unique data types in structure"""
        if types_seen is None:
            types_seen = set()
        
        types_seen.add(type(obj).__name__)
        
        if isinstance(obj, dict):
            for value in obj.values():
                self._count_unique_types(value, types_seen)
        elif isinstance(obj, list):
            for item in obj:
                self._count_unique_types(item, types_seen)
        
        return len(types_seen)
    
    def _analyze_route_efficiency(self, target: str) -> float:
        """Analyze route efficiency for target"""
        try:
            if target not in self.route_performance:
                return 0.8  # Default good efficiency
            
            return self.route_performance[target]
            
        except Exception:
            return 0.8
    
    def _predict_resource_requirements(self, features: List[float], target: str) -> Dict[str, float]:
        """Predict resource requirements using ML"""
        try:
            # Base resource allocation
            base_allocation = {
                'cpu': 0.1,
                'memory': 0.05,
                'network': 0.2,
                'processing_time': 5.0
            }
            
            # Adjust based on features
            if len(features) >= 10:
                data_size_factor = features[0] / 10000
                complexity_factor = features[1]
                
                # Scale resources based on complexity
                multiplier = 1.0 + (complexity_factor * 0.5) + (data_size_factor * 0.3)
                
                for resource in base_allocation:
                    base_allocation[resource] *= multiplier
            
            return base_allocation
            
        except Exception:
            return {'cpu': 0.1, 'memory': 0.05, 'network': 0.2, 'processing_time': 5.0}
    
    def _select_optimization_strategy(self, failure_probability: float, 
                                   complexity: float, target: str) -> OptimizationStrategy:
        """Select optimal delivery strategy using ML"""
        try:
            # Strategy selection based on ML analysis
            if failure_probability > 0.7:
                return OptimizationStrategy.FAILURE_PREVENTION
            elif complexity > 0.8:
                return OptimizationStrategy.RESOURCE_OPTIMIZATION
            elif self.target_load_balancing[target] > 0.8:
                return OptimizationStrategy.LOAD_BALANCING
            elif self.route_performance.get(target, 0.8) < 0.6:
                return OptimizationStrategy.PREDICTIVE_ROUTING
            else:
                return OptimizationStrategy.ADAPTIVE_RETRY
                
        except Exception:
            return OptimizationStrategy.ADAPTIVE_RETRY
    
    def _calculate_ml_priority(self, record: MLDeliveryRecord) -> float:
        """Calculate ML-enhanced priority score"""
        try:
            base_priority = record.priority.value
            
            # Adjust based on ML predictions
            if record.failure_probability > 0.8:
                base_priority += 5  # Increase priority for predicted failures
            
            if record.predicted_delivery_time and record.predicted_delivery_time > 60:
                base_priority += 2  # Increase priority for long deliveries
            
            if record.delivery_complexity > 0.8:
                base_priority += 1  # Slight increase for complex deliveries
            
            # Time-based urgency
            age_minutes = (datetime.now() - record.created_at).total_seconds() / 60
            urgency_factor = min(2.0, age_minutes / 30)  # Increase with age
            
            return -(base_priority + urgency_factor)  # Negative for min-heap
            
        except Exception:
            return -record.priority.value
    
    def _apply_ml_optimizations(self, record: MLDeliveryRecord) -> MLDeliveryRecord:
        """Apply ML-driven optimizations to delivery"""
        try:
            record.status = DeliveryStatus.OPTIMIZING
            
            # Apply strategy-specific optimizations
            if record.optimization_strategy == OptimizationStrategy.FAILURE_PREVENTION:
                record = self._apply_failure_prevention(record)
            elif record.optimization_strategy == OptimizationStrategy.RESOURCE_OPTIMIZATION:
                record = self._apply_resource_optimization(record)
            elif record.optimization_strategy == OptimizationStrategy.PREDICTIVE_ROUTING:
                record = self._apply_predictive_routing(record)
            elif record.optimization_strategy == OptimizationStrategy.LOAD_BALANCING:
                record = self._apply_load_balancing(record)
            else:
                record = self._apply_adaptive_retry(record)
            
            record.status = DeliveryStatus.ROUTING
            self.delivery_stats['ml_optimizations'] += 1
            
            return record
            
        except Exception as e:
            logger.warning(f"ML optimization failed: {e}")
            return record
    
    def _apply_failure_prevention(self, record: MLDeliveryRecord) -> MLDeliveryRecord:
        """Apply failure prevention optimizations"""
        # Increase resource allocation
        for resource in record.resource_allocation:
            record.resource_allocation[resource] *= 1.5
        
        # Add redundancy
        record.data['_ml_redundancy'] = True
        record.data['_failure_prevention'] = True
        
        return record
    
    def _apply_resource_optimization(self, record: MLDeliveryRecord) -> MLDeliveryRecord:
        """Apply resource optimization"""
        # Optimize resource allocation based on complexity
        complexity_factor = record.delivery_complexity
        
        if complexity_factor > 0.8:
            record.resource_allocation['processing_time'] *= 2.0
            record.resource_allocation['memory'] *= 1.5
        
        return record
    
    def _apply_predictive_routing(self, record: MLDeliveryRecord) -> MLDeliveryRecord:
        """Apply predictive routing optimization"""
        # Select optimal route based on ML analysis
        optimal_route = self._select_optimal_route(record.target)
        record.data['_ml_route'] = optimal_route
        
        return record
    
    def _apply_load_balancing(self, record: MLDeliveryRecord) -> MLDeliveryRecord:
        """Apply load balancing optimization"""
        # Adjust timing to balance load
        current_load = self.target_load_balancing[record.target]
        
        if current_load > 0.8:
            # Delay delivery slightly to balance load
            delay_seconds = min(30, current_load * 20)
            record.estimated_completion = datetime.now() + timedelta(seconds=delay_seconds)
        
        return record
    
    def _apply_adaptive_retry(self, record: MLDeliveryRecord) -> MLDeliveryRecord:
        """Apply adaptive retry optimization"""
        # Configure adaptive retry based on ML predictions
        if record.failure_probability > 0.5:
            record.data['_max_retries'] = 5
            record.data['_retry_backoff'] = 'exponential'
        else:
            record.data['_max_retries'] = 3
            record.data['_retry_backoff'] = 'linear'
        
        return record
    
    def _select_optimal_route(self, target: str) -> str:
        """Select optimal delivery route using ML"""
        available_routes = ['direct', 'cached', 'buffered', 'priority']
        
        # Return best performing route for target
        best_route = 'direct'
        best_performance = 0.0
        
        for route in available_routes:
            route_key = f"{target}_{route}"
            performance = self.route_performance.get(route_key, 0.5)
            
            if performance > best_performance:
                best_performance = performance
                best_route = route
        
        return best_route
    
    def _attempt_ml_delivery(self, record: MLDeliveryRecord, worker_id: str) -> bool:
        """Attempt ML-enhanced delivery"""
        try:
            record.status = DeliveryStatus.DELIVERING
            record.last_attempt = datetime.now()
            
            # Simulate ML-enhanced delivery process
            processing_time = record.resource_allocation.get('processing_time', 5.0)
            
            # Apply ML optimizations during delivery
            if record.optimization_strategy == OptimizationStrategy.FAILURE_PREVENTION:
                processing_time *= 0.8  # Faster with prevention optimizations
            
            # Simulate delivery
            time.sleep(min(processing_time / 100, 0.1))  # Scaled simulation
            
            # ML-enhanced success prediction
            success_probability = 1.0 - record.failure_probability
            
            # Add some randomness but bias toward ML prediction
            random_factor = np.random.random()
            success = random_factor < success_probability
            
            if success:
                record.status = DeliveryStatus.DELIVERED
                record.delivered_at = datetime.now()
                
                # Update route performance
                route = record.data.get('_ml_route', 'direct')
                route_key = f"{record.target}_{route}"
                self._update_route_performance(route_key, True)
                
                logger.debug(f"ML delivery {record.delivery_id} successful via {route}")
            else:
                record.status = DeliveryStatus.FAILED
                record.error_message = "ML predicted failure occurred"
                
                # Update route performance
                route = record.data.get('_ml_route', 'direct')
                route_key = f"{record.target}_{route}"
                self._update_route_performance(route_key, False)
            
            return success
            
        except Exception as e:
            record.status = DeliveryStatus.FAILED
            record.error_message = str(e)
            logger.error(f"ML delivery attempt failed: {e}")
            return False
    
    def _update_delivery_record(self, record: MLDeliveryRecord, success: bool, worker_id: str):
        """Update delivery record and statistics"""
        try:
            # Update delivery statistics
            if success:
                self.delivery_stats['successful_deliveries'] += 1
            else:
                self.delivery_stats['failed_deliveries'] += 1
                self.failed_deliveries.append(record)
            
            # Calculate delivery time
            if record.delivered_at and record.created_at:
                delivery_time = (record.delivered_at - record.created_at).total_seconds()
                
                # Update average delivery time
                total_successful = self.delivery_stats['successful_deliveries']
                if total_successful > 0:
                    current_avg = self.delivery_stats['average_delivery_time']
                    self.delivery_stats['average_delivery_time'] = (
                        (current_avg * (total_successful - 1) + delivery_time) / total_successful
                    )
            
            # Update worker performance
            worker_perf = self.worker_performance[worker_id]
            worker_perf['deliveries_completed'] += 1
            
            if worker_perf['deliveries_completed'] > 0:
                success_count = worker_perf.get('success_count', 0)
                if success:
                    success_count += 1
                
                worker_perf['success_count'] = success_count
                worker_perf['success_rate'] = success_count / worker_perf['deliveries_completed']
            
            # Store in history
            self.delivery_history.append(record)
            
            # Update target load balancing
            self.target_load_balancing[record.target] *= 0.95  # Decay factor
            
        except Exception as e:
            logger.error(f"Failed to update delivery record: {e}")
    
    def _learn_from_delivery(self, record: MLDeliveryRecord, success: bool):
        """Learn from delivery outcome to improve ML models"""
        try:
            # Store features and outcome for model retraining
            feature_data = {
                'features': record.ml_features,
                'target': record.target,
                'success': success,
                'delivery_time': None,
                'complexity': record.delivery_complexity,
                'strategy': record.optimization_strategy.value if record.optimization_strategy else None,
                'timestamp': datetime.now()
            }
            
            if record.delivered_at and record.created_at:
                feature_data['delivery_time'] = (record.delivered_at - record.created_at).total_seconds()
            
            self.ml_features_history.append(feature_data)
            
            # Update ML accuracy tracking
            if record.failure_probability is not None:
                predicted_failure = record.failure_probability > 0.5
                actual_failure = not success
                
                if predicted_failure == actual_failure:
                    # Accurate prediction
                    self.delivery_stats['ml_accuracy'] = (
                        self.delivery_stats.get('ml_accuracy', 0.8) * 0.9 + 1.0 * 0.1
                    )
                else:
                    # Inaccurate prediction
                    self.delivery_stats['ml_accuracy'] = (
                        self.delivery_stats.get('ml_accuracy', 0.8) * 0.9 + 0.0 * 0.1
                    )
            
        except Exception as e:
            logger.debug(f"Learning from delivery failed: {e}")
    
    def _update_route_performance(self, route_key: str, success: bool):
        """Update route performance metrics"""
        try:
            current_performance = self.route_performance.get(route_key, 0.5)
            
            # Update with exponential moving average
            if success:
                new_performance = current_performance * 0.9 + 1.0 * 0.1
            else:
                new_performance = current_performance * 0.9 + 0.0 * 0.1
            
            self.route_performance[route_key] = new_performance
            
        except Exception as e:
            logger.debug(f"Route performance update failed: {e}")
    
    def _calculate_recent_success_rate(self, target: str) -> float:
        """Calculate recent success rate for target"""
        try:
            recent_deliveries = [