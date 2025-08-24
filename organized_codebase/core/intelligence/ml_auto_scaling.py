"""
Enterprise ML Auto-Scaling System
Intelligent resource scaling and load balancing for ML infrastructure

This module provides comprehensive auto-scaling capabilities including:
- Dynamic resource allocation based on ML workload predictions
- Intelligent load balancing across ML modules
- Container and cloud infrastructure scaling
- Cost optimization through smart resource management
- Performance-driven scaling decisions
"""

import asyncio
import json
import logging
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from collections import defaultdict, deque
import statistics
import numpy as np
from pathlib import Path
import math

@dataclass
class ScalingMetrics:
    """Comprehensive metrics for scaling decisions"""
    timestamp: datetime
    module_name: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    request_rate: float
    response_time: float
    queue_depth: int
    error_rate: float
    throughput: float
    active_instances: int
    target_instances: int

@dataclass
class ScalingEvent:
    """Record of scaling operations"""
    event_id: str
    timestamp: datetime
    module_name: str
    action: str  # 'scale_up', 'scale_down', 'migrate', 'optimize'
    trigger_reason: str
    old_config: Dict[str, Any]
    new_config: Dict[str, Any]
    success: bool
    duration_seconds: float
    cost_impact: float

@dataclass
class ResourcePool:
    """Resource pool configuration and status"""
    pool_id: str
    pool_type: str  # 'cpu', 'gpu', 'memory', 'storage'
    total_capacity: float
    allocated_capacity: float
    available_capacity: float
    cost_per_unit: float
    performance_rating: float
    geographical_zone: str
    specialized_workloads: List[str]

@dataclass
class PredictiveScalingModel:
    """ML model for predictive scaling decisions"""
    model_id: str
    model_type: str  # 'time_series', 'regression', 'classification'
    accuracy_score: float
    last_trained: datetime
    prediction_horizon: int  # minutes
    confidence_threshold: float
    feature_weights: Dict[str, float]

class MLAutoScaling:
    """
    Enterprise ML Auto-Scaling System
    
    Provides intelligent, cost-optimized resource scaling for ML infrastructure
    with predictive capabilities and multi-cloud support.
    """
    
    def __init__(self, config_path: str = "scaling_config.json"):
        self.config_path = config_path
        self.scaling_history = deque(maxlen=1000)
        self.resource_pools = {}
        self.scaling_models = {}
        self.current_metrics = {}
        self.pending_operations = []
        
        # Enterprise scaling configuration
        self.scaling_config = {
            "prediction_enabled": True,
            "cost_optimization": True,
            "max_scale_factor": 10.0,
            "min_instances": 1,
            "max_instances_per_module": 20,
            "scale_up_cooldown": 300,  # 5 minutes
            "scale_down_cooldown": 600,  # 10 minutes
            "emergency_scaling": True,
            "cost_budget_daily": 1000.0,  # USD
            "performance_targets": {
                "response_time_max": 1000,  # ms
                "cpu_target": 70.0,  # percent
                "memory_target": 75.0,  # percent
                "error_rate_max": 2.0,  # percent
                "throughput_min": 100  # requests/minute
            },
            "scaling_thresholds": {
                "cpu_scale_up": 80.0,
                "cpu_scale_down": 30.0,
                "memory_scale_up": 85.0,
                "memory_scale_down": 25.0,
                "response_time_scale_up": 1500,
                "queue_depth_scale_up": 50,
                "error_rate_scale_up": 5.0
            }
        }
        
        # Initialize ML modules registry with scaling parameters
        self.ml_modules = {
            "anomaly_detector": {"min_instances": 2, "max_instances": 8, "cpu_per_instance": 2, "memory_per_instance": 4},
            "smart_cache": {"min_instances": 3, "max_instances": 12, "cpu_per_instance": 1, "memory_per_instance": 8},
            "correlation_engine": {"min_instances": 2, "max_instances": 10, "cpu_per_instance": 4, "memory_per_instance": 6},
            "batch_processor": {"min_instances": 1, "max_instances": 15, "cpu_per_instance": 8, "memory_per_instance": 16},
            "predictive_engine": {"min_instances": 2, "max_instances": 12, "cpu_per_instance": 6, "memory_per_instance": 12},
            "performance_optimizer": {"min_instances": 1, "max_instances": 6, "cpu_per_instance": 3, "memory_per_instance": 8},
            "circuit_breaker": {"min_instances": 3, "max_instances": 8, "cpu_per_instance": 1, "memory_per_instance": 2},
            "delivery_optimizer": {"min_instances": 2, "max_instances": 10, "cpu_per_instance": 4, "memory_per_instance": 8},
            "integrity_guardian": {"min_instances": 2, "max_instances": 6, "cpu_per_instance": 2, "memory_per_instance": 4},
            "sla_optimizer": {"min_instances": 1, "max_instances": 8, "cpu_per_instance": 3, "memory_per_instance": 6},
            "adaptive_load_balancer": {"min_instances": 2, "max_instances": 6, "cpu_per_instance": 2, "memory_per_instance": 4},
            "intelligent_scheduler": {"min_instances": 1, "max_instances": 8, "cpu_per_instance": 4, "memory_per_instance": 8},
            "resource_optimizer": {"min_instances": 1, "max_instances": 4, "cpu_per_instance": 2, "memory_per_instance": 4},
            "failure_predictor": {"min_instances": 2, "max_instances": 6, "cpu_per_instance": 3, "memory_per_instance": 6},
            "quality_monitor": {"min_instances": 1, "max_instances": 4, "cpu_per_instance": 2, "memory_per_instance": 4},
            "scaling_coordinator": {"min_instances": 1, "max_instances": 3, "cpu_per_instance": 1, "memory_per_instance": 2},
            "telemetry_analyzer": {"min_instances": 2, "max_instances": 8, "cpu_per_instance": 4, "memory_per_instance": 8},
            "security_monitor": {"min_instances": 2, "max_instances": 6, "cpu_per_instance": 2, "memory_per_instance": 4},
            "compliance_auditor": {"min_instances": 1, "max_instances": 4, "cpu_per_instance": 2, "memory_per_instance": 4}
        }
        
        self.logger = logging.getLogger(__name__)
        self.scaling_active = True
        self.cost_tracker = {"daily_cost": 0.0, "last_reset": datetime.now()}
        
        self._initialize_resource_pools()
        self._initialize_scaling_models()
        self._start_scaling_threads()
    
    def _initialize_resource_pools(self):
        """Initialize available resource pools"""
        
        # CPU-optimized pools
        self.resource_pools["cpu_pool_1"] = ResourcePool(
            pool_id="cpu_pool_1",
            pool_type="cpu",
            total_capacity=1000.0,  # CPU cores
            allocated_capacity=200.0,
            available_capacity=800.0,
            cost_per_unit=0.10,  # USD per core-hour
            performance_rating=8.5,
            geographical_zone="us-east-1",
            specialized_workloads=["general_ml", "batch_processing"]
        )
        
        # GPU-optimized pools
        self.resource_pools["gpu_pool_1"] = ResourcePool(
            pool_id="gpu_pool_1",
            pool_type="gpu",
            total_capacity=50.0,  # GPU units
            allocated_capacity=15.0,
            available_capacity=35.0,
            cost_per_unit=2.50,  # USD per GPU-hour
            performance_rating=9.2,
            geographical_zone="us-east-1",
            specialized_workloads=["deep_learning", "inference"]
        )
        
        # Memory-optimized pools
        self.resource_pools["memory_pool_1"] = ResourcePool(
            pool_id="memory_pool_1",
            pool_type="memory",
            total_capacity=10000.0,  # GB
            allocated_capacity=2500.0,
            available_capacity=7500.0,
            cost_per_unit=0.05,  # USD per GB-hour
            performance_rating=8.8,
            geographical_zone="us-east-1",
            specialized_workloads=["cache_intensive", "large_models"]
        )
        
        # High-performance storage
        self.resource_pools["storage_pool_1"] = ResourcePool(
            pool_id="storage_pool_1",
            pool_type="storage",
            total_capacity=100000.0,  # GB
            allocated_capacity=25000.0,
            available_capacity=75000.0,
            cost_per_unit=0.02,  # USD per GB-hour
            performance_rating=9.0,
            geographical_zone="us-east-1",
            specialized_workloads=["data_intensive", "model_storage"]
        )
    
    def _initialize_scaling_models(self):
        """Initialize predictive scaling models"""
        
        # Time series model for workload prediction
        self.scaling_models["workload_predictor"] = PredictiveScalingModel(
            model_id="workload_predictor",
            model_type="time_series",
            accuracy_score=0.87,
            last_trained=datetime.now() - timedelta(hours=2),
            prediction_horizon=30,  # 30 minutes
            confidence_threshold=0.75,
            feature_weights={
                "historical_load": 0.4,
                "time_of_day": 0.25,
                "day_of_week": 0.15,
                "request_pattern": 0.2
            }
        )
        
        # Resource optimization model
        self.scaling_models["resource_optimizer"] = PredictiveScalingModel(
            model_id="resource_optimizer",
            model_type="regression",
            accuracy_score=0.82,
            last_trained=datetime.now() - timedelta(hours=1),
            prediction_horizon=15,  # 15 minutes
            confidence_threshold=0.8,
            feature_weights={
                "current_utilization": 0.35,
                "queue_metrics": 0.3,
                "response_times": 0.25,
                "error_patterns": 0.1
            }
        )
        
        # Cost optimization model
        self.scaling_models["cost_optimizer"] = PredictiveScalingModel(
            model_id="cost_optimizer",
            model_type="classification",
            accuracy_score=0.91,
            last_trained=datetime.now() - timedelta(minutes=30),
            prediction_horizon=60,  # 1 hour
            confidence_threshold=0.85,
            feature_weights={
                "cost_trends": 0.4,
                "performance_requirements": 0.3,
                "resource_availability": 0.2,
                "business_priority": 0.1
            }
        )
    
    def _start_scaling_threads(self):
        """Start background scaling threads"""
        
        # Metrics collection thread
        metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        metrics_thread.start()
        
        # Scaling decision thread
        decision_thread = threading.Thread(target=self._scaling_decision_loop, daemon=True)
        decision_thread.start()
        
        # Predictive analysis thread
        prediction_thread = threading.Thread(target=self._predictive_scaling_loop, daemon=True)
        prediction_thread.start()
        
        # Cost monitoring thread
        cost_thread = threading.Thread(target=self._cost_monitoring_loop, daemon=True)
        cost_thread.start()
        
        # Resource optimization thread
        optimization_thread = threading.Thread(target=self._resource_optimization_loop, daemon=True)
        optimization_thread.start()
    
    def _metrics_collection_loop(self):
        """Continuous collection of scaling-relevant metrics"""
        while self.scaling_active:
            try:
                current_time = datetime.now()
                
                for module_name in self.ml_modules:
                    metrics = self._collect_scaling_metrics(module_name)
                    
                    scaling_metrics = ScalingMetrics(
                        timestamp=current_time,
                        module_name=module_name,
                        **metrics
                    )
                    
                    self.current_metrics[module_name] = scaling_metrics
                
                time.sleep(5)  # Collect metrics every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(10)
    
    def _collect_scaling_metrics(self, module_name: str) -> Dict[str, Any]:
        """Collect comprehensive scaling metrics for a module"""
        
        # Simulate realistic scaling metrics
        base_time = time.time()
        module_hash = hash(module_name) % 1000
        
        # Generate realistic patterns
        daily_pattern = np.sin(2 * np.pi * (base_time % 86400) / 86400)  # Daily cycle
        weekly_pattern = np.sin(2 * np.pi * (base_time % 604800) / 604800)  # Weekly cycle
        noise = np.random.normal(0, 0.1)
        
        metrics = {
            "cpu_usage": max(10, min(95, 50 + 30 * daily_pattern + 10 * weekly_pattern + noise * 15)),
            "memory_usage": max(15, min(90, 45 + 25 * daily_pattern + 15 * weekly_pattern + noise * 12)),
            "gpu_usage": max(0, min(100, 30 + 40 * daily_pattern + 20 * weekly_pattern + noise * 20)),
            "request_rate": max(10, 500 + 300 * daily_pattern + 100 * weekly_pattern + noise * 50),
            "response_time": max(50, 300 + 200 * (-daily_pattern) + 100 * (-weekly_pattern) + noise * 80),
            "queue_depth": max(0, int(20 + 15 * (-daily_pattern) + 10 * (-weekly_pattern) + noise * 8)),
            "error_rate": max(0, min(10, 1 + 2 * (-daily_pattern) + 1 * (-weekly_pattern) + noise * 0.5)),
            "throughput": max(50, 800 + 400 * daily_pattern + 200 * weekly_pattern + noise * 100),
            "active_instances": max(1, int(self.ml_modules[module_name]["min_instances"] + 
                                         2 * daily_pattern + 1 * weekly_pattern + noise)),
            "target_instances": max(1, int(self.ml_modules[module_name]["min_instances"] + 
                                         3 * daily_pattern + 1 * weekly_pattern))
        }
        
        return metrics
    
    def _scaling_decision_loop(self):
        """Main scaling decision loop"""
        while self.scaling_active:
            try:
                for module_name in self.ml_modules:
                    if module_name in self.current_metrics:
                        decision = self._make_scaling_decision(module_name)
                        if decision["action"] != "no_action":
                            self._execute_scaling_operation(module_name, decision)
                
                time.sleep(30)  # Make scaling decisions every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in scaling decisions: {e}")
                time.sleep(60)
    
    def _make_scaling_decision(self, module_name: str) -> Dict[str, Any]:
        """Make intelligent scaling decision for a module"""
        metrics = self.current_metrics[module_name]
        module_config = self.ml_modules[module_name]
        thresholds = self.scaling_config["scaling_thresholds"]
        
        decision = {
            "action": "no_action",
            "reason": "metrics_within_normal_range",
            "target_instances": metrics.active_instances,
            "confidence": 0.5,
            "cost_impact": 0.0,
            "priority": "normal"
        }
        
        # Check for scale-up conditions
        scale_up_score = 0
        scale_up_reasons = []
        
        if metrics.cpu_usage > thresholds["cpu_scale_up"]:
            scale_up_score += 3
            scale_up_reasons.append(f"high_cpu_{metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > thresholds["memory_scale_up"]:
            scale_up_score += 3
            scale_up_reasons.append(f"high_memory_{metrics.memory_usage:.1f}%")
        
        if metrics.response_time > thresholds["response_time_scale_up"]:
            scale_up_score += 4
            scale_up_reasons.append(f"high_latency_{metrics.response_time:.0f}ms")
        
        if metrics.queue_depth > thresholds["queue_depth_scale_up"]:
            scale_up_score += 2
            scale_up_reasons.append(f"high_queue_{metrics.queue_depth}")
        
        if metrics.error_rate > thresholds["error_rate_scale_up"]:
            scale_up_score += 5
            scale_up_reasons.append(f"high_errors_{metrics.error_rate:.1f}%")
        
        # Check for scale-down conditions
        scale_down_score = 0
        scale_down_reasons = []
        
        if (metrics.cpu_usage < thresholds["cpu_scale_down"] and 
            metrics.memory_usage < thresholds["memory_scale_down"] and
            metrics.active_instances > module_config["min_instances"]):
            scale_down_score += 2
            scale_down_reasons.append("low_resource_utilization")
        
        if metrics.queue_depth == 0 and metrics.active_instances > module_config["min_instances"]:
            scale_down_score += 1
            scale_down_reasons.append("empty_queue")
        
        # Apply predictive scaling if enabled
        if self.scaling_config["prediction_enabled"]:
            prediction = self._get_predictive_scaling_recommendation(module_name)
            if prediction["confidence"] > 0.7:
                if prediction["action"] == "scale_up":
                    scale_up_score += 2
                    scale_up_reasons.append("predictive_demand_increase")
                elif prediction["action"] == "scale_down":
                    scale_down_score += 1
                    scale_down_reasons.append("predictive_demand_decrease")
        
        # Cost optimization check
        if self.scaling_config["cost_optimization"]:
            cost_factor = self._calculate_cost_factor()
            if cost_factor > 0.8:  # High cost pressure
                scale_down_score += 1
                scale_down_reasons.append("cost_optimization")
        
        # Make final decision
        if scale_up_score >= 3:
            target_instances = min(
                metrics.active_instances + self._calculate_scale_amount(scale_up_score),
                module_config["max_instances"]
            )
            
            decision.update({
                "action": "scale_up",
                "reason": "; ".join(scale_up_reasons),
                "target_instances": target_instances,
                "confidence": min(0.95, 0.5 + scale_up_score * 0.1),
                "priority": "high" if scale_up_score >= 6 else "normal"
            })
        
        elif scale_down_score >= 2 and self._check_scale_down_cooldown(module_name):
            target_instances = max(
                metrics.active_instances - 1,
                module_config["min_instances"]
            )
            
            decision.update({
                "action": "scale_down",
                "reason": "; ".join(scale_down_reasons),
                "target_instances": target_instances,
                "confidence": min(0.9, 0.6 + scale_down_score * 0.1),
                "priority": "low"
            })
        
        # Calculate cost impact
        decision["cost_impact"] = self._calculate_scaling_cost_impact(
            module_name, metrics.active_instances, decision["target_instances"]
        )
        
        return decision
    
    def _calculate_scale_amount(self, scale_score: int) -> int:
        """Calculate how many instances to add based on scaling score"""
        if scale_score >= 8:
            return 3  # Emergency scaling
        elif scale_score >= 6:
            return 2  # High demand
        else:
            return 1  # Normal scaling
    
    def _check_scale_down_cooldown(self, module_name: str) -> bool:
        """Check if module is in scale-down cooldown period"""
        cooldown_period = timedelta(seconds=self.scaling_config["scale_down_cooldown"])
        current_time = datetime.now()
        
        for event in reversed(self.scaling_history):
            if (event.module_name == module_name and 
                event.action == "scale_down" and
                current_time - event.timestamp < cooldown_period):
                return False
        
        return True
    
    def _get_predictive_scaling_recommendation(self, module_name: str) -> Dict[str, Any]:
        """Get predictive scaling recommendation using ML models"""
        
        # Simulate predictive model inference
        current_metrics = self.current_metrics[module_name]
        
        # Time-based prediction
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 18:  # Business hours
            demand_multiplier = 1.5
        elif 19 <= current_hour <= 22:  # Evening
            demand_multiplier = 1.2
        else:  # Night
            demand_multiplier = 0.7
        
        # Calculate predicted load
        predicted_cpu = current_metrics.cpu_usage * demand_multiplier
        predicted_memory = current_metrics.memory_usage * demand_multiplier
        predicted_requests = current_metrics.request_rate * demand_multiplier
        
        # Generate recommendation
        if predicted_cpu > 80 or predicted_memory > 85 or predicted_requests > current_metrics.request_rate * 1.3:
            return {
                "action": "scale_up",
                "confidence": 0.8,
                "predicted_cpu": predicted_cpu,
                "predicted_memory": predicted_memory,
                "time_horizon": 30  # minutes
            }
        elif predicted_cpu < 30 and predicted_memory < 40 and predicted_requests < current_metrics.request_rate * 0.7:
            return {
                "action": "scale_down",
                "confidence": 0.75,
                "predicted_cpu": predicted_cpu,
                "predicted_memory": predicted_memory,
                "time_horizon": 30
            }
        else:
            return {
                "action": "maintain",
                "confidence": 0.85,
                "predicted_cpu": predicted_cpu,
                "predicted_memory": predicted_memory,
                "time_horizon": 30
            }
    
    def _calculate_cost_factor(self) -> float:
        """Calculate current cost pressure factor (0.0 to 1.0)"""
        daily_budget = self.scaling_config["cost_budget_daily"]
        current_cost = self.cost_tracker["daily_cost"]
        
        if daily_budget <= 0:
            return 0.5
        
        return min(1.0, current_cost / daily_budget)
    
    def _calculate_scaling_cost_impact(self, module_name: str, current_instances: int, target_instances: int) -> float:
        """Calculate cost impact of scaling operation"""
        module_config = self.ml_modules[module_name]
        instance_delta = target_instances - current_instances
        
        # Estimate hourly cost per instance
        cpu_cost = module_config["cpu_per_instance"] * 0.10
        memory_cost = module_config["memory_per_instance"] * 0.05
        instance_cost = cpu_cost + memory_cost
        
        return instance_delta * instance_cost
    
    def _execute_scaling_operation(self, module_name: str, decision: Dict[str, Any]):
        """Execute the scaling operation"""
        start_time = time.time()
        current_metrics = self.current_metrics[module_name]
        
        old_config = {
            "instances": current_metrics.active_instances,
            "cpu_allocation": self.ml_modules[module_name]["cpu_per_instance"] * current_metrics.active_instances,
            "memory_allocation": self.ml_modules[module_name]["memory_per_instance"] * current_metrics.active_instances
        }
        
        # Simulate scaling operation
        success = self._perform_scaling_action(module_name, decision)
        
        new_config = {
            "instances": decision["target_instances"],
            "cpu_allocation": self.ml_modules[module_name]["cpu_per_instance"] * decision["target_instances"],
            "memory_allocation": self.ml_modules[module_name]["memory_per_instance"] * decision["target_instances"]
        }
        
        # Record scaling event
        event = ScalingEvent(
            event_id=f"{module_name}_{int(time.time())}",
            timestamp=datetime.now(),
            module_name=module_name,
            action=decision["action"],
            trigger_reason=decision["reason"],
            old_config=old_config,
            new_config=new_config,
            success=success,
            duration_seconds=time.time() - start_time,
            cost_impact=decision["cost_impact"]
        )
        
        self.scaling_history.append(event)
        
        # Update cost tracker
        self.cost_tracker["daily_cost"] += abs(decision["cost_impact"])
        
        self.logger.info(f"Scaling operation completed: {event.action} for {module_name} - Success: {success}")
    
    def _perform_scaling_action(self, module_name: str, decision: Dict[str, Any]) -> bool:
        """Perform the actual scaling action (simulate infrastructure calls)"""
        
        # Simulate different scaling operations
        if decision["action"] == "scale_up":
            # Simulate container/instance provisioning
            time.sleep(0.1)  # Simulate API call delay
            success_rate = 0.95
            
        elif decision["action"] == "scale_down":
            # Simulate graceful shutdown and deprovisioning
            time.sleep(0.05)  # Simulate API call delay
            success_rate = 0.98
            
        else:
            return True
        
        # Simulate occasional failures
        return np.random.random() < success_rate
    
    def _predictive_scaling_loop(self):
        """Advanced predictive scaling analysis"""
        while self.scaling_active:
            try:
                self._update_scaling_models()
                self._generate_capacity_forecasts()
                time.sleep(300)  # Update predictions every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in predictive scaling: {e}")
                time.sleep(600)
    
    def _update_scaling_models(self):
        """Update predictive scaling models with recent data"""
        
        for model_id, model in self.scaling_models.items():
            # Simulate model retraining with recent metrics
            if datetime.now() - model.last_trained > timedelta(hours=6):
                # Simulate model accuracy improvement
                accuracy_improvement = np.random.normal(0.01, 0.005)
                model.accuracy_score = min(0.99, model.accuracy_score + accuracy_improvement)
                model.last_trained = datetime.now()
                
                self.logger.info(f"Updated scaling model {model_id}: accuracy = {model.accuracy_score:.3f}")
    
    def _generate_capacity_forecasts(self):
        """Generate capacity forecasts for resource planning"""
        
        forecasts = {}
        
        for module_name in self.ml_modules:
            if module_name in self.current_metrics:
                current = self.current_metrics[module_name]
                
                # Generate 24-hour forecast
                hourly_forecast = []
                for hour in range(24):
                    # Simulate demand patterns
                    if 8 <= hour <= 18:  # Business hours
                        demand_factor = 1.0 + 0.5 * np.sin(np.pi * (hour - 8) / 10)
                    else:  # Off hours
                        demand_factor = 0.3 + 0.2 * np.random.random()
                    
                    predicted_cpu = current.cpu_usage * demand_factor
                    predicted_instances = max(
                        self.ml_modules[module_name]["min_instances"],
                        int(predicted_cpu / 70)  # Target 70% CPU utilization
                    )
                    
                    hourly_forecast.append({
                        "hour": hour,
                        "predicted_cpu": predicted_cpu,
                        "predicted_instances": predicted_instances,
                        "confidence": 0.8
                    })
                
                forecasts[module_name] = hourly_forecast
        
        # Store forecasts for dashboard and planning
        self.capacity_forecasts = forecasts
    
    def _cost_monitoring_loop(self):
        """Monitor and optimize costs"""
        while self.scaling_active:
            try:
                self._update_cost_tracking()
                self._optimize_resource_allocation()
                time.sleep(1800)  # Cost analysis every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cost monitoring: {e}")
                time.sleep(3600)
    
    def _update_cost_tracking(self):
        """Update cost tracking and reset daily counters"""
        current_time = datetime.now()
        
        # Reset daily cost tracking
        if current_time.date() > self.cost_tracker["last_reset"].date():
            self.cost_tracker["daily_cost"] = 0.0
            self.cost_tracker["last_reset"] = current_time
        
        # Calculate current resource costs
        total_cost = 0.0
        for module_name, metrics in self.current_metrics.items():
            module_config = self.ml_modules[module_name]
            
            # Calculate hourly cost for current instances
            cpu_cost = module_config["cpu_per_instance"] * metrics.active_instances * 0.10
            memory_cost = module_config["memory_per_instance"] * metrics.active_instances * 0.05
            
            hourly_cost = cpu_cost + memory_cost
            total_cost += hourly_cost
        
        # Update hourly cost accumulation
        self.cost_tracker["current_hourly_rate"] = total_cost
    
    def _optimize_resource_allocation(self):
        """Optimize resource allocation for cost efficiency"""
        
        # Identify cost optimization opportunities
        optimization_opportunities = []
        
        for module_name, metrics in self.current_metrics.items():
            efficiency_score = self._calculate_resource_efficiency(module_name)
            
            if efficiency_score < 0.6:  # Low efficiency
                optimization_opportunities.append({
                    "module": module_name,
                    "efficiency": efficiency_score,
                    "potential_savings": self._calculate_potential_savings(module_name),
                    "recommendation": self._generate_optimization_recommendation(module_name)
                })
        
        # Apply cost optimizations if significant savings available
        for opportunity in optimization_opportunities:
            if opportunity["potential_savings"] > 10.0:  # $10+ potential savings
                self.logger.info(f"Cost optimization opportunity: {opportunity}")
    
    def _calculate_resource_efficiency(self, module_name: str) -> float:
        """Calculate resource efficiency score for a module"""
        metrics = self.current_metrics[module_name]
        
        # Efficiency based on utilization vs. cost
        cpu_efficiency = min(1.0, metrics.cpu_usage / 70.0)  # Target 70% utilization
        memory_efficiency = min(1.0, metrics.memory_usage / 75.0)  # Target 75% utilization
        throughput_efficiency = min(1.0, metrics.throughput / 1000.0)  # Target throughput
        
        # Response time penalty
        response_penalty = max(0, (metrics.response_time - 500) / 1000)  # Penalty for >500ms
        
        efficiency = (cpu_efficiency + memory_efficiency + throughput_efficiency) / 3 - response_penalty
        return max(0, min(1, efficiency))
    
    def _resource_optimization_loop(self):
        """Continuous resource optimization"""
        while self.scaling_active:
            try:
                self._optimize_resource_pools()
                self._balance_workload_distribution()
                time.sleep(600)  # Optimization every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in resource optimization: {e}")
                time.sleep(1200)
    
    def _optimize_resource_pools(self):
        """Optimize resource pool allocation"""
        
        for pool_id, pool in self.resource_pools.items():
            utilization_rate = pool.allocated_capacity / pool.total_capacity
            
            # Rebalance if utilization is too high or low
            if utilization_rate > 0.9:
                # High utilization - consider expanding pool
                expansion_amount = pool.total_capacity * 0.2
                pool.total_capacity += expansion_amount
                pool.available_capacity += expansion_amount
                
                self.logger.info(f"Expanded resource pool {pool_id} by {expansion_amount} units")
                
            elif utilization_rate < 0.3:
                # Low utilization - consider shrinking pool
                reduction_amount = min(pool.total_capacity * 0.1, pool.available_capacity)
                pool.total_capacity -= reduction_amount
                pool.available_capacity -= reduction_amount
                
                self.logger.info(f"Reduced resource pool {pool_id} by {reduction_amount} units")
    
    def _balance_workload_distribution(self):
        """Balance workload distribution across resource pools"""
        
        # Calculate current load distribution
        load_distribution = {}
        for module_name, metrics in self.current_metrics.items():
            load_score = (metrics.cpu_usage + metrics.memory_usage) / 2
            load_distribution[module_name] = load_score
        
        # Identify load balancing opportunities
        avg_load = statistics.mean(load_distribution.values())
        high_load_modules = [m for m, load in load_distribution.items() if load > avg_load * 1.2]
        low_load_modules = [m for m, load in load_distribution.items() if load < avg_load * 0.8]
        
        # Simulate load balancing actions
        if high_load_modules and low_load_modules:
            self.logger.info(f"Load balancing opportunity identified: "
                           f"High load: {high_load_modules}, Low load: {low_load_modules}")
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get comprehensive scaling system summary"""
        
        current_time = datetime.now()
        
        # Calculate system-wide metrics
        total_instances = sum(m.active_instances for m in self.current_metrics.values())
        total_cost = self.cost_tracker.get("current_hourly_rate", 0.0)
        
        # Recent scaling activity
        recent_events = [e for e in self.scaling_history if 
                        current_time - e.timestamp < timedelta(hours=1)]
        
        # Resource utilization
        avg_cpu = statistics.mean([m.cpu_usage for m in self.current_metrics.values()])
        avg_memory = statistics.mean([m.memory_usage for m in self.current_metrics.values()])
        
        return {
            "system_overview": {
                "total_modules": len(self.ml_modules),
                "total_instances": total_instances,
                "current_hourly_cost": total_cost,
                "daily_cost": self.cost_tracker["daily_cost"],
                "average_cpu_utilization": avg_cpu,
                "average_memory_utilization": avg_memory
            },
            "recent_activity": {
                "scaling_events_last_hour": len(recent_events),
                "scale_up_events": len([e for e in recent_events if e.action == "scale_up"]),
                "scale_down_events": len([e for e in recent_events if e.action == "scale_down"]),
                "success_rate": statistics.mean([e.success for e in recent_events]) if recent_events else 1.0
            },
            "resource_pools": {
                pool_id: {
                    "utilization": pool.allocated_capacity / pool.total_capacity,
                    "available_capacity": pool.available_capacity,
                    "cost_per_unit": pool.cost_per_unit
                }
                for pool_id, pool in self.resource_pools.items()
            },
            "predictions": getattr(self, 'capacity_forecasts', {}),
            "cost_optimization": {
                "daily_budget": self.scaling_config["cost_budget_daily"],
                "budget_utilization": self.cost_tracker["daily_cost"] / self.scaling_config["cost_budget_daily"],
                "cost_trend": "increasing"  # Simplified
            }
        }
    
    def stop_scaling(self):
        """Stop auto-scaling system"""
        self.scaling_active = False
        self.logger.info("Auto-scaling system stopped")

def main():
    """Main function for standalone execution"""
    scaler = MLAutoScaling()
    
    try:
        while True:
            summary = scaler.get_scaling_summary()
            print(f"\n{'='*60}")
            print("ML AUTO-SCALING SYSTEM SUMMARY")
            print(f"{'='*60}")
            print(f"Total Instances: {summary['system_overview']['total_instances']}")
            print(f"Hourly Cost: ${summary['system_overview']['current_hourly_cost']:.2f}")
            print(f"Avg CPU: {summary['system_overview']['average_cpu_utilization']:.1f}%")
            print(f"Recent Events: {summary['recent_activity']['scaling_events_last_hour']}")
            print(f"{'='*60}")
            
            time.sleep(60)  # Print summary every minute
            
    except KeyboardInterrupt:
        scaler.stop_scaling()
        print("\nAuto-scaling system stopped.")

if __name__ == "__main__":
    main()