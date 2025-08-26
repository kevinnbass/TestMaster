"""
Advanced ML Delivery Optimization Engine
=======================================
"""Core Module - Split from delivery_optimizer.py"""


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


logger = logging.getLogger(__name__)

class DeliveryStatus(Enum):
    """ML-enhanced delivery status tracking"""
    PENDING = "pending"
    OPTIMIZING = "optimizing"
    ROUTING = "routing"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    ML_PREDICTED_FAILURE = "ml_predicted_failure"

class DeliveryPriority(Enum):
    """ML-driven delivery prioritization"""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10
    ML_URGENT = 15

class OptimizationStrategy(Enum):
    """ML optimization strategies"""
    PREDICTIVE_ROUTING = "predictive_routing"
    ADAPTIVE_RETRY = "adaptive_retry"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    FAILURE_PREVENTION = "failure_prevention"
    LOAD_BALANCING = "load_balancing"

@dataclass
class MLDeliveryRecord:
    """ML-enhanced delivery tracking record"""
    delivery_id: str
    data: Dict[str, Any]
    target: str
    status: DeliveryStatus
    priority: DeliveryPriority
    attempts: int
    created_at: datetime
    
    # ML features
    ml_features: List[float] = field(default_factory=list)
    predicted_delivery_time: Optional[float] = None
    failure_probability: float = 0.0
    optimization_strategy: Optional[OptimizationStrategy] = None
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Timing data
    last_attempt: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # ML analysis
    delivery_complexity: float = 0.0
    route_efficiency: float = 0.0
    error_message: Optional[str] = None

@dataclass
class DeliveryMetrics:
    """ML-driven delivery performance metrics"""
    timestamp: datetime
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    average_delivery_time: float
    ml_accuracy_score: float
    optimization_effectiveness: float
    resource_utilization: Dict[str, float] = field(default_factory=dict)

class AdvancedMLDeliveryOptimizer:
    """
    Advanced ML-driven delivery optimization engine with intelligent
    routing, predictive failure prevention, and adaptive resource management.
    """
    
    def __init__(self, max_workers: int = 4, ml_enabled: bool = True):
        """
        Initialize ML delivery optimizer.
        
        Args:
            max_workers: Maximum number of delivery workers
            ml_enabled: Enable ML optimization features
        """
        self.max_workers = max_workers
        self.ml_enabled = ml_enabled
        
        # ML models for optimization
        self.delivery_time_predictor: Optional[RandomForestRegressor] = None
        self.failure_predictor: Optional[LogisticRegression] = None
        self.route_optimizer: Optional[KMeans] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Delivery management
        self.delivery_queue = queue.PriorityQueue(maxsize=2000)
        self.delivery_records: Dict[str, MLDeliveryRecord] = {}
        self.delivery_history: deque = deque(maxlen=5000)
        self.failed_deliveries: deque = deque(maxlen=1000)
        
        # ML optimization tracking
        self.optimization_history: deque = deque(maxlen=1000)
        self.delivery_metrics: deque = deque(maxlen=1000)
        self.ml_features_history: deque = deque(maxlen=2000)
        
        # Delivery routing and optimization
        self.active_routes: Dict[str, List[str]] = {}
        self.route_performance: Dict[str, float] = {}
        self.target_load_balancing: Dict[str, float] = defaultdict(float)
        
        # Worker management
        self.delivery_workers: List[threading.Thread] = []
        self.ml_optimization_worker: Optional[threading.Thread] = None
        self.workers_active = False
        self.worker_performance: Dict[str, Dict[str, float]] = {}
        
        # ML configuration
        self.ml_config = {
            "prediction_horizon_minutes": 30,
            "failure_threshold": 0.7,
            "optimization_interval_seconds": 120,
            "min_training_samples": 50,
            "model_retrain_hours": 8,
            "feature_window_size": 10,
            "route_optimization_threshold": 0.8,
            "resource_utilization_target": 0.85
        }
        
        # Performance statistics
        self.delivery_stats = {
            'total_deliveries': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'ml_optimizations': 0,
            'route_optimizations': 0,
            'failure_predictions': 0,
            'prevented_failures': 0,
            'average_delivery_time': 0.0,
            'ml_accuracy': 0.0,
            'start_time': datetime.now()
        }
        
        # Data processing for ML optimization
        self.data_processors = {
            'feature_extractor': self._extract_delivery_features,
            'complexity_analyzer': self._analyze_delivery_complexity,
            'route_analyzer': self._analyze_route_efficiency,
            'resource_predictor': self._predict_resource_requirements
        }
        
        self._initialize_ml_models()
        
        logger.info("Advanced ML Delivery Optimizer initialized")
    
    def start_delivery_service(self):
        """Start ML-enhanced delivery service"""
        if self.workers_active:
            return
        
        self.workers_active = True
        
        # Start delivery workers
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._ml_delivery_worker,
                name=f"MLDeliveryWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.delivery_workers.append(worker)
        
        # Start ML optimization worker
        if self.ml_enabled:
            self.ml_optimization_worker = threading.Thread(
                target=self._ml_optimization_loop,
                name="MLOptimizationWorker",
                daemon=True
            )
            self.ml_optimization_worker.start()
        
        logger.info(f"ML Delivery service started with {self.max_workers} workers")
    
    def stop_delivery_service(self):
        """Stop ML delivery service"""
        self.workers_active = False
        
        # Wait for workers to finish
        for worker in self.delivery_workers:
            if worker.is_alive():
                worker.join(timeout=5)
        
        if self.ml_optimization_worker and self.ml_optimization_worker.is_alive():
            self.ml_optimization_worker.join(timeout=5)
        
        self.delivery_workers.clear()
        logger.info("ML Delivery service stopped")
    
    def queue_delivery(self, data: Dict[str, Any], target: str = "dashboard",
                      priority: DeliveryPriority = DeliveryPriority.MEDIUM) -> str:
        """Queue delivery with ML optimization"""
        delivery_id = f"ml_del_{int(time.time() * 1000000)}"
        
        # Extract ML features for optimization
        ml_features = self._extract_delivery_features(data, target)
        
        # Predict delivery characteristics
        predicted_time = self._predict_delivery_time(ml_features, target)
        failure_probability = self._predict_failure_probability(ml_features, target)
        complexity = self._analyze_delivery_complexity(data)
        
        # Select optimization strategy
        optimization_strategy = self._select_optimization_strategy(
            failure_probability, complexity, target
        )
        
        # Predict resource requirements
        resource_allocation = self._predict_resource_requirements(ml_features, target)
        
        # Create ML delivery record
        record = MLDeliveryRecord(
            delivery_id=delivery_id,
            data=self._process_data_for_ml_delivery(data),
            target=target,
            status=DeliveryStatus.PENDING,
            priority=priority,
            attempts=0,
            created_at=datetime.now(),
            ml_features=ml_features,
            predicted_delivery_time=predicted_time,
            failure_probability=failure_probability,
            optimization_strategy=optimization_strategy,
            resource_allocation=resource_allocation,
            delivery_complexity=complexity,
            estimated_completion=datetime.now() + timedelta(seconds=predicted_time or 10)
        )
        
        # Adjust priority based on ML analysis
        if failure_probability > 0.8:
            record.priority = DeliveryPriority.ML_URGENT
        elif predicted_time and predicted_time > 30:
            record.priority = DeliveryPriority.HIGH
        
        try:
            # Queue with ML-enhanced priority
            priority_score = self._calculate_ml_priority(record)
            self.delivery_queue.put_nowait((priority_score, record))
            self.delivery_records[delivery_id] = record
            self.delivery_stats['total_deliveries'] += 1
            
            logger.debug(f"Queued ML delivery {delivery_id} with strategy {optimization_strategy.value}")
            return delivery_id
            
        except queue.Full:
            logger.warning("ML Delivery queue is full, applying emergency optimization")
            self._emergency_queue_optimization()
            return None
    
    def _ml_delivery_worker(self):
        """ML-enhanced delivery worker thread"""
        worker_id = threading.current_thread().name
        self.worker_performance[worker_id] = {
            'deliveries_completed': 0,
            'success_rate': 0.0,
            'average_time': 0.0,
            'ml_optimizations_applied': 0
        }
        
        while self.workers_active:
            try:
                # Get delivery with timeout
                priority_score, record = self.delivery_queue.get(timeout=1)
                
                # Apply ML optimizations before delivery
                optimized_record = self._apply_ml_optimizations(record)
                
                # Attempt ML-enhanced delivery
                success = self._attempt_ml_delivery(optimized_record, worker_id)
                
                # Update record and statistics
                self._update_delivery_record(optimized_record, success, worker_id)
                
                # Learn from delivery outcome for ML improvement
                self._learn_from_delivery(optimized_record, success)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"ML delivery worker {worker_id} error: {e}")
    
    def _ml_optimization_loop(self):
        """Background ML optimization and learning loop"""
        while self.workers_active:
            try:
                time.sleep(self.ml_config["optimization_interval_seconds"])
                
                # Retrain ML models
                await self._retrain_ml_models()
                
                # Optimize routing strategies
                self._optimize_delivery_routes()
                
                # Analyze and optimize resource allocation
                self._optimize_resource_allocation()
                
                # Update ML configuration based on performance
                self._adaptive_ml_tuning()
                
                # Generate ML insights and recommendations
                self._generate_ml_insights()
                
            except Exception as e:
                logger.error(f"ML optimization loop error: {e}")
    
    def _extract_delivery_features(self, data: Dict[str, Any], target: str) -> List[float]:
        """Extract ML features from delivery data"""
        try:
            features = []
            
            # Data characteristics features
            data_size = len(str(data))
            features.append(float(data_size))
            
            complexity_score = self._calculate_data_complexity(data)
            features.append(complexity_score)
            
            # Target characteristics
            target_hash = hash(target) % 1000
            features.append(float(target_hash))
            
            # Current system load features
            current_time = datetime.now()
            features.append(float(current_time.hour))
            features.append(float(current_time.minute))
            features.append(float(current_time.weekday()))
            
            # Queue and system state features
            queue_size = self.delivery_queue.qsize()
            features.append(float(queue_size))
            
            active_deliveries = len([r for r in self.delivery_records.values() 
                                   if r.status in [DeliveryStatus.DELIVERING, DeliveryStatus.ROUTING]])
            features.append(float(active_deliveries))
            
            # Historical performance features
            recent_success_rate = self._calculate_recent_success_rate(target)
            features.append(recent_success_rate)
            
            recent_avg_time = self._calculate_recent_average_time(target)
            features.append(recent_avg_time)
            
            # Ensure consistent feature count
            while len(features) < 15:
                features.append(0.0)
            
            return features[:15]
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return [0.0] * 15
    
    def _calculate_data_complexity(self, data: Dict[str, Any]) -> float:
        """Calculate complexity score for data structure"""
        try:
            complexity = 0.0
            
            def analyze_structure(obj, depth=0):
                nonlocal complexity
                complexity += depth * 0.1
                
                if isinstance(obj, dict):
                    complexity += len(obj) * 0.1
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            analyze_structure(value, depth + 1)
                elif isinstance(obj, list):
                    complexity += len(obj) * 0.05
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            analyze_structure(item, depth + 1)
            
            analyze_structure(data)
            return min(complexity, 10.0)  # Cap at 10.0
            
        except Exception:
            return 1.0
    
    def _predict_delivery_time(self, features: List[float], target: str) -> Optional[float]:
        """Predict delivery time using ML"""
        try:
            if not self.delivery_time_predictor or len(features) < 10:
                return None
            
            # Scale features
            if 'delivery_time' in self.scalers:
                features_scaled = self.scalers['delivery_time'].transform([features])
            else:
                features_scaled = [features]
            
            # Make prediction
            prediction = self.delivery_time_predictor.predict(features_scaled)[0]
            self.delivery_stats['ml_predictions'] = self.delivery_stats.get('ml_predictions', 0) + 1
            
            return max(1.0, min(300.0, prediction))  # 1-300 seconds range
            
        except Exception as e:
            logger.debug(f"Delivery time prediction error: {e}")
            return None
    
    def _predict_failure_probability(self, features: List[float], target: str) -> float:
        """Predict failure probability using ML"""
        try:
            if not self.failure_predictor or len(features) < 10:
                return 0.1  # Default low probability
            
            # Scale features