"""
Advanced ML Performance Optimization Engine
==========================================
"""Core Module - Split from performance_optimizer.py"""


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


logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of ML-driven optimizations"""
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    CACHE = "cache"
    NETWORK = "network"
    ALGORITHM = "algorithm"
    ML_MODEL = "ml_model"
    PREDICTION = "prediction"

class OptimizationLevel(Enum):
    """ML optimization aggressiveness levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ML_ADAPTIVE = "ml_adaptive"

class MLOptimizationStrategy(Enum):
    """ML-based optimization strategies"""
    PREDICTIVE_SCALING = "predictive_scaling"
    ADAPTIVE_TUNING = "adaptive_tuning"
    RESOURCE_LEARNING = "resource_learning"
    PERFORMANCE_CLUSTERING = "performance_clustering"
    ANOMALY_OPTIMIZATION = "anomaly_optimization"

@dataclass
class MLOptimizationRule:
    """ML-enhanced performance optimization rule"""
    rule_id: str
    optimization_type: OptimizationType
    ml_strategy: MLOptimizationStrategy
    condition: Callable[[], bool]
    action: Callable[[], None]
    ml_predictor: Optional[Any] = None
    priority: int = 5
    cooldown_seconds: int = 300
    max_applications: int = 10
    confidence_threshold: float = 0.7
    description: str = ""
    
@dataclass
class PerformanceMetric:
    """Enhanced performance metric with ML features"""
    metric_name: str
    value: float
    timestamp: datetime
    ml_features: List[float] = field(default_factory=list)
    anomaly_score: float = 0.0
    trend_direction: str = "stable"
    prediction_confidence: float = 0.0
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    unit: str = ""
    component: str = "system"

@dataclass
class MLOptimizationResult:
    """Result of ML-driven optimization"""
    rule_id: str
    timestamp: datetime
    success: bool
    improvement_percent: float
    ml_confidence: float
    optimization_strategy: MLOptimizationStrategy
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    ml_analysis: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class AdvancedMLPerformanceOptimizer:
    """
    Enterprise ML-driven performance optimization engine with
    intelligent tuning, predictive scaling, and adaptive algorithms.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.ML_ADAPTIVE,
                 monitoring_interval: int = 60):
        """
        Initialize ML performance optimizer.
        
        Args:
            optimization_level: Level of ML optimization aggressiveness
            monitoring_interval: Interval between optimization checks
        """
        self.optimization_level = optimization_level
        self.monitoring_interval = monitoring_interval
        
        # ML models for optimization
        self.ml_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.clusterers: Dict[str, KMeans] = {}
        
        # Optimization tracking
        self.optimization_rules: Dict[str, MLOptimizationRule] = {}
        self.performance_metrics: defaultdict = defaultdict(deque)
        self.optimization_history: deque = deque(maxlen=1000)
        self.rule_applications: defaultdict = defaultdict(int)
        self.rule_last_applied: Dict[str, datetime] = {}
        
        # ML performance baselines and predictions
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.performance_predictions: Dict[str, List[Tuple[datetime, float]]] = {}
        self.anomaly_scores: Dict[str, float] = {}
        
        # Analytics components references
        self.analytics_aggregator = None
        self.cache_systems: List[Any] = []
        self.data_stores: List[Any] = []
        self.processing_pipelines: List[Any] = []
        
        # ML optimization statistics
        self.optimizer_stats = {
            'ml_optimizations_applied': 0,
            'predictive_optimizations': 0,
            'adaptive_optimizations': 0,
            'ml_accuracy_score': 0.0,
            'performance_improvements': 0,
            'total_cpu_saved_percent': 0,
            'total_memory_saved_mb': 0,
            'ml_predictions_made': 0,
            'anomalies_detected': 0,
            'start_time': datetime.now()
        }
        
        # Threading and async
        self.optimizer_active = False
        self.optimization_thread: Optional[threading.Thread] = None
        self.ml_training_thread: Optional[threading.Thread] = None
        self.prediction_thread: Optional[threading.Thread] = None
        
        # ML configuration
        self.ml_config = {
            "feature_window_size": 10,
            "prediction_horizon_minutes": 30,
            "anomaly_threshold": 0.8,
            "cluster_count": 5,
            "model_retrain_interval_hours": 6,
            "min_training_samples": 20,
            "ml_confidence_threshold": 0.7
        }
        
        self._setup_ml_optimization_rules()
        
        logger.info(f"Advanced ML Performance Optimizer initialized: {optimization_level.value}")
    
    def start_ml_optimization(self):
        """Start ML-driven performance optimization"""
        if self.optimizer_active:
            return
        
        self.optimizer_active = True
        
        # Start optimization threads
        self.optimization_thread = threading.Thread(target=self._ml_optimization_loop, daemon=True)
        self.ml_training_thread = threading.Thread(target=self._ml_training_loop, daemon=True)
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        
        self.optimization_thread.start()
        self.ml_training_thread.start()
        self.prediction_thread.start()
        
        logger.info("ML Performance Optimization started")
    
    def stop_ml_optimization(self):
        """Stop ML performance optimization"""
        self.optimizer_active = False
        
        # Wait for threads to finish
        for thread in [self.optimization_thread, self.ml_training_thread, self.prediction_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("ML Performance Optimization stopped")
    
    def _setup_ml_optimization_rules(self):
        """Setup ML-enhanced optimization rules"""
        
        # ML-driven memory optimization
        def ml_memory_condition():
            current_usage = psutil.virtual_memory().percent
            predicted_usage = self._predict_metric("memory_usage", current_usage)
            return predicted_usage > 85 or current_usage > 80
        
        def ml_memory_action():
            # ML-enhanced memory cleanup
            self._perform_ml_memory_optimization()
            logger.info("Applied ML-driven memory optimization")
        
        memory_rule = MLOptimizationRule(
            rule_id="ml_memory_optimization",
            optimization_type=OptimizationType.MEMORY,
            ml_strategy=MLOptimizationStrategy.PREDICTIVE_SCALING,
            condition=ml_memory_condition,
            action=ml_memory_action,
            priority=9,
            cooldown_seconds=180,
            max_applications=10,
            description="ML-driven predictive memory optimization"
        )
        
        # ML-driven CPU optimization
        def ml_cpu_condition():
            current_usage = psutil.cpu_percent(interval=1)
            anomaly_score = self._calculate_anomaly_score("cpu_usage", current_usage)
            return anomaly_score > self.ml_config["anomaly_threshold"] or current_usage > 75
        
        def ml_cpu_action():
            self._perform_ml_cpu_optimization()
            logger.info("Applied ML-driven CPU optimization")
        
        cpu_rule = MLOptimizationRule(
            rule_id="ml_cpu_optimization",
            optimization_type=OptimizationType.CPU,
            ml_strategy=MLOptimizationStrategy.ANOMALY_OPTIMIZATION,
            condition=ml_cpu_condition,
            action=ml_cpu_action,
            priority=8,
            cooldown_seconds=120,
            max_applications=8,
            description="ML anomaly-based CPU optimization"
        )
        
        # ML adaptive cache optimization
        def ml_cache_condition():
            # Use ML clustering to identify optimal cache patterns
            return self._should_optimize_cache_ml()
        
        def ml_cache_action():
            self._perform_ml_cache_optimization()
            logger.info("Applied ML adaptive cache optimization")
        
        cache_rule = MLOptimizationRule(
            rule_id="ml_cache_optimization",
            optimization_type=OptimizationType.CACHE,
            ml_strategy=MLOptimizationStrategy.ADAPTIVE_TUNING,
            condition=ml_cache_condition,
            action=ml_cache_action,
            priority=7,
            cooldown_seconds=300,
            max_applications=5,
            description="ML clustering-based cache optimization"
        )
        
        # ML algorithm optimization
        def ml_algorithm_condition():
            return self._should_optimize_algorithms_ml()
        
        def ml_algorithm_action():
            self._perform_ml_algorithm_optimization()
            logger.info("Applied ML algorithm optimization")
        
        algorithm_rule = MLOptimizationRule(
            rule_id="ml_algorithm_optimization",
            optimization_type=OptimizationType.ALGORITHM,
            ml_strategy=MLOptimizationStrategy.RESOURCE_LEARNING,
            condition=ml_algorithm_condition,
            action=ml_algorithm_action,
            priority=6,
            cooldown_seconds=600,
            max_applications=3,
            description="ML-driven algorithm optimization"
        )
        
        # Add rules based on optimization level
        if self.optimization_level in [OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE, OptimizationLevel.ML_ADAPTIVE]:
            self.optimization_rules[memory_rule.rule_id] = memory_rule
            self.optimization_rules[cache_rule.rule_id] = cache_rule
        
        if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.ML_ADAPTIVE]:
            self.optimization_rules[cpu_rule.rule_id] = cpu_rule
            self.optimization_rules[algorithm_rule.rule_id] = algorithm_rule
    
    def _ml_optimization_loop(self):
        """Main ML optimization monitoring loop"""
        while self.optimizer_active:
            try:
                time.sleep(self.monitoring_interval)
                
                # Collect current metrics with ML features
                current_metrics = self._collect_ml_enhanced_metrics()
                
                # Record metrics for ML analysis
                for metric_name, metric_data in current_metrics.items():
                    self._record_ml_metric(metric_name, metric_data)
                
                # Check ML optimization rules
                for rule_id, rule in self.optimization_rules.items():
                    if self._can_apply_ml_rule(rule):
                        if rule.condition():
                            result = self._apply_ml_optimization_rule(rule)
                            if result and result.success:
                                self.optimizer_stats['ml_optimizations_applied'] += 1
                                if result.improvement_percent > 0:
                                    self.optimizer_stats['performance_improvements'] += 1
                
            except Exception as e:
                logger.error(f"ML optimization loop error: {e}")
    
    def _ml_training_loop(self):
        """ML model training and updating loop"""
        while self.optimizer_active:
            try:
                time.sleep(3600)  # Train every hour
                
                # Train/update ML models
                self._train_prediction_models()
                self._update_anomaly_detectors()
                self._train_clustering_models()
                
                logger.info("ML models updated")
                
            except Exception as e:
                logger.error(f"ML training loop error: {e}")
    
    def _prediction_loop(self):
        """ML prediction generation loop"""
        while self.optimizer_active:
            try:
                time.sleep(300)  # Predict every 5 minutes
                
                # Generate performance predictions
                self._generate_performance_predictions()
                
                # Update anomaly scores
                self._update_anomaly_scores()
                
            except Exception as e:
                logger.error(f"ML prediction loop error: {e}")
    
    def _collect_ml_enhanced_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect metrics with ML feature engineering"""
        metrics = {}
        
        try:
            # System metrics with ML features
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Create ML features (time-based patterns, trends)
            current_time = datetime.now()
            hour_of_day = current_time.hour
            day_of_week = current_time.weekday()
            
            metrics['cpu_usage'] = {
                'value': cpu_usage,
                'ml_features': [cpu_usage, hour_of_day, day_of_week],
                'timestamp': current_time
            }
            
            metrics['memory_usage'] = {
                'value': memory_usage,
                'ml_features': [memory_usage, memory.available / (1024**3), hour_of_day],
                'timestamp': current_time
            }
            
            # Analytics-specific metrics
            if self.analytics_aggregator:
                start_time = time.time()
                try:
                    # Test analytics response time
                    response_time = (time.time() - start_time) * 1000
                    metrics['analytics_response_time'] = {
                        'value': response_time,
                        'ml_features': [response_time, cpu_usage, memory_usage],
                        'timestamp': current_time
                    }