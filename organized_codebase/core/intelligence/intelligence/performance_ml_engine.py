"""
Advanced Performance ML Engine - Core Module
===========================================

Enterprise-grade ML-driven performance optimization engine with intelligent
bottleneck detection, predictive scaling, adaptive optimization strategies,
and comprehensive performance analytics using advanced ML algorithms.

Part 1 of Performance Booster extraction - Core ML algorithms and optimization.
Coordinates with performance_execution_manager.py for complete system.

Author: TestMaster Intelligence Phase 2B
"""

import asyncio
import concurrent.futures
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import queue

# ML imports
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class PerformanceOptimizationStrategy(Enum):
    """ML-driven performance optimization strategies"""
    PREDICTIVE_CACHING = "predictive_caching"
    INTELLIGENT_PARALLELIZATION = "intelligent_parallelization"
    ADAPTIVE_RESOURCE_ALLOCATION = "adaptive_resource_allocation"
    BOTTLENECK_ELIMINATION = "bottleneck_elimination"
    PREDICTIVE_PRELOADING = "predictive_preloading"
    DYNAMIC_SCALING = "dynamic_scaling"
    PATTERN_OPTIMIZATION = "pattern_optimization"

class PerformanceMLAlgorithm(Enum):
    """ML algorithms for performance optimization"""
    RANDOM_FOREST_PREDICTOR = "random_forest_predictor"
    ISOLATION_FOREST_ANOMALY = "isolation_forest_anomaly"
    KMEANS_CLUSTERING = "kmeans_clustering"
    DBSCAN_PATTERN = "dbscan_pattern"
    RIDGE_REGRESSION = "ridge_regression"
    ELASTIC_NET_REGULARIZATION = "elastic_net_regularization"

class PerformanceBottleneckType(Enum):
    """Types of performance bottlenecks"""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_BOUND = "memory_bound"
    IO_LATENCY = "io_latency"
    NETWORK_OVERHEAD = "network_overhead"
    CACHE_MISSES = "cache_misses"
    SYNCHRONIZATION = "synchronization"
    ALGORITHM_INEFFICIENCY = "algorithm_inefficiency"

@dataclass
class MLPerformanceMetrics:
    """Advanced ML performance metrics with feature engineering"""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    
    # ML features
    ml_features: List[float] = field(default_factory=list)
    performance_score: float = 0.0
    optimization_potential: float = 0.0
    bottleneck_probability: Dict[str, float] = field(default_factory=dict)
    optimization_strategy: Optional[PerformanceOptimizationStrategy] = None
    
    # Advanced metrics
    cache_efficiency: float = 0.0
    parallel_efficiency: float = 0.0
    resource_utilization_score: float = 0.0
    prediction_accuracy: float = 0.0
    
    # Contextual data
    operation_complexity: float = 0.0
    data_size_factor: float = 0.0
    concurrency_level: int = 0
    optimization_applied: bool = False

@dataclass
class MLOptimizationPlan:
    """ML-generated optimization plan"""
    plan_id: str
    target_operation: str
    optimization_strategies: List[PerformanceOptimizationStrategy]
    expected_improvement_percent: float
    confidence_score: float
    ml_reasoning: List[str] = field(default_factory=list)
    
    # Resource predictions
    predicted_duration_ms: float = 0.0
    predicted_memory_mb: float = 0.0
    predicted_cpu_reduction: float = 0.0
    
    # Implementation details
    parallel_workers_recommended: int = 0
    cache_strategies: List[str] = field(default_factory=list)
    resource_adjustments: Dict[str, float] = field(default_factory=dict)
    preload_recommendations: List[str] = field(default_factory=list)

@dataclass
class PerformancePattern:
    """ML-identified performance pattern"""
    pattern_id: str
    pattern_type: str
    frequency: int
    operations: List[str]
    ml_cluster_id: int
    
    # Pattern characteristics
    average_duration_ms: float
    memory_pattern: List[float]
    cpu_pattern: List[float]
    optimization_opportunities: List[str] = field(default_factory=list)
    pattern_strength: float = 0.0
    predictability_score: float = 0.0

class AdvancedPerformanceMLEngine:
    """
    Advanced ML-driven performance optimization engine with sophisticated
    algorithms for bottleneck detection, predictive optimization, and
    intelligent resource management.
    """
    
    def __init__(self, ml_enabled: bool = True, optimization_interval: float = 60.0):
        """
        Initialize advanced performance ML engine.
        
        Args:
            ml_enabled: Enable ML optimization features
            optimization_interval: Interval for ML optimization cycles
        """
        self.ml_enabled = ml_enabled
        self.optimization_interval = optimization_interval
        
        # ML models for performance optimization
        self.duration_predictor: Optional[RandomForestRegressor] = None
        self.bottleneck_detector: Optional[IsolationForest] = None
        self.pattern_clusterer: Optional[KMeans] = None
        self.anomaly_detector: Optional[DBSCAN] = None
        self.resource_optimizer: Optional[Ridge] = None
        self.scaling_predictor: Optional[ElasticNet] = None
        
        # ML scalers and encoders
        self.scalers: Dict[str, StandardScaler] = {}
        self.normalizers: Dict[str, MinMaxScaler] = {}
        
        # Performance tracking and ML data
        self.performance_metrics: deque = deque(maxlen=10000)
        self.optimization_plans: Dict[str, MLOptimizationPlan] = {}
        self.performance_patterns: Dict[str, PerformancePattern] = {}
        self.ml_training_data: deque = deque(maxlen=5000)
        
        # ML feature engineering
        self.feature_extractors: Dict[str, Callable] = {}
        self.bottleneck_classifiers: Dict[str, Any] = {}
        self.optimization_history: deque = deque(maxlen=2000)
        
        # Performance baselines and predictions
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.optimization_predictions: Dict[str, float] = {}
        self.pattern_predictions: Dict[str, List[float]] = {}
        
        # ML configuration
        self.ml_config = {
            "prediction_horizon_minutes": 30,
            "bottleneck_detection_threshold": 0.8,
            "optimization_confidence_threshold": 0.7,
            "pattern_min_frequency": 5,
            "min_training_samples": 50,
            "model_retrain_hours": 8,
            "feature_window_size": 20,
            "clustering_min_samples": 10,
            "anomaly_contamination": 0.1,
            "optimization_target_improvement": 0.2
        }
        
        # Performance statistics
        self.ml_stats = {
            'total_optimizations': 0,
            'ml_predictions_made': 0,
            'bottlenecks_detected': 0,
            'patterns_identified': 0,
            'optimization_success_rate': 0.0,
            'average_improvement_percent': 0.0,
            'ml_model_accuracy': 0.0,
            'predictions_accurate': 0,
            'predictions_total': 0,
            'start_time': datetime.now()
        }
        
        # Background ML processing
        self.ml_engine_active = False
        self.ml_optimization_worker: Optional[threading.Thread] = None
        self.ml_training_worker: Optional[threading.Thread] = None
        self.ml_prediction_worker: Optional[threading.Thread] = None
        
        # Thread safety
        self.engine_lock = threading.RLock()
        
        self._initialize_ml_models()
        self._setup_feature_extractors()
        
        logger.info("Advanced Performance ML Engine initialized")
    
    def start_ml_performance_engine(self):
        """Start ML-driven performance optimization engine"""
        if self.ml_engine_active:
            return
        
        self.ml_engine_active = True
        
        # Start ML workers
        self.ml_optimization_worker = threading.Thread(
            target=self._ml_optimization_loop, daemon=True)
        self.ml_training_worker = threading.Thread(
            target=self._ml_training_loop, daemon=True)
        self.ml_prediction_worker = threading.Thread(
            target=self._ml_prediction_loop, daemon=True)
        
        self.ml_optimization_worker.start()
        self.ml_training_worker.start()
        self.ml_prediction_worker.start()
        
        logger.info("ML Performance Engine started")
    
    def stop_ml_performance_engine(self):
        """Stop ML performance engine"""
        self.ml_engine_active = False
        
        # Wait for workers to finish
        for worker in [self.ml_optimization_worker, self.ml_training_worker, self.ml_prediction_worker]:
            if worker and worker.is_alive():
                worker.join(timeout=5)
        
        logger.info("ML Performance Engine stopped")
    
    def analyze_performance_ml(self, operation_name: str, start_time: float, 
                              end_time: float, context_data: Dict[str, Any] = None) -> MLPerformanceMetrics:
        """Analyze performance with comprehensive ML features"""
        with self.engine_lock:
            duration_ms = (end_time - start_time) * 1000
            
            # Extract ML features
            ml_features = self._extract_performance_features(
                operation_name, duration_ms, context_data or {}
            )
            
            # Create performance metrics
            metrics = MLPerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_usage_mb=self._get_memory_usage(),
                cpu_utilization=self._get_cpu_utilization(),
                ml_features=ml_features
            )
            
            # ML analysis
            metrics.performance_score = self._calculate_performance_score(metrics)
            metrics.optimization_potential = self._calculate_optimization_potential(metrics)
            metrics.bottleneck_probability = self._detect_bottlenecks_ml(metrics)
            metrics.optimization_strategy = self._recommend_optimization_strategy(metrics)
            
            # Advanced ML metrics
            metrics.cache_efficiency = self._calculate_cache_efficiency(metrics)
            metrics.parallel_efficiency = self._calculate_parallel_efficiency(metrics)
            metrics.resource_utilization_score = self._calculate_resource_utilization_score(metrics)
            
            # Store for ML training
            self.performance_metrics.append(metrics)
            self._record_ml_training_data(metrics)
            
            # Update statistics
            self.ml_stats['total_optimizations'] += 1
            
            logger.debug(f"ML performance analysis: {operation_name} - Score: {metrics.performance_score:.2f}")
            return metrics
    
    def generate_optimization_plan_ml(self, operation_name: str, 
                                     target_improvement: float = 0.3) -> MLOptimizationPlan:
        """Generate ML-driven optimization plan"""
        try:
            # Analyze historical performance
            historical_metrics = self._get_historical_metrics(operation_name)
            
            if not historical_metrics:
                return self._create_default_optimization_plan(operation_name)
            
            # ML-driven analysis
            optimization_strategies = self._identify_optimization_strategies_ml(historical_metrics)
            expected_improvement = self._predict_improvement_ml(historical_metrics, optimization_strategies)
            confidence_score = self._calculate_optimization_confidence(historical_metrics, optimization_strategies)
            
            # Generate ML reasoning
            ml_reasoning = self._generate_ml_reasoning(historical_metrics, optimization_strategies)
            
            # Resource predictions
            predicted_duration = self._predict_optimized_duration(historical_metrics, optimization_strategies)
            predicted_memory = self._predict_optimized_memory(historical_metrics, optimization_strategies)
            predicted_cpu_reduction = self._predict_cpu_reduction(historical_metrics, optimization_strategies)
            
            # Implementation recommendations
            parallel_workers = self._recommend_parallel_workers(historical_metrics)
            cache_strategies = self._recommend_cache_strategies(historical_metrics)
            resource_adjustments = self._recommend_resource_adjustments(historical_metrics)
            preload_recommendations = self._recommend_preloading(historical_metrics)
            
            plan = MLOptimizationPlan(
                plan_id=f"opt_plan_{int(time.time())}",
                target_operation=operation_name,
                optimization_strategies=optimization_strategies,
                expected_improvement_percent=expected_improvement,
                confidence_score=confidence_score,
                ml_reasoning=ml_reasoning,
                predicted_duration_ms=predicted_duration,
                predicted_memory_mb=predicted_memory,
                predicted_cpu_reduction=predicted_cpu_reduction,
                parallel_workers_recommended=parallel_workers,
                cache_strategies=cache_strategies,
                resource_adjustments=resource_adjustments,
                preload_recommendations=preload_recommendations
            )
            
            self.optimization_plans[operation_name] = plan
            self.ml_stats['ml_predictions_made'] += 1
            
            logger.info(f"Generated ML optimization plan for {operation_name}: "
                       f"{expected_improvement:.1%} improvement expected")
            return plan
            
        except Exception as e:
            logger.error(f"ML optimization plan generation failed: {e}")
            return self._create_default_optimization_plan(operation_name)
    
    def detect_performance_patterns_ml(self) -> List[PerformancePattern]:
        """Detect performance patterns using ML clustering"""
        try:
            if len(self.performance_metrics) < self.ml_config["clustering_min_samples"]:
                return []
            
            # Prepare data for clustering
            features_matrix = []
            operation_names = []
            
            for metric in list(self.performance_metrics)[-500:]:  # Last 500 metrics
                if len(metric.ml_features) >= 10:
                    features_matrix.append(metric.ml_features[:10])
                    operation_names.append(metric.operation_name)
            
            if len(features_matrix) < self.ml_config["clustering_min_samples"]:
                return []
            
            # Scale features
            features_array = np.array(features_matrix)
            if 'pattern_clustering' not in self.scalers:
                self.scalers['pattern_clustering'] = StandardScaler()
                features_scaled = self.scalers['pattern_clustering'].fit_transform(features_array)
            else:
                features_scaled = self.scalers['pattern_clustering'].transform(features_array)
            
            # Perform clustering
            n_clusters = min(8, len(features_matrix) // 10)
            if n_clusters < 2:
                return []
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features_scaled)
            
            # Analyze patterns
            patterns = []
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                
                if len(cluster_indices) >= self.ml_config["pattern_min_frequency"]:
                    pattern = self._analyze_cluster_pattern(
                        cluster_id, cluster_indices, features_matrix, operation_names
                    )
                    if pattern:
                        patterns.append(pattern)
                        self.performance_patterns[pattern.pattern_id] = pattern
            
            self.ml_stats['patterns_identified'] += len(patterns)
            
            logger.info(f"Detected {len(patterns)} performance patterns using ML clustering")
            return patterns
            
        except Exception as e:
            logger.error(f"ML pattern detection failed: {e}")
            return []
    
    def predict_performance_ml(self, operation_name: str, context_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Predict performance metrics using ML"""
        try:
            if not self.duration_predictor:
                return {'predicted_duration_ms': 100.0, 'confidence': 0.5}
            
            # Extract features for prediction
            features = self._extract_prediction_features(operation_name, context_data or {})
            
            if len(features) < 10:
                return {'predicted_duration_ms': 100.0, 'confidence': 0.5}
            
            # Scale features
            if 'duration_prediction' in self.scalers:
                features_scaled = self.scalers['duration_prediction'].transform([features])
            else:
                features_scaled = [features]
            
            # Make predictions
            predicted_duration = self.duration_predictor.predict(features_scaled)[0]
            
            # Calculate confidence based on historical accuracy
            confidence = self._calculate_prediction_confidence(operation_name)
            
            # Additional ML predictions
            bottleneck_probability = self._predict_bottleneck_probability(features)
            optimization_potential = self._predict_optimization_potential(features)
            
            self.ml_stats['ml_predictions_made'] += 1
            
            return {
                'predicted_duration_ms': max(1.0, predicted_duration),
                'confidence': confidence,
                'bottleneck_probability': bottleneck_probability,
                'optimization_potential': optimization_potential,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"ML performance prediction failed: {e}")
            return {'predicted_duration_ms': 100.0, 'confidence': 0.5}
    
    def _extract_performance_features(self, operation_name: str, duration_ms: float, 
                                    context_data: Dict[str, Any]) -> List[float]:
        """Extract ML features for performance analysis"""
        try:
            features = []
            
            # Basic performance features
            features.append(duration_ms)
            features.append(self._get_memory_usage())
            features.append(self._get_cpu_utilization())
            
            # Operation characteristics
            operation_hash = hash(operation_name) % 1000
            features.append(float(operation_hash))
            
            # Temporal features
            current_time = datetime.now()
            features.append(float(current_time.hour))
            features.append(float(current_time.minute))
            features.append(float(current_time.weekday()))
            
            # Context-based features
            data_size = context_data.get('data_size', 0)
            complexity = context_data.get('complexity', 1.0)
            concurrency = context_data.get('concurrency_level', 1)
            features.extend([float(data_size), complexity, float(concurrency)])
            
            # Historical performance features
            historical_avg = self._get_historical_average_duration(operation_name)
            features.append(historical_avg)
            
            # System load features
            active_operations = len([m for m in self.performance_metrics 
                                   if (datetime.now() - datetime.fromtimestamp(m.end_time)).total_seconds() < 60])
            features.append(float(active_operations))
            
            # Pattern-based features
            pattern_score = self._calculate_pattern_similarity(operation_name)
            features.append(pattern_score)
            
            # Optimization history features
            recent_optimizations = len([o for o in self.optimization_history 
                                      if (datetime.now() - o.get('timestamp', datetime.min)).total_seconds() < 3600])
            features.append(float(recent_optimizations))
            
            # Ensure consistent feature count
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return [0.0] * 20
    
    def _detect_bottlenecks_ml(self, metrics: MLPerformanceMetrics) -> Dict[str, float]:
        """Detect bottlenecks using ML algorithms"""
        try:
            bottleneck_probabilities = {}
            
            # CPU bottleneck detection
            if metrics.cpu_utilization > 80:
                bottleneck_probabilities[PerformanceBottleneckType.CPU_INTENSIVE.value] = min(1.0, metrics.cpu_utilization / 100)
            
            # Memory bottleneck detection
            if metrics.memory_usage_mb > 1000:  # More than 1GB
                bottleneck_probabilities[PerformanceBottleneckType.MEMORY_BOUND.value] = min(1.0, metrics.memory_usage_mb / 2000)
            
            # Duration-based bottleneck detection
            if metrics.duration_ms > 1000:  # More than 1 second
                if metrics.cpu_utilization < 30:  # Low CPU suggests IO bound
                    bottleneck_probabilities[PerformanceBottleneckType.IO_LATENCY.value] = min(1.0, metrics.duration_ms / 5000)
                else:
                    bottleneck_probabilities[PerformanceBottleneckType.ALGORITHM_INEFFICIENCY.value] = min(1.0, metrics.duration_ms / 3000)
            
            # Use anomaly detection if available
            if self.bottleneck_detector and len(metrics.ml_features) >= 10:
                anomaly_score = self._calculate_anomaly_score(metrics.ml_features)
                if anomaly_score > self.ml_config["bottleneck_detection_threshold"]:
                    bottleneck_probabilities["anomaly_detected"] = anomaly_score
            
            if bottleneck_probabilities:
                self.ml_stats['bottlenecks_detected'] += 1
            
            return bottleneck_probabilities
            
        except Exception as e:
            logger.debug(f"Bottleneck detection error: {e}")
            return {}
    
    def _recommend_optimization_strategy(self, metrics: MLPerformanceMetrics) -> Optional[PerformanceOptimizationStrategy]:
        """Recommend optimization strategy based on ML analysis"""
        try:
            # Analyze bottlenecks and recommend strategy
            bottlenecks = metrics.bottleneck_probability
            
            if not bottlenecks:
                return PerformanceOptimizationStrategy.PATTERN_OPTIMIZATION
            
            max_bottleneck = max(bottlenecks.items(), key=lambda x: x[1])
            bottleneck_type, probability = max_bottleneck
            
            if probability < 0.5:
                return PerformanceOptimizationStrategy.PATTERN_OPTIMIZATION
            
            # Strategy mapping
            strategy_mapping = {
                PerformanceBottleneckType.CPU_INTENSIVE.value: PerformanceOptimizationStrategy.INTELLIGENT_PARALLELIZATION,
                PerformanceBottleneckType.MEMORY_BOUND.value: PerformanceOptimizationStrategy.ADAPTIVE_RESOURCE_ALLOCATION,
                PerformanceBottleneckType.IO_LATENCY.value: PerformanceOptimizationStrategy.PREDICTIVE_CACHING,
                PerformanceBottleneckType.CACHE_MISSES.value: PerformanceOptimizationStrategy.PREDICTIVE_PRELOADING,
                PerformanceBottleneckType.ALGORITHM_INEFFICIENCY.value: PerformanceOptimizationStrategy.BOTTLENECK_ELIMINATION
            }
            
            return strategy_mapping.get(bottleneck_type, PerformanceOptimizationStrategy.DYNAMIC_SCALING)
            
        except Exception as e:
            logger.debug(f"Strategy recommendation error: {e}")
            return PerformanceOptimizationStrategy.PATTERN_OPTIMIZATION
    
    def _calculate_performance_score(self, metrics: MLPerformanceMetrics) -> float:
        """Calculate ML-driven performance score"""
        try:
            score = 100.0
            
            # Duration penalty
            if metrics.duration_ms > 1000:
                score -= min(50, (metrics.duration_ms - 1000) / 100)
            
            # CPU utilization penalty
            if metrics.cpu_utilization > 70:
                score -= (metrics.cpu_utilization - 70) * 0.5
            
            # Memory usage penalty
            if metrics.memory_usage_mb > 500:
                score -= min(20, (metrics.memory_usage_mb - 500) / 50)
            
            # Bonus for optimization potential
            if metrics.optimization_potential > 0.5:
                score += metrics.optimization_potential * 10
            
            return max(0.0, min(100.0, score))
            
        except Exception:
            return 50.0
    
    def _calculate_optimization_potential(self, metrics: MLPerformanceMetrics) -> float:
        """Calculate optimization potential using ML"""
        try:
            potential = 0.0
            
            # High duration suggests optimization potential
            if metrics.duration_ms > 500:
                potential += min(0.5, (metrics.duration_ms - 500) / 2000)
            
            # High resource usage suggests optimization potential
            if metrics.cpu_utilization > 60:
                potential += min(0.3, (metrics.cpu_utilization - 60) / 100)
            
            if metrics.memory_usage_mb > 200:
                potential += min(0.2, (metrics.memory_usage_mb - 200) / 1000)
            
            # Pattern-based potential
            pattern_potential = self._calculate_pattern_optimization_potential(metrics.operation_name)
            potential += pattern_potential * 0.3
            
            return min(1.0, potential)
            
        except Exception:
            return 0.5
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024 * 1024)
        except Exception:
            return 100.0
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except Exception:
            return 50.0
    
    def _get_historical_metrics(self, operation_name: str, limit: int = 50) -> List[MLPerformanceMetrics]:
        """Get historical metrics for an operation"""
        return [m for m in list(self.performance_metrics)[-limit*2:] 
                if m.operation_name == operation_name][:limit]
    
    def _get_historical_average_duration(self, operation_name: str) -> float:
        """Get historical average duration for an operation"""
        try:
            historical = self._get_historical_metrics(operation_name, 20)
            if historical:
                return sum(m.duration_ms for m in historical) / len(historical)
            return 100.0
        except Exception:
            return 100.0
    
    def _calculate_pattern_similarity(self, operation_name: str) -> float:
        """Calculate pattern similarity score"""
        try:
            # Simple pattern similarity based on operation name frequency
            recent_operations = [m.operation_name for m in list(self.performance_metrics)[-100:]]
            frequency = recent_operations.count(operation_name)
            return min(1.0, frequency / 20)  # Normalize to 0-1
        except Exception:
            return 0.0
    
    def _calculate_pattern_optimization_potential(self, operation_name: str) -> float:
        """Calculate optimization potential based on patterns"""
        try:
            if operation_name in self.performance_patterns:
                pattern = self.performance_patterns[operation_name]
                return pattern.predictability_score * 0.8
            return 0.3  # Default potential
        except Exception:
            return 0.3
    
    def _setup_feature_extractors(self):
        """Setup ML feature extractors"""
        self.feature_extractors = {
            'duration': lambda m: m.duration_ms,
            'memory': lambda m: m.memory_usage_mb,
            'cpu': lambda m: m.cpu_utilization,
            'performance_score': lambda m: m.performance_score,
            'optimization_potential': lambda m: m.optimization_potential
        }
    
    def _initialize_ml_models(self):
        """Initialize ML models for performance optimization"""
        try:
            if self.ml_enabled:
                # Initialize with basic models
                self.duration_predictor = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
                self.bottleneck_detector = IsolationForest(contamination=0.1, random_state=42)
                self.pattern_clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)
                self.anomaly_detector = DBSCAN(eps=0.5, min_samples=5)
                self.resource_optimizer = Ridge(alpha=1.0)
                self.scaling_predictor = ElasticNet(alpha=1.0, l1_ratio=0.5)
                
                logger.info("ML models initialized for performance engine")
                
        except Exception as e:
            logger.warning(f"ML model initialization failed: {e}")
            self.ml_enabled = False
    
    # ========================================================================
    # BACKGROUND ML LOOPS
    # ========================================================================
    
    def _ml_optimization_loop(self):
        """Background ML optimization loop"""
        while self.ml_engine_active:
            try:
                time.sleep(self.optimization_interval)
                
                # Generate optimization plans for frequent operations
                await self._generate_proactive_optimization_plans()
                
                # Detect performance patterns
                self.detect_performance_patterns_ml()
                
                # Update performance baselines
                self._update_performance_baselines()
                
            except Exception as e:
                logger.error(f"ML optimization loop error: {e}")
    
    def _ml_training_loop(self):
        """Background ML model training loop"""
        while self.ml_engine_active:
            try:
                time.sleep(3600)  # Every hour
                
                # Retrain models with new data
                await self._retrain_ml_models()
                
                # Update optimization strategies
                self._update_optimization_strategies()
                
            except Exception as e:
                logger.error(f"ML training loop error: {e}")
    
    def _ml_prediction_loop(self):
        """Background ML prediction loop"""
        while self.ml_engine_active:
            try:
                time.sleep(300)  # Every 5 minutes
                
                # Generate performance predictions
                self._generate_performance_predictions()
                
                # Update prediction accuracy
                self._update_prediction_accuracy()
                
            except Exception as e:
                logger.error(f"ML prediction loop error: {e}")
    
    async def _generate_proactive_optimization_plans(self):
        """Generate proactive optimization plans"""
        try:
            # Find frequently used operations
            operation_counts = defaultdict(int)
            for metric in list(self.performance_metrics)[-200:]:
                operation_counts[metric.operation_name] += 1
            
            # Generate plans for top operations
            top_operations = sorted(operation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for operation_name, count in top_operations:
                if operation_name not in self.optimization_plans:
                    plan = self.generate_optimization_plan_ml(operation_name)
                    if plan.confidence_score > self.ml_config["optimization_confidence_threshold"]:
                        logger.info(f"Generated proactive optimization plan for {operation_name}")
            
        except Exception as e:
            logger.error(f"Proactive optimization plan generation error: {e}")
    
    async def _retrain_ml_models(self):
        """Retrain ML models with accumulated data"""
        try:
            if len(self.ml_training_data) < self.ml_config["min_training_samples"]:
                return
            
            training_data = list(self.ml_training_data)[-1000:]  # Last 1000 samples
            
            # Train duration predictor
            X_duration, y_duration = [], []
            for record in training_data:
                if 'features' in record and 'duration' in record:
                    X_duration.append(record['features'])
                    y_duration.append(record['duration'])
            
            if len(X_duration) >= 20:
                X_duration_array = np.array(X_duration)
                y_duration_array = np.array(y_duration)
                
                # Train scaler
                scaler_duration = StandardScaler()
                X_duration_scaled = scaler_duration.fit_transform(X_duration_array)
                
                # Train model
                self.duration_predictor = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
                self.duration_predictor.fit(X_duration_scaled, y_duration_array)
                
                self.scalers['duration_prediction'] = scaler_duration
                
                # Calculate accuracy
                predictions = self.duration_predictor.predict(X_duration_scaled)
                r2 = r2_score(y_duration_array, predictions)
                self.ml_stats['ml_model_accuracy'] = max(0, r2)
                
                logger.info(f"Retrained ML models - Duration predictor R²: {r2:.3f}")
            
        except Exception as e:
            logger.error(f"ML model retraining error: {e}")
    
    def _record_ml_training_data(self, metrics: MLPerformanceMetrics):
        """Record data for ML training"""
        try:
            training_record = {
                'timestamp': datetime.now(),
                'operation_name': metrics.operation_name,
                'features': metrics.ml_features,
                'duration': metrics.duration_ms,
                'memory': metrics.memory_usage_mb,
                'cpu': metrics.cpu_utilization,
                'performance_score': metrics.performance_score,
                'optimization_potential': metrics.optimization_potential
            }
            
            self.ml_training_data.append(training_record)
            
        except Exception as e:
            logger.debug(f"ML training data recording error: {e}")
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_ml_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML performance summary"""
        with self.engine_lock:
            recent_metrics = list(self.performance_metrics)[-100:] if self.performance_metrics else []
            
            if recent_metrics:
                avg_duration = sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
                avg_performance_score = sum(m.performance_score for m in recent_metrics) / len(recent_metrics)
                avg_optimization_potential = sum(m.optimization_potential for m in recent_metrics) / len(recent_metrics)
            else:
                avg_duration = avg_performance_score = avg_optimization_potential = 0
            
            return {
                'ml_engine_status': 'active' if self.ml_engine_active else 'inactive',
                'ml_enabled': self.ml_enabled,
                'statistics': self.ml_stats.copy(),
                'current_performance': {
                    'avg_duration_ms': avg_duration,
                    'avg_performance_score': avg_performance_score,
                    'avg_optimization_potential': avg_optimization_potential
                },
                'ml_models': {
                    'duration_predictor': self.duration_predictor is not None,
                    'bottleneck_detector': self.bottleneck_detector is not None,
                    'pattern_clusterer': self.pattern_clusterer is not None,
                    'anomaly_detector': self.anomaly_detector is not None,
                    'resource_optimizer': self.resource_optimizer is not None,
                    'scaling_predictor': self.scaling_predictor is not None
                },
                'optimization_plans': len(self.optimization_plans),
                'performance_patterns': len(self.performance_patterns),
                'ml_configuration': self.ml_config.copy(),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_optimization_recommendations(self, operation_name: str) -> Dict[str, Any]:
        """Get ML optimization recommendations for an operation"""
        try:
            if operation_name in self.optimization_plans:
                plan = self.optimization_plans[operation_name]
                return {
                    'has_plan': True,
                    'plan': plan.__dict__,
                    'confidence': plan.confidence_score,
                    'expected_improvement': plan.expected_improvement_percent
                }
            else:
                # Generate on-demand plan
                plan = self.generate_optimization_plan_ml(operation_name)
                return {
                    'has_plan': True,
                    'plan': plan.__dict__,
                    'confidence': plan.confidence_score,
                    'expected_improvement': plan.expected_improvement_percent,
                    'generated_on_demand': True
                }
                
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return {'has_plan': False, 'error': str(e)}
    
    def shutdown(self):
        """Shutdown ML performance engine"""
        self.stop_ml_performance_engine()
        logger.info("Advanced Performance ML Engine shutdown")

# Global ML performance engine instance
advanced_performance_ml_engine = AdvancedPerformanceMLEngine()

# Export for external use
__all__ = [
    'PerformanceOptimizationStrategy',
    'PerformanceMLAlgorithm',
    'PerformanceBottleneckType',
    'MLPerformanceMetrics',
    'MLOptimizationPlan',
    'PerformancePattern',
    'AdvancedPerformanceMLEngine',
    'advanced_performance_ml_engine'
]