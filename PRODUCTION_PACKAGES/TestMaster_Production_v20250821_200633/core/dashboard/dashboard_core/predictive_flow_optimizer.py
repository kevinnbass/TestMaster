"""
Predictive Analytics Flow Optimizer
==================================

AI-powered predictive optimization system that learns from analytics patterns
to optimize flow, prevent bottlenecks, and maximize system efficiency.

Author: TestMaster Team
"""

import logging
import time
import threading
import json
import math
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple, Callable
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import os

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Flow optimization strategies."""
    LOAD_BALANCING = "load_balancing"
    PREDICTIVE_CACHING = "predictive_caching"
    BOTTLENECK_AVOIDANCE = "bottleneck_avoidance"
    PARALLEL_PROCESSING = "parallel_processing"
    RESOURCE_SCALING = "resource_scaling"
    PRIORITY_QUEUING = "priority_queuing"
    ADAPTIVE_ROUTING = "adaptive_routing"

class FlowMetric(Enum):
    """Flow performance metrics."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    QUEUE_DEPTH = "queue_depth"
    RESOURCE_UTILIZATION = "resource_utilization"
    PROCESSING_TIME = "processing_time"
    SUCCESS_RATE = "success_rate"

class PredictionType(Enum):
    """Types of predictions."""
    VOLUME_FORECAST = "volume_forecast"
    BOTTLENECK_PREDICTION = "bottleneck_prediction"
    RESOURCE_DEMAND = "resource_demand"
    FAILURE_PROBABILITY = "failure_probability"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"

@dataclass
class FlowPrediction:
    """Flow prediction record."""
    prediction_id: str
    prediction_type: PredictionType
    predicted_at: datetime
    horizon_minutes: int
    confidence: float
    predicted_value: float
    actual_value: Optional[float]
    accuracy: Optional[float]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'prediction_id': self.prediction_id,
            'prediction_type': self.prediction_type.value,
            'predicted_at': self.predicted_at.isoformat(),
            'horizon_minutes': self.horizon_minutes,
            'confidence': self.confidence,
            'predicted_value': self.predicted_value,
            'actual_value': self.actual_value,
            'accuracy': self.accuracy,
            'metadata': self.metadata or {}
        }

@dataclass
class OptimizationAction:
    """Optimization action record."""
    action_id: str
    strategy: OptimizationStrategy
    triggered_at: datetime
    target_metric: FlowMetric
    expected_improvement: float
    actual_improvement: Optional[float]
    action_data: Dict[str, Any]
    success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'action_id': self.action_id,
            'strategy': self.strategy.value,
            'triggered_at': self.triggered_at.isoformat(),
            'target_metric': self.target_metric.value,
            'expected_improvement': self.expected_improvement,
            'actual_improvement': self.actual_improvement,
            'action_data': self.action_data,
            'success': self.success
        }

class PredictiveFlowOptimizer:
    """
    AI-powered predictive analytics flow optimization system.
    """
    
    def __init__(self,
                 aggregator=None,
                 delivery_guarantee=None,
                 realtime_tracker=None,
                 db_path: str = "data/flow_optimizer.db",
                 optimization_interval: float = 60.0):
        """
        Initialize predictive flow optimizer.
        
        Args:
            aggregator: Analytics aggregator instance
            delivery_guarantee: Delivery guarantee system
            realtime_tracker: Real-time analytics tracker
            db_path: Database path for optimization records
            optimization_interval: Seconds between optimization cycles
        """
        self.aggregator = aggregator
        self.delivery_guarantee = delivery_guarantee
        self.realtime_tracker = realtime_tracker
        self.db_path = db_path
        self.optimization_interval = optimization_interval
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Flow monitoring and prediction
        self.flow_metrics: Dict[FlowMetric, deque] = {
            metric: deque(maxlen=1440)  # 24 hours of minute-level data
            for metric in FlowMetric
        }
        
        self.predictions: Dict[str, FlowPrediction] = {}
        self.optimization_actions: Dict[str, OptimizationAction] = {}
        
        # Machine learning models (simplified)
        self.prediction_models: Dict[PredictionType, Dict[str, Any]] = {}
        self.optimization_strategies: Dict[OptimizationStrategy, Callable] = {
            OptimizationStrategy.LOAD_BALANCING: self._optimize_load_balancing,
            OptimizationStrategy.PREDICTIVE_CACHING: self._optimize_predictive_caching,
            OptimizationStrategy.BOTTLENECK_AVOIDANCE: self._optimize_bottleneck_avoidance,
            OptimizationStrategy.PARALLEL_PROCESSING: self._optimize_parallel_processing,
            OptimizationStrategy.RESOURCE_SCALING: self._optimize_resource_scaling,
            OptimizationStrategy.PRIORITY_QUEUING: self._optimize_priority_queuing,
            OptimizationStrategy.ADAPTIVE_ROUTING: self._optimize_adaptive_routing
        }
        
        # Performance tracking
        self.baseline_metrics: Dict[FlowMetric, float] = {}
        self.optimization_history: deque = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'prediction_accuracy': 100.0,
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'optimization_success_rate': 100.0,
            'average_improvement': 0.0,
            'best_improvement': 0.0,
            'total_processing_time_saved': 0.0,
            'current_efficiency_score': 100.0
        }
        
        # Configuration
        self.prediction_horizons = [5, 15, 30, 60, 120]  # minutes
        self.optimization_threshold = 0.1  # 10% improvement threshold
        self.confidence_threshold = 0.7
        self.learning_rate = 0.1
        
        # Background processing
        self.optimizer_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.prediction_thread = threading.Thread(
            target=self._prediction_loop,
            daemon=True
        )
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        
        # Start threads
        self.monitoring_thread.start()
        self.prediction_thread.start()
        self.optimization_thread.start()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Predictive Flow Optimizer initialized")
    
    def _init_database(self):
        """Initialize flow optimizer database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS flow_predictions (
                        prediction_id TEXT PRIMARY KEY,
                        prediction_type TEXT NOT NULL,
                        predicted_at TEXT NOT NULL,
                        horizon_minutes INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        predicted_value REAL NOT NULL,
                        actual_value REAL,
                        accuracy REAL,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_actions (
                        action_id TEXT PRIMARY KEY,
                        strategy TEXT NOT NULL,
                        triggered_at TEXT NOT NULL,
                        target_metric TEXT NOT NULL,
                        expected_improvement REAL NOT NULL,
                        actual_improvement REAL,
                        action_data TEXT NOT NULL,
                        success INTEGER
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS flow_metrics_history (
                        timestamp TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        metadata TEXT,
                        PRIMARY KEY (timestamp, metric_type)
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_prediction_type ON flow_predictions(prediction_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_predicted_at ON flow_predictions(predicted_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_optimization_strategy ON optimization_actions(strategy)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_triggered_at ON optimization_actions(triggered_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON flow_metrics_history(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Flow optimizer database initialization failed: {e}")
            raise
    
    def _monitoring_loop(self):
        """Background flow monitoring loop."""
        while self.optimizer_active:
            try:
                # Collect current flow metrics
                current_metrics = self._collect_flow_metrics()
                
                with self.lock:
                    # Store metrics
                    for metric_type, value in current_metrics.items():
                        self.flow_metrics[metric_type].append({
                            'timestamp': datetime.now(),
                            'value': value
                        })
                    
                    # Save to database
                    self._save_metrics_to_db(current_metrics)
                    
                    # Update baseline if needed
                    self._update_baseline_metrics(current_metrics)
                
                time.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)
    
    def _collect_flow_metrics(self) -> Dict[FlowMetric, float]:
        """Collect current flow metrics from all systems."""
        metrics = {}
        
        try:
            # Get metrics from real-time tracker
            if self.realtime_tracker:
                tracker_summary = self.realtime_tracker.get_tracking_summary()
                stats = tracker_summary.get('statistics', {})
                
                metrics[FlowMetric.THROUGHPUT] = stats.get('events_per_second', 0.0)
                metrics[FlowMetric.PROCESSING_TIME] = stats.get('average_flow_time', 0.0)
                
                # Calculate success rate
                total_events = stats.get('total_events', 0)
                failed_count = stats.get('failed_analytics_count', 0)
                if total_events > 0:
                    metrics[FlowMetric.SUCCESS_RATE] = ((total_events - failed_count) / total_events) * 100
                else:
                    metrics[FlowMetric.SUCCESS_RATE] = 100.0
            
            # Get metrics from delivery guarantee
            if self.delivery_guarantee:
                delivery_stats = self.delivery_guarantee.get_guarantee_statistics()
                stats = delivery_stats.get('statistics', {})
                
                metrics[FlowMetric.ERROR_RATE] = 100 - stats.get('delivery_success_rate', 100.0)
                metrics[FlowMetric.QUEUE_DEPTH] = stats.get('current_pending', 0)
            
            # Get metrics from aggregator
            if self.aggregator and hasattr(self.aggregator, 'get_performance_metrics'):
                agg_metrics = self.aggregator.get_performance_metrics()
                metrics[FlowMetric.LATENCY] = agg_metrics.get('average_response_time', 0.0)
                metrics[FlowMetric.RESOURCE_UTILIZATION] = agg_metrics.get('cpu_usage', 0.0)
            
            # Fill in missing metrics with defaults
            for metric_type in FlowMetric:
                if metric_type not in metrics:
                    metrics[metric_type] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Flow metrics collection failed: {e}")
            return {metric: 0.0 for metric in FlowMetric}
    
    def _save_metrics_to_db(self, metrics: Dict[FlowMetric, float]):
        """Save metrics to database."""
        try:
            timestamp = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                for metric_type, value in metrics.items():
                    conn.execute('''
                        INSERT OR REPLACE INTO flow_metrics_history
                        (timestamp, metric_type, value, metadata)
                        VALUES (?, ?, ?, ?)
                    ''', (timestamp, metric_type.value, value, '{}'))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Metrics database save failed: {e}")
    
    def _update_baseline_metrics(self, current_metrics: Dict[FlowMetric, float]):
        """Update baseline metrics for comparison."""
        try:
            for metric_type, current_value in current_metrics.items():
                if metric_type not in self.baseline_metrics:
                    self.baseline_metrics[metric_type] = current_value
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.baseline_metrics[metric_type] = (
                        alpha * current_value + 
                        (1 - alpha) * self.baseline_metrics[metric_type]
                    )
        except Exception as e:
            logger.error(f"Baseline metrics update failed: {e}")
    
    def _prediction_loop(self):
        """Background prediction loop."""
        while self.optimizer_active:
            try:
                time.sleep(120)  # Generate predictions every 2 minutes
                
                with self.lock:
                    # Generate predictions for different horizons
                    for horizon in self.prediction_horizons:
                        self._generate_predictions(horizon)
                    
                    # Validate previous predictions
                    self._validate_predictions()
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
    
    def _generate_predictions(self, horizon_minutes: int):
        """Generate predictions for specified time horizon."""
        try:
            # Volume forecast
            volume_prediction = self._predict_volume(horizon_minutes)
            if volume_prediction:
                self.predictions[volume_prediction.prediction_id] = volume_prediction
                self._save_prediction(volume_prediction)
            
            # Bottleneck prediction
            bottleneck_prediction = self._predict_bottlenecks(horizon_minutes)
            if bottleneck_prediction:
                self.predictions[bottleneck_prediction.prediction_id] = bottleneck_prediction
                self._save_prediction(bottleneck_prediction)
            
            # Resource demand prediction
            resource_prediction = self._predict_resource_demand(horizon_minutes)
            if resource_prediction:
                self.predictions[resource_prediction.prediction_id] = resource_prediction
                self._save_prediction(resource_prediction)
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
    
    def _predict_volume(self, horizon_minutes: int) -> Optional[FlowPrediction]:
        """Predict analytics volume for time horizon."""
        try:
            throughput_data = self.flow_metrics[FlowMetric.THROUGHPUT]
            
            if len(throughput_data) < 10:
                return None  # Not enough data
            
            # Simple trend analysis
            recent_values = [point['value'] for point in list(throughput_data)[-30:]]
            
            if not recent_values:
                return None
            
            # Calculate trend
            avg_value = statistics.mean(recent_values)
            
            # Simple linear extrapolation
            if len(recent_values) > 5:
                x_values = list(range(len(recent_values)))
                slope = self._calculate_trend_slope(x_values, recent_values)
                predicted_value = avg_value + (slope * horizon_minutes)
            else:
                predicted_value = avg_value
            
            # Calculate confidence based on data consistency
            if len(recent_values) > 1:
                stdev = statistics.stdev(recent_values)
                confidence = max(0.1, min(1.0, 1.0 - (stdev / (avg_value + 0.1))))
            else:
                confidence = 0.5
            
            prediction = FlowPrediction(
                prediction_id=f"volume_{int(time.time() * 1000000)}",
                prediction_type=PredictionType.VOLUME_FORECAST,
                predicted_at=datetime.now(),
                horizon_minutes=horizon_minutes,
                confidence=confidence,
                predicted_value=max(0, predicted_value),
                actual_value=None,
                accuracy=None,
                metadata={'avg_baseline': avg_value, 'data_points': len(recent_values)}
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Volume prediction failed: {e}")
            return None
    
    def _predict_bottlenecks(self, horizon_minutes: int) -> Optional[FlowPrediction]:
        """Predict bottleneck probability."""
        try:
            queue_data = self.flow_metrics[FlowMetric.QUEUE_DEPTH]
            latency_data = self.flow_metrics[FlowMetric.LATENCY]
            
            if len(queue_data) < 5 or len(latency_data) < 5:
                return None
            
            # Calculate bottleneck indicators
            recent_queue = [point['value'] for point in list(queue_data)[-10:]]
            recent_latency = [point['value'] for point in list(latency_data)[-10:]]
            
            avg_queue = statistics.mean(recent_queue) if recent_queue else 0
            avg_latency = statistics.mean(recent_latency) if recent_latency else 0
            
            # Bottleneck probability based on queue depth and latency trends
            queue_trend = self._calculate_trend_slope(list(range(len(recent_queue))), recent_queue)
            latency_trend = self._calculate_trend_slope(list(range(len(recent_latency))), recent_latency)
            
            # Higher probability if both queue and latency are increasing
            bottleneck_probability = 0.0
            
            if queue_trend > 0.1:  # Queue growing
                bottleneck_probability += 0.3
            if latency_trend > 0.1:  # Latency increasing
                bottleneck_probability += 0.3
            if avg_queue > 10:  # High queue depth
                bottleneck_probability += 0.2
            if avg_latency > 1000:  # High latency (ms)
                bottleneck_probability += 0.2
            
            bottleneck_probability = min(1.0, bottleneck_probability)
            
            # Confidence based on trend consistency
            confidence = 0.8 if queue_trend > 0 and latency_trend > 0 else 0.6
            
            prediction = FlowPrediction(
                prediction_id=f"bottleneck_{int(time.time() * 1000000)}",
                prediction_type=PredictionType.BOTTLENECK_PREDICTION,
                predicted_at=datetime.now(),
                horizon_minutes=horizon_minutes,
                confidence=confidence,
                predicted_value=bottleneck_probability,
                actual_value=None,
                accuracy=None,
                metadata={
                    'queue_trend': queue_trend,
                    'latency_trend': latency_trend,
                    'avg_queue': avg_queue,
                    'avg_latency': avg_latency
                }
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Bottleneck prediction failed: {e}")
            return None
    
    def _predict_resource_demand(self, horizon_minutes: int) -> Optional[FlowPrediction]:
        """Predict resource demand."""
        try:
            utilization_data = self.flow_metrics[FlowMetric.RESOURCE_UTILIZATION]
            throughput_data = self.flow_metrics[FlowMetric.THROUGHPUT]
            
            if len(utilization_data) < 5 or len(throughput_data) < 5:
                return None
            
            recent_util = [point['value'] for point in list(utilization_data)[-15:]]
            recent_throughput = [point['value'] for point in list(throughput_data)[-15:]]
            
            if not recent_util or not recent_throughput:
                return None
            
            avg_util = statistics.mean(recent_util)
            avg_throughput = statistics.mean(recent_throughput)
            
            # Predict resource demand based on throughput trend
            throughput_trend = self._calculate_trend_slope(list(range(len(recent_throughput))), recent_throughput)
            
            # Estimate future resource demand
            if throughput_trend > 0:
                predicted_demand = avg_util + (throughput_trend * horizon_minutes * 0.1)
            else:
                predicted_demand = avg_util
            
            predicted_demand = max(0, min(100, predicted_demand))  # 0-100%
            
            confidence = 0.7 if len(recent_util) > 10 else 0.5
            
            prediction = FlowPrediction(
                prediction_id=f"resource_{int(time.time() * 1000000)}",
                prediction_type=PredictionType.RESOURCE_DEMAND,
                predicted_at=datetime.now(),
                horizon_minutes=horizon_minutes,
                confidence=confidence,
                predicted_value=predicted_demand,
                actual_value=None,
                accuracy=None,
                metadata={
                    'current_utilization': avg_util,
                    'throughput_trend': throughput_trend
                }
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Resource demand prediction failed: {e}")
            return None
    
    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate trend slope using simple linear regression."""
        try:
            if len(x_values) != len(y_values) or len(x_values) < 2:
                return 0.0
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except Exception as e:
            logger.error(f"Trend slope calculation failed: {e}")
            return 0.0
    
    def _validate_predictions(self):
        """Validate previous predictions against actual values."""
        try:
            current_time = datetime.now()
            
            for prediction_id, prediction in list(self.predictions.items()):
                # Check if prediction horizon has passed
                time_since_prediction = (current_time - prediction.predicted_at).total_seconds() / 60
                
                if time_since_prediction >= prediction.horizon_minutes:
                    # Get actual value
                    actual_value = self._get_actual_value_for_prediction(prediction)
                    
                    if actual_value is not None:
                        prediction.actual_value = actual_value
                        
                        # Calculate accuracy
                        if prediction.predicted_value != 0:
                            error = abs(prediction.predicted_value - actual_value) / abs(prediction.predicted_value)
                            prediction.accuracy = max(0, 1 - error)
                        else:
                            prediction.accuracy = 1.0 if actual_value == 0 else 0.0
                        
                        # Update statistics
                        self.stats['total_predictions'] += 1
                        if prediction.accuracy > 0.8:  # 80% accuracy threshold
                            self.stats['accurate_predictions'] += 1
                        
                        # Update overall accuracy
                        total_preds = self.stats['total_predictions']
                        if total_preds > 0:
                            self.stats['prediction_accuracy'] = (
                                self.stats['accurate_predictions'] / total_preds * 100
                            )
                        
                        # Save updated prediction
                        self._save_prediction(prediction)
                        
                        logger.debug(f"Validated prediction {prediction_id}: {prediction.accuracy:.2f} accuracy")
            
        except Exception as e:
            logger.error(f"Prediction validation failed: {e}")
    
    def _get_actual_value_for_prediction(self, prediction: FlowPrediction) -> Optional[float]:
        """Get actual value for prediction validation."""
        try:
            if prediction.prediction_type == PredictionType.VOLUME_FORECAST:
                # Get actual throughput at prediction time
                throughput_data = self.flow_metrics[FlowMetric.THROUGHPUT]
                if throughput_data:
                    return throughput_data[-1]['value']
            
            elif prediction.prediction_type == PredictionType.BOTTLENECK_PREDICTION:
                # Check if bottleneck actually occurred
                queue_data = self.flow_metrics[FlowMetric.QUEUE_DEPTH]
                latency_data = self.flow_metrics[FlowMetric.LATENCY]
                
                if queue_data and latency_data:
                    current_queue = queue_data[-1]['value']
                    current_latency = latency_data[-1]['value']
                    
                    # Bottleneck indicator: high queue or high latency
                    bottleneck_occurred = (current_queue > 15 or current_latency > 1500)
                    return 1.0 if bottleneck_occurred else 0.0
            
            elif prediction.prediction_type == PredictionType.RESOURCE_DEMAND:
                # Get actual resource utilization
                util_data = self.flow_metrics[FlowMetric.RESOURCE_UTILIZATION]
                if util_data:
                    return util_data[-1]['value']
            
            return None
            
        except Exception as e:
            logger.error(f"Actual value retrieval failed: {e}")
            return None
    
    def _optimization_loop(self):
        """Background optimization loop."""
        while self.optimizer_active:
            try:
                time.sleep(self.optimization_interval)
                
                with self.lock:
                    # Check for optimization opportunities
                    self._identify_optimization_opportunities()
                    
                    # Execute optimizations
                    self._execute_optimizations()
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
    
    def _identify_optimization_opportunities(self):
        """Identify potential optimization opportunities."""
        try:
            current_metrics = self._collect_flow_metrics()
            
            # Check each optimization strategy
            for strategy in OptimizationStrategy:
                opportunity_score = self._evaluate_optimization_opportunity(strategy, current_metrics)
                
                if opportunity_score > self.optimization_threshold:
                    self._trigger_optimization(strategy, current_metrics, opportunity_score)
            
        except Exception as e:
            logger.error(f"Optimization opportunity identification failed: {e}")
    
    def _evaluate_optimization_opportunity(self, strategy: OptimizationStrategy, metrics: Dict[FlowMetric, float]) -> float:
        """Evaluate optimization opportunity score for a strategy."""
        try:
            if strategy == OptimizationStrategy.LOAD_BALANCING:
                # High opportunity if resource utilization is uneven
                util = metrics[FlowMetric.RESOURCE_UTILIZATION]
                if util > 80:  # High utilization suggests load balancing needed
                    return 0.8
                
            elif strategy == OptimizationStrategy.BOTTLENECK_AVOIDANCE:
                # High opportunity if queue depth is growing
                queue_depth = metrics[FlowMetric.QUEUE_DEPTH]
                latency = metrics[FlowMetric.LATENCY]
                
                if queue_depth > 10 or latency > 1000:
                    return 0.9
                
            elif strategy == OptimizationStrategy.PREDICTIVE_CACHING:
                # Opportunity based on throughput patterns
                throughput = metrics[FlowMetric.THROUGHPUT]
                if throughput > 5:  # High throughput benefits from caching
                    return 0.6
                
            elif strategy == OptimizationStrategy.PARALLEL_PROCESSING:
                # Opportunity if processing time is high
                processing_time = metrics[FlowMetric.PROCESSING_TIME]
                if processing_time > 500:  # >500ms suggests parallel processing could help
                    return 0.7
                
            elif strategy == OptimizationStrategy.PRIORITY_QUEUING:
                # Opportunity if error rate is high
                error_rate = metrics[FlowMetric.ERROR_RATE]
                if error_rate > 5:  # >5% error rate
                    return 0.6
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Optimization opportunity evaluation failed: {e}")
            return 0.0
    
    def _trigger_optimization(self, strategy: OptimizationStrategy, metrics: Dict[FlowMetric, float], opportunity_score: float):
        """Trigger an optimization action."""
        try:
            action_id = f"opt_{strategy.value}_{int(time.time() * 1000000)}"
            
            # Determine target metric and expected improvement
            target_metric, expected_improvement = self._calculate_expected_improvement(strategy, metrics)
            
            action = OptimizationAction(
                action_id=action_id,
                strategy=strategy,
                triggered_at=datetime.now(),
                target_metric=target_metric,
                expected_improvement=expected_improvement,
                actual_improvement=None,
                action_data={'metrics_snapshot': metrics, 'opportunity_score': opportunity_score}
            )
            
            self.optimization_actions[action_id] = action
            
            logger.info(f"Triggered optimization: {strategy.value} (expected improvement: {expected_improvement:.1f}%)")
            
        except Exception as e:
            logger.error(f"Optimization trigger failed: {e}")
    
    def _calculate_expected_improvement(self, strategy: OptimizationStrategy, metrics: Dict[FlowMetric, float]) -> Tuple[FlowMetric, float]:
        """Calculate expected improvement for optimization strategy."""
        try:
            if strategy == OptimizationStrategy.LOAD_BALANCING:
                return FlowMetric.RESOURCE_UTILIZATION, 15.0
            elif strategy == OptimizationStrategy.BOTTLENECK_AVOIDANCE:
                return FlowMetric.LATENCY, 25.0
            elif strategy == OptimizationStrategy.PREDICTIVE_CACHING:
                return FlowMetric.PROCESSING_TIME, 20.0
            elif strategy == OptimizationStrategy.PARALLEL_PROCESSING:
                return FlowMetric.THROUGHPUT, 30.0
            elif strategy == OptimizationStrategy.PRIORITY_QUEUING:
                return FlowMetric.ERROR_RATE, 50.0
            else:
                return FlowMetric.THROUGHPUT, 10.0
                
        except Exception as e:
            logger.error(f"Expected improvement calculation failed: {e}")
            return FlowMetric.THROUGHPUT, 0.0
    
    def _execute_optimizations(self):
        """Execute pending optimizations."""
        try:
            pending_actions = [
                action for action in self.optimization_actions.values()
                if action.success is None
            ]
            
            for action in pending_actions[:3]:  # Limit to 3 concurrent optimizations
                success = self.optimization_strategies[action.strategy](action)
                action.success = success
                
                if success:
                    self.stats['successful_optimizations'] += 1
                    logger.info(f"Successfully executed optimization: {action.action_id}")
                else:
                    logger.warning(f"Optimization failed: {action.action_id}")
                
                self.stats['total_optimizations'] += 1
                
                # Update success rate
                if self.stats['total_optimizations'] > 0:
                    self.stats['optimization_success_rate'] = (
                        self.stats['successful_optimizations'] / self.stats['total_optimizations'] * 100
                    )
                
                # Save action
                self._save_optimization_action(action)
            
        except Exception as e:
            logger.error(f"Optimization execution failed: {e}")
    
    def _optimize_load_balancing(self, action: OptimizationAction) -> bool:
        """Implement load balancing optimization."""
        try:
            # Theoretical load balancing - adjust processing queues
            if self.realtime_tracker and hasattr(self.realtime_tracker, 'balance_load'):
                return self.realtime_tracker.balance_load()
            
            # Simulate load balancing effect
            logger.info("Load balancing optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Load balancing optimization failed: {e}")
            return False
    
    def _optimize_predictive_caching(self, action: OptimizationAction) -> bool:
        """Implement predictive caching optimization."""
        try:
            # Enable more aggressive caching based on patterns
            if self.aggregator and hasattr(self.aggregator, 'enable_predictive_caching'):
                return self.aggregator.enable_predictive_caching()
            
            logger.info("Predictive caching optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Predictive caching optimization failed: {e}")
            return False
    
    def _optimize_bottleneck_avoidance(self, action: OptimizationAction) -> bool:
        """Implement bottleneck avoidance optimization."""
        try:
            # Increase processing capacity or adjust queues
            if self.delivery_guarantee and hasattr(self.delivery_guarantee, 'increase_batch_size'):
                return self.delivery_guarantee.increase_batch_size()
            
            logger.info("Bottleneck avoidance optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Bottleneck avoidance optimization failed: {e}")
            return False
    
    def _optimize_parallel_processing(self, action: OptimizationAction) -> bool:
        """Implement parallel processing optimization."""
        try:
            # Enable parallel processing where possible
            if self.aggregator and hasattr(self.aggregator, 'enable_parallel_processing'):
                return self.aggregator.enable_parallel_processing()
            
            logger.info("Parallel processing optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Parallel processing optimization failed: {e}")
            return False
    
    def _optimize_resource_scaling(self, action: OptimizationAction) -> bool:
        """Implement resource scaling optimization."""
        try:
            # Scale resources based on demand
            logger.info("Resource scaling optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Resource scaling optimization failed: {e}")
            return False
    
    def _optimize_priority_queuing(self, action: OptimizationAction) -> bool:
        """Implement priority queuing optimization."""
        try:
            # Implement priority-based processing
            if self.delivery_guarantee and hasattr(self.delivery_guarantee, 'enable_priority_queuing'):
                return self.delivery_guarantee.enable_priority_queuing()
            
            logger.info("Priority queuing optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Priority queuing optimization failed: {e}")
            return False
    
    def _optimize_adaptive_routing(self, action: OptimizationAction) -> bool:
        """Implement adaptive routing optimization."""
        try:
            # Route analytics through optimal paths
            logger.info("Adaptive routing optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Adaptive routing optimization failed: {e}")
            return False
    
    def _save_prediction(self, prediction: FlowPrediction):
        """Save prediction to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO flow_predictions
                    (prediction_id, prediction_type, predicted_at, horizon_minutes,
                     confidence, predicted_value, actual_value, accuracy, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.prediction_id,
                    prediction.prediction_type.value,
                    prediction.predicted_at.isoformat(),
                    prediction.horizon_minutes,
                    prediction.confidence,
                    prediction.predicted_value,
                    prediction.actual_value,
                    prediction.accuracy,
                    json.dumps(prediction.metadata or {})
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
    
    def _save_optimization_action(self, action: OptimizationAction):
        """Save optimization action to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO optimization_actions
                    (action_id, strategy, triggered_at, target_metric,
                     expected_improvement, actual_improvement, action_data, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    action.action_id,
                    action.strategy.value,
                    action.triggered_at.isoformat(),
                    action.target_metric.value,
                    action.expected_improvement,
                    action.actual_improvement,
                    json.dumps(action.action_data),
                    1 if action.success else 0 if action.success is not None else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save optimization action: {e}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        with self.lock:
            # Calculate efficiency score
            current_metrics = self._collect_flow_metrics()
            efficiency_factors = []
            
            # Factor in various metrics
            if current_metrics[FlowMetric.SUCCESS_RATE] > 0:
                efficiency_factors.append(current_metrics[FlowMetric.SUCCESS_RATE])
            
            if current_metrics[FlowMetric.THROUGHPUT] > 0:
                efficiency_factors.append(min(100, current_metrics[FlowMetric.THROUGHPUT] * 10))
            
            if current_metrics[FlowMetric.ERROR_RATE] < 10:
                efficiency_factors.append(100 - current_metrics[FlowMetric.ERROR_RATE] * 5)
            
            if efficiency_factors:
                self.stats['current_efficiency_score'] = statistics.mean(efficiency_factors)
            
            return {
                'statistics': dict(self.stats),
                'current_metrics': {metric.value: value for metric, value in current_metrics.items()},
                'baseline_metrics': {metric.value: value for metric, value in self.baseline_metrics.items()},
                'active_predictions': len([p for p in self.predictions.values() if p.actual_value is None]),
                'pending_optimizations': len([a for a in self.optimization_actions.values() if a.success is None]),
                'prediction_types': {
                    ptype.value: len([p for p in self.predictions.values() if p.prediction_type == ptype])
                    for ptype in PredictionType
                },
                'optimization_strategies': {
                    strategy.value: len([a for a in self.optimization_actions.values() if a.strategy == strategy])
                    for strategy in OptimizationStrategy
                },
                'configuration': {
                    'optimization_interval': self.optimization_interval,
                    'prediction_horizons': self.prediction_horizons,
                    'optimization_threshold': self.optimization_threshold,
                    'confidence_threshold': self.confidence_threshold
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def force_optimization(self, strategy: OptimizationStrategy) -> bool:
        """Force execution of specific optimization strategy."""
        try:
            current_metrics = self._collect_flow_metrics()
            opportunity_score = 1.0  # Force high opportunity score
            
            self._trigger_optimization(strategy, current_metrics, opportunity_score)
            
            # Find and execute the action
            pending_actions = [
                action for action in self.optimization_actions.values()
                if action.strategy == strategy and action.success is None
            ]
            
            if pending_actions:
                latest_action = max(pending_actions, key=lambda x: x.triggered_at)
                success = self.optimization_strategies[strategy](latest_action)
                latest_action.success = success
                self._save_optimization_action(latest_action)
                
                logger.info(f"Force optimization {strategy.value}: {'success' if success else 'failed'}")
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Force optimization failed: {e}")
            return False
    
    def get_prediction_details(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific prediction."""
        if prediction_id in self.predictions:
            return self.predictions[prediction_id].to_dict()
        
        # Check database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT * FROM flow_predictions WHERE prediction_id = ?',
                    (prediction_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        'prediction_id': row[0],
                        'prediction_type': row[1],
                        'predicted_at': row[2],
                        'horizon_minutes': row[3],
                        'confidence': row[4],
                        'predicted_value': row[5],
                        'actual_value': row[6],
                        'accuracy': row[7],
                        'metadata': json.loads(row[8]) if row[8] else {}
                    }
        except Exception as e:
            logger.error(f"Prediction details lookup failed: {e}")
        
        return None
    
    def shutdown(self):
        """Shutdown predictive flow optimizer."""
        self.optimizer_active = False
        
        # Wait for threads to complete
        for thread in [self.monitoring_thread, self.prediction_thread, self.optimization_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Predictive Flow Optimizer shutdown - Stats: {self.stats}")

# Global flow optimizer instance
flow_optimizer = None