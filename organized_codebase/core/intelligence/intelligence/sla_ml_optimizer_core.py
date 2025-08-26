"""
Advanced ML SLA Optimization Engine
==================================
"""Core Module - Split from sla_ml_optimizer.py"""


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


logger = logging.getLogger(__name__)

class MLSLALevel(Enum):
    """ML-enhanced SLA performance levels"""
    PLATINUM = "platinum"    # 99.99% uptime, <50ms latency
    GOLD = "gold"           # 99.9% uptime, <100ms latency  
    SILVER = "silver"       # 99.5% uptime, <250ms latency
    BRONZE = "bronze"       # 99.0% uptime, <500ms latency
    ADAPTIVE = "adaptive"   # ML-determined optimal level

class MLEscalationLevel(Enum):
    """ML-enhanced escalation levels"""
    L1_MONITORING = "l1_monitoring"
    L2_ENGINEERING = "l2_engineering"
    L3_SENIOR_ENG = "l3_senior_engineering"
    L4_MANAGEMENT = "l4_management"
    L5_EXECUTIVE = "l5_executive"
    ML_AUTOMATED = "ml_automated"

class ViolationRisk(Enum):
    """ML-predicted violation risk levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OptimizationStrategy(Enum):
    """ML optimization strategies"""
    PREDICTIVE_SCALING = "predictive_scaling"
    ADAPTIVE_THRESHOLDS = "adaptive_thresholds"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    PATTERN_LEARNING = "pattern_learning"
    ANOMALY_PREVENTION = "anomaly_prevention"

@dataclass
class MLSLAConfiguration:
    """ML-enhanced SLA configuration"""
    level: MLSLALevel
    max_latency_ms: float
    min_availability_percent: float
    min_throughput_tps: float
    max_error_rate_percent: float
    delivery_timeout_seconds: float
    escalation_threshold_minutes: int
    
    # ML-specific parameters
    ml_prediction_horizon_minutes: int = 30
    adaptive_threshold_enabled: bool = True
    ml_confidence_threshold: float = 0.8
    auto_optimization_enabled: bool = True
    anomaly_detection_sensitivity: float = 0.7

@dataclass
class MLSLAMetric:
    """ML-enhanced SLA metric"""
    metric_id: str
    analytics_id: str
    timestamp: datetime
    latency_ms: float
    delivery_success: bool
    component: str
    stage: str
    
    # ML features
    ml_features: List[float] = field(default_factory=list)
    predicted_latency: Optional[float] = None
    violation_probability: float = 0.0
    performance_trend: str = "stable"
    optimization_recommendations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

@dataclass
class MLSLAViolation:
    """ML-enhanced SLA violation"""
    violation_id: str
    analytics_id: str
    violation_type: str
    timestamp: datetime
    current_value: float
    threshold_value: float
    severity: str
    
    # ML analysis
    ml_predicted: bool = False
    prediction_confidence: float = 0.0
    root_cause_analysis: List[str] = field(default_factory=list)
    optimization_strategy: Optional[OptimizationStrategy] = None
    
    # Escalation tracking
    escalated: bool = False
    escalation_level: Optional[MLEscalationLevel] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    impact_description: str = ""

@dataclass
class MLPerformancePrediction:
    """ML performance prediction"""
    analytics_id: str
    prediction_timestamp: datetime
    prediction_horizon_minutes: int
    
    # Predicted metrics
    predicted_latency: float
    predicted_availability: float
    predicted_throughput: float
    predicted_error_rate: float
    
    # ML analysis
    confidence_score: float
    risk_assessment: ViolationRisk
    optimization_opportunities: List[str] = field(default_factory=list)
    resource_recommendations: Dict[str, float] = field(default_factory=dict)

class AdvancedMLSLAOptimizer:
    """
    Advanced ML-driven SLA optimization engine with predictive performance
    management, intelligent threshold adaptation, and smart resource scaling.
    """
    
    def __init__(self, ml_enabled: bool = True, monitoring_interval: float = 30.0):
        """
        Initialize ML SLA optimizer.
        
        Args:
            ml_enabled: Enable ML optimization features
            monitoring_interval: Base monitoring interval in seconds
        """
        self.ml_enabled = ml_enabled
        self.monitoring_interval = monitoring_interval
        
        # ML models for SLA optimization
        self.latency_predictor: Optional[RandomForestRegressor] = None
        self.violation_predictor: Optional[GradientBoostingClassifier] = None
        self.performance_classifier: Optional[LogisticRegression] = None
        self.resource_optimizer: Optional[KMeans] = None
        self.scalers: Dict[str, StandardScaler] = {}
        
        # SLA tracking and optimization
        self.sla_configs: Dict[MLSLALevel, MLSLAConfiguration] = {}
        self.active_violations: Dict[str, MLSLAViolation] = {}
        self.sla_metrics: deque = deque(maxlen=10000)
        self.performance_predictions: Dict[str, MLPerformancePrediction] = {}
        
        # ML optimization tracking
        self.ml_features_history: deque = deque(maxlen=5000)
        self.optimization_history: deque = deque(maxlen=1000)
        self.threshold_adaptations: Dict[str, List[float]] = defaultdict(list)
        
        # Adaptive management
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = {}
        self.escalation_patterns: defaultdict = defaultdict(list)
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # ML configuration
        self.ml_config = {
            "prediction_horizon_minutes": 30,
            "violation_confidence_threshold": 0.8,
            "adaptive_threshold_sensitivity": 0.1,
            "optimization_trigger_threshold": 0.7,
            "min_training_samples": 50,
            "model_retrain_hours": 6,
            "feature_extraction_depth": 20,
            "pattern_analysis_window": 100,
            "resource_optimization_interval": 300
        }
        
        # Performance statistics
        self.optimizer_stats = {
            'total_sla_checks': 0,
            'ml_predictions_made': 0,
            'violations_predicted': 0,
            'violations_prevented': 0,
            'threshold_adaptations': 0,
            'resource_optimizations': 0,
            'escalations_triggered': 0,
            'ml_accuracy': 0.0,
            'optimization_effectiveness': 0.0,
            'auto_resolutions': 0,
            'prediction_accuracy': 0.0,
            'start_time': datetime.now()
        }
        
        # Background ML processing
        self.ml_optimizer_active = False
        self.ml_monitoring_worker: Optional[threading.Thread] = None
        self.ml_prediction_worker: Optional[threading.Thread] = None
        self.ml_optimization_worker: Optional[threading.Thread] = None
        
        # Thread safety
        self.optimizer_lock = threading.RLock()
        
        self._initialize_ml_models()
        self._setup_default_sla_configs()
        
        logger.info("Advanced ML SLA Optimizer initialized")
    
    def start_ml_sla_optimizer(self):
        """Start ML-enhanced SLA optimizer"""
        if self.ml_optimizer_active:
            return
        
        self.ml_optimizer_active = True
        
        # Start ML workers
        self.ml_monitoring_worker = threading.Thread(
            target=self._ml_monitoring_loop, daemon=True)
        self.ml_prediction_worker = threading.Thread(
            target=self._ml_prediction_loop, daemon=True)
        self.ml_optimization_worker = threading.Thread(
            target=self._ml_optimization_loop, daemon=True)
        
        self.ml_monitoring_worker.start()
        self.ml_prediction_worker.start()
        self.ml_optimization_worker.start()
        
        logger.info("ML SLA Optimizer started")
    
    def stop_ml_sla_optimizer(self):
        """Stop ML SLA optimizer"""
        self.ml_optimizer_active = False
        
        # Wait for workers to finish
        for worker in [self.ml_monitoring_worker, self.ml_prediction_worker, self.ml_optimization_worker]:
            if worker and worker.is_alive():
                worker.join(timeout=5)
        
        logger.info("ML SLA Optimizer stopped")
    
    def track_analytics_delivery_ml(self, analytics_id: str, component: str, stage: str,
                                   sla_level: MLSLALevel = MLSLALevel.GOLD) -> str:
        """Track analytics delivery with ML-enhanced SLA monitoring"""
        tracking_id = f"ml_sla_{int(time.time() * 1000000)}"
        
        with self.optimizer_lock:
            # Extract ML features for performance prediction
            ml_features = self._extract_sla_features(analytics_id, component, stage)
            
            # Predict performance characteristics
            predicted_latency = self._predict_delivery_latency(ml_features, sla_level)
            violation_probability = self._predict_violation_probability(ml_features, sla_level)
            
            # Create enhanced tracking record
            tracking_data = {
                'tracking_id': tracking_id,
                'analytics_id': analytics_id,
                'component': component,
                'stage': stage,
                'sla_level': sla_level,
                'start_time': datetime.now(),
                'ml_features': ml_features,
                'predicted_latency': predicted_latency,
                'violation_probability': violation_probability,
                'adaptive_timeout': self._calculate_adaptive_timeout(sla_level, predicted_latency)
            }
            
            # Generate performance prediction
            prediction = self._generate_performance_prediction(analytics_id, ml_features, sla_level)
            if prediction:
                self.performance_predictions[analytics_id] = prediction
            
            # Proactive optimization if high violation risk
            if violation_probability > self.ml_config["optimization_trigger_threshold"]:
                self._apply_proactive_optimization(tracking_data)
            
            self.optimizer_stats['total_sla_checks'] += 1
            
            logger.debug(f"ML SLA tracking started: {analytics_id} (predicted latency: {predicted_latency:.1f}ms)")
            return tracking_id
    
    def record_delivery_success_ml(self, tracking_id: str, latency_ms: float,
                                  additional_metrics: Dict[str, Any] = None):
        """Record successful delivery with ML analysis"""
        with self.optimizer_lock:
            # Extract delivery information (would be stored in actual implementation)
            delivery_info = self._get_tracking_info(tracking_id)
            
            if not delivery_info:
                return
            
            # Create ML-enhanced metric
            ml_features = self._extract_delivery_features(delivery_info, latency_ms, True)
            
            metric = MLSLAMetric(
                metric_id=f"metric_{tracking_id}",
                analytics_id=delivery_info['analytics_id'],
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                delivery_success=True,
                component=delivery_info['component'],
                stage=delivery_info['stage'],
                ml_features=ml_features,
                predicted_latency=delivery_info.get('predicted_latency'),
                performance_trend=self._analyze_performance_trend(delivery_info['analytics_id'], latency_ms)
            )
            
            # Store metric and analyze
            self.sla_metrics.append(metric)
            self._analyze_sla_performance_ml(metric, delivery_info['sla_level'])
            
            # Update ML models with new data
            self._record_ml_training_data(metric, delivery_info)
            
            # Adaptive threshold updates
            if self.ml_config.get("adaptive_threshold_enabled", True):
                self._update_adaptive_thresholds(metric, delivery_info['sla_level'])
            
            self.optimizer_stats['total_sla_checks'] += 1
            
            logger.debug(f"ML SLA success recorded: {delivery_info['analytics_id']} in {latency_ms:.1f}ms")
    
    def record_delivery_failure_ml(self, tracking_id: str, error_message: str, latency_ms: float = 0.0):
        """Record delivery failure with ML analysis"""
        with self.optimizer_lock:
            delivery_info = self._get_tracking_info(tracking_id)
            
            if not delivery_info:
                return
            
            # Create ML-enhanced metric
            ml_features = self._extract_delivery_features(delivery_info, latency_ms, False)
            
            metric = MLSLAMetric(
                metric_id=f"metric_{tracking_id}",
                analytics_id=delivery_info['analytics_id'],
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                delivery_success=False,
                component=delivery_info['component'],
                stage=delivery_info['stage'],
                ml_features=ml_features,
                error_message=error_message,
                performance_trend="degrading"
            )
            
            # Store metric and create violation
            self.sla_metrics.append(metric)
            violation = self._create_ml_violation(metric, delivery_info, error_message)
            
            # ML-enhanced root cause analysis
            root_causes = self._analyze_failure_root_causes_ml(metric, delivery_info)
            violation.root_cause_analysis = root_causes
            
            # Apply ML optimization strategy
            optimization_strategy = self._select_optimization_strategy_ml(violation, metric)
            violation.optimization_strategy = optimization_strategy
            
            # Trigger ML-driven escalation if needed
            self._evaluate_ml_escalation(violation)
            
            self.optimizer_stats['total_sla_checks'] += 1
            
            logger.error(f"ML SLA failure recorded: {delivery_info['analytics_id']} - {error_message}")
    
    def _extract_sla_features(self, analytics_id: str, component: str, stage: str) -> List[float]:
        """Extract ML features for SLA analysis"""
        try:
            features = []
            
            # System load features
            current_time = datetime.now()
            features.append(float(current_time.hour))
            features.append(float(current_time.minute))
            features.append(float(current_time.weekday()))
            
            # Component/stage characteristics
            component_hash = hash(component) % 1000
            stage_hash = hash(stage) % 1000
            features.extend([float(component_hash), float(stage_hash)])
            
            # Historical performance features
            recent_metrics = self._get_recent_metrics(analytics_id, component, stage)
            if recent_metrics:
                avg_latency = np.mean([m.latency_ms for m in recent_metrics if m.delivery_success])
                success_rate = sum(1 for m in recent_metrics if m.delivery_success) / len(recent_metrics)
                features.extend([avg_latency, success_rate])
            else:
                features.extend([100.0, 0.95])  # Default values
            
            # System resource features (simulated)
            cpu_usage = 50.0 + np.random.normal(0, 10)  # Simulated CPU
            memory_usage = 60.0 + np.random.normal(0, 15)  # Simulated memory
            features.extend([cpu_usage, memory_usage])
            
            # Queue and load features
            active_deliveries = len([m for m in self.sla_metrics if 
                                   (current_time - m.timestamp).total_seconds() < 300])
            features.append(float(active_deliveries))
            
            # Trend features
            trend_score = self._calculate_performance_trend_score(analytics_id)
            features.append(trend_score)
            
            # Violation history features
            recent_violations = len([v for v in self.active_violations.values() 
                                   if (current_time - v.timestamp).total_seconds() < 3600])
            features.append(float(recent_violations))
            
            # Adaptive threshold features
            if analytics_id in self.adaptive_thresholds:
                thresholds = self.adaptive_thresholds[analytics_id]
                features.append(thresholds.get('latency', 100.0))
                features.append(thresholds.get('availability', 99.0))
            else:
                features.extend([100.0, 99.0])
            
            # Ensure consistent feature count
            while len(features) < self.ml_config["feature_extraction_depth"]:
                features.append(0.0)
            
            return features[:self.ml_config["feature_extraction_depth"]]
            
        except Exception as e:
            logger.debug(f"SLA feature extraction error: {e}")
            return [0.0] * self.ml_config["feature_extraction_depth"]
    
    def _predict_delivery_latency(self, features: List[float], sla_level: MLSLALevel) -> float:
        """Predict delivery latency using ML"""
        try:
            if not self.latency_predictor or len(features) < 10:
                # Return baseline based on SLA level
                baselines = {
                    MLSLALevel.PLATINUM: 30.0,
                    MLSLALevel.GOLD: 80.0,
                    MLSLALevel.SILVER: 200.0,
                    MLSLALevel.BRONZE: 400.0,
                    MLSLALevel.ADAPTIVE: 100.0
                }
                return baselines.get(sla_level, 100.0)
            
            # Scale features
            if 'latency_prediction' in self.scalers:
                features_scaled = self.scalers['latency_prediction'].transform([features])
            else:
                features_scaled = [features]
            
            # Make prediction
            prediction = self.latency_predictor.predict(features_scaled)[0]
            self.optimizer_stats['ml_predictions_made'] += 1
            
            return max(10.0, min(5000.0, prediction))  # Clamp between 10ms and 5s
            
        except Exception as e:
            logger.debug(f"Latency prediction error: {e}")
            return 100.0
    
    def _predict_violation_probability(self, features: List[float], sla_level: MLSLALevel) -> float:
        """Predict SLA violation probability using ML"""
        try:
            if not self.violation_predictor or len(features) < 10:
                return 0.1  # Default low probability
            
            # Scale features
            if 'violation_prediction' in self.scalers:
                features_scaled = self.scalers['violation_prediction'].transform([features])
            else:
                features_scaled = [features]
            
            # Make prediction
            if hasattr(self.violation_predictor, 'predict_proba'):
                probability = self.violation_predictor.predict_proba(features_scaled)[0][1]
            else:
                probability = self.violation_predictor.predict(features_scaled)[0]
            
            self.optimizer_stats['ml_predictions_made'] += 1
            
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            logger.debug(f"Violation prediction error: {e}")
            return 0.1
    
    def _generate_performance_prediction(self, analytics_id: str, features: List[float], 
                                       sla_level: MLSLALevel) -> Optional[MLPerformancePrediction]:
        """Generate comprehensive performance prediction"""
        try:
            horizon_minutes = self.ml_config["prediction_horizon_minutes"]
            
            # Predict key metrics
            predicted_latency = self._predict_delivery_latency(features, sla_level)
            predicted_availability = self._predict_availability(features)
            predicted_throughput = self._predict_throughput(features)
            predicted_error_rate = self._predict_error_rate(features)
            
            # Calculate confidence score
            confidence_score = self._calculate_prediction_confidence(features)
            
            # Assess risk level
            risk_level = self._assess_violation_risk(
                predicted_latency, predicted_availability, predicted_error_rate, sla_level
            )
            
            # Generate optimization opportunities
            opportunities = self._identify_optimization_opportunities(
                predicted_latency, predicted_availability, predicted_throughput, features
            )
            
            # Generate resource recommendations
            resource_recs = self._generate_resource_recommendations(features, risk_level)
            
            return MLPerformancePrediction(
                analytics_id=analytics_id,
                prediction_timestamp=datetime.now(),
                prediction_horizon_minutes=horizon_minutes,
                predicted_latency=predicted_latency,
                predicted_availability=predicted_availability,
                predicted_throughput=predicted_throughput,
                predicted_error_rate=predicted_error_rate,
                confidence_score=confidence_score,
                risk_assessment=risk_level,
                optimization_opportunities=opportunities,
                resource_recommendations=resource_recs
            )
            
        except Exception as e:
            logger.debug(f"Performance prediction generation error: {e}")
            return None
    
    def _predict_availability(self, features: List[float]) -> float:
        """Predict availability percentage"""
        try:
            # Simple heuristic based on features
            if len(features) >= 7:
                success_rate = features[6]  # Historical success rate
                return min(99.99, success_rate * 100)
            return 99.5
            
        except Exception:
            return 99.5
    