"""
Watchdog ML Monitor (Part 1/3) - TestMaster Advanced ML
ML-driven component monitoring with predictive failure detection
Extracted from analytics_watchdog.py (674 lines) → 3 coordinated ML modules
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import psutil
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


class WatchdogAction(Enum):
    RESTART_COMPONENT = "restart_component"
    RESTART_SERVICE = "restart_service"
    RESTART_PROCESS = "restart_process"
    ALERT_ONLY = "alert_only"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"
    FORCE_RESTART = "force_restart"
    ML_OPTIMIZE = "ml_optimize"
    PREDICTIVE_SCALE = "predictive_scale"


class ComponentState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"
    PREDICTED_FAILURE = "predicted_failure"
    ML_OPTIMIZING = "ml_optimizing"


@dataclass
class MLWatchdogRule:
    """ML-enhanced watchdog monitoring rule"""
    
    rule_id: str
    component_name: str
    check_function: Callable[[], bool]
    failure_threshold: int
    check_interval: int
    action: WatchdogAction
    recovery_timeout: int
    max_restarts: int
    escalation_actions: List[WatchdogAction] = field(default_factory=list)
    
    # ML Enhancement Fields
    ml_enabled: bool = True
    predictive_threshold: float = 0.8
    anomaly_sensitivity: float = 0.1
    pattern_learning: bool = True
    failure_prediction_horizon: int = 300  # seconds


@dataclass
class MLComponentHealth:
    """ML-enhanced component health status"""
    
    component_name: str
    state: ComponentState
    last_check: datetime
    consecutive_failures: int
    total_failures: int
    restart_count: int
    last_restart: Optional[datetime]
    health_score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # ML Enhancement Fields
    failure_probability: float = 0.0
    anomaly_score: float = 0.0
    pattern_cluster: int = -1
    performance_trend: str = "stable"  # improving, stable, degrading
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    predicted_failure_time: Optional[datetime] = None
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class MLHealthFeatures:
    """ML features extracted from component health data"""
    
    timestamp: datetime
    component_name: str
    cpu_usage: float
    memory_usage: float
    response_time: float
    error_rate: float
    throughput: float
    consecutive_failures: int
    restart_count: int
    uptime_seconds: float
    health_score: float
    
    # Derived features
    cpu_trend: float = 0.0
    memory_trend: float = 0.0
    error_rate_change: float = 0.0
    performance_variance: float = 0.0


class AdvancedWatchdogMLMonitor:
    """
    ML-enhanced watchdog monitoring system with predictive failure detection
    Part 1/3 of the complete watchdog system
    """
    
    def __init__(self,
                 check_interval: int = 30,
                 max_restart_attempts: int = 3,
                 enable_ml_prediction: bool = True,
                 prediction_interval: int = 60):
        """Initialize ML-enhanced watchdog monitor"""
        
        self.check_interval = check_interval
        self.max_restart_attempts = max_restart_attempts
        self.enable_ml_prediction = enable_ml_prediction
        self.prediction_interval = prediction_interval
        
        # ML Models for Predictive Monitoring
        self.failure_predictor: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.pattern_clusterer: Optional[KMeans] = None
        self.performance_analyzer: Optional[LogisticRegression] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.ml_feature_history: deque = deque(maxlen=1000)
        
        # Monitoring Components
        self.watchdog_rules: Dict[str, MLWatchdogRule] = {}
        self.component_health: Dict[str, MLComponentHealth] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # ML Insights and Predictions
        self.ml_predictions: Dict[str, Dict[str, Any]] = {}
        self.anomaly_patterns: List[Dict[str, Any]] = []
        self.performance_clusters: Dict[int, List[str]] = {}
        
        # System State
        self.watchdog_active = False
        self.critical_components: set = set()
        
        # Configuration
        self.restart_cooldown = 60
        self.health_score_threshold = 0.7
        self.cascade_failure_threshold = 3
        self.ml_training_threshold = 50  # Minimum samples for ML training
        
        # Statistics
        self.watchdog_stats = {
            'total_checks': 0,
            'failed_checks': 0,
            'ml_predictions_made': 0,
            'ml_predictions_accurate': 0,
            'anomalies_detected': 0,
            'predictive_actions_taken': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.monitoring_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and start monitoring
        if enable_ml_prediction:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_prediction_loop())
        
        asyncio.create_task(self._monitoring_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for predictive monitoring"""
        
        try:
            # Failure prediction classifier
            self.failure_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            )
            
            # Anomaly detection for unusual behavior patterns
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Pattern clustering for component behavior grouping
            self.pattern_clusterer = KMeans(
                n_clusters=6,
                random_state=42,
                n_init=10
            )
            
            # Performance trend analysis
            self.performance_analyzer = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            self.logger.info("Watchdog ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Watchdog ML model initialization failed: {e}")
            self.enable_ml_prediction = False
    
    def register_component(self,
                          component_name: str,
                          component_instance: Any,
                          check_function: Optional[Callable] = None,
                          failure_threshold: int = 3,
                          restart_action: WatchdogAction = WatchdogAction.RESTART_COMPONENT,
                          is_critical: bool = False,
                          enable_ml: bool = True) -> str:
        """Register component for ML-enhanced monitoring"""
        
        # Create health check function
        if check_function is None:
            check_function = self._create_default_health_check(component_instance)
        
        # Create ML-enhanced watchdog rule
        rule_id = f"rule_{component_name}_{int(time.time())}"
        rule = MLWatchdogRule(
            rule_id=rule_id,
            component_name=component_name,
            check_function=check_function,
            failure_threshold=failure_threshold,
            check_interval=self.check_interval,
            action=restart_action,
            recovery_timeout=300,
            max_restarts=self.max_restart_attempts,
            ml_enabled=enable_ml
        )
        
        # Initialize component health
        health = MLComponentHealth(
            component_name=component_name,
            state=ComponentState.HEALTHY,
            last_check=datetime.now(),
            consecutive_failures=0,
            total_failures=0,
            restart_count=0,
            last_restart=None,
            health_score=1.0
        )
        
        with self.monitoring_lock:
            self.watchdog_rules[component_name] = rule
            self.component_health[component_name] = health
            
            if is_critical:
                self.critical_components.add(component_name)
        
        self.logger.info(f"Component registered for ML monitoring: {component_name}")
        return rule_id
    
    async def check_component_health(self, component_name: str) -> MLComponentHealth:
        """Perform ML-enhanced health check on component"""
        
        if component_name not in self.watchdog_rules:
            self.logger.error(f"Component not registered: {component_name}")
            return None
        
        rule = self.watchdog_rules[component_name]
        health = self.component_health[component_name]
        
        try:
            # Collect comprehensive metrics
            start_time = time.time()
            
            # Perform basic health check
            is_healthy = rule.check_function()
            
            # Collect system metrics
            system_metrics = await self._collect_system_metrics()
            
            # Collect component-specific metrics
            component_metrics = await self._collect_component_metrics(component_name)
            
            # Create ML features
            features = MLHealthFeatures(
                timestamp=datetime.now(),
                component_name=component_name,
                cpu_usage=system_metrics.get('cpu_usage', 0.0),
                memory_usage=system_metrics.get('memory_usage', 0.0),
                response_time=(time.time() - start_time) * 1000,
                error_rate=component_metrics.get('error_rate', 0.0),
                throughput=component_metrics.get('throughput', 0.0),
                consecutive_failures=health.consecutive_failures,
                restart_count=health.restart_count,
                uptime_seconds=(datetime.now() - self.watchdog_stats['start_time']).total_seconds(),
                health_score=health.health_score
            )
            
            # Update health based on basic check
            health.last_check = datetime.now()
            
            if is_healthy:
                health.state = ComponentState.HEALTHY
                health.consecutive_failures = 0
                health.health_score = min(1.0, health.health_score + 0.1)
            else:
                health.consecutive_failures += 1
                health.total_failures += 1
                health.health_score = max(0.0, health.health_score - 0.2)
                
                if health.consecutive_failures >= rule.failure_threshold:
                    health.state = ComponentState.FAILED
                else:
                    health.state = ComponentState.DEGRADED
            
            # ML Enhancement
            if rule.ml_enabled and self.enable_ml_prediction:
                await self._enhance_health_with_ml(health, features)
            
            # Store features for ML learning
            self.ml_feature_history.append(features)
            self.health_history[component_name].append(health)
            
            self.watchdog_stats['total_checks'] += 1
            
            return health
            
        except Exception as e:
            self.logger.error(f"Health check failed for {component_name}: {e}")
            health.state = ComponentState.UNKNOWN
            health.consecutive_failures += 1
            return health
    
    async def _enhance_health_with_ml(self, health: MLComponentHealth, features: MLHealthFeatures):
        """Enhance health status with ML analysis"""
        
        try:
            with self.ml_lock:
                # Failure prediction
                if self.failure_predictor and len(self.ml_feature_history) >= self.ml_training_threshold:
                    failure_prob = await self._predict_failure_probability(features)
                    health.failure_probability = failure_prob
                    
                    # Predictive failure detection
                    rule = self.watchdog_rules[health.component_name]
                    if failure_prob > rule.predictive_threshold:
                        health.state = ComponentState.PREDICTED_FAILURE
                        health.predicted_failure_time = datetime.now() + timedelta(
                            seconds=rule.failure_prediction_horizon
                        )
                        
                        self.watchdog_stats['ml_predictions_made'] += 1
                        self.logger.warning(
                            f"Predicted failure for {health.component_name}: {failure_prob:.2f}"
                        )
                
                # Anomaly detection
                if self.anomaly_detector:
                    anomaly_score = await self._detect_anomalies(features)
                    health.anomaly_score = anomaly_score
                    
                    if anomaly_score < -rule.anomaly_sensitivity:
                        health.ml_insights['anomaly_detected'] = True
                        self.watchdog_stats['anomalies_detected'] += 1
                
                # Pattern clustering
                if self.pattern_clusterer and len(self.ml_feature_history) >= 20:
                    cluster_id = await self._assign_pattern_cluster(features)
                    health.pattern_cluster = cluster_id
                    
                    # Update cluster information
                    if cluster_id not in self.performance_clusters:
                        self.performance_clusters[cluster_id] = []
                    
                    if health.component_name not in self.performance_clusters[cluster_id]:
                        self.performance_clusters[cluster_id].append(health.component_name)
                
                # Performance trend analysis
                health.performance_trend = await self._analyze_performance_trend(health.component_name)
                
                # Generate optimization suggestions
                health.optimization_suggestions = await self._generate_ml_suggestions(health, features)
                
        except Exception as e:
            self.logger.error(f"ML enhancement failed for {health.component_name}: {e}")
    
    async def _predict_failure_probability(self, features: MLHealthFeatures) -> float:
        """Predict failure probability using ML model"""
        
        try:
            # Extract feature vector
            feature_vector = self._extract_feature_vector(features)
            
            if self.failure_predictor and len(self.ml_feature_history) >= self.ml_training_threshold:
                # Retrain model if needed
                await self._retrain_models_if_needed()
                
                # Make prediction
                failure_prob = self.failure_predictor.predict_proba([feature_vector])[0][1]
                return float(failure_prob)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failure prediction error: {e}")
            return 0.0
    
    def _extract_feature_vector(self, features: MLHealthFeatures) -> np.ndarray:
        """Extract numerical feature vector for ML processing"""
        
        vector = np.array([
            features.cpu_usage,
            features.memory_usage,
            features.response_time,
            features.error_rate,
            features.throughput,
            features.consecutive_failures,
            features.restart_count,
            np.log1p(features.uptime_seconds),
            features.health_score,
            features.cpu_trend,
            features.memory_trend,
            features.error_rate_change,
            features.performance_variance,
            features.timestamp.hour,
            features.timestamp.weekday()
        ])
        
        return vector.astype(np.float64)
    
    async def _monitoring_loop(self):
        """Main monitoring loop with ML insights"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.check_interval)
                
                # Check all registered components
                for component_name in list(self.watchdog_rules.keys()):
                    if self.shutdown_event.is_set():
                        break
                    
                    await self.check_component_health(component_name)
                
                # Check for cascade failures
                await self._check_cascade_failures()
                
                # Update ML insights
                if self.enable_ml_prediction:
                    await self._update_ml_insights()
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _ml_prediction_loop(self):
        """ML prediction and model maintenance loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.prediction_interval)
                
                if len(self.ml_feature_history) >= self.ml_training_threshold:
                    # Retrain models periodically
                    await self._retrain_models_if_needed()
                    
                    # Generate predictive insights
                    await self._generate_predictive_insights()
                    
                    # Validate predictions
                    await self._validate_ml_predictions()
                
            except Exception as e:
                self.logger.error(f"ML prediction loop error: {e}")
                await asyncio.sleep(10)
    
    def get_ml_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive ML monitoring dashboard"""
        
        # Component health summary
        health_summary = {}
        for name, health in self.component_health.items():
            health_summary[name] = {
                'state': health.state.value,
                'health_score': health.health_score,
                'failure_probability': health.failure_probability,
                'anomaly_score': health.anomaly_score,
                'performance_trend': health.performance_trend,
                'consecutive_failures': health.consecutive_failures,
                'optimization_suggestions': health.optimization_suggestions[:3]
            }
        
        # ML insights summary
        ml_insights = {
            'total_predictions': self.watchdog_stats['ml_predictions_made'],
            'prediction_accuracy': (
                self.watchdog_stats['ml_predictions_accurate'] / 
                max(1, self.watchdog_stats['ml_predictions_made'])
            ),
            'anomalies_detected': self.watchdog_stats['anomalies_detected'],
            'pattern_clusters': len(self.performance_clusters),
            'feature_history_size': len(self.ml_feature_history)
        }
        
        return {
            'system_overview': {
                'active_components': len(self.watchdog_rules),
                'critical_components': len(self.critical_components),
                'healthy_components': len([h for h in self.component_health.values() 
                                         if h.state == ComponentState.HEALTHY]),
                'failed_components': len([h for h in self.component_health.values() 
                                        if h.state == ComponentState.FAILED])
            },
            'component_health': health_summary,
            'ml_insights': ml_insights,
            'statistics': self.watchdog_stats.copy()
        }
    
    async def shutdown(self):
        """Graceful shutdown of ML monitor"""
        
        self.logger.info("Shutting down ML watchdog monitor...")
        self.shutdown_event.set()
        self.watchdog_active = False
        await asyncio.sleep(1)
        self.logger.info("ML watchdog monitor shutdown complete")