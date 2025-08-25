"""
Predictive Maintenance AI - TestMaster Advanced ML
Advanced predictive maintenance with ML-driven failure prediction and maintenance scheduling
Enterprise ML Module #8/8 for comprehensive system intelligence
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
import uuid
import json
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ComponentType(Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    STORAGE = "storage"
    DATABASE = "database"
    APPLICATION = "application"


class HealthStatus(Enum):
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    DEGRADED = 2
    CRITICAL = 1


class MaintenanceType(Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    OPTIMIZATION = "optimization"


class MaintenancePriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    SCHEDULED = 5


@dataclass
class SystemComponent:
    """ML-enhanced system component with predictive capabilities"""
    
    component_id: str
    component_name: str
    component_type: ComponentType
    location: str = ""
    vendor: str = ""
    model: str = ""
    serial_number: str = ""
    
    # Operational metrics
    uptime_hours: float = 0.0
    utilization_rate: float = 0.0
    error_rate: float = 0.0
    performance_score: float = 1.0
    temperature: float = 0.0
    vibration_level: float = 0.0
    
    # Health assessment
    current_health: HealthStatus = HealthStatus.GOOD
    health_trend: str = "stable"  # improving, stable, degrading
    last_maintenance: Optional[datetime] = None
    next_scheduled_maintenance: Optional[datetime] = None
    
    # ML Enhancement Fields
    failure_probability: float = 0.0
    time_to_failure_hours: float = 0.0
    maintenance_urgency_score: float = 0.0
    anomaly_score: float = 0.0
    degradation_rate: float = 0.0
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Maintenance history
    maintenance_count: int = 0
    total_downtime_hours: float = 0.0
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    
    # Sensor data
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    historical_readings: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class MaintenanceTask:
    """ML-optimized maintenance task"""
    
    task_id: str
    component_id: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    description: str
    estimated_duration_hours: float = 0.0
    estimated_cost: float = 0.0
    
    # Scheduling
    scheduled_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    maintenance_window_start: Optional[datetime] = None
    maintenance_window_end: Optional[datetime] = None
    
    # ML Enhancement
    predicted_impact: float = 0.0
    success_probability: float = 1.0
    risk_reduction_score: float = 0.0
    business_impact_score: float = 0.0
    
    # Execution tracking
    status: str = "planned"  # planned, scheduled, in_progress, completed, cancelled
    assigned_technician: Optional[str] = None
    actual_start: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    actual_duration_hours: float = 0.0
    actual_cost: float = 0.0
    
    # Requirements
    required_parts: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    
    # Results
    completed_successfully: bool = True
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PredictiveAlert:
    """ML-generated predictive maintenance alert"""
    
    alert_id: str
    component_id: str
    alert_type: str  # failure_prediction, degradation_detected, maintenance_due
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    
    # ML predictions
    confidence: float = 0.0
    time_to_event_hours: float = 0.0
    recommended_actions: List[str] = field(default_factory=list)
    
    # Alert lifecycle
    acknowledged: bool = False
    resolved: bool = False
    false_positive: bool = False
    resolution_notes: str = ""


class PredictiveMaintenanceAI:
    """
    ML-enhanced predictive maintenance system with intelligent failure prediction
    """
    
    def __init__(self,
                 enable_ml_prediction: bool = True,
                 monitoring_interval: int = 300,  # 5 minutes
                 prediction_horizon_hours: int = 168,  # 1 week
                 maintenance_optimization: bool = True):
        """Initialize predictive maintenance AI system"""
        
        self.enable_ml_prediction = enable_ml_prediction
        self.monitoring_interval = monitoring_interval
        self.prediction_horizon_hours = prediction_horizon_hours
        self.maintenance_optimization = maintenance_optimization
        
        # ML Models for Predictive Maintenance
        self.failure_predictor: Optional[RandomForestClassifier] = None
        self.degradation_detector: Optional[IsolationForest] = None
        self.maintenance_optimizer: Optional[GradientBoostingRegressor] = None
        self.anomaly_detector: Optional[DBSCAN] = None
        self.health_classifier: Optional[LogisticRegression] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.health_scaler = RobustScaler()
        self.maintenance_feature_history: deque = deque(maxlen=50000)
        
        # System Components and Maintenance
        self.system_components: Dict[str, SystemComponent] = {}
        self.maintenance_tasks: Dict[str, MaintenanceTask] = {}
        self.active_alerts: Dict[str, PredictiveAlert] = {}
        self.completed_maintenance: deque = deque(maxlen=2000)
        
        # Performance Monitoring
        self.component_health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.failure_history: List[Dict[str, Any]] = []
        self.maintenance_effectiveness: deque = deque(maxlen=500)
        
        # ML Insights and Predictions
        self.failure_predictions: Dict[str, Dict[str, Any]] = {}
        self.maintenance_recommendations: List[Dict[str, Any]] = []
        self.optimization_insights: List[Dict[str, Any]] = []
        
        # Configuration
        self.failure_threshold = 0.7
        self.degradation_threshold = 0.8
        self.alert_cooldown_hours = 24
        self.maintenance_window_hours = 4
        
        # Statistics
        self.maintenance_stats = {
            'components_monitored': 0,
            'predictions_made': 0,
            'failures_prevented': 0,
            'maintenance_tasks_completed': 0,
            'false_positive_rate': 0.0,
            'prediction_accuracy': 0.0,
            'cost_savings': 0.0,
            'downtime_avoided_hours': 0.0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.maintenance_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and monitoring
        if enable_ml_prediction:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_prediction_loop())
        
        asyncio.create_task(self._maintenance_monitoring_loop())
        asyncio.create_task(self._maintenance_scheduling_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for predictive maintenance"""
        
        try:
            # Failure prediction classifier
            self.failure_predictor = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                class_weight='balanced',
                min_samples_split=5
            )
            
            # Component degradation detection
            self.degradation_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Maintenance scheduling optimization
            self.maintenance_optimizer = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            )
            
            # Sensor anomaly detection clustering
            self.anomaly_detector = DBSCAN(
                eps=0.3,
                min_samples=5,
                metric='euclidean'
            )
            
            # Health status classification
            self.health_classifier = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            self.logger.info("Predictive maintenance ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Predictive maintenance ML model initialization failed: {e}")
            self.enable_ml_prediction = False
    
    def register_component(self,
                          component_id: str,
                          component_name: str,
                          component_type: ComponentType,
                          location: str = "",
                          vendor: str = "",
                          model: str = "") -> bool:
        """Register system component for predictive maintenance monitoring"""
        
        try:
            with self.maintenance_lock:
                component = SystemComponent(
                    component_id=component_id,
                    component_name=component_name,
                    component_type=component_type,
                    location=location,
                    vendor=vendor,
                    model=model
                )
                
                self.system_components[component_id] = component
                self.component_health_history[component_id] = deque(maxlen=1000)
                self.maintenance_stats['components_monitored'] += 1
            
            self.logger.info(f"Component registered for predictive maintenance: {component_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Component registration failed: {e}")
            return False
    
    def update_component_metrics(self,
                                component_id: str,
                                uptime_hours: float = None,
                                utilization_rate: float = None,
                                error_rate: float = None,
                                performance_score: float = None,
                                temperature: float = None,
                                vibration_level: float = None,
                                sensor_readings: Dict[str, float] = None) -> bool:
        """Update component operational metrics with ML analysis"""
        
        try:
            with self.maintenance_lock:
                if component_id not in self.system_components:
                    return False
                
                component = self.system_components[component_id]
                
                # Update metrics
                if uptime_hours is not None:
                    component.uptime_hours = uptime_hours
                if utilization_rate is not None:
                    component.utilization_rate = utilization_rate
                if error_rate is not None:
                    component.error_rate = error_rate
                if performance_score is not None:
                    component.performance_score = performance_score
                if temperature is not None:
                    component.temperature = temperature
                if vibration_level is not None:
                    component.vibration_level = vibration_level
                if sensor_readings:
                    component.sensor_readings.update(sensor_readings)
                
                # Store historical reading
                reading = {
                    'timestamp': datetime.now(),
                    'uptime_hours': component.uptime_hours,
                    'utilization_rate': component.utilization_rate,
                    'error_rate': component.error_rate,
                    'performance_score': component.performance_score,
                    'temperature': component.temperature,
                    'vibration_level': component.vibration_level,
                    'sensor_readings': component.sensor_readings.copy()
                }
                
                component.historical_readings.append(reading)
                self.component_health_history[component_id].append(reading)
            
            # ML analysis if enabled
            if self.enable_ml_prediction:
                asyncio.create_task(self._analyze_component_with_ml(component_id))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component metrics update failed: {e}")
            return False
    
    async def _analyze_component_with_ml(self, component_id: str):
        """Perform ML analysis on component health and predict maintenance needs"""
        
        try:
            component = self.system_components[component_id]
            
            with self.ml_lock:
                # Extract features for ML analysis
                features = await self._extract_component_features(component)
                
                if len(self.maintenance_feature_history) >= 100:
                    # Failure prediction
                    if self.failure_predictor:
                        failure_prob = await self._predict_component_failure(features, component)
                        component.failure_probability = failure_prob
                        
                        if failure_prob > self.failure_threshold:
                            await self._generate_failure_alert(component, failure_prob)
                    
                    # Degradation detection
                    if self.degradation_detector:
                        anomaly_score = self.degradation_detector.decision_function([features])[0]
                        component.anomaly_score = float(anomaly_score)
                        
                        if anomaly_score < -self.degradation_threshold:
                            await self._generate_degradation_alert(component, anomaly_score)
                    
                    # Health status prediction
                    if self.health_classifier:
                        health_prediction = await self._predict_health_status(features, component)
                        if health_prediction != component.current_health:
                            component.current_health = health_prediction
                            component.ml_insights['health_status_changed'] = True
                    
                    # Calculate time to failure
                    component.time_to_failure_hours = await self._estimate_time_to_failure(
                        component, features
                    )
                    
                    # Calculate maintenance urgency
                    component.maintenance_urgency_score = await self._calculate_maintenance_urgency(
                        component
                    )
                    
                    # Update degradation rate
                    component.degradation_rate = await self._calculate_degradation_rate(component)
                
                # Store features for model training
                self.maintenance_feature_history.append(features)
                self.maintenance_stats['predictions_made'] += 1
                
        except Exception as e:
            self.logger.error(f"ML component analysis failed: {e}")
    
    def _extract_component_features(self, component: SystemComponent) -> np.ndarray:
        """Extract ML features from component data"""
        
        try:
            # Basic component features
            component_type_encoded = list(ComponentType).index(component.component_type)
            
            # Operational metrics
            uptime_days = component.uptime_hours / 24.0
            utilization_normalized = min(component.utilization_rate, 1.0)
            error_rate_normalized = min(component.error_rate * 100, 1.0)
            performance_score = component.performance_score
            
            # Environmental conditions
            temperature_normalized = component.temperature / 100.0  # Assume max 100°C
            vibration_normalized = min(component.vibration_level / 10.0, 1.0)  # Normalize
            
            # Historical trends
            recent_readings = list(component.historical_readings)[-10:]
            if recent_readings:
                performance_trend = np.mean([r['performance_score'] for r in recent_readings])
                error_trend = np.mean([r['error_rate'] for r in recent_readings])
                temp_trend = np.mean([r['temperature'] for r in recent_readings]) / 100.0
            else:
                performance_trend = performance_score
                error_trend = component.error_rate
                temp_trend = temperature_normalized
            
            # Maintenance history features
            days_since_maintenance = 0.0
            if component.last_maintenance:
                days_since_maintenance = (datetime.now() - component.last_maintenance).days
            
            maintenance_frequency = component.maintenance_count / max(uptime_days, 1.0)
            failure_frequency = component.failure_count / max(uptime_days, 1.0)
            
            # Sensor aggregations
            sensor_values = list(component.sensor_readings.values()) if component.sensor_readings else [0.0]
            sensor_mean = np.mean(sensor_values)
            sensor_std = np.std(sensor_values) if len(sensor_values) > 1 else 0.0
            
            # Create feature vector
            features = np.array([
                component_type_encoded,
                uptime_days / 365.0,  # Normalize to years
                utilization_normalized,
                error_rate_normalized,
                performance_score,
                temperature_normalized,
                vibration_normalized,
                performance_trend,
                error_trend,
                temp_trend,
                days_since_maintenance / 365.0,  # Normalize
                maintenance_frequency * 365.0,  # Annualized
                failure_frequency * 365.0,  # Annualized
                sensor_mean,
                sensor_std,
                component.total_downtime_hours / max(component.uptime_hours, 1.0),
                datetime.now().hour / 24.0,
                datetime.now().weekday() / 7.0
            ])
            
            return features.astype(np.float64)
            
        except Exception as e:
            self.logger.error(f"Component feature extraction failed: {e}")
            return np.zeros(18)  # Default feature vector
    
    async def _predict_component_failure(self, features: np.ndarray, component: SystemComponent) -> float:
        """Predict component failure probability"""
        
        try:
            if not self.failure_predictor or len(self.maintenance_feature_history) < 100:
                return 0.0
            
            # Use ensemble prediction for better reliability
            failure_prob = self.failure_predictor.predict_proba([features])[0][1]  # Probability of failure class
            
            # Adjust based on component history
            if component.failure_count > 0:
                failure_prob *= (1.0 + component.failure_count * 0.1)
            
            # Adjust based on maintenance history
            if component.last_maintenance:
                days_since_maintenance = (datetime.now() - component.last_maintenance).days
                if days_since_maintenance > 365:  # Over a year
                    failure_prob *= 1.2
            
            return min(1.0, failure_prob)
            
        except Exception as e:
            self.logger.error(f"Failure prediction failed: {e}")
            return 0.0
    
    async def _generate_failure_alert(self, component: SystemComponent, failure_probability: float):
        """Generate predictive failure alert"""
        
        try:
            alert_id = f"failure_alert_{component.component_id}_{int(time.time())}"
            
            # Check cooldown period
            recent_alerts = [
                alert for alert in self.active_alerts.values()
                if (alert.component_id == component.component_id and
                    alert.alert_type == 'failure_prediction' and
                    (datetime.now() - alert.timestamp).total_seconds() < self.alert_cooldown_hours * 3600)
            ]
            
            if recent_alerts:
                return  # Skip if recent similar alert exists
            
            # Determine severity
            if failure_probability > 0.9:
                severity = "critical"
            elif failure_probability > 0.8:
                severity = "high"
            elif failure_probability > 0.7:
                severity = "medium"
            else:
                severity = "low"
            
            # Generate recommended actions
            recommended_actions = []
            if failure_probability > 0.8:
                recommended_actions.extend([
                    "Schedule immediate inspection",
                    "Prepare replacement parts",
                    "Plan maintenance window"
                ])
            else:
                recommended_actions.extend([
                    "Increase monitoring frequency",
                    "Schedule preventive maintenance",
                    "Review maintenance history"
                ])
            
            alert = PredictiveAlert(
                alert_id=alert_id,
                component_id=component.component_id,
                alert_type="failure_prediction",
                severity=severity,
                message=f"High failure probability detected for {component.component_name} ({failure_probability:.2%})",
                timestamp=datetime.now(),
                confidence=failure_probability,
                time_to_event_hours=component.time_to_failure_hours,
                recommended_actions=recommended_actions
            )
            
            with self.maintenance_lock:
                self.active_alerts[alert_id] = alert
            
            self.logger.warning(f"Failure prediction alert generated: {component.component_name}")
            
            # Auto-schedule maintenance if high probability
            if failure_probability > 0.8 and self.maintenance_optimization:
                await self._auto_schedule_maintenance(component, MaintenanceType.PREDICTIVE)
            
        except Exception as e:
            self.logger.error(f"Failure alert generation failed: {e}")
    
    async def _maintenance_monitoring_loop(self):
        """Main maintenance monitoring and health assessment loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Update component health assessments
                await self._update_component_health_assessments()
                
                # Check for overdue maintenance
                await self._check_overdue_maintenance()
                
                # Update maintenance effectiveness metrics
                await self._update_maintenance_effectiveness()
                
                # Generate maintenance insights
                if self.enable_ml_prediction:
                    await self._generate_maintenance_insights()
                
            except Exception as e:
                self.logger.error(f"Maintenance monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _ml_prediction_loop(self):
        """ML prediction and model training loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                if len(self.maintenance_feature_history) >= 500:
                    # Retrain ML models
                    await self._retrain_maintenance_models()
                    
                    # Update prediction accuracy metrics
                    await self._update_prediction_accuracy()
                    
                    # Generate optimization recommendations
                    await self._generate_optimization_recommendations()
                
            except Exception as e:
                self.logger.error(f"ML prediction loop error: {e}")
                await asyncio.sleep(60)
    
    async def _maintenance_scheduling_loop(self):
        """Intelligent maintenance scheduling loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Optimize maintenance schedules
                if self.maintenance_optimization:
                    await self._optimize_maintenance_schedules()
                
                # Check maintenance windows
                await self._check_maintenance_windows()
                
                # Update maintenance task priorities
                await self._update_maintenance_priorities()
                
            except Exception as e:
                self.logger.error(f"Maintenance scheduling loop error: {e}")
                await asyncio.sleep(300)
    
    def get_maintenance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive predictive maintenance dashboard"""
        
        # Component health summary
        health_distribution = defaultdict(int)
        for component in self.system_components.values():
            health_distribution[component.current_health.name] += 1
        
        # High-risk components (failure probability > 50%)
        high_risk_components = [
            {
                'component_id': comp.component_id,
                'component_name': comp.component_name,
                'failure_probability': comp.failure_probability,
                'time_to_failure_hours': comp.time_to_failure_hours,
                'maintenance_urgency_score': comp.maintenance_urgency_score
            }
            for comp in self.system_components.values()
            if comp.failure_probability > 0.5
        ]
        
        # Active maintenance tasks
        maintenance_status = defaultdict(int)
        for task in self.maintenance_tasks.values():
            maintenance_status[task.status] += 1
        
        # Recent predictions
        recent_predictions = len([
            comp for comp in self.system_components.values()
            if comp.failure_probability > 0.0
        ])
        
        # Active alerts by severity
        alert_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            alert_severity[alert.severity] += 1
        
        return {
            'maintenance_overview': {
                'components_monitored': len(self.system_components),
                'active_alerts': len(self.active_alerts),
                'scheduled_maintenance_tasks': len([t for t in self.maintenance_tasks.values() if t.status == 'scheduled']),
                'high_risk_components': len(high_risk_components),
                'prediction_accuracy': self.maintenance_stats['prediction_accuracy'],
                'failures_prevented': self.maintenance_stats['failures_prevented']
            },
            'component_health': dict(health_distribution),
            'high_risk_components': high_risk_components[:10],  # Top 10
            'maintenance_tasks': dict(maintenance_status),
            'active_alerts': dict(alert_severity),
            'recent_predictions': recent_predictions,
            'statistics': self.maintenance_stats.copy(),
            'ml_status': {
                'ml_prediction_enabled': self.enable_ml_prediction,
                'feature_history_size': len(self.maintenance_feature_history),
                'models_trained': self.failure_predictor is not None,
                'maintenance_recommendations': len(self.maintenance_recommendations)
            },
            'recent_insights': self.optimization_insights[-5:] if self.optimization_insights else []
        }
    
    def get_component_details(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about specific component"""
        
        if component_id not in self.system_components:
            return None
        
        component = self.system_components[component_id]
        
        # Recent health history
        recent_readings = list(component.historical_readings)[-20:]
        health_timeline = [
            {
                'timestamp': reading['timestamp'].isoformat(),
                'performance_score': reading['performance_score'],
                'error_rate': reading['error_rate'],
                'temperature': reading['temperature']
            }
            for reading in recent_readings
        ]
        
        # Related alerts
        component_alerts = [
            {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'confidence': alert.confidence
            }
            for alert in self.active_alerts.values()
            if alert.component_id == component_id
        ]
        
        return {
            'component_info': {
                'component_id': component.component_id,
                'component_name': component.component_name,
                'component_type': component.component_type.value,
                'location': component.location,
                'vendor': component.vendor,
                'model': component.model
            },
            'current_health': {
                'health_status': component.current_health.name,
                'health_trend': component.health_trend,
                'failure_probability': component.failure_probability,
                'time_to_failure_hours': component.time_to_failure_hours,
                'maintenance_urgency_score': component.maintenance_urgency_score,
                'anomaly_score': component.anomaly_score
            },
            'operational_metrics': {
                'uptime_hours': component.uptime_hours,
                'utilization_rate': component.utilization_rate,
                'error_rate': component.error_rate,
                'performance_score': component.performance_score,
                'temperature': component.temperature,
                'vibration_level': component.vibration_level
            },
            'maintenance_history': {
                'maintenance_count': component.maintenance_count,
                'total_downtime_hours': component.total_downtime_hours,
                'last_maintenance': component.last_maintenance.isoformat() if component.last_maintenance else None,
                'next_scheduled_maintenance': component.next_scheduled_maintenance.isoformat() if component.next_scheduled_maintenance else None,
                'failure_count': component.failure_count,
                'last_failure': component.last_failure.isoformat() if component.last_failure else None
            },
            'health_timeline': health_timeline,
            'active_alerts': component_alerts,
            'ml_insights': component.ml_insights,
            'sensor_readings': component.sensor_readings
        }
    
    async def shutdown(self):
        """Graceful shutdown of predictive maintenance AI"""
        
        self.logger.info("Shutting down predictive maintenance AI...")
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Predictive maintenance AI shutdown complete")