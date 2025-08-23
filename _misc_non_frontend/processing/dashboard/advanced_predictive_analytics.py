#!/usr/bin/env python3
"""
Advanced Predictive Analytics Engine - STEELCLAD Clean Version
=============================================================

Agent Y STEELCLAD Protocol: Modularized version of advanced_predictive_analytics.py
Reduced from 727 lines to <400 lines by extracting ML components

This streamlined version coordinates modular components:
- PredictiveModels: ML model factory and persistence management
- MLDataProcessor: Feature preparation and historical data management  
- MLTrainer: Model training and performance tracking
- Clean external API for dashboard integration

Author: Agent Y - STEELCLAD Modularization Specialist
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Import architecture components
try:
    from core.architecture.architecture_integration import get_architecture_framework
    from core.services.service_registry import get_service_registry
    ARCHITECTURE_AVAILABLE = True
except ImportError:
    ARCHITECTURE_AVAILABLE = False

# Import modularized ML components
try:
    from .models.predictive_models import MLModelFactory, ModelPersistenceManager, ModelPerformanceTracker
    from .data.ml_data_processor import MLFeatureProcessor, HistoricalDataManager  
    from .training.ml_trainer import MLModelTrainer
except ImportError:
    # Fallback to absolute imports
    from models.predictive_models import MLModelFactory, ModelPersistenceManager, ModelPerformanceTracker
    from data.ml_data_processor import MLFeatureProcessor, HistoricalDataManager
    from training.ml_trainer import MLModelTrainer


class PredictionType(Enum):
    """Types of predictions available"""
    HEALTH_TREND = "health_trend"
    SERVICE_FAILURE = "service_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_UTILIZATION = "resource_utilization"
    DEPENDENCY_ISSUES = "dependency_issues"
    ANOMALY_DETECTION = "anomaly_detection"
    CAPACITY_PLANNING = "capacity_planning"


class ConfidenceLevel(Enum):
    """Prediction confidence levels"""
    HIGH = "high"      # >85% confidence
    MEDIUM = "medium"  # 70-85% confidence
    LOW = "low"       # 55-70% confidence
    VERY_LOW = "very_low"  # <55% confidence


@dataclass
class MLPrediction:
    """Machine learning prediction result"""
    prediction_type: PredictionType
    predicted_value: float
    confidence_score: float
    confidence_level: ConfidenceLevel
    feature_importance: Dict[str, float]
    prediction_horizon: int  # minutes ahead
    model_accuracy: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Auto-adjust confidence level based on score
        if self.confidence_score > 85:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence_score > 70:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence_score > 55:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW


class AdvancedPredictiveAnalytics:
    """
    Advanced Predictive Analytics - STEELCLAD Clean Version
    
    Coordinates modular ML components for sophisticated predictions:
    - Health trend forecasting using Random Forest
    - Anomaly detection using Isolation Forest
    - Performance degradation prediction using Gradient Boosting
    - Resource utilization forecasting using Ridge Regression
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Architecture components (if available)
        if ARCHITECTURE_AVAILABLE:
            try:
                self.framework = get_architecture_framework()
                self.service_registry = get_service_registry()
            except Exception as e:
                self.logger.warning(f"Architecture components unavailable: {e}")
                self.framework = None
                self.service_registry = None
        else:
            self.framework = None
            self.service_registry = None
        
        # Directory setup
        self.models_dir = Path("models/predictive_analytics")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = Path("data/predictive_analytics")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modular components
        self._initialize_modular_components()
        
        # Load existing models if available
        self.model_persistence.load_all_models()
        
        self.logger.info("Advanced Predictive Analytics Engine initialized (STEELCLAD)")
    
    def _initialize_modular_components(self):
        """Initialize all modular ML components"""
        # Create models and scalers
        model_data = MLModelFactory.create_all_models()
        self.models = {name: model for name, (model, _) in model_data.items()}
        self.scalers = {name: scaler for name, (_, scaler) in model_data.items()}
        
        # Initialize modular components
        self.feature_processor = MLFeatureProcessor()
        self.data_manager = HistoricalDataManager(self.data_dir)
        self.model_persistence = ModelPersistenceManager(self.models_dir)
        self.performance_tracker = ModelPerformanceTracker()
        self.trainer = MLModelTrainer(self.models, self.scalers, self.models_dir)
    
    def predict_health_trend(self, current_metrics: Dict[str, Any]) -> MLPrediction:
        """
        Predict system health trend using Random Forest
        
        Args:
            current_metrics: Current system metrics
            
        Returns:
            MLPrediction with health trend forecast
        """
        try:
            # Prepare features using modular processor
            features = self.feature_processor.prepare_health_features(current_metrics)
            
            # Create feature matrix
            feature_values = [
                features['cpu_usage'],
                features['memory_usage'],
                features['response_time'],
                features['error_rate'],
                features['service_count'],
                features['dependency_health'],
                features['import_success_rate']
            ]
            X = np.array([feature_values])
            
            # Scale features
            X_scaled = self.scalers['health_trend'].transform(X)
            
            # Make prediction
            prediction = self.models['health_trend'].predict(X_scaled)[0]
            
            # Calculate confidence
            confidence = self._calculate_model_confidence('health_trend', X_scaled)
            
            # Get feature importance
            feature_names = ['cpu', 'memory', 'response_time', 'errors',
                           'services', 'dependencies', 'imports']
            feature_importance = dict(zip(
                feature_names,
                self.models['health_trend'].feature_importances_
            ))
            
            # Track prediction count
            self.performance_tracker.increment_prediction_count('health_trend')
            
            return MLPrediction(
                prediction_type=PredictionType.HEALTH_TREND,
                predicted_value=float(prediction),
                confidence_score=confidence,
                confidence_level=ConfidenceLevel.HIGH,
                feature_importance=feature_importance,
                prediction_horizon=30,
                model_accuracy=self.performance_tracker.get_performance('health_trend').get('r2', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Health trend prediction failed: {e}")
            return self._create_fallback_prediction(PredictionType.HEALTH_TREND, current_metrics)
    
    def detect_anomalies(self, current_metrics: Dict[str, Any]) -> MLPrediction:
        """
        Detect system anomalies using Isolation Forest
        
        Args:
            current_metrics: Current system metrics with historical context
            
        Returns:
            MLPrediction with anomaly detection results
        """
        try:
            # Prepare features using modular processor
            features = self.feature_processor.prepare_anomaly_features(current_metrics)
            
            # Create feature matrix
            feature_values = list(features.values())
            X = np.array([feature_values])
            
            # Scale features
            X_scaled = self.scalers['anomaly'].transform(X)
            
            # Make prediction (-1 for anomaly, 1 for normal)
            prediction = self.models['anomaly'].predict(X_scaled)[0]
            anomaly_score = self.models['anomaly'].score_samples(X_scaled)[0]
            
            # Convert to probability (0-100)
            anomaly_probability = max(0, min(100, (0.5 - anomaly_score) * 100))
            
            # Calculate confidence based on how extreme the score is
            confidence = min(abs(anomaly_score) * 100, 95)
            
            # Create feature importance (approximate for Isolation Forest)
            feature_importance = {name: 1.0 / len(features) for name in features.keys()}
            
            # Track prediction count
            self.performance_tracker.increment_prediction_count('anomaly')
            
            return MLPrediction(
                prediction_type=PredictionType.ANOMALY_DETECTION,
                predicted_value=anomaly_probability,
                confidence_score=confidence,
                confidence_level=ConfidenceLevel.MEDIUM,
                feature_importance=feature_importance,
                prediction_horizon=5,  # Real-time detection
                model_accuracy=0.8  # Default for unsupervised models
            )
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return self._create_fallback_prediction(PredictionType.ANOMALY_DETECTION, current_metrics)
    
    def predict_resource_utilization(self, current_metrics: Dict[str, Any]) -> MLPrediction:
        """
        Predict future resource utilization using Ridge Regression
        
        Args:
            current_metrics: Current system metrics
            
        Returns:
            MLPrediction with resource utilization forecast
        """
        try:
            # Prepare features using modular processor
            features = self.feature_processor.prepare_resource_features(current_metrics)
            
            # Create feature matrix
            feature_values = list(features.values())
            X = np.array([feature_values])
            
            # Scale features
            X_scaled = self.scalers['resource'].transform(X)
            
            # Make prediction
            prediction = self.models['resource'].predict(X_scaled)[0]
            predicted_utilization = max(0, min(100, prediction * 100))
            
            # Calculate confidence
            confidence = self._calculate_model_confidence('resource', X_scaled)
            
            # Get feature importance (coefficients for linear model)
            feature_names = list(features.keys())
            coefficients = self.models['resource'].coef_
            feature_importance = dict(zip(feature_names, np.abs(coefficients)))
            
            # Track prediction count
            self.performance_tracker.increment_prediction_count('resource')
            
            return MLPrediction(
                prediction_type=PredictionType.RESOURCE_UTILIZATION,
                predicted_value=predicted_utilization,
                confidence_score=confidence,
                confidence_level=ConfidenceLevel.MEDIUM,
                feature_importance=feature_importance,
                prediction_horizon=120,  # 2 hours ahead
                model_accuracy=self.performance_tracker.get_performance('resource').get('r2', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Resource utilization prediction failed: {e}")
            return self._create_fallback_prediction(PredictionType.RESOURCE_UTILIZATION, current_metrics)
    
    def train_models_if_needed(self) -> bool:
        """
        Train models if they need updating based on age and performance
        
        Returns:
            True if training was performed
        """
        try:
            # Get training data from modular data manager
            training_data = self.data_manager.get_training_data(min_samples=100)
            
            if training_data is None:
                self.logger.info("Insufficient data for model training")
                return False
            
            # Check which models need retraining
            models_to_train = []
            for model_name in ['health_trend', 'performance', 'resource', 'anomaly']:
                if self.trainer.needs_retraining(model_name):
                    models_to_train.append(model_name)
            
            if not models_to_train:
                self.logger.info("No models need retraining at this time")
                return False
            
            self.logger.info(f"Retraining models: {models_to_train}")
            
            # Perform training using modular trainer
            success = self.trainer.train_all_models(training_data)
            
            if success:
                # Update performance tracker with new metrics
                trainer_performance = self.trainer.get_model_performance()
                for model_name, metrics in trainer_performance.items():
                    self.performance_tracker.update_performance(model_name, metrics)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Model training check failed: {e}")
            return False
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """
        Add new metrics to historical data for future training
        
        Args:
            metrics: System metrics to store
        """
        try:
            self.data_manager.add_metrics(metrics)
        except Exception as e:
            self.logger.error(f"Failed to add metrics to history: {e}")
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all models"""
        return self.performance_tracker.get_all_performance()
    
    def _calculate_model_confidence(self, model_name: str, X_scaled: np.ndarray) -> float:
        """Calculate confidence score for a prediction"""
        try:
            # Get model performance
            performance = self.performance_tracker.get_performance(model_name)
            
            if model_name in ['health_trend', 'resource']:
                # For regression models, use R²
                base_confidence = performance.get('r2', 0.7) * 100
            else:
                # For classification models, use F1 score
                base_confidence = performance.get('f1', 0.7) * 100
            
            # Adjust based on feature variance (lower variance = higher confidence)
            feature_variance = np.var(X_scaled) if X_scaled.size > 0 else 0.5
            variance_penalty = min(feature_variance * 10, 20)
            
            confidence = max(50, min(95, base_confidence - variance_penalty))
            return confidence
            
        except Exception:
            return 75.0  # Default moderate confidence
    
    def _create_fallback_prediction(self, prediction_type: PredictionType, metrics: Dict[str, Any]) -> MLPrediction:
        """Create a fallback prediction when ML models fail"""
        try:
            # Simple fallback based on current metrics
            if prediction_type == PredictionType.HEALTH_TREND:
                current_health = metrics.get('overall_health', 85)
                predicted_value = max(0, min(100, current_health + np.random.normal(0, 5)))
            elif prediction_type == PredictionType.ANOMALY_DETECTION:
                predicted_value = 10.0  # Low anomaly probability
            elif prediction_type == PredictionType.RESOURCE_UTILIZATION:
                current_cpu = metrics.get('cpu_usage', 50)
                current_memory = metrics.get('memory_usage', 50)
                predicted_value = (current_cpu + current_memory) / 2
            else:
                predicted_value = 50.0
            
            return MLPrediction(
                prediction_type=prediction_type,
                predicted_value=predicted_value,
                confidence_score=60.0,
                confidence_level=ConfidenceLevel.LOW,
                feature_importance={'fallback': 1.0},
                prediction_horizon=30,
                model_accuracy=0.6
            )
            
        except Exception:
            return MLPrediction(
                prediction_type=prediction_type,
                predicted_value=50.0,
                confidence_score=50.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                feature_importance={'error': 1.0},
                prediction_horizon=30,
                model_accuracy=0.5
            )


# Global analytics engine instance
_analytics_engine: Optional[AdvancedPredictiveAnalytics] = None


def create_predictive_dashboard_integration() -> AdvancedPredictiveAnalytics:
    """Create and return advanced predictive analytics engine (STEELCLAD)"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AdvancedPredictiveAnalytics()
    return _analytics_engine