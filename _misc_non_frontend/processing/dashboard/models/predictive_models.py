#!/usr/bin/env python3
"""
Predictive ML Models Module
============================

Machine learning models extracted from advanced_predictive_analytics.py
for STEELCLAD modularization (Agent Y STEELCLAD Protocol)

Contains scikit-learn based models for:
- Health trend prediction (Random Forest Regressor)
- Performance degradation (Gradient Boosting Classifier) 
- Resource utilization (Ridge Regression)
- Anomaly detection (Isolation Forest)
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
from typing import Dict, Any, Optional, Tuple


class MLModelFactory:
    """Factory class for creating and managing ML models"""
    
    @staticmethod
    def create_health_trend_model() -> Tuple[RandomForestRegressor, StandardScaler]:
        """
        Create Random Forest model for health trend prediction
        
        Returns:
            Tuple of (model, scaler) for health trend prediction
        """
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        scaler = StandardScaler()
        return model, scaler
    
    @staticmethod
    def create_performance_model() -> Tuple[GradientBoostingClassifier, StandardScaler]:
        """
        Create Gradient Boosting model for performance degradation prediction
        
        Returns:
            Tuple of (model, scaler) for performance prediction
        """
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        scaler = StandardScaler()
        return model, scaler
    
    @staticmethod
    def create_resource_model() -> Tuple[Ridge, StandardScaler]:
        """
        Create Ridge Regression model for resource utilization prediction
        
        Returns:
            Tuple of (model, scaler) for resource prediction
        """
        model = Ridge(
            alpha=1.0,
            solver='auto',
            random_state=42
        )
        scaler = StandardScaler()
        return model, scaler
    
    @staticmethod
    def create_anomaly_model() -> Tuple[IsolationForest, StandardScaler]:
        """
        Create Isolation Forest model for anomaly detection
        
        Returns:
            Tuple of (model, scaler) for anomaly detection
        """
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            max_samples='auto',
            random_state=42
        )
        scaler = StandardScaler()
        return model, scaler
    
    @staticmethod
    def create_all_models() -> Dict[str, Tuple[Any, StandardScaler]]:
        """
        Create all ML models and their associated scalers
        
        Returns:
            Dictionary mapping model names to (model, scaler) tuples
        """
        return {
            'health_trend': MLModelFactory.create_health_trend_model(),
            'performance': MLModelFactory.create_performance_model(),
            'resource': MLModelFactory.create_resource_model(),
            'anomaly': MLModelFactory.create_anomaly_model()
        }


class ModelPersistenceManager:
    """Handles saving and loading of trained models"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_model(self, model_name: str, model: Any, scaler: StandardScaler) -> bool:
        """
        Save a trained model and scaler to disk
        
        Args:
            model_name: Name identifier for the model
            model: Trained scikit-learn model
            scaler: Fitted StandardScaler
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            self.logger.info(f"Saved model and scaler: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str) -> Optional[Tuple[Any, StandardScaler]]:
        """
        Load a trained model and scaler from disk
        
        Args:
            model_name: Name identifier for the model
            
        Returns:
            Tuple of (model, scaler) if successful, None otherwise
        """
        try:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                return None
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            self.logger.info(f"Loaded model and scaler: {model_name}")
            return model, scaler
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def load_all_models(self) -> Dict[str, Tuple[Any, StandardScaler]]:
        """
        Load all available trained models
        
        Returns:
            Dictionary mapping model names to (model, scaler) tuples
        """
        models = {}
        model_names = ['health_trend', 'performance', 'resource', 'anomaly']
        
        for name in model_names:
            loaded = self.load_model(name)
            if loaded:
                models[name] = loaded
            else:
                # Create fresh models if loading fails
                self.logger.warning(f"Creating fresh model for {name}")
                if name == 'health_trend':
                    models[name] = MLModelFactory.create_health_trend_model()
                elif name == 'performance':
                    models[name] = MLModelFactory.create_performance_model()
                elif name == 'resource':
                    models[name] = MLModelFactory.create_resource_model()
                elif name == 'anomaly':
                    models[name] = MLModelFactory.create_anomaly_model()
        
        return models


class ModelPerformanceTracker:
    """Tracks and manages model performance metrics"""
    
    def __init__(self):
        self.performance_history: Dict[str, Dict[str, Any]] = {
            'health_trend': {'mae': 0, 'rmse': 0, 'r2': 0, 'last_trained': None, 'predictions': 0},
            'performance': {'precision': 0, 'recall': 0, 'f1': 0, 'last_trained': None, 'predictions': 0},
            'resource': {'mae': 0, 'rmse': 0, 'r2': 0, 'last_trained': None, 'predictions': 0},
            'anomaly': {'precision': 0, 'recall': 0, 'f1': 0, 'last_trained': None, 'predictions': 0}
        }
        self.logger = logging.getLogger(__name__)
    
    def update_performance(self, model_name: str, metrics: Dict[str, float]):
        """
        Update performance metrics for a model
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
        """
        if model_name in self.performance_history:
            self.performance_history[model_name].update(metrics)
            self.performance_history[model_name]['last_updated'] = str(np.datetime64('now'))
            self.logger.info(f"Updated performance for {model_name}: {metrics}")
    
    def get_performance(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_history.get(model_name, {})
    
    def get_all_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all models
        
        Returns:
            Dictionary mapping model names to their performance metrics
        """
        return self.performance_history.copy()
    
    def increment_prediction_count(self, model_name: str):
        """
        Increment the prediction count for a model
        
        Args:
            model_name: Name of the model
        """
        if model_name in self.performance_history:
            self.performance_history[model_name]['predictions'] += 1
    
    def needs_retraining(self, model_name: str, threshold_predictions: int = 1000) -> bool:
        """
        Check if a model needs retraining based on prediction count or performance degradation
        
        Args:
            model_name: Name of the model
            threshold_predictions: Number of predictions before suggesting retraining
            
        Returns:
            True if retraining is recommended
        """
        if model_name not in self.performance_history:
            return False
        
        metrics = self.performance_history[model_name]
        
        # Check prediction count
        if metrics.get('predictions', 0) > threshold_predictions:
            return True
        
        # Check performance thresholds
        if model_name in ['health_trend', 'resource']:
            # For regression models, check R²
            if metrics.get('r2', 0) < 0.7:  # R² below 70%
                return True
        else:
            # For classification models, check F1 score
            if metrics.get('f1', 0) < 0.7:  # F1 below 70%
                return True
        
        return False