#!/usr/bin/env python3
"""
ML Training Module
==================

Training functionality extracted from advanced_predictive_analytics.py
for STEELCLAD modularization (Agent Y STEELCLAD Protocol)

Handles:
- Model training and retraining
- Performance evaluation and tracking
- Model persistence and loading
- Automated training pipelines
"""

import logging
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score


class MLModelTrainer:
    """Handles training of machine learning models"""
    
    def __init__(self, models: Dict, scalers: Dict, models_dir: Path):
        self.models = models
        self.scalers = scalers
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.model_performance = {
            'health_trend': {'mae': 0, 'rmse': 0, 'r2': 0, 'last_trained': None},
            'performance': {'precision': 0, 'recall': 0, 'f1': 0, 'last_trained': None},
            'resource': {'mae': 0, 'rmse': 0, 'r2': 0, 'last_trained': None},
            'anomaly': {'precision': 0, 'recall': 0, 'f1': 0, 'last_trained': None}
        }
    
    def train_all_models(self, training_data: pd.DataFrame) -> bool:
        """
        Train all models with provided data
        
        Args:
            training_data: DataFrame containing historical metrics
            
        Returns:
            True if training succeeded, False otherwise
        """
        try:
            if len(training_data) < 100:
                self.logger.warning("Insufficient data for training (need at least 100 samples)")
                return False
            
            self.logger.info(f"Training models with {len(training_data)} samples")
            
            # Train each model
            success_count = 0
            
            if self.train_health_model(training_data):
                success_count += 1
            
            if self.train_performance_model(training_data):
                success_count += 1
            
            if self.train_resource_model(training_data):
                success_count += 1
            
            if self.train_anomaly_model(training_data):
                success_count += 1
            
            # Save trained models
            self.save_all_models()
            
            self.logger.info(f"Model training complete - {success_count}/4 models trained successfully")
            return success_count >= 2  # At least half the models should succeed
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False
    
    def train_health_model(self, data: pd.DataFrame) -> bool:
        """
        Train health trend prediction model
        
        Args:
            data: Training data with health metrics
            
        Returns:
            True if training succeeded
        """
        try:
            # Define feature columns
            feature_cols = ['cpu_usage', 'memory_usage', 'response_time',
                          'error_rate', 'service_count', 'dependency_health',
                          'import_success_rate']
            
            # Create target variable (future health)
            data_copy = data.copy()
            if 'future_health' not in data_copy.columns:
                data_copy['future_health'] = data_copy['overall_health'].shift(-6)  # 30 min ahead
            
            # Clean data
            clean_data = data_copy[feature_cols + ['future_health']].dropna()
            
            if len(clean_data) < 50:
                self.logger.warning("Insufficient clean data for health model training")
                return False
            
            X = clean_data[feature_cols].values
            y = clean_data['future_health'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers['health_trend'].fit_transform(X_train)
            X_test_scaled = self.scalers['health_trend'].transform(X_test)
            
            # Train model
            self.models['health_trend'].fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.models['health_trend'].predict(X_test_scaled)
            
            self.model_performance['health_trend'] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'last_trained': datetime.now(),
                'samples_trained': len(X_train)
            }
            
            r2_score_val = self.model_performance['health_trend']['r2']
            self.logger.info(f"Health model trained - R2: {r2_score_val:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health model training failed: {e}")
            return False
    
    def train_performance_model(self, data: pd.DataFrame) -> bool:
        """
        Train performance degradation prediction model
        
        Args:
            data: Training data with performance metrics
            
        Returns:
            True if training succeeded
        """
        try:
            feature_cols = ['response_time_trend', 'throughput_change', 'error_rate_trend',
                          'cpu_utilization_trend', 'memory_pressure', 'queue_depth',
                          'cache_hit_rate']
            
            # Create binary target for performance degradation
            data_copy = data.copy()
            if 'performance_degraded' not in data_copy.columns:
                # Define degradation as significant increase in response time or error rate
                data_copy['performance_degraded'] = (
                    (data_copy.get('response_time_trend', 0) > 50) |
                    (data_copy.get('error_rate_trend', 0) > 5)
                ).astype(int)
            
            # Clean data
            clean_data = data_copy[feature_cols + ['performance_degraded']].dropna()
            
            if len(clean_data) < 50:
                self.logger.warning("Insufficient clean data for performance model training")
                return False
            
            X = clean_data[feature_cols].values
            y = clean_data['performance_degraded'].values
            
            # Check if we have both classes
            if len(np.unique(y)) < 2:
                self.logger.warning("Need both degraded and non-degraded samples for performance model")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scalers['performance'].fit_transform(X_train)
            X_test_scaled = self.scalers['performance'].transform(X_test)
            
            # Train model
            self.models['performance'].fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.models['performance'].predict(X_test_scaled)
            
            self.model_performance['performance'] = {
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'last_trained': datetime.now(),
                'samples_trained': len(X_train)
            }
            
            f1_val = self.model_performance['performance']['f1']
            self.logger.info(f"Performance model trained - F1: {f1_val:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Performance model training failed: {e}")
            return False
    
    def train_resource_model(self, data: pd.DataFrame) -> bool:
        """
        Train resource utilization prediction model
        
        Args:
            data: Training data with resource metrics
            
        Returns:
            True if training succeeded
        """
        try:
            feature_cols = ['current_cpu', 'current_memory', 'current_disk',
                          'request_rate', 'user_sessions', 'cache_usage']
            
            # Create target variable (future resource utilization)
            data_copy = data.copy()
            if 'future_utilization' not in data_copy.columns:
                # Calculate composite utilization score
                data_copy['utilization_score'] = (
                    data_copy.get('current_cpu', 50) * 0.4 +
                    data_copy.get('current_memory', 50) * 0.4 +
                    data_copy.get('current_disk', 30) * 0.2
                ) / 100
                data_copy['future_utilization'] = data_copy['utilization_score'].shift(-3)  # 15 min ahead
            
            # Clean data
            clean_data = data_copy[feature_cols + ['future_utilization']].dropna()
            
            if len(clean_data) < 50:
                self.logger.warning("Insufficient clean data for resource model training")
                return False
            
            X = clean_data[feature_cols].values
            y = clean_data['future_utilization'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers['resource'].fit_transform(X_train)
            X_test_scaled = self.scalers['resource'].transform(X_test)
            
            # Train model
            self.models['resource'].fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.models['resource'].predict(X_test_scaled)
            
            self.model_performance['resource'] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'last_trained': datetime.now(),
                'samples_trained': len(X_train)
            }
            
            r2_val = self.model_performance['resource']['r2']
            self.logger.info(f"Resource model trained - R2: {r2_val:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource model training failed: {e}")
            return False
    
    def train_anomaly_model(self, data: pd.DataFrame) -> bool:
        """
        Train anomaly detection model
        
        Args:
            data: Training data for anomaly detection
            
        Returns:
            True if training succeeded
        """
        try:
            feature_cols = ['cpu_variance', 'memory_variance', 'response_time_spike',
                          'error_rate_change', 'service_failures', 'dependency_changes']
            
            # Prepare data for unsupervised anomaly detection
            clean_data = data[feature_cols].dropna()
            
            if len(clean_data) < 50:
                self.logger.warning("Insufficient clean data for anomaly model training")
                return False
            
            X = clean_data.values
            
            # Scale features
            X_scaled = self.scalers['anomaly'].fit_transform(X)
            
            # Train model (unsupervised)
            self.models['anomaly'].fit(X_scaled)
            
            # Evaluate using contamination rate
            predictions = self.models['anomaly'].predict(X_scaled)
            anomaly_rate = len(predictions[predictions == -1]) / len(predictions)
            
            self.model_performance['anomaly'] = {
                'contamination_rate': anomaly_rate,
                'samples_trained': len(X),
                'last_trained': datetime.now()
            }
            
            self.logger.info(f"Anomaly model trained - Contamination rate: {anomaly_rate:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Anomaly model training failed: {e}")
            return False
    
    def save_all_models(self) -> bool:
        """
        Save all trained models to disk
        
        Returns:
            True if all models saved successfully
        """
        try:
            success_count = 0
            
            for model_name in ['health_trend', 'performance', 'resource', 'anomaly']:
                if self._save_model(model_name):
                    success_count += 1
            
            self.logger.info(f"Saved {success_count}/4 models successfully")
            return success_count == 4
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            return False
    
    def _save_model(self, model_name: str) -> bool:
        """Save a single model and its scaler"""
        try:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[model_name], f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save {model_name} model: {e}")
            return False
    
    def load_all_models(self) -> bool:
        """
        Load all trained models from disk
        
        Returns:
            True if models loaded successfully
        """
        try:
            success_count = 0
            
            for model_name in ['health_trend', 'performance', 'resource', 'anomaly']:
                if self._load_model(model_name):
                    success_count += 1
            
            self.logger.info(f"Loaded {success_count}/4 models successfully")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def _load_model(self, model_name: str) -> bool:
        """Load a single model and its scaler"""
        try:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                return False
            
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scalers[model_name] = pickle.load(f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load {model_name} model: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all models"""
        return self.model_performance.copy()
    
    def needs_retraining(self, model_name: str, max_age_hours: int = 24) -> bool:
        """
        Check if a model needs retraining
        
        Args:
            model_name: Name of the model to check
            max_age_hours: Maximum age before retraining is recommended
            
        Returns:
            True if retraining is recommended
        """
        if model_name not in self.model_performance:
            return True
        
        perf = self.model_performance[model_name]
        last_trained = perf.get('last_trained')
        
        if not last_trained:
            return True
        
        # Check age
        age_hours = (datetime.now() - last_trained).total_seconds() / 3600
        if age_hours > max_age_hours:
            return True
        
        # Check performance thresholds
        if model_name in ['health_trend', 'resource']:
            return perf.get('r2', 0) < 0.7
        else:
            return perf.get('f1', 0) < 0.7