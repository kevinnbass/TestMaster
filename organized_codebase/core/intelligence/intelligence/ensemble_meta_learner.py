"""
Advanced Ensemble Meta-Learning System
======================================
Extracted and enhanced from archive analytics_components.
Module size: ~299 lines (under 300 limit)

Author: Agent B - Intelligence Specialist
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class ModelPerformance:
    """Enhanced model performance tracking."""
    model_id: str
    accuracy: float
    mse: float
    r2_score: float
    training_time: float
    prediction_latency: float
    memory_usage: float
    stability_score: float
    last_updated: datetime


@dataclass
class EnsembleConfig:
    """Configuration for ensemble learning."""
    voting_strategy: str = "weighted"
    meta_learner_type: str = "ridge"
    retraining_threshold: float = 0.05
    performance_decay_factor: float = 0.95
    min_models: int = 3
    max_models: int = 10


class AdaptiveModelSelector:
    """
    Intelligent model selection based on data characteristics.
    Extracted from analytics_performance_optimizer.py patterns.
    """
    
    def __init__(self):
        self.model_registry = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf_small': RandomForestRegressor(n_estimators=50, max_depth=5),
            'rf_large': RandomForestRegressor(n_estimators=100, max_depth=10),
            'gb_fast': GradientBoostingRegressor(n_estimators=50, learning_rate=0.1),
            'gb_accurate': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
        }
        self.selection_history = deque(maxlen=1000)
        
    def select_best_models(self, X: np.ndarray, y: np.ndarray, 
                          n_models: int = 5) -> List[str]:
        """Select best models based on data characteristics."""
        n_samples, n_features = X.shape
        
        # Data characteristics
        feature_variance = np.var(X, axis=0).mean()
        target_variance = np.var(y)
        sparsity = np.mean(X == 0)
        
        selected_models = []
        
        # Rule-based selection
        if n_samples < 100:
            selected_models.extend(['linear', 'ridge'])
        elif n_samples < 1000:
            selected_models.extend(['ridge', 'rf_small', 'gb_fast'])
        else:
            selected_models.extend(['rf_large', 'gb_accurate'])
            
        if n_features > n_samples:
            selected_models.append('ridge')
            
        if feature_variance > 1.0:
            selected_models.append('rf_large')
            
        if sparsity > 0.3:
            selected_models.append('linear')
            
        # Remove duplicates and limit
        return list(set(selected_models))[:n_models]


class MetaLearner:
    """
    Meta-learning system for ensemble combination.
    Based on predictive_analytics_engine.py patterns.
    """
    
    def __init__(self, meta_learner_type: str = "ridge"):
        self.meta_learner_type = meta_learner_type
        self.meta_model = None
        self.feature_importance = {}
        self.combination_weights = {}
        
    def train_meta_model(self, base_predictions: np.ndarray, 
                        true_values: np.ndarray) -> Dict[str, float]:
        """Train meta-learner on base model predictions."""
        if self.meta_learner_type == "ridge":
            self.meta_model = Ridge(alpha=0.1)
        elif self.meta_learner_type == "linear":
            self.meta_model = LinearRegression()
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")
            
        # Train meta-model
        self.meta_model.fit(base_predictions, true_values)
        
        # Extract combination weights
        self.combination_weights = {
            f'model_{i}': weight 
            for i, weight in enumerate(self.meta_model.coef_)
        }
        
        # Calculate performance
        meta_predictions = self.meta_model.predict(base_predictions)
        mse = mean_squared_error(true_values, meta_predictions)
        r2 = r2_score(true_values, meta_predictions)
        
        return {"mse": mse, "r2": r2, "weights": self.combination_weights}
        
    def predict_ensemble(self, base_predictions: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions using meta-learner."""
        if self.meta_model is None:
            # Fallback to simple averaging
            return np.mean(base_predictions, axis=1)
        return self.meta_model.predict(base_predictions)


class EnsembleMetaLearner:
    """
    Advanced ensemble meta-learning system.
    Combines multiple algorithms with intelligent weighting.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.model_selector = AdaptiveModelSelector()
        self.meta_learner = MetaLearner(self.config.meta_learner_type)
        
        # Ensemble state
        self.base_models = {}
        self.model_performance = {}
        self.ensemble_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_trends = defaultdict(deque)
        self.retraining_triggers = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ensemble with intelligent model selection."""
        start_time = datetime.now()
        
        # Select best models for this data
        selected_models = self.model_selector.select_best_models(
            X, y, self.config.max_models
        )
        
        # Train base models
        base_predictions = []
        valid_models = []
        
        for model_name in selected_models:
            try:
                model = self.model_selector.model_registry[model_name]
                model_start = datetime.now()
                
                # Train model
                model.fit(X, y)
                
                # Validate performance
                predictions = model.predict(X)
                mse = mean_squared_error(y, predictions)
                r2 = r2_score(y, predictions)
                
                training_time = (datetime.now() - model_start).total_seconds()
                
                # Store if performance is acceptable
                if r2 > 0.1:  # Minimum performance threshold
                    self.base_models[model_name] = model
                    base_predictions.append(predictions)
                    valid_models.append(model_name)
                    
                    # Track performance
                    self.model_performance[model_name] = ModelPerformance(
                        model_id=model_name,
                        accuracy=r2,
                        mse=mse,
                        r2_score=r2,
                        training_time=training_time,
                        prediction_latency=0.0,  # Updated during prediction
                        memory_usage=0.0,
                        stability_score=1.0,
                        last_updated=datetime.now()
                    )
                    
            except Exception as e:
                warnings.warn(f"Failed to train {model_name}: {e}")
                continue
                
        if len(valid_models) < self.config.min_models:
            raise RuntimeError(f"Insufficient valid models: {len(valid_models)}")
            
        # Train meta-learner
        base_predictions_array = np.column_stack(base_predictions)
        meta_results = self.meta_learner.train_meta_model(base_predictions_array, y)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Update ensemble history
        ensemble_result = {
            'timestamp': datetime.now(),
            'models_used': valid_models,
            'meta_performance': meta_results,
            'training_time': total_time,
            'data_size': len(X)
        }
        self.ensemble_history.append(ensemble_result)
        
        return ensemble_result
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.base_models:
            raise RuntimeError("Ensemble not trained")
            
        start_time = datetime.now()
        
        # Get predictions from all base models
        base_predictions = []
        for model_name, model in self.base_models.items():
            try:
                pred_start = datetime.now()
                predictions = model.predict(X)
                pred_time = (datetime.now() - pred_start).total_seconds()
                
                # Update prediction latency
                if model_name in self.model_performance:
                    self.model_performance[model_name].prediction_latency = pred_time
                    
                base_predictions.append(predictions)
            except Exception as e:
                warnings.warn(f"Model {model_name} prediction failed: {e}")
                continue
                
        if not base_predictions:
            raise RuntimeError("No valid predictions from base models")
            
        # Combine predictions using meta-learner
        base_predictions_array = np.column_stack(base_predictions)
        ensemble_predictions = self.meta_learner.predict_ensemble(base_predictions_array)
        
        return ensemble_predictions
        
    def update_performance(self, X: np.ndarray, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict[str, float]:
        """Update ensemble performance metrics."""
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Track performance trends
        self.performance_trends['mse'].append(mse)
        self.performance_trends['r2'].append(r2)
        
        # Check for retraining triggers
        if len(self.performance_trends['r2']) > 10:
            recent_r2 = np.mean(list(self.performance_trends['r2'])[-5:])
            older_r2 = np.mean(list(self.performance_trends['r2'])[-10:-5])
            
            if older_r2 - recent_r2 > self.config.retraining_threshold:
                self.retraining_triggers.append({
                    'timestamp': datetime.now(),
                    'trigger': 'performance_decline',
                    'old_r2': older_r2,
                    'new_r2': recent_r2
                })
                
        return {"mse": mse, "r2": r2, "needs_retraining": len(self.retraining_triggers) > 0}
        
    def get_model_importance(self) -> Dict[str, float]:
        """Get model importance scores."""
        importance = {}
        
        for model_name, performance in self.model_performance.items():
            # Combine multiple performance factors
            importance[model_name] = (
                performance.accuracy * 0.4 +
                (1.0 / (1.0 + performance.mse)) * 0.3 +
                performance.stability_score * 0.2 +
                (1.0 / (1.0 + performance.prediction_latency)) * 0.1
            )
            
        return importance
        
    def optimize_ensemble(self) -> Dict[str, Any]:
        """Optimize ensemble performance."""
        if not self.model_performance:
            return {"status": "no_models"}
            
        # Remove underperforming models
        importance_scores = self.get_model_importance()
        avg_importance = np.mean(list(importance_scores.values()))
        
        models_to_remove = [
            name for name, score in importance_scores.items()
            if score < avg_importance * 0.5
        ]
        
        for model_name in models_to_remove:
            if len(self.base_models) > self.config.min_models:
                del self.base_models[model_name]
                del self.model_performance[model_name]
                
        return {
            "models_removed": models_to_remove,
            "remaining_models": list(self.base_models.keys()),
            "avg_importance": avg_importance
        }


# Public API
__all__ = [
    'EnsembleMetaLearner',
    'AdaptiveModelSelector',
    'MetaLearner',
    'EnsembleConfig',
    'ModelPerformance'
]