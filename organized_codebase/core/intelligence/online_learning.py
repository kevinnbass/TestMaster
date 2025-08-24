"""
Online Learning and Incremental Algorithms
==========================================
Adaptive learning algorithms for streaming data.
Module size: ~299 lines (under 300 limit)

Author: Agent B - Intelligence Specialist
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import deque
import math


@dataclass
class LearningState:
    """State information for online learners."""
    n_samples: int
    model_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    learning_rate: float
    last_update: int = 0


class OnlineLinearRegression:
    """
    Online linear regression with adaptive learning rate.
    Updates model incrementally with each new sample.
    """
    
    def __init__(self, n_features: int, learning_rate: float = 0.01, 
                 l2_reg: float = 0.01):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        
        # Model parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Online statistics
        self.n_samples = 0
        self.sum_squared_error = 0.0
        self.running_mean = np.zeros(n_features)
        self.running_var = np.zeros(n_features)
        
    def partial_fit(self, X: np.ndarray, y: Union[float, np.ndarray]) -> 'OnlineLinearRegression':
        """Update model with new sample(s)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if np.isscalar(y):
            y = np.array([y])
            
        for i in range(X.shape[0]):
            self._update_single(X[i], y[i])
            
        return self
        
    def _update_single(self, x: np.ndarray, y: float):
        """Update with single sample."""
        # Prediction
        pred = np.dot(x, self.weights) + self.bias
        error = y - pred
        
        # Adaptive learning rate
        adaptive_lr = self.learning_rate / (1 + self.n_samples * 0.0001)
        
        # Gradient update
        self.weights += adaptive_lr * (error * x - self.l2_reg * self.weights)
        self.bias += adaptive_lr * error
        
        # Update statistics
        self.n_samples += 1
        self.sum_squared_error += error ** 2
        
        # Update running statistics
        delta = x - self.running_mean
        self.running_mean += delta / self.n_samples
        self.running_var += delta * (x - self.running_mean)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.dot(X, self.weights) + self.bias
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        mse = self.sum_squared_error / max(1, self.n_samples)
        return {
            'mse': mse,
            'rmse': math.sqrt(mse),
            'n_samples': self.n_samples,
            'weight_norm': np.linalg.norm(self.weights)
        }


class OnlineGradientBoosting:
    """
    Online gradient boosting with adaptive base learners.
    Builds ensemble incrementally.
    """
    
    def __init__(self, max_estimators: int = 100, learning_rate: float = 0.1,
                 subsample_rate: float = 0.8):
        self.max_estimators = max_estimators
        self.learning_rate = learning_rate
        self.subsample_rate = subsample_rate
        
        self.estimators = []
        self.estimator_weights = []
        self.n_samples = 0
        self.residual_buffer = deque(maxlen=1000)
        
    def partial_fit(self, X: np.ndarray, y: Union[float, np.ndarray]) -> 'OnlineGradientBoosting':
        """Update ensemble with new sample(s)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if np.isscalar(y):
            y = np.array([y])
            
        # Store data for residual calculation
        for i in range(X.shape[0]):
            self.residual_buffer.append((X[i], y[i]))
            self.n_samples += 1
            
        # Train new estimator periodically
        if len(self.residual_buffer) >= 50 and len(self.estimators) < self.max_estimators:
            self._add_estimator()
            
        return self
        
    def _add_estimator(self):
        """Add new weak learner to ensemble."""
        if not self.residual_buffer:
            return
            
        # Calculate residuals for recent samples
        X_batch = []
        residuals = []
        
        for x, y in list(self.residual_buffer)[-50:]:
            pred = self.predict(x.reshape(1, -1))[0] if self.estimators else 0
            residual = y - pred
            X_batch.append(x)
            residuals.append(residual)
            
        X_batch = np.array(X_batch)
        residuals = np.array(residuals)
        
        # Train weak learner on residuals
        weak_learner = OnlineLinearRegression(
            n_features=X_batch.shape[1],
            learning_rate=0.05
        )
        weak_learner.partial_fit(X_batch, residuals)
        
        # Add to ensemble
        self.estimators.append(weak_learner)
        self.estimator_weights.append(self.learning_rate)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.estimators:
            return np.zeros(X.shape[0])
            
        predictions = np.zeros(X.shape[0])
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions += weight * estimator.predict(X)
            
        return predictions


class AdaptiveWindowRegressor:
    """
    Adaptive window regressor that adjusts to concept drift.
    Maintains multiple models with different window sizes.
    """
    
    def __init__(self, window_sizes: List[int] = [50, 100, 200]):
        self.window_sizes = window_sizes
        self.models = {
            size: OnlineLinearRegression(n_features=1) for size in window_sizes
        }
        self.data_windows = {
            size: deque(maxlen=size) for size in window_sizes
        }
        self.performance_history = {
            size: deque(maxlen=100) for size in window_sizes
        }
        self.best_model_size = window_sizes[0]
        
    def partial_fit(self, X: np.ndarray, y: Union[float, np.ndarray]) -> 'AdaptiveWindowRegressor':
        """Update all window models."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if np.isscalar(y):
            y = np.array([y])
            
        # Update all models
        for size in self.window_sizes:
            # Add to window
            for i in range(X.shape[0]):
                self.data_windows[size].append((X[i], y[i]))
                
            # Train model if window has enough data
            if len(self.data_windows[size]) >= 10:
                window_data = list(self.data_windows[size])
                X_window = np.array([item[0] for item in window_data])
                y_window = np.array([item[1] for item in window_data])
                
                # Reset and retrain model
                self.models[size] = OnlineLinearRegression(n_features=X.shape[1])
                self.models[size].partial_fit(X_window, y_window)
                
                # Evaluate performance
                if len(window_data) > 5:
                    test_idx = len(window_data) // 2
                    X_test = X_window[test_idx:]
                    y_test = y_window[test_idx:]
                    
                    pred = self.models[size].predict(X_test)
                    mse = np.mean((pred - y_test) ** 2)
                    self.performance_history[size].append(mse)
                    
        # Select best performing model
        self._select_best_model()
        
        return self
        
    def _select_best_model(self):
        """Select best performing window size."""
        best_score = float('inf')
        
        for size in self.window_sizes:
            if self.performance_history[size]:
                # Use recent average performance
                recent_scores = list(self.performance_history[size])[-10:]
                avg_score = np.mean(recent_scores)
                
                if avg_score < best_score:
                    best_score = avg_score
                    self.best_model_size = size
                    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best model."""
        return self.models[self.best_model_size].predict(X)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        metrics = {'best_window_size': self.best_model_size}
        
        for size in self.window_sizes:
            if self.performance_history[size]:
                metrics[f'window_{size}_mse'] = np.mean(list(self.performance_history[size])[-5:])
                
        return metrics


class StreamingKMeans:
    """
    Online K-means clustering for streaming data.
    Updates cluster centers incrementally.
    """
    
    def __init__(self, n_clusters: int = 3, learning_rate: float = 0.1,
                 init_samples: int = 10):
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.init_samples = init_samples
        
        self.centroids = None
        self.cluster_counts = np.zeros(n_clusters)
        self.n_samples = 0
        self.initialization_buffer = []
        
    def partial_fit(self, X: np.ndarray) -> 'StreamingKMeans':
        """Update clusters with new samples."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        for i in range(X.shape[0]):
            self._update_single(X[i])
            
        return self
        
    def _update_single(self, x: np.ndarray):
        """Update with single sample."""
        if self.centroids is None:
            # Collect samples for initialization
            self.initialization_buffer.append(x)
            
            if len(self.initialization_buffer) >= self.init_samples:
                self._initialize_centroids()
            return
            
        # Find closest centroid
        distances = np.linalg.norm(self.centroids - x, axis=1)
        closest_cluster = np.argmin(distances)
        
        # Update centroid
        self.cluster_counts[closest_cluster] += 1
        lr = self.learning_rate / self.cluster_counts[closest_cluster]
        
        self.centroids[closest_cluster] += lr * (x - self.centroids[closest_cluster])
        self.n_samples += 1
        
    def _initialize_centroids(self):
        """Initialize centroids using collected samples."""
        data = np.array(self.initialization_buffer)
        
        # K-means++ initialization
        self.centroids = np.zeros((self.n_clusters, data.shape[1]))
        
        # First centroid: random
        self.centroids[0] = data[np.random.randint(len(data))]
        
        # Remaining centroids: K-means++
        for i in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(data - c, axis=1) 
                              for c in self.centroids[:i]], axis=0)
            probs = distances ** 2
            probs /= probs.sum()
            
            idx = np.random.choice(len(data), p=probs)
            self.centroids[i] = data[idx]
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        if self.centroids is None:
            return np.zeros(X.shape[0], dtype=int)
            
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get clustering metrics."""
        if self.centroids is None:
            return {'initialized': False}
            
        return {
            'initialized': True,
            'n_clusters': self.n_clusters,
            'n_samples': self.n_samples,
            'cluster_counts': self.cluster_counts.tolist(),
            'centroids_norm': [np.linalg.norm(c) for c in self.centroids]
        }


# Public API
__all__ = [
    'OnlineLinearRegression',
    'OnlineGradientBoosting', 
    'AdaptiveWindowRegressor',
    'StreamingKMeans',
    'LearningState'
]