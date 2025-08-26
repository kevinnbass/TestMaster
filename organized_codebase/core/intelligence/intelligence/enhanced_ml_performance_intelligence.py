#!/usr/bin/env python3
"""
Enhanced ML Performance Intelligence System
Advanced machine learning models for deeper system insights with predictive performance management.

Agent Beta - Phase 2, Hours 55-60
Greek Swarm Coordination - TestMaster Intelligence System
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import threading
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.cluster import DBSCAN, KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    print("WARNING: scikit-learn not available. ML features will use simplified algorithms.")
    SKLEARN_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNING: scipy not available. Advanced statistical features disabled.")
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnhancedMLPerformanceIntelligence')

class PredictionHorizon(Enum):
    """Prediction time horizons for different use cases"""
    SHORT_TERM = "15_minutes"
    MEDIUM_TERM = "1_hour"
    LONG_TERM = "4_hours"
    EXTENDED_TERM = "12_hours"
    STRATEGIC_TERM = "24_hours"

class MLModelType(Enum):
    """Available ML model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    RIDGE_REGRESSION = "ridge_regression"
    ELASTIC_NET = "elastic_net"
    ENSEMBLE_HYBRID = "ensemble_hybrid"

class AnomalyType(Enum):
    """Types of performance anomalies"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_SPIKE = "resource_spike"
    LATENCY_ANOMALY = "latency_anomaly"
    THROUGHPUT_DROP = "throughput_drop"
    ERROR_RATE_INCREASE = "error_rate_increase"
    CACHE_MISS_SPIKE = "cache_miss_spike"
    DATABASE_SLOWDOWN = "database_slowdown"

@dataclass
class EnhancedMLConfig:
    """Configuration for enhanced ML performance intelligence"""
    # Model Configuration
    model_types: List[MLModelType] = None
    prediction_horizons: List[PredictionHorizon] = None
    ensemble_weight_strategy: str = "performance_based"
    model_update_frequency: int = 3600  # seconds
    
    # Data Configuration
    feature_window_size: int = 100
    min_training_samples: int = 50
    feature_engineering_enabled: bool = True
    correlation_threshold: float = 0.7
    
    # Prediction Configuration
    prediction_confidence_threshold: float = 0.8
    forecast_update_interval: int = 300  # seconds
    predictive_maintenance_threshold: float = 0.75
    
    # Anomaly Detection Configuration
    anomaly_sensitivity: float = 0.1
    isolation_forest_contamination: float = 0.1
    anomaly_window_size: int = 50
    
    # Performance Configuration
    max_concurrent_predictions: int = 10
    prediction_cache_ttl: int = 180  # seconds
    model_performance_threshold: float = 0.7
    
    # Database Configuration
    db_path: str = "enhanced_ml_performance_intelligence.db"
    data_retention_days: int = 30
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = [MLModelType.ENSEMBLE_HYBRID]
        if self.prediction_horizons is None:
            self.prediction_horizons = [
                PredictionHorizon.SHORT_TERM,
                PredictionHorizon.MEDIUM_TERM,
                PredictionHorizon.LONG_TERM
            ]

@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics for ML analysis"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io_read: float
    disk_io_write: float
    network_in: float
    network_out: float
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    throughput: float
    error_rate: float
    cache_hit_ratio: float
    database_query_time: float
    active_connections: int
    queue_length: int
    
    # Advanced metrics
    cpu_load_1min: float = 0.0
    cpu_load_5min: float = 0.0
    memory_swap_used: float = 0.0
    gc_collections: int = 0
    thread_count: int = 0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to feature vector for ML models"""
        return np.array([
            self.cpu_usage,
            self.memory_usage,
            self.disk_io_read,
            self.disk_io_write,
            self.network_in,
            self.network_out,
            self.response_time_avg,
            self.response_time_p95,
            self.response_time_p99,
            self.throughput,
            self.error_rate,
            self.cache_hit_ratio,
            self.database_query_time,
            self.active_connections,
            self.queue_length,
            self.cpu_load_1min,
            self.cpu_load_5min,
            self.memory_swap_used,
            self.gc_collections,
            self.thread_count
        ])

@dataclass
class MLPrediction:
    """ML prediction result with confidence and metadata"""
    horizon: PredictionHorizon
    predicted_value: float
    confidence_score: float
    model_type: MLModelType
    feature_importance: Dict[str, float]
    prediction_timestamp: datetime
    target_metric: str
    
    # Prediction metadata
    model_accuracy: float
    training_samples: int
    feature_count: int
    prediction_variance: float = 0.0

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    confidence: float
    affected_metrics: List[str]
    detected_timestamp: datetime
    
    # Anomaly context
    baseline_value: float
    anomalous_value: float
    detection_method: str
    recommended_action: str = ""

class FeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self, config: EnhancedMLConfig):
        self.config = config
        self.scalers = {}
        
    def engineer_features(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features from raw metrics"""
        if len(metrics_df) < 3:
            return metrics_df
            
        try:
            # Time-based features
            metrics_df = self._add_time_features(metrics_df)
            
            # Statistical features
            metrics_df = self._add_statistical_features(metrics_df)
            
            # Ratio and derived features
            metrics_df = self._add_derived_features(metrics_df)
            
            # Trend features
            metrics_df = self._add_trend_features(metrics_df)
            
            # Lag features
            metrics_df = self._add_lag_features(metrics_df)
            
            return metrics_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return metrics_df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' not in df.columns:
            return df
            
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical rolling features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['hour', 'day_of_week', 'is_weekend', 'is_business_hours']:
                continue
                
            # Rolling statistics
            df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
            df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
            df[f'{col}_rolling_max_5'] = df[col].rolling(window=5, min_periods=1).max()
            df[f'{col}_rolling_min_5'] = df[col].rolling(window=5, min_periods=1).min()
            
            # Z-score features
            if df[col].std() > 0:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived ratio and combination features"""
        try:
            # Resource utilization ratios
            if 'cpu_usage' in df.columns and 'memory_usage' in df.columns:
                df['cpu_memory_ratio'] = df['cpu_usage'] / (df['memory_usage'] + 1e-6)
            
            # Performance efficiency metrics
            if 'throughput' in df.columns and 'cpu_usage' in df.columns:
                df['throughput_per_cpu'] = df['throughput'] / (df['cpu_usage'] + 1e-6)
            
            # Network efficiency
            if 'network_in' in df.columns and 'network_out' in df.columns:
                df['network_total'] = df['network_in'] + df['network_out']
                df['network_ratio'] = df['network_in'] / (df['network_out'] + 1e-6)
            
            # Response time efficiency
            if 'response_time_p95' in df.columns and 'response_time_avg' in df.columns:
                df['response_time_variability'] = df['response_time_p95'] / (df['response_time_avg'] + 1e-6)
            
        except Exception as e:
            logger.error(f"Derived feature creation failed: {e}")
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend and momentum features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in ['cpu_usage', 'memory_usage', 'response_time_avg', 'throughput']:
            if col not in df.columns:
                continue
                
            # Trend features
            df[f'{col}_diff_1'] = df[col].diff()
            df[f'{col}_diff_2'] = df[col].diff(periods=2)
            
            # Momentum features
            if len(df) >= 3:
                df[f'{col}_momentum'] = (df[col] - df[col].shift(2)) / 2
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for time series patterns"""
        for col in ['cpu_usage', 'memory_usage', 'response_time_avg']:
            if col not in df.columns:
                continue
                
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_2'] = df[col].shift(2)
            df[f'{col}_lag_3'] = df[col].shift(3)
        
        return df

class EnhancedMLPredictor:
    """Enhanced ML predictor with multiple models and advanced features"""
    
    def __init__(self, config: EnhancedMLConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer(config)
        self.model_performance = {}
        self.prediction_cache = {}
        self.last_cache_clear = time.time()
        
    def train_models(self, metrics_history: List[PerformanceMetrics], target_metric: str):
        """Train ML models for performance prediction"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Using simplified models.")
            return
        
        try:
            # Prepare training data
            df = self._prepare_training_data(metrics_history)
            if len(df) < self.config.min_training_samples:
                logger.warning(f"Insufficient training data: {len(df)} samples")
                return
            
            # Engineer features
            df_engineered = self.feature_engineer.engineer_features(df)
            
            # Prepare features and target
            X, y = self._prepare_features_target(df_engineered, target_metric)
            if X is None or y is None:
                logger.error("Failed to prepare features and target")
                return
            
            # Train models for each horizon
            for horizon in self.config.prediction_horizons:
                self._train_horizon_models(X, y, horizon, target_metric)
                
            logger.info(f"Successfully trained models for {target_metric}")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def _prepare_training_data(self, metrics_history: List[PerformanceMetrics]) -> pd.DataFrame:
        """Convert metrics history to DataFrame"""
        data = []
        for metrics in metrics_history:
            data.append({
                'timestamp': metrics.timestamp,
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'disk_io_read': metrics.disk_io_read,
                'disk_io_write': metrics.disk_io_write,
                'network_in': metrics.network_in,
                'network_out': metrics.network_out,
                'response_time_avg': metrics.response_time_avg,
                'response_time_p95': metrics.response_time_p95,
                'response_time_p99': metrics.response_time_p99,
                'throughput': metrics.throughput,
                'error_rate': metrics.error_rate,
                'cache_hit_ratio': metrics.cache_hit_ratio,
                'database_query_time': metrics.database_query_time,
                'active_connections': metrics.active_connections,
                'queue_length': metrics.queue_length,
                'cpu_load_1min': metrics.cpu_load_1min,
                'cpu_load_5min': metrics.cpu_load_5min,
                'memory_swap_used': metrics.memory_swap_used,
                'gc_collections': metrics.gc_collections,
                'thread_count': metrics.thread_count
            })
        
        return pd.DataFrame(data).sort_values('timestamp')
    
    def _prepare_features_target(self, df: pd.DataFrame, target_metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        if target_metric not in df.columns:
            logger.error(f"Target metric {target_metric} not found in data")
            return None, None
        
        # Remove non-numeric columns
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_metric in feature_columns:
            feature_columns.remove(target_metric)
        
        # Handle missing values
        df_features = df[feature_columns].fillna(df[feature_columns].median())
        
        X = df_features.values
        y = df[target_metric].values
        
        # Scale features
        scaler_key = f"{target_metric}_scaler"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()
        
        X_scaled = self.scalers[scaler_key].fit_transform(X)
        
        return X_scaled, y
    
    def _train_horizon_models(self, X: np.ndarray, y: np.ndarray, horizon: PredictionHorizon, target_metric: str):
        """Train models for specific prediction horizon"""
        # Create shifted target for different horizons
        horizon_steps = self._get_horizon_steps(horizon)
        y_shifted = np.roll(y, -horizon_steps)
        y_shifted = y_shifted[:-horizon_steps] if horizon_steps > 0 else y_shifted
        X_shifted = X[:-horizon_steps] if horizon_steps > 0 else X
        
        if len(X_shifted) < self.config.min_training_samples:
            logger.warning(f"Insufficient data for horizon {horizon.value}")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_shifted, y_shifted, test_size=0.2, random_state=42
        )
        
        # Train different model types
        horizon_models = {}
        for model_type in self.config.model_types:
            model = self._create_model(model_type)
            if model is None:
                continue
                
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Evaluate model
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_key = f"{target_metric}_{horizon.value}_{model_type.value}"
                horizon_models[model_key] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'feature_count': X_train.shape[1],
                    'training_samples': len(X_train)
                }
                
                self.model_performance[model_key] = r2
                
                logger.info(f"Trained {model_type.value} for {horizon.value}: R² = {r2:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type.value}: {e}")
        
        # Store models
        model_key = f"{target_metric}_{horizon.value}"
        self.models[model_key] = horizon_models
    
    def _create_model(self, model_type: MLModelType):
        """Create ML model instance"""
        try:
            if model_type == MLModelType.RANDOM_FOREST:
                return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            elif model_type == MLModelType.GRADIENT_BOOSTING:
                return GradientBoostingRegressor(n_estimators=50, random_state=42)
            elif model_type == MLModelType.RIDGE_REGRESSION:
                return Ridge(alpha=1.0)
            elif model_type == MLModelType.ELASTIC_NET:
                return ElasticNet(alpha=1.0, random_state=42)
            elif model_type == MLModelType.ENSEMBLE_HYBRID:
                # Return the best performing single model for now
                return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        except Exception as e:
            logger.error(f"Failed to create model {model_type.value}: {e}")
            return None
    
    def _get_horizon_steps(self, horizon: PredictionHorizon) -> int:
        """Get number of steps for prediction horizon"""
        horizon_map = {
            PredictionHorizon.SHORT_TERM: 3,    # 15 minutes
            PredictionHorizon.MEDIUM_TERM: 12,  # 1 hour
            PredictionHorizon.LONG_TERM: 48,    # 4 hours
            PredictionHorizon.EXTENDED_TERM: 144, # 12 hours
            PredictionHorizon.STRATEGIC_TERM: 288 # 24 hours
        }
        return horizon_map.get(horizon, 12)
    
    def predict(self, current_metrics: PerformanceMetrics, target_metric: str, 
                horizon: PredictionHorizon) -> Optional[MLPrediction]:
        """Make prediction for specific metric and horizon"""
        # Check cache
        cache_key = f"{target_metric}_{horizon.value}_{int(time.time() // self.config.prediction_cache_ttl)}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Clear old cache entries
        if time.time() - self.last_cache_clear > self.config.prediction_cache_ttl:
            self._clear_prediction_cache()
        
        try:
            model_key = f"{target_metric}_{horizon.value}"
            if model_key not in self.models:
                logger.warning(f"No trained model for {target_metric}_{horizon.value}")
                return None
            
            # Prepare current features
            features = self._prepare_current_features(current_metrics, target_metric)
            if features is None:
                return None
            
            # Get best model for prediction
            best_model_info = self._get_best_model(model_key)
            if best_model_info is None:
                return None
            
            model = best_model_info['model']
            prediction_value = float(model.predict([features])[0])
            
            # Calculate confidence based on model performance
            confidence = min(best_model_info.get('r2', 0.0), 1.0)
            
            # Create prediction result
            prediction = MLPrediction(
                horizon=horizon,
                predicted_value=prediction_value,
                confidence_score=confidence,
                model_type=MLModelType.RANDOM_FOREST,  # Simplified for now
                feature_importance={},  # Simplified for now
                prediction_timestamp=datetime.now(),
                target_metric=target_metric,
                model_accuracy=confidence,
                training_samples=best_model_info.get('training_samples', 0),
                feature_count=best_model_info.get('feature_count', 0)
            )
            
            # Cache prediction
            self.prediction_cache[cache_key] = prediction
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed for {target_metric}: {e}")
            return None
    
    def _prepare_current_features(self, current_metrics: PerformanceMetrics, target_metric: str) -> Optional[np.ndarray]:
        """Prepare current metrics for prediction"""
        try:
            # Convert to DataFrame for feature engineering
            current_data = {
                'timestamp': current_metrics.timestamp,
                'cpu_usage': current_metrics.cpu_usage,
                'memory_usage': current_metrics.memory_usage,
                'disk_io_read': current_metrics.disk_io_read,
                'disk_io_write': current_metrics.disk_io_write,
                'network_in': current_metrics.network_in,
                'network_out': current_metrics.network_out,
                'response_time_avg': current_metrics.response_time_avg,
                'response_time_p95': current_metrics.response_time_p95,
                'response_time_p99': current_metrics.response_time_p99,
                'throughput': current_metrics.throughput,
                'error_rate': current_metrics.error_rate,
                'cache_hit_ratio': current_metrics.cache_hit_ratio,
                'database_query_time': current_metrics.database_query_time,
                'active_connections': current_metrics.active_connections,
                'queue_length': current_metrics.queue_length,
                'cpu_load_1min': current_metrics.cpu_load_1min,
                'cpu_load_5min': current_metrics.cpu_load_5min,
                'memory_swap_used': current_metrics.memory_swap_used,
                'gc_collections': current_metrics.gc_collections,
                'thread_count': current_metrics.thread_count
            }
            
            df = pd.DataFrame([current_data])
            
            # Apply feature engineering (simplified for single sample)
            df = self._add_simple_features(df)
            
            # Select numeric features
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_metric in feature_columns:
                feature_columns.remove(target_metric)
            
            # Handle missing values
            df_features = df[feature_columns].fillna(0)
            
            # Scale features
            scaler_key = f"{target_metric}_scaler"
            if scaler_key in self.scalers:
                features_scaled = self.scalers[scaler_key].transform(df_features.values)
                return features_scaled[0]
            else:
                return df_features.values[0]
                
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None
    
    def _add_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple derived features for single sample prediction"""
        try:
            # Resource utilization ratios
            if 'cpu_usage' in df.columns and 'memory_usage' in df.columns:
                df['cpu_memory_ratio'] = df['cpu_usage'] / (df['memory_usage'] + 1e-6)
            
            # Performance efficiency metrics
            if 'throughput' in df.columns and 'cpu_usage' in df.columns:
                df['throughput_per_cpu'] = df['throughput'] / (df['cpu_usage'] + 1e-6)
            
            # Network efficiency
            if 'network_in' in df.columns and 'network_out' in df.columns:
                df['network_total'] = df['network_in'] + df['network_out']
                df['network_ratio'] = df['network_in'] / (df['network_out'] + 1e-6)
            
            # Time-based features
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
            
        except Exception as e:
            logger.error(f"Simple feature creation failed: {e}")
        
        return df
    
    def _get_best_model(self, model_key: str) -> Optional[Dict]:
        """Get best performing model for prediction"""
        if model_key not in self.models:
            return None
        
        models = self.models[model_key]
        if not models:
            return None
        
        # Find model with highest R²
        best_model = None
        best_r2 = -float('inf')
        
        for name, model_info in models.items():
            r2 = model_info.get('r2', -float('inf'))
            if r2 > best_r2 and r2 > self.config.model_performance_threshold:
                best_r2 = r2
                best_model = model_info
        
        return best_model
    
    def _clear_prediction_cache(self):
        """Clear old prediction cache entries"""
        self.prediction_cache.clear()
        self.last_cache_clear = time.time()

class AdvancedAnomalyDetector:
    """Advanced anomaly detection with multiple algorithms"""
    
    def __init__(self, config: EnhancedMLConfig):
        self.config = config
        self.baseline_models = {}
        self.anomaly_history = []
        
    def fit_baseline(self, metrics_history: List[PerformanceMetrics]):
        """Fit baseline models for anomaly detection"""
        if not metrics_history:
            return
        
        try:
            # Prepare data
            df = self._prepare_anomaly_data(metrics_history)
            
            # Fit isolation forest for multivariate anomaly detection
            if SKLEARN_AVAILABLE and len(df) >= self.config.min_training_samples:
                self.baseline_models['isolation_forest'] = IsolationForest(
                    contamination=self.config.isolation_forest_contamination,
                    random_state=42
                )
                self.baseline_models['isolation_forest'].fit(df.select_dtypes(include=[np.number]).fillna(0))
            
            # Fit statistical baselines
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            self.baseline_models['statistical'] = {}
            
            for col in numeric_columns:
                values = df[col].dropna()
                if len(values) > 0:
                    self.baseline_models['statistical'][col] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                        'iqr': values.quantile(0.75) - values.quantile(0.25)
                    }
            
            logger.info("Fitted baseline anomaly detection models")
            
        except Exception as e:
            logger.error(f"Baseline fitting failed: {e}")
    
    def detect_anomalies(self, current_metrics: PerformanceMetrics) -> List[AnomalyDetection]:
        """Detect anomalies in current metrics"""
        anomalies = []
        
        try:
            # Multivariate anomaly detection
            if 'isolation_forest' in self.baseline_models:
                anomalies.extend(self._detect_multivariate_anomalies(current_metrics))
            
            # Statistical anomaly detection
            if 'statistical' in self.baseline_models:
                anomalies.extend(self._detect_statistical_anomalies(current_metrics))
            
            # Pattern-based anomaly detection
            anomalies.extend(self._detect_pattern_anomalies(current_metrics))
            
            # Store anomaly history
            self.anomaly_history.extend(anomalies)
            
            # Limit history size
            if len(self.anomaly_history) > 1000:
                self.anomaly_history = self.anomaly_history[-1000:]
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _prepare_anomaly_data(self, metrics_history: List[PerformanceMetrics]) -> pd.DataFrame:
        """Prepare data for anomaly detection"""
        data = []
        for metrics in metrics_history:
            data.append({
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'response_time_avg': metrics.response_time_avg,
                'response_time_p95': metrics.response_time_p95,
                'throughput': metrics.throughput,
                'error_rate': metrics.error_rate,
                'cache_hit_ratio': metrics.cache_hit_ratio,
                'database_query_time': metrics.database_query_time,
                'active_connections': metrics.active_connections,
                'queue_length': metrics.queue_length
            })
        
        return pd.DataFrame(data)
    
    def _detect_multivariate_anomalies(self, current_metrics: PerformanceMetrics) -> List[AnomalyDetection]:
        """Detect anomalies using isolation forest"""
        anomalies = []
        
        try:
            model = self.baseline_models['isolation_forest']
            
            # Prepare current data
            current_data = np.array([[
                current_metrics.cpu_usage,
                current_metrics.memory_usage,
                current_metrics.response_time_avg,
                current_metrics.response_time_p95,
                current_metrics.throughput,
                current_metrics.error_rate,
                current_metrics.cache_hit_ratio,
                current_metrics.database_query_time,
                current_metrics.active_connections,
                current_metrics.queue_length
            ]])
            
            # Predict anomaly
            prediction = model.predict(current_data)[0]
            anomaly_score = model.decision_function(current_data)[0]
            
            if prediction == -1:  # Anomaly detected
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                    severity=abs(anomaly_score),
                    confidence=min(abs(anomaly_score) * 2, 1.0),
                    affected_metrics=['multivariate_pattern'],
                    detected_timestamp=datetime.now(),
                    baseline_value=0.0,
                    anomalous_value=anomaly_score,
                    detection_method='isolation_forest',
                    recommended_action='Investigate system-wide performance patterns'
                ))
                
        except Exception as e:
            logger.error(f"Multivariate anomaly detection failed: {e}")
        
        return anomalies
    
    def _detect_statistical_anomalies(self, current_metrics: PerformanceMetrics) -> List[AnomalyDetection]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        try:
            baselines = self.baseline_models['statistical']
            
            # Check each metric
            metrics_to_check = {
                'cpu_usage': current_metrics.cpu_usage,
                'memory_usage': current_metrics.memory_usage,
                'response_time_avg': current_metrics.response_time_avg,
                'response_time_p95': current_metrics.response_time_p95,
                'throughput': current_metrics.throughput,
                'error_rate': current_metrics.error_rate,
                'database_query_time': current_metrics.database_query_time
            }
            
            for metric_name, current_value in metrics_to_check.items():
                if metric_name not in baselines:
                    continue
                
                baseline = baselines[metric_name]
                
                # Z-score anomaly detection
                if baseline['std'] > 0:
                    z_score = abs(current_value - baseline['mean']) / baseline['std']
                    
                    if z_score > 3:  # 3-sigma rule
                        anomaly_type = self._determine_anomaly_type(metric_name, current_value, baseline['mean'])
                        
                        anomalies.append(AnomalyDetection(
                            anomaly_type=anomaly_type,
                            severity=min(z_score / 5, 1.0),
                            confidence=min(z_score / 3, 1.0),
                            affected_metrics=[metric_name],
                            detected_timestamp=datetime.now(),
                            baseline_value=baseline['mean'],
                            anomalous_value=current_value,
                            detection_method='statistical_zscore',
                            recommended_action=self._get_anomaly_recommendation(anomaly_type)
                        ))
                
                # IQR anomaly detection
                q1, q3 = baseline['q25'], baseline['q75']
                iqr = baseline['iqr']
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    if current_value < lower_bound or current_value > upper_bound:
                        anomaly_type = self._determine_anomaly_type(metric_name, current_value, (q1 + q3) / 2)
                        
                        anomalies.append(AnomalyDetection(
                            anomaly_type=anomaly_type,
                            severity=0.7,
                            confidence=0.8,
                            affected_metrics=[metric_name],
                            detected_timestamp=datetime.now(),
                            baseline_value=(q1 + q3) / 2,
                            anomalous_value=current_value,
                            detection_method='statistical_iqr',
                            recommended_action=self._get_anomaly_recommendation(anomaly_type)
                        ))
                        
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
        
        return anomalies
    
    def _detect_pattern_anomalies(self, current_metrics: PerformanceMetrics) -> List[AnomalyDetection]:
        """Detect pattern-based anomalies"""
        anomalies = []
        
        try:
            # Cache miss spike detection
            if current_metrics.cache_hit_ratio < 0.5:  # Below 50% hit rate
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.CACHE_MISS_SPIKE,
                    severity=1.0 - current_metrics.cache_hit_ratio,
                    confidence=0.9,
                    affected_metrics=['cache_hit_ratio'],
                    detected_timestamp=datetime.now(),
                    baseline_value=0.8,
                    anomalous_value=current_metrics.cache_hit_ratio,
                    detection_method='pattern_based',
                    recommended_action='Check cache configuration and invalidation patterns'
                ))
            
            # High error rate detection
            if current_metrics.error_rate > 0.05:  # Above 5% error rate
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.ERROR_RATE_INCREASE,
                    severity=min(current_metrics.error_rate * 10, 1.0),
                    confidence=0.9,
                    affected_metrics=['error_rate'],
                    detected_timestamp=datetime.now(),
                    baseline_value=0.01,
                    anomalous_value=current_metrics.error_rate,
                    detection_method='pattern_based',
                    recommended_action='Investigate error logs and system health'
                ))
            
            # Database slowdown detection
            if current_metrics.database_query_time > 100:  # Above 100ms
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.DATABASE_SLOWDOWN,
                    severity=min(current_metrics.database_query_time / 500, 1.0),
                    confidence=0.8,
                    affected_metrics=['database_query_time'],
                    detected_timestamp=datetime.now(),
                    baseline_value=50.0,
                    anomalous_value=current_metrics.database_query_time,
                    detection_method='pattern_based',
                    recommended_action='Analyze database performance and query optimization'
                ))
                        
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
        
        return anomalies
    
    def _determine_anomaly_type(self, metric_name: str, current_value: float, baseline_value: float) -> AnomalyType:
        """Determine anomaly type based on metric"""
        if current_value > baseline_value:
            if metric_name in ['cpu_usage', 'memory_usage']:
                return AnomalyType.RESOURCE_SPIKE
            elif 'response_time' in metric_name:
                return AnomalyType.LATENCY_ANOMALY
            elif metric_name == 'error_rate':
                return AnomalyType.ERROR_RATE_INCREASE
            elif metric_name == 'database_query_time':
                return AnomalyType.DATABASE_SLOWDOWN
        else:
            if metric_name == 'throughput':
                return AnomalyType.THROUGHPUT_DROP
            elif metric_name == 'cache_hit_ratio':
                return AnomalyType.CACHE_MISS_SPIKE
        
        return AnomalyType.PERFORMANCE_DEGRADATION
    
    def _get_anomaly_recommendation(self, anomaly_type: AnomalyType) -> str:
        """Get recommendation for anomaly type"""
        recommendations = {
            AnomalyType.PERFORMANCE_DEGRADATION: "Review system resources and recent changes",
            AnomalyType.RESOURCE_SPIKE: "Check for memory leaks and resource-intensive processes",
            AnomalyType.LATENCY_ANOMALY: "Investigate network connectivity and processing delays",
            AnomalyType.THROUGHPUT_DROP: "Analyze request processing bottlenecks",
            AnomalyType.ERROR_RATE_INCREASE: "Review error logs and system health checks",
            AnomalyType.CACHE_MISS_SPIKE: "Optimize caching strategy and invalidation patterns",
            AnomalyType.DATABASE_SLOWDOWN: "Review database queries and indexing strategy"
        }
        return recommendations.get(anomaly_type, "Investigate system performance patterns")

class PerformanceIntelligenceDatabase:
    """Database for performance intelligence data"""
    
    def __init__(self, config: EnhancedMLConfig):
        self.config = config
        self.db_path = config.db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Performance metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_io_read REAL,
                    disk_io_write REAL,
                    network_in REAL,
                    network_out REAL,
                    response_time_avg REAL,
                    response_time_p95 REAL,
                    response_time_p99 REAL,
                    throughput REAL,
                    error_rate REAL,
                    cache_hit_ratio REAL,
                    database_query_time REAL,
                    active_connections INTEGER,
                    queue_length INTEGER,
                    cpu_load_1min REAL,
                    cpu_load_5min REAL,
                    memory_swap_used REAL,
                    gc_collections INTEGER,
                    thread_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Predictions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    horizon TEXT NOT NULL,
                    predicted_value REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    prediction_timestamp DATETIME NOT NULL,
                    target_metric TEXT NOT NULL,
                    model_accuracy REAL,
                    training_samples INTEGER,
                    feature_count INTEGER,
                    prediction_variance REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Anomalies table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anomaly_type TEXT NOT NULL,
                    severity REAL NOT NULL,
                    confidence REAL NOT NULL,
                    affected_metrics TEXT NOT NULL,
                    detected_timestamp DATETIME NOT NULL,
                    baseline_value REAL,
                    anomalous_value REAL,
                    detection_method TEXT NOT NULL,
                    recommended_action TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ml_predictions(prediction_timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(detected_timestamp)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def store_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO performance_metrics (
                    timestamp, cpu_usage, memory_usage, disk_io_read, disk_io_write,
                    network_in, network_out, response_time_avg, response_time_p95,
                    response_time_p99, throughput, error_rate, cache_hit_ratio,
                    database_query_time, active_connections, queue_length,
                    cpu_load_1min, cpu_load_5min, memory_swap_used, gc_collections,
                    thread_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp, metrics.cpu_usage, metrics.memory_usage,
                    metrics.disk_io_read, metrics.disk_io_write, metrics.network_in,
                    metrics.network_out, metrics.response_time_avg, metrics.response_time_p95,
                    metrics.response_time_p99, metrics.throughput, metrics.error_rate,
                    metrics.cache_hit_ratio, metrics.database_query_time, metrics.active_connections,
                    metrics.queue_length, metrics.cpu_load_1min, metrics.cpu_load_5min,
                    metrics.memory_swap_used, metrics.gc_collections, metrics.thread_count
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def get_metrics_history(self, hours: int = 24) -> List[PerformanceMetrics]:
        """Get metrics history"""
        try:
            since = datetime.now() - timedelta(hours=hours)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM performance_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp
                ''', (since,))
                
                metrics_list = []
                for row in cursor.fetchall():
                    metrics = PerformanceMetrics(
                        timestamp=datetime.fromisoformat(row[1]),
                        cpu_usage=row[2] or 0.0,
                        memory_usage=row[3] or 0.0,
                        disk_io_read=row[4] or 0.0,
                        disk_io_write=row[5] or 0.0,
                        network_in=row[6] or 0.0,
                        network_out=row[7] or 0.0,
                        response_time_avg=row[8] or 0.0,
                        response_time_p95=row[9] or 0.0,
                        response_time_p99=row[10] or 0.0,
                        throughput=row[11] or 0.0,
                        error_rate=row[12] or 0.0,
                        cache_hit_ratio=row[13] or 0.0,
                        database_query_time=row[14] or 0.0,
                        active_connections=row[15] or 0,
                        queue_length=row[16] or 0,
                        cpu_load_1min=row[17] or 0.0,
                        cpu_load_5min=row[18] or 0.0,
                        memory_swap_used=row[19] or 0.0,
                        gc_collections=row[20] or 0,
                        thread_count=row[21] or 0
                    )
                    metrics_list.append(metrics)
                
                return metrics_list
                
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []
    
    def store_prediction(self, prediction: MLPrediction):
        """Store ML prediction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO ml_predictions (
                    horizon, predicted_value, confidence_score, model_type,
                    prediction_timestamp, target_metric, model_accuracy,
                    training_samples, feature_count, prediction_variance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.horizon.value, prediction.predicted_value,
                    prediction.confidence_score, prediction.model_type.value,
                    prediction.prediction_timestamp, prediction.target_metric,
                    prediction.model_accuracy, prediction.training_samples,
                    prediction.feature_count, prediction.prediction_variance
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
    
    def store_anomaly(self, anomaly: AnomalyDetection):
        """Store anomaly detection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO anomalies (
                    anomaly_type, severity, confidence, affected_metrics,
                    detected_timestamp, baseline_value, anomalous_value,
                    detection_method, recommended_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    anomaly.anomaly_type.value, anomaly.severity, anomaly.confidence,
                    ','.join(anomaly.affected_metrics), anomaly.detected_timestamp,
                    anomaly.baseline_value, anomaly.anomalous_value,
                    anomaly.detection_method, anomaly.recommended_action
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store anomaly: {e}")

class EnhancedMLPerformanceIntelligence:
    """Enhanced ML Performance Intelligence System"""
    
    def __init__(self, config: EnhancedMLConfig = None):
        self.config = config or EnhancedMLConfig()
        self.database = PerformanceIntelligenceDatabase(self.config)
        self.ml_predictor = EnhancedMLPredictor(self.config)
        self.anomaly_detector = AdvancedAnomalyDetector(self.config)
        
        # System state
        self.is_running = False
        self.last_training_time = 0
        self.prediction_threads = {}
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the enhanced ML performance intelligence system"""
        try:
            # Load historical data for training
            historical_metrics = self.database.get_metrics_history(hours=24)
            
            if len(historical_metrics) >= self.config.min_training_samples:
                # Train models for key metrics
                target_metrics = ['cpu_usage', 'memory_usage', 'response_time_avg', 'throughput']
                for metric in target_metrics:
                    self.ml_predictor.train_models(historical_metrics, metric)
                
                # Fit anomaly detection baselines
                self.anomaly_detector.fit_baseline(historical_metrics)
                
                logger.info("Enhanced ML Performance Intelligence initialized successfully")
            else:
                logger.warning(f"Insufficient historical data for training: {len(historical_metrics)} samples")
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
    
    def analyze_performance(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Comprehensive performance analysis with ML predictions and anomaly detection"""
        try:
            # Store current metrics
            self.database.store_metrics(current_metrics)
            
            # Generate predictions
            predictions = self._generate_predictions(current_metrics)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
            
            # Store results
            for prediction in predictions.values():
                if prediction:
                    self.database.store_prediction(prediction)
            
            for anomaly in anomalies:
                self.database.store_anomaly(anomaly)
            
            # Generate insights
            insights = self._generate_insights(current_metrics, predictions, anomalies)
            
            # Prepare analysis result
            analysis_result = {
                'timestamp': current_metrics.timestamp.isoformat(),
                'current_metrics': asdict(current_metrics),
                'predictions': {k: asdict(v) if v else None for k, v in predictions.items()},
                'anomalies': [asdict(anomaly) for anomaly in anomalies],
                'insights': insights,
                'system_health': self._assess_system_health(current_metrics, predictions, anomalies),
                'recommendations': self._generate_recommendations(current_metrics, predictions, anomalies)
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'system_health': 'unknown'
            }
    
    def _generate_predictions(self, current_metrics: PerformanceMetrics) -> Dict[str, MLPrediction]:
        """Generate ML predictions for key metrics"""
        predictions = {}
        target_metrics = ['cpu_usage', 'memory_usage', 'response_time_avg', 'throughput']
        
        try:
            for metric in target_metrics:
                for horizon in [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]:
                    prediction = self.ml_predictor.predict(current_metrics, metric, horizon)
                    if prediction and prediction.confidence_score >= self.config.prediction_confidence_threshold:
                        key = f"{metric}_{horizon.value}"
                        predictions[key] = prediction
                        
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
        
        return predictions
    
    def _generate_insights(self, current_metrics: PerformanceMetrics, 
                         predictions: Dict[str, MLPrediction], 
                         anomalies: List[AnomalyDetection]) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        try:
            # Performance trend insights
            cpu_prediction = predictions.get('cpu_usage_medium_term')
            if cpu_prediction:
                if cpu_prediction.predicted_value > current_metrics.cpu_usage * 1.2:
                    insights.append(f"CPU usage expected to increase by {(cpu_prediction.predicted_value - current_metrics.cpu_usage):.1f}% in next hour")
                elif cpu_prediction.predicted_value < current_metrics.cpu_usage * 0.8:
                    insights.append(f"CPU usage expected to decrease by {(current_metrics.cpu_usage - cpu_prediction.predicted_value):.1f}% in next hour")
            
            # Memory trend insights
            memory_prediction = predictions.get('memory_usage_medium_term')
            if memory_prediction:
                if memory_prediction.predicted_value > 80:
                    insights.append("Memory usage approaching critical levels - consider scaling or optimization")
            
            # Response time insights
            response_prediction = predictions.get('response_time_avg_medium_term')
            if response_prediction:
                if response_prediction.predicted_value > current_metrics.response_time_avg * 1.5:
                    insights.append("Response time degradation predicted - investigate potential bottlenecks")
            
            # Anomaly insights
            if anomalies:
                high_severity_anomalies = [a for a in anomalies if a.severity > 0.7]
                if high_severity_anomalies:
                    insights.append(f"High-severity anomalies detected in {len(high_severity_anomalies)} metrics")
            
            # Cache performance insights
            if current_metrics.cache_hit_ratio < 0.8:
                insights.append(f"Cache hit ratio below optimal: {current_metrics.cache_hit_ratio:.1%}")
            
            # Database performance insights
            if current_metrics.database_query_time > 50:
                insights.append(f"Database queries slower than target: {current_metrics.database_query_time:.1f}ms avg")
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
        
        return insights
    
    def _assess_system_health(self, current_metrics: PerformanceMetrics,
                            predictions: Dict[str, MLPrediction],
                            anomalies: List[AnomalyDetection]) -> str:
        """Assess overall system health"""
        try:
            health_score = 100.0
            
            # Current performance impact
            if current_metrics.cpu_usage > 80:
                health_score -= 20
            elif current_metrics.cpu_usage > 60:
                health_score -= 10
            
            if current_metrics.memory_usage > 85:
                health_score -= 20
            elif current_metrics.memory_usage > 70:
                health_score -= 10
            
            if current_metrics.response_time_avg > 100:
                health_score -= 15
            elif current_metrics.response_time_avg > 50:
                health_score -= 8
            
            if current_metrics.error_rate > 0.05:
                health_score -= 25
            elif current_metrics.error_rate > 0.01:
                health_score -= 10
            
            # Predicted performance impact
            for prediction in predictions.values():
                if prediction and prediction.confidence_score > 0.7:
                    if 'cpu_usage' in prediction.target_metric and prediction.predicted_value > 85:
                        health_score -= 10
                    elif 'response_time' in prediction.target_metric and prediction.predicted_value > 150:
                        health_score -= 10
            
            # Anomaly impact
            for anomaly in anomalies:
                if anomaly.severity > 0.8:
                    health_score -= 15
                elif anomaly.severity > 0.5:
                    health_score -= 8
            
            # Determine health status
            if health_score >= 90:
                return 'excellent'
            elif health_score >= 75:
                return 'good'
            elif health_score >= 60:
                return 'fair'
            elif health_score >= 40:
                return 'poor'
            else:
                return 'critical'
                
        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            return 'unknown'
    
    def _generate_recommendations(self, current_metrics: PerformanceMetrics,
                               predictions: Dict[str, MLPrediction],
                               anomalies: List[AnomalyDetection]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        try:
            # Resource optimization recommendations
            if current_metrics.cpu_usage > 75:
                recommendations.append("Consider CPU optimization or scaling up instances")
            
            if current_metrics.memory_usage > 80:
                recommendations.append("Memory usage high - investigate memory leaks or increase memory allocation")
            
            # Cache optimization recommendations
            if current_metrics.cache_hit_ratio < 0.7:
                recommendations.append("Cache hit ratio low - review caching strategy and TTL settings")
            
            # Database optimization recommendations
            if current_metrics.database_query_time > 100:
                recommendations.append("Database queries slow - analyze query performance and indexing")
            
            # Predictive recommendations
            cpu_prediction = predictions.get('cpu_usage_medium_term')
            if cpu_prediction and cpu_prediction.predicted_value > 85:
                recommendations.append("CPU usage predicted to be high - prepare for scaling or optimization")
            
            # Anomaly-based recommendations
            for anomaly in anomalies:
                if anomaly.recommended_action and anomaly.severity > 0.5:
                    recommendations.append(f"Anomaly detected: {anomaly.recommended_action}")
            
            # Performance trend recommendations
            if current_metrics.response_time_avg > current_metrics.response_time_p95 * 0.8:
                recommendations.append("Response time variance high - investigate request processing consistency")
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    def get_performance_forecast(self, hours_ahead: int = 4) -> Dict[str, Any]:
        """Get performance forecast for specified hours"""
        try:
            # Get recent metrics for context
            recent_metrics = self.database.get_metrics_history(hours=1)
            if not recent_metrics:
                return {'error': 'No recent metrics available for forecasting'}
            
            # Use most recent metrics as baseline
            current_metrics = recent_metrics[-1]
            
            # Generate forecasts for different horizons
            forecasts = {}
            horizons = [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM, PredictionHorizon.LONG_TERM]
            target_metrics = ['cpu_usage', 'memory_usage', 'response_time_avg', 'throughput']
            
            for metric in target_metrics:
                metric_forecasts = {}
                for horizon in horizons:
                    prediction = self.ml_predictor.predict(current_metrics, metric, horizon)
                    if prediction:
                        metric_forecasts[horizon.value] = {
                            'predicted_value': prediction.predicted_value,
                            'confidence': prediction.confidence_score,
                            'current_value': getattr(current_metrics, metric, 0),
                            'change_percent': ((prediction.predicted_value - getattr(current_metrics, metric, 0)) / 
                                             max(getattr(current_metrics, metric, 1), 1)) * 100
                        }
                forecasts[metric] = metric_forecasts
            
            return {
                'forecast_timestamp': datetime.now().isoformat(),
                'baseline_timestamp': current_metrics.timestamp.isoformat(),
                'forecasts': forecasts,
                'forecast_horizon_hours': hours_ahead
            }
            
        except Exception as e:
            logger.error(f"Performance forecasting failed: {e}")
            return {'error': str(e)}
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get anomaly summary for specified time period"""
        try:
            # This would typically query the database
            # For now, return recent anomalies from detector
            recent_anomalies = self.anomaly_detector.anomaly_history[-50:] if self.anomaly_detector.anomaly_history else []
            
            # Categorize anomalies
            anomaly_categories = {}
            severity_distribution = {'low': 0, 'medium': 0, 'high': 0}
            
            for anomaly in recent_anomalies:
                category = anomaly.anomaly_type.value
                if category not in anomaly_categories:
                    anomaly_categories[category] = 0
                anomaly_categories[category] += 1
                
                if anomaly.severity > 0.7:
                    severity_distribution['high'] += 1
                elif anomaly.severity > 0.4:
                    severity_distribution['medium'] += 1
                else:
                    severity_distribution['low'] += 1
            
            return {
                'summary_timestamp': datetime.now().isoformat(),
                'time_period_hours': hours,
                'total_anomalies': len(recent_anomalies),
                'anomaly_categories': anomaly_categories,
                'severity_distribution': severity_distribution,
                'recent_anomalies': [
                    {
                        'type': anomaly.anomaly_type.value,
                        'severity': anomaly.severity,
                        'confidence': anomaly.confidence,
                        'timestamp': anomaly.detected_timestamp.isoformat(),
                        'affected_metrics': anomaly.affected_metrics,
                        'recommendation': anomaly.recommended_action
                    }
                    for anomaly in recent_anomalies[-10:]  # Last 10 anomalies
                ]
            }
            
        except Exception as e:
            logger.error(f"Anomaly summary failed: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get enhanced ML performance intelligence system status"""
        try:
            # Get recent metrics
            recent_metrics = self.database.get_metrics_history(hours=1)
            
            system_status = {
                'system_name': 'Enhanced ML Performance Intelligence',
                'version': '2.0.0',
                'status': 'operational' if recent_metrics else 'no_data',
                'timestamp': datetime.now().isoformat(),
                
                # Data status
                'data_status': {
                    'recent_metrics_count': len(recent_metrics),
                    'training_data_sufficient': len(recent_metrics) >= self.config.min_training_samples,
                    'last_metric_timestamp': recent_metrics[-1].timestamp.isoformat() if recent_metrics else None
                },
                
                # Model status
                'model_status': {
                    'trained_models_count': len(self.ml_predictor.models),
                    'model_performance': dict(list(self.ml_predictor.model_performance.items())[:5]),  # Top 5 models
                    'anomaly_baseline_fitted': len(self.anomaly_detector.baseline_models) > 0
                },
                
                # Configuration
                'configuration': {
                    'prediction_horizons': [h.value for h in self.config.prediction_horizons],
                    'model_types': [m.value for m in self.config.model_types],
                    'confidence_threshold': self.config.prediction_confidence_threshold,
                    'anomaly_sensitivity': self.config.anomaly_sensitivity
                },
                
                # Capabilities
                'capabilities': {
                    'ml_prediction': True,
                    'anomaly_detection': True,
                    'performance_forecasting': True,
                    'advanced_feature_engineering': self.config.feature_engineering_enabled,
                    'sklearn_available': SKLEARN_AVAILABLE,
                    'scipy_available': SCIPY_AVAILABLE
                }
            }
            
            return system_status
            
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {
                'system_name': 'Enhanced ML Performance Intelligence',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Demonstration of Enhanced ML Performance Intelligence System"""
    print("=== Enhanced ML Performance Intelligence System Demo ===")
    
    # Initialize system
    config = EnhancedMLConfig()
    intelligence = EnhancedMLPerformanceIntelligence(config)
    
    # Generate sample metrics
    import random
    sample_metrics = []
    base_time = datetime.now()
    
    for i in range(100):
        timestamp = base_time - timedelta(minutes=i*5)
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=random.uniform(20, 80) + random.gauss(0, 5),
            memory_usage=random.uniform(30, 70) + random.gauss(0, 5),
            disk_io_read=random.uniform(10, 100),
            disk_io_write=random.uniform(5, 50),
            network_in=random.uniform(100, 1000),
            network_out=random.uniform(50, 500),
            response_time_avg=random.uniform(20, 100) + random.gauss(0, 10),
            response_time_p95=random.uniform(50, 200),
            response_time_p99=random.uniform(80, 300),
            throughput=random.uniform(50, 200),
            error_rate=random.uniform(0, 0.1),
            cache_hit_ratio=random.uniform(0.6, 0.95),
            database_query_time=random.uniform(10, 100),
            active_connections=random.randint(10, 100),
            queue_length=random.randint(0, 20),
            cpu_load_1min=random.uniform(0.5, 3.0),
            cpu_load_5min=random.uniform(0.3, 2.5),
            memory_swap_used=random.uniform(0, 1024),
            gc_collections=random.randint(0, 10),
            thread_count=random.randint(10, 50)
        )
        sample_metrics.append(metrics)
        intelligence.database.store_metrics(metrics)
    
    print(f"Generated {len(sample_metrics)} sample metrics")
    
    # Train models
    print("\nTraining ML models...")
    for metric in ['cpu_usage', 'memory_usage', 'response_time_avg']:
        intelligence.ml_predictor.train_models(sample_metrics, metric)
    
    # Fit anomaly baselines
    print("Fitting anomaly detection baselines...")
    intelligence.anomaly_detector.fit_baseline(sample_metrics)
    
    # Analyze current performance
    print("\nAnalyzing current performance...")
    current_metrics = sample_metrics[0]  # Most recent
    analysis = intelligence.analyze_performance(current_metrics)
    
    print("Analysis Results:")
    print(f"- System Health: {analysis['system_health']}")
    print(f"- Anomalies Detected: {len(analysis['anomalies'])}")
    print(f"- Predictions Generated: {len([p for p in analysis['predictions'].values() if p])}")
    print(f"- Insights: {len(analysis['insights'])}")
    
    # Show insights
    if analysis['insights']:
        print("\nKey Insights:")
        for insight in analysis['insights'][:3]:
            print(f"  • {insight}")
    
    # Show recommendations
    if analysis['recommendations']:
        print("\nRecommendations:")
        for rec in analysis['recommendations'][:3]:
            print(f"  • {rec}")
    
    # Performance forecast
    print("\nGenerating performance forecast...")
    forecast = intelligence.get_performance_forecast(hours_ahead=4)
    if 'forecasts' in forecast:
        print("Forecast Results:")
        for metric, predictions in forecast['forecasts'].items():
            if predictions:
                medium_term = predictions.get('1_hour')
                if medium_term:
                    print(f"  • {metric}: {medium_term['predicted_value']:.1f} "
                          f"(confidence: {medium_term['confidence']:.2f})")
    
    # System status
    print("\nSystem Status:")
    status = intelligence.get_system_status()
    print(f"- Status: {status['status']}")
    print(f"- Trained Models: {status['model_status']['trained_models_count']}")
    print(f"- Recent Metrics: {status['data_status']['recent_metrics_count']}")
    
    print("\n=== Enhanced ML Performance Intelligence Demo Complete ===")

if __name__ == "__main__":
    main()