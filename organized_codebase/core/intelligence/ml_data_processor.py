#!/usr/bin/env python3
"""
ML Data Processor Module
=========================

Data processing functionality extracted from advanced_predictive_analytics.py
for STEELCLAD modularization (Agent Y STEELCLAD Protocol)

Handles:
- Feature preparation for ML models
- Data validation and cleaning
- Historical data management
- Feature engineering for predictions
"""

import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path


class MLFeatureProcessor:
    """Processes and prepares features for ML models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def prepare_health_features(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Prepare features for health trend prediction
        
        Args:
            metrics: Raw system metrics
            
        Returns:
            Processed features for health prediction model
        """
        try:
            features = {
                'cpu_usage': float(metrics.get('cpu_usage', 50.0)),
                'memory_usage': float(metrics.get('memory_usage', 50.0)),
                'response_time': float(metrics.get('avg_response_time', 100.0)),
                'error_rate': float(metrics.get('error_rate', 0.0)),
                'service_count': float(metrics.get('active_services', 10)),
                'dependency_health': float(metrics.get('dependency_health', 100.0)),
                'import_success_rate': float(metrics.get('import_success_rate', 100.0))
            }
            
            # Validate and clamp values
            features = self._validate_features(features, {
                'cpu_usage': (0, 100),
                'memory_usage': (0, 100),
                'response_time': (0, 5000),
                'error_rate': (0, 100),
                'service_count': (0, 1000),
                'dependency_health': (0, 100),
                'import_success_rate': (0, 100)
            })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Health feature preparation failed: {e}")
            return self._default_health_features()
    
    def prepare_anomaly_features(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Prepare features for anomaly detection
        
        Args:
            metrics: Raw system metrics with historical data
            
        Returns:
            Processed features for anomaly detection model
        """
        try:
            # Extract historical data for variance calculation
            cpu_history = metrics.get('cpu_history', [50.0] * 10)
            memory_history = metrics.get('memory_history', [50.0] * 10)
            
            # Ensure we have enough data points
            if len(cpu_history) < 5:
                cpu_history = [50.0] * 10
            if len(memory_history) < 5:
                memory_history = [50.0] * 10
            
            features = {
                'cpu_variance': float(np.var(cpu_history)),
                'memory_variance': float(np.var(memory_history)),
                'cpu_spike_factor': float(max(cpu_history) - np.mean(cpu_history)),
                'memory_spike_factor': float(max(memory_history) - np.mean(memory_history)),
                'response_time_spike': float(metrics.get('response_time_spike', 0)),
                'error_rate_change': float(metrics.get('error_rate_change', 0)),
                'service_failures': float(metrics.get('service_failures', 0)),
                'dependency_changes': float(metrics.get('dependency_changes', 0)),
                'concurrent_connections': float(metrics.get('concurrent_connections', 50))
            }
            
            # Validate and clamp values
            features = self._validate_features(features, {
                'cpu_variance': (0, 1000),
                'memory_variance': (0, 1000),
                'cpu_spike_factor': (0, 100),
                'memory_spike_factor': (0, 100),
                'response_time_spike': (0, 1000),
                'error_rate_change': (-100, 100),
                'service_failures': (0, 100),
                'dependency_changes': (0, 50),
                'concurrent_connections': (0, 10000)
            })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Anomaly feature preparation failed: {e}")
            return self._default_anomaly_features()
    
    def prepare_performance_features(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Prepare features for performance degradation prediction
        
        Args:
            metrics: Raw system metrics
            
        Returns:
            Processed features for performance prediction model
        """
        try:
            features = {
                'response_time_trend': float(metrics.get('response_time_trend', 0)),
                'throughput_change': float(metrics.get('throughput_change', 0)),
                'error_rate_trend': float(metrics.get('error_rate_trend', 0)),
                'cpu_utilization_trend': float(metrics.get('cpu_trend', 0)),
                'memory_pressure': float(metrics.get('memory_pressure', 0)),
                'queue_depth': float(metrics.get('queue_depth', 0)),
                'cache_hit_rate': float(metrics.get('cache_hit_rate', 90.0)),
                'concurrent_requests': float(metrics.get('concurrent_requests', 10)),
                'database_connections': float(metrics.get('db_connections', 5))
            }
            
            # Validate and clamp values
            features = self._validate_features(features, {
                'response_time_trend': (-1000, 1000),
                'throughput_change': (-100, 100),
                'error_rate_trend': (-100, 100),
                'cpu_utilization_trend': (-100, 100),
                'memory_pressure': (0, 100),
                'queue_depth': (0, 1000),
                'cache_hit_rate': (0, 100),
                'concurrent_requests': (0, 10000),
                'database_connections': (0, 1000)
            })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Performance feature preparation failed: {e}")
            return self._default_performance_features()
    
    def prepare_resource_features(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Prepare features for resource utilization prediction
        
        Args:
            metrics: Raw system metrics
            
        Returns:
            Processed features for resource prediction model
        """
        try:
            now = datetime.now()
            hour_of_day = now.hour
            day_of_week = now.weekday()
            
            features = {
                'current_cpu': float(metrics.get('cpu_usage', 50.0)),
                'current_memory': float(metrics.get('memory_usage', 50.0)),
                'current_disk': float(metrics.get('disk_usage', 30.0)),
                'request_rate': float(metrics.get('request_rate', 100)),
                'user_sessions': float(metrics.get('user_sessions', 10)),
                'cache_usage': float(metrics.get('cache_usage', 25.0)),
                'hour_of_day': float(hour_of_day),
                'day_of_week': float(day_of_week),
                'is_business_hours': float(1 if 9 <= hour_of_day <= 17 else 0),
                'is_weekend': float(1 if day_of_week >= 5 else 0)
            }
            
            # Validate and clamp values
            features = self._validate_features(features, {
                'current_cpu': (0, 100),
                'current_memory': (0, 100),
                'current_disk': (0, 100),
                'request_rate': (0, 10000),
                'user_sessions': (0, 10000),
                'cache_usage': (0, 100),
                'hour_of_day': (0, 23),
                'day_of_week': (0, 6),
                'is_business_hours': (0, 1),
                'is_weekend': (0, 1)
            })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Resource feature preparation failed: {e}")
            return self._default_resource_features()
    
    def _validate_features(self, features: Dict[str, float], bounds: Dict[str, tuple]) -> Dict[str, float]:
        """
        Validate and clamp feature values to acceptable ranges
        
        Args:
            features: Dictionary of feature values
            bounds: Dictionary of (min, max) bounds for each feature
            
        Returns:
            Validated and clamped features
        """
        validated = {}
        for key, value in features.items():
            if key in bounds:
                min_val, max_val = bounds[key]
                validated[key] = max(min_val, min(max_val, value))
            else:
                validated[key] = value
        return validated
    
    def _default_health_features(self) -> Dict[str, float]:
        """Default health features when processing fails"""
        return {
            'cpu_usage': 50.0,
            'memory_usage': 50.0,
            'response_time': 100.0,
            'error_rate': 0.0,
            'service_count': 10.0,
            'dependency_health': 100.0,
            'import_success_rate': 100.0
        }
    
    def _default_anomaly_features(self) -> Dict[str, float]:
        """Default anomaly features when processing fails"""
        return {
            'cpu_variance': 10.0,
            'memory_variance': 10.0,
            'cpu_spike_factor': 0.0,
            'memory_spike_factor': 0.0,
            'response_time_spike': 0.0,
            'error_rate_change': 0.0,
            'service_failures': 0.0,
            'dependency_changes': 0.0,
            'concurrent_connections': 50.0
        }
    
    def _default_performance_features(self) -> Dict[str, float]:
        """Default performance features when processing fails"""
        return {
            'response_time_trend': 0.0,
            'throughput_change': 0.0,
            'error_rate_trend': 0.0,
            'cpu_utilization_trend': 0.0,
            'memory_pressure': 0.0,
            'queue_depth': 0.0,
            'cache_hit_rate': 90.0,
            'concurrent_requests': 10.0,
            'database_connections': 5.0
        }
    
    def _default_resource_features(self) -> Dict[str, float]:
        """Default resource features when processing fails"""
        now = datetime.now()
        return {
            'current_cpu': 50.0,
            'current_memory': 50.0,
            'current_disk': 30.0,
            'request_rate': 100.0,
            'user_sessions': 10.0,
            'cache_usage': 25.0,
            'hour_of_day': float(now.hour),
            'day_of_week': float(now.weekday()),
            'is_business_hours': float(1 if 9 <= now.hour <= 17 else 0),
            'is_weekend': float(1 if now.weekday() >= 5 else 0)
        }


class HistoricalDataManager:
    """Manages historical data for ML model training and predictions"""
    
    def __init__(self, data_dir: Path, max_history_size: int = 10000):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_size = max_history_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize empty DataFrame
        self.historical_data = pd.DataFrame()
        self._load_historical_data()
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """
        Add new metrics to historical data
        
        Args:
            metrics: Dictionary of system metrics
        """
        try:
            # Ensure timestamp
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now()
            
            # Convert to DataFrame row
            new_row = pd.DataFrame([metrics])
            
            # Add to historical data
            if self.historical_data.empty:
                self.historical_data = new_row
            else:
                self.historical_data = pd.concat([self.historical_data, new_row], ignore_index=True)
            
            # Maintain size limit
            if len(self.historical_data) > self.max_history_size:
                self.historical_data = self.historical_data.tail(self.max_history_size)
            
            # Save to disk periodically
            if len(self.historical_data) % 100 == 0:
                self._save_historical_data()
                
        except Exception as e:
            self.logger.error(f"Failed to add metrics to history: {e}")
    
    def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Get recent historical data within specified hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            DataFrame with recent data
        """
        try:
            if self.historical_data.empty:
                return pd.DataFrame()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Ensure timestamp column is datetime
            if 'timestamp' in self.historical_data.columns:
                self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
                recent_data = self.historical_data[self.historical_data['timestamp'] > cutoff_time]
                return recent_data
            else:
                # If no timestamp column, return most recent data
                return self.historical_data.tail(min(hours * 60, len(self.historical_data)))
                
        except Exception as e:
            self.logger.error(f"Failed to get recent data: {e}")
            return pd.DataFrame()
    
    def get_training_data(self, min_samples: int = 100) -> Optional[pd.DataFrame]:
        """
        Get data suitable for model training
        
        Args:
            min_samples: Minimum number of samples required
            
        Returns:
            DataFrame with training data or None if insufficient data
        """
        try:
            if len(self.historical_data) < min_samples:
                self.logger.warning(f"Insufficient data for training: {len(self.historical_data)} < {min_samples}")
                return None
            
            # Return cleaned data
            cleaned_data = self.historical_data.dropna()
            
            if len(cleaned_data) < min_samples:
                self.logger.warning(f"Insufficient clean data for training: {len(cleaned_data)} < {min_samples}")
                return None
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Failed to get training data: {e}")
            return None
    
    def _load_historical_data(self):
        """Load historical data from disk"""
        try:
            data_file = self.data_dir / "historical_metrics.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data_list = json.load(f)
                
                if data_list:
                    self.historical_data = pd.DataFrame(data_list)
                    # Maintain size limit
                    if len(self.historical_data) > self.max_history_size:
                        self.historical_data = self.historical_data.tail(self.max_history_size)
                    
                    self.logger.info(f"Loaded {len(self.historical_data)} historical records")
                else:
                    self.historical_data = pd.DataFrame()
            else:
                self.historical_data = pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            self.historical_data = pd.DataFrame()
    
    def _save_historical_data(self):
        """Save historical data to disk"""
        try:
            data_file = self.data_dir / "historical_metrics.json"
            
            # Convert DataFrame to list of dictionaries
            data_list = self.historical_data.to_dict('records')
            
            # Convert datetime objects to strings
            for record in data_list:
                for key, value in record.items():
                    if isinstance(value, (datetime, pd.Timestamp)):
                        record[key] = value.isoformat()
            
            with open(data_file, 'w') as f:
                json.dump(data_list, f, indent=2)
            
            self.logger.debug(f"Saved {len(data_list)} historical records")
            
        except Exception as e:
            self.logger.error(f"Failed to save historical data: {e}")