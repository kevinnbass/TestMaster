#!/usr/bin/env python3
"""
Dashboard Configuration Module
==============================

Configuration management extracted from performance_analytics_dashboard.py
for STEELCLAD modularization (Agent Y supporting Agent Z)

Handles:
- Dashboard configuration settings
- Performance thresholds management
- Feature flags and integration settings
- Runtime configuration validation
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import logging


@dataclass
class DashboardConfig:
    """
    Configuration for performance analytics dashboard
    
    Manages all dashboard settings including features, thresholds, and integration points.
    """
    
    # Server configuration
    host: str = "localhost"
    port: int = 5001
    debug: bool = False
    
    # Data and refresh settings
    auto_refresh_seconds: int = 5
    max_data_points: int = 1000
    cache_ttl_seconds: int = 30
    
    # Feature flags
    enable_real_time: bool = True
    enable_predictions: bool = True
    enable_alpha_integration: bool = True
    enable_testing_integration: bool = True
    
    # Dashboard features
    enable_correlation_analysis: bool = True
    enable_trend_analysis: bool = True
    enable_anomaly_highlighting: bool = True
    enable_drill_down: bool = True
    
    # Performance thresholds for visualization
    performance_thresholds: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize default performance thresholds if not provided"""
        if self.performance_thresholds is None:
            self.performance_thresholds = {
                'response_time_ms': {'good': 50, 'warning': 100, 'critical': 200},
                'cpu_usage_percent': {'good': 70, 'warning': 80, 'critical': 90},
                'memory_usage_percent': {'good': 75, 'warning': 85, 'critical': 95},
                'cache_hit_ratio': {'good': 0.9, 'warning': 0.8, 'critical': 0.7},
                'error_rate': {'good': 0.01, 'warning': 0.05, 'critical': 0.1},
                'throughput_rps': {'good': 1000, 'warning': 500, 'critical': 100},
                'latency_p95_ms': {'good': 100, 'warning': 250, 'critical': 500},
                'disk_usage_percent': {'good': 80, 'warning': 90, 'critical': 95},
                'network_utilization': {'good': 0.7, 'warning': 0.85, 'critical': 0.95}
            }
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate basic settings
            if self.port < 1 or self.port > 65535:
                raise ValueError(f"Invalid port: {self.port}")
            
            if self.auto_refresh_seconds < 1:
                raise ValueError(f"Invalid refresh interval: {self.auto_refresh_seconds}")
            
            if self.max_data_points < 10:
                raise ValueError(f"Invalid max data points: {self.max_data_points}")
            
            if self.cache_ttl_seconds < 1:
                raise ValueError(f"Invalid cache TTL: {self.cache_ttl_seconds}")
            
            # Validate performance thresholds
            required_metrics = ['response_time_ms', 'cpu_usage_percent', 'memory_usage_percent']
            for metric in required_metrics:
                if metric not in self.performance_thresholds:
                    raise ValueError(f"Missing performance threshold for: {metric}")
                
                thresholds = self.performance_thresholds[metric]
                required_levels = ['good', 'warning', 'critical']
                for level in required_levels:
                    if level not in thresholds:
                        raise ValueError(f"Missing threshold level '{level}' for metric: {metric}")
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
    
    def get_threshold_level(self, metric_name: str, value: float) -> str:
        """
        Determine threshold level for a metric value
        
        Args:
            metric_name: Name of the metric
            value: Current value of the metric
            
        Returns:
            Threshold level: 'good', 'warning', or 'critical'
        """
        if metric_name not in self.performance_thresholds:
            return 'good'  # Default for unknown metrics
        
        thresholds = self.performance_thresholds[metric_name]
        
        # Handle metrics where lower is better (like response_time, error_rate)
        if metric_name in ['response_time_ms', 'error_rate', 'latency_p95_ms']:
            if value <= thresholds['good']:
                return 'good'
            elif value <= thresholds['warning']:
                return 'warning'
            else:
                return 'critical'
        
        # Handle metrics where higher is better (like cache_hit_ratio)
        elif metric_name in ['cache_hit_ratio']:
            if value >= thresholds['good']:
                return 'good'
            elif value >= thresholds['warning']:
                return 'warning'
            else:
                return 'critical'
        
        # Handle usage metrics (cpu, memory, disk) where higher is worse
        else:
            if value <= thresholds['good']:
                return 'good'
            elif value <= thresholds['warning']:
                return 'warning'
            else:
                return 'critical'
    
    def get_color_for_threshold(self, threshold_level: str) -> str:
        """
        Get color code for a threshold level
        
        Args:
            threshold_level: 'good', 'warning', or 'critical'
            
        Returns:
            Color code for dashboard visualization
        """
        colors = {
            'good': '#10b981',      # Green
            'warning': '#f59e0b',   # Yellow
            'critical': '#ef4444'   # Red
        }
        return colors.get(threshold_level, '#6b7280')  # Gray for unknown
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a specific feature is enabled
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            True if feature is enabled, False otherwise
        """
        feature_mapping = {
            'real_time': self.enable_real_time,
            'predictions': self.enable_predictions,
            'alpha_integration': self.enable_alpha_integration,
            'testing_integration': self.enable_testing_integration,
            'correlation_analysis': self.enable_correlation_analysis,
            'trend_analysis': self.enable_trend_analysis,
            'anomaly_highlighting': self.enable_anomaly_highlighting,
            'drill_down': self.enable_drill_down
        }
        return feature_mapping.get(feature_name, False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
            'auto_refresh_seconds': self.auto_refresh_seconds,
            'max_data_points': self.max_data_points,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'features': {
                'real_time': self.enable_real_time,
                'predictions': self.enable_predictions,
                'alpha_integration': self.enable_alpha_integration,
                'testing_integration': self.enable_testing_integration,
                'correlation_analysis': self.enable_correlation_analysis,
                'trend_analysis': self.enable_trend_analysis,
                'anomaly_highlighting': self.enable_anomaly_highlighting,
                'drill_down': self.enable_drill_down
            },
            'performance_thresholds': self.performance_thresholds
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DashboardConfig':
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            DashboardConfig instance
        """
        # Extract features if present
        features = config_dict.get('features', {})
        
        return cls(
            host=config_dict.get('host', 'localhost'),
            port=config_dict.get('port', 5001),
            debug=config_dict.get('debug', False),
            auto_refresh_seconds=config_dict.get('auto_refresh_seconds', 5),
            max_data_points=config_dict.get('max_data_points', 1000),
            cache_ttl_seconds=config_dict.get('cache_ttl_seconds', 30),
            enable_real_time=features.get('real_time', True),
            enable_predictions=features.get('predictions', True),
            enable_alpha_integration=features.get('alpha_integration', True),
            enable_testing_integration=features.get('testing_integration', True),
            enable_correlation_analysis=features.get('correlation_analysis', True),
            enable_trend_analysis=features.get('trend_analysis', True),
            enable_anomaly_highlighting=features.get('anomaly_highlighting', True),
            enable_drill_down=features.get('drill_down', True),
            performance_thresholds=config_dict.get('performance_thresholds')
        )


# Global configuration instance
_dashboard_config: DashboardConfig = None


def get_dashboard_config() -> DashboardConfig:
    """
    Get global dashboard configuration instance
    
    Returns:
        DashboardConfig instance
    """
    global _dashboard_config
    if _dashboard_config is None:
        _dashboard_config = DashboardConfig()
    return _dashboard_config


def set_dashboard_config(config: DashboardConfig):
    """
    Set global dashboard configuration instance
    
    Args:
        config: DashboardConfig instance to set as global
    """
    global _dashboard_config
    if config.validate_config():
        _dashboard_config = config
    else:
        raise ValueError("Invalid dashboard configuration")


def load_config_from_file(file_path: str) -> DashboardConfig:
    """
    Load configuration from file
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        DashboardConfig instance
    """
    import json
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return DashboardConfig.from_dict(config_dict)
    except Exception as e:
        logging.error(f"Failed to load config from {file_path}: {e}")
        return DashboardConfig()  # Return default config


def save_config_to_file(config: DashboardConfig, file_path: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: DashboardConfig instance to save
        file_path: Path to save configuration file
        
    Returns:
        True if successful, False otherwise
    """
    import json
    try:
        with open(file_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Failed to save config to {file_path}: {e}")
        return False