"""
Feature Flags System for TestMaster

Centralized feature flag management for toggleable enhancements.
All new features from the enhanced roadmap can be enabled/disabled
independently through this system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import threading
from functools import lru_cache


class FeatureFlags:
    """
    Centralized feature flag management for TestMaster enhancements.
    
    Provides toggle control for all enhanced features while maintaining
    backward compatibility with the original system.
    """
    
    _instance = None
    _lock = threading.Lock()
    _config = None
    _config_path = None
    _runtime_overrides = {}
    
    def __new__(cls):
        """Singleton pattern for feature flags."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, config_path: str = "testmaster_config.yaml"):
        """
        Initialize the feature flags system with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        cls._config_path = Path(config_path)
        cls._load_config()
    
    @classmethod
    def _load_config(cls):
        """Load configuration from file."""
        if cls._config_path and cls._config_path.exists():
            try:
                with open(cls._config_path, 'r') as f:
                    cls._config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading feature flags config: {e}")
                cls._config = cls._get_default_config()
        else:
            cls._config = cls._get_default_config()
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration with all features disabled."""
        return {
            'layers': {
                'layer1_test_foundation': {
                    'enabled': True,
                    'enhancements': {
                        'shared_state': {
                            'enabled': False,
                            'backend': 'memory'
                        },
                        'advanced_config': {
                            'enabled': False,
                            'hot_reload': False
                        },
                        'context_preservation': {
                            'enabled': False,
                            'deep_copy': True
                        },
                        'performance_monitoring': {
                            'enabled': False,
                            'include_memory': False
                        },
                        'streaming_generation': {
                            'enabled': False,
                            'buffer_size': 1024
                        },
                        'agent_qa': {
                            'enabled': False,
                            'similarity_threshold': 0.7
                        }
                    }
                },
                'layer2_monitoring': {
                    'enabled': True,
                    'enhancements': {
                        'graph_workflows': {
                            'enabled': False,
                            'max_parallel_branches': 4
                        },
                        'dynamic_handoff': {
                            'enabled': False,
                            'preserve_context': True
                        },
                        'async_processing': {
                            'enabled': False,
                            'max_workers': 4
                        },
                        'tracking_manager': {
                            'enabled': False,
                            'chain_depth': 5
                        },
                        'handoff_tools': {
                            'enabled': False,
                            'validation': True
                        }
                    }
                },
                'layer3_orchestration': {
                    'enabled': True,
                    'enhancements': {
                        'performance_dashboard': {
                            'enabled': False,
                            'port': 8080,
                            'auto_refresh': 5
                        },
                        'telemetry': {
                            'enabled': False,
                            'collectors': ['cpu', 'memory', 'api']
                        },
                        'flow_optimizer': {
                            'enabled': False,
                            'learning_rate': 0.1
                        },
                        'collaboration_matrix': {
                            'enabled': False,
                            'update_interval': 3600
                        },
                        'report_generator': {
                            'enabled': False,
                            'schedule': 'daily',
                            'formats': ['html', 'json']
                        }
                    }
                }
            }
        }
    
    @classmethod
    def is_enabled(cls, layer: str, enhancement: str) -> bool:
        """
        Check if a specific enhancement is enabled.
        
        Args:
            layer: Layer name (e.g., 'layer1_test_foundation')
            enhancement: Enhancement name (e.g., 'shared_state')
            
        Returns:
            True if the enhancement is enabled, False otherwise
        """
        # Check runtime overrides first
        override_key = f"{layer}.{enhancement}"
        if override_key in cls._runtime_overrides:
            return cls._runtime_overrides[override_key]
        
        # Check configuration
        if cls._config is None:
            cls.initialize()
        
        try:
            return (cls._config.get('layers', {})
                              .get(layer, {})
                              .get('enhancements', {})
                              .get(enhancement, {})
                              .get('enabled', False))
        except:
            return False
    
    @classmethod
    def get_config(cls, layer: str, enhancement: str) -> Dict[str, Any]:
        """
        Get configuration for a specific enhancement.
        
        Args:
            layer: Layer name
            enhancement: Enhancement name
            
        Returns:
            Configuration dictionary for the enhancement
        """
        if cls._config is None:
            cls.initialize()
        
        try:
            return (cls._config.get('layers', {})
                              .get(layer, {})
                              .get('enhancements', {})
                              .get(enhancement, {}))
        except:
            return {}
    
    @classmethod
    def enable(cls, layer: str, enhancement: str):
        """
        Enable a feature at runtime.
        
        Args:
            layer: Layer name
            enhancement: Enhancement name
        """
        override_key = f"{layer}.{enhancement}"
        cls._runtime_overrides[override_key] = True
        cls._clear_cache()
        print(f"Enabled: {layer}.{enhancement}")
    
    @classmethod
    def disable(cls, layer: str, enhancement: str):
        """
        Disable a feature at runtime.
        
        Args:
            layer: Layer name
            enhancement: Enhancement name
        """
        override_key = f"{layer}.{enhancement}"
        cls._runtime_overrides[override_key] = False
        cls._clear_cache()
        print(f"Disabled: {layer}.{enhancement}")
    
    @classmethod
    def set_feature(cls, layer: str, enhancement: str, enabled: bool):
        """
        Set feature state at runtime.
        
        Args:
            layer: Layer name
            enhancement: Enhancement name
            enabled: Whether to enable or disable
        """
        if enabled:
            cls.enable(layer, enhancement)
        else:
            cls.disable(layer, enhancement)
    
    @classmethod
    def get_all_features(cls) -> Dict[str, Dict[str, bool]]:
        """
        Get status of all features.
        
        Returns:
            Dictionary of layer -> enhancement -> enabled status
        """
        if cls._config is None:
            cls.initialize()
        
        result = {}
        for layer_name, layer_config in cls._config.get('layers', {}).items():
            result[layer_name] = {}
            for enhancement_name, enhancement_config in layer_config.get('enhancements', {}).items():
                result[layer_name][enhancement_name] = cls.is_enabled(layer_name, enhancement_name)
        
        return result
    
    @classmethod
    def show_status(cls):
        """Print status of all features."""
        features = cls.get_all_features()
        
        print("\n🎛️ TestMaster Feature Status")
        print("=" * 50)
        
        for layer, enhancements in features.items():
            print(f"\n📦 {layer}")
            for enhancement, enabled in enhancements.items():
                status = "✅" if enabled else "❌"
                print(f"  {status} {enhancement}")
    
    @classmethod
    def save_config(cls, path: str = None):
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration (uses default if None)
        """
        if path is None:
            path = cls._config_path
        
        if cls._config is None:
            cls.initialize()
        
        # Apply runtime overrides to config
        for override_key, enabled in cls._runtime_overrides.items():
            layer, enhancement = override_key.split('.')
            if layer in cls._config.get('layers', {}):
                if enhancement in cls._config['layers'][layer].get('enhancements', {}):
                    cls._config['layers'][layer]['enhancements'][enhancement]['enabled'] = enabled
        
        try:
            with open(path, 'w') as f:
                yaml.dump(cls._config, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to {path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    @classmethod
    def reload_config(cls):
        """Reload configuration from file."""
        cls._load_config()
        cls._runtime_overrides.clear()
        cls._clear_cache()
        print("Configuration reloaded")
    
    @classmethod
    def _clear_cache(cls):
        """Clear any cached values (for functions using lru_cache)."""
        pass
    
    @classmethod
    def enable_layer_enhancements(cls, layer: str):
        """
        Enable all enhancements for a specific layer.
        
        Args:
            layer: Layer name
        """
        if cls._config is None:
            cls.initialize()
        
        if layer in cls._config.get('layers', {}):
            for enhancement in cls._config['layers'][layer].get('enhancements', {}):
                cls.enable(layer, enhancement)
            print(f"Enabled all enhancements for {layer}")
    
    @classmethod
    def disable_layer_enhancements(cls, layer: str):
        """
        Disable all enhancements for a specific layer.
        
        Args:
            layer: Layer name
        """
        if cls._config is None:
            cls.initialize()
        
        if layer in cls._config.get('layers', {}):
            for enhancement in cls._config['layers'][layer].get('enhancements', {}):
                cls.disable(layer, enhancement)
            print(f"Disabled all enhancements for {layer}")


def feature_enabled(layer: str, enhancement: str):
    """
    Decorator to conditionally execute function based on feature flag.
    
    Args:
        layer: Layer name
        enhancement: Enhancement name
    
    Example:
        @feature_enabled('layer1_test_foundation', 'performance_monitoring')
        def monitor_performance():
            # This only runs if the feature is enabled
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if FeatureFlags.is_enabled(layer, enhancement):
                return func(*args, **kwargs)
            return None
        return wrapper
    return decorator


# Convenience functions for common checks
def is_shared_state_enabled() -> bool:
    """Check if shared state management is enabled."""
    return FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state')


def is_performance_monitoring_enabled() -> bool:
    """Check if performance monitoring is enabled."""
    return FeatureFlags.is_enabled('layer1_test_foundation', 'performance_monitoring')


def is_graph_workflows_enabled() -> bool:
    """Check if graph-based workflows are enabled."""
    return FeatureFlags.is_enabled('layer2_monitoring', 'graph_workflows')


def is_async_processing_enabled() -> bool:
    """Check if async processing is enabled."""
    return FeatureFlags.is_enabled('layer2_monitoring', 'async_processing')


def is_dashboard_enabled() -> bool:
    """Check if performance dashboard is enabled."""
    return FeatureFlags.is_enabled('layer3_orchestration', 'performance_dashboard')


def is_telemetry_enabled() -> bool:
    """Check if telemetry collection is enabled."""
    return FeatureFlags.is_enabled('layer3_orchestration', 'telemetry')