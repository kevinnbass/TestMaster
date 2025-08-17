"""
TestMaster Layer Management System

Provides toggleable layer architecture where each layer can be independently
enabled/disabled via configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .exceptions import TestMasterException


class ConfigError(TestMasterException):
    """Configuration-related errors"""
    pass


@dataclass
class LayerConfig:
    """Configuration for a single layer"""
    enabled: bool
    features: Dict[str, bool]
    requires: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerConfig':
        return cls(
            enabled=data.get('enabled', False),
            features=data.get('features', {}),
            requires=data.get('requires', [])
        )


class LayerManager:
    """
    Manages TestMaster's layered architecture with toggleable features.
    
    Layers:
    - Layer 1: Test Foundation (always enabled)
    - Layer 2: Monitoring & Communication (toggleable)
    - Layer 3: Orchestration (toggleable, requires Layer 2)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "testmaster_config.yaml"
        self.config = self._load_config()
        self._validate_dependencies()
    
    def _load_config(self) -> Dict[str, LayerConfig]:
        """Load configuration from YAML file or create default"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
                return {
                    layer_name: LayerConfig.from_dict(layer_data)
                    for layer_name, layer_data in data.get('layers', {}).items()
                }
        else:
            # Create default configuration
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, LayerConfig]:
        """Create default layer configuration"""
        return {
            'layer1_test_foundation': LayerConfig(
                enabled=True,  # Always enabled
                features={
                    'test_generation': True,
                    'test_verification': True,
                    'test_mapping': True,
                    'failure_detection': True
                },
                requires=[]
            ),
            'layer2_monitoring': LayerConfig(
                enabled=False,
                features={
                    'file_monitoring': True,
                    'idle_detection': True,
                    'claude_communication': True,
                    'dashboard_ui': False
                },
                requires=['layer1_test_foundation']
            ),
            'layer3_orchestration': LayerConfig(
                enabled=False,
                features={
                    'auto_tagging': True,
                    'work_distribution': True,
                    'smart_handoffs': True,
                    'codebase_intelligence': True
                },
                requires=['layer1_test_foundation', 'layer2_monitoring']
            )
        }
    
    def _validate_dependencies(self):
        """Validate that all layer dependencies can be satisfied"""
        for layer_name, layer_config in self.config.items():
            if not layer_config.enabled:
                continue
                
            for dep in layer_config.requires:
                if dep not in self.config:
                    raise ConfigError(f"Layer {layer_name} requires {dep}, but {dep} is not defined")
                if not self.config[dep].enabled:
                    raise ConfigError(f"Layer {layer_name} requires {dep} to be enabled")
    
    def is_enabled(self, layer: str, feature: Optional[str] = None) -> bool:
        """
        Check if a layer or specific feature is enabled
        
        Args:
            layer: Layer name (e.g., 'layer1_test_foundation')
            feature: Optional feature name (e.g., 'test_generation')
            
        Returns:
            True if layer/feature is enabled, False otherwise
        """
        if layer not in self.config:
            return False
            
        layer_config = self.config[layer]
        
        # Check if layer itself is enabled
        if not layer_config.enabled:
            return False
            
        # If checking a specific feature
        if feature:
            return layer_config.features.get(feature, False)
            
        return True
    
    def get_active_features(self) -> Dict[str, List[str]]:
        """Return all active features by layer"""
        active = {}
        for layer_name, layer_config in self.config.items():
            if layer_config.enabled:
                active[layer_name] = [
                    feature for feature, enabled in layer_config.features.items()
                    if enabled
                ]
        return active
    
    def enable_layer(self, layer: str, feature: Optional[str] = None):
        """Enable a layer or specific feature"""
        if layer not in self.config:
            raise ConfigError(f"Unknown layer: {layer}")
            
        if feature:
            # Enable specific feature
            self.config[layer].features[feature] = True
        else:
            # Enable entire layer
            self.config[layer].enabled = True
            
        # Re-validate dependencies
        self._validate_dependencies()
    
    def disable_layer(self, layer: str, feature: Optional[str] = None):
        """Disable a layer or specific feature"""
        if layer not in self.config:
            raise ConfigError(f"Unknown layer: {layer}")
            
        # Don't allow disabling Layer 1
        if layer == 'layer1_test_foundation' and not feature:
            raise ConfigError("Layer 1 (test foundation) cannot be disabled")
            
        if feature:
            # Disable specific feature
            self.config[layer].features[feature] = False
        else:
            # Disable entire layer
            self.config[layer].enabled = False
            
            # Disable dependent layers
            self._disable_dependent_layers(layer)
    
    def _disable_dependent_layers(self, disabled_layer: str):
        """Disable layers that depend on the disabled layer"""
        for layer_name, layer_config in self.config.items():
            if disabled_layer in layer_config.requires and layer_config.enabled:
                layer_config.enabled = False
                # Recursively disable dependents
                self._disable_dependent_layers(layer_name)
    
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            'layers': {
                layer_name: {
                    'enabled': layer_config.enabled,
                    'features': layer_config.features,
                    'requires': layer_config.requires
                }
                for layer_name, layer_config in self.config.items()
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all layers and features"""
        status = {}
        for layer_name, layer_config in self.config.items():
            layer_status = {
                'enabled': layer_config.enabled,
                'features': {}
            }
            
            for feature, enabled in layer_config.features.items():
                layer_status['features'][feature] = {
                    'enabled': enabled and layer_config.enabled,
                    'available': layer_config.enabled
                }
            
            status[layer_name] = layer_status
        
        return status
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check for dependency cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(layer: str) -> bool:
            if layer in rec_stack:
                return True
            if layer in visited:
                return False
                
            visited.add(layer)
            rec_stack.add(layer)
            
            if layer in self.config:
                for dep in self.config[layer].requires:
                    if has_cycle(dep):
                        return True
            
            rec_stack.remove(layer)
            return False
        
        for layer in self.config:
            if has_cycle(layer):
                issues.append(f"Dependency cycle detected involving {layer}")
        
        # Check for missing dependencies
        for layer_name, layer_config in self.config.items():
            for dep in layer_config.requires:
                if dep not in self.config:
                    issues.append(f"Layer {layer_name} requires undefined layer {dep}")
        
        return issues


# Global layer manager instance
_layer_manager: Optional[LayerManager] = None


def get_layer_manager() -> LayerManager:
    """Get the global layer manager instance"""
    global _layer_manager
    if _layer_manager is None:
        _layer_manager = LayerManager()
    return _layer_manager


def is_layer_enabled(layer: str, feature: Optional[str] = None) -> bool:
    """Convenience function to check if a layer/feature is enabled"""
    return get_layer_manager().is_enabled(layer, feature)


# Decorator for layer-conditional functionality
def requires_layer(layer: str, feature: Optional[str] = None):
    """Decorator to require a specific layer/feature to be enabled"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_layer_enabled(layer, feature):
                feature_desc = f".{feature}" if feature else ""
                raise ConfigError(f"Function {func.__name__} requires {layer}{feature_desc} to be enabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator