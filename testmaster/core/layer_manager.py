"""
TestMaster Layer Management System

Provides toggleable layer architecture where each layer can be independently
enabled/disabled via configuration.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
from .exceptions import TestMasterException
from .feature_flags import FeatureFlags
from .shared_state import get_shared_state
from .monitoring_decorators import monitor_performance


class ConfigError(TestMasterException):
    """Configuration-related errors"""
    pass


@dataclass
class LayerConfig:
    """Configuration for a single layer"""
    enabled: bool
    features: Dict[str, bool]
    requires: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    last_modified: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerConfig':
        return cls(
            enabled=data.get('enabled', False),
            features=data.get('features', {}),
            requires=data.get('requires', []),
            metadata=data.get('metadata', {}),
            version=data.get('version', '1.0.0'),
            last_modified=datetime.fromisoformat(data['last_modified']) if 'last_modified' in data else None
        )


class LayerManager:
    """
    Manages TestMaster's layered architecture with toggleable features.
    
    Enhanced with advanced configuration management:
    - Hot-reload configuration changes
    - Environment variable overrides
    - Schema validation
    - Configuration inheritance
    - Audit trail
    
    Layers:
    - Layer 1: Test Foundation (always enabled)
    - Layer 2: Monitoring & Communication (toggleable)
    - Layer 3: Orchestration (toggleable, requires Layer 2)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "testmaster_config.yaml"
        self.config = self._load_config()
        self._validate_dependencies()
        self._lock = threading.RLock()
        self._callbacks: List[Callable] = []
        self._audit_trail: List[Dict[str, Any]] = []
        
        # NEW: Setup advanced configuration if enabled
        if FeatureFlags.is_enabled('layer1_test_foundation', 'advanced_config'):
            self._setup_advanced_config()
            print("✅ Advanced configuration management enabled")
        else:
            self._hot_reload_enabled = False
            self._observer = None
            self._schema_validator = None
    
    @monitor_performance(name="config_load")
    def _load_config(self) -> Dict[str, LayerConfig]:
        """Load configuration from YAML file or create default"""
        config_file = Path(self.config_path)
        config_data = None
        
        # Try multiple config sources
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        
        # NEW: Check for JSON config as fallback
        elif Path(self.config_path.replace('.yaml', '.json')).exists():
            with open(self.config_path.replace('.yaml', '.json'), 'r') as f:
                config_data = json.load(f)
        
        if config_data:
            config = {
                layer_name: LayerConfig.from_dict(layer_data)
                for layer_name, layer_data in config_data.get('layers', {}).items()
            }
            
            # NEW: Apply environment variable overrides
            if FeatureFlags.is_enabled('layer1_test_foundation', 'advanced_config'):
                config = self._apply_env_overrides(config)
            
            return config
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
    
    @monitor_performance(name="config_save")
    def save_config(self):
        """Save current configuration to file"""
        with self._lock:
            # Update timestamps
            for layer_config in self.config.values():
                layer_config.last_modified = datetime.now()
            
            config_data = {
                'version': '2.0.0',
                'generated_at': datetime.now().isoformat(),
                'layers': {
                    layer_name: {
                        'enabled': layer_config.enabled,
                        'features': layer_config.features,
                        'requires': layer_config.requires,
                        'metadata': layer_config.metadata,
                        'version': layer_config.version,
                        'last_modified': layer_config.last_modified.isoformat() if layer_config.last_modified else None
                    }
                    for layer_name, layer_config in self.config.items()
                }
            }
            
            # Save as YAML
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            # NEW: Also save as JSON backup if advanced config enabled
            if FeatureFlags.is_enabled('layer1_test_foundation', 'advanced_config'):
                json_path = self.config_path.replace('.yaml', '.json')
                with open(json_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            
            # Record audit trail
            self._add_audit_entry('config_saved', {'path': self.config_path})
            
            # Notify callbacks
            self._notify_callbacks('save', config_data)
    
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
    
    def _setup_advanced_config(self):
        """Setup advanced configuration features."""
        config = FeatureFlags.get_config('layer1_test_foundation', 'advanced_config')
        
        # Setup hot-reload if enabled
        if config.get('hot_reload', False):
            self._setup_hot_reload()
        else:
            self._hot_reload_enabled = False
            self._observer = None
        
        # Setup schema validator
        self._setup_schema_validator()
        
        # Setup configuration inheritance
        self._merge_inherited_configs()
        
        # Initialize shared state integration
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self._sync_with_shared_state()
    
    def _setup_hot_reload(self):
        """Setup hot-reload for configuration changes."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class ConfigReloadHandler(FileSystemEventHandler):
                def __init__(self, layer_manager):
                    self.layer_manager = layer_manager
                
                def on_modified(self, event):
                    if event.src_path.endswith(('.yaml', '.yml', '.json')):
                        print(f"🔄 Config file changed: {event.src_path}")
                        self.layer_manager.reload_config()
            
            self._observer = Observer()
            handler = ConfigReloadHandler(self)
            self._observer.schedule(handler, str(Path(self.config_path).parent), recursive=False)
            self._observer.start()
            self._hot_reload_enabled = True
            print("🔥 Hot-reload enabled for configuration")
            
        except ImportError:
            print("⚠️ watchdog not installed, hot-reload disabled")
            self._hot_reload_enabled = False
            self._observer = None
    
    def _setup_schema_validator(self):
        """Setup schema validation for configuration."""
        self._schema = {
            'type': 'object',
            'properties': {
                'layers': {
                    'type': 'object',
                    'patternProperties': {
                        '^layer\\d+_.*$': {
                            'type': 'object',
                            'properties': {
                                'enabled': {'type': 'boolean'},
                                'features': {'type': 'object'},
                                'requires': {'type': 'array', 'items': {'type': 'string'}},
                                'metadata': {'type': 'object'},
                                'version': {'type': 'string'}
                            },
                            'required': ['enabled', 'features']
                        }
                    }
                }
            }
        }
        
        try:
            import jsonschema
            self._schema_validator = jsonschema.Draft7Validator(self._schema)
        except ImportError:
            self._schema_validator = None
    
    def _apply_env_overrides(self, config: Dict[str, LayerConfig]) -> Dict[str, LayerConfig]:
        """Apply environment variable overrides to configuration."""
        # Format: TESTMASTER_LAYER2_MONITORING_ENABLED=true
        #         TESTMASTER_LAYER3_ORCHESTRATION_FEATURE_AUTO_TAGGING=false
        
        for key, value in os.environ.items():
            if key.startswith('TESTMASTER_'):
                parts = key[11:].lower().split('_')  # Remove TESTMASTER_ prefix
                
                if len(parts) >= 3:  # At least layer_name_setting
                    layer_name = '_'.join(parts[:2])  # e.g., layer2_monitoring
                    
                    if layer_name in config:
                        if parts[2] == 'enabled':
                            config[layer_name].enabled = value.lower() == 'true'
                            print(f"🌍 ENV override: {layer_name}.enabled = {value}")
                        elif parts[2] == 'feature' and len(parts) >= 4:
                            feature_name = '_'.join(parts[3:])
                            config[layer_name].features[feature_name] = value.lower() == 'true'
                            print(f"🌍 ENV override: {layer_name}.features.{feature_name} = {value}")
        
        return config
    
    def _merge_inherited_configs(self):
        """Merge configuration from parent config files."""
        # Check for parent config files
        parent_configs = [
            'testmaster_base.yaml',
            'testmaster_defaults.yaml',
            '.testmaster.yaml'
        ]
        
        for parent_path in parent_configs:
            if Path(parent_path).exists():
                try:
                    with open(parent_path, 'r') as f:
                        parent_data = yaml.safe_load(f)
                    
                    # Merge with current config (current config takes precedence)
                    for layer_name, layer_data in parent_data.get('layers', {}).items():
                        if layer_name not in self.config:
                            self.config[layer_name] = LayerConfig.from_dict(layer_data)
                            print(f"📥 Inherited layer {layer_name} from {parent_path}")
                        else:
                            # Merge features
                            parent_features = layer_data.get('features', {})
                            for feature, enabled in parent_features.items():
                                if feature not in self.config[layer_name].features:
                                    self.config[layer_name].features[feature] = enabled
                                    print(f"📥 Inherited feature {layer_name}.{feature} from {parent_path}")
                except Exception as e:
                    print(f"⚠️ Error loading parent config {parent_path}: {e}")
    
    def _sync_with_shared_state(self):
        """Sync configuration with shared state."""
        shared_state = get_shared_state()
        
        # Save current config to shared state
        config_dict = {
            layer_name: {
                'enabled': layer_config.enabled,
                'features': layer_config.features
            }
            for layer_name, layer_config in self.config.items()
        }
        
        shared_state.set('layer_config', config_dict, ttl=3600)
        
        # Check for remote config updates
        remote_config = shared_state.get('layer_config_override')
        if remote_config:
            print("🌐 Applying remote configuration overrides")
            for layer_name, overrides in remote_config.items():
                if layer_name in self.config:
                    if 'enabled' in overrides:
                        self.config[layer_name].enabled = overrides['enabled']
                    if 'features' in overrides:
                        self.config[layer_name].features.update(overrides['features'])
    
    def reload_config(self):
        """Reload configuration from file."""
        with self._lock:
            old_config = self.config
            try:
                self.config = self._load_config()
                self._validate_dependencies()
                
                # Detect changes
                changes = self._detect_config_changes(old_config, self.config)
                if changes:
                    print(f"🔄 Configuration reloaded with {len(changes)} changes")
                    for change in changes:
                        print(f"   - {change}")
                    
                    # Record audit trail
                    self._add_audit_entry('config_reloaded', {'changes': changes})
                    
                    # Notify callbacks
                    self._notify_callbacks('reload', changes)
                    
            except Exception as e:
                print(f"⚠️ Error reloading config: {e}")
                self.config = old_config  # Restore old config
    
    def _detect_config_changes(self, old_config: Dict[str, LayerConfig], 
                               new_config: Dict[str, LayerConfig]) -> List[str]:
        """Detect changes between configurations."""
        changes = []
        
        # Check for added/removed layers
        old_layers = set(old_config.keys())
        new_layers = set(new_config.keys())
        
        for layer in new_layers - old_layers:
            changes.append(f"Added layer: {layer}")
        
        for layer in old_layers - new_layers:
            changes.append(f"Removed layer: {layer}")
        
        # Check for changed layers
        for layer in old_layers & new_layers:
            old_layer = old_config[layer]
            new_layer = new_config[layer]
            
            if old_layer.enabled != new_layer.enabled:
                changes.append(f"{layer}.enabled: {old_layer.enabled} → {new_layer.enabled}")
            
            # Check features
            for feature in set(old_layer.features.keys()) | set(new_layer.features.keys()):
                old_val = old_layer.features.get(feature)
                new_val = new_layer.features.get(feature)
                if old_val != new_val:
                    changes.append(f"{layer}.features.{feature}: {old_val} → {new_val}")
        
        return changes
    
    def register_callback(self, callback: Callable[[str, Any], None]):
        """Register a callback for configuration changes."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, event: str, data: Any):
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                print(f"⚠️ Error in config callback: {e}")
    
    def _add_audit_entry(self, action: str, details: Dict[str, Any]):
        """Add entry to audit trail."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        
        self._audit_trail.append(entry)
        
        # Keep only last 100 entries
        if len(self._audit_trail) > 100:
            self._audit_trail = self._audit_trail[-100:]
        
        # Save to shared state if enabled
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            shared_state = get_shared_state()
            shared_state.append('config_audit_trail', entry)
    
    def get_audit_trail(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent audit trail entries."""
        return self._audit_trail[-limit:]
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # NEW: Schema validation if enabled
        if self._schema_validator:
            config_dict = {
                'layers': {
                    layer_name: {
                        'enabled': layer_config.enabled,
                        'features': layer_config.features,
                        'requires': layer_config.requires
                    }
                    for layer_name, layer_config in self.config.items()
                }
            }
            
            for error in self._schema_validator.iter_errors(config_dict):
                issues.append(f"Schema validation error: {error.message}")
        
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


def export_config(output_path: str = "config_export.json"):
    """Export current configuration to JSON."""
    manager = get_layer_manager()
    config_data = {
        'exported_at': datetime.now().isoformat(),
        'status': manager.get_status(),
        'active_features': manager.get_active_features(),
        'audit_trail': manager.get_audit_trail(20)
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)
    
    print(f"💾 Configuration exported to {output_path}")


def import_config(input_path: str):
    """Import configuration from JSON."""
    manager = get_layer_manager()
    
    with open(input_path, 'r') as f:
        config_data = json.load(f)
    
    # Apply imported configuration
    if 'layers' in config_data:
        for layer_name, layer_data in config_data['layers'].items():
            if layer_name in manager.config:
                manager.config[layer_name].enabled = layer_data.get('enabled', False)
                manager.config[layer_name].features.update(layer_data.get('features', {}))
    
    manager.save_config()
    print(f"📥 Configuration imported from {input_path}")