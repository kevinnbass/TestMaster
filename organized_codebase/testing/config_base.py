"""
Configuration Base Classes
==========================

Core configuration abstractions providing the foundation for
hierarchical configuration management across all TestMaster layers.

Author: Agent E - Infrastructure Consolidation
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Type
from pathlib import Path
from enum import Enum


logger = logging.getLogger(__name__)


class ConfigurationLayer(Enum):
    """Configuration hierarchy layers."""
    FOUNDATION = "foundation"
    DOMAIN = "domain"
    ORCHESTRATION = "orchestration"
    SERVICES = "services"
    UNIFIED = "unified"


class ConfigurationScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    LAYER = "layer"
    DOMAIN = "domain"
    MODULE = "module"
    INSTANCE = "instance"


@dataclass
class ConfigurationMetadata:
    """Metadata for configuration entries."""
    layer: ConfigurationLayer
    scope: ConfigurationScope
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ConfigurationBase(ABC):
    """
    Abstract base class for all configuration management.
    
    Provides hierarchical configuration inheritance, validation,
    and coordination across the TestMaster architecture layers.
    """
    
    def __init__(
        self,
        layer: ConfigurationLayer,
        scope: ConfigurationScope = ConfigurationScope.LAYER,
        parent: Optional['ConfigurationBase'] = None
    ):
        self.layer = layer
        self.scope = scope
        self.parent = parent
        self.children: List['ConfigurationBase'] = []
        self.metadata = ConfigurationMetadata(layer=layer, scope=scope, source=self.__class__.__name__)
        
        # Configuration storage
        self._config_data: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}
        self._computed_cache: Dict[str, Any] = {}
        self._cache_valid = False
        
        # Validation and lifecycle
        self._validators: List[callable] = []
        self._change_listeners: List[callable] = []
        
        # Register with parent if provided
        if parent:
            parent.add_child(self)
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.debug(f"Configuration initialized: {self.layer.value}/{self.scope.value}")
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this layer/scope."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation."""
        pass
    
    def add_child(self, child: 'ConfigurationBase'):
        """Add child configuration."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            self._invalidate_cache()
    
    def remove_child(self, child: 'ConfigurationBase'):
        """Remove child configuration."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            self._invalidate_cache()
    
    def set(self, key: str, value: Any, scope: ConfigurationScope = None):
        """Set configuration value."""
        scope = scope or self.scope
        
        # Validate the value
        if not self._validate_value(key, value):
            raise ValueError(f"Invalid configuration value for {key}: {value}")
        
        # Store in appropriate scope
        if scope == ConfigurationScope.INSTANCE:
            self._overrides[key] = value
        else:
            self._config_data[key] = value
        
        self._invalidate_cache()
        self._notify_change(key, value)
        
        self.logger.debug(f"Configuration set: {key}={value} (scope: {scope.value})")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with hierarchical inheritance."""
        # Check cache first
        if self._cache_valid and key in self._computed_cache:
            return self._computed_cache[key]
        
        # Check instance overrides first
        if key in self._overrides:
            value = self._overrides[key]
        # Check local configuration
        elif key in self._config_data:
            value = self._config_data[key]
        # Check parent hierarchy
        elif self.parent:
            value = self.parent.get(key, default)
        # Use default
        else:
            value = default
        
        # Cache the result
        self._computed_cache[key] = value
        return value
    
    def update(self, config_dict: Dict[str, Any], scope: ConfigurationScope = None):
        """Update multiple configuration values."""
        scope = scope or self.scope
        
        for key, value in config_dict.items():
            self.set(key, value, scope)
    
    def merge_from_parent(self):
        """Merge configuration from parent hierarchy."""
        if not self.parent:
            return
        
        # Get parent's computed configuration
        parent_config = self.parent.to_dict()
        
        # Merge with local configuration (local takes precedence)
        merged_config = {**parent_config, **self._config_data}
        self._config_data = merged_config
        
        self._invalidate_cache()
        self.logger.debug("Configuration merged from parent")
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Ensure cache is valid
        self._compute_final_config()
        
        config_dict = {**self._computed_cache}
        
        if include_metadata:
            config_dict['_metadata'] = {
                'layer': self.layer.value,
                'scope': self.scope.value,
                'source': self.metadata.source,
                'created_at': self.metadata.created_at.isoformat(),
                'version': self.metadata.version,
                'description': self.metadata.description,
                'tags': self.metadata.tags
            }
        
        return config_dict
    
    def validate(self) -> List[str]:
        """Validate current configuration."""
        errors = []
        
        # Run custom validators
        for validator in self._validators:
            try:
                validator_errors = validator(self.to_dict())
                if validator_errors:
                    errors.extend(validator_errors)
            except Exception as e:
                errors.append(f"Validator error: {str(e)}")
        
        # Validate against schema
        schema_errors = self._validate_against_schema()
        errors.extend(schema_errors)
        
        return errors
    
    def add_validator(self, validator: callable):
        """Add custom configuration validator."""
        self._validators.append(validator)
    
    def add_change_listener(self, listener: callable):
        """Add configuration change listener."""
        self._change_listeners.append(listener)
    
    def reload(self):
        """Reload configuration from sources."""
        # Subclasses should override to implement specific reloading logic
        self._invalidate_cache()
        self.logger.info("Configuration reloaded")
    
    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Get information about configuration hierarchy."""
        return {
            'layer': self.layer.value,
            'scope': self.scope.value,
            'has_parent': self.parent is not None,
            'parent_layer': self.parent.layer.value if self.parent else None,
            'child_count': len(self.children),
            'child_layers': [child.layer.value for child in self.children],
            'inheritance_depth': self._get_inheritance_depth()
        }
    
    def _compute_final_config(self):
        """Compute final configuration with inheritance."""
        if self._cache_valid:
            return
        
        # Start with default configuration
        final_config = self.get_default_config().copy()
        
        # Apply parent configuration
        if self.parent:
            parent_config = self.parent.to_dict()
            final_config.update(parent_config)
        
        # Apply local configuration
        final_config.update(self._config_data)
        
        # Apply instance overrides
        final_config.update(self._overrides)
        
        # Update cache
        self._computed_cache = final_config
        self._cache_valid = True
    
    def _invalidate_cache(self):
        """Invalidate configuration cache."""
        self._cache_valid = False
        self._computed_cache.clear()
        
        # Invalidate children's cache as well
        for child in self.children:
            child._invalidate_cache()
    
    def _validate_value(self, key: str, value: Any) -> bool:
        """Validate individual configuration value."""
        # Basic validation - subclasses can override for specific validation
        return True
    
    def _validate_against_schema(self) -> List[str]:
        """Validate configuration against schema."""
        # Basic schema validation - subclasses should implement specific validation
        return []
    
    def _notify_change(self, key: str, value: Any):
        """Notify change listeners."""
        for listener in self._change_listeners:
            try:
                listener(key, value, self)
            except Exception as e:
                self.logger.warning(f"Change listener error: {e}")
    
    def _get_inheritance_depth(self) -> int:
        """Get depth in inheritance hierarchy."""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.layer.value}/{self.scope.value})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(layer={self.layer.value}, "
                f"scope={self.scope.value}, children={len(self.children)})")


class LayeredConfiguration(ConfigurationBase):
    """
    Base class for layer-specific configurations.
    
    Provides specialized functionality for each architectural layer
    while maintaining hierarchical inheritance.
    """
    
    def __init__(
        self,
        layer: ConfigurationLayer,
        domain: Optional[str] = None,
        parent: Optional[ConfigurationBase] = None
    ):
        super().__init__(layer, ConfigurationScope.LAYER, parent)
        self.domain = domain
        self.metadata.tags.append(f"layer:{layer.value}")
        if domain:
            self.metadata.tags.append(f"domain:{domain}")
    
    def get_layer_config(self) -> Dict[str, Any]:
        """Get layer-specific configuration."""
        return self.to_dict()
    
    def merge_domain_config(self, domain_config: Dict[str, Any]):
        """Merge domain-specific configuration."""
        self.update(domain_config)
        self.metadata.tags.append("domain_merged")
        self.logger.info(f"Merged domain configuration for {self.domain}")


# Export key classes
__all__ = [
    'ConfigurationLayer',
    'ConfigurationScope', 
    'ConfigurationMetadata',
    'ConfigurationBase',
    'LayeredConfiguration'
]