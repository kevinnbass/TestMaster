"""
Hierarchical Configuration Coordinator
=====================================

Coordinates hierarchical configuration management with the existing
unified configuration system, providing seamless integration between
the 4-tier architecture and legacy configuration.

Author: Agent E - Infrastructure Consolidation
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Import hierarchical configuration foundation
from ..core.foundation.configuration.base.config_base import (
    ConfigurationBase,
    LayeredConfiguration,
    ConfigurationLayer,
    ConfigurationScope
)

# Import orchestration configuration
from ..core.orchestration.configuration.orchestration_config import OrchestrationConfiguration

# Import existing unified configuration
from .enhanced_unified_config import EnhancedConfigManager, ConfigCategory, Environment


logger = logging.getLogger(__name__)


class HierarchicalConfigurationCoordinator:
    """
    Coordinates hierarchical configuration management across all layers.
    
    Provides unified access to:
    - Foundation layer configuration (base abstractions)
    - Domain layer configuration (intelligence, security, testing, etc.)
    - Orchestration layer configuration (workflow, swarm, integration)
    - Services layer configuration (API, analytics, enterprise)
    - Legacy unified configuration (backward compatibility)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize hierarchical configuration layers
        self._foundation_config = None
        self._domain_configs: Dict[str, LayeredConfiguration] = {}
        self._orchestration_config = None
        self._services_configs: Dict[str, LayeredConfiguration] = {}
        
        # Initialize legacy unified configuration
        self._unified_config_manager = EnhancedConfigManager()
        
        # Configuration coordination state
        self._coordination_active = False
        self._layer_dependencies = self._build_layer_dependencies()
        
        # Initialize hierarchical structure
        self._initialize_hierarchical_configuration()
        
        self.logger.info("Hierarchical Configuration Coordinator initialized")
    
    def _initialize_hierarchical_configuration(self):
        """Initialize the hierarchical configuration structure."""
        try:
            # Initialize foundation layer (root of hierarchy)
            self._foundation_config = FoundationConfiguration()
            
            # Initialize orchestration layer with foundation as parent
            self._orchestration_config = OrchestrationConfiguration(parent=self._foundation_config)
            
            # Initialize domain configurations
            self._initialize_domain_configurations()
            
            # Initialize services configurations  
            self._initialize_services_configurations()
            
            # Integrate with legacy unified configuration
            self._integrate_with_unified_config()
            
            self._coordination_active = True
            self.logger.info("Hierarchical configuration structure initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hierarchical configuration: {e}")
            raise
    
    def _initialize_domain_configurations(self):
        """Initialize domain-specific configurations."""
        # Intelligence domain configuration
        intelligence_config = DomainConfiguration(
            layer=ConfigurationLayer.DOMAIN,
            domain="intelligence",
            parent=self._foundation_config
        )
        self._domain_configs["intelligence"] = intelligence_config
        
        # Security domain configuration
        security_config = DomainConfiguration(
            layer=ConfigurationLayer.DOMAIN,
            domain="security", 
            parent=self._foundation_config
        )
        self._domain_configs["security"] = security_config
        
        # Testing domain configuration
        testing_config = DomainConfiguration(
            layer=ConfigurationLayer.DOMAIN,
            domain="testing",
            parent=self._foundation_config
        )
        self._domain_configs["testing"] = testing_config
        
        # Coordination domain configuration
        coordination_config = DomainConfiguration(
            layer=ConfigurationLayer.DOMAIN,
            domain="coordination",
            parent=self._foundation_config
        )
        self._domain_configs["coordination"] = coordination_config
    
    def _initialize_services_configurations(self):
        """Initialize services layer configurations."""
        # API services configuration
        api_config = ServicesConfiguration(
            service_type="api",
            parent=self._orchestration_config
        )
        self._services_configs["api"] = api_config
        
        # Analytics services configuration
        analytics_config = ServicesConfiguration(
            service_type="analytics",
            parent=self._orchestration_config
        )
        self._services_configs["analytics"] = analytics_config
        
        # Enterprise services configuration
        enterprise_config = ServicesConfiguration(
            service_type="enterprise",
            parent=self._orchestration_config
        )
        self._services_configs["enterprise"] = enterprise_config
    
    def _integrate_with_unified_config(self):
        """Integrate hierarchical configuration with legacy unified config."""
        try:
            # Get current unified configuration
            unified_config = self._unified_config_manager.get_full_config()
            
            # Map unified config categories to hierarchical layers
            category_mappings = {
                ConfigCategory.API: ("services", "api"),
                ConfigCategory.GENERATION: ("domain", "intelligence"),
                ConfigCategory.MONITORING: ("orchestration", None),
                ConfigCategory.EXECUTION: ("orchestration", None),
                ConfigCategory.CACHING: ("foundation", None),
                ConfigCategory.REPORTING: ("services", "analytics"),
                ConfigCategory.QUALITY: ("domain", "testing")
            }
            
            # Apply unified config to hierarchical structure
            for category, config_data in unified_config.items():
                if hasattr(ConfigCategory, category.upper()):
                    config_category = ConfigCategory(category)
                    layer_info = category_mappings.get(config_category)
                    
                    if layer_info:
                        layer_type, domain = layer_info
                        self._apply_config_to_layer(layer_type, domain, config_data)
            
            self.logger.info("Integrated hierarchical configuration with unified config")
            
        except Exception as e:
            self.logger.warning(f"Failed to integrate with unified config: {e}")
    
    def _apply_config_to_layer(self, layer_type: str, domain: Optional[str], config_data: Dict[str, Any]):
        """Apply configuration data to specific layer."""
        try:
            if layer_type == "foundation":
                self._foundation_config.update(config_data)
            elif layer_type == "domain" and domain in self._domain_configs:
                self._domain_configs[domain].update(config_data)
            elif layer_type == "orchestration":
                self._orchestration_config.update(config_data)
            elif layer_type == "services" and domain in self._services_configs:
                self._services_configs[domain].update(config_data)
                
        except Exception as e:
            self.logger.warning(f"Failed to apply config to {layer_type}/{domain}: {e}")
    
    def get_config(self, layer: ConfigurationLayer, domain: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for specific layer and domain."""
        if layer == ConfigurationLayer.FOUNDATION:
            return self._foundation_config.to_dict() if self._foundation_config else {}
        elif layer == ConfigurationLayer.DOMAIN and domain in self._domain_configs:
            return self._domain_configs[domain].to_dict()
        elif layer == ConfigurationLayer.ORCHESTRATION:
            return self._orchestration_config.to_dict() if self._orchestration_config else {}
        elif layer == ConfigurationLayer.SERVICES and domain in self._services_configs:
            return self._services_configs[domain].to_dict()
        else:
            return {}
    
    def get_unified_config(self) -> Dict[str, Any]:
        """Get complete unified configuration from all layers."""
        unified_config = {}
        
        # Foundation layer configuration
        if self._foundation_config:
            unified_config['foundation'] = self._foundation_config.to_dict()
        
        # Domain configurations
        for domain, config in self._domain_configs.items():
            unified_config[f'domain_{domain}'] = config.to_dict()
        
        # Orchestration configuration
        if self._orchestration_config:
            unified_config['orchestration'] = self._orchestration_config.to_dict()
        
        # Services configurations
        for service, config in self._services_configs.items():
            unified_config[f'services_{service}'] = config.to_dict()
        
        # Legacy unified configuration for backward compatibility
        unified_config['legacy'] = self._unified_config_manager.get_full_config()
        
        return unified_config
    
    def set_config(self, layer: ConfigurationLayer, key: str, value: Any, domain: Optional[str] = None):
        """Set configuration value in specific layer."""
        if layer == ConfigurationLayer.FOUNDATION and self._foundation_config:
            self._foundation_config.set(key, value)
        elif layer == ConfigurationLayer.DOMAIN and domain in self._domain_configs:
            self._domain_configs[domain].set(key, value)
        elif layer == ConfigurationLayer.ORCHESTRATION and self._orchestration_config:
            self._orchestration_config.set(key, value)
        elif layer == ConfigurationLayer.SERVICES and domain in self._services_configs:
            self._services_configs[domain].set(key, value)
        else:
            raise ValueError(f"Invalid layer/domain combination: {layer.value}/{domain}")
    
    def validate_all_configurations(self) -> Dict[str, List[str]]:
        """Validate all configurations across all layers."""
        validation_results = {}
        
        # Validate foundation layer
        if self._foundation_config:
            foundation_errors = self._foundation_config.validate()
            if foundation_errors:
                validation_results['foundation'] = foundation_errors
        
        # Validate domain configurations
        for domain, config in self._domain_configs.items():
            domain_errors = config.validate()
            if domain_errors:
                validation_results[f'domain_{domain}'] = domain_errors
        
        # Validate orchestration configuration
        if self._orchestration_config:
            orchestration_errors = self._orchestration_config.validate()
            if orchestration_errors:
                validation_results['orchestration'] = orchestration_errors
        
        # Validate services configurations
        for service, config in self._services_configs.items():
            service_errors = config.validate()
            if service_errors:
                validation_results[f'services_{service}'] = service_errors
        
        return validation_results
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get status of hierarchical configuration coordination."""
        return {
            'coordination_active': self._coordination_active,
            'foundation_config_initialized': self._foundation_config is not None,
            'domain_configs_count': len(self._domain_configs),
            'orchestration_config_initialized': self._orchestration_config is not None,
            'services_configs_count': len(self._services_configs),
            'unified_config_integrated': True,
            'layer_dependencies': self._layer_dependencies,
            'validation_status': len(self.validate_all_configurations()) == 0
        }
    
    def _build_layer_dependencies(self) -> Dict[str, List[str]]:
        """Build layer dependency mapping."""
        return {
            'foundation': [],  # Foundation has no dependencies
            'domain': ['foundation'],  # Domains depend on foundation
            'orchestration': ['foundation', 'domain'],  # Orchestration depends on foundation and domains
            'services': ['foundation', 'domain', 'orchestration']  # Services depend on all lower layers
        }


class FoundationConfiguration(LayeredConfiguration):
    """Foundation layer configuration."""
    
    def __init__(self):
        super().__init__(ConfigurationLayer.FOUNDATION)
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'paths': {
                'base_dir': str(Path.cwd()),
                'config_dir': 'config',
                'cache_dir': 'cache',
                'logs_dir': 'logs'
            },
            'performance': {
                'max_workers': 4,
                'timeout_seconds': 30,
                'memory_limit_mb': 1024
            }
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'required': ['logging', 'paths'],
            'properties': {
                'logging': {'type': 'object'},
                'paths': {'type': 'object'},
                'performance': {'type': 'object'}
            }
        }


class DomainConfiguration(LayeredConfiguration):
    """Domain-specific configuration."""
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': True,
            'domain_specific': {}
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'required': ['enabled'],
            'properties': {
                'enabled': {'type': 'boolean'}
            }
        }


class ServicesConfiguration(LayeredConfiguration):
    """Services layer configuration."""
    
    def __init__(self, service_type: str, parent: Optional[LayeredConfiguration] = None):
        super().__init__(ConfigurationLayer.SERVICES, parent=parent)
        self.service_type = service_type
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'service_type': self.service_type,
            'enabled': True,
            'service_specific': {}
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'required': ['service_type', 'enabled'],
            'properties': {
                'service_type': {'type': 'string'},
                'enabled': {'type': 'boolean'}
            }
        }


# Global instance
hierarchical_config_coordinator = HierarchicalConfigurationCoordinator()


# Export key components
__all__ = [
    'HierarchicalConfigurationCoordinator',
    'FoundationConfiguration',
    'DomainConfiguration',
    'ServicesConfiguration',
    'hierarchical_config_coordinator'
]