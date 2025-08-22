"""
Foundation Configuration Layer
============================

Core configuration abstractions and base functionality providing the foundation
for hierarchical configuration management across the TestMaster system.

Components:
- Base configuration classes and validation
- Configuration loaders (YAML, JSON, Environment)
- Configuration providers (File, Environment, Remote)

Author: Agent E - Infrastructure Consolidation
"""

from .base.config_base import ConfigurationBase
from .base.validation import ConfigurationValidator
from .base.serialization import ConfigurationSerializer

from .loaders.yaml_loader import YAMLConfigurationLoader
from .loaders.json_loader import JSONConfigurationLoader
from .loaders.env_loader import EnvironmentConfigurationLoader

from .providers.file_provider import FileConfigurationProvider
from .providers.env_provider import EnvironmentConfigurationProvider

__all__ = [
    # Base configuration
    'ConfigurationBase',
    'ConfigurationValidator',
    'ConfigurationSerializer',
    
    # Configuration loaders
    'YAMLConfigurationLoader',
    'JSONConfigurationLoader', 
    'EnvironmentConfigurationLoader',
    
    # Configuration providers
    'FileConfigurationProvider',
    'EnvironmentConfigurationProvider'
]