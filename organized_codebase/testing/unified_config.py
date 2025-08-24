"""
Unified Configuration System for TestMaster
============================================

Centralized configuration management consolidating all scattered Config classes.
Provides single source of truth for all TestMaster configurations.

Consolidates from:
- testmaster_config.py (base configuration)
- 30+ scattered Config classes across modules
- Various environment and profile configurations

Author: Agent E - Infrastructure Consolidation
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


class ConfigCategory(Enum):
    """Configuration categories for organization."""
    API = "api"
    SECURITY = "security"
    MONITORING = "monitoring"
    CACHING = "caching"
    TESTING = "testing"
    ML = "ml"
    INFRASTRUCTURE = "infrastructure"
    INTEGRATION = "integration"


@dataclass
class ConfigBase:
    """Base configuration class with validation."""
    
    def validate(self) -> bool:
        """Validate configuration. Override in subclasses."""
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary."""
        return cls(**data)


@dataclass
class APIConfig(ConfigBase):
    """Consolidated API configuration."""
    endpoint: str = "http://localhost:5000"
    timeout: int = 30
    retry_count: int = 3
    rate_limit: int = 100
    auth_required: bool = True
    ssl_verify: bool = True
    api_version: str = "v1"
    cors_enabled: bool = True
    
    def validate(self) -> bool:
        """Validate API configuration."""
        if self.timeout <= 0:
            raise ConfigurationError("API timeout must be positive")
        if self.retry_count < 0:
            raise ConfigurationError("Retry count cannot be negative")
        return True


@dataclass
class SecurityConfig(ConfigBase):
    """Consolidated security configuration."""
    encryption_enabled: bool = True
    auth_method: str = "jwt"
    session_timeout: int = 3600
    password_min_length: int = 8
    max_login_attempts: int = 5
    ssl_required: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["localhost"])
    rate_limiting_enabled: bool = True
    
    def validate(self) -> bool:
        """Validate security configuration."""
        if self.session_timeout <= 0:
            raise ConfigurationError("Session timeout must be positive")
        if self.password_min_length < 4:
            raise ConfigurationError("Password minimum length too short")
        return True


@dataclass
class MonitoringConfig(ConfigBase):
    """Consolidated monitoring configuration."""
    enabled: bool = True
    metrics_interval: int = 60
    health_check_interval: int = 30
    log_level: str = "INFO"
    retention_days: int = 30
    alert_enabled: bool = True
    performance_tracking: bool = True
    
    def validate(self) -> bool:
        """Validate monitoring configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ConfigurationError(f"Invalid log level: {self.log_level}")
        return True


@dataclass
class CachingConfig(ConfigBase):
    """Consolidated caching configuration."""
    enabled: bool = True
    backend: str = "intelligent_cache"
    ttl_seconds: int = 3600
    max_size_mb: int = 1000
    compression_enabled: bool = True
    persistence_enabled: bool = True
    strategy: str = "LRU"
    
    def validate(self) -> bool:
        """Validate caching configuration."""
        valid_strategies = ["LRU", "LFU", "FIFO", "TTL"]
        if self.strategy not in valid_strategies:
            raise ConfigurationError(f"Invalid cache strategy: {self.strategy}")
        return True


@dataclass
class TestingConfig(ConfigBase):
    """Consolidated testing configuration."""
    framework: str = "pytest"
    coverage_threshold: float = 80.0
    parallel_workers: int = 4
    test_timeout: int = 300
    generate_reports: bool = True
    auto_healing_enabled: bool = True
    ml_test_optimization: bool = True
    
    def validate(self) -> bool:
        """Validate testing configuration."""
        if not 0.0 <= self.coverage_threshold <= 100.0:
            raise ConfigurationError("Coverage threshold must be 0-100")
        if self.parallel_workers <= 0:
            raise ConfigurationError("Parallel workers must be positive")
        return True


@dataclass
class MLConfig(ConfigBase):
    """Consolidated ML/AI configuration."""
    provider: str = "gemini"
    model: str = "gemini-2.5-pro"
    temperature: float = 0.1
    max_tokens: int = 4000
    rate_limit: int = 30
    timeout: int = 60
    optimization_enabled: bool = True
    
    def validate(self) -> bool:
        """Validate ML configuration."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError("Temperature must be 0.0-2.0")
        if self.max_tokens <= 0:
            raise ConfigurationError("Max tokens must be positive")
        return True


@dataclass
class InfrastructureConfig(ConfigBase):
    """Consolidated infrastructure configuration."""
    orchestration_enabled: bool = True
    distributed_mode: bool = False
    worker_count: int = 4
    memory_limit_mb: int = 2048
    disk_space_limit_gb: int = 10
    backup_enabled: bool = True
    scaling_enabled: bool = True
    
    def validate(self) -> bool:
        """Validate infrastructure configuration."""
        if self.worker_count <= 0:
            raise ConfigurationError("Worker count must be positive")
        if self.memory_limit_mb <= 0:
            raise ConfigurationError("Memory limit must be positive")
        return True


@dataclass
class IntegrationConfig(ConfigBase):
    """Consolidated integration configuration."""
    cross_system_enabled: bool = True
    message_bus_enabled: bool = True
    event_processing: bool = True
    workflow_integration: bool = True
    analytics_integration: bool = True
    real_time_sync: bool = False
    
    def validate(self) -> bool:
        """Validate integration configuration."""
        return True


class UnifiedConfigManager:
    """
    Centralized configuration manager.
    
    Consolidates all scattered Config classes into a single, unified system.
    Provides environment-based configuration with validation and type safety.
    """
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.config_dir = Path("config")
        self.profiles_dir = self.config_dir / "profiles"
        
        # Initialize configuration categories
        self._configs: Dict[ConfigCategory, ConfigBase] = {}
        self._load_configurations()
        
        logger.info(f"Unified configuration manager initialized for {environment.value}")
    
    def _load_configurations(self):
        """Load configurations from files and defaults."""
        # Load default configurations
        self._configs[ConfigCategory.API] = APIConfig()
        self._configs[ConfigCategory.SECURITY] = SecurityConfig()
        self._configs[ConfigCategory.MONITORING] = MonitoringConfig()
        self._configs[ConfigCategory.CACHING] = CachingConfig()
        self._configs[ConfigCategory.TESTING] = TestingConfig()
        self._configs[ConfigCategory.ML] = MLConfig()
        self._configs[ConfigCategory.INFRASTRUCTURE] = InfrastructureConfig()
        self._configs[ConfigCategory.INTEGRATION] = IntegrationConfig()
        
        # Override with environment-specific settings
        self._load_environment_overrides()
        
        # Validate all configurations
        self._validate_all()
    
    def _load_environment_overrides(self):
        """Load environment-specific configuration overrides."""
        env_file = self.profiles_dir / f"{self.environment.value}.yaml"
        
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    overrides = yaml.safe_load(f)
                
                for category_name, config_data in overrides.items():
                    try:
                        category = ConfigCategory(category_name)
                        if category in self._configs:
                            # Update configuration with overrides
                            current_config = self._configs[category]
                            updated_dict = current_config.to_dict()
                            updated_dict.update(config_data)
                            
                            # Create new configuration instance
                            config_class = type(current_config)
                            self._configs[category] = config_class.from_dict(updated_dict)
                            
                    except ValueError:
                        logger.warning(f"Unknown configuration category: {category_name}")
                        
            except Exception as e:
                logger.error(f"Failed to load environment overrides: {e}")
    
    def _validate_all(self):
        """Validate all configurations."""
        for category, config in self._configs.items():
            try:
                config.validate()
            except ConfigurationError as e:
                logger.error(f"Configuration validation failed for {category.value}: {e}")
                raise
    
    def get_config(self, category: ConfigCategory) -> ConfigBase:
        """Get configuration for a category."""
        if category not in self._configs:
            raise ConfigurationError(f"No configuration found for category: {category}")
        
        return self._configs[category]
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        return self.get_config(ConfigCategory.API)
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self.get_config(ConfigCategory.SECURITY)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.get_config(ConfigCategory.MONITORING)
    
    def get_caching_config(self) -> CachingConfig:
        """Get caching configuration."""
        return self.get_config(ConfigCategory.CACHING)
    
    def get_testing_config(self) -> TestingConfig:
        """Get testing configuration."""
        return self.get_config(ConfigCategory.TESTING)
    
    def get_ml_config(self) -> MLConfig:
        """Get ML configuration."""
        return self.get_config(ConfigCategory.ML)
    
    def get_infrastructure_config(self) -> InfrastructureConfig:
        """Get infrastructure configuration."""
        return self.get_config(ConfigCategory.INFRASTRUCTURE)
    
    def get_integration_config(self) -> IntegrationConfig:
        """Get integration configuration."""
        return self.get_config(ConfigCategory.INTEGRATION)
    
    def update_config(self, category: ConfigCategory, **kwargs):
        """Update configuration values."""
        if category not in self._configs:
            raise ConfigurationError(f"No configuration found for category: {category}")
        
        current_config = self._configs[category]
        config_dict = current_config.to_dict()
        config_dict.update(kwargs)
        
        # Create new configuration instance
        config_class = type(current_config)
        new_config = config_class.from_dict(config_dict)
        
        # Validate new configuration
        new_config.validate()
        
        # Update stored configuration
        self._configs[category] = new_config
        
        logger.info(f"Updated {category.value} configuration")
    
    def save_profile(self, profile_name: str):
        """Save current configuration as a profile."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_file = self.profiles_dir / f"{profile_name}.yaml"
        
        profile_data = {}
        for category, config in self._configs.items():
            profile_data[category.value] = config.to_dict()
        
        with open(profile_file, 'w') as f:
            yaml.dump(profile_data, f, default_flow_style=False)
        
        logger.info(f"Saved configuration profile: {profile_name}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary."""
        return {
            category.value: config.to_dict() 
            for category, config in self._configs.items()
        }
    
    def reload(self):
        """Reload configurations from files."""
        logger.info("Reloading configurations...")
        self._load_configurations()


# Global configuration manager instance
_config_manager: Optional[UnifiedConfigManager] = None


def get_config_manager(environment: Optional[Environment] = None) -> UnifiedConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        env = environment or Environment.DEVELOPMENT
        _config_manager = UnifiedConfigManager(env)
    
    return _config_manager


def get_config(category: ConfigCategory) -> ConfigBase:
    """Convenience function to get configuration."""
    return get_config_manager().get_config(category)


# Convenience functions for common configurations
def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_config_manager().get_api_config()


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_config_manager().get_security_config()


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_config_manager().get_monitoring_config()


def get_testing_config() -> TestingConfig:
    """Get testing configuration."""
    return get_config_manager().get_testing_config()


# Export main classes and functions
__all__ = [
    'Environment',
    'ConfigCategory',
    'ConfigBase',
    'APIConfig',
    'SecurityConfig', 
    'MonitoringConfig',
    'CachingConfig',
    'TestingConfig',
    'MLConfig',
    'InfrastructureConfig',
    'IntegrationConfig',
    'UnifiedConfigManager',
    'get_config_manager',
    'get_config',
    'get_api_config',
    'get_security_config',
    'get_monitoring_config',
    'get_testing_config',
    'ConfigurationError'
]