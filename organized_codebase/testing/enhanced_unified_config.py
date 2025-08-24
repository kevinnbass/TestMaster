"""
Enhanced Unified Configuration System for TestMaster
=====================================================

Extends testmaster_config.py with unified architecture features.
Provides single source of truth for all TestMaster configurations.

Features:
- Preserves existing TestMaster configuration structure
- Adds environment profiles and category organization
- Maintains backward compatibility with existing usage
- Integrates with cache and other systems

Author: Agent E - Infrastructure Consolidation
"""

import os
import sys
import json
import yaml
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime

# Import existing TestMaster configuration
sys.path.append(str(Path(__file__).parent))
from testmaster_config import (
    TestMasterConfig, APIConfig, GenerationConfig, MonitoringConfig,
    CachingConfig, ExecutionConfig, ReportingConfig, QualityConfig,
    OptimizationConfig, Environment, ConfigSection
)

logger = logging.getLogger(__name__)


# Extend existing enums with additional categories
class ConfigCategory(Enum):
    """Extended configuration categories mapping to TestMaster sections."""
    # Map to existing TestMaster sections
    API = "api"
    GENERATION = "generation"
    MONITORING = "monitoring"
    CACHING = "caching"
    EXECUTION = "execution"
    REPORTING = "reporting"
    QUALITY = "quality"
    OPTIMIZATION = "optimization"
    
    # Additional unified categories
    SECURITY = "security"
    TESTING = "testing"
    ML = "ml"
    INFRASTRUCTURE = "infrastructure"
    INTEGRATION = "integration"


# Extended security configuration
@dataclass
class SecurityConfig:
    """Security configuration extending TestMaster."""
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
            raise ValueError("Session timeout must be positive")
        if self.password_min_length < 4:
            raise ValueError("Password minimum length too short")
        return True


# Extended testing configuration that complements GenerationConfig
@dataclass
class TestingConfig:
    """Testing configuration extending TestMaster GenerationConfig."""
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
            raise ValueError("Coverage threshold must be 0-100")
        if self.parallel_workers <= 0:
            raise ValueError("Parallel workers must be positive")
        return True


# Extended ML configuration
@dataclass
class MLConfig:
    """ML/AI configuration complementing APIConfig."""
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
            raise ValueError("Temperature must be 0.0-2.0")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        return True


# Extended infrastructure configuration
@dataclass
class InfrastructureConfig:
    """Infrastructure configuration complementing ExecutionConfig."""
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
            raise ValueError("Worker count must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")
        return True


# Extended integration configuration
@dataclass
class IntegrationConfig:
    """Integration configuration for cross-system communication."""
    cross_system_enabled: bool = True
    message_bus_enabled: bool = True
    event_processing: bool = True
    workflow_integration: bool = True
    analytics_integration: bool = True
    real_time_sync: bool = False
    
    def validate(self) -> bool:
        """Validate integration configuration."""
        return True


class EnhancedConfigManager:
    """Enhanced configuration manager wrapping TestMasterConfig.
    
    Comprehensive configuration system that extends TestMasterConfig with
    unified architecture features, environment profiles, and category
    organization while maintaining full backward compatibility.
    
    Key Features:
    - **TestMaster Integration**: Wraps and extends existing TestMasterConfig
    - **Environment Profiles**: Support for dev/test/prod configuration profiles
    - **Category Organization**: Extended config categories (Security, ML, Infrastructure)
    - **Singleton Pattern**: Thread-safe singleton for consistent access
    - **Validation Framework**: Comprehensive validation for all config sections
    - **Dynamic Loading**: Runtime configuration reloading and updates
    - **YAML/JSON Support**: Multiple configuration file format support
    
    Architecture:
        The manager acts as a facade over TestMasterConfig, adding enhanced
        features without breaking existing usage patterns. Extended configurations
        complement the base config sections with domain-specific settings.
    
    Configuration Categories:
        - **Core TestMaster**: API, Generation, Monitoring, Caching, Execution
        - **Extended Categories**: Security, Testing, ML, Infrastructure, Integration
        - **Environment Profiles**: Development, Testing, Production configurations
        - **Dynamic Categories**: Runtime-loaded configuration extensions
    
    Usage:
        >>> config = EnhancedConfigManager(Environment.DEVELOPMENT)
        >>> api_config = config.get_api_config()
        >>> security_config = config.get_security_config()
        >>> config.set_category_config(ConfigCategory.ML, ml_settings)
        >>> config.reload_configuration()
    
    Thread Safety:
        Uses thread-safe singleton pattern with locking to ensure consistent
        configuration access across multi-threaded applications.
    
    Validation:
        All configuration sections include validation methods that are
        automatically called during loading and updating to ensure configuration
        integrity and prevent invalid settings.
    
    Example:
        >>> # Initialize with environment
        >>> config = EnhancedConfigManager(Environment.PRODUCTION)
        >>> 
        >>> # Access TestMaster configurations
        >>> api_settings = config.api_config
        >>> generation_settings = config.generation_config
        >>> 
        >>> # Access extended configurations
        >>> security_settings = config.get_security_config()
        >>> ml_settings = config.get_ml_config()
        >>> 
        >>> # Update configuration dynamically
        >>> config.update_security_config({'ssl_required': True})
        >>> config.save_profile('production_secure')
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, environment: Optional[Environment] = None):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, environment: Optional[Environment] = None):
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
        
        # Set environment before TestMaster initialization if provided
        if environment:
            os.environ["TESTMASTER_ENV"] = environment.value
        
        # Initialize TestMaster configuration
        self._testmaster_config = TestMasterConfig()
        
        self.profiles_dir = self._testmaster_config.config_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        
        # Initialize extended configuration categories
        self._extended_configs: Dict[ConfigCategory, Any] = {}
        self._load_extended_configurations()
        
        self._initialized = True
        logger.info(f"Enhanced configuration manager initialized for {self._testmaster_config.environment.value}")
    
    # Delegate TestMaster properties and methods
    @property
    def environment(self) -> Environment:
        return self._testmaster_config.environment
    
    @property
    def config_dir(self):
        return self._testmaster_config.config_dir
    
    @property
    def api(self) -> APIConfig:
        return self._testmaster_config.api
    
    @property
    def generation(self) -> GenerationConfig:
        return self._testmaster_config.generation
    
    @property
    def monitoring(self) -> MonitoringConfig:
        return self._testmaster_config.monitoring
    
    @property
    def caching(self) -> CachingConfig:
        return self._testmaster_config.caching
    
    @property
    def execution(self) -> ExecutionConfig:
        return self._testmaster_config.execution
    
    @property
    def reporting(self) -> ReportingConfig:
        return self._testmaster_config.reporting
    
    @property
    def quality(self) -> QualityConfig:
        return self._testmaster_config.quality
    
    @property
    def optimization(self) -> OptimizationConfig:
        return self._testmaster_config.optimization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._testmaster_config.to_dict()
    
    def save(self, file_path: Optional[Path] = None):
        """Save current configuration to file."""
        return self._testmaster_config.save(file_path)
    
    def _load_extended_configurations(self):
        """Load extended configurations from files and defaults."""
        # Load extended configurations that complement TestMaster configs
        self._extended_configs[ConfigCategory.SECURITY] = SecurityConfig()
        self._extended_configs[ConfigCategory.TESTING] = TestingConfig()
        self._extended_configs[ConfigCategory.ML] = MLConfig()
        self._extended_configs[ConfigCategory.INFRASTRUCTURE] = InfrastructureConfig()
        self._extended_configs[ConfigCategory.INTEGRATION] = IntegrationConfig()
        
        # Override with environment-specific settings
        self._load_environment_overrides()
        
        # Validate all configurations
        self._validate_extended_configs()
    
    def _load_environment_overrides(self):
        """Load environment-specific configuration overrides."""
        env_file = self.profiles_dir / f"{self.environment.value}.yaml"
        
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    overrides = yaml.safe_load(f)
                
                for category_name, config_data in overrides.items():
                    try:
                        # Handle TestMaster native categories
                        if hasattr(self, category_name) and isinstance(config_data, dict):
                            section = getattr(self, category_name)
                            for key, value in config_data.items():
                                if hasattr(section, key):
                                    setattr(section, key, value)
                        
                        # Handle extended categories
                        try:
                            category = ConfigCategory(category_name)
                            if category in self._extended_configs:
                                current_config = self._extended_configs[category]
                                for key, value in config_data.items():
                                    if hasattr(current_config, key):
                                        setattr(current_config, key, value)
                        except ValueError:
                            logger.warning(f"Unknown configuration category: {category_name}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to update category {category_name}: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to load environment overrides: {e}")
    
    def _validate_extended_configs(self):
        """Validate extended configurations."""
        for category, config in self._extended_configs.items():
            try:
                if hasattr(config, 'validate'):
                    config.validate()
            except Exception as e:
                logger.error(f"Extended configuration validation failed for {category.value}: {e}")
                raise
    
    def get_config(self, category: ConfigCategory) -> Any:
        """Get configuration for a category (TestMaster or extended)."""
        # Check TestMaster native configurations first
        if hasattr(self, category.value):
            return getattr(self, category.value)
        
        # Check extended configurations
        if category in self._extended_configs:
            return self._extended_configs[category]
            
        raise ValueError(f"No configuration found for category: {category}")
    
    # Convenience methods that map to TestMaster or extended configs
    def get_api_config(self) -> APIConfig:
        """Get API configuration from TestMaster."""
        return self.api
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration (extended)."""
        return self.get_config(ConfigCategory.SECURITY)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration from TestMaster."""
        return self.monitoring
    
    def get_caching_config(self) -> CachingConfig:
        """Get caching configuration from TestMaster."""
        return self.caching
    
    def get_testing_config(self) -> TestingConfig:
        """Get testing configuration (extended)."""
        return self.get_config(ConfigCategory.TESTING)
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation configuration from TestMaster."""
        return self.generation
    
    def get_execution_config(self) -> ExecutionConfig:
        """Get execution configuration from TestMaster."""
        return self.execution
    
    def get_reporting_config(self) -> ReportingConfig:
        """Get reporting configuration from TestMaster."""
        return self.reporting
    
    def get_quality_config(self) -> QualityConfig:
        """Get quality configuration from TestMaster."""
        return self.quality
    
    def get_optimization_config(self) -> OptimizationConfig:
        """Get optimization configuration from TestMaster."""
        return self.optimization
    
    def get_ml_config(self) -> MLConfig:
        """Get ML configuration (extended)."""
        return self.get_config(ConfigCategory.ML)
    
    def get_infrastructure_config(self) -> InfrastructureConfig:
        """Get infrastructure configuration (extended)."""
        return self.get_config(ConfigCategory.INFRASTRUCTURE)
    
    def get_integration_config(self) -> IntegrationConfig:
        """Get integration configuration (extended)."""
        return self.get_config(ConfigCategory.INTEGRATION)
    
    def update_config(self, category: ConfigCategory, **kwargs):
        """Update configuration values (TestMaster or extended)."""
        # Handle TestMaster native configurations
        if hasattr(self, category.value):
            section = getattr(self, category.value)
            for key, value in kwargs.items():
                if hasattr(section, key):
                    setattr(section, key, value)
                else:
                    logger.warning(f"Unknown field {key} in {category.value}")
            logger.info(f"Updated {category.value} configuration")
            return
        
        # Handle extended configurations
        if category in self._extended_configs:
            current_config = self._extended_configs[category]
            for key, value in kwargs.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
                else:
                    logger.warning(f"Unknown field {key} in {category.value}")
            
            # Validate if possible
            if hasattr(current_config, 'validate'):
                current_config.validate()
            
            logger.info(f"Updated {category.value} configuration")
            return
        
        raise ValueError(f"No configuration found for category: {category}")
    
    def save_profile(self, profile_name: str):
        """Save current configuration as a profile."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_file = self.profiles_dir / f"{profile_name}.yaml"
        
        profile_data = {}
        
        # Save TestMaster configurations
        testmaster_dict = self.to_dict()
        for key in ['api', 'generation', 'monitoring', 'caching', 'execution', 'reporting', 'quality', 'optimization']:
            if key in testmaster_dict:
                profile_data[key] = testmaster_dict[key]
        
        # Save extended configurations
        for category, config in self._extended_configs.items():
            if hasattr(config, '__dict__'):
                profile_data[category.value] = asdict(config)
        
        with open(profile_file, 'w') as f:
            yaml.dump(profile_data, f, default_flow_style=False)
        
        logger.info(f"Saved configuration profile: {profile_name}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary."""
        all_configs = self.to_dict()  # TestMaster configurations
        
        # Add extended configurations
        for category, config in self._extended_configs.items():
            if hasattr(config, '__dict__'):
                all_configs[category.value] = asdict(config)
        
        return all_configs
    
    def reload(self):
        """Reload configurations from files."""
        logger.info("Reloading configurations...")
        # Reload TestMaster configurations
        self._testmaster_config.reload()
        # Reload extended configurations
        self._load_extended_configurations()


# Global enhanced configuration manager instance
_enhanced_config_manager: Optional[EnhancedConfigManager] = None


def get_config_manager(environment: Optional[Environment] = None) -> EnhancedConfigManager:
    """Get the global enhanced configuration manager instance."""
    global _enhanced_config_manager
    
    if _enhanced_config_manager is None:
        _enhanced_config_manager = EnhancedConfigManager(environment)
    
    return _enhanced_config_manager


def get_config(category: ConfigCategory) -> Any:
    """Convenience function to get configuration."""
    return get_config_manager().get_config(category)


# Backward compatibility - provide access to the original TestMaster config
def get_testmaster_config() -> TestMasterConfig:
    """Get the underlying TestMaster configuration instance."""
    return get_config_manager()


# Convenience functions for all configurations
def get_api_config() -> APIConfig:
    """Get API configuration from TestMaster."""
    return get_config_manager().get_api_config()


def get_security_config() -> SecurityConfig:
    """Get security configuration (extended)."""
    return get_config_manager().get_security_config()


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration from TestMaster."""
    return get_config_manager().get_monitoring_config()


def get_caching_config() -> CachingConfig:
    """Get caching configuration from TestMaster."""
    return get_config_manager().get_caching_config()


def get_testing_config() -> TestingConfig:
    """Get testing configuration (extended)."""
    return get_config_manager().get_testing_config()


def get_generation_config() -> GenerationConfig:
    """Get generation configuration from TestMaster."""
    return get_config_manager().get_generation_config()


def get_execution_config() -> ExecutionConfig:
    """Get execution configuration from TestMaster."""
    return get_config_manager().get_execution_config()


def get_reporting_config() -> ReportingConfig:
    """Get reporting configuration from TestMaster."""
    return get_config_manager().get_reporting_config()


def get_quality_config() -> QualityConfig:
    """Get quality configuration from TestMaster."""
    return get_config_manager().get_quality_config()


def get_optimization_config() -> OptimizationConfig:
    """Get optimization configuration from TestMaster."""
    return get_config_manager().get_optimization_config()


def get_ml_config() -> MLConfig:
    """Get ML configuration (extended)."""
    return get_config_manager().get_ml_config()


def get_infrastructure_config() -> InfrastructureConfig:
    """Get infrastructure configuration (extended)."""
    return get_config_manager().get_infrastructure_config()


def get_integration_config() -> IntegrationConfig:
    """Get integration configuration (extended)."""
    return get_config_manager().get_integration_config()


# Export main classes and functions
__all__ = [
    # Inherited from testmaster_config
    'Environment',
    'ConfigSection',
    'TestMasterConfig',
    'APIConfig',
    'GenerationConfig',
    'MonitoringConfig', 
    'CachingConfig',
    'ExecutionConfig',
    'ReportingConfig',
    'QualityConfig',
    'OptimizationConfig',
    
    # Extended categories and configs
    'ConfigCategory',
    'SecurityConfig',
    'TestingConfig',
    'MLConfig',
    'InfrastructureConfig',
    'IntegrationConfig',
    
    # Enhanced manager
    'EnhancedConfigManager',
    'get_config_manager',
    'get_testmaster_config',
    'get_config',
    
    # Convenience functions
    'get_api_config',
    'get_security_config',
    'get_monitoring_config',
    'get_caching_config',
    'get_testing_config',
    'get_generation_config',
    'get_execution_config',
    'get_reporting_config',
    'get_quality_config',
    'get_optimization_config',
    'get_ml_config',
    'get_infrastructure_config',
    'get_integration_config'
]