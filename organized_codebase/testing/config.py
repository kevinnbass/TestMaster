#!/usr/bin/env python3
"""
Enhanced TestMaster Configuration Management - Phase 1A Agent 1

Intelligent configuration system for the TestMaster Hybrid Intelligence Platform.
Provides dynamic configuration management, environment adaptation, and 
intelligent defaults for all system components.
"""

import os
import json
import yaml
import threading
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConfigScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    MODULE = "module"
    AGENT = "agent"
    SESSION = "session"
    RUNTIME = "runtime"

class ConfigSource(Enum):
    """Configuration source priority."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"
    OVERRIDE = "override"

@dataclass
class ConfigValue:
    """Configuration value with metadata."""
    value: Any
    source: ConfigSource
    scope: ConfigScope
    timestamp: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    validator: Optional[Callable] = None
    
    def validate(self) -> bool:
        """Validate the configuration value."""
        if self.validator:
            try:
                return self.validator(self.value)
            except Exception:
                return False
        return True

@dataclass
class IntelligentConfigProfile:
    """Configuration profile for different operational modes."""
    name: str
    description: str
    config_overrides: Dict[str, Any]
    prerequisites: List[str] = field(default_factory=list)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)

class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass

class ConfigurationIntelligenceAgent:
    """
    Phase 1A Agent 1: Configuration Intelligence Agent
    
    Provides intelligent configuration management with:
    - Dynamic configuration adaptation
    - Environment-specific optimization
    - Performance-driven auto-tuning
    - Multi-source configuration hierarchy
    - Intelligent defaults and validation
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the configuration intelligence agent."""
        self.config_file = config_file
        self._config_values: Dict[str, ConfigValue] = {}
        self._config_lock = threading.RLock()
        self._observers: List[Callable] = []
        self._profiles: Dict[str, IntelligentConfigProfile] = {}
        self._active_profile: Optional[str] = None
        self._performance_metrics: Dict[str, float] = {}
        
        # Initialize configuration hierarchy
        self._initialize_default_configuration()
        self._load_configuration_profiles()
        
        if config_file:
            self._load_config_file(config_file)
        
        self._load_env_overrides()
        self._detect_optimal_profile()
        
        logger.info("Configuration Intelligence Agent initialized")
        logger.info(f"Active profile: {self._active_profile}")
        logger.info(f"Configuration sources: {len(self._config_values)} values loaded")
    
    def _initialize_default_configuration(self):
        """Initialize comprehensive default configuration."""
        defaults = {
            # Core System Configuration
            "core.orchestrator.max_parallel_tasks": ConfigValue(
                value=4, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Maximum parallel tasks in workflow orchestration",
                validator=lambda x: isinstance(x, int) and 1 <= x <= 100
            ),
            "core.orchestrator.task_timeout": ConfigValue(
                value=300, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Default task timeout in seconds",
                validator=lambda x: isinstance(x, (int, float)) and x > 0
            ),
            "core.shared_state.persistence_enabled": ConfigValue(
                value=True, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Enable persistent state storage"
            ),
            "core.shared_state.cleanup_interval": ConfigValue(
                value=3600, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="State cleanup interval in seconds"
            ),
            
            # Intelligence Layer Configuration
            "intelligence.hierarchical_planning.max_depth": ConfigValue(
                value=5, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Maximum planning tree depth",
                validator=lambda x: isinstance(x, int) and 1 <= x <= 10
            ),
            "intelligence.hierarchical_planning.reasoning_strategies": ConfigValue(
                value=["breadth_first", "depth_first", "best_first"], 
                source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Available reasoning strategies"
            ),
            "intelligence.consensus.min_participants": ConfigValue(
                value=2, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Minimum consensus participants",
                validator=lambda x: isinstance(x, int) and x >= 1
            ),
            "intelligence.consensus.threshold": ConfigValue(
                value=0.6, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Consensus agreement threshold",
                validator=lambda x: isinstance(x, (int, float)) and 0 < x <= 1
            ),
            "intelligence.llm.provider_fallback_enabled": ConfigValue(
                value=True, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Enable LLM provider fallback"
            ),
            "intelligence.llm.request_timeout": ConfigValue(
                value=60, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="LLM request timeout in seconds"
            ),
            
            # Security Configuration
            "security.intelligence.vulnerability_scan_enabled": ConfigValue(
                value=True, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Enable vulnerability scanning"
            ),
            "security.intelligence.compliance_standards": ConfigValue(
                value=["owasp_asvs", "sox", "gdpr"], 
                source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Active compliance standards"
            ),
            "security.intelligence.min_severity": ConfigValue(
                value="medium", source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Minimum security issue severity to report"
            ),
            
            # Monitoring Configuration
            "monitoring.performance.metrics_collection_enabled": ConfigValue(
                value=True, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Enable performance metrics collection"
            ),
            "monitoring.performance.alert_threshold": ConfigValue(
                value=0.8, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Performance alert threshold (0-1)"
            ),
            "monitoring.bottleneck.detection_enabled": ConfigValue(
                value=True, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Enable bottleneck detection"
            ),
            "monitoring.resource.adaptive_scaling_enabled": ConfigValue(
                value=True, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Enable adaptive resource scaling"
            ),
            
            # Bridge Configuration
            "bridges.protocol.message_timeout": ConfigValue(
                value=30, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Protocol message timeout in seconds"
            ),
            "bridges.event.max_events_memory": ConfigValue(
                value=10000, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Maximum events in memory"
            ),
            "bridges.session.checkpoint_interval": ConfigValue(
                value=300, source=ConfigSource.DEFAULT, scope=ConfigScope.MODULE,
                description="Session checkpoint interval in seconds"
            ),
            
            # Legacy API Configuration (for compatibility)
            "api.timeout": ConfigValue(
                value=120, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="API request timeout"
            ),
            "api.rate_limit_rpm": ConfigValue(
                value=30, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="API rate limit requests per minute"
            ),
            "api.max_retries": ConfigValue(
                value=3, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Maximum API retry attempts"
            ),
            "generation.mode": ConfigValue(
                value="auto", source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Test generation mode"
            ),
            "generation.quality_threshold": ConfigValue(
                value=70.0, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Test quality threshold"
            ),
            "generation.max_workers": ConfigValue(
                value=4, source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Maximum worker threads"
            ),
            "paths.output_directory": ConfigValue(
                value="tests/unit", source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Test output directory"
            ),
            "paths.cache_directory": ConfigValue(
                value=".testmaster_cache", source=ConfigSource.DEFAULT, scope=ConfigScope.GLOBAL,
                description="Cache directory"
            )
        }
        
        with self._config_lock:
            self._config_values.update(defaults)
    
    def _load_configuration_profiles(self):
        """Load intelligent configuration profiles."""
        profiles = {
            "development": IntelligentConfigProfile(
                name="development",
                description="Development environment optimized for debugging and flexibility",
                config_overrides={
                    "core.orchestrator.max_parallel_tasks": 2,
                    "intelligence.hierarchical_planning.max_depth": 3,
                    "monitoring.performance.metrics_collection_enabled": True,
                    "security.intelligence.min_severity": "low"
                },
                performance_targets={"response_time": 5.0, "throughput": 10.0},
                resource_limits={"memory_mb": 1024, "cpu_percent": 50}
            ),
            
            "production": IntelligentConfigProfile(
                name="production",
                description="Production environment optimized for performance and reliability",
                config_overrides={
                    "core.orchestrator.max_parallel_tasks": 8,
                    "intelligence.hierarchical_planning.max_depth": 7,
                    "monitoring.performance.alert_threshold": 0.9,
                    "security.intelligence.min_severity": "medium"
                },
                performance_targets={"response_time": 2.0, "throughput": 50.0},
                resource_limits={"memory_mb": 4096, "cpu_percent": 80}
            ),
            
            "security_focused": IntelligentConfigProfile(
                name="security_focused", 
                description="Security-first configuration with enhanced scanning",
                config_overrides={
                    "security.intelligence.vulnerability_scan_enabled": True,
                    "security.intelligence.min_severity": "low",
                    "security.intelligence.compliance_standards": ["owasp_asvs", "sox", "gdpr", "pci_dss"],
                    "intelligence.consensus.threshold": 0.8
                },
                performance_targets={"security_coverage": 95.0},
                resource_limits={"memory_mb": 2048, "cpu_percent": 70}
            ),
            
            "high_performance": IntelligentConfigProfile(
                name="high_performance",
                description="Maximum performance configuration for large-scale operations",
                config_overrides={
                    "core.orchestrator.max_parallel_tasks": 16,
                    "generation.max_workers": 8,
                    "bridges.event.max_events_memory": 50000,
                    "monitoring.resource.adaptive_scaling_enabled": True
                },
                performance_targets={"response_time": 1.0, "throughput": 100.0},
                resource_limits={"memory_mb": 8192, "cpu_percent": 90}
            )
        }
        
        self._profiles.update(profiles)
    
    def _detect_optimal_profile(self):
        """Intelligently detect the optimal configuration profile."""
        # Check environment indicators
        env_profile = os.getenv("TESTMASTER_PROFILE")
        if env_profile and env_profile in self._profiles:
            self._active_profile = env_profile
            self._apply_profile(env_profile)
            return
        
        # Auto-detect based on environment
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            profile = "production"
        elif os.getenv("TESTMASTER_SECURITY_SCAN"):
            profile = "security_focused"
        elif os.getenv("TESTMASTER_HIGH_PERFORMANCE"):
            profile = "high_performance"
        else:
            profile = "development"
        
        self._active_profile = profile
        self._apply_profile(profile)
    
    def _apply_profile(self, profile_name: str):
        """Apply a configuration profile."""
        if profile_name not in self._profiles:
            logger.warning(f"Unknown profile: {profile_name}")
            return
        
        profile = self._profiles[profile_name]
        
        with self._config_lock:
            for key, value in profile.config_overrides.items():
                if key in self._config_values:
                    # Update existing config value
                    old_value = self._config_values[key]
                    self._config_values[key] = ConfigValue(
                        value=value,
                        source=ConfigSource.OVERRIDE,
                        scope=old_value.scope,
                        description=old_value.description,
                        validator=old_value.validator
                    )
                else:
                    # Create new config value
                    self._config_values[key] = ConfigValue(
                        value=value,
                        source=ConfigSource.OVERRIDE,
                        scope=ConfigScope.RUNTIME,
                        description=f"Profile override from {profile_name}"
                    )
        
        logger.info(f"Applied configuration profile: {profile_name}")
        self._notify_observers()
    
    def _load_config_file(self, config_file: str):
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {config_path.suffix}")
                    return
            
            # Flatten nested configuration
            flat_config = self._flatten_config(file_config)
            
            with self._config_lock:
                for key, value in flat_config.items():
                    if key in self._config_values:
                        old_value = self._config_values[key]
                        self._config_values[key] = ConfigValue(
                            value=value,
                            source=ConfigSource.FILE,
                            scope=old_value.scope,
                            description=old_value.description,
                            validator=old_value.validator
                        )
                    else:
                        self._config_values[key] = ConfigValue(
                            value=value,
                            source=ConfigSource.FILE,
                            scope=ConfigScope.GLOBAL,
                            description=f"Loaded from {config_file}"
                        )
            
            logger.info(f"Loaded configuration from: {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
    
    def _flatten_config(self, config_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration dictionary."""
        flat_config = {}
        
        for key, value in config_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat_config.update(self._flatten_config(value, new_key))
            else:
                flat_config[new_key] = value
        
        return flat_config
    
    def _load_env_overrides(self):
        """Load environment variable overrides."""
        env_prefix = "TESTMASTER_"
        
        with self._config_lock:
            for env_var, env_value in os.environ.items():
                if env_var.startswith(env_prefix):
                    # Convert environment variable to config key
                    config_key = env_var[len(env_prefix):].lower().replace('_', '.')
                    
                    # Try to parse the value
                    parsed_value = self._parse_env_value(env_value)
                    
                    if config_key in self._config_values:
                        old_value = self._config_values[config_key]
                        self._config_values[config_key] = ConfigValue(
                            value=parsed_value,
                            source=ConfigSource.ENVIRONMENT,
                            scope=old_value.scope,
                            description=old_value.description,
                            validator=old_value.validator
                        )
                    else:
                        self._config_values[config_key] = ConfigValue(
                            value=parsed_value,
                            source=ConfigSource.ENVIRONMENT,
                            scope=ConfigScope.RUNTIME,
                            description=f"Environment override: {env_var}"
                        )
        
        logger.debug("Loaded environment variable overrides")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        with self._config_lock:
            config_value = self._config_values.get(key)
            if config_value:
                if config_value.validate():
                    return config_value.value
                else:
                    logger.warning(f"Configuration validation failed for {key}, using default")
                    return default
            return default
    
    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.RUNTIME):
        """Set configuration value at runtime."""
        with self._config_lock:
            old_value = self._config_values.get(key)
            validator = old_value.validator if old_value else None
            description = old_value.description if old_value else f"Runtime override for {key}"
            
            self._config_values[key] = ConfigValue(
                value=value,
                source=ConfigSource.RUNTIME,
                scope=scope,
                description=description,
                validator=validator
            )
        
        self._notify_observers()
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def get_profile_info(self) -> Dict[str, Any]:
        """Get active profile information."""
        if not self._active_profile:
            return {}
        
        profile = self._profiles.get(self._active_profile, {})
        return {
            "name": self._active_profile,
            "description": getattr(profile, 'description', ''),
            "performance_targets": getattr(profile, 'performance_targets', {}),
            "resource_limits": getattr(profile, 'resource_limits', {})
        }
    
    def add_observer(self, callback: Callable):
        """Add configuration change observer."""
        self._observers.append(callback)
    
    def _notify_observers(self):
        """Notify configuration change observers."""
        for observer in self._observers:
            try:
                observer()
            except Exception as e:
                logger.error(f"Configuration observer error: {e}")
    
    def validate_all(self) -> List[str]:
        """Validate all configuration values."""
        errors = []
        
        with self._config_lock:
            for key, config_value in self._config_values.items():
                if not config_value.validate():
                    errors.append(f"Validation failed for {key}: {config_value.value}")
        
        return errors
    
    def export_config(self, format: str = "json") -> str:
        """Export current configuration."""
        config_dict = {}
        
        with self._config_lock:
            for key, config_value in self._config_values.items():
                config_dict[key] = {
                    "value": config_value.value,
                    "source": config_value.source.value,
                    "scope": config_value.scope.value,
                    "description": config_value.description
                }
        
        if format.lower() == "yaml":
            return yaml.dump(config_dict, default_flow_style=False)
        else:
            return json.dumps(config_dict, indent=2)

# Backwards compatibility
class TestMasterConfig(ConfigurationIntelligenceAgent):
    """Legacy configuration class for backwards compatibility."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize with legacy interface."""
        super().__init__(config_file)
        logger.info("Using legacy TestMasterConfig interface - consider upgrading to ConfigurationIntelligenceAgent")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Legacy method - returns simple dict for compatibility."""
        return {
            "api": {
                "timeout": self.get("api.timeout", 120),
                "rate_limit_rpm": self.get("api.rate_limit_rpm", 30), 
                "max_retries": self.get("api.max_retries", 3)
            },
            "generation": {
                "mode": self.get("generation.mode", "auto"),
                "quality_threshold": self.get("generation.quality_threshold", 70.0),
                "max_workers": self.get("generation.max_workers", 4)
            },
            "paths": {
                "output_directory": self.get("paths.output_directory", "tests/unit"),
                "cache_directory": self.get("paths.cache_directory", ".testmaster_cache")
            }
        }
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        # Use parent class implementation
        super()._load_env_overrides()

# Global configuration instance for legacy compatibility
_global_config = None

def get_config(config_file: Optional[str] = None) -> ConfigurationIntelligenceAgent:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigurationIntelligenceAgent(config_file)
    return _global_config

def reset_config():
    """Reset global configuration (useful for testing)."""
    global _global_config
    _global_config = None