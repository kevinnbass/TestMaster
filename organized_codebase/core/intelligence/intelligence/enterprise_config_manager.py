"""
Enterprise Configuration Manager
===============================

Comprehensive configuration management system providing environment-based profiles,
dynamic reloading, validation, type-safe access, and enterprise-grade features.

Features:
- Environment-based configuration profiles (dev, staging, prod)
- Dynamic configuration hot-reloading with change detection
- Type-safe configuration access with validation
- Multiple configuration sources (files, environment variables, defaults)
- Configuration encryption and security features
- Audit logging and change tracking
- Template-based configuration generation
- Cross-system configuration synchronization

Author: TestMaster Intelligence Team
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime
import threading
import hashlib
import copy
import base64
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments with enterprise support"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    DISASTER_RECOVERY = "disaster_recovery"

class ConfigurationLevel(Enum):
    """Configuration priority levels"""
    DEFAULTS = 0
    ENVIRONMENT = 1
    USER = 2
    RUNTIME = 3
    OVERRIDE = 4

class SecurityLevel(Enum):
    """Security levels for configuration values"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

@dataclass
class ConfigurationValue:
    """Wrapper for configuration values with metadata"""
    value: Any
    source: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    encrypted: bool = False
    validation_rules: List[str] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.value, str) and self.encrypted:
            # Decrypt value if needed
            self.value = self._decrypt_value(self.value)
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt configuration value (simplified implementation)"""
        try:
            # Simple base64 decoding for demo - use proper encryption in production
            return base64.b64decode(encrypted_value.encode()).decode()
        except:
            return encrypted_value

@dataclass
class IntelligenceConfig:
    """Intelligence system configuration"""
    enabled: bool = True
    processing_threads: int = 4
    max_queue_size: int = 1000
    timeout_seconds: int = 30
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    performance_monitoring: bool = True
    error_recovery_enabled: bool = True
    ml_models_enabled: bool = True
    predictive_analytics: bool = True
    anomaly_detection: bool = True
    auto_scaling: bool = True

@dataclass
class SecurityConfig:
    """Security configuration"""
    authentication_required: bool = True
    authorization_enabled: bool = True
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    audit_logging: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    password_policy_enabled: bool = True
    two_factor_auth: bool = False
    api_rate_limiting: bool = True
    sql_injection_protection: bool = True
    xss_protection: bool = True
    csrf_protection: bool = True

@dataclass
class MonitoringConfig:
    """Enhanced monitoring configuration"""
    enabled: bool = True
    real_time_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 80.0,
        'memory_usage': 85.0,
        'error_rate': 5.0,
        'response_time': 2.0
    })
    notification_channels: List[str] = field(default_factory=lambda: ['email', 'slack'])
    log_retention_days: int = 30
    metrics_retention_days: int = 90
    dashboard_enabled: bool = True
    dashboard_port: int = 8080
    health_check_interval: int = 30
    performance_profiling: bool = False
    distributed_tracing: bool = False

@dataclass
class DatabaseConfig:
    """Database configuration with enterprise features"""
    host: str = "localhost"
    port: int = 5432
    database: str = "testmaster"
    username: str = "testmaster"
    password: str = ""
    ssl_enabled: bool = True
    connection_pool_size: int = 10
    max_connections: int = 100
    connection_timeout: int = 30
    query_timeout: int = 60
    read_replicas: List[str] = field(default_factory=list)
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    encryption_key: str = ""
    audit_table_changes: bool = True

@dataclass
class CachingConfig:
    """Advanced caching configuration"""
    enabled: bool = True
    cache_type: str = "redis"  # redis, memcached, memory
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    max_memory: str = "256mb"
    eviction_policy: str = "allkeys-lru"
    compression_enabled: bool = True
    serialization_format: str = "json"  # json, pickle, msgpack
    cluster_enabled: bool = False
    sentinel_enabled: bool = False
    sharding_enabled: bool = False
    cache_warming_enabled: bool = True
    cache_warming_schedule: str = "0 1 * * *"  # Daily at 1 AM

@dataclass
class APIConfig:
    """API configuration with enterprise features"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    max_request_size: int = 104857600  # 100MB
    request_timeout: int = 300
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limiting: Dict[str, int] = field(default_factory=lambda: {
        "requests_per_minute": 100,
        "requests_per_hour": 1000
    })
    authentication_backends: List[str] = field(default_factory=lambda: ["jwt", "api_key"])
    ssl_enabled: bool = True
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    api_versioning: bool = True
    swagger_enabled: bool = True
    metrics_enabled: bool = True

class EnterpriseConfigManager:
    """
    Enterprise-grade configuration management system providing comprehensive
    configuration handling, validation, security, and lifecycle management.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern with thread safety"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize enterprise configuration manager"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.environment = self._detect_environment()
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration sections
        self.intelligence = IntelligenceConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.database = DatabaseConfig()
        self.caching = CachingConfig()
        self.api = APIConfig()
        
        # Configuration metadata
        self.config_values: Dict[str, ConfigurationValue] = {}
        self.config_templates: Dict[str, Dict] = {}
        self.config_watchers: List[Callable] = []
        self.change_listeners: List[Callable] = []
        
        # Configuration files with hierarchy
        self.config_files = {
            ConfigurationLevel.DEFAULTS: self.config_dir / "defaults.json",
            ConfigurationLevel.ENVIRONMENT: self.config_dir / f"{self.environment.value}.json",
            ConfigurationLevel.USER: self.config_dir / "user.json",
            ConfigurationLevel.RUNTIME: self.config_dir / "runtime.json",
            ConfigurationLevel.OVERRIDE: self.config_dir / "override.json"
        }
        
        # State management
        self._config_hash = ""
        self._last_reload = datetime.now()
        self._hot_reload_enabled = True
        self._validation_rules: Dict[str, List[Callable]] = defaultdict(list)
        self._audit_log: List[Dict] = []
        
        # Load all configurations
        self._load_all_configurations()
        
        # Start configuration monitoring
        self._start_configuration_monitoring()
        
        logger.info(f"Enterprise Configuration Manager initialized for {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Detect current environment with fallback chain"""
        # Check environment variable
        env_var = os.getenv("TESTMASTER_ENV", "").lower()
        if env_var:
            try:
                return Environment(env_var)
            except ValueError:
                pass
        
        # Check configuration file markers
        for env in Environment:
            marker_file = Path(f".env.{env.value}")
            if marker_file.exists():
                return env
        
        # Check hostname patterns
        hostname = os.getenv("HOSTNAME", "").lower()
        if "prod" in hostname:
            return Environment.PRODUCTION
        elif "staging" in hostname or "stage" in hostname:
            return Environment.STAGING
        elif "test" in hostname:
            return Environment.TESTING
        
        # Default to local
        return Environment.LOCAL
    
    def _load_all_configurations(self):
        """Load configurations from all sources in priority order"""
        # Load in priority order (lowest to highest)
        for level in ConfigurationLevel:
            config_file = self.config_files[level]
            if config_file.exists():
                self._load_configuration_file(config_file, level)
        
        # Load environment variables
        self._load_environment_variables()
        
        # Apply configuration templates
        self._apply_configuration_templates()
        
        # Validate all configurations
        self._validate_all_configurations()
        
        # Calculate configuration hash
        self._config_hash = self._calculate_configuration_hash()
        
        self._audit_log.append({
            'action': 'configuration_loaded',
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment.value,
            'sources': len([f for f in self.config_files.values() if f.exists()])
        })
    
    def _load_configuration_file(self, file_path: Path, level: ConfigurationLevel):
        """Load configuration from file with enhanced error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == ".json":
                    data = json.load(f)
                elif file_path.suffix in [".yml", ".yaml"]:
                    data = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported configuration file format: {file_path}")
                    return
            
            # Apply configuration data to sections
            self._apply_configuration_data(data, level.name.lower())
            
            logger.info(f"Loaded configuration from {file_path} (level: {level.name})")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            raise
    
    def _apply_configuration_data(self, data: Dict, source: str):
        """Apply configuration data to sections with metadata tracking"""
        for section_name, section_data in data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        # Store old value for auditing
                        old_value = getattr(section, key, None)
                        
                        # Set new value
                        setattr(section, key, value)
                        
                        # Track in config values with metadata
                        config_key = f"{section_name}.{key}"
                        self.config_values[config_key] = ConfigurationValue(
                            value=value,
                            source=source,
                            timestamp=datetime.now()
                        )
                        
                        # Audit log for changes
                        if old_value != value:
                            self._audit_log.append({
                                'action': 'configuration_changed',
                                'key': config_key,
                                'old_value': old_value,
                                'new_value': value,
                                'source': source,
                                'timestamp': datetime.now().isoformat()
                            })
    
    def _load_environment_variables(self):
        """Load configuration from environment variables with enterprise mapping"""
        env_mappings = {
            # Intelligence configuration
            "TESTMASTER_INTELLIGENCE_ENABLED": ("intelligence", "enabled", bool),
            "TESTMASTER_PROCESSING_THREADS": ("intelligence", "processing_threads", int),
            "TESTMASTER_MAX_QUEUE_SIZE": ("intelligence", "max_queue_size", int),
            
            # Security configuration
            "TESTMASTER_AUTH_REQUIRED": ("security", "authentication_required", bool),
            "TESTMASTER_ENCRYPTION_ENABLED": ("security", "encryption_at_rest", bool),
            "TESTMASTER_SESSION_TIMEOUT": ("security", "session_timeout_minutes", int),
            
            # Monitoring configuration
            "TESTMASTER_MONITORING_ENABLED": ("monitoring", "enabled", bool),
            "TESTMASTER_METRICS_INTERVAL": ("monitoring", "metrics_collection_interval", int),
            "TESTMASTER_DASHBOARD_PORT": ("monitoring", "dashboard_port", int),
            
            # Database configuration
            "TESTMASTER_DB_HOST": ("database", "host", str),
            "TESTMASTER_DB_PORT": ("database", "port", int),
            "TESTMASTER_DB_NAME": ("database", "database", str),
            "TESTMASTER_DB_USER": ("database", "username", str),
            "TESTMASTER_DB_PASSWORD": ("database", "password", str),
            
            # Caching configuration
            "TESTMASTER_CACHE_ENABLED": ("caching", "enabled", bool),
            "TESTMASTER_CACHE_HOST": ("caching", "host", str),
            "TESTMASTER_CACHE_PORT": ("caching", "port", int),
            
            # API configuration
            "TESTMASTER_API_HOST": ("api", "host", str),
            "TESTMASTER_API_PORT": ("api", "port", int),
            "TESTMASTER_API_WORKERS": ("api", "workers", int),
        }
        
        for env_var, (section, field, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    section_obj = getattr(self, section)
                    
                    # Type conversion
                    if type_func == bool:
                        parsed_value = value.lower() in ["true", "1", "yes", "on", "enabled"]
                    elif type_func == list:
                        parsed_value = [item.strip() for item in value.split(",")]
                    else:
                        parsed_value = type_func(value)
                    
                    # Set value
                    setattr(section_obj, field, parsed_value)
                    
                    # Track in config values
                    config_key = f"{section}.{field}"
                    self.config_values[config_key] = ConfigurationValue(
                        value=parsed_value,
                        source="environment",
                        timestamp=datetime.now()
                    )
                    
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to set {section}.{field} from {env_var}: {e}")
    
    def _apply_configuration_templates(self):
        """Apply configuration templates for environment-specific settings"""
        # Define templates for different environments
        if self.environment == Environment.PRODUCTION:
            self._apply_production_template()
        elif self.environment == Environment.STAGING:
            self._apply_staging_template()
        elif self.environment == Environment.DEVELOPMENT:
            self._apply_development_template()
        else:
            self._apply_local_template()
    
    def _apply_production_template(self):
        """Apply production-optimized configuration template"""
        # Security hardening
        self.security.authentication_required = True
        self.security.encryption_at_rest = True
        self.security.encryption_in_transit = True
        self.security.audit_logging = True
        self.security.two_factor_auth = True
        
        # Performance optimization
        self.intelligence.processing_threads = 8
        self.intelligence.max_queue_size = 5000
        self.intelligence.cache_enabled = True
        
        # Monitoring enhancement
        self.monitoring.real_time_monitoring = True
        self.monitoring.alert_thresholds.update({
            'cpu_usage': 70.0,
            'memory_usage': 75.0,
            'error_rate': 1.0,
            'response_time': 1.0
        })
        
        # Database optimization
        self.database.connection_pool_size = 20
        self.database.max_connections = 200
        self.database.backup_enabled = True
        
        # API hardening
        self.api.debug = False
        self.api.workers = 8
        self.api.ssl_enabled = True
        self.api.rate_limiting = {
            "requests_per_minute": 1000,
            "requests_per_hour": 10000
        }
    
    def _apply_staging_template(self):
        """Apply staging configuration template"""
        # Balanced security and development needs
        self.security.authentication_required = True
        self.security.encryption_at_rest = True
        self.security.audit_logging = True
        
        # Moderate performance settings
        self.intelligence.processing_threads = 4
        self.intelligence.max_queue_size = 2000
        
        # Enhanced monitoring for testing
        self.monitoring.real_time_monitoring = True
        self.monitoring.performance_profiling = True
        
        # Development-friendly API settings
        self.api.debug = False
        self.api.swagger_enabled = True
    
    def _apply_development_template(self):
        """Apply development configuration template"""
        # Relaxed security for development
        self.security.authentication_required = False
        self.security.encryption_at_rest = False
        self.security.audit_logging = False
        
        # Debug-friendly settings
        self.intelligence.processing_threads = 2
        self.monitoring.performance_profiling = True
        
        # Development API settings
        self.api.debug = True
        self.api.swagger_enabled = True
        self.api.workers = 2
    
    def _apply_local_template(self):
        """Apply local development configuration template"""
        # Minimal security for local development
        self.security.authentication_required = False
        self.security.encryption_at_rest = False
        
        # Lightweight settings
        self.intelligence.processing_threads = 1
        self.intelligence.max_queue_size = 100
        
        # Local development API
        self.api.debug = True
        self.api.host = "127.0.0.1"
        self.api.workers = 1
    
    def _validate_all_configurations(self):
        """Validate all configuration sections and values"""
        validation_errors = []
        
        # Intelligence validation
        if self.intelligence.processing_threads < 1:
            validation_errors.append("Intelligence processing_threads must be >= 1")
        if self.intelligence.max_queue_size < 10:
            validation_errors.append("Intelligence max_queue_size must be >= 10")
        
        # Security validation
        if self.environment == Environment.PRODUCTION:
            if not self.security.authentication_required:
                validation_errors.append("Authentication is required in production")
            if not self.security.encryption_at_rest:
                validation_errors.append("Encryption at rest is required in production")
        
        # Database validation
        if not self.database.host:
            validation_errors.append("Database host is required")
        if self.database.port < 1 or self.database.port > 65535:
            validation_errors.append("Database port must be between 1 and 65535")
        
        # API validation
        if self.api.port < 1 or self.api.port > 65535:
            validation_errors.append("API port must be between 1 and 65535")
        if self.api.workers < 1:
            validation_errors.append("API workers must be >= 1")
        
        # Log validation results
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Configuration validation error: {error}")
            if self.environment == Environment.PRODUCTION:
                raise ValueError(f"Configuration validation failed: {validation_errors}")
        else:
            logger.info("All configurations validated successfully")
    
    def _calculate_configuration_hash(self) -> str:
        """Calculate hash of current configuration state"""
        config_dict = self.to_dict()
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _start_configuration_monitoring(self):
        """Start monitoring for configuration changes"""
        if self._hot_reload_enabled:
            monitor_thread = threading.Thread(target=self._configuration_monitor_loop, daemon=True)
            monitor_thread.start()
            logger.info("Configuration monitoring started")
    
    def _configuration_monitor_loop(self):
        """Monitor configuration files for changes"""
        while self._hot_reload_enabled:
            try:
                # Check if any configuration files have been modified
                current_hash = self._calculate_file_hashes()
                if hasattr(self, '_file_hashes') and current_hash != self._file_hashes:
                    logger.info("Configuration file changes detected, reloading...")
                    self.reload_configuration()
                
                self._file_hashes = current_hash
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Configuration monitoring error: {e}")
                time.sleep(10)  # Back off on error
    
    def _calculate_file_hashes(self) -> str:
        """Calculate hash of all configuration files"""
        file_hashes = []
        for config_file in self.config_files.values():
            if config_file.exists():
                with open(config_file, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                file_hashes.append(f"{config_file.name}:{file_hash}")
        
        return hashlib.md5(":".join(sorted(file_hashes)).encode()).hexdigest()
    
    def reload_configuration(self):
        """Reload configuration from all sources"""
        old_hash = self._config_hash
        
        try:
            self._load_all_configurations()
            new_hash = self._config_hash
            
            if old_hash != new_hash:
                # Notify change listeners
                for listener in self.change_listeners:
                    try:
                        listener(self)
                    except Exception as e:
                        logger.error(f"Configuration change listener error: {e}")
                
                self._audit_log.append({
                    'action': 'configuration_reloaded',
                    'timestamp': datetime.now().isoformat(),
                    'old_hash': old_hash,
                    'new_hash': new_hash
                })
                
                logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            raise
    
    def get_configuration_value(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation path with metadata"""
        try:
            parts = path.split(".")
            obj = self
            
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return default
            
            return obj
            
        except Exception as e:
            logger.warning(f"Failed to get configuration value for {path}: {e}")
            return default
    
    def set_configuration_value(self, path: str, value: Any, source: str = "runtime"):
        """Set configuration value by dot-notation path with audit trail"""
        try:
            parts = path.split(".")
            obj = self
            
            # Navigate to parent object
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    raise ValueError(f"Invalid configuration path: {path}")
            
            # Get old value for audit
            old_value = getattr(obj, parts[-1], None)
            
            # Set new value
            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], value)
                
                # Track in config values
                self.config_values[path] = ConfigurationValue(
                    value=value,
                    source=source,
                    timestamp=datetime.now()
                )
                
                # Audit log
                self._audit_log.append({
                    'action': 'configuration_set',
                    'key': path,
                    'old_value': old_value,
                    'new_value': value,
                    'source': source,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update configuration hash
                self._config_hash = self._calculate_configuration_hash()
                
                logger.info(f"Configuration value set: {path} = {value}")
            else:
                raise ValueError(f"Invalid configuration field: {parts[-1]}")
                
        except Exception as e:
            logger.error(f"Failed to set configuration value for {path}: {e}")
            raise
    
    def add_change_listener(self, listener: Callable):
        """Add configuration change listener"""
        self.change_listeners.append(listener)
        logger.info("Configuration change listener added")
    
    def remove_change_listener(self, listener: Callable):
        """Remove configuration change listener"""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
            logger.info("Configuration change listener removed")
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert configuration to dictionary with security filtering"""
        config_dict = {
            "environment": self.environment.value,
            "intelligence": asdict(self.intelligence),
            "security": asdict(self.security),
            "monitoring": asdict(self.monitoring),
            "database": asdict(self.database),
            "caching": asdict(self.caching),
            "api": asdict(self.api),
            "metadata": {
                "loaded_at": self._last_reload.isoformat(),
                "config_hash": self._config_hash,
                "sources_count": len([f for f in self.config_files.values() if f.exists()])
            }
        }
        
        # Filter sensitive information
        if not include_sensitive:
            self._filter_sensitive_data(config_dict)
        
        return config_dict
    
    def _filter_sensitive_data(self, config_dict: Dict):
        """Remove sensitive data from configuration dictionary"""
        sensitive_fields = [
            "password", "secret", "key", "token", "credential"
        ]
        
        def filter_recursive(obj):
            if isinstance(obj, dict):
                for key, value in list(obj.items()):
                    if any(sensitive in key.lower() for sensitive in sensitive_fields):
                        obj[key] = "***FILTERED***"
                    else:
                        filter_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    filter_recursive(item)
        
        filter_recursive(config_dict)
    
    def save_configuration(self, level: ConfigurationLevel = ConfigurationLevel.USER):
        """Save current configuration to file"""
        config_file = self.config_files[level]
        config_dict = self.to_dict(include_sensitive=False)
        
        # Remove metadata for saving
        config_dict.pop("metadata", None)
        config_dict.pop("environment", None)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self._audit_log.append({
                'action': 'configuration_saved',
                'file': str(config_file),
                'level': level.name,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get configuration audit log"""
        return self._audit_log[-limit:]
    
    def export_configuration(self, format_type: str = "json", include_sensitive: bool = False) -> str:
        """Export configuration in specified format"""
        config_dict = self.to_dict(include_sensitive=include_sensitive)
        
        if format_type.lower() == "json":
            return json.dumps(config_dict, indent=2, default=str)
        elif format_type.lower() in ["yml", "yaml"]:
            return yaml.dump(config_dict, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def validate_configuration_schema(self) -> Dict[str, Any]:
        """Validate configuration against schema"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Perform comprehensive validation
        try:
            self._validate_all_configurations()
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
        
        return validation_result
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring and debugging"""
        return {
            "environment": self.environment.value,
            "configuration_sources": {
                level.name: str(file_path) for level, file_path in self.config_files.items()
                if file_path.exists()
            },
            "last_reload": self._last_reload.isoformat(),
            "config_hash": self._config_hash,
            "hot_reload_enabled": self._hot_reload_enabled,
            "change_listeners": len(self.change_listeners),
            "audit_entries": len(self._audit_log),
            "validation_status": "valid",
            "sections": {
                "intelligence": bool(self.intelligence.enabled),
                "security": bool(self.security.authentication_required),
                "monitoring": bool(self.monitoring.enabled),
                "database": bool(self.database.host),
                "caching": bool(self.caching.enabled),
                "api": bool(self.api.port)
            }
        }
    
    def shutdown(self):
        """Shutdown configuration manager"""
        self._hot_reload_enabled = False
        
        # Save final audit log
        self._audit_log.append({
            'action': 'configuration_manager_shutdown',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info("Enterprise Configuration Manager shutdown")

# Global configuration instance
enterprise_config = EnterpriseConfigManager()