#!/usr/bin/env python3
"""
TestMaster Unified Configuration System
Single source of truth for all TestMaster settings.

Features:
- Environment-based configuration profiles
- Dynamic configuration reloading
- Validation and defaults management
- Type-safe configuration access
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime
import threading
import hashlib

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


class ConfigSection(Enum):
    """Configuration sections."""
    API = "api"
    GENERATION = "generation"
    MONITORING = "monitoring"
    CACHING = "caching"
    EXECUTION = "execution"
    REPORTING = "reporting"
    QUALITY = "quality"
    OPTIMIZATION = "optimization"


@dataclass
class APIConfig:
    """API configuration settings."""
    gemini_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    rate_limit_rpm: int = 30
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0
    preferred_model: str = "gemini-2.5-pro"
    fallback_models: List[str] = field(default_factory=lambda: ["gemini-2.0-flash", "gemini-1.5-pro"])
    
    def __post_init__(self):
        # Load from environment if not set
        self.gemini_api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.google_api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")


@dataclass
class GenerationConfig:
    """Test generation configuration."""
    max_iterations: int = 5
    self_healing_enabled: bool = True
    verification_passes: int = 3
    min_quality_score: int = 70
    max_quality_score: int = 100
    parallel_workers: int = 4
    batch_size: int = 10
    use_real_imports: bool = True
    generate_edge_cases: bool = True
    generate_error_tests: bool = True
    test_framework: str = "pytest"
    coverage_target: float = 95.0
    incremental_mode: bool = True
    context_window_size: int = 8000
    temperature: float = 0.1
    max_output_tokens: int = 4000


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    continuous_mode: bool = False
    interval_minutes: int = 120
    idle_threshold_minutes: int = 10
    watch_directories: List[str] = field(default_factory=list)
    ignore_patterns: List[str] = field(default_factory=lambda: ["*.pyc", "__pycache__", ".git", "*.log"])
    track_refactorings: bool = True
    track_renames: bool = True
    track_splits: bool = True
    track_merges: bool = True
    similarity_threshold: float = 0.3
    change_detection_method: str = "git"  # git, filesystem, hybrid
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class CachingConfig:
    """Caching configuration."""
    enabled: bool = True
    cache_directory: str = "cache"
    cache_ttl_seconds: int = 3600
    max_cache_size_mb: int = 500
    cache_strategy: str = "lru"  # lru, lfu, fifo
    persistent_cache: bool = True
    cache_api_responses: bool = True
    cache_test_results: bool = True
    cache_analysis_results: bool = True
    compression_enabled: bool = True
    encryption_enabled: bool = False


@dataclass
class ExecutionConfig:
    """Test execution configuration."""
    parallel_execution: bool = True
    max_parallel_tests: int = 10
    timeout_per_test: int = 30
    fail_fast: bool = False
    retry_failed_tests: bool = True
    max_test_retries: int = 2
    capture_output: bool = True
    verbose_output: bool = False
    profile_tests: bool = False
    memory_limit_mb: int = 1024
    cpu_limit_percent: int = 80
    distributed_execution: bool = False
    execution_order: str = "priority"  # priority, random, alphabetical, dependency


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    generate_html_report: bool = True
    generate_json_report: bool = True
    generate_junit_xml: bool = False
    report_directory: str = "reports"
    include_coverage_report: bool = True
    include_quality_metrics: bool = True
    include_performance_metrics: bool = True
    include_trend_analysis: bool = True
    email_reports: bool = False
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook: Optional[str] = None
    dashboard_enabled: bool = True
    dashboard_port: int = 8080


@dataclass
class QualityConfig:
    """Quality assurance configuration."""
    min_test_coverage: float = 80.0
    min_branch_coverage: float = 70.0
    max_cyclomatic_complexity: int = 10
    max_test_duration_seconds: float = 10.0
    require_docstrings: bool = True
    require_type_hints: bool = True
    lint_tests: bool = True
    security_scanning: bool = False
    mutation_testing: bool = False
    property_based_testing: bool = False
    fuzz_testing: bool = False


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    enable_test_selection: bool = True
    enable_test_prioritization: bool = True
    enable_deduplication: bool = True
    enable_caching: bool = True
    enable_parallel_generation: bool = True
    optimize_imports: bool = True
    minimize_api_calls: bool = True
    batch_api_requests: bool = True
    use_incremental_updates: bool = True
    profile_performance: bool = False
    auto_tune_parameters: bool = False


class TestMasterConfig:
    """Main configuration manager for TestMaster."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.environment = self._detect_environment()
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize configuration sections
        self.api = APIConfig()
        self.generation = GenerationConfig()
        self.monitoring = MonitoringConfig()
        self.caching = CachingConfig()
        self.execution = ExecutionConfig()
        self.reporting = ReportingConfig()
        self.quality = QualityConfig()
        self.optimization = OptimizationConfig()
        
        # Configuration files
        self.default_config_file = self.config_dir / "default.json"
        self.environment_config_file = self.config_dir / f"{self.environment.value}.json"
        self.user_config_file = self.config_dir / "user.json"
        
        # Load configurations
        self._load_all_configs()
        
        # Watch for changes
        self._config_hash = self._calculate_config_hash()
        self._watch_for_changes = False
    
    def _detect_environment(self) -> Environment:
        """Detect current environment."""
        env_var = os.getenv("TESTMASTER_ENV", "local").lower()
        try:
            return Environment(env_var)
        except ValueError:
            return Environment.LOCAL
    
    def _load_all_configs(self):
        """Load all configuration sources in order of precedence."""
        # 1. Load defaults
        self._load_defaults()
        
        # 2. Load environment-specific config
        if self.environment_config_file.exists():
            self._load_config_file(self.environment_config_file)
        
        # 3. Load user config
        if self.user_config_file.exists():
            self._load_config_file(self.user_config_file)
        
        # 4. Load environment variables
        self._load_env_vars()
        
        # 5. Validate configuration
        self._validate_config()
    
    def _load_defaults(self):
        """Load or create default configuration."""
        if not self.default_config_file.exists():
            self._save_defaults()
        else:
            self._load_config_file(self.default_config_file)
    
    def _save_defaults(self):
        """Save default configuration to file."""
        defaults = {
            "api": asdict(self.api),
            "generation": asdict(self.generation),
            "monitoring": asdict(self.monitoring),
            "caching": asdict(self.caching),
            "execution": asdict(self.execution),
            "reporting": asdict(self.reporting),
            "quality": asdict(self.quality),
            "optimization": asdict(self.optimization)
        }
        
        # Remove sensitive data from defaults
        defaults["api"]["gemini_api_key"] = None
        defaults["api"]["google_api_key"] = None
        defaults["api"]["openai_api_key"] = None
        defaults["api"]["anthropic_api_key"] = None
        
        with open(self.default_config_file, "w") as f:
            json.dump(defaults, f, indent=2)
    
    def _load_config_file(self, file_path: Path):
        """Load configuration from file."""
        try:
            with open(file_path) as f:
                if file_path.suffix == ".json":
                    config = json.load(f)
                elif file_path.suffix in [".yml", ".yaml"]:
                    config = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported config file format: {file_path}")
                    return
            
            # Update configuration sections
            for section_name, section_data in config.items():
                if hasattr(self, section_name) and isinstance(section_data, dict):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            logger.info(f"Loaded configuration from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
    
    def _load_env_vars(self):
        """Load configuration from environment variables."""
        # Map environment variables to configuration fields
        env_mappings = {
            "TESTMASTER_API_RATE_LIMIT": ("api", "rate_limit_rpm", int),
            "TESTMASTER_MAX_WORKERS": ("generation", "parallel_workers", int),
            "TESTMASTER_QUALITY_THRESHOLD": ("generation", "min_quality_score", int),
            "TESTMASTER_CACHE_ENABLED": ("caching", "enabled", bool),
            "TESTMASTER_CACHE_DIR": ("caching", "cache_directory", str),
            "TESTMASTER_MONITORING_ENABLED": ("monitoring", "enabled", bool),
            "TESTMASTER_MONITORING_INTERVAL": ("monitoring", "interval_minutes", int),
        }
        
        for env_var, (section, field, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    section_obj = getattr(self, section)
                    if type_func == bool:
                        parsed_value = value.lower() in ["true", "1", "yes", "on"]
                    else:
                        parsed_value = type_func(value)
                    setattr(section_obj, field, parsed_value)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to set {section}.{field} from {env_var}: {e}")
    
    def _validate_config(self):
        """Validate configuration values."""
        # API validation
        if not any([self.api.gemini_api_key, self.api.google_api_key, 
                   self.api.openai_api_key, self.api.anthropic_api_key]):
            logger.warning("No API keys configured. Test generation will fail.")
        
        # Generation validation
        if self.generation.max_iterations < 1:
            self.generation.max_iterations = 1
        if self.generation.min_quality_score < 0:
            self.generation.min_quality_score = 0
        if self.generation.min_quality_score > 100:
            self.generation.min_quality_score = 100
        
        # Caching validation
        if self.caching.max_cache_size_mb < 10:
            self.caching.max_cache_size_mb = 10
        
        # Execution validation
        if self.execution.max_parallel_tests < 1:
            self.execution.max_parallel_tests = 1
        
        logger.info("Configuration validated successfully")
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration."""
        config_dict = self.to_dict()
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def reload(self):
        """Reload configuration from files."""
        logger.info("Reloading configuration...")
        self._load_all_configs()
        self._config_hash = self._calculate_config_hash()
    
    def has_changed(self) -> bool:
        """Check if configuration has changed."""
        current_hash = self._calculate_config_hash()
        return current_hash != self._config_hash
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation path."""
        parts = path.split(".")
        obj = self
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        
        return obj
    
    def set(self, path: str, value: Any):
        """Set configuration value by dot-notation path."""
        parts = path.split(".")
        obj = self
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise ValueError(f"Invalid configuration path: {path}")
        
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
        else:
            raise ValueError(f"Invalid configuration field: {parts[-1]}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "api": asdict(self.api),
            "generation": asdict(self.generation),
            "monitoring": asdict(self.monitoring),
            "caching": asdict(self.caching),
            "execution": asdict(self.execution),
            "reporting": asdict(self.reporting),
            "quality": asdict(self.quality),
            "optimization": asdict(self.optimization)
        }
    
    def save(self, file_path: Optional[Path] = None):
        """Save current configuration to file."""
        if file_path is None:
            file_path = self.user_config_file
        
        config_dict = self.to_dict()
        
        # Remove sensitive data if saving to default location
        if file_path == self.user_config_file:
            config_dict["api"]["gemini_api_key"] = None
            config_dict["api"]["google_api_key"] = None
            config_dict["api"]["openai_api_key"] = None
            config_dict["api"]["anthropic_api_key"] = None
        
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are configured."""
        return any([
            self.api.gemini_api_key,
            self.api.google_api_key,
            self.api.openai_api_key,
            self.api.anthropic_api_key
        ])
    
    def get_active_model(self) -> str:
        """Get the currently active AI model."""
        if self.api.gemini_api_key or self.api.google_api_key:
            return self.api.preferred_model
        elif self.api.openai_api_key:
            return "gpt-4"
        elif self.api.anthropic_api_key:
            return "claude-3"
        else:
            return "none"
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*50)
        print("TestMaster Configuration Summary")
        print("="*50)
        print(f"Environment: {self.environment.value}")
        print(f"Active Model: {self.get_active_model()}")
        print(f"\nGeneration Settings:")
        print(f"  Max Iterations: {self.generation.max_iterations}")
        print(f"  Quality Threshold: {self.generation.min_quality_score}")
        print(f"  Parallel Workers: {self.generation.parallel_workers}")
        print(f"  Coverage Target: {self.generation.coverage_target}%")
        print(f"\nMonitoring:")
        print(f"  Enabled: {self.monitoring.enabled}")
        print(f"  Interval: {self.monitoring.interval_minutes} minutes")
        print(f"\nCaching:")
        print(f"  Enabled: {self.caching.enabled}")
        print(f"  Directory: {self.caching.cache_directory}")
        print(f"  TTL: {self.caching.cache_ttl_seconds} seconds")
        print(f"\nExecution:")
        print(f"  Parallel: {self.execution.parallel_execution}")
        print(f"  Max Parallel: {self.execution.max_parallel_tests}")
        print(f"  Timeout: {self.execution.timeout_per_test} seconds")
        print("="*50 + "\n")


# Global configuration instance
config = TestMasterConfig()


def main():
    """CLI for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TestMaster Configuration Manager")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--save", help="Save configuration to file")
    parser.add_argument("--load", help="Load configuration from file")
    parser.add_argument("--get", help="Get configuration value by path")
    parser.add_argument("--set", nargs=2, metavar=("PATH", "VALUE"), help="Set configuration value")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--env", help="Set environment (development, testing, production)")
    
    args = parser.parse_args()
    
    if args.env:
        os.environ["TESTMASTER_ENV"] = args.env
        config.reload()
        print(f"Environment set to: {args.env}")
    
    if args.show:
        config.print_summary()
        print("\nFull configuration:")
        print(json.dumps(config.to_dict(), indent=2))
    
    if args.save:
        config.save(Path(args.save))
        print(f"Configuration saved to {args.save}")
    
    if args.load:
        config._load_config_file(Path(args.load))
        print(f"Configuration loaded from {args.load}")
    
    if args.get:
        value = config.get(args.get)
        print(f"{args.get} = {value}")
    
    if args.set:
        path, value = args.set
        # Try to parse value as JSON first
        try:
            import ast
            value = ast.literal_eval(value)
        except:
            pass  # Keep as string
        
        config.set(path, value)
        print(f"Set {path} = {value}")
    
    if args.validate:
        if config.validate_api_keys():
            print("✓ API keys configured")
        else:
            print("✗ No API keys configured")
        
        config._validate_config()
        print("✓ Configuration is valid")


if __name__ == "__main__":
    main()