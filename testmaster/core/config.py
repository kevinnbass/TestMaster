#!/usr/bin/env python3
"""
TestMaster Configuration Management

Central configuration system for TestMaster.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

class TestMasterConfig:
    """Central configuration management for TestMaster."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration."""
        self.config_file = config_file
        self._config = self._load_default_config()
        
        if config_file and Path(config_file).exists():
            self._load_config_file(config_file)
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "api": {
                "timeout": 120,
                "rate_limit_rpm": 30,
                "max_retries": 3
            },
            "generation": {
                "mode": "auto",
                "quality_threshold": 70.0,
                "max_workers": 4
            },
            "paths": {
                "output_directory": "tests/unit",
                "cache_directory": ".testmaster_cache"
            }
        }
    
    def _load_config_file(self, config_file: str):
        """Load configuration from file."""
        # Placeholder - would load YAML/JSON config
        pass
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        # API settings
        if api_key := os.getenv("GOOGLE_API_KEY"):
            self._config.setdefault("api", {})["key"] = api_key
        
        if timeout := os.getenv("TESTMASTER_TIMEOUT"):
            try:
                self._config["api"]["timeout"] = int(timeout)
            except ValueError:
                pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value