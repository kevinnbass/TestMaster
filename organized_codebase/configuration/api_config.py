"""
API Configuration Module
=======================

API-related configuration settings including keys, rate limits, and model preferences.
Modularized from testmaster_config.py and unified_config.py.

Author: Agent E - Infrastructure Consolidation
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .data_models import ConfigBase


@dataclass
class APIConfig(ConfigBase):
    """API configuration settings."""
    
    # API Keys
    gemini_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Rate Limiting
    rate_limit_rpm: int = 30
    rate_limit_daily: int = 10000
    concurrent_requests: int = 5
    
    # Timeouts and Retries
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0
    exponential_backoff: bool = True
    
    # Model Configuration
    preferred_model: str = "gemini-2.5-pro"
    fallback_models: List[str] = field(default_factory=lambda: [
        "gemini-2.0-flash", 
        "gemini-1.5-pro",
        "gpt-4",
        "claude-3"
    ])
    
    # Endpoint Configuration
    base_urls: Dict[str, str] = field(default_factory=lambda: {
        "gemini": "https://generativelanguage.googleapis.com/v1",
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1"
    })
    
    # Request Configuration
    max_tokens: int = 4000
    temperature: float = 0.1
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def __post_init__(self):
        """Load API keys from environment if not set."""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.google_api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    
    def validate(self) -> List[str]:
        """Validate API configuration."""
        errors = []
        
        # Check if at least one API key is configured
        if not any([
            self.gemini_api_key,
            self.google_api_key,
            self.openai_api_key,
            self.anthropic_api_key
        ]):
            errors.append("No API keys configured. At least one API key is required.")
        
        # Validate rate limits
        if self.rate_limit_rpm <= 0:
            errors.append("Rate limit RPM must be positive")
        
        if self.timeout_seconds <= 0:
            errors.append("Timeout seconds must be positive")
        
        if self.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        # Validate model configuration
        if not self.preferred_model:
            errors.append("Preferred model must be specified")
        
        if self.temperature < 0 or self.temperature > 2:
            errors.append("Temperature must be between 0 and 2")
        
        return errors
    
    def get_active_api_key(self) -> Optional[str]:
        """Get the first available API key."""
        for key in [self.gemini_api_key, self.google_api_key, 
                   self.openai_api_key, self.anthropic_api_key]:
            if key:
                return key
        return None
    
    def get_api_config_for_model(self, model: str) -> Dict[str, Any]:
        """Get API configuration for a specific model."""
        config = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout_seconds
        }
        
        # Model-specific adjustments
        if "gemini" in model.lower():
            config["api_key"] = self.gemini_api_key or self.google_api_key
            config["base_url"] = self.base_urls.get("gemini")
        elif "gpt" in model.lower():
            config["api_key"] = self.openai_api_key
            config["base_url"] = self.base_urls.get("openai")
            config["frequency_penalty"] = self.frequency_penalty
            config["presence_penalty"] = self.presence_penalty
        elif "claude" in model.lower():
            config["api_key"] = self.anthropic_api_key
            config["base_url"] = self.base_urls.get("anthropic")
        
        return config


__all__ = ['APIConfig']