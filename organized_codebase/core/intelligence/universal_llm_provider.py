"""
Universal LLM Provider Interface

Provides unified interface for any LLM provider.
Adapted from Agency Swarm's provider abstraction and PraisonAI's agent communication.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, AsyncIterator, Iterator
from enum import Enum
from abc import ABC, abstractmethod
import time
import json
from datetime import datetime
import asyncio


class MessageRole(Enum):
    """Message roles for LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ProviderType(Enum):
    """Types of LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    LOCAL = "local"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


@dataclass
class LLMMessage:
    """Universal message format for LLM communication."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            'role': self.role.value,
            'content': self.content
        }
        
        if self.name:
            result['name'] = self.name
        if self.function_call:
            result['function_call'] = self.function_call
        if self.tool_calls:
            result['tool_calls'] = self.tool_calls
        if self.metadata:
            result['metadata'] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMMessage':
        """Create from dictionary."""
        return cls(
            role=MessageRole(data['role']),
            content=data['content'],
            name=data.get('name'),
            function_call=data.get('function_call'),
            tool_calls=data.get('tool_calls'),
            metadata=data.get('metadata', {})
        )


@dataclass
class LLMResponse:
    """Universal response format from LLM providers."""
    content: str
    provider: str
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    response_time: float = 0.0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'provider': self.provider,
            'model': self.model,
            'usage': self.usage,
            'finish_reason': self.finish_reason,
            'function_calls': self.function_calls,
            'tool_calls': self.tool_calls,
            'response_time': self.response_time,
            'tokens_used': self.tokens_used,
            'cost_estimate': self.cost_estimate,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id,
            'metadata': self.metadata
        }


@dataclass
class LLMProviderConfig:
    """Configuration for LLM providers."""
    provider_type: ProviderType
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    
    # Retry and timeout
    max_retries: int = 3
    timeout: float = 30.0
    retry_delay: float = 1.0
    
    # Cost and rate limiting
    max_cost_per_request: float = 1.0
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    
    # Provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'provider_type': self.provider_type.value,
            'model': self.model,
            'api_key': '***' if self.api_key else None,  # Hide sensitive data
            'api_base': self.api_base,
            'api_version': self.api_version,
            'organization': self.organization,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stop': self.stop,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'retry_delay': self.retry_delay,
            'max_cost_per_request': self.max_cost_per_request,
            'rate_limit_rpm': self.rate_limit_rpm,
            'rate_limit_tpm': self.rate_limit_tpm,
            'extra_params': self.extra_params
        }


class UniversalLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.provider_type = config.provider_type
        self.model = config.model
        
        # Metrics tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.avg_response_time = 0.0
        self.error_count = 0
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_times = []
        
        print(f"{self.provider_type.value.title()} Provider initialized")
        print(f"   Model: {self.model}")
    
    @abstractmethod
    async def generate(self, 
                      messages: List[LLMMessage],
                      **kwargs) -> LLMResponse:
        """Generate response from messages."""
        pass
    
    @abstractmethod
    def generate_sync(self, 
                     messages: List[LLMMessage],
                     **kwargs) -> LLMResponse:
        """Synchronous generate method."""
        pass
    
    @abstractmethod
    async def stream_generate(self, 
                             messages: List[LLMMessage],
                             **kwargs) -> AsyncIterator[str]:
        """Stream generate response."""
        pass
    
    def validate_request(self, messages: List[LLMMessage]) -> bool:
        """Validate request before sending."""
        # Check rate limits
        if not self._check_rate_limits():
            return False
        
        # Check token limits
        estimated_tokens = self._estimate_tokens(messages)
        if self.config.max_tokens and estimated_tokens > self.config.max_tokens:
            return False
        
        # Check cost limits
        estimated_cost = self._estimate_cost(estimated_tokens)
        if estimated_cost > self.config.max_cost_per_request:
            return False
        
        return True
    
    def _check_rate_limits(self) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        minute_ago = current_time - 60
        self._request_times = [t for t in self._request_times if t > minute_ago]
        
        # Check RPM limit
        if self.config.rate_limit_rpm:
            if len(self._request_times) >= self.config.rate_limit_rpm:
                return False
        
        return True
    
    def _estimate_tokens(self, messages: List[LLMMessage]) -> int:
        """Estimate tokens for messages."""
        # Rough estimation: 4 characters per token
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost for tokens."""
        # Default cost estimation (should be overridden by providers)
        cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens (GPT-3.5 rate)
        return (tokens / 1000) * cost_per_1k_tokens
    
    def _record_request(self, response: LLMResponse):
        """Record request metrics."""
        self.request_count += 1
        self.total_tokens += response.tokens_used
        self.total_cost += response.cost_estimate
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.request_count - 1) + response.response_time) 
            / self.request_count
        )
        
        # Record request time for rate limiting
        self._request_times.append(time.time())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics."""
        return {
            'provider': self.provider_type.value,
            'model': self.model,
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_response_time': self.avg_response_time,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1)
        }
    
    def reset_metrics(self):
        """Reset metrics tracking."""
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.avg_response_time = 0.0
        self.error_count = 0


class LLMProviderManager:
    """Manages multiple LLM providers and provides unified interface."""
    
    def __init__(self):
        self.providers: Dict[str, UniversalLLMProvider] = {}
        self.primary_provider: Optional[str] = None
        self.fallback_providers: List[str] = []
        
        # Load balancing
        self.load_balancing_strategy = "round_robin"  # round_robin, least_cost, fastest
        self._request_counter = 0
        
        print("LLM Provider Manager initialized")
    
    def register_provider(self, name: str, provider: UniversalLLMProvider, 
                         is_primary: bool = False):
        """Register a new provider."""
        self.providers[name] = provider
        
        if is_primary or not self.primary_provider:
            self.primary_provider = name
        
        print(f"   Registered provider: {name} ({provider.provider_type.value})")
    
    def set_fallback_order(self, provider_names: List[str]):
        """Set fallback provider order."""
        # Validate all providers exist
        for name in provider_names:
            if name not in self.providers:
                raise ValueError(f"Provider '{name}' not registered")
        
        self.fallback_providers = provider_names
        print(f"   Fallback order set: {' -> '.join(provider_names)}")
    
    async def generate(self, 
                      messages: List[LLMMessage],
                      preferred_provider: Optional[str] = None,
                      **kwargs) -> LLMResponse:
        """Generate response using best available provider."""
        
        # Determine provider order
        if preferred_provider and preferred_provider in self.providers:
            provider_order = [preferred_provider] + self.fallback_providers
        else:
            provider_order = self._get_provider_order()
        
        last_error = None
        
        # Try providers in order
        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            
            try:
                # Validate request
                if not provider.validate_request(messages):
                    print(f"   Provider {provider_name} rejected request (rate limit/cost)")
                    continue
                
                # Generate response
                start_time = time.time()
                response = await provider.generate(messages, **kwargs)
                response.response_time = time.time() - start_time
                
                # Record metrics
                provider._record_request(response)
                
                print(f"   Generated response using {provider_name}")
                return response
                
            except Exception as e:
                last_error = e
                provider.error_count += 1
                print(f"   Provider {provider_name} failed: {str(e)}")
                continue
        
        # All providers failed
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def generate_sync(self, 
                     messages: List[LLMMessage],
                     preferred_provider: Optional[str] = None,
                     **kwargs) -> LLMResponse:
        """Synchronous generate method."""
        return asyncio.run(self.generate(messages, preferred_provider, **kwargs))
    
    def _get_provider_order(self) -> List[str]:
        """Get provider order based on load balancing strategy."""
        available_providers = list(self.providers.keys())
        
        if self.load_balancing_strategy == "round_robin":
            # Round robin selection
            self._request_counter += 1
            start_index = self._request_counter % len(available_providers)
            return available_providers[start_index:] + available_providers[:start_index]
        
        elif self.load_balancing_strategy == "least_cost":
            # Sort by lowest average cost
            return sorted(available_providers, 
                         key=lambda p: self.providers[p].total_cost / max(self.providers[p].request_count, 1))
        
        elif self.load_balancing_strategy == "fastest":
            # Sort by fastest average response time
            return sorted(available_providers,
                         key=lambda p: self.providers[p].avg_response_time)
        
        else:
            # Default to primary + fallbacks
            if self.primary_provider:
                return [self.primary_provider] + self.fallback_providers
            return available_providers
    
    def get_provider_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all providers."""
        return {name: provider.get_metrics() 
                for name, provider in self.providers.items()}
    
    def get_best_provider(self, criteria: str = "cost") -> Optional[str]:
        """Get best provider based on criteria."""
        if not self.providers:
            return None
        
        if criteria == "cost":
            return min(self.providers.keys(),
                      key=lambda p: self.providers[p].total_cost / max(self.providers[p].request_count, 1))
        elif criteria == "speed":
            return min(self.providers.keys(),
                      key=lambda p: self.providers[p].avg_response_time)
        elif criteria == "reliability":
            return min(self.providers.keys(),
                      key=lambda p: self.providers[p].error_count / max(self.providers[p].request_count, 1))
        else:
            return self.primary_provider
    
    def optimize_provider_selection(self, objective: str = "balanced"):
        """Optimize provider selection based on metrics."""
        metrics = self.get_provider_metrics()
        
        if objective == "cost":
            self.load_balancing_strategy = "least_cost"
        elif objective == "speed":
            self.load_balancing_strategy = "fastest"
        elif objective == "balanced":
            # Use round robin for balanced load
            self.load_balancing_strategy = "round_robin"
        
        print(f"   Provider selection optimized for: {objective}")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_status = {}
        
        for name, provider in self.providers.items():
            try:
                # Simple health check with minimal message
                test_messages = [
                    LLMMessage(role=MessageRole.USER, content="Hi")
                ]
                
                # Quick validation check
                is_healthy = provider.validate_request(test_messages)
                health_status[name] = is_healthy
                
            except Exception:
                health_status[name] = False
        
        return health_status