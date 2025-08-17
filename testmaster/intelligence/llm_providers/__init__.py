"""
Universal LLM Provider Management

Provides unified interface for any LLM provider (OpenAI, Anthropic, Local, etc.).
Adapted from Agency Swarm and PraisonAI's provider abstractions.
"""

from .universal_llm_provider import (
    UniversalLLMProvider,
    LLMProviderConfig,
    LLMResponse,
    LLMMessage,
    MessageRole,
    ProviderType,
    LLMProviderManager
)

from .provider_implementations import (
    OpenAIProvider,
    AnthropicProvider,
    LocalLLMProvider,
    AzureOpenAIProvider,
    GoogleProvider,
    OllamaProvider
)

from .provider_optimization import (
    ProviderOptimizer,
    CostOptimizer,
    LatencyOptimizer,
    QualityOptimizer,
    ProviderMetrics
)

__all__ = [
    # Core LLM Management
    'UniversalLLMProvider',
    'LLMProviderConfig',
    'LLMResponse',
    'LLMMessage',
    'MessageRole',
    'ProviderType',
    'LLMProviderManager',
    
    # Provider Implementations
    'OpenAIProvider',
    'AnthropicProvider',
    'LocalLLMProvider',
    'AzureOpenAIProvider',
    'GoogleProvider',
    'OllamaProvider',
    
    # Optimization
    'ProviderOptimizer',
    'CostOptimizer',
    'LatencyOptimizer',
    'QualityOptimizer',
    'ProviderMetrics'
]