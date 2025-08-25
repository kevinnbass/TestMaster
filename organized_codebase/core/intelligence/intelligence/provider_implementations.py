"""
LLM Provider Implementations

Concrete implementations for various LLM providers.
Adapted from Agency Swarm and PraisonAI's provider implementations.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, AsyncIterator
import aiohttp
import requests

from .universal_llm_provider import (
    UniversalLLMProvider, LLMProviderConfig, LLMMessage, LLMResponse,
    MessageRole, ProviderType
)


class OpenAIProvider(UniversalLLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.openai.com/v1"
        
        # Model-specific pricing (per 1K tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        }
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Convert messages to OpenAI format
        openai_messages = [self._convert_message(msg) for msg in messages]
        
        payload = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            **kwargs
        }
        
        # Add stop sequences if configured
        if self.config.stop:
            payload["stop"] = self.config.stop
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
                
                result = await response.json()
        
        response_time = time.time() - start_time
        
        # Extract response
        choice = result["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        
        # Extract usage
        usage = result.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        
        # Calculate cost
        cost = self._calculate_cost(usage)
        
        return LLMResponse(
            content=content,
            provider=self.provider_type.value,
            model=self.config.model,
            usage=usage,
            finish_reason=finish_reason,
            response_time=response_time,
            tokens_used=tokens_used,
            cost_estimate=cost,
            request_id=result.get("id")
        )
    
    def generate_sync(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Synchronous generation."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        openai_messages = [self._convert_message(msg) for msg in messages]
        
        payload = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            **kwargs
        }
        
        if self.config.stop:
            payload["stop"] = self.config.stop
        
        start_time = time.time()
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout
        )
        
        response_time = time.time() - start_time
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error {response.status_code}: {response.text}")
        
        result = response.json()
        
        # Extract response
        choice = result["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        usage = result.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        cost = self._calculate_cost(usage)
        
        return LLMResponse(
            content=content,
            provider=self.provider_type.value,
            model=self.config.model,
            usage=usage,
            finish_reason=finish_reason,
            response_time=response_time,
            tokens_used=tokens_used,
            cost_estimate=cost,
            request_id=result.get("id")
        )
    
    async def stream_generate(self, messages: List[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Stream generate response."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        openai_messages = [self._convert_message(msg) for msg in messages]
        
        payload = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "stream": True,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data != '[DONE]':
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
    
    def _convert_message(self, message: LLMMessage) -> Dict[str, Any]:
        """Convert universal message to OpenAI format."""
        return {
            "role": message.role.value,
            "content": message.content
        }
    
    def _calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculate cost based on usage."""
        if self.config.model not in self.pricing:
            return super()._estimate_cost(usage.get("total_tokens", 0))
        
        pricing = self.pricing[self.config.model]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost


class AnthropicProvider(UniversalLLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.anthropic.com/v1"
        
        # Anthropic pricing
        self.pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages to Anthropic format
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        payload = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens or 1000,
            "temperature": self.config.temperature,
            **kwargs
        }
        
        if system_message:
            payload["system"] = system_message
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_text}")
                
                result = await response.json()
        
        response_time = time.time() - start_time
        
        # Extract response
        content = result["content"][0]["text"]
        usage = result.get("usage", {})
        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        cost = self._calculate_cost(usage)
        
        return LLMResponse(
            content=content,
            provider=self.provider_type.value,
            model=self.config.model,
            usage=usage,
            response_time=response_time,
            tokens_used=tokens_used,
            cost_estimate=cost,
            request_id=result.get("id")
        )
    
    def generate_sync(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Synchronous generation."""
        return asyncio.run(self.generate(messages, **kwargs))
    
    async def stream_generate(self, messages: List[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Stream generate (simplified implementation)."""
        response = await self.generate(messages, **kwargs)
        # Anthropic streaming would need proper SSE handling
        yield response.content
    
    def _calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculate cost for Anthropic."""
        if self.config.model not in self.pricing:
            return super()._estimate_cost(usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
        
        pricing = self.pricing[self.config.model]
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost


class LocalLLMProvider(UniversalLLMProvider):
    """Local LLM provider (for local models via API)."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base or "http://localhost:11434"  # Default Ollama port
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate using local LLM API."""
        # Convert messages to simple prompt
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens or -1,
            }
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Local LLM API error {response.status}: {error_text}")
                
                result = await response.json()
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=result.get("response", ""),
            provider=self.provider_type.value,
            model=self.config.model,
            response_time=response_time,
            tokens_used=result.get("eval_count", 0),
            cost_estimate=0.0  # Local models are free
        )
    
    def generate_sync(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Synchronous generation."""
        return asyncio.run(self.generate(messages, **kwargs))
    
    async def stream_generate(self, messages: List[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Stream generate."""
        response = await self.generate(messages, **kwargs)
        yield response.content
    
    def _messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert messages to simple prompt format."""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        
        # Azure requires different URL structure
        if not config.api_base:
            raise ValueError("Azure OpenAI requires api_base URL")
        
        self.api_base = config.api_base.rstrip('/')
        self.api_version = config.api_version or "2023-12-01-preview"
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate using Azure OpenAI."""
        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json"
        }
        
        # Convert messages
        openai_messages = [self._convert_message(msg) for msg in messages]
        
        payload = {
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        
        # Azure URL includes deployment name
        url = f"{self.api_base}/openai/deployments/{self.config.model}/chat/completions?api-version={self.api_version}"
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Azure OpenAI API error {response.status}: {error_text}")
                
                result = await response.json()
        
        response_time = time.time() - start_time
        
        # Process similar to OpenAI
        choice = result["choices"][0]
        content = choice["message"]["content"]
        usage = result.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        cost = self._calculate_cost(usage)
        
        return LLMResponse(
            content=content,
            provider=self.provider_type.value,
            model=self.config.model,
            usage=usage,
            response_time=response_time,
            tokens_used=tokens_used,
            cost_estimate=cost,
            request_id=result.get("id")
        )


class GoogleProvider(UniversalLLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://generativelanguage.googleapis.com/v1beta"
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate using Google Gemini API."""
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg.role == MessageRole.USER else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
                "maxOutputTokens": self.config.max_tokens or 1000,
            }
        }
        
        url = f"{self.api_base}/models/{self.config.model}:generateContent?key={self.config.api_key}"
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Google API error {response.status}: {error_text}")
                
                result = await response.json()
        
        response_time = time.time() - start_time
        
        # Extract response
        if "candidates" in result and result["candidates"]:
            content = result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            content = ""
        
        # Google doesn't provide detailed usage, estimate
        estimated_tokens = len(content.split())
        
        return LLMResponse(
            content=content,
            provider=self.provider_type.value,
            model=self.config.model,
            response_time=response_time,
            tokens_used=estimated_tokens,
            cost_estimate=self._estimate_cost(estimated_tokens)
        )
    
    def generate_sync(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Synchronous generation."""
        return asyncio.run(self.generate(messages, **kwargs))
    
    async def stream_generate(self, messages: List[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Stream generate."""
        response = await self.generate(messages, **kwargs)
        yield response.content


class OllamaProvider(LocalLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base or "http://localhost:11434"
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate using Ollama API."""
        # Ollama supports chat format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        payload = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens or -1,
            }
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                
                result = await response.json()
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=result["message"]["content"],
            provider=self.provider_type.value,
            model=self.config.model,
            response_time=response_time,
            tokens_used=result.get("eval_count", 0),
            cost_estimate=0.0  # Local models are free
        )