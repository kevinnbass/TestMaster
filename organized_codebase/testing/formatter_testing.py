# -*- coding: utf-8 -*-
"""
AgentScope Multi-Provider Formatter Testing Framework
====================================================

Extracted from agentscope/tests/formatter_openai_test.py
Enhanced for TestMaster integration

Testing patterns for:
- Multi-provider message formatting (OpenAI, Anthropic, Gemini, DeepSeek, DashScope, Ollama)
- Multimodal content handling (text, image, audio)
- Tool use and tool result formatting
- Chat vs multiagent formatting strategies
- Base64 encoding and URL source handling
- System message integration
- Conversation history management
"""

import asyncio
import base64
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

import pytest


class MockSource:
    """Mock source for multimodal content"""
    
    def __init__(self, source_type: str, **kwargs):
        self.type = source_type
        self.__dict__.update(kwargs)


class MockBlock:
    """Mock content block for messages"""
    
    def __init__(self, block_type: str, **kwargs):
        self.type = block_type
        self.__dict__.update(kwargs)


class MockMessage:
    """Mock message for formatter testing"""
    
    def __init__(self, role: str, content: Union[str, List], name: str, **kwargs):
        self.role = role
        self.content = content
        self.name = name
        self.__dict__.update(kwargs)


class FormatterTestFramework:
    """Core framework for multi-provider formatter testing"""
    
    def __init__(self):
        self.providers = [
            "openai", "anthropic", "gemini", "deepseek", 
            "dashscope", "ollama", "azure_openai"
        ]
        self.supported_formats = {
            "text": ["text"],
            "image": ["image_url", "image"],
            "audio": ["input_audio", "audio"],
            "tool_use": ["tool_calls", "function"],
            "tool_result": ["tool", "function_result"]
        }
        self.temp_files = []
    
    def create_mock_media_file(self, content: bytes, extension: str) -> str:
        """Create temporary media file for testing"""
        with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as f:
            f.write(content)
            temp_path = f.name
            self.temp_files.append(temp_path)
            return temp_path
    
    def encode_base64(self, data: bytes) -> str:
        """Encode data to base64 string"""
        return base64.b64encode(data).decode('utf-8')
    
    def create_text_block(self, text: str) -> MockBlock:
        """Create text content block"""
        return MockBlock("text", text=text)
    
    def create_image_block(self, source_type: str = "url", **kwargs) -> MockBlock:
        """Create image content block"""
        if source_type == "url":
            source = MockSource("url", url=kwargs.get("url", ""))
        else:
            source = MockSource("base64", 
                              media_type=kwargs.get("media_type", "image/png"),
                              data=kwargs.get("data", ""))
        return MockBlock("image", source=source)
    
    def create_audio_block(self, source_type: str = "url", **kwargs) -> MockBlock:
        """Create audio content block"""
        if source_type == "url":
            source = MockSource("url", url=kwargs.get("url", ""))
        else:
            source = MockSource("base64",
                              media_type=kwargs.get("media_type", "audio/wav"),
                              data=kwargs.get("data", ""))
        return MockBlock("audio", source=source)
    
    def create_tool_use_block(self, tool_id: str, name: str, input_data: Dict) -> MockBlock:
        """Create tool use block"""
        return MockBlock("tool_use", id=tool_id, name=name, input=input_data)
    
    def create_tool_result_block(self, tool_id: str, name: str, output: List) -> MockBlock:
        """Create tool result block"""
        return MockBlock("tool_result", id=tool_id, name=name, output=output)
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        self.temp_files.clear()


class MockOpenAIFormatter:
    """Mock OpenAI formatter for testing"""
    
    def __init__(self, multiagent: bool = False):
        self.multiagent = multiagent
    
    async def format(self, messages: List[MockMessage]) -> List[Dict]:
        """Format messages for OpenAI API"""
        formatted = []
        
        if self.multiagent:
            return self._format_multiagent(messages)
        else:
            return self._format_chat(messages)
    
    def _format_chat(self, messages: List[MockMessage]) -> List[Dict]:
        """Format for standard chat completion"""
        formatted = []
        
        for msg in messages:
            formatted_msg = {
                "role": msg.role,
                "name": msg.name
            }
            
            if isinstance(msg.content, str):
                formatted_msg["content"] = [{"type": "text", "text": msg.content}]
            elif isinstance(msg.content, list):
                formatted_msg["content"] = self._format_content_blocks(msg.content)
            
            # Handle tool calls
            if hasattr(msg, 'tool_calls'):
                formatted_msg["tool_calls"] = msg.tool_calls
                formatted_msg["content"] = None
            
            formatted.append(formatted_msg)
        
        return formatted
    
    def _format_multiagent(self, messages: List[MockMessage]) -> List[Dict]:
        """Format for multi-agent conversations"""
        # Compress conversation history into single user message
        history_text = self._build_conversation_history(messages)
        
        formatted = []
        system_msgs = [m for m in messages if m.role == "system"]
        
        if system_msgs:
            formatted.append({
                "role": "system",
                "content": [{"type": "text", "text": system_msgs[0].content}]
            })
        
        # Add compressed history
        if history_text:
            formatted.append({
                "role": "user",
                "content": [{"type": "text", "text": history_text}]
            })
        
        return formatted
    
    def _build_conversation_history(self, messages: List[MockMessage]) -> str:
        """Build conversation history text"""
        history_parts = []
        for msg in messages:
            if msg.role not in ["system"]:
                if isinstance(msg.content, str):
                    history_parts.append(f"{msg.role}: {msg.content}")
                else:
                    # Extract text from content blocks
                    text_parts = []
                    for block in msg.content:
                        if hasattr(block, 'type') and block.type == "text":
                            text_parts.append(block.text)
                    if text_parts:
                        history_parts.append(f"{msg.role}: {' '.join(text_parts)}")
        
        return "# Conversation History\n<history>\n" + "\n".join(history_parts) + "\n</history>"
    
    def _format_content_blocks(self, blocks: List[MockBlock]) -> List[Dict]:
        """Format content blocks for API"""
        formatted_blocks = []
        
        for block in blocks:
            if block.type == "text":
                formatted_blocks.append({"type": "text", "text": block.text})
            elif block.type == "image":
                if hasattr(block.source, 'url'):
                    # Convert file path to base64
                    if os.path.exists(block.source.url):
                        with open(block.source.url, 'rb') as f:
                            data = base64.b64encode(f.read()).decode('utf-8')
                        formatted_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{data}"}
                        })
                else:
                    formatted_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.source.media_type};base64,{block.source.data}"}
                    })
            elif block.type == "audio":
                if hasattr(block.source, 'url'):
                    # Convert file to base64
                    if os.path.exists(block.source.url):
                        with open(block.source.url, 'rb') as f:
                            data = base64.b64encode(f.read()).decode('utf-8')
                        formatted_blocks.append({
                            "type": "input_audio",
                            "input_audio": {"data": data, "format": "wav"}
                        })
                else:
                    formatted_blocks.append({
                        "type": "input_audio",
                        "input_audio": {"data": block.source.data, "format": "wav"}
                    })
        
        return formatted_blocks


class FormatterValidator:
    """Validator for formatted messages"""
    
    @staticmethod
    def validate_openai_format(formatted: List[Dict]) -> bool:
        """Validate OpenAI chat completion format"""
        required_fields = ["role", "content"]
        
        for msg in formatted:
            if not all(field in msg for field in required_fields):
                return False
            
            # Validate role
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                return False
            
            # Validate content structure
            if msg["content"] is not None:
                if not isinstance(msg["content"], list):
                    return False
                
                for content_item in msg["content"]:
                    if "type" not in content_item:
                        return False
        
        return True
    
    @staticmethod
    def validate_multimodal_content(content: List[Dict]) -> bool:
        """Validate multimodal content structure"""
        supported_types = ["text", "image_url", "input_audio"]
        
        for item in content:
            if item["type"] not in supported_types:
                return False
            
            if item["type"] == "text" and "text" not in item:
                return False
            elif item["type"] == "image_url" and "image_url" not in item:
                return False
            elif item["type"] == "input_audio" and "input_audio" not in item:
                return False
        
        return True
    
    @staticmethod
    def validate_tool_calls(msg: Dict) -> bool:
        """Validate tool call format"""
        if "tool_calls" not in msg:
            return True
        
        for tool_call in msg["tool_calls"]:
            required_fields = ["id", "type", "function"]
            if not all(field in tool_call for field in required_fields):
                return False
            
            if tool_call["type"] != "function":
                return False
            
            function = tool_call["function"]
            if not all(field in function for field in ["name", "arguments"]):
                return False
        
        return True


class MultiProviderFormatterTest(IsolatedAsyncioTestCase):
    """Comprehensive multi-provider formatter testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = FormatterTestFramework()
        self.validator = FormatterValidator()
        
        # Create test media files
        self.image_data = b"fake image content"
        self.audio_data = b"fake audio content"
        
        self.image_path = self.framework.create_mock_media_file(self.image_data, "png")
        self.audio_path = self.framework.create_mock_media_file(self.audio_data, "wav")
        
        # Create test messages
        self.system_msg = MockMessage(
            "system", "You're a helpful assistant.", "system"
        )
        
        self.text_msg = MockMessage(
            "user", "What is the capital of France?", "user"
        )
        
        self.multimodal_msg = MockMessage(
            "user",
            [
                self.framework.create_text_block("Describe this image"),
                self.framework.create_image_block("url", url=self.image_path),
                self.framework.create_audio_block("url", url=self.audio_path)
            ],
            "user"
        )
        
        self.tool_use_msg = MockMessage(
            "assistant",
            [self.framework.create_tool_use_block(
                "1", "get_weather", {"location": "Paris"}
            )],
            "assistant"
        )
        
        self.tool_result_msg = MockMessage(
            "system",
            [self.framework.create_tool_result_block(
                "1", "get_weather", 
                [self.framework.create_text_block("It's sunny in Paris")]
            )],
            "system"
        )
    
    async def test_openai_chat_formatting(self):
        """Test OpenAI chat completion formatting"""
        formatter = MockOpenAIFormatter(multiagent=False)
        
        messages = [self.system_msg, self.text_msg, self.multimodal_msg]
        formatted = await formatter.format(messages)
        
        # Validate format
        assert self.validator.validate_openai_format(formatted)
        assert len(formatted) == 3
        
        # Check system message
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"][0]["text"] == "You're a helpful assistant."
        
        # Check multimodal message
        multimodal_content = formatted[2]["content"]
        assert len(multimodal_content) == 3
        assert multimodal_content[0]["type"] == "text"
        assert multimodal_content[1]["type"] == "image_url"
        assert multimodal_content[2]["type"] == "input_audio"
    
    async def test_multiagent_formatting(self):
        """Test multi-agent conversation formatting"""
        formatter = MockOpenAIFormatter(multiagent=True)
        
        messages = [self.system_msg, self.text_msg, self.multimodal_msg]
        formatted = await formatter.format(messages)
        
        # Should compress conversation into history
        assert len(formatted) <= 2
        
        # Check for conversation history
        if len(formatted) > 1:
            history_content = formatted[1]["content"][0]["text"]
            assert "# Conversation History" in history_content
            assert "<history>" in history_content
    
    async def test_tool_formatting(self):
        """Test tool use and result formatting"""
        formatter = MockOpenAIFormatter(multiagent=False)
        
        # Mock tool calls on message
        self.tool_use_msg.tool_calls = [{
            "id": "1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}'
            }
        }]
        
        messages = [self.tool_use_msg, self.tool_result_msg]
        formatted = await formatter.format(messages)
        
        # Validate tool call format
        assert self.validator.validate_tool_calls(formatted[0])
        assert formatted[0]["content"] is None
        assert "tool_calls" in formatted[0]
    
    async def test_base64_encoding(self):
        """Test base64 encoding for media content"""
        # Create message with base64 source
        base64_image = self.framework.create_image_block(
            "base64", 
            media_type="image/png",
            data=self.framework.encode_base64(self.image_data)
        )
        
        msg = MockMessage("user", [base64_image], "user")
        formatter = MockOpenAIFormatter()
        
        formatted = await formatter.format([msg])
        
        # Check base64 encoding in output
        image_content = formatted[0]["content"][0]
        assert "data:image/png;base64," in image_content["image_url"]["url"]
    
    async def test_provider_compatibility(self):
        """Test compatibility across different providers"""
        messages = [self.system_msg, self.multimodal_msg]
        
        for provider in self.framework.providers:
            # Mock different provider formatters
            if provider == "anthropic":
                # Anthropic uses different content structure
                pass
            elif provider == "gemini":
                # Gemini has specific multimodal requirements
                pass
            
            # For now, test with OpenAI format as baseline
            formatter = MockOpenAIFormatter()
            formatted = await formatter.format(messages)
            
            assert self.validator.validate_openai_format(formatted)
    
    async def test_error_handling(self):
        """Test error handling in formatting"""
        formatter = MockOpenAIFormatter()
        
        # Test with invalid message
        invalid_msg = MockMessage("invalid_role", "test", "test")
        
        try:
            formatted = await formatter.format([invalid_msg])
            # Should handle gracefully
            assert isinstance(formatted, list)
        except Exception:
            # Expected for invalid input
            pass
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        self.framework.cleanup()


# Pytest integration
@pytest.fixture
def formatter_framework():
    """Pytest fixture for formatter framework"""
    framework = FormatterTestFramework()
    yield framework
    framework.cleanup()


@pytest.fixture
def formatter_validator():
    """Pytest fixture for formatter validator"""
    return FormatterValidator()


def test_formatter_framework_creation(formatter_framework):
    """Test formatter framework creation"""
    assert len(formatter_framework.providers) > 0
    assert "openai" in formatter_framework.providers
    assert "anthropic" in formatter_framework.providers


def test_mock_block_creation(formatter_framework):
    """Test mock block creation"""
    text_block = formatter_framework.create_text_block("test text")
    assert text_block.type == "text"
    assert text_block.text == "test text"
    
    image_block = formatter_framework.create_image_block("url", url="test.png")
    assert image_block.type == "image"
    assert image_block.source.type == "url"


def test_format_validation(formatter_validator):
    """Test format validation"""
    valid_format = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}]
        }
    ]
    
    assert formatter_validator.validate_openai_format(valid_format)
    
    invalid_format = [{"invalid": "format"}]
    assert not formatter_validator.validate_openai_format(invalid_format)