"""
Agency-Swarm Test Patterns - AGENT B COMPREHENSIVE TESTING EXCELLENCE
======================================================================

Extracted testing patterns from agency-swarm repository for enhanced testing capabilities.
Focus: Async/sync coordination, communication testing, timeouts, and multi-agent coordination.

AGENT B Enhancement: Phase 1.1 - Agency-Swarm Pattern Integration
- Async/sync testing patterns
- Communication validation frameworks
- Multi-agent coordination testing
- Timeout and retry mechanisms
"""

import asyncio
import time
import pytest
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging


class AsyncTestPatterns:
    """
    Async testing patterns extracted from agency-swarm test_sync_async.py
    """
    
    @staticmethod
    async def test_async_function(func: Callable, *args, **kwargs) -> Any:
        """Test async function with basic validation"""
        await asyncio.sleep(0)  # Yield control
        return await func(*args, **kwargs)
    
    @staticmethod
    def test_run_async_sync_basic(func: Callable, *args, expected_result: Any = None) -> bool:
        """Test async function in sync context"""
        from agency_swarm.util.helpers import run_async_sync
        result = run_async_sync(func, *args)
        if expected_result is not None:
            return result == expected_result
        return result is not None
    
    @staticmethod
    @pytest.mark.asyncio
    async def test_run_async_sync_in_running_loop(func: Callable, *args) -> bool:
        """Test that running async in running loop raises RuntimeError"""
        from agency_swarm.util.helpers import run_async_sync
        try:
            run_async_sync(func, *args)
            return False
        except RuntimeError:
            return True


@dataclass
class CommunicationTestConfig:
    """Configuration for communication testing patterns"""
    timeout: int = 30
    retry_interval: int = 1
    max_retries: int = 30
    expected_keywords: List[str] = None
    forbidden_keywords: List[str] = None


class CommunicationTestPatterns:
    """
    Communication testing patterns extracted from agency-swarm test_communication.py
    """
    
    def __init__(self, config: CommunicationTestConfig = None):
        self.config = config or CommunicationTestConfig()
        self.logger = logging.getLogger(__name__)
    
    def test_message_routing_with_timeout(self, 
                                        agency, 
                                        message: str,
                                        expected_agent = None) -> Dict[str, Any]:
        """Test message routing with timeout and retry logic"""
        start_time = time.time()
        response = None
        attempts = 0
        
        while time.time() - start_time < self.config.timeout:
            try:
                response = agency.get_completion(message)
                attempts += 1
                break
            except Exception as e:
                self.logger.warning(f"Attempt {attempts + 1} failed: {e}")
                attempts += 1
                time.sleep(self.config.retry_interval)
                continue
        
        result = {
            'response': response,
            'attempts': attempts,
            'duration': time.time() - start_time,
            'success': response is not None
        }
        
        if response is None:
            result['error'] = f"Test timed out after {self.config.timeout} seconds"
        
        # Validate response content
        if response and self.config.forbidden_keywords:
            for keyword in self.config.forbidden_keywords:
                if keyword.lower() in response.lower():
                    result['validation_error'] = f"Forbidden keyword '{keyword}' found in response"
                    result['success'] = False
        
        if response and self.config.expected_keywords:
            for keyword in self.config.expected_keywords:
                if keyword.lower() not in response.lower():
                    result['validation_error'] = f"Expected keyword '{keyword}' not found in response"
                    result['success'] = False
        
        # Validate recipient agent
        if expected_agent and hasattr(agency, 'main_thread'):
            if agency.main_thread.recipient_agent != expected_agent:
                result['routing_error'] = f"Message routed to wrong agent"
                result['success'] = False
        
        return result
    
    def test_conversation_continuity(self, agency, messages: List[str]) -> Dict[str, Any]:
        """Test conversation continuity across multiple messages"""
        results = []
        thread_info = {}
        
        for i, message in enumerate(messages):
            response = agency.get_completion(message)
            results.append({
                'message_index': i,
                'message': message,
                'response': response,
                'timestamp': time.time()
            })
        
        # Validate thread continuity
        if hasattr(agency, 'main_thread'):
            main_thread = agency.main_thread
            thread_info = {
                'thread_id': getattr(main_thread, 'thread_url', None),
                'message_count': len(main_thread.get_messages()) if hasattr(main_thread, 'get_messages') else 0,
                'recipient_agent': main_thread.recipient_agent if hasattr(main_thread, 'recipient_agent') else None
            }
        
        return {
            'conversation': results,
            'thread_info': thread_info,
            'success': len(results) == len(messages)
        }
    
    def test_concurrent_routing_error_detection(self, agency, concurrent_message: str) -> Dict[str, Any]:
        """Test detection of concurrent routing errors"""
        response = agency.get_completion(concurrent_message)
        
        # Check for error detection
        has_error = "error" in response.lower()
        has_fatal = "fatal" in response.lower()
        
        return {
            'response': response,
            'detected_error': has_error,
            'detected_fatal': has_fatal,
            'success': has_error and not has_fatal  # Should detect error but not be fatal
        }


class MCPTestPatterns:
    """
    MCP (Model Context Protocol) testing patterns
    """
    
    @staticmethod
    def create_mock_mcp_tool():
        """Create mock MCP tool for testing"""
        from pydantic import Field
        from agency_swarm.tools import BaseTool
        
        class MockMCPTool(BaseTool):
            """Mock MCP tool for testing"""
            message: str = Field(..., description="The message to process.")
            
            def run(self):
                return f"MCP Processed: {self.message}"
        
        return MockMCPTool
    
    @staticmethod
    def test_mcp_tool_execution(tool_class, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test MCP tool execution"""
        try:
            tool_instance = tool_class(**test_data)
            result = tool_instance.run()
            return {
                'success': True,
                'result': result,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }


class ResponseValidationPatterns:
    """
    Response validation testing patterns
    """
    
    def __init__(self):
        self.validation_rules = {}
    
    def add_validation_rule(self, name: str, validator: Callable[[str], bool]):
        """Add a validation rule"""
        self.validation_rules[name] = validator
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """Validate response against all rules"""
        results = {}
        overall_success = True
        
        for rule_name, validator in self.validation_rules.items():
            try:
                is_valid = validator(response)
                results[rule_name] = {
                    'valid': is_valid,
                    'error': None
                }
                if not is_valid:
                    overall_success = False
            except Exception as e:
                results[rule_name] = {
                    'valid': False,
                    'error': str(e)
                }
                overall_success = False
        
        return {
            'overall_success': overall_success,
            'rule_results': results,
            'response_length': len(response),
            'response_preview': response[:100] + "..." if len(response) > 100 else response
        }


class ThreadRetryPatterns:
    """
    Thread retry testing patterns
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def test_with_exponential_backoff(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Test with exponential backoff retry pattern"""
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return {
                    'success': True,
                    'result': result,
                    'attempts': attempt + 1,
                    'error': None
                }
            except Exception as e:
                if attempt == self.max_retries:
                    return {
                        'success': False,
                        'result': None,
                        'attempts': attempt + 1,
                        'error': str(e)
                    }
                
                delay = self.base_delay * (2 ** attempt)
                time.sleep(delay)
        
        return {
            'success': False,
            'result': None,
            'attempts': self.max_retries + 1,
            'error': 'Max retries exceeded'
        }


class ToolFactoryPatterns:
    """
    Tool factory testing patterns
    """
    
    @staticmethod
    def create_test_tool(name: str, description: str, fields: Dict[str, Any]):
        """Create a test tool dynamically"""
        from pydantic import Field
        from agency_swarm.tools import BaseTool
        
        # Create field annotations
        annotations = {}
        defaults = {}
        
        for field_name, field_config in fields.items():
            field_type = field_config.get('type', str)
            field_desc = field_config.get('description', f'{field_name} field')
            field_default = field_config.get('default', ...)
            
            annotations[field_name] = field_type
            defaults[field_name] = Field(field_default, description=field_desc)
        
        # Create tool class dynamically
        class_name = f"Test{name}Tool"
        tool_class = type(class_name, (BaseTool,), {
            '__doc__': description,
            '__annotations__': annotations,
            **defaults,
            'run': lambda self: f"Executed {name} with data: {self.dict()}"
        })
        
        return tool_class
    
    @staticmethod
    def test_tool_factory(tool_class, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test tool factory with multiple test cases"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                tool_instance = tool_class(**test_case)
                result = tool_instance.run()
                results.append({
                    'test_case': i,
                    'success': True,
                    'result': result,
                    'error': None
                })
            except Exception as e:
                results.append({
                    'test_case': i,
                    'success': False,
                    'result': None,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        return {
            'total_tests': len(test_cases),
            'passed': success_count,
            'failed': len(test_cases) - success_count,
            'success_rate': success_count / len(test_cases) if test_cases else 0,
            'results': results
        }


# Export all patterns
__all__ = [
    'AsyncTestPatterns',
    'CommunicationTestPatterns', 
    'CommunicationTestConfig',
    'MCPTestPatterns',
    'ResponseValidationPatterns',
    'ThreadRetryPatterns',
    'ToolFactoryPatterns'
]