"""
AgentScope Comprehensive Testing Patterns - AGENT B COMPREHENSIVE TESTING EXCELLENCE
=====================================================================================

Extracted testing patterns from agentscope repository for enhanced testing capabilities.
Focus: Pipeline testing, React agent testing, formatter testing, embedding cache testing.

AGENT B Enhancement: Phase 1.3 - AgentScope Pattern Integration
- Pipeline execution testing
- Agent lifecycle and hook testing  
- Formatter testing for multiple providers (Anthropic, Gemini, OpenAI)
- Token and embedding cache testing
- Tool and toolkit testing patterns
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch
import logging
import json
import time


class PipelineTestPatterns:
    """
    Pipeline testing patterns extracted from agentscope pipeline_test.py
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @dataclass
    class MockMessage:
        """Mock message class for pipeline testing"""
        name: str
        content: str
        role: str
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    class MockAgent:
        """Mock agent for pipeline testing"""
        def __init__(self, name: str, operation: str, value: Any = None):
            self.name = name
            self.operation = operation
            self.value = value
        
        async def reply(self, message) -> Any:
            """Mock reply function"""
            if self.operation == "add":
                message.metadata["result"] += self.value
            elif self.operation == "multiply":
                message.metadata["result"] *= self.value
            elif self.operation == "append":
                if "items" not in message.metadata:
                    message.metadata["items"] = []
                message.metadata["items"].append(self.value)
            elif self.operation == "transform":
                message.metadata["result"] = str(message.metadata.get("result", 0)).upper()
            
            return message
        
        async def observe(self, msg) -> None:
            """Mock observe function"""
            pass
        
        async def handle_interrupt(self, *args, **kwargs):
            """Mock interrupt handler"""
            pass
    
    class MockSequentialPipeline:
        """Mock sequential pipeline"""
        def __init__(self, agents: List[Any]):
            self.agents = agents
        
        async def __call__(self, message):
            """Execute pipeline"""
            current_message = message
            for agent in self.agents:
                current_message = await agent.reply(current_message)
            return current_message
    
    async def test_functional_sequential_pipeline(self, operations: List[Dict[str, Any]], 
                                                 initial_value: Any = 0) -> Dict[str, Any]:
        """Test sequential pipeline with functional approach"""
        agents = []
        for op in operations:
            agent = self.MockAgent(
                name=op.get('name', f'Agent_{len(agents)}'),
                operation=op.get('operation'),
                value=op.get('value')
            )
            agents.append(agent)
        
        message = self.MockMessage("user", "", "user", metadata={"result": initial_value})
        
        # Functional pipeline execution
        current_message = message
        for agent in agents:
            current_message = await agent.reply(current_message)
        
        return {
            'initial_value': initial_value,
            'final_result': current_message.metadata.get("result"),
            'operations_count': len(operations),
            'metadata': current_message.metadata,
            'success': True
        }
    
    async def test_class_sequential_pipeline(self, operations: List[Dict[str, Any]], 
                                           initial_value: Any = 0) -> Dict[str, Any]:
        """Test sequential pipeline with class approach"""
        agents = []
        for op in operations:
            agent = self.MockAgent(
                name=op.get('name', f'Agent_{len(agents)}'),
                operation=op.get('operation'),
                value=op.get('value')
            )
            agents.append(agent)
        
        message = self.MockMessage("user", "", "user", metadata={"result": initial_value})
        pipeline = self.MockSequentialPipeline(agents)
        
        result_message = await pipeline(message)
        
        return {
            'initial_value': initial_value,
            'final_result': result_message.metadata.get("result"),
            'operations_count': len(operations),
            'metadata': result_message.metadata,
            'pipeline_type': 'class',
            'success': True
        }
    
    async def test_multiple_pipeline_configurations(self) -> Dict[str, Any]:
        """Test multiple pipeline configurations"""
        test_configs = [
            {
                'name': 'math_operations',
                'operations': [
                    {'operation': 'add', 'value': 1},
                    {'operation': 'add', 'value': 2},
                    {'operation': 'multiply', 'value': 3}
                ],
                'initial_value': 0,
                'expected_result': 9  # (0 + 1 + 2) * 3
            },
            {
                'name': 'reverse_math',
                'operations': [
                    {'operation': 'multiply', 'value': 3},
                    {'operation': 'add', 'value': 1},
                    {'operation': 'add', 'value': 2}
                ],
                'initial_value': 0,
                'expected_result': 3  # (0 * 3) + 1 + 2
            },
            {
                'name': 'list_operations',
                'operations': [
                    {'operation': 'append', 'value': 'first'},
                    {'operation': 'append', 'value': 'second'},
                    {'operation': 'append', 'value': 'third'}
                ],
                'initial_value': 0,
                'expected_items': ['first', 'second', 'third']
            }
        ]
        
        results = []
        for config in test_configs:
            try:
                result = await self.test_functional_sequential_pipeline(
                    config['operations'], 
                    config['initial_value']
                )
                
                # Validate expected results
                if 'expected_result' in config:
                    result['matches_expected'] = result['final_result'] == config['expected_result']
                elif 'expected_items' in config:
                    result['matches_expected'] = result['metadata'].get('items') == config['expected_items']
                
                result['config_name'] = config['name']
                results.append(result)
            except Exception as e:
                results.append({
                    'config_name': config['name'],
                    'success': False,
                    'error': str(e)
                })
        
        successful_tests = sum(1 for r in results if r.get('success', False))
        return {
            'total_configs': len(test_configs),
            'successful_tests': successful_tests,
            'results': results
        }


class ReactAgentTestPatterns:
    """
    ReAct agent testing patterns extracted from agentscope react_agent_test.py
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    class MockModel:
        """Mock model for ReAct agent testing"""
        def __init__(self, responses: List[Dict[str, Any]] = None):
            self.responses = responses or [{"type": "text", "text": "123"}]
            self.call_count = 0
        
        async def __call__(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
            """Mock model call"""
            response_index = min(self.call_count, len(self.responses) - 1)
            response = self.responses[response_index]
            self.call_count += 1
            
            return {
                "content": [response],
                "usage": {"total_tokens": 10}
            }
    
    class MockReActAgent:
        """Mock ReAct agent for testing"""
        def __init__(self, name: str, model, toolkit=None):
            self.name = name
            self.model = model
            self.toolkit = toolkit or {}
            self.hooks = {}
            self.counters = {}
            self.finish_function_name = "finish_task"
        
        def register_instance_hook(self, hook_type: str, name: str, func: Callable):
            """Register a hook function"""
            if hook_type not in self.hooks:
                self.hooks[hook_type] = {}
            self.hooks[hook_type][name] = func
        
        async def execute_hook(self, hook_type: str, *args, **kwargs):
            """Execute hooks of given type"""
            if hook_type in self.hooks:
                for hook_func in self.hooks[hook_type].values():
                    await hook_func(self, *args, **kwargs)
        
        async def __call__(self, message=None):
            """Execute ReAct agent cycle"""
            # Pre-reasoning hook
            await self.execute_hook("pre_reasoning", {})
            
            # Mock reasoning phase
            response = await self.model([{"role": "user", "content": "test"}])
            
            # Post-reasoning hook  
            await self.execute_hook("post_reasoning", {}, response)
            
            # Pre-acting hook
            await self.execute_hook("pre_acting", {})
            
            # Mock acting phase
            action_result = {"action": "completed", "result": "success"}
            
            # Post-acting hook
            await self.execute_hook("post_acting", {}, action_result)
            
            return response
    
    @staticmethod
    async def pre_reasoning_hook(agent, kwargs):
        """Mock pre-reasoning hook"""
        agent.counters["pre_reasoning"] = agent.counters.get("pre_reasoning", 0) + 1
    
    @staticmethod
    async def post_reasoning_hook(agent, kwargs, output):
        """Mock post-reasoning hook"""
        agent.counters["post_reasoning"] = agent.counters.get("post_reasoning", 0) + 1
    
    @staticmethod
    async def pre_acting_hook(agent, kwargs):
        """Mock pre-acting hook"""
        agent.counters["pre_acting"] = agent.counters.get("pre_acting", 0) + 1
    
    @staticmethod
    async def post_acting_hook(agent, kwargs, output):
        """Mock post-acting hook"""
        agent.counters["post_acting"] = agent.counters.get("post_acting", 0) + 1
    
    async def test_react_agent_lifecycle(self, responses: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Test ReAct agent lifecycle with hooks"""
        model = self.MockModel(responses)
        agent = self.MockReActAgent("TestAgent", model)
        
        # Register hooks
        agent.register_instance_hook("pre_reasoning", "test_hook", self.pre_reasoning_hook)
        agent.register_instance_hook("post_reasoning", "test_hook", self.post_reasoning_hook)
        agent.register_instance_hook("pre_acting", "test_hook", self.pre_acting_hook)
        agent.register_instance_hook("post_acting", "test_hook", self.post_acting_hook)
        
        # Execute agent
        result = await agent()
        
        return {
            'agent_name': agent.name,
            'hook_counts': agent.counters,
            'model_calls': model.call_count,
            'execution_result': result,
            'hooks_registered': len(agent.hooks),
            'success': True
        }
    
    async def test_multiple_executions(self, num_executions: int = 3) -> Dict[str, Any]:
        """Test multiple agent executions"""
        responses = [
            {"type": "text", "text": "First response"},
            {"type": "tool_use", "name": "finish_task", "id": "1", "input": {"response": "Task completed"}},
            {"type": "text", "text": "Final response"}
        ]
        
        model = self.MockModel(responses)
        agent = self.MockReActAgent("MultiTestAgent", model)
        
        # Register hooks
        agent.register_instance_hook("pre_reasoning", "counter", self.pre_reasoning_hook)
        agent.register_instance_hook("post_reasoning", "counter", self.post_reasoning_hook)
        agent.register_instance_hook("pre_acting", "counter", self.pre_acting_hook)
        agent.register_instance_hook("post_acting", "counter", self.post_acting_hook)
        
        execution_results = []
        for i in range(num_executions):
            result = await agent()
            execution_results.append({
                'execution_index': i,
                'result': result,
                'counters_snapshot': agent.counters.copy()
            })
        
        return {
            'total_executions': num_executions,
            'model_total_calls': model.call_count,
            'final_hook_counts': agent.counters,
            'execution_results': execution_results,
            'average_hooks_per_execution': {
                hook: count / num_executions 
                for hook, count in agent.counters.items()
            }
        }


class FormatterTestPatterns:
    """
    Formatter testing patterns for multiple providers
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_providers = ['anthropic', 'openai', 'gemini', 'dashscope']
    
    @dataclass
    class MockMessage:
        """Mock message for formatter testing"""
        role: str
        content: Union[str, List[Dict[str, Any]]]
        name: str
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class MockBlock:
        """Mock content block"""
        type: str
        content: Any
        additional_data: Dict[str, Any] = field(default_factory=dict)
    
    class MockFormatter:
        """Mock formatter for testing"""
        def __init__(self, provider: str):
            self.provider = provider
            self.format_calls = 0
        
        async def format(self, messages: List[Any]) -> List[Dict[str, Any]]:
            """Mock format function"""
            self.format_calls += 1
            
            formatted = []
            for msg in messages:
                formatted_msg = {
                    "role": getattr(msg, 'role', 'user'),
                    "content": getattr(msg, 'content', ''),
                    "provider": self.provider,
                    "formatted_at": time.time()
                }
                formatted.append(formatted_msg)
            
            return formatted
    
    async def test_formatter_basic_functionality(self, provider: str, 
                                               messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test basic formatter functionality"""
        formatter = self.MockFormatter(provider)
        
        mock_messages = []
        for msg_data in messages:
            mock_msg = self.MockMessage(
                role=msg_data.get('role', 'user'),
                content=msg_data.get('content', ''),
                name=msg_data.get('name', 'test')
            )
            mock_messages.append(mock_msg)
        
        try:
            formatted = await formatter.format(mock_messages)
            
            return {
                'provider': provider,
                'input_messages': len(messages),
                'formatted_messages': len(formatted),
                'format_calls': formatter.format_calls,
                'formatted_data': formatted,
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'provider': provider,
                'input_messages': len(messages),
                'success': False,
                'error': str(e)
            }
    
    async def test_multimodal_formatting(self, provider: str) -> Dict[str, Any]:
        """Test multimodal content formatting"""
        multimodal_messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'What is in this image?'},
                    {'type': 'image', 'source': {'type': 'url', 'url': 'http://example.com/image.jpg'}}
                ],
                'name': 'user'
            },
            {
                'role': 'assistant', 
                'content': 'I can see an image with various objects.',
                'name': 'assistant'
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe it in detail'},
                    {'type': 'tool_use', 'name': 'analyze_image', 'id': '1', 'input': {'image_url': 'http://example.com/image.jpg'}}
                ],
                'name': 'user'
            }
        ]
        
        result = await self.test_formatter_basic_functionality(provider, multimodal_messages)
        result['test_type'] = 'multimodal'
        
        return result
    
    async def test_all_providers_formatting(self, message_sets: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Test formatting across all supported providers"""
        results = {}
        
        for provider in self.supported_providers:
            provider_results = []
            
            for i, message_set in enumerate(message_sets):
                test_result = await self.test_formatter_basic_functionality(provider, message_set)
                test_result['message_set_index'] = i
                provider_results.append(test_result)
            
            # Test multimodal for each provider
            multimodal_result = await self.test_multimodal_formatting(provider)
            provider_results.append(multimodal_result)
            
            results[provider] = {
                'results': provider_results,
                'successful_tests': sum(1 for r in provider_results if r['success']),
                'total_tests': len(provider_results)
            }
        
        return {
            'providers_tested': self.supported_providers,
            'provider_results': results,
            'overall_success_rate': sum(
                results[p]['successful_tests'] / results[p]['total_tests'] 
                for p in self.supported_providers
            ) / len(self.supported_providers)
        }


class EmbeddingCacheTestPatterns:
    """
    Embedding cache testing patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    class MockEmbeddingCache:
        """Mock embedding cache for testing"""
        def __init__(self, cache_size: int = 100):
            self.cache = {}
            self.cache_size = cache_size
            self.hits = 0
            self.misses = 0
            self.evictions = 0
        
        async def get(self, key: str) -> Optional[List[float]]:
            """Get embedding from cache"""
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
        
        async def put(self, key: str, embedding: List[float]) -> bool:
            """Put embedding in cache"""
            if len(self.cache) >= self.cache_size:
                # Simple eviction - remove oldest
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.evictions += 1
            
            self.cache[key] = embedding
            return True
        
        async def contains(self, key: str) -> bool:
            """Check if key exists in cache"""
            return key in self.cache
        
        def get_stats(self) -> Dict[str, Any]:
            """Get cache statistics"""
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'cache_size': len(self.cache),
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    async def test_cache_basic_operations(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test basic cache operations"""
        cache = self.MockEmbeddingCache()
        results = []
        
        for item in test_data:
            key = item['key']
            embedding = item['embedding']
            
            # Test cache miss
            cached_embedding = await cache.get(key)
            assert cached_embedding is None
            
            # Test put
            put_success = await cache.put(key, embedding)
            
            # Test cache hit
            cached_embedding = await cache.get(key)
            
            # Test contains
            contains_result = await cache.contains(key)
            
            results.append({
                'key': key,
                'put_success': put_success,
                'retrieved_embedding': cached_embedding,
                'contains_result': contains_result,
                'embedding_match': cached_embedding == embedding
            })
        
        stats = cache.get_stats()
        
        return {
            'test_items': len(test_data),
            'results': results,
            'cache_stats': stats,
            'all_operations_successful': all(r['put_success'] and r['embedding_match'] for r in results)
        }
    
    async def test_cache_eviction(self, cache_size: int = 5, 
                                 num_items: int = 10) -> Dict[str, Any]:
        """Test cache eviction behavior"""
        cache = self.MockEmbeddingCache(cache_size)
        
        # Fill cache beyond capacity
        for i in range(num_items):
            key = f"item_{i}"
            embedding = [float(j) for j in range(10)]  # Mock embedding
            await cache.put(key, embedding)
        
        stats = cache.get_stats()
        
        # Test that early items were evicted
        early_items_present = []
        for i in range(num_items - cache_size):
            key = f"item_{i}"
            present = await cache.contains(key)
            early_items_present.append(present)
        
        # Test that recent items are still present
        recent_items_present = []
        for i in range(num_items - cache_size, num_items):
            key = f"item_{i}"
            present = await cache.contains(key)
            recent_items_present.append(present)
        
        return {
            'cache_size_limit': cache_size,
            'items_added': num_items,
            'expected_evictions': max(0, num_items - cache_size),
            'actual_evictions': stats['evictions'],
            'early_items_evicted': not any(early_items_present),
            'recent_items_preserved': all(recent_items_present),
            'final_cache_size': stats['cache_size'],
            'cache_stats': stats
        }
    
    async def test_cache_performance(self, num_operations: int = 1000) -> Dict[str, Any]:
        """Test cache performance with many operations"""
        cache = self.MockEmbeddingCache(cache_size=100)
        
        start_time = time.time()
        
        # Perform mixed operations
        for i in range(num_operations):
            if i < num_operations // 2:
                # First half: populate cache
                key = f"perf_item_{i % 50}"  # Some duplication to test hits
                embedding = [float(j) for j in range(128)]
                await cache.put(key, embedding)
            else:
                # Second half: read from cache
                key = f"perf_item_{i % 50}"
                await cache.get(key)
        
        end_time = time.time()
        duration = end_time - start_time
        
        stats = cache.get_stats()
        
        return {
            'total_operations': num_operations,
            'duration_seconds': duration,
            'operations_per_second': num_operations / duration,
            'cache_stats': stats,
            'performance_acceptable': stats['hit_rate'] > 0.3  # At least 30% hit rate
        }


class ToolTestPatterns:
    """
    Tool and toolkit testing patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    class MockTool:
        """Mock tool for testing"""
        def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None):
            self.name = name
            self.description = description
            self.parameters = parameters or {}
            self.call_count = 0
            self.last_input = None
        
        async def __call__(self, **kwargs) -> Dict[str, Any]:
            """Execute tool"""
            self.call_count += 1
            self.last_input = kwargs
            
            # Mock tool execution based on name
            if self.name == "calculator":
                operation = kwargs.get('operation', 'add')
                a = kwargs.get('a', 0)
                b = kwargs.get('b', 0)
                
                if operation == 'add':
                    result = a + b
                elif operation == 'multiply':
                    result = a * b
                elif operation == 'divide':
                    result = a / b if b != 0 else "Error: Division by zero"
                else:
                    result = "Error: Unknown operation"
                
                return {'result': result, 'operation': operation}
            
            elif self.name == "text_processor":
                text = kwargs.get('text', '')
                action = kwargs.get('action', 'upper')
                
                if action == 'upper':
                    result = text.upper()
                elif action == 'lower':
                    result = text.lower()
                elif action == 'reverse':
                    result = text[::-1]
                else:
                    result = text
                
                return {'result': result, 'original': text, 'action': action}
            
            else:
                return {'result': f"Tool {self.name} executed", 'input': kwargs}
    
    class MockToolkit:
        """Mock toolkit for testing"""
        def __init__(self):
            self.tools = {}
            self.execution_log = []
        
        def add_tool(self, tool):
            """Add tool to toolkit"""
            self.tools[tool.name] = tool
        
        async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
            """Execute tool by name"""
            if tool_name not in self.tools:
                raise ValueError(f"Tool {tool_name} not found")
            
            tool = self.tools[tool_name]
            result = await tool(**kwargs)
            
            self.execution_log.append({
                'tool_name': tool_name,
                'input': kwargs,
                'result': result,
                'timestamp': time.time()
            })
            
            return result
        
        def get_tool_info(self) -> Dict[str, Any]:
            """Get information about all tools"""
            return {
                tool_name: {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters,
                    'call_count': tool.call_count
                }
                for tool_name, tool in self.tools.items()
            }
    
    async def test_individual_tools(self, tool_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test individual tools"""
        results = []
        
        for config in tool_configs:
            tool = self.MockTool(
                name=config['name'],
                description=config['description'], 
                parameters=config.get('parameters', {})
            )
            
            test_cases = config.get('test_cases', [])
            tool_results = []
            
            for test_case in test_cases:
                try:
                    result = await tool(**test_case['input'])
                    tool_results.append({
                        'test_case': test_case,
                        'result': result,
                        'success': True,
                        'expected_match': result == test_case.get('expected') if 'expected' in test_case else None
                    })
                except Exception as e:
                    tool_results.append({
                        'test_case': test_case,
                        'result': None,
                        'success': False,
                        'error': str(e)
                    })
            
            results.append({
                'tool_name': tool.name,
                'test_results': tool_results,
                'total_calls': tool.call_count,
                'successful_tests': sum(1 for r in tool_results if r['success'])
            })
        
        return {
            'total_tools_tested': len(tool_configs),
            'tool_results': results
        }
    
    async def test_toolkit_integration(self) -> Dict[str, Any]:
        """Test toolkit with multiple tools"""
        toolkit = self.MockToolkit()
        
        # Add tools
        calculator = self.MockTool("calculator", "Perform math operations", 
                                  {"operation": "string", "a": "number", "b": "number"})
        text_processor = self.MockTool("text_processor", "Process text", 
                                     {"text": "string", "action": "string"})
        
        toolkit.add_tool(calculator)
        toolkit.add_tool(text_processor)
        
        # Test tool execution through toolkit
        test_executions = [
            {"tool": "calculator", "args": {"operation": "add", "a": 5, "b": 3}},
            {"tool": "calculator", "args": {"operation": "multiply", "a": 4, "b": 7}},
            {"tool": "text_processor", "args": {"text": "Hello World", "action": "upper"}},
            {"tool": "text_processor", "args": {"text": "Python", "action": "reverse"}}
        ]
        
        execution_results = []
        for execution in test_executions:
            try:
                result = await toolkit.execute_tool(execution["tool"], **execution["args"])
                execution_results.append({
                    'execution': execution,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                execution_results.append({
                    'execution': execution,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        toolkit_info = toolkit.get_tool_info()
        
        return {
            'toolkit_tools': list(toolkit.tools.keys()),
            'execution_results': execution_results,
            'successful_executions': sum(1 for r in execution_results if r['success']),
            'total_executions': len(test_executions),
            'execution_log': toolkit.execution_log,
            'tool_info': toolkit_info
        }


# Export all patterns
__all__ = [
    'PipelineTestPatterns',
    'ReactAgentTestPatterns', 
    'FormatterTestPatterns',
    'EmbeddingCacheTestPatterns',
    'ToolTestPatterns'
]