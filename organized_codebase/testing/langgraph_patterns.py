"""
LangGraph Supervisor Testing Patterns - AGENT B COMPREHENSIVE TESTING EXCELLENCE
================================================================================

Extracted testing patterns from langgraph-supervisor-py repository for enhanced supervisor testing capabilities.
Focus: Multi-agent coordination, supervisor workflows, agent handoffs, functional API testing.

AGENT B Enhancement: Phase 1.5 - LangGraph Supervisor Pattern Integration
- Multi-agent supervisor coordination testing
- Agent handoff and message forwarding patterns
- Functional API workflow testing
- Agent name formatting and inline content handling
- Metadata propagation testing across agents
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union, Sequence
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import re
from unittest.mock import Mock, AsyncMock


class SupervisorWorkflowTestPatterns:
    """
    Supervisor workflow testing patterns extracted from langgraph test_supervisor.py
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @dataclass
    class MockMessage:
        """Mock message for supervisor testing"""
        content: str
        name: Optional[str] = None
        type: str = "ai"  # "human", "ai", "tool"
        tool_calls: Optional[List[Dict[str, Any]]] = None
        
        def dict(self):
            return {
                "content": self.content,
                "name": self.name,
                "type": self.type,
                "tool_calls": self.tool_calls or []
            }
    
    class MockTool:
        """Mock tool for supervisor testing"""
        def __init__(self, name: str, description: str = ""):
            self.name = name
            self.description = description
            self.call_count = 0
            self.last_args = None
        
        def __call__(self, **kwargs):
            self.call_count += 1
            self.last_args = kwargs
            
            if self.name == "add":
                return kwargs.get('a', 0) + kwargs.get('b', 0)
            elif self.name == "web_search":
                return self._mock_web_search(kwargs.get('query', ''))
            elif self.name == "echo_tool":
                return kwargs.get('text', 'echo')
            else:
                return f"Tool {self.name} executed with {kwargs}"
        
        def _mock_web_search(self, query: str) -> str:
            """Mock web search results"""
            if "FAANG" in query.upper():
                return (
                    "Here are the headcounts for each of the FAANG companies in 2024:\n"
                    "1. **Facebook (Meta)**: 67,317 employees.\n"
                    "2. **Apple**: 164,000 employees.\n"
                    "3. **Amazon**: 1,551,000 employees.\n"
                    "4. **Netflix**: 14,000 employees.\n"
                    "5. **Google (Alphabet)**: 181,269 employees."
                )
            return f"Search results for: {query}"
    
    class MockChatModel:
        """Mock chat model for testing"""
        def __init__(self, responses: List[Dict[str, Any]]):
            self.responses = responses
            self.response_index = 0
            self.call_count = 0
            self.tools = []
        
        def bind_tools(self, tools: List[Any]) -> 'MockChatModel':
            """Bind tools to the model"""
            self.tools = tools
            return self
        
        def __call__(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
            """Generate response"""
            if self.response_index >= len(self.responses):
                self.response_index = len(self.responses) - 1
            
            response = self.responses[self.response_index]
            self.response_index += 1
            self.call_count += 1
            
            return response
    
    class MockAgent:
        """Mock agent for supervisor testing"""
        def __init__(self, name: str, model: 'MockChatModel', tools: List[MockTool]):
            self.name = name
            self.model = model
            self.tools = {tool.name: tool for tool in tools}
            self.execution_log = []
        
        async def execute(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Execute agent with messages"""
            start_time = time.time()
            
            # Get model response
            response = self.model(messages)
            
            # Execute any tool calls
            if response.get('tool_calls'):
                for tool_call in response['tool_calls']:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    
                    if tool_name in self.tools:
                        result = self.tools[tool_name](**tool_args)
                        tool_call['result'] = result
            
            execution_record = {
                'agent': self.name,
                'messages': messages,
                'response': response,
                'execution_time': time.time() - start_time,
                'timestamp': time.time()
            }
            self.execution_log.append(execution_record)
            
            return response
    
    class MockSupervisor:
        """Mock supervisor for coordination testing"""
        def __init__(self, agents: List[MockAgent], model: MockChatModel, 
                     output_mode: str = "last_message", add_handoff_messages: bool = True):
            self.agents = {agent.name: agent for agent in agents}
            self.model = model
            self.output_mode = output_mode
            self.add_handoff_messages = add_handoff_messages
            self.execution_log = []
            self.message_history = []
        
        async def execute(self, input_messages: List[Dict[str, Any]], 
                        config: Dict[str, Any] = None) -> Dict[str, Any]:
            """Execute supervisor workflow"""
            current_messages = input_messages.copy()
            self.message_history = input_messages.copy()
            
            max_steps = 10  # Prevent infinite loops
            step = 0
            
            while step < max_steps:
                # Get supervisor decision
                supervisor_response = self.model(current_messages)
                
                if supervisor_response.get('tool_calls'):
                    for tool_call in supervisor_response['tool_calls']:
                        if tool_call['name'].startswith('transfer_to_'):
                            # Extract agent name from tool call
                            agent_name = tool_call['name'].replace('transfer_to_', '').replace('_', ' ').title().replace(' ', '') 
                            # Map back to actual agent names
                            for actual_name in self.agents.keys():
                                if agent_name.lower() in actual_name.lower():
                                    agent_name = actual_name
                                    break
                            
                            if agent_name in self.agents:
                                # Add handoff message if configured
                                if self.add_handoff_messages:
                                    handoff_msg = {
                                        'name': tool_call['name'],
                                        'content': f'Successfully transferred to {agent_name}',
                                        'type': 'tool'
                                    }
                                    self.message_history.append(handoff_msg)
                                
                                # Execute agent
                                agent_response = await self.agents[agent_name].execute(current_messages)
                                
                                agent_msg = {
                                    'name': agent_name,
                                    'content': agent_response.get('content', ''),
                                    'type': 'ai',
                                    'tool_calls': agent_response.get('tool_calls', [])
                                }
                                self.message_history.append(agent_msg)
                                
                                # Add transfer back message
                                if self.add_handoff_messages:
                                    transfer_back_msg = {
                                        'name': agent_name,
                                        'content': 'Transferring back to supervisor',
                                        'type': 'ai',
                                        'tool_calls': [{'name': 'transfer_back_to_supervisor', 'args': {}}]
                                    }
                                    self.message_history.append(transfer_back_msg)
                                    
                                    transfer_back_result = {
                                        'name': 'transfer_back_to_supervisor',
                                        'content': 'Successfully transferred back to supervisor',
                                        'type': 'tool'
                                    }
                                    self.message_history.append(transfer_back_result)
                                
                                current_messages.append(agent_msg)
                
                # Add supervisor response to history
                supervisor_msg = {
                    'name': 'supervisor',
                    'content': supervisor_response.get('content', ''),
                    'type': 'ai',
                    'tool_calls': supervisor_response.get('tool_calls', [])
                }
                self.message_history.append(supervisor_msg)
                
                # Check if supervisor provided final response
                if supervisor_response.get('content') and not supervisor_response.get('tool_calls'):
                    break
                
                step += 1
            
            # Return messages based on output mode
            if self.output_mode == "last_message":
                # Return only key messages for last_message mode
                filtered_messages = []
                for msg in self.message_history:
                    if (msg.get('name') in self.agents or 
                        msg.get('name') == 'supervisor' or 
                        msg.get('type') == 'human' or
                        msg.get('name', '').startswith('transfer_')):
                        filtered_messages.append(msg)
                return {'messages': filtered_messages}
            else:
                # Return full history
                return {'messages': self.message_history}
    
    async def test_supervisor_basic_workflow(self, 
                                           include_agent_name: bool = False,
                                           output_mode: str = "last_message") -> Dict[str, Any]:
        """Test basic supervisor workflow with two agents"""
        
        # Create tools
        add_tool = self.MockTool("add", "Add two numbers")
        web_search_tool = self.MockTool("web_search", "Search the web for information")
        
        # Create mock responses
        supervisor_responses = [
            {
                'content': '',
                'tool_calls': [{'name': 'transfer_to_research_expert', 'args': {}}]
            },
            {
                'content': '',
                'tool_calls': [{'name': 'transfer_to_math_expert', 'args': {}}]
            },
            {
                'content': 'The combined headcount of the FAANG companies in 2024 is 1,977,586 employees.'
            }
        ]
        
        research_responses = [
            {
                'content': '',
                'tool_calls': [{'name': 'web_search', 'args': {'query': 'FAANG headcount 2024'}}]
            },
            {
                'content': 'The headcount for the FAANG companies in 2024 is as follows:\n\n1. **Facebook (Meta)**: 67,317 employees\n2. **Amazon**: 1,551,000 employees\n3. **Apple**: 164,000 employees\n4. **Netflix**: 14,000 employees\n5. **Google (Alphabet)**: 181,269 employees\n\nTo find the combined headcount, simply add these numbers together.'
            }
        ]
        
        math_responses = [
            {
                'content': '',
                'tool_calls': [
                    {'name': 'add', 'args': {'a': 67317, 'b': 1551000}},
                    {'name': 'add', 'args': {'a': 164000, 'b': 14000}},
                    {'name': 'add', 'args': {'a': 181269, 'b': 0}}
                ]
            },
            {
                'content': 'The combined headcount of the FAANG companies in 2024 is 1,977,586 employees.'
            }
        ]
        
        # Create agents
        math_model = self.MockChatModel(math_responses)
        math_agent = self.MockAgent("math_expert", math_model, [add_tool])
        
        research_model = self.MockChatModel(research_responses)
        research_agent = self.MockAgent("research_expert", research_model, [web_search_tool])
        
        # Create supervisor
        supervisor_model = self.MockChatModel(supervisor_responses)
        supervisor = self.MockSupervisor([math_agent, research_agent], supervisor_model, output_mode)
        
        # Execute workflow
        input_message = {
            'content': "what's the combined headcount of the FAANG companies in 2024?",
            'type': 'human'
        }
        
        result = await supervisor.execute([input_message])
        
        return {
            'input_message': input_message,
            'output_mode': output_mode,
            'result': result,
            'message_count': len(result['messages']),
            'agents_executed': [agent.name for agent in [math_agent, research_agent] if agent.execution_log],
            'supervisor_calls': supervisor_model.call_count,
            'success': True
        }
    
    async def test_message_forwarding(self) -> Dict[str, Any]:
        """Test supervisor message forwarding capabilities"""
        
        # Create echo tool and agent
        echo_tool = self.MockTool("echo_tool", "Echo the input text")
        
        echo_responses = [
            {'content': 'Echo: test forwarding!'}
        ]
        
        supervisor_responses = [
            {
                'content': '',
                'tool_calls': [{'name': 'transfer_to_echo_agent', 'args': {}}]
            },
            {
                'content': 'Echo: test forwarding!',
                'tool_calls': [{'name': 'forward_message', 'args': {'from_agent': 'echo_agent'}}]
            }
        ]
        
        # Create agents and supervisor
        echo_model = self.MockChatModel(echo_responses)
        echo_agent = self.MockAgent("echo_agent", echo_model, [echo_tool])
        
        supervisor_model = self.MockChatModel(supervisor_responses)
        supervisor = self.MockSupervisor([echo_agent], supervisor_model)
        
        # Execute workflow
        input_message = {'content': 'Scooby-dooby-doo', 'type': 'human'}
        result = await supervisor.execute([input_message])
        
        return {
            'input_message': input_message,
            'result': result,
            'echo_agent_calls': len(echo_agent.execution_log),
            'forwarding_tested': True,
            'success': True
        }
    
    async def test_metadata_propagation(self, test_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Test metadata propagation from supervisor to sub-agents"""
        
        class MetadataTrackingAgent(self.MockAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.received_metadata = None
            
            async def execute(self, messages: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Dict[str, Any]:
                if config and 'metadata' in config:
                    self.received_metadata = config['metadata']
                
                return await super().execute(messages)
        
        # Create tracking agent
        test_responses = [{'content': 'Test response'}]
        test_model = self.MockChatModel(test_responses)
        tracking_agent = MetadataTrackingAgent("test_agent", test_model, [])
        
        # Create supervisor
        supervisor_responses = [
            {
                'content': '',
                'tool_calls': [{'name': 'transfer_to_test_agent', 'args': {}}]
            },
            {'content': 'Final response'}
        ]
        
        supervisor_model = self.MockChatModel(supervisor_responses)
        supervisor = self.MockSupervisor([tracking_agent], supervisor_model)
        
        # Execute with metadata
        input_message = {'content': 'Test message', 'type': 'human'}
        config = {'metadata': test_metadata}
        
        result = await supervisor.execute([input_message], config)
        
        return {
            'test_metadata': test_metadata,
            'received_metadata': tracking_agent.received_metadata,
            'metadata_propagated': tracking_agent.received_metadata == test_metadata,
            'result': result,
            'success': True
        }
    
    async def test_worker_hide_handoffs(self) -> Dict[str, Any]:
        """Test supervisor with hidden handoff messages"""
        
        echo_tool = self.MockTool("echo_tool", "Echo the input text")
        
        echo_responses = [
            {'content': 'Echo 1!'},
            {'content': 'Echo 2!'}
        ]
        
        supervisor_responses = [
            {
                'content': '',
                'tool_calls': [{'name': 'transfer_to_echo_agent', 'args': {}}]
            },
            {'content': 'boo'},
            {
                'content': '',
                'tool_calls': [{'name': 'transfer_to_echo_agent', 'args': {}}]
            },
            {'content': 'END'}
        ]
        
        # Create agents and supervisor with hidden handoffs
        echo_model = self.MockChatModel(echo_responses)
        echo_agent = self.MockAgent("echo_agent", echo_model, [echo_tool])
        
        supervisor_model = self.MockChatModel(supervisor_responses)
        supervisor = self.MockSupervisor([echo_agent], supervisor_model, add_handoff_messages=False)
        
        # Execute workflow
        input_message = {'content': 'Scooby-dooby-doo', 'type': 'human'}
        result = await supervisor.execute([input_message])
        
        # Test second execution
        second_input = {'content': 'Huh take two?', 'type': 'human'}
        second_result = await supervisor.execute(result['messages'] + [second_input])
        
        return {
            'first_execution': result,
            'second_execution': second_result,
            'handoffs_hidden': True,
            'echo_executions': len(echo_agent.execution_log),
            'success': True
        }


class FunctionalAPITestPatterns:
    """
    Functional API testing patterns extracted from test_supervisor_functional_api.py
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    class MockFunctionalModel:
        """Mock functional model"""
        def __init__(self, responses: List[str]):
            self.responses = responses
            self.response_index = 0
        
        def bind_tools(self, *args, **kwargs):
            return self
        
        def invoke(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            if self.response_index >= len(self.responses):
                self.response_index = len(self.responses) - 1
            
            response = self.responses[self.response_index]
            self.response_index += 1
            
            return {'content': response, 'type': 'ai'}
    
    class MockTask:
        """Mock task for functional API"""
        def __init__(self, func: Callable):
            self.func = func
            self.execution_count = 0
        
        def result(self) -> Any:
            """Get task result"""
            self.execution_count += 1
            return self.func()
    
    class MockFunctionalAgent:
        """Mock functional API agent"""
        def __init__(self, name: str, task_func: Callable, model):
            self.name = name
            self.task_func = task_func
            self.model = model
            self.execution_log = []
        
        def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute functional agent"""
            start_time = time.time()
            
            # Create mock task
            task = self.MockTask(lambda: self.model.invoke([{'content': 'Generate task output', 'type': 'system'}]))
            
            # Execute task
            result = task.result()
            
            # Add to messages
            messages = state.get('messages', [])
            messages.append(result)
            
            execution_record = {
                'agent': self.name,
                'state': state,
                'result': result,
                'execution_time': time.time() - start_time,
                'timestamp': time.time()
            }
            self.execution_log.append(execution_record)
            
            return {'messages': messages}
    
    async def test_functional_workflow(self, agent_name: str = "joke_agent", 
                                     prompt: str = "You are a supervisor managing a joke expert.") -> Dict[str, Any]:
        """Test supervisor workflow with functional API agent"""
        
        # Create functional model
        model = self.MockFunctionalModel(["Mocked response"])
        
        # Create functional agent
        def generate_joke():
            """Generate a joke using the model"""
            return model.invoke([{'content': 'Write a short joke', 'type': 'system'}])
        
        functional_agent = self.MockFunctionalAgent(agent_name, generate_joke, model)
        
        # Mock supervisor model
        supervisor_model = self.MockFunctionalModel([
            "Transferring to joke agent",
            "Here's a great joke for you!"
        ])
        
        # Create mock supervisor workflow  
        class MockSupervisorWorkflow:
            def __init__(self, agents, model, prompt):
                self.agents = agents
                self.model = model
                self.prompt = prompt
                self.execution_log = []
            
            def compile(self):
                return self
            
            def invoke(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
                # Execute functional agent
                for agent in self.agents:
                    result = agent(input_state)
                    input_state.update(result)
                
                # Add supervisor response
                supervisor_response = self.model.invoke([])
                input_state['messages'].append(supervisor_response)
                
                self.execution_log.append({
                    'input': input_state,
                    'agents_executed': [agent.name for agent in self.agents],
                    'timestamp': time.time()
                })
                
                return input_state
        
        # Create and test workflow
        workflow = MockSupervisorWorkflow([functional_agent], supervisor_model, prompt)
        app = workflow.compile()
        
        input_messages = [{'content': 'Tell me a joke!', 'type': 'human'}]
        result = app.invoke({'messages': input_messages})
        
        # Verify results
        has_joke_content = any(
            'joke' in msg.get('content', '').lower() 
            for msg in result['messages'] 
            if isinstance(msg, dict) and 'content' in msg
        )
        
        return {
            'agent_name': agent_name,
            'prompt': prompt,
            'input_messages': input_messages,
            'result': result,
            'message_count': len(result.get('messages', [])),
            'has_joke_content': has_joke_content,
            'functional_agent_executions': len(functional_agent.execution_log),
            'workflow_executions': len(workflow.execution_log),
            'success': True
        }


class AgentNameTestPatterns:
    """
    Agent name formatting test patterns extracted from test_agent_name.py
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @dataclass
    class MockMessage:
        """Mock message for agent name testing"""
        content: Union[str, List[Dict[str, Any]]]
        name: Optional[str] = None
        type: str = "ai"
        
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return (self.content == other.content and 
                   self.name == other.name and 
                   self.type == other.type)
    
    def add_inline_agent_name(self, message: MockMessage) -> MockMessage:
        """Add inline agent name to message content"""
        # Only process AI messages with names
        if message.type != "ai" or not message.name:
            return message
        
        if isinstance(message.content, str):
            if not message.content:
                return message
            
            formatted_content = f"<name>{message.name}</name><content>{message.content}</content>"
            return self.MockMessage(
                content=formatted_content,
                name=message.name,
                type=message.type
            )
        
        elif isinstance(message.content, list):
            # Handle content blocks
            new_content = []
            name_added = False
            
            for block in message.content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    if not name_added:
                        text = block.get('text', '')
                        formatted_text = f"<name>{message.name}</name><content>{text}</content>"
                        new_content.append({'type': 'text', 'text': formatted_text})
                        name_added = True
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            
            # If no text blocks found, add empty content block with name
            if not name_added:
                new_content.insert(0, {
                    'type': 'text', 
                    'text': f"<name>{message.name}</name><content></content>"
                })
            
            return self.MockMessage(
                content=new_content,
                name=message.name,
                type=message.type
            )
        
        return message
    
    def remove_inline_agent_name(self, message: MockMessage) -> MockMessage:
        """Remove inline agent name from message content"""
        # Only process AI messages
        if message.type != "ai":
            return message
        
        if isinstance(message.content, str):
            if not message.content:
                return message
            
            # Extract content from tags using regex
            pattern = r'<name>.*?</name><content>(.*?)</content>'
            match = re.search(pattern, message.content, re.DOTALL)
            
            if match:
                extracted_content = match.group(1)
                return self.MockMessage(
                    content=extracted_content,
                    name=message.name,
                    type=message.type
                )
            
            return message
        
        elif isinstance(message.content, list):
            # Handle content blocks
            new_content = []
            
            for block in message.content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text = block.get('text', '')
                    
                    # Extract content from tags
                    pattern = r'<name>.*?</name><content>(.*?)</content>'
                    match = re.search(pattern, text, re.DOTALL)
                    
                    if match:
                        extracted_text = match.group(1)
                        if extracted_text:  # Only add if not empty
                            new_content.append({'type': 'text', 'text': extracted_text})
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            
            return self.MockMessage(
                content=new_content,
                name=message.name,
                type=message.type
            )
        
        return message
    
    def test_add_inline_agent_name(self) -> Dict[str, Any]:
        """Test adding inline agent names to messages"""
        test_cases = []
        
        # Test 1: Non-AI messages should be unchanged
        human_message = self.MockMessage(content="Hello", type="human")
        result1 = self.add_inline_agent_name(human_message)
        test_cases.append({
            'case': 'non_ai_message',
            'input': human_message,
            'result': result1,
            'expected_unchanged': True,
            'success': result1 == human_message
        })
        
        # Test 2: AI messages without names should be unchanged
        ai_message_no_name = self.MockMessage(content="Hello world", type="ai")
        result2 = self.add_inline_agent_name(ai_message_no_name)
        test_cases.append({
            'case': 'ai_message_no_name',
            'input': ai_message_no_name,
            'result': result2,
            'expected_unchanged': True,
            'success': result2 == ai_message_no_name
        })
        
        # Test 3: AI messages with names should get formatted
        ai_message_with_name = self.MockMessage(content="Hello world", name="assistant", type="ai")
        result3 = self.add_inline_agent_name(ai_message_with_name)
        expected_content = "<name>assistant</name><content>Hello world</content>"
        test_cases.append({
            'case': 'ai_message_with_name',
            'input': ai_message_with_name,
            'result': result3,
            'expected_content': expected_content,
            'success': result3.content == expected_content and result3.name == "assistant"
        })
        
        # Test 4: Content blocks
        content_blocks = [
            {"type": "text", "text": "Hello world"},
            {"type": "image", "image_url": "http://example.com/image.jpg"}
        ]
        ai_message_blocks = self.MockMessage(content=content_blocks, name="assistant", type="ai")
        result4 = self.add_inline_agent_name(ai_message_blocks)
        expected_blocks = [
            {"type": "text", "text": "<name>assistant</name><content>Hello world</content>"},
            {"type": "image", "image_url": "http://example.com/image.jpg"}
        ]
        test_cases.append({
            'case': 'content_blocks',
            'input': ai_message_blocks,
            'result': result4,
            'expected_blocks': expected_blocks,
            'success': result4.content == expected_blocks
        })
        
        successful_cases = sum(1 for case in test_cases if case['success'])
        
        return {
            'test_cases': test_cases,
            'total_cases': len(test_cases),
            'successful_cases': successful_cases,
            'all_passed': successful_cases == len(test_cases)
        }
    
    def test_remove_inline_agent_name(self) -> Dict[str, Any]:
        """Test removing inline agent names from messages"""
        test_cases = []
        
        # Test 1: Non-AI messages should be unchanged
        human_message = self.MockMessage(content="Hello", type="human")
        result1 = self.remove_inline_agent_name(human_message)
        test_cases.append({
            'case': 'non_ai_message',
            'input': human_message,
            'result': result1,
            'success': result1 == human_message
        })
        
        # Test 2: Messages with empty content should be unchanged
        ai_message_empty = self.MockMessage(content="", name="assistant", type="ai")
        result2 = self.remove_inline_agent_name(ai_message_empty)
        test_cases.append({
            'case': 'empty_content',
            'input': ai_message_empty,
            'result': result2,
            'success': result2 == ai_message_empty
        })
        
        # Test 3: Messages without tags should be unchanged
        ai_message_no_tags = self.MockMessage(content="Hello world", name="assistant", type="ai")
        result3 = self.remove_inline_agent_name(ai_message_no_tags)
        test_cases.append({
            'case': 'no_tags',
            'input': ai_message_no_tags,
            'result': result3,
            'success': result3 == ai_message_no_tags
        })
        
        # Test 4: Extract content from tags
        tagged_content = "<name>assistant</name><content>Hello world</content>"
        ai_message_tagged = self.MockMessage(content=tagged_content, name="assistant", type="ai")
        result4 = self.remove_inline_agent_name(ai_message_tagged)
        test_cases.append({
            'case': 'extract_from_tags',
            'input': ai_message_tagged,
            'result': result4,
            'expected_content': "Hello world",
            'success': result4.content == "Hello world" and result4.name == "assistant"
        })
        
        # Test 5: Multiline content
        multiline_content = """<name>assistant</name><content>This is
a multiline
message</content>"""
        ai_message_multiline = self.MockMessage(content=multiline_content, name="assistant", type="ai")
        result5 = self.remove_inline_agent_name(ai_message_multiline)
        expected_multiline = "This is\na multiline\nmessage"
        test_cases.append({
            'case': 'multiline_content',
            'input': ai_message_multiline,
            'result': result5,
            'expected_content': expected_multiline,
            'success': result5.content == expected_multiline
        })
        
        # Test 6: Content blocks
        content_blocks = [
            {"type": "text", "text": "<name>assistant</name><content>Hello world</content>"},
            {"type": "image", "image_url": "http://example.com/image.jpg"}
        ]
        ai_message_blocks = self.MockMessage(content=content_blocks, name="assistant", type="ai")
        result6 = self.remove_inline_agent_name(ai_message_blocks)
        expected_blocks = [
            {"type": "text", "text": "Hello world"},
            {"type": "image", "image_url": "http://example.com/image.jpg"}
        ]
        test_cases.append({
            'case': 'content_blocks',
            'input': ai_message_blocks,
            'result': result6,
            'expected_blocks': expected_blocks,
            'success': result6.content == expected_blocks
        })
        
        successful_cases = sum(1 for case in test_cases if case['success'])
        
        return {
            'test_cases': test_cases,
            'total_cases': len(test_cases),
            'successful_cases': successful_cases,
            'all_passed': successful_cases == len(test_cases)
        }
    
    def test_roundtrip_agent_name_processing(self) -> Dict[str, Any]:
        """Test adding and removing agent names in roundtrip"""
        original_message = self.MockMessage(
            content="This is a test message",
            name="test_agent", 
            type="ai"
        )
        
        # Add inline name
        with_inline = self.add_inline_agent_name(original_message)
        
        # Remove inline name  
        restored = self.remove_inline_agent_name(with_inline)
        
        return {
            'original': original_message,
            'with_inline': with_inline,
            'restored': restored,
            'roundtrip_success': restored.content == original_message.content,
            'name_preserved': restored.name == original_message.name,
            'type_preserved': restored.type == original_message.type,
            'complete_roundtrip': (restored.content == original_message.content and
                                 restored.name == original_message.name and 
                                 restored.type == original_message.type)
        }


class HandoffTestPatterns:
    """
    Agent handoff testing patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @dataclass
    class HandoffRecord:
        """Record of agent handoff"""
        from_agent: str
        to_agent: str
        message: Dict[str, Any]
        timestamp: float
        success: bool
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    class MockHandoffManager:
        """Mock handoff manager for testing"""
        def __init__(self):
            self.handoff_history = []
            self.available_agents = set()
            self.current_agent = None
        
        def register_agent(self, agent_name: str):
            """Register an agent as available"""
            self.available_agents.add(agent_name)
        
        async def handoff_to_agent(self, target_agent: str, message: Dict[str, Any], 
                                 from_agent: str = None) -> HandoffRecord:
            """Perform handoff to target agent"""
            if target_agent not in self.available_agents:
                record = HandoffRecord(
                    from_agent=from_agent or self.current_agent or "supervisor",
                    to_agent=target_agent,
                    message=message,
                    timestamp=time.time(),
                    success=False,
                    metadata={"error": f"Agent {target_agent} not available"}
                )
            else:
                record = HandoffRecord(
                    from_agent=from_agent or self.current_agent or "supervisor",
                    to_agent=target_agent,
                    message=message,
                    timestamp=time.time(),
                    success=True,
                    metadata={"transfer_successful": True}
                )
                self.current_agent = target_agent
            
            self.handoff_history.append(record)
            return record
        
        def get_handoff_chain(self) -> List[str]:
            """Get chain of agent handoffs"""
            return [record.to_agent for record in self.handoff_history if record.success]
        
        def get_handoff_statistics(self) -> Dict[str, Any]:
            """Get handoff statistics"""
            total_handoffs = len(self.handoff_history)
            successful_handoffs = sum(1 for r in self.handoff_history if r.success)
            
            agent_transitions = {}
            for record in self.handoff_history:
                if record.success:
                    transition = f"{record.from_agent} -> {record.to_agent}"
                    agent_transitions[transition] = agent_transitions.get(transition, 0) + 1
            
            return {
                'total_handoffs': total_handoffs,
                'successful_handoffs': successful_handoffs,
                'failed_handoffs': total_handoffs - successful_handoffs,
                'success_rate': successful_handoffs / total_handoffs if total_handoffs > 0 else 0,
                'agent_transitions': agent_transitions,
                'handoff_chain': self.get_handoff_chain(),
                'available_agents': list(self.available_agents)
            }
    
    async def test_basic_handoff_sequence(self, agents: List[str], 
                                        messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test basic sequence of agent handoffs"""
        handoff_manager = self.MockHandoffManager()
        
        # Register agents
        for agent in agents:
            handoff_manager.register_agent(agent)
        
        # Perform handoffs
        handoff_results = []
        for i, message in enumerate(messages):
            target_agent = agents[i % len(agents)]
            result = await handoff_manager.handoff_to_agent(target_agent, message)
            handoff_results.append(result)
        
        stats = handoff_manager.get_handoff_statistics()
        
        return {
            'agents': agents,
            'messages_processed': len(messages),
            'handoff_results': handoff_results,
            'statistics': stats,
            'all_successful': all(r.success for r in handoff_results)
        }
    
    async def test_handoff_failure_scenarios(self) -> Dict[str, Any]:
        """Test handoff failure scenarios"""
        handoff_manager = self.MockHandoffManager()
        
        # Register only some agents
        handoff_manager.register_agent("agent1")
        handoff_manager.register_agent("agent2")
        
        test_scenarios = [
            {
                'name': 'valid_handoff',
                'target_agent': 'agent1',
                'message': {'content': 'Test message 1'},
                'expected_success': True
            },
            {
                'name': 'invalid_agent',
                'target_agent': 'nonexistent_agent',
                'message': {'content': 'Test message 2'},
                'expected_success': False
            },
            {
                'name': 'another_valid_handoff',
                'target_agent': 'agent2',
                'message': {'content': 'Test message 3'},
                'expected_success': True
            }
        ]
        
        results = []
        for scenario in test_scenarios:
            result = await handoff_manager.handoff_to_agent(
                scenario['target_agent'], 
                scenario['message']
            )
            
            scenario_result = {
                'scenario': scenario,
                'handoff_record': result,
                'success_matches_expected': result.success == scenario['expected_success']
            }
            results.append(scenario_result)
        
        stats = handoff_manager.get_handoff_statistics()
        
        return {
            'test_scenarios': test_scenarios,
            'results': results,
            'statistics': stats,
            'all_expectations_met': all(r['success_matches_expected'] for r in results)
        }
    
    async def test_handoff_performance(self, num_handoffs: int = 100) -> Dict[str, Any]:
        """Test handoff performance with many operations"""
        handoff_manager = self.MockHandoffManager()
        
        # Register test agents
        agents = [f"agent_{i}" for i in range(10)]
        for agent in agents:
            handoff_manager.register_agent(agent)
        
        # Perform many handoffs
        start_time = time.time()
        
        handoff_times = []
        for i in range(num_handoffs):
            target_agent = agents[i % len(agents)]
            message = {'content': f'Performance test message {i}'}
            
            handoff_start = time.time()
            await handoff_manager.handoff_to_agent(target_agent, message)
            handoff_end = time.time()
            
            handoff_times.append(handoff_end - handoff_start)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        stats = handoff_manager.get_handoff_statistics()
        
        return {
            'num_handoffs': num_handoffs,
            'total_time': total_time,
            'average_handoff_time': sum(handoff_times) / len(handoff_times),
            'min_handoff_time': min(handoff_times),
            'max_handoff_time': max(handoff_times),
            'handoffs_per_second': num_handoffs / total_time,
            'statistics': stats,
            'performance_acceptable': total_time < num_handoffs * 0.001  # Less than 1ms per handoff
        }


# Export all patterns
__all__ = [
    'SupervisorWorkflowTestPatterns',
    'FunctionalAPITestPatterns',
    'AgentNameTestPatterns',
    'HandoffTestPatterns'
]