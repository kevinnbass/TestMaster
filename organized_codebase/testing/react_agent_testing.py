# -*- coding: utf-8 -*-
"""
AgentScope ReAct Agent Testing Framework
========================================

Extracted from agentscope/tests/react_agent_test.py
Enhanced for TestMaster integration

Testing patterns for:
- ReAct (Reason + Act) agent behavior
- Hook system (pre/post reasoning, acting)
- Tool use and reasoning cycles
- Agent lifecycle management
- Model integration and response handling
- Memory and formatter integration
- Toolkit execution patterns
- Agent hook registration and execution
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock

import pytest


class MockMessage:
    """Mock message for agent testing"""
    
    def __init__(self, role: str, content: Union[str, List], name: str, **kwargs):
        self.role = role
        self.content = content
        self.name = name
        self.__dict__.update(kwargs)


class MockTextBlock:
    """Mock text block content"""
    
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    """Mock tool use block content"""
    
    def __init__(self, name: str, tool_id: str, input_data: Dict):
        self.type = "tool_use"
        self.name = name
        self.id = tool_id
        self.input = input_data


class MockChatResponse:
    """Mock chat model response"""
    
    def __init__(self, content: List):
        self.content = content
        self.status = "success"
        self.usage = {"input_tokens": 10, "output_tokens": 5}


class MockChatModel:
    """Mock chat model for testing"""
    
    def __init__(self, name: str = "test_model"):
        self.name = name
        self.stream = False
        self.fake_content = [MockTextBlock("Default response")]
        self.call_count = 0
        self.last_messages = []
    
    async def __call__(self, messages: List[Dict], **kwargs) -> MockChatResponse:
        """Mock model call"""
        self.call_count += 1
        self.last_messages = messages.copy()
        
        return MockChatResponse(content=self.fake_content)
    
    def set_response(self, content: List):
        """Set the next response content"""
        self.fake_content = content
    
    def reset_stats(self):
        """Reset call statistics"""
        self.call_count = 0
        self.last_messages.clear()


class MockMemory:
    """Mock memory for agent testing"""
    
    def __init__(self):
        self.messages = []
        self.clear_count = 0
    
    def add(self, message: MockMessage):
        """Add message to memory"""
        self.messages.append(message)
    
    def get_all(self) -> List[MockMessage]:
        """Get all messages"""
        return self.messages.copy()
    
    def clear(self):
        """Clear memory"""
        self.messages.clear()
        self.clear_count += 1
    
    def size(self) -> int:
        """Get memory size"""
        return len(self.messages)


class MockFormatter:
    """Mock formatter for message formatting"""
    
    def __init__(self):
        self.format_count = 0
        self.last_messages = []
    
    async def format(self, messages: List[MockMessage]) -> List[Dict]:
        """Format messages"""
        self.format_count += 1
        self.last_messages = messages.copy()
        
        formatted = []
        for msg in messages:
            formatted_msg = {
                "role": msg.role,
                "content": msg.content if isinstance(msg.content, str) else str(msg.content)
            }
            formatted.append(formatted_msg)
        
        return formatted


class MockTool:
    """Mock tool for agent testing"""
    
    def __init__(self, name: str, response: str = "Tool executed"):
        self.name = name
        self.response = response
        self.call_count = 0
        self.last_input = None
    
    async def execute(self, input_data: Dict) -> str:
        """Execute tool"""
        self.call_count += 1
        self.last_input = input_data
        return self.response


class MockToolkit:
    """Mock toolkit for agent testing"""
    
    def __init__(self):
        self.tools = {}
        self.execution_count = 0
    
    def add_tool(self, tool: MockTool):
        """Add tool to toolkit"""
        self.tools[tool.name] = tool
    
    async def execute_tool(self, tool_name: str, input_data: Dict) -> str:
        """Execute tool by name"""
        self.execution_count += 1
        
        if tool_name in self.tools:
            return await self.tools[tool_name].execute(input_data)
        else:
            raise ValueError(f"Tool '{tool_name}' not found")
    
    def get_tool_names(self) -> List[str]:
        """Get available tool names"""
        return list(self.tools.keys())
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if tool exists"""
        return tool_name in self.tools


class ReActAgent:
    """ReAct (Reason + Act) Agent implementation"""
    
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: MockChatModel,
        formatter: MockFormatter,
        memory: MockMemory,
        toolkit: MockToolkit,
        max_iterations: int = 5
    ):
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.formatter = formatter
        self.memory = memory
        self.toolkit = toolkit
        self.max_iterations = max_iterations
        self.finish_function_name = "finish"
        
        # Hook system
        self.hooks = {
            "pre_reasoning": {},
            "post_reasoning": {},
            "pre_acting": {},
            "post_acting": {}
        }
        
        # Statistics
        self.reasoning_count = 0
        self.acting_count = 0
        self.iteration_count = 0
        self.finished = False
    
    def register_instance_hook(
        self,
        hook_type: str,
        hook_name: str,
        hook_function: Callable
    ):
        """Register instance hook"""
        if hook_type in self.hooks:
            self.hooks[hook_type][hook_name] = hook_function
    
    async def execute_hooks(self, hook_type: str, **kwargs):
        """Execute hooks of given type"""
        if hook_type in self.hooks:
            for hook_name, hook_func in self.hooks[hook_type].items():
                await hook_func(self, kwargs)
    
    async def __call__(self, message: Optional[MockMessage] = None) -> MockMessage:
        """Execute ReAct cycle"""
        if message:
            self.memory.add(message)
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            
            # Reasoning phase
            await self.execute_hooks("pre_reasoning")
            reasoning_result = await self.reason()
            await self.execute_hooks("post_reasoning", output=reasoning_result)
            
            # Check if agent wants to finish
            if self.should_finish(reasoning_result):
                self.finished = True
                break
            
            # Acting phase
            await self.execute_hooks("pre_acting")
            acting_result = await self.act(reasoning_result)
            await self.execute_hooks("post_acting", output=acting_result)
            
            # Add result to memory
            if acting_result:
                self.memory.add(acting_result)
        
        return self.create_final_response()
    
    async def reason(self) -> MockMessage:
        """Reasoning step"""
        self.reasoning_count += 1
        
        # Get conversation history
        messages = self.memory.get_all()
        
        # Add system prompt if memory is empty
        if not messages:
            system_msg = MockMessage("system", self.sys_prompt, "system")
            messages = [system_msg]
        
        # Format messages
        formatted_messages = await self.formatter.format(messages)
        
        # Get model response
        response = await self.model(formatted_messages)
        
        # Create reasoning result message
        reasoning_msg = MockMessage(
            "assistant",
            response.content,
            self.name,
            reasoning_step=self.reasoning_count
        )
        
        return reasoning_msg
    
    def should_finish(self, reasoning_result: MockMessage) -> bool:
        """Check if agent should finish"""
        content = reasoning_result.content
        
        # Check for finish tool use
        if isinstance(content, list):
            for item in content:
                if hasattr(item, 'type') and item.type == "tool_use":
                    if item.name == self.finish_function_name:
                        return True
        
        # Check for finish keywords in text
        if isinstance(content, str):
            finish_keywords = ["FINISH", "DONE", "COMPLETE"]
            return any(keyword in content.upper() for keyword in finish_keywords)
        
        return False
    
    async def act(self, reasoning_result: MockMessage) -> Optional[MockMessage]:
        """Acting step"""
        self.acting_count += 1
        
        content = reasoning_result.content
        
        # Extract tool use from content
        tool_uses = self.extract_tool_uses(content)
        
        if not tool_uses:
            return None
        
        # Execute tools
        results = []
        for tool_use in tool_uses:
            try:
                result = await self.toolkit.execute_tool(
                    tool_use["name"],
                    tool_use["input"]
                )
                results.append(f"Tool '{tool_use['name']}' result: {result}")
            except Exception as e:
                results.append(f"Tool '{tool_use['name']}' error: {str(e)}")
        
        # Create acting result message
        acting_msg = MockMessage(
            "tool",
            "\n".join(results),
            "tool_results",
            acting_step=self.acting_count
        )
        
        return acting_msg
    
    def extract_tool_uses(self, content) -> List[Dict]:
        """Extract tool uses from content"""
        tool_uses = []
        
        if isinstance(content, list):
            for item in content:
                if hasattr(item, 'type') and item.type == "tool_use":
                    tool_uses.append({
                        "name": item.name,
                        "id": item.id,
                        "input": item.input
                    })
        
        return tool_uses
    
    def create_final_response(self) -> MockMessage:
        """Create final response message"""
        return MockMessage(
            "assistant",
            f"ReAct cycle completed. Iterations: {self.iteration_count}, "
            f"Reasoning steps: {self.reasoning_count}, Acting steps: {self.acting_count}",
            self.name,
            finished=self.finished,
            total_iterations=self.iteration_count
        )
    
    def reset(self):
        """Reset agent state"""
        self.reasoning_count = 0
        self.acting_count = 0
        self.iteration_count = 0
        self.finished = False
        self.memory.clear()


class ReActTestFramework:
    """Core framework for ReAct agent testing"""
    
    def __init__(self):
        self.agents = {}
        self.models = {}
        self.toolkits = {}
        self.test_scenarios = {}
        self.hook_registrations = []
    
    def create_test_model(self, name: str) -> MockChatModel:
        """Create test model"""
        model = MockChatModel(name)
        self.models[name] = model
        return model
    
    def create_test_toolkit(self, name: str) -> MockToolkit:
        """Create test toolkit"""
        toolkit = MockToolkit()
        self.toolkits[name] = toolkit
        return toolkit
    
    def create_react_agent(
        self,
        name: str,
        sys_prompt: str = "You are a helpful assistant.",
        model_name: str = "default",
        toolkit_name: str = "default"
    ) -> ReActAgent:
        """Create ReAct agent"""
        # Create or get model
        if model_name not in self.models:
            self.create_test_model(model_name)
        
        # Create or get toolkit
        if toolkit_name not in self.toolkits:
            self.create_test_toolkit(toolkit_name)
        
        agent = ReActAgent(
            name=name,
            sys_prompt=sys_prompt,
            model=self.models[model_name],
            formatter=MockFormatter(),
            memory=MockMemory(),
            toolkit=self.toolkits[toolkit_name]
        )
        
        self.agents[name] = agent
        return agent
    
    def add_tool_to_toolkit(self, toolkit_name: str, tool_name: str, response: str = "Tool executed"):
        """Add tool to toolkit"""
        if toolkit_name in self.toolkits:
            tool = MockTool(tool_name, response)
            self.toolkits[toolkit_name].add_tool(tool)
    
    def create_test_scenario(
        self,
        name: str,
        agent_name: str,
        model_responses: List[List],
        expected_iterations: int,
        expected_finish: bool = True
    ):
        """Create test scenario"""
        self.test_scenarios[name] = {
            "agent_name": agent_name,
            "model_responses": model_responses,
            "expected_iterations": expected_iterations,
            "expected_finish": expected_finish
        }
    
    async def run_test_scenario(self, scenario_name: str) -> Dict:
        """Run test scenario"""
        if scenario_name not in self.test_scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.test_scenarios[scenario_name]
        agent = self.agents[scenario["agent_name"]]
        model = agent.model
        
        # Reset agent state
        agent.reset()
        
        # Set up model responses
        response_queue = scenario["model_responses"].copy()
        
        async def mock_model_call(messages, **kwargs):
            if response_queue:
                content = response_queue.pop(0)
                model.fake_content = content
            return MockChatResponse(content=model.fake_content)
        
        # Replace model call
        original_call = model.__call__
        model.__call__ = mock_model_call
        
        try:
            # Run agent
            start_time = asyncio.get_event_loop().time()
            result = await agent()
            end_time = asyncio.get_event_loop().time()
            
            # Collect results
            test_result = {
                "scenario": scenario_name,
                "execution_time": end_time - start_time,
                "iterations": agent.iteration_count,
                "reasoning_steps": agent.reasoning_count,
                "acting_steps": agent.acting_count,
                "finished": agent.finished,
                "memory_size": agent.memory.size(),
                "model_calls": model.call_count,
                "toolkit_executions": agent.toolkit.execution_count,
                "final_response": result,
                "expectations_met": {
                    "iterations": agent.iteration_count == scenario["expected_iterations"],
                    "finished": agent.finished == scenario["expected_finish"]
                }
            }
            
            return test_result
        
        finally:
            # Restore original model call
            model.__call__ = original_call
    
    def register_global_hooks(self, agent_name: str, hooks: Dict[str, Callable]):
        """Register hooks for agent"""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            for hook_type, hook_func in hooks.items():
                agent.register_instance_hook(hook_type, "test_hook", hook_func)


class HookValidator:
    """Validator for hook system testing"""
    
    @staticmethod
    def create_counting_hook(hook_type: str):
        """Create hook that counts executions"""
        async def counting_hook(agent, kwargs):
            counter_name = f"cnt_{hook_type}"
            if hasattr(agent, counter_name):
                setattr(agent, counter_name, getattr(agent, counter_name) + 1)
            else:
                setattr(agent, counter_name, 1)
        
        return counting_hook
    
    @staticmethod
    def validate_hook_executions(agent: ReActAgent, expected_counts: Dict[str, int]) -> Dict:
        """Validate hook execution counts"""
        validation_results = {}
        
        for hook_type, expected_count in expected_counts.items():
            counter_name = f"cnt_{hook_type}"
            actual_count = getattr(agent, counter_name, 0)
            
            validation_results[hook_type] = {
                "expected": expected_count,
                "actual": actual_count,
                "is_valid": actual_count == expected_count
            }
        
        return validation_results
    
    @staticmethod
    def validate_agent_state(agent: ReActAgent, expected_state: Dict) -> Dict:
        """Validate agent state"""
        validation_results = {}
        
        for state_key, expected_value in expected_state.items():
            actual_value = getattr(agent, state_key, None)
            validation_results[state_key] = {
                "expected": expected_value,
                "actual": actual_value,
                "is_valid": actual_value == expected_value
            }
        
        return validation_results


class ReActAgentTest(IsolatedAsyncioTestCase):
    """Comprehensive ReAct agent testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = ReActTestFramework()
        self.validator = HookValidator()
        
        # Create default agent
        self.agent = self.framework.create_react_agent("test_agent", "You are a helpful assistant named Friday.")
        
        # Add test tools
        self.framework.add_tool_to_toolkit("default", "calculator", "42")
        self.framework.add_tool_to_toolkit("default", "weather", "Sunny, 25°C")
        self.framework.add_tool_to_toolkit("default", "finish", "Task completed")
    
    async def test_basic_react_cycle(self):
        """Test basic ReAct reasoning and acting cycle"""
        # Set up model to return text response
        self.agent.model.set_response([MockTextBlock("I need to think about this.")])
        
        # Run agent
        result = await self.agent()
        
        # Validate basic execution
        assert self.agent.reasoning_count > 0
        assert self.agent.iteration_count > 0
        assert result is not None
        assert hasattr(result, 'total_iterations')
    
    async def test_hook_system_execution(self):
        """Test hook system with counting hooks"""
        # Register counting hooks
        hooks = {
            "pre_reasoning": self.validator.create_counting_hook("pre_reasoning"),
            "post_reasoning": self.validator.create_counting_hook("post_reasoning"),
            "pre_acting": self.validator.create_counting_hook("pre_acting"),
            "post_acting": self.validator.create_counting_hook("post_acting")
        }
        
        for hook_type, hook_func in hooks.items():
            self.agent.register_instance_hook(hook_type, "test_hook", hook_func)
        
        # Run agent once
        self.agent.model.set_response([MockTextBlock("Simple response")])
        await self.agent()
        
        # Validate hook executions
        expected_counts = {
            "pre_reasoning": 1,
            "post_reasoning": 1,
            "pre_acting": 1,
            "post_acting": 1
        }
        
        validation_results = self.validator.validate_hook_executions(self.agent, expected_counts)
        
        for hook_type, result in validation_results.items():
            assert result["is_valid"], f"Hook {hook_type} count mismatch: expected {result['expected']}, got {result['actual']}"
    
    async def test_tool_execution_cycle(self):
        """Test tool execution in ReAct cycle"""
        # Set up model to return tool use
        tool_use_block = MockToolUseBlock("calculator", "calc_1", {"expression": "2+2"})
        self.agent.model.set_response([tool_use_block])
        
        # Run agent
        result = await self.agent()
        
        # Validate tool execution
        assert self.agent.acting_count > 0
        assert self.agent.toolkit.execution_count > 0
        
        # Check if calculator tool was called
        calculator_tool = self.agent.toolkit.tools["calculator"]
        assert calculator_tool.call_count > 0
        assert calculator_tool.last_input == {"expression": "2+2"}
    
    async def test_finish_condition(self):
        """Test agent finish condition"""
        # Set up model to return finish tool use
        finish_tool_use = MockToolUseBlock(self.agent.finish_function_name, "finish_1", {"response": "Task completed"})
        self.agent.model.set_response([finish_tool_use])
        
        # Run agent
        result = await self.agent()
        
        # Validate finish behavior
        assert self.agent.finished == True
        assert "completed" in result.content.lower()
    
    async def test_multiple_iterations(self):
        """Test multiple ReAct iterations"""
        # Register counting hooks
        hooks = {
            "pre_reasoning": self.validator.create_counting_hook("pre_reasoning"),
            "post_reasoning": self.validator.create_counting_hook("post_reasoning")
        }
        
        for hook_type, hook_func in hooks.items():
            self.agent.register_instance_hook(hook_type, "test_hook", hook_func)
        
        # Set up model responses for multiple iterations
        responses = [
            [MockTextBlock("First reasoning step")],
            [MockToolUseBlock(self.agent.finish_function_name, "finish_1", {"response": "Done"})]
        ]
        
        # Create test scenario
        self.framework.create_test_scenario(
            "multi_iteration",
            "test_agent",
            responses,
            expected_iterations=2,
            expected_finish=True
        )
        
        # Run scenario
        result = await self.framework.run_test_scenario("multi_iteration")
        
        # Validate multiple iterations
        assert result["expectations_met"]["iterations"]
        assert result["expectations_met"]["finished"]
        assert result["reasoning_steps"] == 2
    
    async def test_memory_integration(self):
        """Test memory integration with ReAct cycle"""
        # Add initial message to memory
        initial_msg = MockMessage("user", "What's 2+2?", "user")
        self.agent.memory.add(initial_msg)
        
        # Run agent
        self.agent.model.set_response([MockTextBlock("Let me calculate that.")])
        await self.agent()
        
        # Validate memory usage
        assert self.agent.memory.size() > 1  # Initial message + agent responses
        assert self.agent.formatter.format_count > 0  # Formatter was used
    
    async def test_error_handling_in_tools(self):
        """Test error handling when tools fail"""
        # Add a tool that will fail
        failing_tool = MockTool("failing_tool", "This will fail")
        
        async def failing_execute(input_data):
            raise ValueError("Tool execution failed")
        
        failing_tool.execute = failing_execute
        self.agent.toolkit.add_tool(failing_tool)
        
        # Set up model to use the failing tool
        tool_use_block = MockToolUseBlock("failing_tool", "fail_1", {"input": "test"})
        self.agent.model.set_response([tool_use_block])
        
        # Run agent (should handle error gracefully)
        result = await self.agent()
        
        # Validate error handling
        assert result is not None
        # Memory should contain error message
        messages = self.agent.memory.get_all()
        error_messages = [msg for msg in messages if "error" in str(msg.content).lower()]
        assert len(error_messages) > 0
    
    async def test_max_iterations_limit(self):
        """Test max iterations limit"""
        # Set low max iterations
        self.agent.max_iterations = 2
        
        # Set up model to never finish
        self.agent.model.set_response([MockTextBlock("Keep thinking...")])
        
        # Run agent
        result = await self.agent()
        
        # Validate iteration limit
        assert self.agent.iteration_count == 2
        assert not self.agent.finished  # Should not finish naturally
    
    async def test_agent_state_validation(self):
        """Test comprehensive agent state validation"""
        # Run complete cycle
        finish_tool_use = MockToolUseBlock(self.agent.finish_function_name, "finish_1", {"response": "Complete"})
        self.agent.model.set_response([finish_tool_use])
        
        await self.agent()
        
        # Validate final state
        expected_state = {
            "finished": True,
            "reasoning_count": 1,
            "acting_count": 1,
            "iteration_count": 1
        }
        
        validation_results = self.validator.validate_agent_state(self.agent, expected_state)
        
        for state_key, result in validation_results.items():
            assert result["is_valid"], f"State {state_key} mismatch: expected {result['expected']}, got {result['actual']}"
    
    async def test_framework_scenario_system(self):
        """Test framework scenario creation and execution"""
        # Create complex scenario
        responses = [
            [MockTextBlock("I need to use the calculator")],
            [MockToolUseBlock("calculator", "calc_1", {"expression": "10*5"})],
            [MockToolUseBlock(self.agent.finish_function_name, "finish_1", {"response": "The answer is 50"})]
        ]
        
        self.framework.create_test_scenario(
            "calculation_scenario",
            "test_agent",
            responses,
            expected_iterations=3,
            expected_finish=True
        )
        
        # Run scenario
        result = await self.framework.run_test_scenario("calculation_scenario")
        
        # Validate scenario execution
        assert result["expectations_met"]["iterations"]
        assert result["expectations_met"]["finished"]
        assert result["toolkit_executions"] > 0
        assert "execution_time" in result
        assert result["execution_time"] >= 0


# Pytest integration
@pytest.fixture
def react_framework():
    """Pytest fixture for ReAct framework"""
    return ReActTestFramework()


@pytest.fixture
def hook_validator():
    """Pytest fixture for hook validator"""
    return HookValidator()


def test_react_framework_creation(react_framework):
    """Test ReAct framework creation"""
    agent = react_framework.create_react_agent("test", "Test agent")
    assert agent.name == "test"
    assert "test" in react_framework.agents


def test_model_creation(react_framework):
    """Test model creation"""
    model = react_framework.create_test_model("test_model")
    assert model.name == "test_model"
    assert "test_model" in react_framework.models


def test_toolkit_creation(react_framework):
    """Test toolkit creation"""
    toolkit = react_framework.create_test_toolkit("test_toolkit")
    assert "test_toolkit" in react_framework.toolkits
    
    # Add tool
    react_framework.add_tool_to_toolkit("test_toolkit", "test_tool", "Test response")
    assert toolkit.has_tool("test_tool")


@pytest.mark.asyncio
async def test_simple_hook_execution(hook_validator):
    """Test simple hook execution"""
    hook = hook_validator.create_counting_hook("test")
    
    # Create mock agent
    agent = MagicMock()
    
    # Execute hook
    await hook(agent, {})
    
    # Validate counter was set
    assert hasattr(agent, "cnt_test")
    assert agent.cnt_test == 1


def test_hook_validation(hook_validator):
    """Test hook validation"""
    # Create mock agent with counters
    agent = MagicMock()
    agent.cnt_pre_reasoning = 2
    agent.cnt_post_reasoning = 2
    
    expected_counts = {
        "pre_reasoning": 2,
        "post_reasoning": 2
    }
    
    results = hook_validator.validate_hook_executions(agent, expected_counts)
    
    for hook_type, result in results.items():
        assert result["is_valid"]