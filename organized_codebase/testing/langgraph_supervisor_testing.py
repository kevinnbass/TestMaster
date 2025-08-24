"""
LangGraph Supervisor Testing Framework
Extracted from LangGraph supervisor multi-agent orchestration patterns.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Sequence
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time
import uuid


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MockMessage:
    """Mock message object for agent communication"""
    content: str
    name: Optional[str] = None
    type: str = "human"
    tool_calls: Optional[List[Dict[str, Any]]] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


@dataclass 
class MockToolCall:
    """Mock tool call for agent actions"""
    name: str
    args: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "tool_call"


class MockChatModel:
    """Mock chat model for supervisor and agent testing"""
    
    def __init__(self, responses: List[MockMessage]):
        self.responses = responses
        self.idx = 0
        self.bound_tools = []
    
    def bind_tools(self, tools: List[Any]) -> 'MockChatModel':
        """Bind tools to the model"""
        self.bound_tools = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]
        return self
    
    def invoke(self, messages: List[MockMessage]) -> MockMessage:
        """Invoke the model with messages"""
        if self.idx < len(self.responses):
            response = self.responses[self.idx]
            self.idx += 1
            return response
        else:
            return MockMessage(content="Default response", type="ai")
    
    def generate(self, messages: List[MockMessage], **kwargs) -> Dict[str, Any]:
        """Generate response with metadata"""
        response = self.invoke(messages)
        return {
            "generations": [{"message": response}],
            "llm_output": {"model_name": "mock-model"}
        }


class MockAgent:
    """Mock agent for supervisor testing"""
    
    def __init__(self, name: str, model: MockChatModel, tools: List[Any] = None):
        self.name = name
        self.model = model
        self.tools = tools or []
        self.state = AgentState.IDLE
        self.message_history = []
        self.execution_count = 0
    
    def invoke(self, state: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        """Invoke agent with state"""
        self.state = AgentState.ACTIVE
        self.execution_count += 1
        
        messages = state.get("messages", [])
        self.message_history.extend(messages)
        
        # Generate response
        response = self.model.invoke(messages)
        response.name = self.name
        
        # Update state
        self.state = AgentState.COMPLETED
        
        return {"messages": [response]}
    
    def reset(self) -> None:
        """Reset agent state"""
        self.state = AgentState.IDLE
        self.message_history.clear()
        self.execution_count = 0
        self.model.idx = 0


class MockSupervisor:
    """Mock supervisor for multi-agent orchestration"""
    
    def __init__(self, agents: List[MockAgent], model: MockChatModel, 
                 output_mode: str = "last_message", add_handoff_messages: bool = True):
        self.agents = {agent.name: agent for agent in agents}
        self.model = model
        self.output_mode = output_mode
        self.add_handoff_messages = add_handoff_messages
        self.message_history = []
        self.handoff_tools = self._create_handoff_tools()
        self.current_agent = None
        self.execution_stats = {
            'total_handoffs': 0,
            'agent_executions': defaultdict(int),
            'message_count': 0
        }
    
    def _create_handoff_tools(self) -> List[Dict[str, Any]]:
        """Create handoff tools for agent transfers"""
        tools = []
        for agent_name in self.agents.keys():
            tools.append({
                "name": f"transfer_to_{agent_name}",
                "description": f"Transfer control to {agent_name}",
                "agent": agent_name
            })
        return tools
    
    def invoke(self, state: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        """Invoke supervisor workflow"""
        messages = state.get("messages", [])
        self.message_history.extend(messages)
        self.execution_stats['message_count'] += len(messages)
        
        # Supervisor decision making
        supervisor_response = self.model.invoke(messages)
        supervisor_response.name = "supervisor"
        
        result_messages = list(messages)
        result_messages.append(supervisor_response)
        
        # Handle tool calls (agent handoffs)
        if supervisor_response.tool_calls:
            for tool_call in supervisor_response.tool_calls:
                if tool_call.get("name", "").startswith("transfer_to_"):
                    agent_name = tool_call["name"].replace("transfer_to_", "")
                    agent_response = self._execute_agent(agent_name, {"messages": result_messages})
                    
                    if self.add_handoff_messages:
                        # Add handoff message
                        handoff_msg = MockMessage(
                            content=f"Successfully transferred to {agent_name}",
                            name=tool_call["name"],
                            type="tool"
                        )
                        result_messages.append(handoff_msg)
                    
                    # Add agent response based on output mode
                    if self.output_mode == "full_history":
                        result_messages.extend(agent_response.get("messages", []))
                    else:  # last_message
                        if agent_response.get("messages"):
                            result_messages.append(agent_response["messages"][-1])
                    
                    # Add transfer back message
                    transfer_back_msg = MockMessage(
                        content="Transferring back to supervisor",
                        name=agent_name,
                        type="ai",
                        tool_calls=[{"name": "transfer_back_to_supervisor", "args": {}}]
                    )
                    result_messages.append(transfer_back_msg)
                    
                    transfer_confirm_msg = MockMessage(
                        content="Successfully transferred back to supervisor",
                        name="transfer_back_to_supervisor",
                        type="tool"
                    )
                    result_messages.append(transfer_confirm_msg)
                    
                    self.execution_stats['total_handoffs'] += 1
        
        # Final supervisor response
        if len(self.model.responses) > 1:
            final_response = self.model.invoke([])
            final_response.name = "supervisor"
            result_messages.append(final_response)
        
        return {"messages": result_messages}
    
    def _execute_agent(self, agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        self.current_agent = agent_name
        self.execution_stats['agent_executions'][agent_name] += 1
        
        return agent.invoke(state)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get supervisor execution statistics"""
        return {
            'total_handoffs': self.execution_stats['total_handoffs'],
            'agent_executions': dict(self.execution_stats['agent_executions']),
            'message_count': self.execution_stats['message_count'],
            'agents_used': list(self.execution_stats['agent_executions'].keys()),
            'total_agents': len(self.agents)
        }


class MockTool:
    """Mock tool for agent capabilities"""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
        self.call_count = 0
        self.call_history = []
    
    def invoke(self, args: Dict[str, Any]) -> Any:
        """Invoke tool with arguments"""
        self.call_count += 1
        self.call_history.append(args)
        return self.func(**args)
    
    def reset(self) -> None:
        """Reset tool statistics"""
        self.call_count = 0
        self.call_history.clear()


class LangGraphSupervisorTestFramework:
    """Comprehensive test framework for LangGraph supervisor patterns"""
    
    def __init__(self):
        self.supervisors = {}
        self.agents = {}
        self.tools = {}
        self.test_scenarios = []
        self.performance_metrics = {
            'handoff_times': [],
            'agent_response_times': [],
            'message_processing_times': []
        }
    
    def create_test_tools(self) -> Dict[str, MockTool]:
        """Create standard test tools"""
        def add(a: float, b: float) -> float:
            return a + b
        
        def web_search(query: str) -> str:
            return f"Search results for: {query}"
        
        def echo_tool(text: str) -> str:
            return f"Echo: {text}"
        
        return {
            "add": MockTool("add", "Add two numbers", add),
            "web_search": MockTool("web_search", "Search the web", web_search),
            "echo_tool": MockTool("echo_tool", "Echo input text", echo_tool)
        }
    
    def test_basic_supervisor_workflow(self) -> bool:
        """Test basic supervisor workflow with agent handoffs"""
        try:
            # Create test tools
            tools = self.create_test_tools()
            
            # Create supervisor messages for handoffs
            supervisor_messages = [
                MockMessage(
                    content="",
                    type="ai",
                    tool_calls=[{"name": "transfer_to_research_agent", "args": {}}]
                ),
                MockMessage(
                    content="",
                    type="ai", 
                    tool_calls=[{"name": "transfer_to_math_agent", "args": {}}]
                ),
                MockMessage(content="Task completed successfully", type="ai")
            ]
            
            # Create agent responses
            research_responses = [
                MockMessage(content="Research completed: Found relevant data", type="ai")
            ]
            math_responses = [
                MockMessage(content="Calculation result: 42", type="ai")
            ]
            
            # Create agents
            research_agent = MockAgent(
                "research_agent",
                MockChatModel(research_responses),
                [tools["web_search"]]
            )
            
            math_agent = MockAgent(
                "math_agent",
                MockChatModel(math_responses),
                [tools["add"]]
            )
            
            # Create supervisor
            supervisor = MockSupervisor(
                agents=[research_agent, math_agent],
                model=MockChatModel(supervisor_messages)
            )
            
            # Test execution
            result = supervisor.invoke({
                "messages": [MockMessage(content="Solve this complex problem", type="human")]
            })
            
            # Verify results
            assert len(result["messages"]) > 3  # Initial + handoffs + responses
            assert supervisor.execution_stats['total_handoffs'] == 2
            assert supervisor.execution_stats['agent_executions']['research_agent'] == 1
            assert supervisor.execution_stats['agent_executions']['math_agent'] == 1
            
            return True
        except Exception as e:
            pytest.fail(f"Basic supervisor workflow test failed: {e}")
    
    def test_supervisor_message_forwarding(self) -> bool:
        """Test supervisor message forwarding capabilities"""
        try:
            # Create echo agent
            echo_responses = [MockMessage(content="Echo: test forwarding!", type="ai")]
            echo_agent = MockAgent("echo_agent", MockChatModel(echo_responses))
            
            # Supervisor with forwarding capability
            supervisor_messages = [
                MockMessage(
                    content="",
                    type="ai",
                    tool_calls=[{"name": "transfer_to_echo_agent", "args": {}}]
                ),
                MockMessage(
                    content="",
                    type="ai",
                    tool_calls=[{"name": "forward_message", "args": {"from_agent": "echo_agent"}}]
                ),
                MockMessage(content="Echo: test forwarding!", type="ai")
            ]
            
            supervisor = MockSupervisor(
                agents=[echo_agent],
                model=MockChatModel(supervisor_messages)
            )
            
            result = supervisor.invoke({
                "messages": [MockMessage(content="Test message", type="human")]
            })
            
            # Verify forwarding worked
            assert len(result["messages"]) >= 4
            assert any("Echo: test forwarding!" in msg.content for msg in result["messages"])
            
            return True
        except Exception as e:
            pytest.fail(f"Message forwarding test failed: {e}")
    
    def test_supervisor_output_modes(self) -> bool:
        """Test different supervisor output modes"""
        try:
            # Create multi-step agent
            multi_step_responses = [
                MockMessage(content="Step 1 complete", type="ai"),
                MockMessage(content="Step 2 complete", type="ai"),
                MockMessage(content="Final result", type="ai")
            ]
            
            agent = MockAgent("test_agent", MockChatModel(multi_step_responses))
            
            supervisor_messages = [
                MockMessage(
                    content="",
                    type="ai",
                    tool_calls=[{"name": "transfer_to_test_agent", "args": {}}]
                )
            ]
            
            # Test last_message mode
            supervisor_last = MockSupervisor(
                agents=[agent],
                model=MockChatModel(supervisor_messages.copy()),
                output_mode="last_message"
            )
            
            result_last = supervisor_last.invoke({
                "messages": [MockMessage(content="Test", type="human")]
            })
            
            # Test full_history mode  
            agent.reset()  # Reset agent for second test
            supervisor_full = MockSupervisor(
                agents=[agent],
                model=MockChatModel(supervisor_messages.copy()),
                output_mode="full_history"
            )
            
            result_full = supervisor_full.invoke({
                "messages": [MockMessage(content="Test", type="human")]
            })
            
            # Verify different output modes produce different message counts
            assert len(result_full["messages"]) >= len(result_last["messages"])
            
            return True
        except Exception as e:
            pytest.fail(f"Output modes test failed: {e}")
    
    def test_supervisor_error_handling(self) -> bool:
        """Test supervisor error handling"""
        try:
            # Create failing agent
            failing_agent = MockAgent("failing_agent", MockChatModel([]))
            failing_agent.invoke = lambda state, config=None: {"error": "Agent failed"}
            
            supervisor_messages = [
                MockMessage(
                    content="",
                    type="ai",
                    tool_calls=[{"name": "transfer_to_failing_agent", "args": {}}]
                )
            ]
            
            supervisor = MockSupervisor(
                agents=[failing_agent],
                model=MockChatModel(supervisor_messages)
            )
            
            # Test error handling
            try:
                supervisor._execute_agent("failing_agent", {"messages": []})
                # Should handle gracefully or raise appropriate exception
            except Exception:
                pass  # Expected behavior
            
            # Test invalid agent transfer
            try:
                supervisor._execute_agent("nonexistent_agent", {"messages": []})
                assert False, "Should have raised ValueError"
            except ValueError:
                pass  # Expected
            
            return True
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
    
    def test_supervisor_metadata_passing(self) -> bool:
        """Test metadata passing to sub-agents"""
        try:
            # Create metadata-aware agent
            def metadata_check_agent(state: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
                assert config is not None
                assert config.get("metadata", {}).get("test_key") == "test_value"
                return {"messages": [MockMessage(content="Metadata received", type="ai")]}
            
            agent = MockAgent("metadata_agent", MockChatModel([]))
            agent.invoke = metadata_check_agent
            
            supervisor_messages = [
                MockMessage(
                    content="",
                    type="ai",
                    tool_calls=[{"name": "transfer_to_metadata_agent", "args": {}}]
                )
            ]
            
            supervisor = MockSupervisor(
                agents=[agent],
                model=MockChatModel(supervisor_messages)
            )
            
            # Test with metadata config
            config = {"metadata": {"test_key": "test_value", "another_key": 123}}
            
            result = supervisor.invoke({
                "messages": [MockMessage(content="Test", type="human")]
            }, config=config)
            
            assert len(result["messages"]) > 0
            
            return True
        except Exception as e:
            pytest.fail(f"Metadata passing test failed: {e}")
    
    def test_concurrent_agent_execution(self) -> bool:
        """Test concurrent agent execution patterns"""
        try:
            # Create multiple agents
            agents = []
            for i in range(3):
                responses = [MockMessage(content=f"Agent {i} response", type="ai")]
                agent = MockAgent(f"agent_{i}", MockChatModel(responses))
                agents.append(agent)
            
            supervisor_messages = [
                MockMessage(
                    content="",
                    type="ai",
                    tool_calls=[
                        {"name": "transfer_to_agent_0", "args": {}},
                        {"name": "transfer_to_agent_1", "args": {}},
                        {"name": "transfer_to_agent_2", "args": {}}
                    ]
                )
            ]
            
            supervisor = MockSupervisor(
                agents=agents,
                model=MockChatModel(supervisor_messages)
            )
            
            result = supervisor.invoke({
                "messages": [MockMessage(content="Test concurrent execution", type="human")]
            })
            
            # Verify all agents were executed
            stats = supervisor.get_execution_stats()
            assert stats['agents_used'] == ["agent_0", "agent_1", "agent_2"]
            assert stats['total_handoffs'] == 3
            
            return True
        except Exception as e:
            pytest.fail(f"Concurrent execution test failed: {e}")
    
    def test_supervisor_performance_metrics(self) -> bool:
        """Test supervisor performance monitoring"""
        try:
            # Create performance test scenario
            responses = [MockMessage(content="Performance test response", type="ai")]
            agent = MockAgent("perf_agent", MockChatModel(responses))
            
            supervisor_messages = [
                MockMessage(
                    content="",
                    type="ai",
                    tool_calls=[{"name": "transfer_to_perf_agent", "args": {}}]
                )
            ]
            
            supervisor = MockSupervisor(
                agents=[agent],
                model=MockChatModel(supervisor_messages)
            )
            
            # Measure performance
            start_time = time.time()
            for _ in range(10):
                supervisor.invoke({
                    "messages": [MockMessage(content="Performance test", type="human")]
                })
                agent.reset()
                supervisor.model.idx = 0
            
            execution_time = time.time() - start_time
            
            # Verify performance metrics
            stats = supervisor.get_execution_stats()
            assert stats['agent_executions']['perf_agent'] == 10
            assert stats['total_handoffs'] == 10
            assert execution_time < 1.0  # Should complete quickly
            
            return True
        except Exception as e:
            pytest.fail(f"Performance metrics test failed: {e}")
    
    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all supervisor tests"""
        results = {}
        
        test_methods = [
            'test_basic_supervisor_workflow',
            'test_supervisor_message_forwarding',
            'test_supervisor_output_modes',
            'test_supervisor_error_handling',
            'test_supervisor_metadata_passing',
            'test_concurrent_agent_execution',
            'test_supervisor_performance_metrics'
        ]
        
        for test_method in test_methods:
            try:
                results[test_method] = getattr(self, test_method)()
            except Exception as e:
                results[test_method] = False
                print(f"{test_method} failed: {e}")
        
        return results


# Pytest integration patterns
class TestLangGraphSupervisor:
    """Pytest test class for LangGraph supervisor patterns"""
    
    @pytest.fixture
    def framework(self):
        return LangGraphSupervisorTestFramework()
    
    @pytest.fixture
    def test_agents(self, framework):
        tools = framework.create_test_tools()
        
        research_agent = MockAgent(
            "research_agent",
            MockChatModel([MockMessage(content="Research complete", type="ai")]),
            [tools["web_search"]]
        )
        
        math_agent = MockAgent(
            "math_agent",
            MockChatModel([MockMessage(content="Math complete", type="ai")]),
            [tools["add"]]
        )
        
        return [research_agent, math_agent]
    
    def test_supervisor_workflow_execution(self, framework, test_agents):
        """Test supervisor workflow execution"""
        supervisor_messages = [
            MockMessage(
                content="",
                type="ai",
                tool_calls=[{"name": "transfer_to_research_agent", "args": {}}]
            ),
            MockMessage(content="Task completed", type="ai")
        ]
        
        supervisor = MockSupervisor(
            agents=test_agents,
            model=MockChatModel(supervisor_messages)
        )
        
        result = supervisor.invoke({
            "messages": [MockMessage(content="Test task", type="human")]
        })
        
        assert len(result["messages"]) > 2
        assert supervisor.execution_stats['total_handoffs'] == 1
    
    def test_multi_agent_handoffs(self, framework, test_agents):
        """Test multiple agent handoffs"""
        supervisor_messages = [
            MockMessage(
                content="",
                type="ai",
                tool_calls=[{"name": "transfer_to_research_agent", "args": {}}]
            ),
            MockMessage(
                content="",
                type="ai",
                tool_calls=[{"name": "transfer_to_math_agent", "args": {}}]
            ),
            MockMessage(content="All tasks completed", type="ai")
        ]
        
        supervisor = MockSupervisor(
            agents=test_agents,
            model=MockChatModel(supervisor_messages)
        )
        
        result = supervisor.invoke({
            "messages": [MockMessage(content="Complex multi-step task", type="human")]
        })
        
        stats = supervisor.get_execution_stats()
        assert stats['total_handoffs'] == 2
        assert len(stats['agents_used']) == 2
        assert 'research_agent' in stats['agents_used']
        assert 'math_agent' in stats['agents_used']