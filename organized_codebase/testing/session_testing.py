# -*- coding: utf-8 -*-
"""
AgentScope Session Management Testing Framework
==============================================

Extracted from agentscope/tests/session_test.py
Enhanced for TestMaster integration

Testing patterns for:
- JSON session persistence and restoration
- Agent state serialization and deserialization
- Memory state management across sessions
- Multi-agent session coordination
- Session file handling and cleanup
- Agent registration and state tracking
- Cross-session continuity validation
- Session metadata management
"""

import asyncio
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Union
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

import pytest


class MockMessage:
    """Mock message for session testing"""
    
    def __init__(self, role: str, content: str, name: str, timestamp: Optional[float] = None, **kwargs):
        self.role = role
        self.content = content
        self.name = name
        self.timestamp = timestamp or time.time()
        self.__dict__.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "timestamp": self.timestamp,
            **{k: v for k, v in self.__dict__.items() if k not in ['role', 'content', 'name', 'timestamp']}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MockMessage':
        """Create message from dictionary"""
        return cls(**data)


class MockMemory:
    """Mock memory for session testing"""
    
    def __init__(self):
        self.messages = []
        self.metadata = {}
    
    async def add(self, message: MockMessage):
        """Add message to memory"""
        self.messages.append(message)
    
    def get_all(self) -> List[MockMessage]:
        """Get all messages"""
        return self.messages.copy()
    
    def clear(self):
        """Clear memory"""
        self.messages.clear()
    
    def size(self) -> int:
        """Get memory size"""
        return len(self.messages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary"""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata.copy()
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load memory from dictionary"""
        self.messages = [MockMessage.from_dict(msg_data) for msg_data in data.get("messages", [])]
        self.metadata = data.get("metadata", {})


class SessionableAgent:
    """Agent with session state management"""
    
    def __init__(self, name: str, sys_prompt: str = "You are a helpful assistant."):
        self.name = name
        self.sys_prompt = sys_prompt
        self.memory = MockMemory()
        self.agent_id = f"agent_{name}_{int(time.time())}"
        self.creation_time = time.time()
        
        # Registered state attributes
        self.registered_states = set()
        self.register_state("name")
        self.register_state("sys_prompt")
        self.register_state("agent_id")
        self.register_state("creation_time")
        
        # Runtime statistics
        self.reply_count = 0
        self.observe_count = 0
        self.last_activity = time.time()
    
    def register_state(self, attribute_name: str):
        """Register attribute for state persistence"""
        self.registered_states.add(attribute_name)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        state = {}
        for attr_name in self.registered_states:
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                # Handle serializable types
                if isinstance(value, (str, int, float, bool, list, dict)):
                    state[attr_name] = value
                else:
                    state[attr_name] = str(value)
        
        # Add memory state
        state["memory"] = self.memory.to_dict()
        
        # Add runtime stats
        state["runtime_stats"] = {
            "reply_count": self.reply_count,
            "observe_count": self.observe_count,
            "last_activity": self.last_activity
        }
        
        return state
    
    def set_state(self, state: Dict[str, Any]):
        """Restore agent state"""
        for attr_name, value in state.items():
            if attr_name == "memory":
                self.memory.from_dict(value)
            elif attr_name == "runtime_stats":
                self.reply_count = value.get("reply_count", 0)
                self.observe_count = value.get("observe_count", 0)
                self.last_activity = value.get("last_activity", time.time())
            elif attr_name in self.registered_states:
                setattr(self, attr_name, value)
    
    async def reply(self, message: MockMessage) -> MockMessage:
        """Reply to message with state tracking"""
        self.reply_count += 1
        self.last_activity = time.time()
        
        # Add to memory
        await self.memory.add(message)
        
        # Generate response
        response = MockMessage(
            "assistant",
            f"Response from {self.name}: {message.content}",
            self.name
        )
        
        await self.memory.add(response)
        return response
    
    async def observe(self, message: MockMessage):
        """Observe message with state tracking"""
        self.observe_count += 1
        self.last_activity = time.time()
        await self.memory.add(message)
    
    async def handle_interrupt(self, message: Optional[MockMessage] = None) -> MockMessage:
        """Handle interrupt"""
        return MockMessage("system", "Interrupt handled", "system")


class JSONSession:
    """JSON-based session manager"""
    
    def __init__(self, session_id: str, save_dir: str = "./sessions"):
        self.session_id = session_id
        self.save_dir = save_dir
        self.session_file = os.path.join(save_dir, f"{session_id}.json")
        self.agents = {}
        self.session_metadata = {
            "session_id": session_id,
            "created_at": time.time(),
            "last_saved": None,
            "version": "1.0"
        }
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
    
    def add_agent(self, agent_name: str, agent: SessionableAgent):
        """Add agent to session"""
        self.agents[agent_name] = agent
        self.session_metadata["last_updated"] = time.time()
    
    async def save_session_state(self, **agents):
        """Save session state to file"""
        # Update agents
        for agent_name, agent in agents.items():
            self.agents[agent_name] = agent
        
        # Prepare session data
        session_data = {
            "metadata": self.session_metadata.copy(),
            "agents": {}
        }
        
        # Save agent states
        for agent_name, agent in self.agents.items():
            session_data["agents"][agent_name] = {
                "class_name": agent.__class__.__name__,
                "state": agent.get_state()
            }
        
        # Update metadata
        session_data["metadata"]["last_saved"] = time.time()
        session_data["metadata"]["agent_count"] = len(self.agents)
        
        # Write to file
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        self.session_metadata["last_saved"] = session_data["metadata"]["last_saved"]
    
    async def load_session_state(self) -> Dict[str, Any]:
        """Load session state from file"""
        if not os.path.exists(self.session_file):
            raise FileNotFoundError(f"Session file not found: {self.session_file}")
        
        with open(self.session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        return session_data
    
    async def restore_agents(self, agent_classes: Dict[str, type]) -> Dict[str, SessionableAgent]:
        """Restore agents from session state"""
        session_data = await self.load_session_state()
        restored_agents = {}
        
        for agent_name, agent_data in session_data.get("agents", {}).items():
            class_name = agent_data.get("class_name", "SessionableAgent")
            agent_state = agent_data.get("state", {})
            
            # Create agent instance
            if class_name in agent_classes:
                agent_class = agent_classes[class_name]
                agent = agent_class(agent_name)
            else:
                # Default to SessionableAgent
                agent = SessionableAgent(agent_name)
            
            # Restore state
            agent.set_state(agent_state)
            restored_agents[agent_name] = agent
        
        return restored_agents
    
    def session_exists(self) -> bool:
        """Check if session file exists"""
        return os.path.exists(self.session_file)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        info = self.session_metadata.copy()
        info["file_path"] = self.session_file
        info["file_exists"] = self.session_exists()
        info["current_agent_count"] = len(self.agents)
        
        if self.session_exists():
            file_stat = os.stat(self.session_file)
            info["file_size"] = file_stat.st_size
            info["file_modified"] = file_stat.st_mtime
        
        return info
    
    def cleanup(self):
        """Clean up session file"""
        if os.path.exists(self.session_file):
            os.remove(self.session_file)


class SessionTestFramework:
    """Core framework for session testing"""
    
    def __init__(self):
        self.sessions = {}
        self.temp_dirs = []
        self.test_agents = {}
        self.session_scenarios = {}
    
    def create_temp_session_dir(self) -> str:
        """Create temporary directory for session files"""
        temp_dir = tempfile.mkdtemp(prefix="session_test_")
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_session(self, session_id: str, save_dir: Optional[str] = None) -> JSONSession:
        """Create session instance"""
        if save_dir is None:
            save_dir = self.create_temp_session_dir()
        
        session = JSONSession(session_id, save_dir)
        self.sessions[session_id] = session
        return session
    
    def create_test_agent(self, name: str, sys_prompt: str = None) -> SessionableAgent:
        """Create test agent"""
        agent = SessionableAgent(name, sys_prompt or f"I am {name}, a test agent.")
        self.test_agents[name] = agent
        return agent
    
    async def create_session_scenario(
        self,
        scenario_name: str,
        session_id: str,
        agents: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]]
    ):
        """Create session test scenario"""
        scenario = {
            "session_id": session_id,
            "agents": agents,
            "interactions": interactions,
            "created_at": time.time()
        }
        
        self.session_scenarios[scenario_name] = scenario
    
    async def run_session_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Run session scenario test"""
        if scenario_name not in self.session_scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.session_scenarios[scenario_name]
        session = self.create_session(scenario["session_id"])
        
        # Create agents
        created_agents = {}
        for agent_config in scenario["agents"]:
            agent = self.create_test_agent(
                agent_config["name"],
                agent_config.get("sys_prompt", None)
            )
            created_agents[agent_config["name"]] = agent
            session.add_agent(agent_config["name"], agent)
        
        # Execute interactions
        interaction_results = []
        for interaction in scenario["interactions"]:
            if interaction["type"] == "message":
                sender = created_agents[interaction["from"]]
                receiver = created_agents[interaction["to"]]
                
                message = MockMessage(
                    "user",
                    interaction["content"],
                    interaction["from"]
                )
                
                response = await receiver.reply(message)
                interaction_results.append({
                    "type": "message",
                    "from": interaction["from"],
                    "to": interaction["to"],
                    "message": message.to_dict(),
                    "response": response.to_dict()
                })
            
            elif interaction["type"] == "observe":
                observer = created_agents[interaction["agent"]]
                message = MockMessage(
                    "system",
                    interaction["content"],
                    "system"
                )
                
                await observer.observe(message)
                interaction_results.append({
                    "type": "observe",
                    "agent": interaction["agent"],
                    "message": message.to_dict()
                })
        
        # Save session
        await session.save_session_state(**created_agents)
        
        # Test restoration
        agent_classes = {"SessionableAgent": SessionableAgent}
        restored_agents = await session.restore_agents(agent_classes)
        
        return {
            "scenario_name": scenario_name,
            "session_id": scenario["session_id"],
            "agents_created": len(created_agents),
            "interactions_executed": len(interaction_results),
            "agents_restored": len(restored_agents),
            "session_file_exists": session.session_exists(),
            "session_info": session.get_session_info(),
            "interaction_results": interaction_results,
            "restoration_successful": len(restored_agents) == len(created_agents)
        }
    
    def cleanup(self):
        """Clean up temporary files and directories"""
        # Clean up sessions
        for session in self.sessions.values():
            session.cleanup()
        
        # Clean up temp directories
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Clear collections
        self.sessions.clear()
        self.temp_dirs.clear()
        self.test_agents.clear()
        self.session_scenarios.clear()


class SessionValidator:
    """Validator for session functionality"""
    
    @staticmethod
    def validate_session_file_structure(session_file: str) -> Dict[str, Any]:
        """Validate session file structure"""
        if not os.path.exists(session_file):
            return {"is_valid": False, "error": "Session file does not exist"}
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            required_keys = ["metadata", "agents"]
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                return {
                    "is_valid": False,
                    "error": f"Missing required keys: {missing_keys}"
                }
            
            # Validate metadata structure
            metadata = data["metadata"]
            required_metadata_keys = ["session_id", "created_at"]
            missing_metadata = [key for key in required_metadata_keys if key not in metadata]
            
            if missing_metadata:
                return {
                    "is_valid": False,
                    "error": f"Missing metadata keys: {missing_metadata}"
                }
            
            return {
                "is_valid": True,
                "agent_count": len(data["agents"]),
                "metadata": metadata,
                "file_size": os.path.getsize(session_file)
            }
        
        except json.JSONDecodeError as e:
            return {"is_valid": False, "error": f"Invalid JSON: {str(e)}"}
        except Exception as e:
            return {"is_valid": False, "error": f"Validation error: {str(e)}"}
    
    @staticmethod
    def validate_agent_state_restoration(
        original_agent: SessionableAgent,
        restored_agent: SessionableAgent
    ) -> Dict[str, Any]:
        """Validate agent state restoration"""
        validation_results = {}
        
        # Compare registered states
        for state_name in original_agent.registered_states:
            original_value = getattr(original_agent, state_name, None)
            restored_value = getattr(restored_agent, state_name, None)
            
            validation_results[state_name] = {
                "original": original_value,
                "restored": restored_value,
                "matches": original_value == restored_value
            }
        
        # Compare memory
        original_memory_size = original_agent.memory.size()
        restored_memory_size = restored_agent.memory.size()
        
        validation_results["memory"] = {
            "original_size": original_memory_size,
            "restored_size": restored_memory_size,
            "size_matches": original_memory_size == restored_memory_size
        }
        
        # Compare runtime stats
        validation_results["runtime_stats"] = {
            "reply_count_matches": original_agent.reply_count == restored_agent.reply_count,
            "observe_count_matches": original_agent.observe_count == restored_agent.observe_count
        }
        
        # Overall validation
        all_states_match = all(result.get("matches", False) for key, result in validation_results.items() 
                             if isinstance(result, dict) and "matches" in result)
        
        return {
            "is_valid": all_states_match,
            "detailed_results": validation_results,
            "mismatched_states": [state for state, result in validation_results.items() 
                                if isinstance(result, dict) and not result.get("matches", True)]
        }
    
    @staticmethod
    def validate_session_metadata(session: JSONSession, expected_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate session metadata"""
        actual_metadata = session.session_metadata
        validation_results = {}
        
        for key, expected_value in expected_metadata.items():
            actual_value = actual_metadata.get(key)
            validation_results[key] = {
                "expected": expected_value,
                "actual": actual_value,
                "matches": actual_value == expected_value
            }
        
        return {
            "is_valid": all(result["matches"] for result in validation_results.values()),
            "detailed_results": validation_results
        }


class SessionTest(IsolatedAsyncioTestCase):
    """Comprehensive session management testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = SessionTestFramework()
        self.validator = SessionValidator()
    
    async def test_basic_session_creation_and_saving(self):
        """Test basic session creation and saving"""
        session = self.framework.create_session("test_session_1")
        
        # Create test agents
        agent1 = self.framework.create_test_agent("Alice", "I am Alice, a helpful assistant.")
        agent2 = self.framework.create_test_agent("Bob", "I am Bob, a coding assistant.")
        
        # Add some interactions
        message = MockMessage("user", "Hello Alice!", "user")
        await agent1.reply(message)
        
        await agent2.observe(MockMessage("system", "Alice replied to user", "system"))
        
        # Save session
        await session.save_session_state(agent1=agent1, agent2=agent2)
        
        # Validate session file
        file_validation = self.validator.validate_session_file_structure(session.session_file)
        assert file_validation["is_valid"]
        assert file_validation["agent_count"] == 2
    
    async def test_session_state_restoration(self):
        """Test session state restoration"""
        session = self.framework.create_session("test_session_2")
        
        # Create and configure agent
        original_agent = self.framework.create_test_agent("TestAgent", "Original prompt")
        original_agent.register_state("custom_attribute")
        original_agent.custom_attribute = "test_value"
        
        # Add memory content
        await original_agent.memory.add(MockMessage("user", "Test message 1", "user"))
        await original_agent.memory.add(MockMessage("assistant", "Test response 1", "assistant"))
        
        # Update stats
        await original_agent.reply(MockMessage("user", "Test message 2", "user"))
        
        # Save session
        await session.save_session_state(test_agent=original_agent)
        
        # Restore session
        agent_classes = {"SessionableAgent": SessionableAgent}
        restored_agents = await session.restore_agents(agent_classes)
        
        assert "test_agent" in restored_agents
        restored_agent = restored_agents["test_agent"]
        
        # Validate restoration
        restoration_validation = self.validator.validate_agent_state_restoration(
            original_agent,
            restored_agent
        )
        
        assert restoration_validation["is_valid"]
        assert len(restoration_validation["mismatched_states"]) == 0
    
    async def test_multi_agent_session_coordination(self):
        """Test multi-agent session coordination"""
        session = self.framework.create_session("multi_agent_session")
        
        # Create multiple agents
        agents = {}
        for i in range(3):
            agent_name = f"Agent_{i}"
            agent = self.framework.create_test_agent(agent_name, f"I am agent number {i}")
            agents[agent_name] = agent
        
        # Simulate interactions between agents
        message = MockMessage("user", "Hello everyone!", "user")
        
        for agent_name, agent in agents.items():
            response = await agent.reply(message)
            # Let other agents observe the response
            for other_name, other_agent in agents.items():
                if other_name != agent_name:
                    await other_agent.observe(response)
        
        # Save session
        await session.save_session_state(**agents)
        
        # Validate session file
        file_validation = self.validator.validate_session_file_structure(session.session_file)
        assert file_validation["is_valid"]
        assert file_validation["agent_count"] == 3
        
        # Test restoration
        agent_classes = {"SessionableAgent": SessionableAgent}
        restored_agents = await session.restore_agents(agent_classes)
        
        assert len(restored_agents) == 3
        
        # Validate each agent's memory contains observations
        for agent_name, restored_agent in restored_agents.items():
            # Each agent should have replied once and observed twice
            assert restored_agent.reply_count == 1
            assert restored_agent.observe_count == 2
    
    async def test_session_metadata_validation(self):
        """Test session metadata validation"""
        session = self.framework.create_session("metadata_test_session")
        
        # Set custom metadata
        session.session_metadata.update({
            "test_key": "test_value",
            "version": "2.0"
        })
        
        agent = self.framework.create_test_agent("MetadataAgent")
        await session.save_session_state(metadata_agent=agent)
        
        # Validate metadata
        expected_metadata = {
            "session_id": "metadata_test_session",
            "test_key": "test_value",
            "version": "2.0"
        }
        
        metadata_validation = self.validator.validate_session_metadata(session, expected_metadata)
        assert metadata_validation["is_valid"]
    
    async def test_session_file_corruption_handling(self):
        """Test handling of corrupted session files"""
        session = self.framework.create_session("corruption_test")
        
        # Create valid session first
        agent = self.framework.create_test_agent("CorruptionAgent")
        await session.save_session_state(corruption_agent=agent)
        
        # Corrupt the file
        with open(session.session_file, 'w') as f:
            f.write("invalid json content {")
        
        # Try to validate corrupted file
        file_validation = self.validator.validate_session_file_structure(session.session_file)
        assert not file_validation["is_valid"]
        assert "Invalid JSON" in file_validation["error"]
        
        # Try to load corrupted session
        try:
            await session.load_session_state()
            assert False, "Should have raised exception for corrupted file"
        except json.JSONDecodeError:
            pass  # Expected
    
    async def test_session_scenario_framework(self):
        """Test session scenario framework"""
        # Create complex scenario
        await self.framework.create_session_scenario(
            "conversation_scenario",
            "conversation_session",
            [
                {"name": "Alice", "sys_prompt": "I am Alice, a helpful assistant."},
                {"name": "Bob", "sys_prompt": "I am Bob, a coding expert."}
            ],
            [
                {"type": "message", "from": "Alice", "to": "Bob", "content": "Can you help with Python?"},
                {"type": "message", "from": "Bob", "to": "Alice", "content": "Sure! What do you need?"},
                {"type": "observe", "agent": "Alice", "content": "Bob is ready to help"},
                {"type": "message", "from": "Alice", "to": "Bob", "content": "I need help with loops."}
            ]
        )
        
        # Run scenario
        result = await self.framework.run_session_scenario("conversation_scenario")
        
        assert result["agents_created"] == 2
        assert result["interactions_executed"] == 4
        assert result["restoration_successful"]
        assert result["session_file_exists"]
    
    async def test_memory_state_persistence(self):
        """Test memory state persistence across sessions"""
        session = self.framework.create_session("memory_persistence_test")
        
        agent = self.framework.create_test_agent("MemoryAgent")
        
        # Add various types of messages to memory
        messages = [
            MockMessage("user", "First user message", "user"),
            MockMessage("assistant", "First assistant response", "MemoryAgent"),
            MockMessage("system", "System notification", "system"),
            MockMessage("user", "Second user message", "user")
        ]
        
        for msg in messages:
            await agent.memory.add(msg)
        
        original_memory_size = agent.memory.size()
        original_messages = agent.memory.get_all()
        
        # Save and restore
        await session.save_session_state(memory_agent=agent)
        
        agent_classes = {"SessionableAgent": SessionableAgent}
        restored_agents = await session.restore_agents(agent_classes)
        restored_agent = restored_agents["memory_agent"]
        
        # Validate memory restoration
        assert restored_agent.memory.size() == original_memory_size
        restored_messages = restored_agent.memory.get_all()
        
        for i, (original, restored) in enumerate(zip(original_messages, restored_messages)):
            assert original.role == restored.role
            assert original.content == restored.content
            assert original.name == restored.name
    
    async def test_session_info_and_cleanup(self):
        """Test session information and cleanup"""
        session = self.framework.create_session("info_cleanup_test")
        
        agent = self.framework.create_test_agent("InfoAgent")
        await session.save_session_state(info_agent=agent)
        
        # Get session info
        info = session.get_session_info()
        
        assert info["session_id"] == "info_cleanup_test"
        assert info["file_exists"] == True
        assert info["current_agent_count"] == 1
        assert "file_size" in info
        assert "file_modified" in info
        
        # Test cleanup
        session.cleanup()
        
        # Verify file is gone
        assert not os.path.exists(session.session_file)
        
        # Update info after cleanup
        info_after_cleanup = session.get_session_info()
        assert info_after_cleanup["file_exists"] == False
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        self.framework.cleanup()


# Pytest integration
@pytest.fixture
def session_framework():
    """Pytest fixture for session framework"""
    framework = SessionTestFramework()
    yield framework
    framework.cleanup()


@pytest.fixture
def session_validator():
    """Pytest fixture for session validator"""
    return SessionValidator()


def test_session_framework_creation(session_framework):
    """Test session framework creation"""
    session = session_framework.create_session("test")
    assert session.session_id == "test"
    assert "test" in session_framework.sessions


def test_agent_creation(session_framework):
    """Test agent creation"""
    agent = session_framework.create_test_agent("TestAgent", "Test prompt")
    assert agent.name == "TestAgent"
    assert agent.sys_prompt == "Test prompt"
    assert "TestAgent" in session_framework.test_agents


@pytest.mark.asyncio
async def test_agent_state_management(session_framework):
    """Test agent state management"""
    agent = session_framework.create_test_agent("StateAgent")
    agent.register_state("custom_prop")
    agent.custom_prop = "test_value"
    
    # Get and set state
    state = agent.get_state()
    assert "custom_prop" in state
    assert state["custom_prop"] == "test_value"
    
    # Create new agent and restore state
    new_agent = session_framework.create_test_agent("NewAgent")
    new_agent.set_state(state)
    
    assert new_agent.custom_prop == "test_value"


def test_session_file_validation(session_validator):
    """Test session file validation"""
    # Test non-existent file
    result = session_validator.validate_session_file_structure("non_existent.json")
    assert not result["is_valid"]
    assert "does not exist" in result["error"]


def test_session_metadata_validation(session_validator):
    """Test session metadata validation"""
    # Create mock session
    session = MagicMock()
    session.session_metadata = {
        "session_id": "test",
        "version": "1.0"
    }
    
    expected = {"session_id": "test", "version": "1.0"}
    result = session_validator.validate_session_metadata(session, expected)
    
    assert result["is_valid"]


@pytest.mark.asyncio
async def test_simple_session_scenario(session_framework):
    """Test simple session scenario execution"""
    await session_framework.create_session_scenario(
        "simple_test",
        "simple_session",
        [{"name": "TestAgent", "sys_prompt": "I am a test agent"}],
        [{"type": "observe", "agent": "TestAgent", "content": "Test observation"}]
    )
    
    result = await session_framework.run_session_scenario("simple_test")
    
    assert result["agents_created"] == 1
    assert result["interactions_executed"] == 1
    assert result["restoration_successful"]