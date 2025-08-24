# -*- coding: utf-8 -*-
"""
AgentScope Hook System Testing Framework
========================================

Extracted from agentscope/tests/hook_test.py
Enhanced for TestMaster integration

Testing patterns for:
- Pre and post hook execution
- Async and sync hook compatibility
- Hook parameter modification and output modification
- Instance vs class hook registration
- Hook inheritance and isolation
- Multiple inheritance hook behavior
- Hook clearing and cleanup
- Agent lifecycle hook integration
"""

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Type
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock

import pytest


class MockTextBlock:
    """Mock text block content"""
    
    def __init__(self, text: str, block_type: str = "text"):
        self.type = block_type
        self.text = text
    
    def __eq__(self, other):
        """Equality comparison for testing"""
        if not isinstance(other, MockTextBlock):
            return False
        return self.type == other.type and self.text == other.text
    
    def __repr__(self):
        return f"MockTextBlock(type='{self.type}', text='{self.text}')"


class MockMessage:
    """Mock message for agent testing"""
    
    def __init__(self, role: str, content: Union[str, List], name: str, **kwargs):
        self.role = role
        self.content = content if isinstance(content, list) else [MockTextBlock(content)]
        self.name = name
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        return f"MockMessage(role='{self.role}', content={self.content}, name='{self.name}')"


class HookableAgent:
    """Base agent with hook system support"""
    
    def __init__(self, name: str = "agent"):
        self.name = name
        self.records = []
        self.memory = []
        self.instance_hooks = {
            "pre_reply": {},
            "post_reply": {},
            "pre_observe": {},
            "post_observe": {},
            "pre_print": {},
            "post_print": {}
        }
        # Class-level hooks (simulated)
        if not hasattr(self.__class__, '_class_hooks'):
            self.__class__._class_hooks = {
                "pre_reply": {},
                "post_reply": {},
                "pre_observe": {},
                "post_observe": {},
                "pre_print": {},
                "post_print": {}
            }
    
    def register_instance_hook(
        self,
        hook_type: str,
        hook_name: str,
        hook_function: Callable
    ):
        """Register instance-level hook"""
        if hook_type in self.instance_hooks:
            self.instance_hooks[hook_type][hook_name] = hook_function
    
    @classmethod
    def register_class_hook(
        cls,
        hook_type: str,
        hook_name: str,
        hook_function: Callable
    ):
        """Register class-level hook"""
        if not hasattr(cls, '_class_hooks'):
            cls._class_hooks = {
                "pre_reply": {},
                "post_reply": {},
                "pre_observe": {},
                "post_observe": {},
                "pre_print": {},
                "post_print": {}
            }
        
        if hook_type in cls._class_hooks:
            cls._class_hooks[hook_type][hook_name] = hook_function
    
    def clear_instance_hooks(self):
        """Clear all instance hooks"""
        for hook_type in self.instance_hooks:
            self.instance_hooks[hook_type].clear()
    
    @classmethod
    def clear_class_hooks(cls):
        """Clear all class hooks"""
        if hasattr(cls, '_class_hooks'):
            for hook_type in cls._class_hooks:
                cls._class_hooks[hook_type].clear()
    
    @classmethod
    def remove_class_hook(cls, hook_type: str, hook_name: str):
        """Remove specific class hook"""
        if hasattr(cls, '_class_hooks') and hook_type in cls._class_hooks:
            cls._class_hooks[hook_type].pop(hook_name, None)
    
    async def execute_hooks(
        self,
        hook_type: str,
        kwargs: Dict[str, Any],
        output: Any = None
    ) -> tuple[Dict[str, Any], Any]:
        """Execute hooks of given type"""
        current_kwargs = kwargs.copy()
        current_output = output
        
        # Execute class hooks first (if any)
        if hasattr(self.__class__, '_class_hooks') and hook_type in self.__class__._class_hooks:
            for hook_name, hook_func in self.__class__._class_hooks[hook_type].items():
                if hook_type.startswith("pre_"):
                    # Pre-hooks can modify kwargs
                    if asyncio.iscoroutinefunction(hook_func):
                        result = await hook_func(self, current_kwargs)
                    else:
                        result = hook_func(self, current_kwargs)
                    
                    if result is not None:
                        current_kwargs = result
                else:
                    # Post-hooks can modify output
                    if asyncio.iscoroutinefunction(hook_func):
                        result = await hook_func(self, current_kwargs, current_output)
                    else:
                        result = hook_func(self, current_kwargs, current_output)
                    
                    if result is not None:
                        current_output = result
        
        # Execute instance hooks
        if hook_type in self.instance_hooks:
            for hook_name, hook_func in self.instance_hooks[hook_type].items():
                if hook_type.startswith("pre_"):
                    # Pre-hooks can modify kwargs
                    if asyncio.iscoroutinefunction(hook_func):
                        result = await hook_func(self, current_kwargs)
                    else:
                        result = hook_func(self, current_kwargs)
                    
                    if result is not None:
                        current_kwargs = result
                else:
                    # Post-hooks can modify output
                    if asyncio.iscoroutinefunction(hook_func):
                        result = await hook_func(self, current_kwargs, current_output)
                    else:
                        result = hook_func(self, current_kwargs, current_output)
                    
                    if result is not None:
                        current_output = result
        
        return current_kwargs, current_output
    
    async def __call__(self, message: MockMessage) -> MockMessage:
        """Main agent call with hook support"""
        return await self.reply(message)
    
    async def reply(self, message: MockMessage) -> MockMessage:
        """Reply with pre and post hook support"""
        # Execute pre-reply hooks
        kwargs = {"msg": message}
        kwargs, _ = await self.execute_hooks("pre_reply", kwargs)
        modified_message = kwargs["msg"]
        
        # Core reply logic
        await self.print(modified_message)
        result = self._core_reply(modified_message)
        
        # Execute post-reply hooks
        _, modified_result = await self.execute_hooks("post_reply", kwargs, result)
        
        return modified_result if modified_result is not None else result
    
    def _core_reply(self, message: MockMessage) -> MockMessage:
        """Core reply logic (can be overridden)"""
        # Default behavior: add a "mark" text block
        result_content = message.content.copy()
        result_content.append(MockTextBlock("mark"))
        
        return MockMessage(
            role="assistant",
            content=result_content,
            name=self.name
        )
    
    async def print(self, message: MockMessage):
        """Print with hook support"""
        kwargs = {"msg": message}
        await self.execute_hooks("pre_print", kwargs)
    
    async def observe(self, message: MockMessage):
        """Observe with hook support"""
        # Execute pre-observe hooks
        kwargs = {"msg": message}
        kwargs, _ = await self.execute_hooks("pre_observe", kwargs)
        modified_message = kwargs["msg"]
        
        # Core observe logic
        self.memory.append(modified_message)
        
        # Execute post-observe hooks
        await self.execute_hooks("post_observe", kwargs, None)


class TestChildAgent(HookableAgent):
    """Child agent for testing inheritance"""
    pass


class TestGrandChildAgent(TestChildAgent):
    """Grandchild agent for testing deeper inheritance"""
    pass


class TestAgentA(HookableAgent):
    """First parent class for multiple inheritance testing"""
    pass


class TestAgentB(HookableAgent):
    """Second parent class for multiple inheritance testing"""
    pass


class TestAgentC(TestAgentA, TestAgentB):
    """Multiple inheritance class for testing"""
    pass


class HookTestFramework:
    """Core framework for hook system testing"""
    
    def __init__(self):
        self.agents = {}
        self.hook_functions = {}
        self.execution_records = []
    
    def create_agent(self, name: str, agent_class: Type[HookableAgent] = HookableAgent) -> HookableAgent:
        """Create agent instance"""
        agent = agent_class(name)
        self.agents[name] = agent
        return agent
    
    def create_pre_hook(
        self,
        name: str,
        modifies_content: bool = False,
        is_async: bool = False,
        content_to_add: str = None
    ) -> Callable:
        """Create pre-hook function"""
        content_text = content_to_add or name
        
        if is_async:
            async def async_pre_hook(agent: HookableAgent, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                # Record execution
                agent.records.append(name)
                
                # Modify message content if requested
                if "msg" in kwargs and isinstance(kwargs["msg"], MockMessage):
                    kwargs["msg"].content.append(MockTextBlock(content_text))
                
                return kwargs if modifies_content else None
            
            self.hook_functions[name] = async_pre_hook
            return async_pre_hook
        else:
            def sync_pre_hook(agent: HookableAgent, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                # Record execution
                agent.records.append(name)
                
                # Modify message content if requested
                if "msg" in kwargs and isinstance(kwargs["msg"], MockMessage):
                    kwargs["msg"].content.append(MockTextBlock(content_text))
                
                return kwargs if modifies_content else None
            
            self.hook_functions[name] = sync_pre_hook
            return sync_pre_hook
    
    def create_post_hook(
        self,
        name: str,
        modifies_output: bool = False,
        is_async: bool = False,
        content_to_add: str = None
    ) -> Callable:
        """Create post-hook function"""
        content_text = content_to_add or name
        
        if is_async:
            async def async_post_hook(
                agent: HookableAgent,
                kwargs: Dict[str, Any],
                output: Any
            ) -> Optional[Any]:
                # Record execution
                agent.records.append(name)
                
                # Modify output if requested
                if modifies_output and isinstance(output, MockMessage):
                    output.content.append(MockTextBlock(content_text))
                    return output
                elif not modifies_output and isinstance(output, MockMessage):
                    # Still add content but don't return modified output
                    output.content.append(MockTextBlock(content_text))
                
                return None
            
            self.hook_functions[name] = async_post_hook
            return async_post_hook
        else:
            def sync_post_hook(
                agent: HookableAgent,
                kwargs: Dict[str, Any],
                output: Any
            ) -> Optional[Any]:
                # Record execution
                agent.records.append(name)
                
                # Modify output if requested
                if modifies_output and isinstance(output, MockMessage):
                    output.content.append(MockTextBlock(content_text))
                    return output
                elif not modifies_output and isinstance(output, MockMessage):
                    # Still add content but don't return modified output
                    output.content.append(MockTextBlock(content_text))
                
                return None
            
            self.hook_functions[name] = sync_post_hook
            return sync_post_hook
    
    def register_multiple_hooks(
        self,
        agent: HookableAgent,
        hook_configs: List[Dict[str, Any]]
    ):
        """Register multiple hooks from configuration"""
        for config in hook_configs:
            hook_type = config["type"]
            hook_name = config["name"]
            is_pre = hook_type.startswith("pre_")
            
            if is_pre:
                hook_func = self.create_pre_hook(
                    hook_name,
                    config.get("modifies_content", False),
                    config.get("is_async", False),
                    config.get("content", None)
                )
            else:
                hook_func = self.create_post_hook(
                    hook_name,
                    config.get("modifies_output", False),
                    config.get("is_async", False),
                    config.get("content", None)
                )
            
            agent.register_instance_hook(hook_type, hook_name, hook_func)
    
    async def run_hook_sequence_test(
        self,
        agent_name: str,
        hook_sequence: List[str],
        test_message: MockMessage
    ) -> Dict[str, Any]:
        """Run test with specific hook execution sequence"""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        # Clear previous records
        agent.records.clear()
        agent.memory.clear()
        
        # Execute agent
        start_time = asyncio.get_event_loop().time()
        result = await agent(test_message)
        end_time = asyncio.get_event_loop().time()
        
        # Validate hook execution order
        execution_order_correct = agent.records == hook_sequence
        
        return {
            "agent_name": agent_name,
            "expected_sequence": hook_sequence,
            "actual_sequence": agent.records,
            "execution_order_correct": execution_order_correct,
            "result_message": result,
            "execution_time": end_time - start_time,
            "memory_size": len(agent.memory)
        }


class HookValidator:
    """Validator for hook system behavior"""
    
    @staticmethod
    def validate_hook_execution_order(
        expected_order: List[str],
        actual_order: List[str]
    ) -> Dict[str, Any]:
        """Validate hook execution order"""
        is_valid = expected_order == actual_order
        
        # Find mismatches
        mismatches = []
        max_len = max(len(expected_order), len(actual_order))
        
        for i in range(max_len):
            expected = expected_order[i] if i < len(expected_order) else None
            actual = actual_order[i] if i < len(actual_order) else None
            
            if expected != actual:
                mismatches.append({
                    "position": i,
                    "expected": expected,
                    "actual": actual
                })
        
        return {
            "validator": "hook_execution_order",
            "expected_order": expected_order,
            "actual_order": actual_order,
            "is_valid": is_valid,
            "mismatches": mismatches,
            "total_expected": len(expected_order),
            "total_actual": len(actual_order)
        }
    
    @staticmethod
    def validate_message_content(
        message: MockMessage,
        expected_texts: List[str]
    ) -> Dict[str, Any]:
        """Validate message content contains expected texts"""
        actual_texts = [block.text for block in message.content if hasattr(block, 'text')]
        
        missing_texts = [text for text in expected_texts if text not in actual_texts]
        extra_texts = [text for text in actual_texts if text not in expected_texts]
        
        return {
            "validator": "message_content",
            "expected_texts": expected_texts,
            "actual_texts": actual_texts,
            "missing_texts": missing_texts,
            "extra_texts": extra_texts,
            "is_valid": len(missing_texts) == 0 and len(extra_texts) == 0
        }
    
    @staticmethod
    def validate_hook_registration(
        agent: HookableAgent,
        expected_hooks: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Validate hook registration"""
        validation_results = {}
        
        for hook_type, expected_hook_names in expected_hooks.items():
            actual_hook_names = list(agent.instance_hooks.get(hook_type, {}).keys())
            
            missing_hooks = [name for name in expected_hook_names if name not in actual_hook_names]
            extra_hooks = [name for name in actual_hook_names if name not in expected_hook_names]
            
            validation_results[hook_type] = {
                "expected_hooks": expected_hook_names,
                "actual_hooks": actual_hook_names,
                "missing_hooks": missing_hooks,
                "extra_hooks": extra_hooks,
                "is_valid": len(missing_hooks) == 0 and len(extra_hooks) == 0
            }
        
        return {
            "validator": "hook_registration",
            "by_hook_type": validation_results,
            "is_valid": all(result["is_valid"] for result in validation_results.values())
        }


class HookTest(IsolatedAsyncioTestCase):
    """Comprehensive hook system testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = HookTestFramework()
        self.validator = HookValidator()
        
        # Create default agent
        self.agent = self.framework.create_agent("test_agent")
        
        # Create test message
        self.test_message = MockMessage("user", [MockTextBlock("0")], "user")
    
    def get_test_message(self) -> MockMessage:
        """Get fresh test message"""
        return MockMessage("user", [MockTextBlock("0")], "user")
    
    async def test_basic_hook_execution(self):
        """Test basic pre and post hook execution"""
        # No hooks - should just add "mark"
        result = await self.agent(self.get_test_message())
        expected_content = [MockTextBlock("0"), MockTextBlock("mark")]
        
        content_validation = self.validator.validate_message_content(
            result,
            ["0", "mark"]
        )
        assert content_validation["is_valid"]
    
    async def test_pre_hook_with_modification(self):
        """Test pre-hook that modifies parameters"""
        # Register async pre-hook that modifies content
        pre_hook = self.framework.create_pre_hook("pre_1", modifies_content=True, is_async=True)
        self.agent.register_instance_hook("pre_reply", "pre_1", pre_hook)
        
        result = await self.agent(self.get_test_message())
        
        # Should have: original "0" + "pre_1" + "mark"
        content_validation = self.validator.validate_message_content(
            result,
            ["0", "pre_1", "mark"]
        )
        assert content_validation["is_valid"]
        
        # Check execution record
        assert self.agent.records == ["pre_1"]
    
    async def test_multiple_pre_hooks(self):
        """Test multiple pre-hooks execution order"""
        # Register multiple hooks
        hook_configs = [
            {"type": "pre_reply", "name": "pre_1", "modifies_content": True, "is_async": True},
            {"type": "pre_reply", "name": "pre_2", "modifies_content": False, "is_async": True},
            {"type": "pre_reply", "name": "pre_3", "modifies_content": True, "is_async": False},
            {"type": "pre_reply", "name": "pre_4", "modifies_content": False, "is_async": False}
        ]
        
        self.framework.register_multiple_hooks(self.agent, hook_configs)
        
        result = await self.agent(self.get_test_message())
        
        # Validate execution order
        expected_order = ["pre_1", "pre_2", "pre_3", "pre_4"]
        order_validation = self.validator.validate_hook_execution_order(
            expected_order,
            self.agent.records
        )
        assert order_validation["is_valid"]
        
        # Content should have modifications from hooks that modify content
        content_validation = self.validator.validate_message_content(
            result,
            ["0", "pre_1", "pre_3", "mark"]
        )
        assert content_validation["is_valid"]
    
    async def test_post_hook_execution(self):
        """Test post-hook execution and output modification"""
        # Register post-hooks
        hook_configs = [
            {"type": "post_reply", "name": "post_1", "modifies_output": True, "is_async": True},
            {"type": "post_reply", "name": "post_2", "modifies_output": False, "is_async": True},
            {"type": "post_reply", "name": "post_3", "modifies_output": True, "is_async": False},
            {"type": "post_reply", "name": "post_4", "modifies_output": False, "is_async": False}
        ]
        
        self.framework.register_multiple_hooks(self.agent, hook_configs)
        
        result = await self.agent(self.get_test_message())
        
        # Validate execution order
        expected_order = ["post_1", "post_2", "post_3", "post_4"]
        order_validation = self.validator.validate_hook_execution_order(
            expected_order,
            self.agent.records
        )
        assert order_validation["is_valid"]
        
        # Content should have modifications from hooks that modify output
        content_validation = self.validator.validate_message_content(
            result,
            ["0", "mark", "post_1", "post_3"]
        )
        assert content_validation["is_valid"]
    
    async def test_combined_pre_and_post_hooks(self):
        """Test combination of pre and post hooks"""
        hook_configs = [
            {"type": "pre_reply", "name": "pre_1", "modifies_content": True, "is_async": True},
            {"type": "pre_reply", "name": "pre_2", "modifies_content": False, "is_async": True},
            {"type": "post_reply", "name": "post_1", "modifies_output": True, "is_async": True},
            {"type": "post_reply", "name": "post_2", "modifies_output": False, "is_async": False}
        ]
        
        self.framework.register_multiple_hooks(self.agent, hook_configs)
        
        result = await self.agent(self.get_test_message())
        
        # Validate execution order (pre hooks first, then post hooks)
        expected_order = ["pre_1", "pre_2", "post_1", "post_2"]
        order_validation = self.validator.validate_hook_execution_order(
            expected_order,
            self.agent.records
        )
        assert order_validation["is_valid"]
        
        # Content should include modifications from both pre and post hooks
        content_validation = self.validator.validate_message_content(
            result,
            ["0", "pre_1", "mark", "post_1"]
        )
        assert content_validation["is_valid"]
    
    async def test_print_hooks(self):
        """Test print hook execution"""
        hook_configs = [
            {"type": "pre_print", "name": "pre_1", "modifies_content": True, "is_async": True},
            {"type": "pre_print", "name": "pre_2", "modifies_content": False, "is_async": True},
            {"type": "pre_print", "name": "pre_3", "modifies_content": True, "is_async": False},
            {"type": "pre_print", "name": "pre_4", "modifies_content": False, "is_async": False}
        ]
        
        self.framework.register_multiple_hooks(self.agent, hook_configs)
        
        await self.agent(self.get_test_message())
        
        # Validate print hook execution
        expected_order = ["pre_1", "pre_2", "pre_3", "pre_4"]
        order_validation = self.validator.validate_hook_execution_order(
            expected_order,
            self.agent.records
        )
        assert order_validation["is_valid"]
    
    async def test_observe_hooks(self):
        """Test observe hook execution"""
        hook_configs = [
            {"type": "pre_observe", "name": "pre_1", "modifies_content": True, "is_async": True},
            {"type": "pre_observe", "name": "pre_2", "modifies_content": False, "is_async": True},
            {"type": "post_observe", "name": "post_1", "modifies_output": True, "is_async": True},
            {"type": "post_observe", "name": "post_2", "modifies_output": False, "is_async": True}
        ]
        
        self.framework.register_multiple_hooks(self.agent, hook_configs)
        
        await self.agent.observe(self.get_test_message())
        
        # Validate hook execution
        expected_order = ["pre_1", "pre_2", "post_1", "post_2"]
        order_validation = self.validator.validate_hook_execution_order(
            expected_order,
            self.agent.records
        )
        assert order_validation["is_valid"]
        
        # Check memory was populated
        assert len(self.agent.memory) == 1
        
        # Check message content modification
        observed_message = self.agent.memory[0]
        content_validation = self.validator.validate_message_content(
            observed_message,
            ["0", "pre_1"]  # Only pre_1 modifies content
        )
        assert content_validation["is_valid"]
    
    async def test_hook_clearing(self):
        """Test hook clearing functionality"""
        # Register some hooks
        hook_configs = [
            {"type": "pre_reply", "name": "pre_1", "modifies_content": True, "is_async": True},
            {"type": "post_reply", "name": "post_1", "modifies_output": True, "is_async": True}
        ]
        
        self.framework.register_multiple_hooks(self.agent, hook_configs)
        
        # Verify hooks are registered
        registration_validation = self.validator.validate_hook_registration(
            self.agent,
            {"pre_reply": ["pre_1"], "post_reply": ["post_1"]}
        )
        assert registration_validation["is_valid"]
        
        # Clear hooks
        self.agent.clear_instance_hooks()
        
        # Verify hooks are cleared
        registration_validation = self.validator.validate_hook_registration(
            self.agent,
            {"pre_reply": [], "post_reply": []}
        )
        assert registration_validation["is_valid"]
        
        # Test execution without hooks
        self.agent.records.clear()
        result = await self.agent(self.get_test_message())
        
        # Should have no hook executions
        assert self.agent.records == []
        
        # Should have only basic content
        content_validation = self.validator.validate_message_content(
            result,
            ["0", "mark"]
        )
        assert content_validation["is_valid"]
    
    async def test_class_hook_system(self):
        """Test class-level hook registration and execution"""
        # Create child agent class instance
        child_agent = self.framework.create_agent("child_agent", TestChildAgent)
        
        # Register class-level hook
        pre_hook = self.framework.create_pre_hook("class_pre", modifies_content=True, is_async=False)
        TestChildAgent.register_class_hook("pre_reply", "class_pre", pre_hook)
        
        # Test class hook execution
        result = await child_agent(self.get_test_message())
        
        # Should execute class hook
        assert "class_pre" in child_agent.records
        
        # Content should be modified
        content_validation = self.validator.validate_message_content(
            result,
            ["0", "class_pre", "mark"]
        )
        assert content_validation["is_valid"]
        
        # Clean up
        TestChildAgent.clear_class_hooks()
    
    async def test_hook_inheritance_isolation(self):
        """Test hook inheritance and isolation between classes"""
        # Create agents of different classes
        parent_agent = self.framework.create_agent("parent", HookableAgent)
        child_agent = self.framework.create_agent("child", TestChildAgent)
        grandchild_agent = self.framework.create_agent("grandchild", TestGrandChildAgent)
        
        # Register different hooks on different classes
        parent_hook = self.framework.create_pre_hook("parent_hook", modifies_content=True, is_async=False)
        child_hook = self.framework.create_pre_hook("child_hook", modifies_content=True, is_async=True)
        grandchild_hook = self.framework.create_pre_hook("grandchild_hook", modifies_content=False, is_async=False)
        
        HookableAgent.register_class_hook("pre_reply", "parent_hook", parent_hook)
        TestChildAgent.register_class_hook("pre_reply", "child_hook", child_hook)
        TestGrandChildAgent.register_class_hook("pre_reply", "grandchild_hook", grandchild_hook)
        
        # Test parent agent
        result = await parent_agent(self.get_test_message())
        assert "parent_hook" in parent_agent.records
        
        # Test child agent
        result = await child_agent(self.get_test_message())
        assert "child_hook" in child_agent.records
        assert "parent_hook" not in child_agent.records  # Should be isolated
        
        # Test grandchild agent
        result = await grandchild_agent(self.get_test_message())
        assert "grandchild_hook" in grandchild_agent.records
        assert "child_hook" not in grandchild_agent.records  # Should be isolated
        assert "parent_hook" not in grandchild_agent.records  # Should be isolated
        
        # Clean up
        HookableAgent.clear_class_hooks()
        TestChildAgent.clear_class_hooks()
        TestGrandChildAgent.clear_class_hooks()
    
    async def test_multiple_inheritance_hooks(self):
        """Test hook behavior with multiple inheritance"""
        # Create agents with multiple inheritance
        agent_a = self.framework.create_agent("agent_a", TestAgentA)
        agent_b = self.framework.create_agent("agent_b", TestAgentB)
        agent_c = self.framework.create_agent("agent_c", TestAgentC)
        
        # Register hooks on different parent classes
        hook_a = self.framework.create_pre_hook("hook_a", modifies_content=True, is_async=False)
        hook_b = self.framework.create_pre_hook("hook_b", modifies_content=True, is_async=True)
        hook_c = self.framework.create_pre_hook("hook_c", modifies_content=False, is_async=False)
        
        TestAgentA.register_class_hook("pre_reply", "hook_a", hook_a)
        TestAgentB.register_class_hook("pre_reply", "hook_b", hook_b)
        TestAgentC.register_class_hook("pre_reply", "hook_c", hook_c)
        
        # Test each agent executes only its own hooks
        result_a = await agent_a(self.get_test_message())
        assert "hook_a" in agent_a.records
        assert "hook_b" not in agent_a.records
        assert "hook_c" not in agent_a.records
        
        result_b = await agent_b(self.get_test_message())
        assert "hook_b" in agent_b.records
        assert "hook_a" not in agent_b.records
        assert "hook_c" not in agent_b.records
        
        result_c = await agent_c(self.get_test_message())
        assert "hook_c" in agent_c.records
        assert "hook_a" not in agent_c.records
        assert "hook_b" not in agent_c.records
        
        # Clean up
        TestAgentA.clear_class_hooks()
        TestAgentB.clear_class_hooks()
        TestAgentC.clear_class_hooks()
    
    async def test_framework_sequence_testing(self):
        """Test framework's sequence testing capabilities"""
        # Set up complex hook sequence
        hook_configs = [
            {"type": "pre_reply", "name": "pre_1", "modifies_content": True, "is_async": True},
            {"type": "pre_reply", "name": "pre_2", "modifies_content": False, "is_async": False},
            {"type": "post_reply", "name": "post_1", "modifies_output": True, "is_async": True},
            {"type": "post_reply", "name": "post_2", "modifies_output": False, "is_async": False}
        ]
        
        self.framework.register_multiple_hooks(self.agent, hook_configs)
        
        expected_sequence = ["pre_1", "pre_2", "post_1", "post_2"]
        
        test_result = await self.framework.run_hook_sequence_test(
            "test_agent",
            expected_sequence,
            self.get_test_message()
        )
        
        assert test_result["execution_order_correct"]
        assert test_result["expected_sequence"] == expected_sequence
        assert test_result["actual_sequence"] == expected_sequence
        assert "execution_time" in test_result
        assert test_result["execution_time"] >= 0
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        # Clear all instance hooks
        for agent in self.framework.agents.values():
            agent.clear_instance_hooks()
        
        # Clear all class hooks
        HookableAgent.clear_class_hooks()
        TestChildAgent.clear_class_hooks()
        TestGrandChildAgent.clear_class_hooks()
        TestAgentA.clear_class_hooks()
        TestAgentB.clear_class_hooks()
        TestAgentC.clear_class_hooks()


# Pytest integration
@pytest.fixture
def hook_framework():
    """Pytest fixture for hook framework"""
    return HookTestFramework()


@pytest.fixture
def hook_validator():
    """Pytest fixture for hook validator"""
    return HookValidator()


def test_hook_framework_creation(hook_framework):
    """Test hook framework creation"""
    agent = hook_framework.create_agent("test")
    assert agent.name == "test"
    assert "test" in hook_framework.agents


def test_pre_hook_creation(hook_framework):
    """Test pre-hook creation"""
    hook = hook_framework.create_pre_hook("test_hook", modifies_content=True, is_async=True)
    assert callable(hook)
    assert asyncio.iscoroutinefunction(hook)


def test_post_hook_creation(hook_framework):
    """Test post-hook creation"""
    hook = hook_framework.create_post_hook("test_hook", modifies_output=True, is_async=False)
    assert callable(hook)
    assert not asyncio.iscoroutinefunction(hook)


def test_hook_execution_order_validation(hook_validator):
    """Test hook execution order validation"""
    expected = ["hook1", "hook2", "hook3"]
    actual = ["hook1", "hook2", "hook3"]
    
    validation = hook_validator.validate_hook_execution_order(expected, actual)
    assert validation["is_valid"]
    assert len(validation["mismatches"]) == 0


def test_message_content_validation(hook_validator):
    """Test message content validation"""
    message = MockMessage("user", [MockTextBlock("test1"), MockTextBlock("test2")], "user")
    expected_texts = ["test1", "test2"]
    
    validation = hook_validator.validate_message_content(message, expected_texts)
    assert validation["is_valid"]
    assert len(validation["missing_texts"]) == 0
    assert len(validation["extra_texts"]) == 0


@pytest.mark.asyncio
async def test_simple_hook_execution(hook_framework):
    """Test simple hook execution"""
    agent = hook_framework.create_agent("test")
    hook = hook_framework.create_pre_hook("test_hook", modifies_content=True, is_async=True)
    
    agent.register_instance_hook("pre_reply", "test_hook", hook)
    
    message = MockMessage("user", [MockTextBlock("test")], "user")
    result = await agent.reply(message)
    
    assert "test_hook" in agent.records
    assert len(result.content) >= 2  # Original + hook modification