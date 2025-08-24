# -*- coding: utf-8 -*-
"""
AgentScope Pipeline Testing Framework
====================================

Extracted from agentscope/tests/pipeline_test.py
Enhanced for TestMaster integration

Testing patterns for:
- Sequential pipeline execution
- Functional vs class-based pipelines
- Agent chaining and composition
- Message flow and metadata handling
- Pipeline state management
- Error propagation in pipelines
- Async pipeline execution
- Custom pipeline operators
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock

import pytest


class MockMessage:
    """Mock message for pipeline testing"""
    
    def __init__(self, role: str, content: str, name: str, metadata: Optional[Dict] = None):
        self.role = role
        self.content = content
        self.name = name
        self.metadata = metadata or {}
    
    def copy(self) -> 'MockMessage':
        """Create a copy of the message"""
        return MockMessage(
            self.role,
            self.content,
            self.name,
            self.metadata.copy()
        )


class MockAgent:
    """Mock agent for pipeline testing"""
    
    def __init__(self, name: str, behavior: Optional[Callable] = None):
        self.name = name
        self.behavior = behavior or self.default_behavior
        self.call_count = 0
        self.received_messages = []
        self.observation_count = 0
        self.interrupt_count = 0
    
    async def reply(self, message: MockMessage) -> MockMessage:
        """Reply to a message"""
        self.call_count += 1
        self.received_messages.append(message.copy())
        
        if asyncio.iscoroutinefunction(self.behavior):
            result = await self.behavior(message)
        else:
            result = self.behavior(message)
        
        return result
    
    async def observe(self, msg: Union[MockMessage, List[MockMessage], None]) -> None:
        """Observe messages"""
        self.observation_count += 1
    
    async def handle_interrupt(self, *args: Any, **kwargs: Any) -> MockMessage:
        """Handle interruption"""
        self.interrupt_count += 1
        return MockMessage("system", "interrupted", "system")
    
    def default_behavior(self, message: MockMessage) -> MockMessage:
        """Default agent behavior"""
        return MockMessage(
            "assistant",
            f"{self.name} processed: {message.content}",
            self.name,
            message.metadata.copy()
        )
    
    def reset_stats(self):
        """Reset agent statistics"""
        self.call_count = 0
        self.received_messages.clear()
        self.observation_count = 0
        self.interrupt_count = 0


class ArithmeticAgent(MockAgent):
    """Agent that performs arithmetic operations"""
    
    def __init__(self, name: str, operation: str, value: Union[int, float]):
        super().__init__(name)
        self.operation = operation
        self.value = value
    
    def default_behavior(self, message: MockMessage) -> MockMessage:
        """Perform arithmetic operation"""
        current_result = message.metadata.get("result", 0)
        
        if self.operation == "add":
            new_result = current_result + self.value
        elif self.operation == "multiply":
            new_result = current_result * self.value
        elif self.operation == "subtract":
            new_result = current_result - self.value
        elif self.operation == "divide":
            new_result = current_result / self.value if self.value != 0 else current_result
        else:
            new_result = current_result
        
        result_msg = message.copy()
        result_msg.metadata["result"] = new_result
        result_msg.metadata["last_operation"] = f"{self.operation}({self.value})"
        
        return result_msg


class ConditionalAgent(MockAgent):
    """Agent with conditional behavior"""
    
    def __init__(self, name: str, condition: Callable, true_behavior: Callable, false_behavior: Callable):
        super().__init__(name)
        self.condition = condition
        self.true_behavior = true_behavior
        self.false_behavior = false_behavior
    
    def default_behavior(self, message: MockMessage) -> MockMessage:
        """Execute conditional behavior"""
        if self.condition(message):
            return self.true_behavior(message)
        else:
            return self.false_behavior(message)


class SequentialPipeline:
    """Sequential pipeline implementation"""
    
    def __init__(self, agents: List[MockAgent]):
        self.agents = agents
        self.execution_log = []
        self.error_handlers = {}
    
    async def __call__(self, message: MockMessage) -> MockMessage:
        """Execute pipeline"""
        return await self.execute(message)
    
    async def execute(self, message: MockMessage) -> MockMessage:
        """Execute agents sequentially"""
        current_message = message.copy()
        self.execution_log.clear()
        
        for i, agent in enumerate(self.agents):
            try:
                self.execution_log.append({
                    "step": i,
                    "agent": agent.name,
                    "input": current_message.copy(),
                    "status": "executing"
                })
                
                result = await agent.reply(current_message)
                current_message = result
                
                self.execution_log[-1]["output"] = result.copy()
                self.execution_log[-1]["status"] = "completed"
                
            except Exception as e:
                self.execution_log[-1]["error"] = str(e)
                self.execution_log[-1]["status"] = "error"
                
                # Handle error
                if agent.name in self.error_handlers:
                    current_message = await self.error_handlers[agent.name](current_message, e)
                else:
                    raise e
        
        return current_message
    
    def add_error_handler(self, agent_name: str, handler: Callable):
        """Add error handler for specific agent"""
        self.error_handlers[agent_name] = handler
    
    def get_execution_summary(self) -> Dict:
        """Get execution summary"""
        total_steps = len(self.execution_log)
        completed_steps = sum(1 for step in self.execution_log if step["status"] == "completed")
        error_steps = sum(1 for step in self.execution_log if step["status"] == "error")
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "error_steps": error_steps,
            "success_rate": completed_steps / total_steps if total_steps > 0 else 0,
            "execution_time": self._calculate_execution_time()
        }
    
    def _calculate_execution_time(self) -> float:
        """Calculate mock execution time"""
        return len(self.execution_log) * 0.1  # Mock timing


class ParallelPipeline:
    """Parallel pipeline implementation"""
    
    def __init__(self, agents: List[MockAgent]):
        self.agents = agents
        self.execution_log = []
    
    async def execute(self, message: MockMessage) -> List[MockMessage]:
        """Execute agents in parallel"""
        self.execution_log.clear()
        
        tasks = []
        for i, agent in enumerate(self.agents):
            task = self._execute_agent(i, agent, message.copy())
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.execution_log.append({
                    "step": i,
                    "agent": self.agents[i].name,
                    "status": "error",
                    "error": str(result)
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def _execute_agent(self, step: int, agent: MockAgent, message: MockMessage) -> MockMessage:
        """Execute single agent"""
        try:
            self.execution_log.append({
                "step": step,
                "agent": agent.name,
                "status": "executing"
            })
            
            result = await agent.reply(message)
            
            self.execution_log[-1]["status"] = "completed"
            return result
            
        except Exception as e:
            self.execution_log[-1]["status"] = "error"
            self.execution_log[-1]["error"] = str(e)
            raise e


class PipelineTestFramework:
    """Core framework for pipeline testing"""
    
    def __init__(self):
        self.pipelines = {}
        self.agents = {}
        self.test_messages = {}
        self.validators = []
    
    def create_arithmetic_pipeline(self, operations: List[tuple]) -> SequentialPipeline:
        """Create arithmetic pipeline from operations"""
        agents = []
        for i, (op, value) in enumerate(operations):
            agent = ArithmeticAgent(f"{op}_{i}", op, value)
            agents.append(agent)
        
        return SequentialPipeline(agents)
    
    def create_test_message(self, name: str, initial_value: int = 0) -> MockMessage:
        """Create test message with initial value"""
        msg = MockMessage("user", "", "user", {"result": initial_value})
        self.test_messages[name] = msg
        return msg
    
    def register_pipeline(self, name: str, pipeline: Union[SequentialPipeline, ParallelPipeline]):
        """Register pipeline for testing"""
        self.pipelines[name] = pipeline
    
    def register_agent(self, name: str, agent: MockAgent):
        """Register agent for testing"""
        self.agents[name] = agent
    
    def add_validator(self, validator: Callable):
        """Add result validator"""
        self.validators.append(validator)
    
    async def run_pipeline_test(self, pipeline_name: str, message_name: str) -> Dict:
        """Run pipeline test and return results"""
        pipeline = self.pipelines[pipeline_name]
        message = self.test_messages[message_name].copy()
        
        start_time = asyncio.get_event_loop().time()
        result = await pipeline.execute(message)
        end_time = asyncio.get_event_loop().time()
        
        test_result = {
            "pipeline": pipeline_name,
            "input_message": message,
            "output_message": result,
            "execution_time": end_time - start_time,
            "pipeline_summary": pipeline.get_execution_summary() if hasattr(pipeline, 'get_execution_summary') else {},
            "validation_results": []
        }
        
        # Run validators
        for validator in self.validators:
            validation_result = validator(result)
            test_result["validation_results"].append(validation_result)
        
        return test_result
    
    def create_conditional_pipeline(self, conditions: List[tuple]) -> SequentialPipeline:
        """Create pipeline with conditional agents"""
        agents = []
        for i, (condition, true_op, false_op) in enumerate(conditions):
            def true_behavior(msg, op=true_op):
                result = msg.copy()
                result.metadata["result"] += op
                return result
            
            def false_behavior(msg, op=false_op):
                result = msg.copy()
                result.metadata["result"] *= op
                return result
            
            agent = ConditionalAgent(f"conditional_{i}", condition, true_behavior, false_behavior)
            agents.append(agent)
        
        return SequentialPipeline(agents)


class PipelineValidator:
    """Validator for pipeline execution results"""
    
    @staticmethod
    def validate_arithmetic_result(expected: float, tolerance: float = 1e-6) -> Callable:
        """Create validator for arithmetic results"""
        def validator(result: MockMessage) -> Dict:
            actual = result.metadata.get("result", 0)
            is_valid = abs(actual - expected) <= tolerance
            
            return {
                "validator": "arithmetic_result",
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance,
                "is_valid": is_valid,
                "error": abs(actual - expected) if not is_valid else 0
            }
        
        return validator
    
    @staticmethod
    def validate_message_structure(result: MockMessage) -> Dict:
        """Validate message structure"""
        required_fields = ["role", "content", "name", "metadata"]
        missing_fields = [field for field in required_fields if not hasattr(result, field)]
        
        return {
            "validator": "message_structure",
            "required_fields": required_fields,
            "missing_fields": missing_fields,
            "is_valid": len(missing_fields) == 0
        }
    
    @staticmethod
    def validate_metadata_keys(required_keys: List[str]) -> Callable:
        """Create validator for metadata keys"""
        def validator(result: MockMessage) -> Dict:
            missing_keys = [key for key in required_keys if key not in result.metadata]
            
            return {
                "validator": "metadata_keys",
                "required_keys": required_keys,
                "missing_keys": missing_keys,
                "actual_keys": list(result.metadata.keys()),
                "is_valid": len(missing_keys) == 0
            }
        
        return validator
    
    @staticmethod
    def validate_execution_log(pipeline: SequentialPipeline) -> Dict:
        """Validate pipeline execution log"""
        log = pipeline.execution_log
        
        # Check for required fields in each step
        required_step_fields = ["step", "agent", "status"]
        invalid_steps = []
        
        for i, step in enumerate(log):
            missing_fields = [field for field in required_step_fields if field not in step]
            if missing_fields:
                invalid_steps.append({"step_index": i, "missing_fields": missing_fields})
        
        return {
            "validator": "execution_log",
            "total_steps": len(log),
            "invalid_steps": invalid_steps,
            "is_valid": len(invalid_steps) == 0,
            "step_statuses": [step.get("status", "unknown") for step in log]
        }


class PipelineTest(IsolatedAsyncioTestCase):
    """Comprehensive pipeline testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = PipelineTestFramework()
        self.validator = PipelineValidator()
    
    async def test_sequential_arithmetic_pipeline(self):
        """Test sequential arithmetic operations"""
        # Create pipeline: add 1, add 2, multiply 3
        operations = [("add", 1), ("add", 2), ("multiply", 3)]
        pipeline = self.framework.create_arithmetic_pipeline(operations)
        
        message = self.framework.create_test_message("arithmetic", 0)
        result = await pipeline.execute(message)
        
        # (0 + 1 + 2) * 3 = 9
        assert result.metadata["result"] == 9
        
        # Validate execution log
        log_validation = self.validator.validate_execution_log(pipeline)
        assert log_validation["is_valid"]
        assert log_validation["total_steps"] == 3
    
    async def test_different_operation_orders(self):
        """Test different operation orders produce different results"""
        message = self.framework.create_test_message("order_test", 0)
        
        # Order 1: add 1, add 2, multiply 3 = (0+1+2)*3 = 9
        pipeline1 = self.framework.create_arithmetic_pipeline([("add", 1), ("add", 2), ("multiply", 3)])
        result1 = await pipeline1.execute(message.copy())
        
        # Order 2: add 1, multiply 3, add 2 = (0+1)*3+2 = 5
        pipeline2 = self.framework.create_arithmetic_pipeline([("add", 1), ("multiply", 3), ("add", 2)])
        result2 = await pipeline2.execute(message.copy())
        
        # Order 3: multiply 3, add 1, add 2 = 0*3+1+2 = 3
        pipeline3 = self.framework.create_arithmetic_pipeline([("multiply", 3), ("add", 1), ("add", 2)])
        result3 = await pipeline3.execute(message.copy())
        
        assert result1.metadata["result"] == 9
        assert result2.metadata["result"] == 5
        assert result3.metadata["result"] == 3
    
    async def test_parallel_pipeline_execution(self):
        """Test parallel pipeline execution"""
        agents = [
            ArithmeticAgent("add_1", "add", 1),
            ArithmeticAgent("add_2", "add", 2),
            ArithmeticAgent("multiply_3", "multiply", 3)
        ]
        
        pipeline = ParallelPipeline(agents)
        message = self.framework.create_test_message("parallel", 5)
        
        results = await pipeline.execute(message)
        
        # All agents work on original message (5)
        # Results should be: 5+1=6, 5+2=7, 5*3=15
        result_values = [r.metadata["result"] for r in results]
        expected_values = [6, 7, 15]
        
        assert sorted(result_values) == sorted(expected_values)
    
    async def test_conditional_pipeline(self):
        """Test conditional pipeline behavior"""
        def positive_condition(msg):
            return msg.metadata.get("result", 0) > 0
        
        # If positive: add 10, else: multiply by 2
        conditions = [(positive_condition, 10, 2)]
        pipeline = self.framework.create_conditional_pipeline(conditions)
        
        # Test with positive value
        positive_msg = self.framework.create_test_message("positive", 5)
        positive_result = await pipeline.execute(positive_msg)
        assert positive_result.metadata["result"] == 15  # 5 + 10
        
        # Test with zero/negative value
        zero_msg = self.framework.create_test_message("zero", 0)
        zero_result = await pipeline.execute(zero_msg)
        assert zero_result.metadata["result"] == 0  # 0 * 2
    
    async def test_error_handling_in_pipeline(self):
        """Test error handling in pipeline execution"""
        # Create agent that raises exception
        class ErrorAgent(MockAgent):
            def default_behavior(self, message):
                raise ValueError("Simulated error")
        
        agents = [
            ArithmeticAgent("add_1", "add", 1),
            ErrorAgent("error_agent"),
            ArithmeticAgent("add_2", "add", 2)
        ]
        
        pipeline = SequentialPipeline(agents)
        
        # Add error handler
        async def error_handler(message, error):
            # Continue with original message
            result = message.copy()
            result.metadata["error_handled"] = True
            return result
        
        pipeline.add_error_handler("error_agent", error_handler)
        
        message = self.framework.create_test_message("error_test", 0)
        result = await pipeline.execute(message)
        
        # Should complete despite error
        assert result.metadata.get("error_handled") == True
        assert result.metadata["result"] == 3  # 0 + 1 + 2
    
    async def test_agent_statistics(self):
        """Test agent call statistics"""
        agent1 = ArithmeticAgent("stats_1", "add", 1)
        agent2 = ArithmeticAgent("stats_2", "multiply", 2)
        
        pipeline = SequentialPipeline([agent1, agent2])
        message = self.framework.create_test_message("stats", 0)
        
        # Run pipeline multiple times
        for _ in range(3):
            await pipeline.execute(message.copy())
        
        assert agent1.call_count == 3
        assert agent2.call_count == 3
        assert len(agent1.received_messages) == 3
        assert len(agent2.received_messages) == 3
    
    async def test_message_copying(self):
        """Test message copying and isolation"""
        agent = ArithmeticAgent("copy_test", "add", 1)
        
        original_message = self.framework.create_test_message("original", 5)
        result_message = await agent.reply(original_message)
        
        # Original should be unchanged
        assert original_message.metadata["result"] == 5
        # Result should be modified
        assert result_message.metadata["result"] == 6
    
    async def test_pipeline_framework_integration(self):
        """Test pipeline framework features"""
        # Register pipeline and agents
        operations = [("add", 1), ("multiply", 2)]
        pipeline = self.framework.create_arithmetic_pipeline(operations)
        self.framework.register_pipeline("test_pipeline", pipeline)
        
        message = self.framework.create_test_message("framework_test", 3)
        
        # Add validators
        self.framework.add_validator(self.validator.validate_arithmetic_result(8))  # (3+1)*2 = 8
        self.framework.add_validator(self.validator.validate_message_structure)
        self.framework.add_validator(self.validator.validate_metadata_keys(["result"]))
        
        # Run test
        test_result = await self.framework.run_pipeline_test("test_pipeline", "framework_test")
        
        assert test_result["output_message"].metadata["result"] == 8
        assert all(v["is_valid"] for v in test_result["validation_results"])


# Pytest integration
@pytest.fixture
def pipeline_framework():
    """Pytest fixture for pipeline framework"""
    return PipelineTestFramework()


@pytest.fixture
def pipeline_validator():
    """Pytest fixture for pipeline validator"""
    return PipelineValidator()


def test_arithmetic_agent_creation(pipeline_framework):
    """Test arithmetic agent creation"""
    agent = ArithmeticAgent("test", "add", 5)
    assert agent.name == "test"
    assert agent.operation == "add"
    assert agent.value == 5


def test_sequential_pipeline_creation(pipeline_framework):
    """Test sequential pipeline creation"""
    operations = [("add", 1), ("multiply", 2)]
    pipeline = pipeline_framework.create_arithmetic_pipeline(operations)
    
    assert len(pipeline.agents) == 2
    assert pipeline.agents[0].operation == "add"
    assert pipeline.agents[1].operation == "multiply"


@pytest.mark.asyncio
async def test_simple_pipeline_execution(pipeline_framework):
    """Test simple pipeline execution"""
    operations = [("add", 1)]
    pipeline = pipeline_framework.create_arithmetic_pipeline(operations)
    
    message = pipeline_framework.create_test_message("simple", 0)
    result = await pipeline.execute(message)
    
    assert result.metadata["result"] == 1


def test_message_structure_validation(pipeline_validator):
    """Test message structure validation"""
    message = MockMessage("user", "test", "user", {"key": "value"})
    validation_result = pipeline_validator.validate_message_structure(message)
    
    assert validation_result["is_valid"]
    assert len(validation_result["missing_fields"]) == 0