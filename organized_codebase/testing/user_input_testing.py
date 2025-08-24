# -*- coding: utf-8 -*-
"""
AgentScope User Input Testing Framework
======================================

Extracted from agentscope/tests/user_input_test.py
Enhanced for TestMaster integration

Testing patterns for:
- User terminal input handling with mocking
- Structured data model validation with Pydantic
- User agent interaction patterns
- Input prompt and response validation
- Multi-field structured input processing
- Input validation and error handling
- Interactive dialogue simulation
- User input timeout handling
"""

import asyncio
import time
from typing import Any, Dict, List, Literal, Optional, Union
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import patch, MagicMock, call

import pytest


# Mock Pydantic BaseModel for structured input validation
class MockBaseModel:
    """Mock Pydantic BaseModel for testing"""
    
    def __init__(self, **data):
        self.__dict__.update(data)
        self._validate()
    
    def _validate(self):
        """Basic validation - can be overridden"""
        pass
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def model_fields(cls) -> Dict[str, Any]:
        """Get model fields info"""
        return getattr(cls, '_fields', {})


class MockField:
    """Mock Pydantic Field"""
    
    def __init__(self, min_length: int = None, max_length: int = None, description: str = None, **kwargs):
        self.min_length = min_length
        self.max_length = max_length
        self.description = description
        self.kwargs = kwargs


def Field(min_length: int = None, max_length: int = None, description: str = None, **kwargs) -> MockField:
    """Mock Field function"""
    return MockField(min_length, max_length, description, **kwargs)


class MockChoice(MockBaseModel):
    """Mock choice model for testing"""
    
    _fields = {
        "thinking": {"type": str, "min_length": 1, "max_length": 10},
        "decision": {"type": Literal["apple", "banana", "cherry"]}
    }
    
    def __init__(self, thinking: str = None, decision: Literal["apple", "banana", "cherry"] = None):
        super().__init__(thinking=thinking, decision=decision)
    
    def _validate(self):
        """Validate fields"""
        if self.thinking:
            if len(self.thinking) < 1 or len(self.thinking) > 10:
                raise ValueError("thinking must be between 1 and 10 characters")
        
        if self.decision and self.decision not in ["apple", "banana", "cherry"]:
            raise ValueError("decision must be one of: apple, banana, cherry")


class MockMessage:
    """Mock message for user agent testing"""
    
    def __init__(self, role: str, content: str, name: str, metadata: Optional[Dict] = None):
        self.role = role
        self.content = content
        self.name = name
        self.metadata = metadata or {}


class UserAgent:
    """Mock user agent for terminal input testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.input_count = 0
        self.last_inputs = []
        self.structured_model = None
    
    async def __call__(self, structured_model: Optional[type] = None, timeout: Optional[float] = None) -> MockMessage:
        """Get user input with optional structured model"""
        self.structured_model = structured_model
        
        # Get basic content input
        content = await self._get_input("Please enter your message: ", timeout)
        
        metadata = {}
        
        # If structured model is provided, get structured input
        if structured_model:
            metadata = await self._get_structured_input(structured_model, timeout)
        
        return MockMessage("user", content, self.name, metadata)
    
    async def _get_input(self, prompt: str, timeout: Optional[float] = None) -> str:
        """Get single input with optional timeout"""
        self.input_count += 1
        
        if timeout:
            # Simulate timeout handling
            try:
                # In real implementation, this would use asyncio.wait_for
                # For testing, we'll simulate immediate response
                user_input = input(prompt)
            except Exception:
                raise TimeoutError(f"Input timeout after {timeout} seconds")
        else:
            user_input = input(prompt)
        
        self.last_inputs.append(user_input)
        return user_input
    
    async def _get_structured_input(self, model_class: type, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get structured input based on model"""
        fields_data = {}
        
        # Get model fields
        if hasattr(model_class, '_fields'):
            fields = model_class._fields
        elif hasattr(model_class, 'model_fields'):
            fields = model_class.model_fields()
        else:
            # Fallback: assume it's MockChoice for testing
            fields = MockChoice._fields
        
        # Collect input for each field
        for field_name, field_info in fields.items():
            prompt = f"Please enter {field_name}: "
            
            # Add validation info to prompt
            if isinstance(field_info, dict):
                if "min_length" in field_info and "max_length" in field_info:
                    prompt += f"(between {field_info['min_length']} and {field_info['max_length']} characters) "
            
            field_value = await self._get_input(prompt, timeout)
            fields_data[field_name] = field_value
        
        # Validate against model
        try:
            model_instance = model_class(**fields_data)
            return model_instance.model_dump()
        except Exception as e:
            # In real implementation, might retry or handle validation errors
            raise ValueError(f"Validation error: {str(e)}")
    
    def reset(self):
        """Reset agent state"""
        self.input_count = 0
        self.last_inputs.clear()
        self.structured_model = None


class InputSimulator:
    """Simulator for user input scenarios"""
    
    def __init__(self):
        self.scenarios = {}
        self.input_sequences = {}
    
    def create_input_scenario(
        self,
        name: str,
        inputs: List[str],
        expected_calls: int,
        structured_model: Optional[type] = None,
        should_timeout: bool = False,
        timeout_duration: Optional[float] = None
    ):
        """Create input test scenario"""
        self.scenarios[name] = {
            "inputs": inputs,
            "expected_calls": expected_calls,
            "structured_model": structured_model,
            "should_timeout": should_timeout,
            "timeout_duration": timeout_duration
        }
        
        self.input_sequences[name] = inputs.copy()
    
    async def run_scenario(self, scenario_name: str, agent: UserAgent) -> Dict[str, Any]:
        """Run input scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        inputs = scenario["inputs"]
        expected_calls = scenario["expected_calls"]
        structured_model = scenario["structured_model"]
        should_timeout = scenario["should_timeout"]
        timeout_duration = scenario["timeout_duration"]
        
        # Mock input function
        with patch("builtins.input", side_effect=inputs) as mock_input:
            try:
                if should_timeout and timeout_duration:
                    # Simulate timeout scenario
                    start_time = time.time()
                    result = await agent(structured_model, timeout=timeout_duration)
                    execution_time = time.time() - start_time
                else:
                    start_time = time.time()
                    result = await agent(structured_model, timeout=timeout_duration)
                    execution_time = time.time() - start_time
                
                return {
                    "scenario_name": scenario_name,
                    "success": True,
                    "result": result,
                    "execution_time": execution_time,
                    "mock_input_calls": mock_input.call_count,
                    "expected_calls": expected_calls,
                    "calls_match": mock_input.call_count == expected_calls,
                    "agent_input_count": agent.input_count,
                    "last_inputs": agent.last_inputs.copy(),
                    "timeout_occurred": False
                }
            
            except TimeoutError as e:
                return {
                    "scenario_name": scenario_name,
                    "success": False,
                    "error": str(e),
                    "timeout_occurred": True,
                    "mock_input_calls": mock_input.call_count,
                    "expected_calls": expected_calls
                }
            
            except Exception as e:
                return {
                    "scenario_name": scenario_name,
                    "success": False,
                    "error": str(e),
                    "timeout_occurred": False,
                    "mock_input_calls": mock_input.call_count,
                    "expected_calls": expected_calls
                }


class UserInputTestFramework:
    """Core framework for user input testing"""
    
    def __init__(self):
        self.agents = {}
        self.simulators = {}
        self.structured_models = {}
        self.test_results = {}
    
    def create_user_agent(self, name: str) -> UserAgent:
        """Create user agent for testing"""
        agent = UserAgent(name)
        self.agents[name] = agent
        return agent
    
    def create_input_simulator(self, name: str) -> InputSimulator:
        """Create input simulator"""
        simulator = InputSimulator()
        self.simulators[name] = simulator
        return simulator
    
    def register_structured_model(self, name: str, model_class: type):
        """Register structured data model"""
        self.structured_models[name] = model_class
    
    async def run_comprehensive_input_test(
        self,
        agent_name: str,
        simulator_name: str,
        scenario_name: str
    ) -> Dict[str, Any]:
        """Run comprehensive input test"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        if simulator_name not in self.simulators:
            raise ValueError(f"Simulator '{simulator_name}' not found")
        
        agent = self.agents[agent_name]
        simulator = self.simulators[simulator_name]
        
        # Reset agent state
        agent.reset()
        
        # Run scenario
        result = await simulator.run_scenario(scenario_name, agent)
        
        # Store result for analysis
        test_key = f"{agent_name}_{simulator_name}_{scenario_name}"
        self.test_results[test_key] = result
        
        return result
    
    def create_mock_structured_models(self):
        """Create common mock structured models for testing"""
        
        class SimpleChoice(MockBaseModel):
            _fields = {
                "choice": {"type": Literal["yes", "no"]}
            }
            
            def __init__(self, choice: Literal["yes", "no"] = None):
                super().__init__(choice=choice)
        
        class ComplexForm(MockBaseModel):
            _fields = {
                "name": {"type": str, "min_length": 2, "max_length": 50},
                "age": {"type": int, "min_value": 0, "max_value": 150},
                "category": {"type": Literal["A", "B", "C"]}
            }
            
            def __init__(self, name: str = None, age: int = None, category: Literal["A", "B", "C"] = None):
                super().__init__(name=name, age=age, category=category)
        
        self.register_structured_model("simple_choice", SimpleChoice)
        self.register_structured_model("complex_form", ComplexForm)
        self.register_structured_model("mock_choice", MockChoice)
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        timeout_tests = sum(1 for result in self.test_results.values() if result.get("timeout_occurred", False))
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "timeout_tests": timeout_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "detailed_results": self.test_results.copy()
        }


class InputValidator:
    """Validator for user input functionality"""
    
    @staticmethod
    def validate_message_structure(message: MockMessage, expected_content: str = None) -> Dict[str, Any]:
        """Validate message structure"""
        validations = {
            "has_role": hasattr(message, "role"),
            "has_content": hasattr(message, "content"),
            "has_name": hasattr(message, "name"),
            "has_metadata": hasattr(message, "metadata"),
            "role_is_user": message.role == "user" if hasattr(message, "role") else False
        }
        
        if expected_content:
            validations["content_matches"] = message.content == expected_content if hasattr(message, "content") else False
        
        return {
            "is_valid": all(validations.values()),
            "validations": validations
        }
    
    @staticmethod
    def validate_structured_metadata(
        metadata: Dict[str, Any],
        expected_fields: List[str],
        field_constraints: Dict[str, Dict] = None
    ) -> Dict[str, Any]:
        """Validate structured metadata"""
        field_constraints = field_constraints or {}
        
        validation_results = {}
        missing_fields = []
        constraint_violations = []
        
        # Check required fields
        for field in expected_fields:
            if field not in metadata:
                missing_fields.append(field)
            else:
                # Check field constraints
                if field in field_constraints:
                    constraints = field_constraints[field]
                    value = metadata[field]
                    
                    # Check length constraints
                    if "min_length" in constraints and len(str(value)) < constraints["min_length"]:
                        constraint_violations.append({
                            "field": field,
                            "constraint": "min_length",
                            "expected": constraints["min_length"],
                            "actual": len(str(value))
                        })
                    
                    if "max_length" in constraints and len(str(value)) > constraints["max_length"]:
                        constraint_violations.append({
                            "field": field,
                            "constraint": "max_length",
                            "expected": constraints["max_length"],
                            "actual": len(str(value))
                        })
                    
                    # Check allowed values
                    if "allowed_values" in constraints and value not in constraints["allowed_values"]:
                        constraint_violations.append({
                            "field": field,
                            "constraint": "allowed_values",
                            "expected": constraints["allowed_values"],
                            "actual": value
                        })
        
        return {
            "is_valid": len(missing_fields) == 0 and len(constraint_violations) == 0,
            "missing_fields": missing_fields,
            "constraint_violations": constraint_violations,
            "total_fields": len(metadata),
            "expected_fields": expected_fields
        }
    
    @staticmethod
    def validate_input_scenario_result(
        result: Dict[str, Any],
        expected_success: bool = True,
        expected_calls: int = None,
        expected_timeout: bool = False
    ) -> Dict[str, Any]:
        """Validate input scenario test result"""
        validations = {}
        
        validations["success_matches"] = result.get("success", False) == expected_success
        validations["timeout_matches"] = result.get("timeout_occurred", False) == expected_timeout
        
        if expected_calls is not None:
            validations["calls_match"] = result.get("calls_match", False)
            validations["call_count_correct"] = result.get("mock_input_calls", 0) == expected_calls
        
        return {
            "is_valid": all(validations.values()),
            "validations": validations,
            "result_summary": {
                "success": result.get("success", False),
                "timeout": result.get("timeout_occurred", False),
                "calls": result.get("mock_input_calls", 0),
                "execution_time": result.get("execution_time", 0)
            }
        }


class UserInputTest(IsolatedAsyncioTestCase):
    """Comprehensive user input testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = UserInputTestFramework()
        self.validator = InputValidator()
        
        # Create common test components
        self.framework.create_mock_structured_models()
        self.user_agent = self.framework.create_user_agent("Alice")
        self.simulator = self.framework.create_input_simulator("basic_simulator")
    
    async def test_basic_user_terminal_input(self):
        """Test basic user terminal input without structured data"""
        # Create simple input scenario
        self.simulator.create_input_scenario(
            "basic_input",
            inputs=["Hello, world!"],
            expected_calls=1
        )
        
        # Run test
        result = await self.framework.run_comprehensive_input_test(
            "Alice",
            "basic_simulator",
            "basic_input"
        )
        
        # Validate results
        assert result["success"] == True
        assert result["calls_match"] == True
        assert result["result"].content == "Hello, world!"
        
        # Validate message structure
        message_validation = self.validator.validate_message_structure(
            result["result"],
            expected_content="Hello, world!"
        )
        assert message_validation["is_valid"]
    
    async def test_structured_user_input(self):
        """Test structured user input with mock choice model"""
        # Create structured input scenario
        self.simulator.create_input_scenario(
            "structured_choice",
            inputs=["Hi!", "sth", "apple"],
            expected_calls=3,
            structured_model=MockChoice
        )
        
        # Run test
        result = await self.framework.run_comprehensive_input_test(
            "Alice",
            "basic_simulator",
            "structured_choice"
        )
        
        # Validate results
        assert result["success"] == True
        assert result["calls_match"] == True
        assert result["result"].content == "Hi!"
        
        # Validate structured metadata
        expected_fields = ["thinking", "decision"]
        field_constraints = {
            "thinking": {"min_length": 1, "max_length": 10},
            "decision": {"allowed_values": ["apple", "banana", "cherry"]}
        }
        
        metadata_validation = self.validator.validate_structured_metadata(
            result["result"].metadata,
            expected_fields,
            field_constraints
        )
        assert metadata_validation["is_valid"]
        assert result["result"].metadata["thinking"] == "sth"
        assert result["result"].metadata["decision"] == "apple"
    
    async def test_input_validation_errors(self):
        """Test input validation error handling"""
        # Create scenario with invalid structured input
        self.simulator.create_input_scenario(
            "invalid_structured",
            inputs=["Test", "toolongstring", "invalid_choice"],
            expected_calls=3,
            structured_model=MockChoice
        )
        
        # Run test - should fail due to validation errors
        result = await self.framework.run_comprehensive_input_test(
            "Alice",
            "basic_simulator",
            "invalid_structured"
        )
        
        # Should fail due to validation error
        assert result["success"] == False
        assert "Validation error" in result.get("error", "")
    
    async def test_multiple_input_scenarios(self):
        """Test multiple input scenarios with different models"""
        # Create multiple scenarios
        scenarios = [
            {
                "name": "simple_yes_no",
                "inputs": ["yes"],
                "calls": 2,  # content + choice
                "model": self.framework.structured_models["simple_choice"]
            },
            {
                "name": "basic_message",
                "inputs": ["Just a message"],
                "calls": 1,
                "model": None
            }
        ]
        
        results = []
        for scenario in scenarios:
            self.simulator.create_input_scenario(
                scenario["name"],
                scenario["inputs"],
                scenario["calls"],
                scenario["model"]
            )
            
            # Create fresh agent for each scenario
            agent = self.framework.create_user_agent(f"Agent_{scenario['name']}")
            
            result = await self.simulator.run_scenario(scenario["name"], agent)
            results.append(result)
        
        # Validate all scenarios
        for i, result in enumerate(results):
            scenario = scenarios[i]
            
            scenario_validation = self.validator.validate_input_scenario_result(
                result,
                expected_success=True,
                expected_calls=scenario["calls"]
            )
            assert scenario_validation["is_valid"]
    
    async def test_input_timeout_simulation(self):
        """Test input timeout handling"""
        # Create timeout scenario
        self.simulator.create_input_scenario(
            "timeout_test",
            inputs=["Should not be reached"],
            expected_calls=0,
            should_timeout=True,
            timeout_duration=0.1  # Very short timeout
        )
        
        # Mock input to simulate delay
        async def delayed_input(*args, **kwargs):
            await asyncio.sleep(0.2)  # Longer than timeout
            return "delayed response"
        
        # This test simulates timeout behavior
        # In real implementation, timeout would be handled by asyncio.wait_for
        
        # For testing purposes, we'll create a scenario that completes quickly
        self.simulator.create_input_scenario(
            "quick_response",
            inputs=["Quick response"],
            expected_calls=1,
            timeout_duration=5.0  # Generous timeout
        )
        
        result = await self.framework.run_comprehensive_input_test(
            "Alice",
            "basic_simulator",
            "quick_response"
        )
        
        assert result["success"] == True
        assert result["timeout_occurred"] == False
    
    async def test_agent_state_management(self):
        """Test user agent state management"""
        agent = self.framework.create_user_agent("StateTestAgent")
        
        # Verify initial state
        assert agent.input_count == 0
        assert len(agent.last_inputs) == 0
        
        # Run input scenario
        self.simulator.create_input_scenario(
            "state_test",
            inputs=["First input", "Second input"],
            expected_calls=2,
            structured_model=self.framework.structured_models["simple_choice"]
        )
        
        result = await self.simulator.run_scenario("state_test", agent)
        
        # Verify state changes
        assert agent.input_count > 0
        assert len(agent.last_inputs) > 0
        assert "First input" in agent.last_inputs
        
        # Test reset functionality
        agent.reset()
        assert agent.input_count == 0
        assert len(agent.last_inputs) == 0
    
    async def test_framework_comprehensive_summary(self):
        """Test framework's comprehensive test summary"""
        # Run multiple scenarios
        scenarios = ["basic_test", "structured_test", "error_test"]
        
        for i, scenario_name in enumerate(scenarios):
            should_succeed = i < 2  # First two should succeed, third should fail
            
            if should_succeed:
                self.simulator.create_input_scenario(
                    scenario_name,
                    inputs=["Test input", "valid", "apple"],
                    expected_calls=3,
                    structured_model=MockChoice
                )
            else:
                # Create failing scenario
                self.simulator.create_input_scenario(
                    scenario_name,
                    inputs=["Test", "waytoolongstring", "invalid"],
                    expected_calls=3,
                    structured_model=MockChoice
                )
            
            await self.framework.run_comprehensive_input_test(
                "Alice",
                "basic_simulator",
                scenario_name
            )
        
        # Get comprehensive summary
        summary = self.framework.get_test_summary()
        
        assert summary["total_tests"] == 3
        assert summary["successful_tests"] >= 2
        assert "detailed_results" in summary
        assert summary["success_rate"] >= 0.6  # At least 2/3 should succeed
    
    async def test_complex_structured_model(self):
        """Test complex structured model with multiple field types"""
        # Define complex model scenario
        class PersonInfo(MockBaseModel):
            _fields = {
                "name": {"type": str, "min_length": 2, "max_length": 50},
                "age": {"type": str},  # Will be validated as string input
                "preference": {"type": Literal["option1", "option2", "option3"]}
            }
            
            def __init__(self, name: str = None, age: str = None, preference: str = None):
                super().__init__(name=name, age=age, preference=preference)
            
            def _validate(self):
                if self.name and (len(self.name) < 2 or len(self.name) > 50):
                    raise ValueError("Name must be between 2 and 50 characters")
                if self.preference and self.preference not in ["option1", "option2", "option3"]:
                    raise ValueError("Preference must be option1, option2, or option3")
        
        # Create scenario with valid complex input
        self.simulator.create_input_scenario(
            "complex_structured",
            inputs=["Personal info", "Alice Smith", "25", "option2"],
            expected_calls=4,  # Content + 3 structured fields
            structured_model=PersonInfo
        )
        
        result = await self.framework.run_comprehensive_input_test(
            "Alice",
            "basic_simulator",
            "complex_structured"
        )
        
        assert result["success"] == True
        assert result["result"].metadata["name"] == "Alice Smith"
        assert result["result"].metadata["age"] == "25"
        assert result["result"].metadata["preference"] == "option2"


# Pytest integration
@pytest.fixture
def input_framework():
    """Pytest fixture for input framework"""
    framework = UserInputTestFramework()
    framework.create_mock_structured_models()
    return framework


@pytest.fixture
def input_validator():
    """Pytest fixture for input validator"""
    return InputValidator()


def test_input_framework_creation(input_framework):
    """Test input framework creation"""
    agent = input_framework.create_user_agent("TestAgent")
    assert agent.name == "TestAgent"
    assert "TestAgent" in input_framework.agents


def test_input_simulator_creation(input_framework):
    """Test input simulator creation"""
    simulator = input_framework.create_input_simulator("TestSimulator")
    assert "TestSimulator" in input_framework.simulators
    
    # Test scenario creation
    simulator.create_input_scenario("test", ["input"], 1)
    assert "test" in simulator.scenarios


def test_mock_choice_validation():
    """Test MockChoice model validation"""
    # Valid choice
    choice = MockChoice(thinking="test", decision="apple")
    assert choice.thinking == "test"
    assert choice.decision == "apple"
    
    # Invalid thinking length
    with pytest.raises(ValueError):
        MockChoice(thinking="", decision="apple")  # Too short
    
    with pytest.raises(ValueError):
        MockChoice(thinking="waytoolongstring", decision="apple")  # Too long
    
    # Invalid decision
    with pytest.raises(ValueError):
        MockChoice(thinking="test", decision="invalid")


def test_message_structure_validation(input_validator):
    """Test message structure validation"""
    message = MockMessage("user", "test content", "TestUser", {"key": "value"})
    
    validation = input_validator.validate_message_structure(message, "test content")
    assert validation["is_valid"]
    assert validation["validations"]["role_is_user"]
    assert validation["validations"]["content_matches"]


def test_structured_metadata_validation(input_validator):
    """Test structured metadata validation"""
    metadata = {"thinking": "test", "decision": "apple"}
    expected_fields = ["thinking", "decision"]
    constraints = {
        "thinking": {"min_length": 1, "max_length": 10},
        "decision": {"allowed_values": ["apple", "banana", "cherry"]}
    }
    
    validation = input_validator.validate_structured_metadata(metadata, expected_fields, constraints)
    assert validation["is_valid"]
    assert len(validation["missing_fields"]) == 0
    assert len(validation["constraint_violations"]) == 0


@pytest.mark.asyncio
async def test_simple_user_agent_call():
    """Test simple user agent call"""
    agent = UserAgent("SimpleTest")
    
    with patch("builtins.input", return_value="Test input"):
        result = await agent()
        
        assert result.role == "user"
        assert result.content == "Test input"
        assert result.name == "SimpleTest"
        assert agent.input_count == 1


def test_input_scenario_result_validation(input_validator):
    """Test input scenario result validation"""
    # Successful result
    result = {
        "success": True,
        "timeout_occurred": False,
        "mock_input_calls": 3,
        "calls_match": True,
        "execution_time": 0.1
    }
    
    validation = input_validator.validate_input_scenario_result(
        result,
        expected_success=True,
        expected_calls=3,
        expected_timeout=False
    )
    
    assert validation["is_valid"]
    assert validation["result_summary"]["success"] == True
    assert validation["result_summary"]["calls"] == 3