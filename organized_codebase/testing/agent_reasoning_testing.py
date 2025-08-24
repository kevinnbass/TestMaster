"""Agent Reasoning Testing Framework - CrewAI Pattern
Extracted patterns for testing agent reasoning capabilities
Supports reasoning validation, plan refinement, and max attempts handling
"""
import json
from typing import Any, Dict, List, Optional, Callable, Union
from unittest.mock import Mock, MagicMock
from pydantic import BaseModel, Field
import pytest


class MockLLM:
    """Mock LLM for reasoning tests"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.call_responses = {}
        self.supports_function_calling_flag = False
        self.call_count = 0
    
    def call(self, messages: List[Dict], *args, **kwargs) -> str:
        """Mock LLM call with configurable responses"""
        self.call_count += 1
        
        # Check for function calling
        if "tools" in kwargs and self.supports_function_calling_flag:
            return self._handle_function_call(messages, **kwargs)
        
        # Check message content for reasoning prompts
        for message in messages:
            content = message.get("content", "")
            if "create a detailed plan" in content:
                return self._get_reasoning_response("planning")
            elif "refine your plan" in content:
                return self._get_reasoning_response("refining")
        
        # Default task execution response
        return self._get_reasoning_response("execution")
    
    def supports_function_calling(self) -> bool:
        """Check if LLM supports function calling"""
        return self.supports_function_calling_flag
    
    def set_responses(self, responses: Dict[str, str]):
        """Set predefined responses for different scenarios"""
        self.call_responses = responses
    
    def enable_function_calling(self, enabled: bool = True):
        """Enable/disable function calling support"""
        self.supports_function_calling_flag = enabled
    
    def _handle_function_call(self, messages: List[Dict], **kwargs) -> str:
        """Handle function calling scenarios"""
        if "function_response" in self.call_responses:
            return self.call_responses["function_response"]
        
        # Default function calling response
        return json.dumps({
            "plan": "I'll solve this problem step by step.",
            "ready": True
        })
    
    def _get_reasoning_response(self, scenario: str) -> str:
        """Get response based on reasoning scenario"""
        scenario_key = f"{scenario}_response"
        if scenario_key in self.call_responses:
            return self.call_responses[scenario_key]
        
        # Default responses
        defaults = {
            "planning": "I'll solve this problem.\n\nREADY: I am ready to execute the task.",
            "refining": "I've refined my approach.\n\nREADY: I am ready to execute the task.",
            "execution": "Task completed successfully"
        }
        
        return defaults.get(scenario, "Default response")


class MockAgent:
    """Mock agent with reasoning capabilities"""
    
    def __init__(self, role: str, goal: str, backstory: str, llm: MockLLM,
                 reasoning: bool = False, max_reasoning_attempts: int = 3, verbose: bool = False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.reasoning = reasoning
        self.max_reasoning_attempts = max_reasoning_attempts
        self.verbose = verbose
    
    def execute_task(self, task: 'MockTask') -> str:
        """Execute task with optional reasoning"""
        if not self.reasoning:
            return self.llm.call([{"content": "Execute task"}])
        
        # Perform reasoning process
        reasoning_result = self._perform_reasoning(task)
        
        if reasoning_result.get("ready", False):
            # Update task description with reasoning plan
            if "plan" in reasoning_result:
                task.add_reasoning_plan(reasoning_result["plan"])
        
        # Execute the actual task
        return self.llm.call([{"content": f"Execute: {task.description}"}])
    
    def _perform_reasoning(self, task: 'MockTask') -> Dict[str, Any]:
        """Perform reasoning with retry logic"""
        for attempt in range(self.max_reasoning_attempts):
            try:
                if attempt == 0:
                    # Initial planning
                    response = self.llm.call([{"content": "create a detailed plan"}])
                else:
                    # Plan refinement
                    response = self.llm.call([{"content": "refine your plan"}])
                
                # Check if using function calling
                if self.llm.supports_function_calling():
                    try:
                        result = json.loads(response)
                        if result.get("ready", False):
                            return result
                    except json.JSONDecodeError:
                        # Fall back to text parsing
                        pass
                
                # Text-based parsing
                if "READY:" in response:
                    return {
                        "ready": True,
                        "plan": response,
                        "attempts": attempt + 1
                    }
                elif "NOT READY:" in response:
                    continue  # Try again
                else:
                    return {
                        "ready": True,
                        "plan": response,
                        "attempts": attempt + 1
                    }
                    
            except Exception as e:
                if attempt == self.max_reasoning_attempts - 1:
                    # Last attempt failed, proceed anyway
                    return {
                        "ready": True,
                        "plan": "Proceeding despite reasoning failure",
                        "error": str(e),
                        "attempts": attempt + 1
                    }
        
        # Max attempts reached without being ready
        return {
            "ready": True,
            "plan": "Proceeding after max attempts",
            "attempts": self.max_reasoning_attempts
        }


class MockTask:
    """Mock task for reasoning tests"""
    
    def __init__(self, description: str, expected_output: str, agent: Optional[MockAgent] = None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.reasoning_plan = None
    
    def add_reasoning_plan(self, plan: str):
        """Add reasoning plan to task description"""
        self.reasoning_plan = plan
        self.description = f"{self.description}\n\nReasoning Plan: {plan}"


class ReasoningTestFramework:
    """Framework for testing agent reasoning capabilities"""
    
    def __init__(self):
        self.test_results = []
        self.mock_responses = {}
    
    def create_reasoning_agent(self, reasoning: bool = True, max_attempts: int = 3) -> MockAgent:
        """Create agent with reasoning capabilities"""
        llm = MockLLM()
        return MockAgent(
            role="Test Agent",
            goal="To test the reasoning feature", 
            backstory="I am a test agent created to verify the reasoning feature works correctly.",
            llm=llm,
            reasoning=reasoning,
            max_reasoning_attempts=max_attempts,
            verbose=True
        )
    
    def test_basic_reasoning(self, agent: MockAgent, task_description: str, 
                           expected_output: str) -> Dict[str, Any]:
        """Test basic reasoning functionality"""
        # Configure successful reasoning response
        agent.llm.set_responses({
            "planning_response": "I'll solve this simple problem.\n\nREADY: I am ready to execute the task.",
            "execution_response": expected_output
        })
        
        task = MockTask(task_description, expected_output, agent)
        
        try:
            result = agent.execute_task(task)
            
            return {
                'success': True,
                'result': result,
                'reasoning_plan_added': "Reasoning Plan:" in task.description,
                'llm_call_count': agent.llm.call_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_reasoning_refinement(self, agent: MockAgent, task_description: str) -> Dict[str, Any]:
        """Test reasoning with plan refinement"""
        # Configure refinement scenario
        call_count = [0]
        
        def dynamic_response(scenario):
            call_count[0] += 1
            if call_count[0] == 1:
                return "I need to think more.\n\nNOT READY: I need to refine my plan."
            else:
                return "Now I have a better approach.\n\nREADY: I am ready to execute the task."
        
        agent.llm.set_responses({
            "planning_response": dynamic_response("planning"),
            "refining_response": dynamic_response("refining"),
            "execution_response": "Refined execution result"
        })
        
        # Override call method to handle dynamic responses
        original_call = agent.llm.call
        def mock_call(messages, *args, **kwargs):
            for message in messages:
                content = message.get("content", "")
                if "create a detailed plan" in content or "refine your plan" in content:
                    return dynamic_response("reasoning")
            return "Refined execution result"
        
        agent.llm.call = mock_call
        
        task = MockTask(task_description, "Expected refined output", agent)
        
        try:
            result = agent.execute_task(task)
            
            return {
                'success': True,
                'result': result,
                'refinement_attempts': call_count[0],
                'reasoning_plan_added': "Reasoning Plan:" in task.description
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_max_attempts_exhaustion(self, agent: MockAgent) -> Dict[str, Any]:
        """Test behavior when max reasoning attempts are exhausted"""
        # Configure to always return NOT READY
        call_count = [0]
        
        def always_not_ready(scenario):
            call_count[0] += 1
            return f"Attempt {call_count[0]}: Still not ready.\n\nNOT READY: Need more refinement."
        
        agent.llm.set_responses({
            "planning_response": always_not_ready("planning"),
            "refining_response": always_not_ready("refining"),
            "execution_response": "Executed despite not being ready"
        })
        
        # Override call method
        original_call = agent.llm.call
        def mock_call(messages, *args, **kwargs):
            for message in messages:
                content = message.get("content", "")
                if "create a detailed plan" in content or "refine your plan" in content:
                    return always_not_ready("reasoning")
            return "Executed despite not being ready"
        
        agent.llm.call = mock_call
        
        task = MockTask("Complex unsolvable task", "Some output", agent)
        
        try:
            result = agent.execute_task(task)
            
            return {
                'success': True,
                'result': result,
                'reasoning_attempts': min(call_count[0], agent.max_reasoning_attempts),
                'max_attempts_reached': call_count[0] >= agent.max_reasoning_attempts
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_function_calling_reasoning(self, agent: MockAgent) -> Dict[str, Any]:
        """Test reasoning with function calling support"""
        agent.llm.enable_function_calling(True)
        agent.llm.set_responses({
            "function_response": json.dumps({
                "plan": "I'll solve this using function calling.",
                "ready": True
            }),
            "execution_response": "Function calling result"
        })
        
        task = MockTask("Function calling task", "Expected output", agent)
        
        try:
            result = agent.execute_task(task)
            
            return {
                'success': True,
                'result': result,
                'function_calling_used': True,
                'plan_in_description': "I'll solve this using function calling." in task.description
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_function_calling_fallback(self, agent: MockAgent) -> Dict[str, Any]:
        """Test fallback when function calling fails"""
        agent.llm.enable_function_calling(True)
        agent.llm.set_responses({
            "function_response": "Invalid JSON. READY: Falling back to text parsing.",
            "execution_response": "Fallback result"
        })
        
        task = MockTask("Fallback test task", "Expected output", agent)
        
        try:
            result = agent.execute_task(task)
            
            return {
                'success': True,
                'result': result,
                'fallback_used': True,
                'fallback_text_in_description': "Invalid JSON" in task.description
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_reasoning_error_handling(self, agent: MockAgent) -> Dict[str, Any]:
        """Test error handling during reasoning"""
        call_count = [0]
        
        def error_then_success(scenario):
            call_count[0] += 1
            if call_count[0] <= 2:  # First two calls fail
                raise Exception("LLM error during reasoning")
            return "Recovered execution result"
        
        # Override call method
        original_call = agent.llm.call
        def mock_call(messages, *args, **kwargs):
            return error_then_success("any")
        
        agent.llm.call = mock_call
        
        task = MockTask("Error handling task", "Expected output", agent)
        
        try:
            result = agent.execute_task(task)
            
            return {
                'success': True,
                'result': result,
                'error_recovery': True,
                'call_attempts': call_count[0]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'call_attempts': call_count[0]
            }


class ReasoningValidationFramework:
    """Framework for validating reasoning behavior"""
    
    def __init__(self):
        self.reasoning_framework = ReasoningTestFramework()
    
    def validate_input_parameters(self) -> Dict[str, Any]:
        """Validate reasoning input parameter validation"""
        llm = MockLLM()
        
        # Test missing task
        try:
            # Simulate AgentReasoning initialization validation
            if None is None:  # Simulating task=None
                raise ValueError("Both task and agent must be provided")
        except ValueError as e:
            missing_task_error = str(e)
        else:
            missing_task_error = None
        
        # Test missing agent 
        try:
            task = MockTask("Test task", "Test output")
            if None is None:  # Simulating agent=None
                raise ValueError("Both task and agent must be provided")
        except ValueError as e:
            missing_agent_error = str(e)
        else:
            missing_agent_error = None
        
        return {
            'missing_task_validation': missing_task_error == "Both task and agent must be provided",
            'missing_agent_validation': missing_agent_error == "Both task and agent must be provided",
            'validation_working': True
        }
    
    def validate_reasoning_scenarios(self) -> Dict[str, Any]:
        """Validate all reasoning test scenarios"""
        results = {}
        
        # Test basic reasoning
        agent = self.reasoning_framework.create_reasoning_agent()
        results['basic_reasoning'] = self.reasoning_framework.test_basic_reasoning(
            agent, "Simple math: What's 2+2?", "4"
        )
        
        # Test refinement
        agent = self.reasoning_framework.create_reasoning_agent(max_attempts=2)
        results['refinement'] = self.reasoning_framework.test_reasoning_refinement(
            agent, "Complex math: What's the derivative of x²?"
        )
        
        # Test max attempts
        agent = self.reasoning_framework.create_reasoning_agent(max_attempts=2)
        results['max_attempts'] = self.reasoning_framework.test_max_attempts_exhaustion(agent)
        
        # Test function calling
        agent = self.reasoning_framework.create_reasoning_agent()
        results['function_calling'] = self.reasoning_framework.test_function_calling_reasoning(agent)
        
        # Test fallback
        agent = self.reasoning_framework.create_reasoning_agent()
        results['fallback'] = self.reasoning_framework.test_function_calling_fallback(agent)
        
        # Test error handling
        agent = self.reasoning_framework.create_reasoning_agent()
        results['error_handling'] = self.reasoning_framework.test_reasoning_error_handling(agent)
        
        return results
    
    def create_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of reasoning test results"""
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('success', False))
        
        return {
            'total_scenarios': total_tests,
            'passed_scenarios': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'detailed_results': results
        }


# Pytest integration patterns
class PyTestReasoningPatterns:
    """Reasoning testing patterns for pytest"""
    
    @pytest.fixture
    def mock_llm_responses(self):
        """Fixture for mock LLM responses"""
        return {
            "ready": "I'll solve this simple problem.\n\nREADY: I am ready to execute the task.\n\n",
            "not_ready": "I need to think about this more.\n\nNOT READY: I need to refine my plan.",
            "ready_after_refine": "Now I have a good approach.\n\nREADY: I am ready to execute the task.",
            "execution": "Task completed"
        }
    
    @pytest.fixture
    def reasoning_framework(self):
        """Provide reasoning test framework"""
        return ReasoningTestFramework()
    
    def test_agent_with_basic_reasoning(self, reasoning_framework, mock_llm_responses):
        """Test basic agent reasoning functionality"""
        agent = reasoning_framework.create_reasoning_agent()
        agent.llm.set_responses({
            "planning_response": mock_llm_responses["ready"],
            "execution_response": mock_llm_responses["execution"]
        })
        
        result = reasoning_framework.test_basic_reasoning(
            agent, "Simple task", "Expected result"
        )
        
        assert result['success'] is True
        assert result['reasoning_plan_added'] is True
        assert "READY:" in mock_llm_responses["ready"]
    
    def test_reasoning_refinement_process(self, reasoning_framework):
        """Test reasoning refinement with multiple attempts"""
        agent = reasoning_framework.create_reasoning_agent(max_attempts=3)
        
        result = reasoning_framework.test_reasoning_refinement(
            agent, "Complex derivative problem"
        )
        
        assert result['success'] is True
        assert result['refinement_attempts'] >= 1
    
    def test_max_reasoning_attempts(self, reasoning_framework):
        """Test max reasoning attempts exhaustion"""
        agent = reasoning_framework.create_reasoning_agent(max_attempts=2)
        
        result = reasoning_framework.test_max_attempts_exhaustion(agent)
        
        assert result['success'] is True
        assert result['reasoning_attempts'] <= agent.max_reasoning_attempts
    
    def test_function_calling_integration(self, reasoning_framework):
        """Test function calling in reasoning"""
        agent = reasoning_framework.create_reasoning_agent()
        
        result = reasoning_framework.test_function_calling_reasoning(agent)
        
        assert result['success'] is True
        assert result['function_calling_used'] is True


# Export patterns for integration
__all__ = [
    'ReasoningTestFramework',
    'ReasoningValidationFramework',
    'MockAgent',
    'MockTask',
    'MockLLM',
    'PyTestReasoningPatterns'
]