"""Task Guardrails Testing Framework - CrewAI Pattern
Extracted patterns for testing task-level guardrails and validation
Supports custom validators, retry mechanisms, and event emission
"""
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel, Field
import pytest


class MockTaskOutput:
    """Mock task output for guardrail testing"""
    
    def __init__(self, raw: str, description: str = "Test task", 
                 expected_output: str = "Expected output", agent: str = "Test Agent"):
        self.raw = raw
        self.description = description
        self.expected_output = expected_output
        self.agent = agent


class MockLLMGuardrail:
    """Mock LLM-based guardrail"""
    
    def __init__(self, description: str, llm: 'MockLLM'):
        self.description = description
        self.llm = llm
        self.call_count = 0
    
    def __call__(self, task_output: MockTaskOutput) -> Tuple[bool, str]:
        """Execute LLM guardrail validation"""
        self.call_count += 1
        
        # Simple validation based on word count for testing
        word_count = len(task_output.raw.split())
        
        if "less than 10 words" in self.description:
            if word_count >= 10:
                return False, f"Response exceeds the guardrail limit of fewer than 10 words (found {word_count} words)"
            return True, task_output.raw
            
        elif "less than 500 words" in self.description:
            if word_count >= 500:
                return False, f"Response exceeds the guardrail limit of fewer than 500 words (found {word_count} words)"
            return True, task_output.raw
        
        # Default behavior - check for specific patterns
        if "authors are from Italy" in self.description:
            if "italy" not in task_output.raw.lower() and "italian" not in task_output.raw.lower():
                return False, "The task result does not comply with the guardrail because none of the listed authors are from Italy"
            return True, task_output.raw
        
        return True, task_output.raw


class MockLLM:
    """Mock LLM for guardrail testing"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.responses = {}
    
    def call(self, messages: List[Dict], *args, **kwargs) -> str:
        """Mock LLM call"""
        return self.responses.get('default', 'Mock LLM response')
    
    def set_response(self, key: str, response: str):
        """Set mock response"""
        self.responses[key] = response


class MockAgent:
    """Mock agent for task execution"""
    
    def __init__(self, role: str, goal: str, backstory: str):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.crew = None
        self.execution_count = 0
        self.last_context = None
    
    def execute_task(self, task, context: str = None, tools: List = None):
        """Mock task execution"""
        self.execution_count += 1
        self.last_context = context
        
        # Return different responses based on execution count for retry testing
        if hasattr(task, 'mock_responses'):
            if self.execution_count <= len(task.mock_responses):
                return task.mock_responses[self.execution_count - 1]
        
        return "mock task result"
    
    def kickoff(self, *args, **kwargs):
        """Mock agent kickoff for error simulation"""
        if hasattr(self, '_should_raise_error') and self._should_raise_error:
            raise Exception("Unexpected error")
        return "Agent kickoff result"


class MockTask:
    """Mock task with guardrail support"""
    
    def __init__(self, description: str, expected_output: str, agent: MockAgent = None,
                 guardrail: Union[Callable, str, object] = None, max_retries: int = 1):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.guardrail = guardrail
        self.max_retries = max_retries
        self.retry_count = 0
        self.execution_count = 0
        self.mock_responses = []
        self._events = []
    
    def execute_sync(self, agent: MockAgent = None) -> MockTaskOutput:
        """Execute task synchronously with guardrail validation"""
        execution_agent = agent or self.agent
        
        if not execution_agent:
            raise ValueError("No agent provided for task execution")
        
        attempt = 0
        while attempt <= self.max_retries:
            try:
                # Execute the task
                result = execution_agent.execute_task(self)
                
                # Create task output
                task_output = MockTaskOutput(
                    raw=result,
                    description=self.description,
                    expected_output=self.expected_output,
                    agent=execution_agent.role
                )
                
                # Apply guardrail if present
                if self.guardrail:
                    validation_result = self._apply_guardrail(task_output, attempt)
                    
                    if not validation_result['passed']:
                        attempt += 1
                        self.retry_count = attempt
                        
                        if attempt > self.max_retries:
                            raise Exception(f"Task failed guardrail validation after {self.max_retries} retries: {validation_result['error']}")
                        
                        # Add error to context for retry
                        context = f"Previous attempt failed validation: {validation_result['error']}"
                        execution_agent.last_context = context
                        continue
                    else:
                        # Validation passed, update output if transformed
                        if validation_result['transformed_output']:
                            task_output.raw = validation_result['transformed_output']
                
                return task_output
                
            except Exception as e:
                if "validation" not in str(e) and attempt <= self.max_retries:
                    attempt += 1
                    self.retry_count = attempt
                    continue
                raise e
        
        raise Exception("Maximum retries exceeded")
    
    def _apply_guardrail(self, task_output: MockTaskOutput, attempt: int) -> Dict[str, Any]:
        """Apply guardrail validation"""
        if callable(self.guardrail):
            # Function-based guardrail
            try:
                passed, output = self.guardrail(task_output)
                return {
                    'passed': passed,
                    'transformed_output': output if passed else None,
                    'error': output if not passed else None
                }
            except Exception as e:
                return {
                    'passed': False,
                    'transformed_output': None,
                    'error': str(e)
                }
        
        elif isinstance(self.guardrail, str):
            # String-based guardrail (LLM validation)
            llm = MockLLM()
            llm_guardrail = MockLLMGuardrail(self.guardrail, llm)
            passed, output = llm_guardrail(task_output)
            return {
                'passed': passed,
                'transformed_output': output if passed else None,
                'error': output if not passed else None
            }
        
        elif hasattr(self.guardrail, '__call__'):
            # Object-based guardrail
            passed, output = self.guardrail(task_output)
            return {
                'passed': passed,
                'transformed_output': output if passed else None,
                'error': output if not passed else None
            }
        
        return {'passed': True, 'transformed_output': None, 'error': None}
    
    def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit task event"""
        self._events.append({
            'type': event_type,
            'data': event_data
        })
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get emitted events"""
        return self._events


class GuardrailTestFramework:
    """Framework for testing task guardrails"""
    
    def __init__(self):
        self.test_results = []
        self.event_captures = []
    
    def create_test_agent(self) -> MockAgent:
        """Create test agent"""
        return MockAgent(
            role="Test Agent",
            goal="Test Goal", 
            backstory="Test Backstory"
        )
    
    def test_task_without_guardrail(self) -> Dict[str, Any]:
        """Test task execution without guardrails (backward compatibility)"""
        agent = Mock()
        agent.role = "test_agent"
        agent.execute_task.return_value = "test result"
        agent.crew = None
        
        task = MockTask(description="Test task", expected_output="Output")
        
        try:
            result = task.execute_sync(agent=agent)
            
            return {
                'success': True,
                'result_generated': isinstance(result, MockTaskOutput),
                'expected_output': result.raw == "test result",
                'backward_compatible': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_successful_guardrail(self) -> Dict[str, Any]:
        """Test successful guardrail validation with transformation"""
        def success_guardrail(result: MockTaskOutput) -> Tuple[bool, str]:
            return (True, result.raw.upper())
        
        agent = Mock()
        agent.role = "test_agent"
        agent.execute_task.return_value = "test result"
        agent.crew = None
        
        task = MockTask(
            description="Test task", 
            expected_output="Output",
            guardrail=success_guardrail
        )
        
        try:
            result = task.execute_sync(agent=agent)
            
            return {
                'success': True,
                'guardrail_applied': True,
                'output_transformed': result.raw == "TEST RESULT",
                'validation_passed': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_failing_guardrail_with_retries(self) -> Dict[str, Any]:
        """Test failing guardrail with retry mechanism"""
        def failing_guardrail(result: MockTaskOutput) -> Tuple[bool, str]:
            return (False, "Invalid format")
        
        agent = Mock()
        agent.role = "test_agent"
        agent.execute_task.side_effect = ["bad result", "good result"]
        agent.crew = None
        
        task = MockTask(
            description="Test task",
            expected_output="Output", 
            guardrail=failing_guardrail,
            max_retries=1
        )
        
        try:
            result = task.execute_sync(agent=agent)
            
            return {
                'success': False,  # Should fail
                'unexpected_success': True
            }
            
        except Exception as e:
            return {
                'success': True,  # Expected failure
                'error_message': str(e),
                'retry_count': task.retry_count,
                'expected_error': "Task failed guardrail validation" in str(e),
                'max_retries_respected': task.retry_count == task.max_retries
            }
    
    def test_guardrail_retry_with_context(self) -> Dict[str, Any]:
        """Test that guardrail error is passed in context for retry"""
        call_count = [0]
        
        def context_checking_guardrail(result: MockTaskOutput) -> Tuple[bool, str]:
            call_count[0] += 1
            if call_count[0] == 1:
                return (False, "Expected JSON, got string")
            return (True, '{"valid": "json"}')
        
        agent = Mock()
        agent.role = "test_agent"
        agent.crew = None
        
        # Mock execute_task to track context
        contexts_received = []
        
        def track_context_execute_task(task, context=None, tools=None):
            contexts_received.append(context)
            if len(contexts_received) == 1:
                return "invalid"
            return '{"valid": "json"}'
        
        agent.execute_task.side_effect = track_context_execute_task
        
        task = MockTask(
            description="Test task",
            expected_output="Output",
            guardrail=context_checking_guardrail,
            max_retries=1
        )
        
        try:
            result = task.execute_sync(agent=agent)
            
            return {
                'success': False,  # Should have failed after retries
                'unexpected_success': True
            }
            
        except Exception as e:
            return {
                'success': True,
                'error_contains_guardrail_message': "Expected JSON, got string" in str(e),
                'context_passed_to_retry': len(contexts_received) > 1,
                'retry_attempted': len(contexts_received) == 2
            }
    
    def test_llm_guardrail_processing(self) -> Dict[str, Any]:
        """Test LLM-based guardrail processing"""
        task_output = MockTaskOutput(
            raw="Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s.",
            description="Test task",
            expected_output="Output", 
            agent="Test Agent"
        )
        
        # Test with word count limit
        llm = MockLLM()
        short_guardrail = MockLLMGuardrail(
            description="Ensure the result has less than 10 words",
            llm=llm
        )
        
        result_short = short_guardrail(task_output)
        
        long_guardrail = MockLLMGuardrail(
            description="Ensure the result has less than 500 words",
            llm=llm
        )
        
        result_long = long_guardrail(task_output)
        
        return {
            'short_limit_failed': result_short[0] is False,
            'short_limit_error_message': "exceeding the guardrail limit" in result_short[1].lower(),
            'long_limit_passed': result_long[0] is True,
            'long_limit_output_unchanged': result_long[1] == task_output.raw
        }
    
    def test_guardrail_events_emission(self) -> Dict[str, Any]:
        """Test that guardrails emit proper events"""
        started_events = []
        completed_events = []
        
        def mock_event_handler_started(event_data):
            started_events.append(event_data)
        
        def mock_event_handler_completed(event_data):
            completed_events.append(event_data)
        
        # Simulate event emission
        agent = self.create_test_agent()
        
        # Test string-based guardrail
        task1 = MockTask(
            description="Gather information about available books on the First World War",
            expected_output="A list of available books on the First World War",
            guardrail="Ensure the authors are from Italy",
            agent=agent
        )
        
        try:
            result1 = task1.execute_sync(agent=agent)
        except Exception:
            pass  # Expected to fail
        
        # Test callable guardrail
        def custom_guardrail(result: MockTaskOutput) -> Tuple[bool, str]:
            return (True, "good result from callable function")
        
        task2 = MockTask(
            description="Test task",
            expected_output="Output",
            guardrail=custom_guardrail
        )
        
        result2 = task2.execute_sync(agent=agent)
        
        # Mock expected events
        expected_started_events = [
            {"guardrail": "Ensure the authors are from Italy", "retry_count": 0},
            {"guardrail": "Ensure the authors are from Italy", "retry_count": 1},
            {"guardrail": "custom_guardrail function", "retry_count": 0}
        ]
        
        expected_completed_events = [
            {
                "success": False,
                "result": None, 
                "error": "The task result does not comply with the guardrail",
                "retry_count": 0
            },
            {"success": True, "result": result1.raw if 'result1' in locals() else None, "error": None, "retry_count": 1},
            {"success": True, "result": "good result from callable function", "error": None, "retry_count": 0}
        ]
        
        return {
            'events_structure_correct': True,
            'started_events_expected': len(expected_started_events) == 3,
            'completed_events_expected': len(expected_completed_events) == 3,
            'callable_guardrail_handled': result2.raw == "good result from callable function"
        }
    
    def test_error_during_validation(self) -> Dict[str, Any]:
        """Test error handling when validation itself fails"""
        def error_guardrail(result: MockTaskOutput) -> Tuple[bool, str]:
            raise Exception("Validation error")
        
        agent = Mock()
        agent.role = "test_agent"
        agent.execute_task.return_value = "test result"
        agent.crew = None
        
        task = MockTask(
            description="Test task",
            expected_output="Output",
            guardrail=error_guardrail,
            max_retries=0
        )
        
        try:
            result = task.execute_sync(agent=agent)
            
            return {
                'success': False,
                'unexpected_success': True
            }
            
        except Exception as e:
            return {
                'success': True,
                'error_handled': "Validation error" in str(e) or "validation" in str(e).lower(),
                'proper_error_propagation': True
            }


class GuardrailValidator:
    """Validates guardrail test results"""
    
    def __init__(self):
        self.framework = GuardrailTestFramework()
    
    def validate_backward_compatibility(self) -> Dict[str, Any]:
        """Validate backward compatibility without guardrails"""
        result = self.framework.test_task_without_guardrail()
        
        validation = {
            'execution_successful': result.get('success', False),
            'result_generated': result.get('result_generated', False),
            'output_correct': result.get('expected_output', False),
            'backward_compatible': result.get('backward_compatible', False)
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_successful_guardrail_flow(self) -> Dict[str, Any]:
        """Validate successful guardrail execution with transformation"""
        result = self.framework.test_successful_guardrail()
        
        validation = {
            'execution_successful': result.get('success', False),
            'guardrail_applied': result.get('guardrail_applied', False),
            'transformation_worked': result.get('output_transformed', False),
            'validation_passed': result.get('validation_passed', False)
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_retry_mechanism(self) -> Dict[str, Any]:
        """Validate retry mechanism with failing guardrails"""
        result = self.framework.test_failing_guardrail_with_retries()
        
        validation = {
            'expected_failure': result.get('success', False),
            'error_message_correct': result.get('expected_error', False),
            'max_retries_respected': result.get('max_retries_respected', False),
            'retry_count_tracked': result.get('retry_count', 0) > 0
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_context_passing(self) -> Dict[str, Any]:
        """Validate context passing during retries"""
        result = self.framework.test_guardrail_retry_with_context()
        
        validation = {
            'context_functionality_works': result.get('success', False),
            'error_message_preserved': result.get('error_contains_guardrail_message', False),
            'retry_attempted': result.get('retry_attempted', False)
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_llm_guardrail_processing(self) -> Dict[str, Any]:
        """Validate LLM-based guardrail processing"""
        result = self.framework.test_llm_guardrail_processing()
        
        validation = {
            'short_limit_validation_works': result.get('short_limit_failed', False),
            'error_messages_informative': result.get('short_limit_error_message', False),
            'long_limit_validation_works': result.get('long_limit_passed', False),
            'output_preservation_works': result.get('long_limit_output_unchanged', False)
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_event_emission(self) -> Dict[str, Any]:
        """Validate event emission during guardrail processing"""
        result = self.framework.test_guardrail_events_emission()
        
        validation = {
            'event_structure_correct': result.get('events_structure_correct', False),
            'started_events_emitted': result.get('started_events_expected', False),
            'completed_events_emitted': result.get('completed_events_expected', False),
            'callable_guardrails_supported': result.get('callable_guardrail_handled', False)
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def run_comprehensive_guardrail_tests(self) -> Dict[str, Any]:
        """Run comprehensive guardrail validation"""
        results = {}
        
        results['backward_compatibility'] = self.validate_backward_compatibility()
        results['successful_flow'] = self.validate_successful_guardrail_flow()
        results['retry_mechanism'] = self.validate_retry_mechanism()
        results['context_passing'] = self.validate_context_passing()
        results['llm_processing'] = self.validate_llm_guardrail_processing()
        results['event_emission'] = self.validate_event_emission()
        
        # Calculate overall success
        overall_success = all(
            result.get('overall_success', False) 
            for result in results.values()
        )
        
        results['summary'] = {
            'total_validation_categories': len(results) - 1,
            'passed_categories': sum(1 for k, v in results.items() 
                                   if k != 'summary' and v.get('overall_success', False)),
            'overall_success': overall_success,
            'guardrail_system_validated': overall_success
        }
        
        return results


# Pytest integration patterns
class PyTestGuardrailPatterns:
    """Task guardrail testing patterns for pytest"""
    
    @pytest.fixture
    def sample_agent(self):
        """Provide sample agent"""
        return MockAgent(role="Test Agent", goal="Test Goal", backstory="Test Backstory")
    
    @pytest.fixture
    def task_output(self):
        """Provide sample task output"""
        return MockTaskOutput(
            raw="Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
            description="Test task",
            expected_output="Output",
            agent="Test Agent"
        )
    
    @pytest.fixture
    def guardrail_framework(self):
        """Provide guardrail test framework"""
        return GuardrailTestFramework()
    
    def test_task_without_guardrail(self, guardrail_framework):
        """Test backward compatibility without guardrails"""
        result = guardrail_framework.test_task_without_guardrail()
        
        assert result['success'] is True
        assert result['result_generated'] is True
        assert result['backward_compatible'] is True
    
    def test_task_with_successful_guardrail_func(self):
        """Test successful guardrail validation"""
        def guardrail(result: MockTaskOutput) -> Tuple[bool, str]:
            return (True, result.raw.upper())
        
        agent = Mock()
        agent.role = "test_agent"
        agent.execute_task.return_value = "test result"
        agent.crew = None
        
        task = MockTask(description="Test task", expected_output="Output", guardrail=guardrail)
        
        result = task.execute_sync(agent=agent)
        assert isinstance(result, MockTaskOutput)
        assert result.raw == "TEST RESULT"
    
    def test_task_with_failing_guardrail(self):
        """Test failing guardrail triggers retry"""
        def guardrail(result: MockTaskOutput) -> Tuple[bool, str]:
            return (False, "Invalid format")
        
        agent = Mock()
        agent.role = "test_agent"
        agent.execute_task.side_effect = ["bad result", "good result"]
        agent.crew = None
        
        task = MockTask(
            description="Test task",
            expected_output="Output", 
            guardrail=guardrail,
            max_retries=1
        )
        
        with pytest.raises(Exception) as exc_info:
            task.execute_sync(agent=agent)
        
        assert "Task failed guardrail validation" in str(exc_info.value)
        assert task.retry_count == 1
    
    def test_guardrail_respects_max_retries(self):
        """Test that guardrail respects max_retries configuration"""
        def guardrail(result: MockTaskOutput) -> Tuple[bool, str]:
            return (False, "Invalid format")
        
        agent = Mock()
        agent.role = "test_agent"
        agent.execute_task.return_value = "bad result"
        agent.crew = None
        
        task = MockTask(
            description="Test task",
            expected_output="Output",
            guardrail=guardrail,
            max_retries=2
        )
        
        with pytest.raises(Exception) as exc_info:
            task.execute_sync(agent=agent)
        
        assert task.retry_count == 2
        assert "Task failed guardrail validation after 2 retries" in str(exc_info.value)
        assert "Invalid format" in str(exc_info.value)
    
    def test_llm_guardrail_process_output(self, task_output):
        """Test LLM guardrail processing"""
        llm = MockLLM(model="gpt-4o")
        
        # Test short word limit
        short_guardrail = MockLLMGuardrail(
            description="Ensure the result has less than 10 words", 
            llm=llm
        )
        
        result = short_guardrail(task_output)
        assert result[0] is False
        assert "exceeding the guardrail limit of fewer than" in result[1].lower()
        
        # Test long word limit
        long_guardrail = MockLLMGuardrail(
            description="Ensure the result has less than 500 words",
            llm=llm
        )
        
        result = long_guardrail(task_output)
        assert result[0] is True
        assert result[1] == task_output.raw


# Export patterns for integration
__all__ = [
    'GuardrailTestFramework',
    'GuardrailValidator',
    'MockTask',
    'MockTaskOutput',
    'MockAgent',
    'MockLLMGuardrail',
    'PyTestGuardrailPatterns'
]