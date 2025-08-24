"""Response Validation Testing Framework - Agency-Swarm Pattern
Extracted patterns for response validation with retry mechanisms
Supports custom validators, JSON validation, and content policy enforcement
"""
import json
import re
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock, MagicMock
from pydantic import BaseModel, Field, ValidationError
import pytest


class ResponseValidator:
    """Core response validation functionality"""
    
    def __init__(self, validator_func: Optional[Callable[[str], str]] = None, 
                 max_attempts: int = 3):
        self.validator_func = validator_func
        self.max_attempts = max_attempts
        self.validation_attempts = 0
    
    def validate(self, message: str, current_attempt: int = 0) -> Dict[str, Any]:
        """Validate response with retry logic"""
        if not self.validator_func:
            return {'valid': True, 'message': message}
        
        try:
            # Run validation
            validated_message = self.validator_func(message)
            return {
                'valid': True,
                'message': validated_message,
                'attempts': current_attempt
            }
            
        except Exception as e:
            # Check if we have retries left
            if current_attempt >= self.max_attempts:
                # No more retries, raise the exception
                raise e
            
            # Return retry information
            return {
                'valid': False,
                'retry': True,
                'error_message': str(e),
                'attempts': current_attempt + 1,
                'continue_loop': True
            }


class JSONResponseValidator:
    """Validates JSON format responses"""
    
    @staticmethod
    def create_json_validator(required_fields: List[str] = None) -> Callable[[str], str]:
        """Create a JSON validation function"""
        required_fields = required_fields or []
        
        def json_validator(message: str) -> str:
            try:
                data = json.loads(message)
                
                # Check if it's a dictionary
                if not isinstance(data, dict):
                    raise ValueError("Response must be a JSON object")
                
                # Check required fields
                for field in required_fields:
                    if field not in data:
                        raise ValueError(f"Response must include '{field}' field")
                
                return message
                
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format: {message}")
        
        return json_validator
    
    @staticmethod
    def create_structured_validator(schema_model: type) -> Callable[[str], str]:
        """Create validator using Pydantic model"""
        def pydantic_validator(message: str) -> str:
            try:
                data = json.loads(message)
                schema_model(**data)  # Validate against Pydantic model
                return message
            except json.JSONDecodeError:
                raise ValueError("Response must be valid JSON")
            except ValidationError as e:
                raise ValueError(f"Response validation failed: {str(e)}")
            except Exception as e:
                raise ValueError(f"Response validation failed: {str(e)}")
        
        return pydantic_validator


class ContentPolicyValidator:
    """Validates content against policy rules"""
    
    def __init__(self, forbidden_words: List[str] = None, 
                 required_patterns: List[str] = None):
        self.forbidden_words = forbidden_words or []
        self.required_patterns = required_patterns or []
    
    def create_content_validator(self) -> Callable[[str], str]:
        """Create content policy validation function"""
        def content_validator(message: str) -> str:
            message_lower = message.lower()
            
            # Check forbidden words
            for word in self.forbidden_words:
                if word.lower() in message_lower:
                    raise ValueError(
                        f"Response contains forbidden word: '{word}'. "
                        f"Please rephrase without sensitive information."
                    )
            
            # Check required patterns
            for pattern in self.required_patterns:
                if not re.search(pattern, message, re.IGNORECASE):
                    raise ValueError(
                        f"Response must include pattern: {pattern}"
                    )
            
            return message
        
        return content_validator


class MockValidationThread:
    """Mock thread for validation testing"""
    
    def __init__(self):
        self.client = Mock()
        self.id = "test_thread_id"
        self._thread = Mock()
        self._run = Mock()
        self._run.id = "test_run_id"
        self.create_message = Mock()
        self._create_run = Mock()
        self.messages = []
    
    def setup_message_response(self, response_text: str):
        """Setup mock message response"""
        mock_message = Mock()
        mock_message.content = [Mock()]
        mock_message.content[0].text = Mock()
        mock_message.content[0].text.value = response_text
        self.create_message.return_value = mock_message


class ValidationTestFramework:
    """Framework for testing response validation patterns"""
    
    def __init__(self):
        self.test_results = []
        self.mock_threads = []
    
    def create_mock_agent(self, validator_func: Optional[Callable] = None, 
                         max_attempts: int = 3) -> 'MockValidationAgent':
        """Create mock agent with validation"""
        agent = MockValidationAgent(validator_func, max_attempts)
        return agent
    
    def test_successful_validation(self, agent: 'MockValidationAgent', 
                                 message: str) -> Dict[str, Any]:
        """Test successful validation scenario"""
        try:
            result = agent.validate_response(message, attempt=0)
            
            return {
                'success': result.get('valid', False),
                'message': result.get('message'),
                'attempts': result.get('attempts', 0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_validation_retry(self, agent: 'MockValidationAgent', 
                            invalid_message: str, valid_message: str) -> Dict[str, Any]:
        """Test validation with retry logic"""
        results = []
        
        # First attempt with invalid message
        try:
            result1 = agent.validate_response(invalid_message, attempt=0)
            results.append(result1)
            
            # If retry is needed, try with valid message
            if result1.get('retry'):
                result2 = agent.validate_response(valid_message, 
                                                attempt=result1.get('attempts', 1))
                results.append(result2)
            
            return {
                'success': True,
                'retry_triggered': len(results) > 1,
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': results
            }
    
    def test_validation_exhaustion(self, agent: 'MockValidationAgent', 
                                 invalid_message: str) -> Dict[str, Any]:
        """Test validation when retries are exhausted"""
        try:
            # Attempt validation at max attempts
            result = agent.validate_response(invalid_message, 
                                          attempt=agent.max_attempts)
            
            return {
                'success': False,  # Should have raised exception
                'result': result
            }
            
        except Exception as e:
            return {
                'success': True,  # Expected exception
                'error_type': type(e).__name__,
                'error_message': str(e)
            }


class MockValidationAgent:
    """Mock agent with response validation"""
    
    def __init__(self, validator_func: Optional[Callable] = None, 
                 max_attempts: int = 3):
        self.response_validator = validator_func
        self.validation_attempts = max_attempts
        self.max_attempts = max_attempts
    
    def validate_response(self, message: str, attempt: int = 0) -> Dict[str, Any]:
        """Validate response using configured validator"""
        validator = ResponseValidator(self.response_validator, self.max_attempts)
        return validator.validate(message, attempt)


class ValidationTestScenarios:
    """Pre-built validation test scenarios"""
    
    @staticmethod
    def json_validation_scenario() -> Dict[str, Any]:
        """JSON validation test scenario"""
        # Create JSON validator
        json_validator = JSONResponseValidator.create_json_validator(['status'])
        agent = MockValidationAgent(json_validator, max_attempts=2)
        framework = ValidationTestFramework()
        
        # Test scenarios
        scenarios = {}
        
        # Valid JSON
        scenarios['valid'] = framework.test_successful_validation(
            agent, '{"status": "success", "data": "test"}'
        )
        
        # Invalid JSON
        scenarios['invalid_json'] = framework.test_validation_exhaustion(
            agent, '{"incomplete": json'
        )
        
        # Missing required field
        scenarios['missing_field'] = framework.test_validation_exhaustion(
            agent, '{"data": "test"}'
        )
        
        return scenarios
    
    @staticmethod
    def content_policy_scenario() -> Dict[str, Any]:
        """Content policy validation test scenario"""
        # Create content policy validator
        policy_validator = ContentPolicyValidator(
            forbidden_words=['password', 'secret', 'confidential']
        )
        content_validator = policy_validator.create_content_validator()
        
        agent = MockValidationAgent(content_validator, max_attempts=2)
        framework = ValidationTestFramework()
        
        # Test scenarios
        scenarios = {}
        
        # Valid content
        scenarios['valid'] = framework.test_successful_validation(
            agent, "Here is the public information you requested"
        )
        
        # Forbidden content
        scenarios['forbidden'] = framework.test_validation_exhaustion(
            agent, "The password is 12345"
        )
        
        return scenarios
    
    @staticmethod
    def pydantic_validation_scenario() -> Dict[str, Any]:
        """Pydantic model validation test scenario"""
        # Define response model
        class ResponseModel(BaseModel):
            action: str = Field(..., description="The action taken")
            result: str = Field(..., description="The result of the action")
            confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
        
        # Create Pydantic validator
        pydantic_validator = JSONResponseValidator.create_structured_validator(ResponseModel)
        agent = MockValidationAgent(pydantic_validator, max_attempts=2)
        framework = ValidationTestFramework()
        
        # Test scenarios
        scenarios = {}
        
        # Valid data
        scenarios['valid'] = framework.test_successful_validation(
            agent, '{"action": "test", "result": "success", "confidence": 0.95}'
        )
        
        # Invalid confidence range
        scenarios['invalid_confidence'] = framework.test_validation_exhaustion(
            agent, '{"action": "test", "result": "success", "confidence": 1.5}'
        )
        
        # Missing fields
        scenarios['missing_fields'] = framework.test_validation_exhaustion(
            agent, '{"action": "test"}'
        )
        
        return scenarios


class ValidationTestValidator:
    """Validates validation test execution"""
    
    def __init__(self):
        self.framework = ValidationTestFramework()
    
    def run_comprehensive_validation_tests(self) -> Dict[str, Any]:
        """Run comprehensive validation tests"""
        results = {}
        
        # JSON validation tests
        results['json_scenarios'] = ValidationTestScenarios.json_validation_scenario()
        
        # Content policy tests  
        results['content_scenarios'] = ValidationTestScenarios.content_policy_scenario()
        
        # Pydantic validation tests
        results['pydantic_scenarios'] = ValidationTestScenarios.pydantic_validation_scenario()
        
        return results
    
    def validate_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that test results meet expectations"""
        validation_summary = {
            'total_scenarios': 0,
            'passed_scenarios': 0,
            'failed_scenarios': 0,
            'details': {}
        }
        
        for category, scenarios in results.items():
            category_summary = {'passed': 0, 'failed': 0, 'total': len(scenarios)}
            
            for scenario_name, scenario_result in scenarios.items():
                validation_summary['total_scenarios'] += 1
                category_summary['total'] += 1
                
                if scenario_result.get('success', False):
                    validation_summary['passed_scenarios'] += 1
                    category_summary['passed'] += 1
                else:
                    validation_summary['failed_scenarios'] += 1
                    category_summary['failed'] += 1
            
            validation_summary['details'][category] = category_summary
        
        return validation_summary


# Pytest integration patterns
class PyTestValidationPatterns:
    """Response validation testing patterns for pytest"""
    
    @pytest.fixture
    def mock_thread(self):
        """Create mock thread for validation testing"""
        thread = MockValidationThread()
        yield thread
    
    @pytest.fixture
    def json_validator_agent(self):
        """Create agent with JSON validator"""
        json_validator = JSONResponseValidator.create_json_validator(['status'])
        return MockValidationAgent(json_validator, max_attempts=2)
    
    def test_json_validation_success(self, json_validator_agent):
        """Test successful JSON validation"""
        result = json_validator_agent.validate_response(
            '{"status": "success", "data": "test"}'
        )
        assert result['valid'] is True
        assert result['message'] == '{"status": "success", "data": "test"}'
    
    def test_json_validation_failure(self, json_validator_agent):
        """Test JSON validation failure"""
        with pytest.raises(ValueError, match="Invalid JSON format"):
            json_validator_agent.validate_response('{"incomplete": json', attempt=2)
    
    def test_content_policy_validation(self):
        """Test content policy validation"""
        policy_validator = ContentPolicyValidator(['password', 'secret'])
        content_validator = policy_validator.create_content_validator()
        agent = MockValidationAgent(content_validator, max_attempts=1)
        
        with pytest.raises(ValueError, match="forbidden word"):
            agent.validate_response("The password is 12345", attempt=1)


# Export patterns for integration
__all__ = [
    'ResponseValidator',
    'JSONResponseValidator', 
    'ContentPolicyValidator',
    'ValidationTestFramework',
    'MockValidationAgent',
    'ValidationTestScenarios',
    'ValidationTestValidator',
    'PyTestValidationPatterns'
]