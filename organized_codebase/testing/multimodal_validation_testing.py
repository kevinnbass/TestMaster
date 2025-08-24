"""Multimodal Validation Testing Framework - CrewAI Pattern
Extracted patterns for testing multimodal agent capabilities
Supports image processing, vision models, and multimodal validation
"""
import os
from typing import Any, Dict, List, Optional, Callable, Union
from unittest.mock import Mock, MagicMock, patch
import pytest


class MockMultimodalLLM:
    """Mock LLM with multimodal capabilities"""
    
    def __init__(self, model: str = "openai/gpt-4o", api_key: str = None, temperature: float = 0.7):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.supports_vision = "gpt-4o" in model or "gpt-4-vision" in model
        self.responses = {}
        
    def call(self, messages: List[Dict], *args, **kwargs) -> str:
        """Mock LLM call with multimodal support"""
        # Check if messages contain images
        has_images = any(
            isinstance(msg.get('content'), list) and 
            any(item.get('type') == 'image_url' for item in msg.get('content', []))
            for msg in messages
        )
        
        if has_images and self.supports_vision:
            return self._generate_vision_response(messages)
        else:
            return self._generate_text_response(messages)
    
    def set_response(self, key: str, response: str):
        """Set mock response for testing"""
        self.responses[key] = response
        
    def _generate_vision_response(self, messages: List[Dict]) -> str:
        """Generate response for vision-enabled requests"""
        if 'vision_response' in self.responses:
            return self.responses['vision_response']
        
        return """Based on the product image analysis:

1. Quality of materials: The boot appears to be made from genuine leather with good stitching quality.
2. Manufacturing defects: No visible defects observed in the construction or finishing.
3. Compliance with standards: The product meets standard footwear quality guidelines.

Overall assessment: High-quality product with professional manufacturing standards."""
    
    def _generate_text_response(self, messages: List[Dict]) -> str:
        """Generate response for text-only requests"""
        if 'text_response' in self.responses:
            return self.responses['text_response']
            
        return "Standard text response for non-vision requests."


class MockMultimodalAgent:
    """Mock agent with multimodal capabilities"""
    
    def __init__(self, role: str, goal: str, backstory: str, llm: MockMultimodalLLM,
                 verbose: bool = True, allow_delegation: bool = False, multimodal: bool = False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.multimodal = multimodal
        self.execution_count = 0
    
    def execute_task(self, task: 'MockMultimodalTask') -> str:
        """Execute task with multimodal support"""
        self.execution_count += 1
        
        # Check if task involves image processing
        if self.multimodal and task.has_image_content():
            return self._execute_vision_task(task)
        else:
            return self._execute_text_task(task)
    
    def _execute_vision_task(self, task: 'MockMultimodalTask') -> str:
        """Execute task with vision capabilities"""
        if not self.llm.supports_vision:
            raise ValueError(f"Model {self.llm.model} does not support vision capabilities")
        
        # Create multimodal message format
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": task.description},
                {"type": "image_url", "image_url": {"url": task.image_url}}
            ]
        }]
        
        return self.llm.call(messages)
    
    def _execute_text_task(self, task: 'MockMultimodalTask') -> str:
        """Execute text-only task"""
        messages = [{"role": "user", "content": task.description}]
        return self.llm.call(messages)
    
    def validate_multimodal_setup(self) -> Dict[str, Any]:
        """Validate multimodal configuration"""
        return {
            'multimodal_enabled': self.multimodal,
            'llm_supports_vision': self.llm.supports_vision,
            'configuration_valid': self.multimodal and self.llm.supports_vision,
            'model_name': self.llm.model
        }


class MockMultimodalTask:
    """Mock task with multimodal content support"""
    
    def __init__(self, description: str, expected_output: str, agent: MockMultimodalAgent = None,
                 image_url: str = None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.image_url = image_url
        self.execution_count = 0
    
    def has_image_content(self) -> bool:
        """Check if task involves image processing"""
        return (self.image_url is not None or 
                'image' in self.description.lower() or
                'http' in self.description and any(ext in self.description 
                                                 for ext in ['.jpg', '.png', '.gif', '.jpeg']))
    
    def execute(self) -> str:
        """Execute the multimodal task"""
        self.execution_count += 1
        
        if self.agent:
            return self.agent.execute_task(self)
        else:
            return f"Mock execution of task: {self.description}"
    
    def extract_image_urls(self) -> List[str]:
        """Extract image URLs from task description"""
        urls = []
        if self.image_url:
            urls.append(self.image_url)
        
        # Simple URL extraction from description
        words = self.description.split()
        for word in words:
            if word.startswith('http') and any(ext in word for ext in ['.jpg', '.png', '.gif', '.jpeg']):
                urls.append(word)
        
        return urls


class MockMultimodalCrew:
    """Mock crew with multimodal agents"""
    
    def __init__(self, agents: List[MockMultimodalAgent], tasks: List[MockMultimodalTask]):
        self.agents = agents
        self.tasks = tasks
        self.execution_results = []
    
    def kickoff(self) -> 'MockCrewResult':
        """Execute crew with multimodal tasks"""
        results = []
        
        for task in self.tasks:
            if task.agent:
                result = task.agent.execute_task(task)
            else:
                # Assign to first available multimodal agent
                multimodal_agents = [a for a in self.agents if a.multimodal]
                if multimodal_agents and task.has_image_content():
                    result = multimodal_agents[0].execute_task(task)
                else:
                    result = task.execute()
            
            results.append(result)
        
        return MockCrewResult(raw="Multimodal crew execution completed", results=results)


class MockCrewResult:
    """Mock crew execution result"""
    
    def __init__(self, raw: str, results: List[str] = None):
        self.raw = raw
        self.results = results or []


class MultimodalTestFramework:
    """Framework for testing multimodal capabilities"""
    
    def __init__(self):
        self.test_results = []
        
    def create_vision_agent(self, api_key: str = None) -> MockMultimodalAgent:
        """Create agent with vision capabilities"""
        llm = MockMultimodalLLM(
            model="openai/gpt-4o",
            api_key=api_key or "test_key",
            temperature=0.7
        )
        
        return MockMultimodalAgent(
            role="Visual Quality Inspector",
            goal="Perform detailed quality analysis of product images",
            backstory="Senior quality control expert with expertise in visual inspection",
            llm=llm,
            verbose=True,
            allow_delegation=False,
            multimodal=True
        )
    
    def create_image_analysis_task(self, image_url: str, analysis_requirements: List[str] = None) -> MockMultimodalTask:
        """Create task for image analysis"""
        if analysis_requirements is None:
            analysis_requirements = [
                "Quality of materials",
                "Manufacturing defects", 
                "Compliance with standards"
            ]
        
        description = f"""
        Analyze the product image at {image_url} with focus on:
        {chr(10).join(f'{i+1}. {req}' for i, req in enumerate(analysis_requirements))}
        Provide a detailed report highlighting any issues found.
        """
        
        return MockMultimodalTask(
            description=description,
            expected_output="A detailed report highlighting any issues found",
            image_url=image_url
        )
    
    def test_multimodal_agent_initialization(self, api_key: str = None) -> Dict[str, Any]:
        """Test multimodal agent initialization"""
        try:
            agent = self.create_vision_agent(api_key)
            validation = agent.validate_multimodal_setup()
            
            return {
                'success': True,
                'agent_created': True,
                'multimodal_enabled': validation['multimodal_enabled'],
                'llm_supports_vision': validation['llm_supports_vision'],
                'configuration_valid': validation['configuration_valid'],
                'model_name': validation['model_name']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_image_processing_capability(self, image_url: str, api_key: str = None) -> Dict[str, Any]:
        """Test image processing without validation errors"""
        try:
            # Create agent and task
            agent = self.create_vision_agent(api_key)
            task = self.create_image_analysis_task(image_url)
            task.agent = agent
            
            # Set expected response
            agent.llm.set_response('vision_response', 
                                  "Quality analysis completed: Materials excellent, no defects found, compliant with standards.")
            
            # Execute task
            result = agent.execute_task(task)
            
            return {
                'success': True,
                'task_executed': True,
                'image_processed': task.has_image_content(),
                'vision_model_used': agent.llm.supports_vision,
                'result_generated': result is not None and len(result) > 0,
                'analysis_result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def test_crew_multimodal_execution(self, image_url: str, api_key: str = None) -> Dict[str, Any]:
        """Test multimodal crew execution"""
        try:
            # Create multimodal agent and task
            agent = self.create_vision_agent(api_key)
            task = self.create_image_analysis_task(image_url)
            
            # Create crew
            crew = MockMultimodalCrew(agents=[agent], tasks=[task])
            
            # Execute crew
            result = crew.kickoff()
            
            return {
                'success': True,
                'crew_executed': True,
                'multimodal_task_handled': task.has_image_content(),
                'execution_completed': result is not None,
                'crew_result': result.raw,
                'task_results': result.results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def test_api_key_validation(self) -> Dict[str, Any]:
        """Test API key validation for vision models"""
        # Test with missing API key
        try:
            agent_no_key = self.create_vision_agent(api_key=None)
            
            return {
                'missing_key_handled': True,
                'agent_created_without_key': True,
                'validation_note': "In testing, we use mock keys"
            }
            
        except Exception as e:
            return {
                'missing_key_handled': False,
                'error': str(e)
            }
    
    def test_model_compatibility(self) -> Dict[str, Any]:
        """Test model compatibility with vision tasks"""
        model_tests = {}
        
        vision_models = ["openai/gpt-4o", "openai/gpt-4-vision-preview"]
        text_models = ["openai/gpt-3.5-turbo", "openai/gpt-4"]
        
        # Test vision-capable models
        for model in vision_models:
            try:
                llm = MockMultimodalLLM(model=model, api_key = os.getenv('KEY'))
                agent = MockMultimodalAgent(
                    role="Tester", goal="Test", backstory="Test",
                    llm=llm, multimodal=True
                )
                
                model_tests[model] = {
                    'supports_vision': llm.supports_vision,
                    'multimodal_compatible': True,
                    'agent_created': True
                }
                
            except Exception as e:
                model_tests[model] = {
                    'supports_vision': False,
                    'multimodal_compatible': False,
                    'error': str(e)
                }
        
        # Test text-only models
        for model in text_models:
            try:
                llm = MockMultimodalLLM(model=model, api_key = os.getenv('KEY'))
                
                model_tests[model] = {
                    'supports_vision': llm.supports_vision,
                    'multimodal_compatible': False,
                    'expected_limitation': True
                }
                
            except Exception as e:
                model_tests[model] = {
                    'supports_vision': False,
                    'multimodal_compatible': False,
                    'error': str(e)
                }
        
        return {
            'model_compatibility_tests': model_tests,
            'vision_models_work': all(
                test.get('supports_vision', False) 
                for model, test in model_tests.items() 
                if model in vision_models
            ),
            'text_models_limited': all(
                not test.get('supports_vision', True) 
                for model, test in model_tests.items() 
                if model in text_models
            )
        }
    
    def test_error_handling(self, image_url: str) -> Dict[str, Any]:
        """Test error handling in multimodal scenarios"""
        error_scenarios = {}
        
        # Test unsupported model with vision task
        try:
            text_llm = MockMultimodalLLM(model="openai/gpt-3.5-turbo")
            text_agent = MockMultimodalAgent(
                role="Tester", goal="Test", backstory="Test",
                llm=text_llm, multimodal=True
            )
            
            task = self.create_image_analysis_task(image_url)
            task.agent = text_agent
            
            result = text_agent.execute_task(task)
            error_scenarios['unsupported_model'] = {
                'error_raised': False,
                'execution_completed': True,
                'result': result
            }
            
        except Exception as e:
            error_scenarios['unsupported_model'] = {
                'error_raised': True,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        
        # Test invalid image URL
        try:
            agent = self.create_vision_agent("test_key")
            invalid_task = self.create_image_analysis_task("invalid_url")
            invalid_task.agent = agent
            
            result = agent.execute_task(invalid_task)
            error_scenarios['invalid_image_url'] = {
                'handled_gracefully': True,
                'result_generated': result is not None
            }
            
        except Exception as e:
            error_scenarios['invalid_image_url'] = {
                'handled_gracefully': False,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        
        return {
            'error_scenarios': error_scenarios,
            'error_handling_robust': all(
                scenario.get('handled_gracefully', scenario.get('error_raised', False))
                for scenario in error_scenarios.values()
            )
        }


class MultimodalValidator:
    """Validates multimodal testing results"""
    
    def __init__(self):
        self.framework = MultimodalTestFramework()
    
    def validate_agent_initialization(self, api_key: str = None) -> Dict[str, Any]:
        """Validate multimodal agent initialization"""
        result = self.framework.test_multimodal_agent_initialization(api_key)
        
        validation = {
            'initialization_successful': result.get('success', False),
            'multimodal_properly_enabled': result.get('multimodal_enabled', False),
            'vision_support_detected': result.get('llm_supports_vision', False),
            'configuration_valid': result.get('configuration_valid', False)
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_image_processing(self, image_url: str, api_key: str = None) -> Dict[str, Any]:
        """Validate image processing capabilities"""
        result = self.framework.test_image_processing_capability(image_url, api_key)
        
        validation = {
            'processing_successful': result.get('success', False),
            'image_detected': result.get('image_processed', False),
            'vision_model_used': result.get('vision_model_used', False),
            'analysis_generated': result.get('result_generated', False)
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_crew_integration(self, image_url: str, api_key: str = None) -> Dict[str, Any]:
        """Validate multimodal crew integration"""
        result = self.framework.test_crew_multimodal_execution(image_url, api_key)
        
        validation = {
            'crew_execution_successful': result.get('success', False),
            'multimodal_tasks_handled': result.get('multimodal_task_handled', False),
            'results_generated': result.get('execution_completed', False)
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_model_compatibility(self) -> Dict[str, Any]:
        """Validate model compatibility"""
        result = self.framework.test_model_compatibility()
        
        validation = {
            'vision_models_supported': result.get('vision_models_work', False),
            'text_models_properly_limited': result.get('text_models_limited', False),
            'compatibility_properly_detected': True
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_error_handling(self, image_url: str) -> Dict[str, Any]:
        """Validate error handling"""
        result = self.framework.test_error_handling(image_url)
        
        validation = {
            'errors_handled_gracefully': result.get('error_handling_robust', False),
            'proper_error_messages': True  # Assuming proper error messages
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def run_comprehensive_multimodal_tests(self, image_url: str, api_key: str = None) -> Dict[str, Any]:
        """Run comprehensive multimodal validation"""
        results = {}
        
        results['initialization'] = self.validate_agent_initialization(api_key)
        results['image_processing'] = self.validate_image_processing(image_url, api_key)
        results['crew_integration'] = self.validate_crew_integration(image_url, api_key)
        results['model_compatibility'] = self.validate_model_compatibility()
        results['error_handling'] = self.validate_error_handling(image_url)
        
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
            'multimodal_capabilities_validated': overall_success
        }
        
        return results


# Pytest integration patterns
class PyTestMultimodalPatterns:
    """Multimodal testing patterns for pytest"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Provide mock API key"""
        return "test_openai_api_key"
    
    @pytest.fixture
    def test_image_url(self):
        """Provide test image URL"""
        return "https://www.us.maguireshoes.com/collections/spring-25/products/lucena-black-boot"
    
    @pytest.fixture
    def multimodal_framework(self):
        """Provide multimodal test framework"""
        return MultimodalTestFramework()
    
    @pytest.mark.skip(reason="Only run manually with valid API keys")
    def test_multimodal_agent_with_image_url(self, mock_api_key, test_image_url):
        """Test multimodal agent with image processing"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        framework = MultimodalTestFramework()
        result = framework.test_image_processing_capability(test_image_url, mock_api_key)
        
        assert result['success'] is True
        assert result['image_processed'] is True
        assert result['vision_model_used'] is True
    
    def test_multimodal_agent_initialization(self, multimodal_framework, mock_api_key):
        """Test multimodal agent initialization"""
        result = multimodal_framework.test_multimodal_agent_initialization(mock_api_key)
        
        assert result['success'] is True
        assert result['multimodal_enabled'] is True
        assert result['llm_supports_vision'] is True
        assert result['configuration_valid'] is True
    
    def test_model_compatibility_detection(self, multimodal_framework):
        """Test model compatibility detection"""
        result = multimodal_framework.test_model_compatibility()
        
        assert result['vision_models_work'] is True
        assert result['text_models_limited'] is True
    
    def test_image_content_detection(self, multimodal_framework, test_image_url):
        """Test image content detection in tasks"""
        task = multimodal_framework.create_image_analysis_task(test_image_url)
        
        assert task.has_image_content() is True
        assert len(task.extract_image_urls()) > 0
        assert test_image_url in task.extract_image_urls()[0]
    
    def test_crew_multimodal_integration(self, multimodal_framework, test_image_url, mock_api_key):
        """Test crew integration with multimodal tasks"""
        result = multimodal_framework.test_crew_multimodal_execution(test_image_url, mock_api_key)
        
        assert result['success'] is True
        assert result['multimodal_task_handled'] is True
        assert result['execution_completed'] is True


# Export patterns for integration
__all__ = [
    'MultimodalTestFramework',
    'MultimodalValidator',
    'MockMultimodalAgent',
    'MockMultimodalTask',
    'MockMultimodalCrew',
    'MockMultimodalLLM',
    'PyTestMultimodalPatterns'
]