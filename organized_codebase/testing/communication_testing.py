"""Communication Testing Framework - Agency-Swarm Pattern
Extracted patterns for agent-to-agent communication testing
Supports message routing, timeout handling, and error detection
"""
import time
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, MagicMock
from pydantic import Field, BaseModel
import pytest


class MockAgent:
    """Mock agent for communication testing"""
    
    def __init__(self, name: str, description: str = "", instructions: str = ""):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.tools = []
        self.message_history = []
    
    def receive_message(self, message: str, sender: Optional[str] = None):
        """Simulate receiving a message"""
        self.message_history.append({
            'message': message,
            'sender': sender,
            'timestamp': time.time()
        })
        return f"Agent {self.name} received: {message}"


class MockTool(BaseModel):
    """Mock tool for communication testing"""
    message: str = Field(..., description="The message to process")
    
    def run(self):
        """Execute mock tool"""
        return f"Tool processed: {self.message}"


class CommunicationTestFramework:
    """Framework for testing agent communication patterns"""
    
    def __init__(self):
        self.agencies = []
        self.test_results = []
        self.timeouts = {}
    
    def create_test_agency(self, agents: List[MockAgent]) -> 'MockAgency':
        """Create a mock agency for testing"""
        agency = MockAgency(agents)
        self.agencies.append(agency)
        return agency
    
    def test_message_routing(self, agency: 'MockAgency', message: str, 
                           expected_recipient: str, timeout: int = 30) -> Dict[str, Any]:
        """Test message routing between agents"""
        start_time = time.time()
        
        try:
            response = agency.get_completion(message, timeout=timeout)
            
            # Verify routing
            routing_success = expected_recipient.lower() in response.lower()
            
            return {
                'success': True,
                'response': response,
                'routing_correct': routing_success,
                'execution_time': time.time() - start_time,
                'recipient_agent': agency.get_current_recipient()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def test_timeout_handling(self, agency: 'MockAgency', timeout: int = 5) -> Dict[str, Any]:
        """Test timeout handling in communication"""
        start_time = time.time()
        
        try:
            # Simulate slow operation
            response = agency.get_completion_with_delay("Test timeout", delay=timeout+1, timeout=timeout)
            
            return {
                'success': False,  # Should have timed out
                'response': response,
                'execution_time': time.time() - start_time
            }
            
        except TimeoutError:
            return {
                'success': True,  # Expected timeout
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def test_error_detection(self, agency: 'MockAgency', error_pattern: str) -> Dict[str, Any]:
        """Test error detection in communication"""
        try:
            response = agency.get_completion("Generate double recipient error")
            
            # Check for error patterns
            error_detected = error_pattern.lower() in response.lower()
            fatal_error = "fatal" in response.lower()
            
            return {
                'success': error_detected and not fatal_error,
                'response': response,
                'error_detected': error_detected,
                'fatal_error': fatal_error
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class MockAgency:
    """Mock agency for communication testing"""
    
    def __init__(self, agents: List[MockAgent]):
        self.agents = agents
        self.main_thread = MockThread()
        self.current_recipient = None
        self.temperature = 0
    
    def get_completion(self, message: str, timeout: int = 30) -> str:
        """Simulate getting completion from agency"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Simulate message processing
                if "customer support" in message.lower():
                    self.current_recipient = self._find_agent("Customer Support")
                    if self.current_recipient:
                        return "I'll connect you to customer support right away."
                
                # Check for routing errors
                if "route" in message.lower() and "twice" in message.lower():
                    return "error"
                
                # Default response
                return f"Processed: {message}"
                
            except Exception as e:
                time.sleep(1)
                continue
        
        raise TimeoutError(f"Operation timed out after {timeout} seconds")
    
    def get_completion_with_delay(self, message: str, delay: int, timeout: int) -> str:
        """Simulate completion with artificial delay"""
        start_time = time.time()
        
        # Simulate processing delay
        time.sleep(min(delay, timeout + 1))
        
        if time.time() - start_time > timeout:
            raise TimeoutError("Operation exceeded timeout")
        
        return f"Delayed response: {message}"
    
    def _find_agent(self, name: str) -> Optional[MockAgent]:
        """Find agent by name"""
        for agent in self.agents:
            if agent.name.lower() == name.lower():
                return agent
        return None
    
    def get_current_recipient(self) -> Optional[MockAgent]:
        """Get current recipient agent"""
        return self.current_recipient


class MockThread:
    """Mock thread for communication testing"""
    
    def __init__(self):
        self.thread_url = "mock://thread/12345"
        self.recipient_agent = None
        self.messages = []
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get thread messages"""
        return self.messages
    
    def add_message(self, message: str, sender: str = "user"):
        """Add message to thread"""
        self.messages.append({
            'content': message,
            'sender': sender,
            'timestamp': time.time()
        })


class CommunicationTestValidator:
    """Validates communication test patterns"""
    
    def __init__(self):
        self.framework = CommunicationTestFramework()
    
    def run_basic_routing_test(self) -> Dict[str, Any]:
        """Test basic message routing"""
        # Create test agents
        ceo = MockAgent("CEO", "Handles routing", "Route messages to appropriate agents")
        support = MockAgent("Customer Support", "Handles support", "Answer customer questions")
        
        # Create agency
        agency = self.framework.create_test_agency([ceo, support])
        
        # Test routing
        result = self.framework.test_message_routing(
            agency, 
            "Hello, I need customer support please.",
            "customer support"
        )
        
        return result
    
    def run_timeout_test(self) -> Dict[str, Any]:
        """Test timeout handling"""
        ceo = MockAgent("CEO")
        agency = self.framework.create_test_agency([ceo])
        
        result = self.framework.test_timeout_handling(agency, timeout=2)
        return result
    
    def run_error_detection_test(self) -> Dict[str, Any]:
        """Test error detection"""
        ceo = MockAgent("CEO")
        support = MockAgent("Customer Support")
        agency = self.framework.create_test_agency([ceo, support])
        
        result = self.framework.test_error_detection(agency, "error")
        return result


# Pytest integration patterns
class PyTestCommunicationPatterns:
    """Communication testing patterns for pytest"""
    
    @pytest.fixture
    def test_agents(self):
        """Create test agents fixture"""
        ceo = MockAgent(
            name="CEO",
            description="Responsible for client communication, task planning and management.",
            instructions="Route messages to appropriate agents"
        )
        
        customer_support = MockAgent(
            name="Customer Support", 
            description="Responsible for customer support.",
            instructions="Answer customer questions and help with issues"
        )
        
        agency = MockAgency([ceo, customer_support])
        return ceo, customer_support, agency
    
    def test_send_message_routing(self, test_agents):
        """Test message routing functionality"""
        _, customer_support, agency = test_agents
        
        response = agency.get_completion("Hello, I need customer support please.")
        assert response is not None
        assert "error" not in response.lower()
        
        # Verify routing
        follow_up = agency.get_completion("Who are you?")
        assert "support" in follow_up.lower() or agency.current_recipient == customer_support
    
    def test_double_recipient_error(self, test_agents):
        """Test double recipient error detection"""
        _, _, agency = test_agents
        
        response = agency.get_completion(
            "Route me to customer support TWICE simultaneously. This is a test."
        )
        assert "error" in response.lower()
        assert "fatal" not in response.lower()


# Export patterns for integration
__all__ = [
    'CommunicationTestFramework',
    'MockAgent',
    'MockAgency', 
    'CommunicationTestValidator',
    'PyTestCommunicationPatterns'
]