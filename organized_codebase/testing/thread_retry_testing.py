"""Thread Retry Testing Framework - Agency-Swarm Pattern
Extracted patterns for testing thread retry mechanisms
Supports rate limiting, error recovery, and backoff strategies
"""
import time
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, MagicMock, patch
import pytest


class RetryManager:
    """Manages retry logic for thread operations"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_count = 0
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if operation should be retried"""
        if attempt >= self.max_retries:
            return False
        
        # Check for retryable errors
        error_msg = str(error).lower()
        retryable_conditions = [
            'rate limit',
            'timeout',
            'connection error',
            'service unavailable',
            'temporarily unavailable'
        ]
        
        return any(condition in error_msg for condition in retryable_conditions)
    
    def calculate_delay(self, attempt: int, error: Exception = None) -> float:
        """Calculate delay before retry"""
        # Extract delay from rate limit messages
        if error and 'rate limit' in str(error).lower():
            delay = self._extract_rate_limit_delay(str(error))
            if delay:
                return delay
        
        # Exponential backoff
        return self.base_delay * (2 ** attempt)
    
    def _extract_rate_limit_delay(self, error_message: str) -> Optional[float]:
        """Extract delay time from rate limit error message"""
        import re
        
        # Look for patterns like "Try again in X seconds"
        patterns = [
            r'try again in (\d+) seconds?',
            r'retry after (\d+) seconds?',
            r'wait (\d+) seconds?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message.lower())
            if match:
                return float(match.group(1))
        
        return None


class MockThreadError:
    """Mock thread error for testing"""
    
    def __init__(self, message: str):
        self.message = message


class MockThread:
    """Mock thread for retry testing"""
    
    def __init__(self):
        self.client = Mock()
        self.id = "test_thread_id"
        self._thread = Mock()
        self._run = Mock()
        self._run.id = "test_run_id"
        self._create_run = Mock()
        self.error_simulation = None
        self.call_count = 0
    
    def simulate_error(self, error_message: str):
        """Simulate an error condition"""
        self._run.last_error = MockThreadError(error_message)
        self.error_simulation = error_message
    
    def clear_error(self):
        """Clear error simulation"""
        self._run.last_error = None
        self.error_simulation = None


class RetryTestFramework:
    """Framework for testing retry mechanisms"""
    
    def __init__(self):
        self.test_results = []
        self.sleep_calls = []
    
    def test_rate_limit_retry(self, thread: MockThread, 
                            rate_limit_message: str = "Rate limit is exceeded. Try again in 2 seconds.") -> Dict[str, Any]:
        """Test rate limit retry behavior"""
        # Setup thread with rate limit error
        thread.simulate_error(rate_limit_message)
        
        retry_manager = RetryManager()
        
        # Mock time.sleep to track calls
        with patch('time.sleep') as mock_sleep:
            result = self._try_run_failed_recovery(
                thread=thread,
                retry_manager=retry_manager,
                error_attempts=0
            )
            
            # Check if retry was attempted
            sleep_called = mock_sleep.called
            sleep_duration = None
            if sleep_called:
                sleep_duration = mock_sleep.call_args[0][0]
            
            return {
                'success': result,
                'sleep_called': sleep_called,
                'sleep_duration': sleep_duration,
                'create_run_called': thread._create_run.called
            }
    
    def test_non_retryable_error(self, thread: MockThread) -> Dict[str, Any]:
        """Test behavior with non-retryable errors"""
        # Setup thread with non-retryable error
        thread.simulate_error("Authentication failed")
        
        retry_manager = RetryManager()
        
        with patch('time.sleep') as mock_sleep:
            result = self._try_run_failed_recovery(
                thread=thread,
                retry_manager=retry_manager,
                error_attempts=0
            )
            
            return {
                'success': result,
                'sleep_called': mock_sleep.called,
                'create_run_called': thread._create_run.called
            }
    
    def test_max_retries_exceeded(self, thread: MockThread) -> Dict[str, Any]:
        """Test behavior when max retries are exceeded"""
        thread.simulate_error("Rate limit is exceeded. Try again in 1 second.")
        
        retry_manager = RetryManager(max_retries=2)
        
        with patch('time.sleep') as mock_sleep:
            result = self._try_run_failed_recovery(
                thread=thread,
                retry_manager=retry_manager,
                error_attempts=3  # Exceeds max retries
            )
            
            return {
                'success': result,
                'sleep_called': mock_sleep.called,
                'create_run_called': thread._create_run.called
            }
    
    def test_exponential_backoff(self, thread: MockThread) -> Dict[str, Any]:
        """Test exponential backoff delay calculation"""
        retry_manager = RetryManager(base_delay=1.0)
        
        delays = []
        for attempt in range(5):
            delay = retry_manager.calculate_delay(attempt)
            delays.append(delay)
        
        return {
            'delays': delays,
            'exponential_pattern': all(delays[i] <= delays[i+1] for i in range(len(delays)-1))
        }
    
    def test_custom_delay_extraction(self, thread: MockThread) -> Dict[str, Any]:
        """Test custom delay extraction from error messages"""
        retry_manager = RetryManager()
        
        test_messages = [
            "Rate limit exceeded. Try again in 5 seconds.",
            "Service unavailable. Retry after 10 seconds.",
            "Please wait 3 seconds before retrying.",
            "No delay information available"
        ]
        
        extracted_delays = []
        for message in test_messages:
            error = Exception(message)
            delay = retry_manager.calculate_delay(0, error)
            extracted_delays.append(delay)
        
        return {
            'test_messages': test_messages,
            'extracted_delays': extracted_delays,
            'custom_extraction_working': any(d >= 3.0 for d in extracted_delays[:3])
        }
    
    def _try_run_failed_recovery(self, thread: MockThread, retry_manager: RetryManager, 
                               error_attempts: int) -> bool:
        """Simulate thread retry recovery logic"""
        if not thread._run.last_error:
            return False
        
        error_message = thread._run.last_error.message
        error = Exception(error_message)
        
        # Check if should retry
        if not retry_manager.should_retry(error, error_attempts):
            return False
        
        # Calculate and apply delay
        delay = retry_manager.calculate_delay(error_attempts, error)
        time.sleep(delay)
        
        # Attempt to create new run
        thread._create_run()
        
        return True


class RetryTestValidator:
    """Validates retry test behavior and results"""
    
    def __init__(self):
        self.framework = RetryTestFramework()
    
    def validate_rate_limit_handling(self) -> Dict[str, Any]:
        """Validate rate limit retry handling"""
        thread = MockThread()
        
        result = self.framework.test_rate_limit_retry(thread)
        
        validation = {
            'retry_attempted': result['success'],
            'sleep_called': result['sleep_called'],
            'correct_delay': result.get('sleep_duration') == 2.0,
            'run_created': result['create_run_called']
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_non_retryable_handling(self) -> Dict[str, Any]:
        """Validate non-retryable error handling"""
        thread = MockThread()
        
        result = self.framework.test_non_retryable_error(thread)
        
        validation = {
            'no_retry_attempted': not result['success'],
            'no_sleep_called': not result['sleep_called'],
            'no_run_created': not result['create_run_called']
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_backoff_strategy(self) -> Dict[str, Any]:
        """Validate exponential backoff strategy"""
        thread = MockThread()
        
        result = self.framework.test_exponential_backoff(thread)
        
        validation = {
            'has_delays': len(result['delays']) > 0,
            'exponential_pattern': result['exponential_pattern'],
            'reasonable_delays': all(0.5 <= d <= 32.0 for d in result['delays'])
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_delay_extraction(self) -> Dict[str, Any]:
        """Validate custom delay extraction"""
        thread = MockThread()
        
        result = self.framework.test_custom_delay_extraction(thread)
        
        validation = {
            'extraction_working': result['custom_extraction_working'],
            'correct_delays': result['extracted_delays'][:3] == [5.0, 10.0, 3.0] or
                           any(d in [5.0, 10.0, 3.0] for d in result['extracted_delays'][:3])
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def run_comprehensive_retry_tests(self) -> Dict[str, Any]:
        """Run comprehensive retry mechanism tests"""
        results = {}
        
        results['rate_limit'] = self.validate_rate_limit_handling()
        results['non_retryable'] = self.validate_non_retryable_handling() 
        results['backoff_strategy'] = self.validate_backoff_strategy()
        results['delay_extraction'] = self.validate_delay_extraction()
        
        # Calculate overall success
        overall_success = all(
            result.get('overall_success', False) 
            for result in results.values()
        )
        
        results['summary'] = {
            'total_tests': len(results) - 1,  # Exclude summary itself
            'passed_tests': sum(1 for k, v in results.items() 
                              if k != 'summary' and v.get('overall_success', False)),
            'overall_success': overall_success
        }
        
        return results


class RetryPatternBuilder:
    """Builds common retry patterns for reuse"""
    
    @staticmethod
    def create_rate_limit_pattern(delay_seconds: float = 2.0) -> Dict[str, Any]:
        """Create rate limit retry pattern"""
        return {
            'error_message': f"Rate limit is exceeded. Try again in {int(delay_seconds)} seconds.",
            'expected_delay': delay_seconds,
            'retryable': True
        }
    
    @staticmethod
    def create_timeout_pattern(timeout_seconds: float = 5.0) -> Dict[str, Any]:
        """Create timeout retry pattern"""
        return {
            'error_message': f"Request timeout after {timeout_seconds} seconds",
            'expected_delay': 1.0,  # Default exponential backoff
            'retryable': True
        }
    
    @staticmethod
    def create_auth_error_pattern() -> Dict[str, Any]:
        """Create authentication error pattern (non-retryable)"""
        return {
            'error_message': "Authentication failed - invalid credentials",
            'expected_delay': None,
            'retryable': False
        }
    
    @staticmethod
    def create_service_unavailable_pattern() -> Dict[str, Any]:
        """Create service unavailable pattern"""
        return {
            'error_message': "Service temporarily unavailable",
            'expected_delay': 1.0,
            'retryable': True
        }


# Pytest integration patterns
class PyTestRetryPatterns:
    """Retry testing patterns for pytest"""
    
    @pytest.fixture
    def mock_thread(self):
        """Create mock thread for retry testing"""
        return MockThread()
    
    @pytest.fixture  
    def retry_framework(self):
        """Create retry test framework"""
        return RetryTestFramework()
    
    def test_rate_limit_retry_mechanism(self, mock_thread, retry_framework, monkeypatch):
        """Test rate limit retry with proper delay"""
        # Setup rate limit error
        mock_thread.simulate_error("Rate limit is exceeded. Try again in 2 seconds.")
        
        # Track sleep calls
        sleep_calls = []
        def fake_sleep(seconds):
            sleep_calls.append(seconds)
        monkeypatch.setattr(time, "sleep", fake_sleep)
        
        # Test retry
        result = retry_framework.test_rate_limit_retry(mock_thread)
        
        assert result['success'] is True
        assert len(sleep_calls) > 0
        assert sleep_calls[0] == 2.0
        assert mock_thread._create_run.called
    
    def test_non_retryable_error_handling(self, mock_thread, retry_framework):
        """Test that non-retryable errors don't trigger retries"""
        mock_thread.simulate_error("Authentication failed")
        
        result = retry_framework.test_non_retryable_error(mock_thread)
        
        assert result['success'] is False
        assert result['sleep_called'] is False
        assert result['create_run_called'] is False
    
    def test_exponential_backoff_calculation(self, retry_framework):
        """Test exponential backoff delay calculation"""
        result = retry_framework.test_exponential_backoff(MockThread())
        
        delays = result['delays']
        assert len(delays) >= 5
        assert delays[0] == 1.0  # Base delay
        assert delays[1] == 2.0  # 1 * 2^1
        assert delays[2] == 4.0  # 1 * 2^2
        assert result['exponential_pattern'] is True


# Export patterns for integration
__all__ = [
    'RetryManager',
    'RetryTestFramework',
    'RetryTestValidator',
    'RetryPatternBuilder',
    'MockThread',
    'PyTestRetryPatterns'
]