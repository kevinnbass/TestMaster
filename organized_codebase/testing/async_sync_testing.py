"""Async/Sync Testing Framework - Agency-Swarm Pattern
Extracted patterns for handling mixed async/sync test execution
Supports runtime loop detection and proper asyncio management
"""
import asyncio
import functools
from typing import Any, Callable, Coroutine, Optional, Union
import pytest


class AsyncSyncTestManager:
    """Manages async/sync test execution patterns"""
    
    @staticmethod
    def run_async_sync(coro_func: Callable[..., Coroutine], *args, **kwargs) -> Any:
        """Execute async function in sync context with loop detection"""
        try:
            # Try to get existing loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call async function from within running loop")
        except RuntimeError:
            # No loop running, safe to create new one
            return asyncio.run(coro_func(*args, **kwargs))
        
    @staticmethod
    def create_async_test_wrapper(func: Callable) -> Callable:
        """Wraps function for async test execution"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            await asyncio.sleep(0)  # Yield control
            return await func(*args, **kwargs)
        return wrapper
    
    @staticmethod
    def validate_loop_state():
        """Validates current event loop state"""
        try:
            loop = asyncio.get_running_loop()
            return {
                'running': True,
                'closed': loop.is_closed(),
                'debug': loop.get_debug()
            }
        except RuntimeError:
            return {'running': False, 'closed': None, 'debug': None}


class AsyncTestFixtures:
    """Reusable async test fixtures and helpers"""
    
    @staticmethod
    async def async_add(a: int, b: int) -> int:
        """Simple async operation for testing"""
        await asyncio.sleep(0)
        return a + b
    
    @staticmethod
    async def async_multiply(a: int, b: int) -> int:
        """Another async operation for testing"""
        await asyncio.sleep(0.001)
        return a * b
    
    @staticmethod
    async def async_divide(a: int, b: int) -> float:
        """Async operation with potential error"""
        await asyncio.sleep(0.001)
        if b == 0:
            raise ValueError("Division by zero")
        return a / b


class AsyncSyncTestValidator:
    """Validates proper async/sync test behavior"""
    
    def __init__(self):
        self.test_results = []
    
    def test_basic_async_sync_execution(self):
        """Test basic async function execution in sync context"""
        manager = AsyncSyncTestManager()
        fixtures = AsyncTestFixtures()
        
        # Test successful execution
        result = manager.run_async_sync(fixtures.async_add, 1, 2)
        assert result == 3
        self.test_results.append(("async_sync_basic", True))
        
        return result
    
    def test_running_loop_detection(self):
        """Test detection of running event loop"""
        manager = AsyncSyncTestManager()
        
        async def test_in_loop():
            # This should raise RuntimeError when called from within running loop
            with pytest.raises(RuntimeError):
                manager.run_async_sync(AsyncTestFixtures.async_add, 1, 2)
        
        # Run the test
        asyncio.run(test_in_loop())
        self.test_results.append(("loop_detection", True))
    
    def test_error_propagation(self):
        """Test that async errors propagate correctly"""
        manager = AsyncSyncTestManager()
        fixtures = AsyncTestFixtures()
        
        with pytest.raises(ValueError, match="Division by zero"):
            manager.run_async_sync(fixtures.async_divide, 10, 0)
        
        self.test_results.append(("error_propagation", True))
    
    def get_test_summary(self):
        """Get summary of test results"""
        return {
            'total_tests': len(self.test_results),
            'passed': sum(1 for _, passed in self.test_results if passed),
            'failed': sum(1 for _, passed in self.test_results if not passed),
            'results': self.test_results
        }


# Pytest integration patterns
class PyTestAsyncSyncPatterns:
    """Integration patterns for pytest async/sync testing"""
    
    @pytest.fixture
    def async_manager(self):
        """Provide async test manager"""
        return AsyncSyncTestManager()
    
    @pytest.fixture
    def async_fixtures(self):
        """Provide async test fixtures"""
        return AsyncTestFixtures()
    
    @pytest.mark.asyncio
    async def test_async_context(self):
        """Test execution within async context"""
        manager = AsyncSyncTestManager()
        
        # This should raise RuntimeError in async context
        with pytest.raises(RuntimeError):
            manager.run_async_sync(AsyncTestFixtures.async_add, 1, 2)
    
    def test_sync_context(self, async_manager, async_fixtures):
        """Test execution in sync context"""
        result = async_manager.run_async_sync(async_fixtures.async_add, 5, 10)
        assert result == 15


# Export patterns for integration
__all__ = [
    'AsyncSyncTestManager',
    'AsyncTestFixtures', 
    'AsyncSyncTestValidator',
    'PyTestAsyncSyncPatterns'
]