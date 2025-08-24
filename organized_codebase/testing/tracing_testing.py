# -*- coding: utf-8 -*-
"""
AgentScope Tracing System Testing Framework
==========================================

Extracted from agentscope/tests/tracer_test.py
Enhanced for TestMaster integration

Testing patterns for:
- Function and method tracing with decorators
- LLM call tracing with stream and non-stream modes
- Agent reply tracing and lifecycle monitoring
- Formatter operation tracing
- Toolkit execution tracing with error handling
- Embedding model tracing
- Async and sync function tracing
- Generator and async generator tracing
- Error propagation in traced functions
"""

import asyncio
import functools
import inspect
import time
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union, Callable
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

import pytest


class MockTextBlock:
    """Mock text block for tracing tests"""
    
    def __init__(self, text: str, block_type: str = "text"):
        self.type = block_type
        self.text = text
    
    def __eq__(self, other):
        if not isinstance(other, MockTextBlock):
            return False
        return self.type == other.type and self.text == other.text
    
    def __repr__(self):
        return f"MockTextBlock(type='{self.type}', text='{self.text}')"


class MockToolUseBlock:
    """Mock tool use block for tracing tests"""
    
    def __init__(self, tool_id: str, name: str, input_data: Dict):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = input_data


class MockChatResponse:
    """Mock chat response for LLM tracing tests"""
    
    def __init__(self, response_id: str, content: List[MockTextBlock], usage: Optional[Dict] = None):
        self.id = response_id
        self.content = content
        self.usage = usage or {"input_tokens": 10, "output_tokens": 5}


class MockMessage:
    """Mock message for agent tracing tests"""
    
    def __init__(self, role: str, content: Union[str, List], name: str, **kwargs):
        self.role = role
        self.content = content if isinstance(content, list) else [MockTextBlock(content)]
        self.name = name
        self.__dict__.update(kwargs)


class MockToolResponse:
    """Mock tool response for toolkit tracing tests"""
    
    def __init__(self, content: List[MockTextBlock]):
        self.content = content


class TraceRecord:
    """Record of a traced function call"""
    
    def __init__(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        start_time: float,
        trace_type: str = "function"
    ):
        self.function_name = function_name
        self.args = args
        self.kwargs = kwargs
        self.start_time = start_time
        self.end_time = None
        self.result = None
        self.error = None
        self.trace_type = trace_type
        self.metadata = {}
    
    def finish(self, result: Any = None, error: Exception = None):
        """Mark trace as finished"""
        self.end_time = time.time()
        self.result = result
        self.error = error
    
    def duration(self) -> Optional[float]:
        """Get execution duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "function_name": self.function_name,
            "args": str(self.args),
            "kwargs": str(self.kwargs),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "success": self.error is None,
            "error": str(self.error) if self.error else None,
            "trace_type": self.trace_type,
            "metadata": self.metadata
        }


class TracingManager:
    """Manager for tracing system"""
    
    def __init__(self):
        self.traces = []
        self.enabled = True
        self.filters = []
        self.listeners = []
    
    def add_trace(self, trace: TraceRecord):
        """Add trace record"""
        if self.enabled and self._should_trace(trace):
            self.traces.append(trace)
            self._notify_listeners(trace)
    
    def _should_trace(self, trace: TraceRecord) -> bool:
        """Check if trace should be recorded based on filters"""
        if not self.filters:
            return True
        
        for filter_func in self.filters:
            if not filter_func(trace):
                return False
        return True
    
    def _notify_listeners(self, trace: TraceRecord):
        """Notify trace listeners"""
        for listener in self.listeners:
            try:
                listener(trace)
            except Exception:
                pass  # Don't let listener errors break tracing
    
    def clear_traces(self):
        """Clear all traces"""
        self.traces.clear()
    
    def get_traces_by_type(self, trace_type: str) -> List[TraceRecord]:
        """Get traces by type"""
        return [trace for trace in self.traces if trace.trace_type == trace_type]
    
    def get_traces_by_function(self, function_name: str) -> List[TraceRecord]:
        """Get traces by function name"""
        return [trace for trace in self.traces if trace.function_name == function_name]
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        if not self.traces:
            return {"total_traces": 0}
        
        by_type = {}
        by_function = {}
        total_duration = 0
        success_count = 0
        
        for trace in self.traces:
            # Count by type
            by_type[trace.trace_type] = by_type.get(trace.trace_type, 0) + 1
            
            # Count by function
            by_function[trace.function_name] = by_function.get(trace.function_name, 0) + 1
            
            # Duration
            if trace.duration():
                total_duration += trace.duration()
            
            # Success rate
            if trace.error is None:
                success_count += 1
        
        return {
            "total_traces": len(self.traces),
            "by_type": by_type,
            "by_function": by_function,
            "total_duration": total_duration,
            "average_duration": total_duration / len(self.traces) if self.traces else 0,
            "success_rate": success_count / len(self.traces) if self.traces else 0
        }


# Global tracing manager instance
_tracing_manager = TracingManager()


def trace(name: str = None, trace_type: str = "function"):
    """General purpose tracing decorator"""
    def decorator(func: Callable):
        function_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                trace_record = TraceRecord(function_name, args, kwargs, time.time(), trace_type)
                _tracing_manager.add_trace(trace_record)
                
                try:
                    result = await func(*args, **kwargs)
                    trace_record.finish(result=result)
                    return result
                except Exception as e:
                    trace_record.finish(error=e)
                    raise
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                trace_record = TraceRecord(function_name, args, kwargs, time.time(), trace_type)
                _tracing_manager.add_trace(trace_record)
                
                try:
                    result = func(*args, **kwargs)
                    trace_record.finish(result=result)
                    return result
                except Exception as e:
                    trace_record.finish(error=e)
                    raise
            
            return sync_wrapper
    
    return decorator


def trace_llm(func: Callable):
    """Decorator for tracing LLM calls"""
    return trace(name=func.__name__, trace_type="llm")(func)


def trace_reply(func: Callable):
    """Decorator for tracing agent replies"""
    return trace(name=func.__name__, trace_type="reply")(func)


def trace_format(func: Callable):
    """Decorator for tracing formatter operations"""
    return trace(name=func.__name__, trace_type="format")(func)


def trace_embedding(func: Callable):
    """Decorator for tracing embedding operations"""
    return trace(name=func.__name__, trace_type="embedding")(func)


def trace_toolkit(func: Callable):
    """Decorator for tracing toolkit operations"""
    return trace(name=func.__name__, trace_type="toolkit")(func)


class MockLLMModel:
    """Mock LLM model for tracing tests"""
    
    def __init__(self, model_name: str, stream: bool = False, raise_error: bool = False):
        self.model_name = model_name
        self.stream = stream
        self.raise_error = raise_error
    
    @trace_llm
    async def __call__(
        self,
        messages: List[Dict],
        **kwargs
    ) -> Union[MockChatResponse, AsyncGenerator[MockChatResponse, None]]:
        """Simulate LLM call"""
        if self.raise_error:
            raise ValueError("Simulated LLM error")
        
        if self.stream:
            async def stream_generator():
                for i in range(3):
                    yield MockChatResponse(
                        f"msg_{i}",
                        [MockTextBlock("x" * (i + 1))]
                    )
            
            return stream_generator()
        else:
            return MockChatResponse(
                "msg_0",
                [MockTextBlock("Hello, world!")]
            )


class MockAgent:
    """Mock agent for tracing tests"""
    
    def __init__(self, name: str = "TestAgent"):
        self.name = name
        self.reply_count = 0
    
    @trace_reply
    async def reply(self, message: Optional[MockMessage] = None, raise_error: bool = False) -> MockMessage:
        """Simulate agent reply"""
        self.reply_count += 1
        
        if raise_error:
            raise ValueError("Simulated agent reply error")
        
        return MockMessage(
            "assistant",
            [MockTextBlock("Hello, world!")],
            self.name
        )
    
    async def __call__(self, message: Optional[MockMessage] = None, **kwargs):
        """Call agent reply"""
        return await self.reply(message, **kwargs)
    
    async def observe(self, message: MockMessage):
        """Observe message (not traced by default)"""
        pass
    
    async def handle_interrupt(self, *args, **kwargs) -> MockMessage:
        """Handle interrupt (not traced by default)"""
        return MockMessage("system", "Interrupt handled", "system")


class MockFormatter:
    """Mock formatter for tracing tests"""
    
    def __init__(self, formatter_name: str = "TestFormatter"):
        self.formatter_name = formatter_name
    
    @trace_format
    async def format(self, messages: List[MockMessage], raise_error: bool = False) -> List[Dict]:
        """Simulate message formatting"""
        if raise_error:
            raise ValueError("Simulated formatter error")
        
        return [
            {"role": msg.role, "content": msg.content[0].text if msg.content else ""}
            for msg in messages
        ]


class MockToolkit:
    """Mock toolkit for tracing tests"""
    
    def __init__(self):
        self.tools = {}
        self.call_count = 0
    
    def register_tool_function(self, func: Callable, name: Optional[str] = None):
        """Register tool function"""
        tool_name = name or func.__name__
        self.tools[tool_name] = func
    
    @trace_toolkit
    async def call_tool_function(self, tool_use: MockToolUseBlock) -> AsyncGenerator[MockToolResponse, None]:
        """Call tool function with tracing"""
        self.call_count += 1
        
        if tool_use.name not in self.tools:
            yield MockToolResponse([MockTextBlock(f"Error: Tool '{tool_use.name}' not found")])
            return
        
        tool_func = self.tools[tool_use.name]
        
        try:
            # Call tool function
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_use.input)
            else:
                result = tool_func(**tool_use.input)
            
            # Handle different result types
            if hasattr(result, '__aiter__'):
                # AsyncGenerator result
                async for chunk in result:
                    yield chunk
            elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                # Generator result
                for chunk in result:
                    yield chunk
            else:
                # Single result
                if isinstance(result, MockToolResponse):
                    yield result
                else:
                    yield MockToolResponse([MockTextBlock(str(result))])
        
        except Exception as e:
            yield MockToolResponse([MockTextBlock(f"Error: {str(e)}")])


class MockEmbeddingModel:
    """Mock embedding model for tracing tests"""
    
    def __init__(self, model_name: str = "TestEmbedding"):
        self.model_name = model_name
    
    @trace_embedding
    async def __call__(self, texts: List[str], raise_error: bool = False) -> List[List[float]]:
        """Simulate embedding generation"""
        if raise_error:
            raise ValueError("Simulated embedding error")
        
        # Return mock embeddings
        return [[float(i), float(i+1), float(i+2)] for i in range(len(texts))]


class TracingTestFramework:
    """Core framework for tracing system testing"""
    
    def __init__(self):
        self.manager = _tracing_manager
        self.models = {}
        self.agents = {}
        self.formatters = {}
        self.toolkits = {}
        self.embedding_models = {}
    
    def setup_tracing(self, enabled: bool = True):
        """Setup tracing configuration"""
        self.manager.enabled = enabled
        self.manager.clear_traces()
    
    def create_llm_model(
        self,
        name: str,
        stream: bool = False,
        raise_error: bool = False
    ) -> MockLLMModel:
        """Create mock LLM model"""
        model = MockLLMModel(name, stream, raise_error)
        self.models[name] = model
        return model
    
    def create_agent(self, name: str) -> MockAgent:
        """Create mock agent"""
        agent = MockAgent(name)
        self.agents[name] = agent
        return agent
    
    def create_formatter(self, name: str) -> MockFormatter:
        """Create mock formatter"""
        formatter = MockFormatter(name)
        self.formatters[name] = formatter
        return formatter
    
    def create_toolkit(self, name: str) -> MockToolkit:
        """Create mock toolkit"""
        toolkit = MockToolkit()
        self.toolkits[name] = toolkit
        return toolkit
    
    def create_embedding_model(self, name: str) -> MockEmbeddingModel:
        """Create mock embedding model"""
        model = MockEmbeddingModel(name)
        self.embedding_models[name] = model
        return model
    
    async def run_traced_function_test(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Dict = None,
        expected_trace_type: str = "function",
        should_raise: bool = False
    ) -> Dict[str, Any]:
        """Run test on traced function"""
        kwargs = kwargs or {}
        initial_trace_count = len(self.manager.traces)
        
        result = None
        error = None
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
        except Exception as e:
            error = e
            if not should_raise:
                raise
        
        # Verify trace was added
        new_traces = self.manager.traces[initial_trace_count:]
        
        return {
            "result": result,
            "error": error,
            "traces_added": len(new_traces),
            "new_traces": new_traces,
            "expected_trace_type": expected_trace_type,
            "trace_types_match": all(t.trace_type == expected_trace_type for t in new_traces)
        }
    
    def add_trace_filter(self, filter_func: Callable[[TraceRecord], bool]):
        """Add trace filter"""
        self.manager.filters.append(filter_func)
    
    def add_trace_listener(self, listener_func: Callable[[TraceRecord], None]):
        """Add trace listener"""
        self.manager.listeners.append(listener_func)
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get comprehensive trace summary"""
        stats = self.manager.get_trace_statistics()
        
        # Add detailed trace information
        traces_by_type = {}
        for trace_type in stats.get("by_type", {}):
            traces_by_type[trace_type] = [
                trace.to_dict() for trace in self.manager.get_traces_by_type(trace_type)
            ]
        
        return {
            "statistics": stats,
            "traces_by_type": traces_by_type,
            "total_traces": len(self.manager.traces),
            "enabled": self.manager.enabled
        }


class TracingValidator:
    """Validator for tracing functionality"""
    
    @staticmethod
    def validate_trace_record(
        trace: TraceRecord,
        expected_function: str = None,
        expected_type: str = None,
        expected_success: bool = True
    ) -> Dict[str, Any]:
        """Validate individual trace record"""
        validations = {}
        
        if expected_function:
            validations["function_name"] = {
                "expected": expected_function,
                "actual": trace.function_name,
                "matches": trace.function_name == expected_function
            }
        
        if expected_type:
            validations["trace_type"] = {
                "expected": expected_type,
                "actual": trace.trace_type,
                "matches": trace.trace_type == expected_type
            }
        
        validations["success"] = {
            "expected": expected_success,
            "actual": trace.error is None,
            "matches": (trace.error is None) == expected_success
        }
        
        validations["has_timing"] = {
            "start_time_set": trace.start_time is not None,
            "end_time_set": trace.end_time is not None,
            "duration_available": trace.duration() is not None
        }
        
        return {
            "is_valid": all(
                val.get("matches", val) if isinstance(val, dict) else val
                for val in validations.values()
            ),
            "validations": validations
        }
    
    @staticmethod
    def validate_trace_sequence(
        traces: List[TraceRecord],
        expected_sequence: List[str]
    ) -> Dict[str, Any]:
        """Validate sequence of trace function names"""
        actual_sequence = [trace.function_name for trace in traces]
        
        return {
            "is_valid": actual_sequence == expected_sequence,
            "expected_sequence": expected_sequence,
            "actual_sequence": actual_sequence,
            "sequence_length_matches": len(actual_sequence) == len(expected_sequence)
        }
    
    @staticmethod
    def validate_tracing_statistics(
        stats: Dict[str, Any],
        expected_total: int = None,
        expected_success_rate: float = None
    ) -> Dict[str, Any]:
        """Validate tracing statistics"""
        validations = {}
        
        if expected_total is not None:
            validations["total_traces"] = {
                "expected": expected_total,
                "actual": stats.get("total_traces", 0),
                "matches": stats.get("total_traces", 0) == expected_total
            }
        
        if expected_success_rate is not None:
            actual_rate = stats.get("success_rate", 0)
            validations["success_rate"] = {
                "expected": expected_success_rate,
                "actual": actual_rate,
                "matches": abs(actual_rate - expected_success_rate) < 0.01  # Allow small floating point differences
            }
        
        return {
            "is_valid": all(val["matches"] for val in validations.values()),
            "validations": validations
        }


class TracingTest(IsolatedAsyncioTestCase):
    """Comprehensive tracing system testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = TracingTestFramework()
        self.validator = TracingValidator()
        self.framework.setup_tracing(enabled=True)
    
    async def test_basic_function_tracing(self):
        """Test basic function tracing"""
        
        @trace(name="test_func")
        async def async_test_func(x: int) -> int:
            """Test async function"""
            return x * 2
        
        @trace(name="sync_func")
        def sync_test_func(x: int) -> int:
            """Test sync function"""
            return x + 3
        
        # Test async function
        result = await self.framework.run_traced_function_test(
            async_test_func,
            args=(5,),
            expected_trace_type="function"
        )
        
        assert result["result"] == 10
        assert result["traces_added"] == 1
        assert result["trace_types_match"]
        
        # Test sync function
        result = await self.framework.run_traced_function_test(
            sync_test_func,
            args=(4,),
            expected_trace_type="function"
        )
        
        assert result["result"] == 7
        assert result["traces_added"] == 1
        assert result["trace_types_match"]
    
    async def test_generator_tracing(self):
        """Test tracing of generators and async generators"""
        
        @trace(name="async_gen")
        async def async_generator() -> AsyncGenerator[str, None]:
            """Test async generator"""
            for i in range(3):
                yield f"chunk_{i}"
        
        @trace(name="sync_gen")
        def sync_generator() -> Generator[str, None, None]:
            """Test sync generator"""
            for i in range(3):
                yield f"sync_chunk_{i}"
        
        # Test async generator
        initial_count = len(self.framework.manager.traces)
        results = [chunk async for chunk in async_generator()]
        
        assert results == ["chunk_0", "chunk_1", "chunk_2"]
        assert len(self.framework.manager.traces) == initial_count + 1
        
        # Test sync generator
        initial_count = len(self.framework.manager.traces)
        results = list(sync_generator())
        
        assert results == ["sync_chunk_0", "sync_chunk_1", "sync_chunk_2"]
        assert len(self.framework.manager.traces) == initial_count + 1
    
    async def test_error_handling_in_tracing(self):
        """Test error handling in traced functions"""
        
        @trace(name="error_func")
        async def error_async_func() -> int:
            """Test error in async function"""
            raise ValueError("Test error")
        
        @trace(name="error_sync")
        def error_sync_func() -> int:
            """Test error in sync function"""
            raise ValueError("Test sync error")
        
        # Test async error
        result = await self.framework.run_traced_function_test(
            error_async_func,
            should_raise=True
        )
        
        assert result["error"] is not None
        assert isinstance(result["error"], ValueError)
        assert result["traces_added"] == 1
        
        # Validate trace recorded error
        error_traces = [t for t in result["new_traces"] if t.error is not None]
        assert len(error_traces) == 1
        
        # Test sync error
        with pytest.raises(ValueError):
            result = await self.framework.run_traced_function_test(
                error_sync_func,
                should_raise=True
            )
    
    async def test_llm_tracing(self):
        """Test LLM call tracing"""
        # Test non-streaming LLM
        non_stream_llm = self.framework.create_llm_model("non_stream", stream=False)
        
        result = await non_stream_llm([{"role": "user", "content": "Hello"}])
        assert isinstance(result, MockChatResponse)
        assert result.content[0].text == "Hello, world!"
        
        # Verify LLM trace
        llm_traces = self.framework.manager.get_traces_by_type("llm")
        assert len(llm_traces) >= 1
        
        latest_trace = llm_traces[-1]
        validation = self.validator.validate_trace_record(
            latest_trace,
            expected_function="__call__",
            expected_type="llm",
            expected_success=True
        )
        assert validation["is_valid"]
        
        # Test streaming LLM
        stream_llm = self.framework.create_llm_model("stream", stream=True)
        
        response_generator = await stream_llm([{"role": "user", "content": "Hello"}])
        results = [chunk.content async for chunk in response_generator]
        
        expected_results = [
            [MockTextBlock("x")],
            [MockTextBlock("xx")],
            [MockTextBlock("xxx")]
        ]
        
        assert results == expected_results
        
        # Test LLM error
        error_llm = self.framework.create_llm_model("error", raise_error=True)
        
        with pytest.raises(ValueError):
            await error_llm([])
        
        # Verify error was traced
        error_traces = [t for t in self.framework.manager.traces if t.error is not None and t.trace_type == "llm"]
        assert len(error_traces) >= 1
    
    async def test_agent_reply_tracing(self):
        """Test agent reply tracing"""
        agent = self.framework.create_agent("TestAgent")
        
        # Test successful reply
        response = await agent.reply()
        
        assert isinstance(response, MockMessage)
        assert response.content[0].text == "Hello, world!"
        assert agent.reply_count == 1
        
        # Verify reply trace
        reply_traces = self.framework.manager.get_traces_by_type("reply")
        assert len(reply_traces) >= 1
        
        latest_trace = reply_traces[-1]
        validation = self.validator.validate_trace_record(
            latest_trace,
            expected_function="reply",
            expected_type="reply",
            expected_success=True
        )
        assert validation["is_valid"]
        
        # Test reply with error
        with pytest.raises(ValueError):
            await agent.reply(raise_error=True)
        
        # Verify error trace
        error_reply_traces = [t for t in reply_traces if t.error is not None]
        assert len(error_reply_traces) >= 1
    
    async def test_formatter_tracing(self):
        """Test formatter operation tracing"""
        formatter = self.framework.create_formatter("TestFormatter")
        
        messages = [
            MockMessage("user", "Hello", "user"),
            MockMessage("assistant", "Hi there", "assistant")
        ]
        
        # Test successful formatting
        result = await formatter.format(messages)
        
        expected_result = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        assert result == expected_result
        
        # Verify format trace
        format_traces = self.framework.manager.get_traces_by_type("format")
        assert len(format_traces) >= 1
        
        latest_trace = format_traces[-1]
        validation = self.validator.validate_trace_record(
            latest_trace,
            expected_function="format",
            expected_type="format",
            expected_success=True
        )
        assert validation["is_valid"]
        
        # Test formatter error
        with pytest.raises(ValueError):
            await formatter.format(messages, raise_error=True)
    
    async def test_toolkit_tracing(self):
        """Test toolkit operation tracing"""
        toolkit = self.framework.create_toolkit("TestToolkit")
        
        # Register simple tool
        def simple_tool(text: str) -> MockToolResponse:
            return MockToolResponse([MockTextBlock(f"Processed: {text}")])
        
        toolkit.register_tool_function(simple_tool)
        
        # Test tool execution
        tool_use = MockToolUseBlock("tool_1", "simple_tool", {"text": "hello"})
        
        results = []
        async for response in toolkit.call_tool_function(tool_use):
            results.append(response)
        
        assert len(results) == 1
        assert results[0].content[0].text == "Processed: hello"
        
        # Verify toolkit trace
        toolkit_traces = self.framework.manager.get_traces_by_type("toolkit")
        assert len(toolkit_traces) >= 1
        
        # Register async generator tool
        async def async_gen_tool(count: int) -> AsyncGenerator[MockToolResponse, None]:
            for i in range(count):
                yield MockToolResponse([MockTextBlock(f"Chunk {i}")])
        
        toolkit.register_tool_function(async_gen_tool)
        
        # Test async generator tool
        tool_use = MockToolUseBlock("tool_2", "async_gen_tool", {"count": 3})
        
        results = []
        async for response in toolkit.call_tool_function(tool_use):
            results.append(response)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.content[0].text == f"Chunk {i}"
        
        # Test tool error
        def error_tool(raise_error: bool) -> MockToolResponse:
            if raise_error:
                raise ValueError("Tool error")
            return MockToolResponse([MockTextBlock("Success")])
        
        toolkit.register_tool_function(error_tool)
        
        tool_use = MockToolUseBlock("tool_3", "error_tool", {"raise_error": True})
        
        results = []
        async for response in toolkit.call_tool_function(tool_use):
            results.append(response)
        
        # Should get error message, not raise exception
        assert len(results) == 1
        assert "Error:" in results[0].content[0].text
    
    async def test_embedding_tracing(self):
        """Test embedding model tracing"""
        embedding_model = self.framework.create_embedding_model("TestEmbedding")
        
        # Test successful embedding
        texts = ["hello", "world"]
        embeddings = await embedding_model(texts)
        
        expected_embeddings = [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
        assert embeddings == expected_embeddings
        
        # Verify embedding trace
        embedding_traces = self.framework.manager.get_traces_by_type("embedding")
        assert len(embedding_traces) >= 1
        
        latest_trace = embedding_traces[-1]
        validation = self.validator.validate_trace_record(
            latest_trace,
            expected_function="__call__",
            expected_type="embedding",
            expected_success=True
        )
        assert validation["is_valid"]
        
        # Test embedding error
        with pytest.raises(ValueError):
            await embedding_model(["test"], raise_error=True)
    
    async def test_trace_filtering_and_statistics(self):
        """Test trace filtering and statistics"""
        # Add filter to only trace LLM calls
        def llm_only_filter(trace: TraceRecord) -> bool:
            return trace.trace_type == "llm"
        
        self.framework.add_trace_filter(llm_only_filter)
        
        # Create and call different types
        llm = self.framework.create_llm_model("filtered_test")
        agent = self.framework.create_agent("filtered_agent")
        
        initial_count = len(self.framework.manager.traces)
        
        # These should be traced (LLM)
        await llm([])
        await llm([])
        
        # These should be filtered out (not LLM)
        await agent.reply()
        await agent.reply()
        
        # Only LLM traces should be recorded
        new_traces = self.framework.manager.traces[initial_count:]
        llm_traces = [t for t in new_traces if t.trace_type == "llm"]
        non_llm_traces = [t for t in new_traces if t.trace_type != "llm"]
        
        assert len(llm_traces) == 2
        assert len(non_llm_traces) == 0
        
        # Test statistics
        stats = self.framework.manager.get_trace_statistics()
        
        stats_validation = self.validator.validate_tracing_statistics(
            stats,
            expected_success_rate=1.0  # All should be successful
        )
        assert stats_validation["is_valid"]
    
    async def test_trace_listeners(self):
        """Test trace event listeners"""
        listener_calls = []
        
        def trace_listener(trace: TraceRecord):
            listener_calls.append(trace.function_name)
        
        self.framework.add_trace_listener(trace_listener)
        
        # Clear filters for this test
        self.framework.manager.filters.clear()
        
        # Perform various operations
        agent = self.framework.create_agent("listener_test")
        await agent.reply()
        
        llm = self.framework.create_llm_model("listener_llm")
        await llm([])
        
        # Verify listener was called
        assert len(listener_calls) >= 2
        assert "reply" in listener_calls
        assert "__call__" in listener_calls
    
    async def test_comprehensive_trace_summary(self):
        """Test comprehensive trace summary generation"""
        # Clear previous traces and filters
        self.framework.manager.clear_traces()
        self.framework.manager.filters.clear()
        
        # Perform various operations
        llm = self.framework.create_llm_model("summary_llm")
        agent = self.framework.create_agent("summary_agent")
        formatter = self.framework.create_formatter("summary_formatter")
        
        await llm([])
        await agent.reply()
        await formatter.format([MockMessage("user", "test", "user")])
        
        # Get comprehensive summary
        summary = self.framework.get_trace_summary()
        
        assert summary["total_traces"] >= 3
        assert "statistics" in summary
        assert "traces_by_type" in summary
        
        # Verify different trace types are present
        assert "llm" in summary["statistics"]["by_type"]
        assert "reply" in summary["statistics"]["by_type"]
        assert "format" in summary["statistics"]["by_type"]
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        self.framework.manager.clear_traces()
        self.framework.manager.filters.clear()
        self.framework.manager.listeners.clear()


# Pytest integration
@pytest.fixture
def tracing_framework():
    """Pytest fixture for tracing framework"""
    framework = TracingTestFramework()
    framework.setup_tracing(enabled=True)
    yield framework
    framework.manager.clear_traces()


@pytest.fixture
def tracing_validator():
    """Pytest fixture for tracing validator"""
    return TracingValidator()


def test_tracing_framework_creation(tracing_framework):
    """Test tracing framework creation"""
    assert tracing_framework.manager.enabled
    llm = tracing_framework.create_llm_model("test")
    assert llm.model_name == "test"


def test_trace_record_creation():
    """Test trace record creation and methods"""
    record = TraceRecord("test_func", (1, 2), {"key": "value"}, time.time())
    
    assert record.function_name == "test_func"
    assert record.args == (1, 2)
    assert record.kwargs == {"key": "value"}
    
    # Test finish
    record.finish(result="test_result")
    assert record.result == "test_result"
    assert record.duration() is not None
    
    # Test to_dict
    record_dict = record.to_dict()
    assert "function_name" in record_dict
    assert record_dict["success"] == True


def test_tracing_manager():
    """Test tracing manager functionality"""
    manager = TracingManager()
    
    # Test adding traces
    trace1 = TraceRecord("func1", (), {}, time.time())
    trace2 = TraceRecord("func2", (), {}, time.time(), "llm")
    
    manager.add_trace(trace1)
    manager.add_trace(trace2)
    
    assert len(manager.traces) == 2
    
    # Test filtering by type
    llm_traces = manager.get_traces_by_type("llm")
    assert len(llm_traces) == 1
    assert llm_traces[0].function_name == "func2"
    
    # Test statistics
    stats = manager.get_trace_statistics()
    assert stats["total_traces"] == 2
    assert "by_type" in stats


@pytest.mark.asyncio
async def test_simple_function_tracing(tracing_framework):
    """Test simple function tracing"""
    
    @trace(name="simple_test")
    async def simple_func(x: int) -> int:
        return x * 2
    
    result = await simple_func(5)
    assert result == 10
    
    # Check trace was recorded
    traces = tracing_framework.manager.get_traces_by_function("simple_test")
    assert len(traces) == 1


def test_trace_validation(tracing_validator):
    """Test trace validation"""
    trace = TraceRecord("test_func", (), {}, time.time())
    trace.finish(result="success")
    
    validation = tracing_validator.validate_trace_record(
        trace,
        expected_function="test_func",
        expected_success=True
    )
    
    assert validation["is_valid"]


def test_trace_sequence_validation(tracing_validator):
    """Test trace sequence validation"""
    traces = [
        TraceRecord("func1", (), {}, time.time()),
        TraceRecord("func2", (), {}, time.time()),
        TraceRecord("func3", (), {}, time.time())
    ]
    
    validation = tracing_validator.validate_trace_sequence(
        traces,
        ["func1", "func2", "func3"]
    )
    
    assert validation["is_valid"]
    assert validation["sequence_length_matches"]