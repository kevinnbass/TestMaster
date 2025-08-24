"""Flow Testing Framework - CrewAI Pattern
Extracted patterns for testing flow execution and control structures
Supports sequential, parallel, conditional, and cyclic flows
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union
from pydantic import BaseModel
import pytest


class MockFlow:
    """Mock flow for testing flow patterns"""
    
    def __init__(self, state_class: type = None):
        self.execution_order = []
        self.method_registry = {}
        self.listeners = {}
        self.routers = {}
        self.start_methods = []
        self._state = {} if state_class is None else state_class()
        self._events = []
        self.name = self.__class__.__name__
    
    @property
    def state(self):
        """Get flow state"""
        if isinstance(self._state, dict):
            # Auto-generate UUID for unstructured state
            if 'id' not in self._state:
                import uuid
                self._state['id'] = str(uuid.uuid4())
        elif hasattr(self._state, '__dict__') and not hasattr(self._state, 'id'):
            # Auto-generate UUID for structured state
            import uuid
            self._state.id = str(uuid.uuid4())
        return self._state
    
    @state.setter
    def state(self, value):
        """Set flow state"""
        self._state = value
    
    def start(self, step_name: str = None):
        """Decorator for start methods"""
        def decorator(func):
            self.start_methods.append(func.__name__)
            if step_name:
                self.method_registry[step_name] = func
            return func
        return decorator
    
    def listen(self, trigger):
        """Decorator for listener methods"""
        def decorator(func):
            trigger_name = self._get_trigger_name(trigger)
            if trigger_name not in self.listeners:
                self.listeners[trigger_name] = []
            self.listeners[trigger_name].append(func.__name__)
            return func
        return decorator
    
    def router(self, trigger):
        """Decorator for router methods"""
        def decorator(func):
            trigger_name = self._get_trigger_name(trigger)
            if trigger_name not in self.routers:
                self.routers[trigger_name] = []
            self.routers[trigger_name].append(func.__name__)
            return func
        return decorator
    
    def _get_trigger_name(self, trigger):
        """Get trigger name from various trigger types"""
        if callable(trigger):
            return trigger.__name__
        elif isinstance(trigger, str):
            return trigger
        elif hasattr(trigger, 'triggers'):  # For and_/or_ conditions
            return str(trigger)
        else:
            return str(trigger)
    
    def kickoff(self, inputs: Dict[str, Any] = None):
        """Execute flow synchronously"""
        if inputs:
            self._apply_inputs(inputs)
        
        self._emit_event("FlowStartedEvent", {
            "flow_name": self.name,
            "inputs": inputs,
            "timestamp": datetime.now()
        })
        
        # Execute start methods
        for start_method in self.start_methods:
            self._execute_method(start_method)
        
        # Process execution queue
        self._process_execution_queue()
        
        result = self._get_final_result()
        
        self._emit_event("FlowFinishedEvent", {
            "flow_name": self.name, 
            "result": result,
            "timestamp": datetime.now()
        })
        
        return result
    
    async def kickoff_async(self, inputs: Dict[str, Any] = None):
        """Execute flow asynchronously"""
        if inputs:
            self._apply_inputs(inputs)
        
        self._emit_event("FlowStartedEvent", {
            "flow_name": self.name,
            "inputs": inputs,
            "timestamp": datetime.now()
        })
        
        # Execute start methods asynchronously
        for start_method in self.start_methods:
            await self._execute_method_async(start_method)
        
        # Process execution queue asynchronously
        await self._process_execution_queue_async()
        
        result = self._get_final_result()
        
        self._emit_event("FlowFinishedEvent", {
            "flow_name": self.name,
            "result": result, 
            "timestamp": datetime.now()
        })
        
        return result
    
    def plot(self, filename: str):
        """Generate flow visualization"""
        self._emit_event("FlowPlotEvent", {
            "flow_name": self.name,
            "filename": filename,
            "timestamp": datetime.now()
        })
        return f"Flow plotted to {filename}"
    
    def _apply_inputs(self, inputs: Dict[str, Any]):
        """Apply inputs to flow state"""
        if isinstance(self._state, dict):
            self._state.update(inputs)
        else:
            for key, value in inputs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
    
    def _execute_method(self, method_name: str, *args):
        """Execute a flow method"""
        self._emit_event("MethodExecutionStartedEvent", {
            "method_name": method_name,
            "params": {f"_{i}": arg for i, arg in enumerate(args)} if args else {},
            "state": self.state,
            "flow_name": self.name
        })
        
        method = getattr(self, method_name, None)
        if method:
            result = method(*args)
            
            self._emit_event("MethodExecutionFinishedEvent", {
                "method_name": method_name,
                "result": result,
                "state": self.state,
                "flow_name": self.name
            })
            
            return result
        return None
    
    async def _execute_method_async(self, method_name: str, *args):
        """Execute a flow method asynchronously"""
        self._emit_event("MethodExecutionStartedEvent", {
            "method_name": method_name,
            "params": {f"_{i}": arg for i, arg in enumerate(args)} if args else {},
            "state": self.state,
            "flow_name": self.name
        })
        
        method = getattr(self, method_name, None)
        if method:
            if asyncio.iscoroutinefunction(method):
                result = await method(*args)
            else:
                result = method(*args)
            
            self._emit_event("MethodExecutionFinishedEvent", {
                "method_name": method_name,
                "result": result,
                "state": self.state,
                "flow_name": self.name
            })
            
            return result
        return None
    
    def _process_execution_queue(self):
        """Process execution queue for listeners and routers"""
        executed_methods = set(self.start_methods)
        queue = list(self.start_methods)
        
        while queue:
            current_method = queue.pop(0)
            
            # Check for listeners
            if current_method in self.listeners:
                for listener in self.listeners[current_method]:
                    if listener not in executed_methods:
                        result = self._execute_method(listener)
                        executed_methods.add(listener)
                        queue.append(listener)
            
            # Check for routers
            if current_method in self.routers:
                for router in self.routers[current_method]:
                    if router not in executed_methods:
                        route_result = self._execute_method(router)
                        executed_methods.add(router)
                        
                        # Handle router result
                        if route_result:
                            if route_result in self.listeners:
                                for listener in self.listeners[route_result]:
                                    if listener not in executed_methods:
                                        self._execute_method(listener)
                                        executed_methods.add(listener)
                                        queue.append(listener)
    
    async def _process_execution_queue_async(self):
        """Process execution queue asynchronously"""
        executed_methods = set(self.start_methods)
        queue = list(self.start_methods)
        
        while queue:
            current_method = queue.pop(0)
            
            # Check for listeners
            if current_method in self.listeners:
                for listener in self.listeners[current_method]:
                    if listener not in executed_methods:
                        result = await self._execute_method_async(listener)
                        executed_methods.add(listener)
                        queue.append(listener)
            
            # Check for routers
            if current_method in self.routers:
                for router in self.routers[current_method]:
                    if router not in executed_methods:
                        route_result = await self._execute_method_async(router)
                        executed_methods.add(router)
                        
                        # Handle router result
                        if route_result:
                            if route_result in self.listeners:
                                for listener in self.listeners[route_result]:
                                    if listener not in executed_methods:
                                        await self._execute_method_async(listener)
                                        executed_methods.add(listener)
                                        queue.append(listener)
    
    def _get_final_result(self):
        """Get final flow result"""
        # Return the last execution result or a default
        return "Flow execution completed"
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit flow event"""
        self._events.append({
            "type": event_type,
            "data": event_data
        })
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get emitted events"""
        return self._events


class ConditionalFlow:
    """Helper for conditional flow logic"""
    
    @staticmethod
    def and_(*triggers):
        """AND condition - all triggers must complete"""
        return AndCondition(triggers)
    
    @staticmethod  
    def or_(*triggers):
        """OR condition - any trigger can complete"""
        return OrCondition(triggers)


class AndCondition:
    """AND condition for flow triggers"""
    
    def __init__(self, triggers):
        self.triggers = triggers
    
    def __str__(self):
        return f"and_({', '.join(str(t) for t in self.triggers)})"


class OrCondition:
    """OR condition for flow triggers"""
    
    def __init__(self, triggers):
        self.triggers = triggers
    
    def __str__(self):
        return f"or_({', '.join(str(t) for t in self.triggers)})"


class FlowTestFramework:
    """Framework for testing flow execution patterns"""
    
    def __init__(self):
        self.test_results = []
        
    def create_simple_sequential_flow(self) -> MockFlow:
        """Create a simple sequential flow"""
        class SimpleFlow(MockFlow):
            
            @MockFlow.start()
            def step_1(self):
                self.execution_order.append("step_1")
                return "step_1_result"
            
            @MockFlow.listen(step_1)
            def step_2(self):
                self.execution_order.append("step_2")
                return "step_2_result"
        
        return SimpleFlow()
    
    def create_multi_start_flow(self) -> MockFlow:
        """Create flow with multiple start methods"""
        class MultiStartFlow(MockFlow):
            
            @MockFlow.start()
            def step_a(self):
                self.execution_order.append("step_a")
                return "step_a_result"
            
            @MockFlow.start()
            def step_b(self):
                self.execution_order.append("step_b")
                return "step_b_result"
            
            @MockFlow.listen(step_a)
            def step_c(self):
                self.execution_order.append("step_c")
                return "step_c_result"
            
            @MockFlow.listen(step_b)
            def step_d(self):
                self.execution_order.append("step_d")
                return "step_d_result"
        
        return MultiStartFlow()
    
    def create_cyclic_flow(self, max_iterations: int = 3) -> MockFlow:
        """Create a cyclic flow"""
        class CyclicFlow(MockFlow):
            def __init__(self):
                super().__init__()
                self.iteration = 0
                self.max_iterations = max_iterations
            
            @MockFlow.start("loop")
            def step_1(self):
                if self.iteration >= self.max_iterations:
                    return
                self.execution_order.append(f"step_1_{self.iteration}")
                return f"step_1_{self.iteration}_result"
            
            @MockFlow.listen(step_1)
            def step_2(self):
                self.execution_order.append(f"step_2_{self.iteration}")
                return f"step_2_{self.iteration}_result"
            
            @MockFlow.router(step_2)
            def step_3(self):
                self.execution_order.append(f"step_3_{self.iteration}")
                self.iteration += 1
                if self.iteration < self.max_iterations:
                    return "loop"
                return "exit"
        
        return CyclicFlow()
    
    def create_conditional_flow(self, condition_type: str = "and") -> MockFlow:
        """Create flow with conditional logic"""
        if condition_type == "and":
            class AndConditionFlow(MockFlow):
                
                @MockFlow.start()
                def step_1(self):
                    self.execution_order.append("step_1")
                    return "step_1_result"
                
                @MockFlow.start()
                def step_2(self):
                    self.execution_order.append("step_2")
                    return "step_2_result"
                
                # Simulated AND condition
                def step_3(self):
                    # Only execute if both step_1 and step_2 completed
                    if "step_1" in self.execution_order and "step_2" in self.execution_order:
                        self.execution_order.append("step_3")
                        return "step_3_result"
            
            flow = AndConditionFlow()
            # Manually set up the AND logic
            flow.listeners["step_1"] = ["step_3"]
            flow.listeners["step_2"] = ["step_3"]
            return flow
            
        else:  # OR condition
            class OrConditionFlow(MockFlow):
                
                @MockFlow.start()
                def step_a(self):
                    self.execution_order.append("step_a")
                    return "step_a_result"
                
                @MockFlow.start()
                def step_b(self):
                    self.execution_order.append("step_b")
                    return "step_b_result"
                
                def step_c(self):
                    # Execute if either step_a or step_b completed
                    if "step_a" in self.execution_order or "step_b" in self.execution_order:
                        self.execution_order.append("step_c")
                        return "step_c_result"
            
            flow = OrConditionFlow()
            # Manually set up the OR logic
            flow.listeners["step_a"] = ["step_c"]
            flow.listeners["step_b"] = ["step_c"]
            return flow
    
    def create_router_flow(self) -> MockFlow:
        """Create flow with router logic"""
        class RouterFlow(MockFlow):
            
            @MockFlow.start()
            def start_method(self):
                self.execution_order.append("start_method")
                return "start_result"
            
            @MockFlow.router(start_method)
            def router(self):
                self.execution_order.append("router")
                condition = True  # Configurable condition
                return "step_if_true" if condition else "step_if_false"
            
            def truthy(self):
                self.execution_order.append("step_if_true")
                return "truthy_result"
            
            def falsy(self):
                self.execution_order.append("step_if_false")
                return "falsy_result"
        
        flow = RouterFlow()
        # Set up router listeners
        flow.listeners["step_if_true"] = ["truthy"]
        flow.listeners["step_if_false"] = ["falsy"]
        return flow
    
    def test_flow_execution(self, flow: MockFlow, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test flow execution and capture results"""
        try:
            result = flow.kickoff(inputs)
            
            return {
                'success': True,
                'result': result,
                'execution_order': flow.execution_order,
                'events': flow.get_events(),
                'final_state': flow.state
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_order': flow.execution_order
            }
    
    async def test_async_flow_execution(self, flow: MockFlow, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test async flow execution"""
        try:
            result = await flow.kickoff_async(inputs)
            
            return {
                'success': True,
                'result': result,
                'execution_order': flow.execution_order,
                'events': flow.get_events(),
                'final_state': flow.state
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_order': flow.execution_order
            }
    
    def test_flow_restart(self, flow: MockFlow) -> Dict[str, Any]:
        """Test flow restart capability"""
        # First execution
        first_result = flow.kickoff()
        first_order = flow.execution_order.copy()
        
        # Second execution (restart)
        second_result = flow.kickoff()
        second_order = flow.execution_order.copy()
        
        return {
            'first_execution': {
                'result': first_result,
                'order': first_order
            },
            'second_execution': {
                'result': second_result, 
                'order': second_order
            },
            'restart_successful': len(second_order) > len(first_order)
        }


class FlowValidator:
    """Validates flow execution and behavior"""
    
    def __init__(self):
        self.framework = FlowTestFramework()
    
    def validate_sequential_flow(self) -> Dict[str, Any]:
        """Validate simple sequential flow"""
        flow = self.framework.create_simple_sequential_flow()
        result = self.framework.test_flow_execution(flow)
        
        validation = {
            'execution_successful': result['success'],
            'correct_order': result.get('execution_order') == ["step_1", "step_2"],
            'events_emitted': len(result.get('events', [])) > 0
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_multi_start_flow(self) -> Dict[str, Any]:
        """Validate multi-start flow"""
        flow = self.framework.create_multi_start_flow()
        result = self.framework.test_flow_execution(flow)
        
        execution_order = result.get('execution_order', [])
        validation = {
            'execution_successful': result['success'],
            'both_starts_executed': "step_a" in execution_order and "step_b" in execution_order,
            'both_listeners_executed': "step_c" in execution_order and "step_d" in execution_order,
            'correct_sequence': (
                execution_order.index("step_c") > execution_order.index("step_a") and
                execution_order.index("step_d") > execution_order.index("step_b")
            ) if all(s in execution_order for s in ["step_a", "step_b", "step_c", "step_d"]) else False
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_cyclic_flow(self) -> Dict[str, Any]:
        """Validate cyclic flow behavior"""
        max_iterations = 3
        flow = self.framework.create_cyclic_flow(max_iterations)
        result = self.framework.test_flow_execution(flow)
        
        execution_order = result.get('execution_order', [])
        
        # Check expected pattern
        expected_order = []
        for i in range(max_iterations):
            expected_order.extend([f"step_1_{i}", f"step_2_{i}", f"step_3_{i}"])
        
        validation = {
            'execution_successful': result['success'],
            'correct_iteration_count': flow.iteration == max_iterations,
            'correct_execution_order': execution_order == expected_order,
            'all_iterations_executed': len([s for s in execution_order if s.startswith("step_1_")]) == max_iterations
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_router_flow(self) -> Dict[str, Any]:
        """Validate router flow behavior"""
        flow = self.framework.create_router_flow()
        result = self.framework.test_flow_execution(flow)
        
        execution_order = result.get('execution_order', [])
        expected_order = ["start_method", "router", "step_if_true"]
        
        validation = {
            'execution_successful': result['success'],
            'correct_routing': execution_order == expected_order,
            'router_executed': "router" in execution_order,
            'correct_branch_taken': "step_if_true" in execution_order and "step_if_false" not in execution_order
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    async def validate_async_flow(self) -> Dict[str, Any]:
        """Validate async flow execution"""
        flow = self.framework.create_simple_sequential_flow()
        
        # Add async method
        async def async_step_1(self):
            await asyncio.sleep(0.01)
            self.execution_order.append("step_1")
            return "async_step_1_result"
        
        flow.step_1 = async_step_1.__get__(flow, type(flow))
        
        result = await self.framework.test_async_flow_execution(flow)
        
        validation = {
            'async_execution_successful': result['success'],
            'correct_order': result.get('execution_order') == ["step_1", "step_2"]
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }


# Pytest integration patterns  
class PyTestFlowPatterns:
    """Flow testing patterns for pytest"""
    
    @pytest.fixture
    def flow_framework(self):
        """Provide flow test framework"""
        return FlowTestFramework()
    
    def test_simple_sequential_flow(self, flow_framework):
        """Test simple sequential flow"""
        flow = flow_framework.create_simple_sequential_flow()
        result = flow_framework.test_flow_execution(flow)
        
        assert result['success'] is True
        assert result['execution_order'] == ["step_1", "step_2"]
    
    def test_flow_with_multiple_starts(self, flow_framework):
        """Test flow with multiple start points"""
        flow = flow_framework.create_multi_start_flow()
        result = flow_framework.test_flow_execution(flow)
        
        execution_order = result['execution_order']
        assert "step_a" in execution_order
        assert "step_b" in execution_order
        assert "step_c" in execution_order
        assert "step_d" in execution_order
        assert execution_order.index("step_c") > execution_order.index("step_a")
        assert execution_order.index("step_d") > execution_order.index("step_b")
    
    def test_cyclic_flow(self, flow_framework):
        """Test cyclic flow execution"""
        flow = flow_framework.create_cyclic_flow(max_iterations=3)
        result = flow_framework.test_flow_execution(flow)
        
        execution_order = result['execution_order']
        expected_order = []
        for i in range(3):
            expected_order.extend([f"step_1_{i}", f"step_2_{i}", f"step_3_{i}"])
        
        assert result['success'] is True
        assert execution_order == expected_order
    
    @pytest.mark.asyncio
    async def test_async_flow(self, flow_framework):
        """Test asynchronous flow execution"""
        flow = flow_framework.create_simple_sequential_flow()
        
        # Make step_1 async
        async def async_step_1(self):
            await asyncio.sleep(0.01)
            self.execution_order.append("step_1")
            return "async_result"
        
        flow.step_1 = async_step_1.__get__(flow, type(flow))
        
        result = await flow_framework.test_async_flow_execution(flow)
        
        assert result['success'] is True
        assert result['execution_order'] == ["step_1", "step_2"]
    
    def test_flow_restart(self, flow_framework):
        """Test flow restart capability"""
        flow = flow_framework.create_simple_sequential_flow()
        result = flow_framework.test_flow_restart(flow)
        
        assert result['restart_successful'] is True
        assert result['first_execution']['order'] == ["step_1", "step_2"]
        assert result['second_execution']['order'] == ["step_1", "step_2", "step_1", "step_2"]


# Export patterns for integration
__all__ = [
    'FlowTestFramework',
    'FlowValidator', 
    'MockFlow',
    'ConditionalFlow',
    'AndCondition',
    'OrCondition',
    'PyTestFlowPatterns'
]