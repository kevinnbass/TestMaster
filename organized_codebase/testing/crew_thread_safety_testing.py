"""Crew Thread Safety Testing Framework - CrewAI Pattern
Extracted patterns for testing multi-threaded crew execution
Supports context isolation, parallel execution, and thread safety validation
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, patch
import pytest
import uuid


class MockCrewContext:
    """Mock crew context for thread safety testing"""
    
    def __init__(self, crew_id: str = None, crew_key: str = None):
        self.id = crew_id or str(uuid.uuid4())
        self.key = crew_key or f"crew_{self.id}"


class MockAgent:
    """Mock agent for crew testing"""
    
    def __init__(self, name: str, role: str = None, goal: str = None, backstory: str = None):
        self.name = name
        self.role = role or f"{name} Agent"
        self.goal = goal or f"Complete {name} task"
        self.backstory = backstory or f"I am agent for {name}"
        self.execution_count = 0
    
    def execute_task(self, task, *args, **kwargs):
        """Mock task execution"""
        self.execution_count += 1
        return "Task completed"


class MockTask:
    """Mock task for crew testing"""
    
    def __init__(self, name: str, callback: Callable = None):
        self.name = name
        self.description = f"Task for {name}"
        self.expected_output = "Done"
        self.callback = callback
        self.agent = None
        self.execution_count = 0
    
    def execute(self, *args, **kwargs):
        """Execute task with optional callback"""
        self.execution_count += 1
        output = f"Task {self.name} executed"
        
        if self.callback:
            self.callback(output)
        
        return output


class MockCrew:
    """Mock crew for thread safety testing"""
    
    def __init__(self, agents: List[MockAgent], tasks: List[MockTask], verbose: bool = False):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
        self.id = str(uuid.uuid4())
        self._context = None
        self.execution_count = 0
    
    def kickoff(self, inputs: Dict[str, Any] = None) -> 'MockCrewResult':
        """Synchronous crew execution"""
        self.execution_count += 1
        
        # Set crew context for this execution
        self._set_crew_context()
        
        try:
            # Execute tasks
            results = []
            for task in self.tasks:
                if task.agent:
                    result = task.agent.execute_task(task)
                else:
                    result = task.execute()
                results.append(result)
            
            return MockCrewResult(raw="Crew execution completed", results=results)
        
        finally:
            # Clear context after execution
            self._clear_crew_context()
    
    async def kickoff_async(self, inputs: Dict[str, Any] = None) -> 'MockCrewResult':
        """Asynchronous crew execution"""
        self.execution_count += 1
        
        # Set crew context for async execution
        self._set_crew_context()
        
        try:
            # Simulate async task execution
            await asyncio.sleep(0.01)
            
            results = []
            for task in self.tasks:
                await asyncio.sleep(0.01)
                if task.agent:
                    result = task.agent.execute_task(task)
                else:
                    result = task.execute()
                results.append(result)
            
            return MockCrewResult(raw="Async crew execution completed", results=results)
        
        finally:
            self._clear_crew_context()
    
    def kickoff_for_each(self, inputs: List[Dict[str, Any]]) -> List['MockCrewResult']:
        """Execute crew for each input (parallel execution)"""
        results = []
        
        for input_data in inputs:
            # Each execution gets its own context
            result = self.kickoff(input_data)
            results.append(result)
        
        return results
    
    def _set_crew_context(self):
        """Set crew context (simulated thread-local storage)"""
        self._context = MockCrewContext(self.id)
        # In real implementation, this would set thread-local context
        _thread_local_context.set_context(self._context)
    
    def _clear_crew_context(self):
        """Clear crew context"""
        _thread_local_context.clear_context()
        self._context = None


class MockCrewResult:
    """Mock crew execution result"""
    
    def __init__(self, raw: str, results: List[str] = None):
        self.raw = raw
        self.results = results or []


class ThreadLocalContext:
    """Simulated thread-local context storage"""
    
    def __init__(self):
        self._contexts = {}
    
    def set_context(self, context: MockCrewContext):
        """Set context for current thread"""
        thread_id = threading.get_ident()
        self._contexts[thread_id] = context
    
    def get_context(self) -> Optional[MockCrewContext]:
        """Get context for current thread"""
        thread_id = threading.get_ident()
        return self._contexts.get(thread_id)
    
    def clear_context(self):
        """Clear context for current thread"""
        thread_id = threading.get_ident()
        if thread_id in self._contexts:
            del self._contexts[thread_id]


# Global thread-local context instance
_thread_local_context = ThreadLocalContext()

def get_crew_context() -> Optional[MockCrewContext]:
    """Get current crew context"""
    return _thread_local_context.get_context()


class ThreadSafetyTestFramework:
    """Framework for testing crew thread safety"""
    
    def __init__(self):
        self.test_results = []
        self.context_captures = []
    
    def create_test_crew(self, name: str, task_callback: Callable = None) -> MockCrew:
        """Create a test crew with agent and task"""
        agent = MockAgent(name)
        task = MockTask(name, callback=task_callback)
        task.agent = agent
        
        return MockCrew(agents=[agent], tasks=[task], verbose=False)
    
    def test_parallel_crews_context_isolation(self, num_crews: int = 5) -> Dict[str, Any]:
        """Test context isolation between parallel crews"""
        contexts_captured = []
        
        def run_crew_with_context_check(crew_id: str) -> Dict[str, Any]:
            results = {"crew_id": crew_id, "contexts": []}
            
            def capture_context_in_task(output):
                context = get_crew_context()
                results["contexts"].append({
                    "stage": "task_callback",
                    "crew_id": context.id if context else None,
                    "crew_key": context.key if context else None,
                    "thread": threading.current_thread().name
                })
                return output
            
            # Check context before kickoff
            context_before = get_crew_context()
            results["contexts"].append({
                "stage": "before_kickoff",
                "crew_id": context_before.id if context_before else None,
                "thread": threading.current_thread().name
            })
            
            # Create and execute crew
            crew = self.create_test_crew(crew_id, task_callback=capture_context_in_task)
            output = crew.kickoff()
            
            # Check context after kickoff
            context_after = get_crew_context()
            results["contexts"].append({
                "stage": "after_kickoff",
                "crew_id": context_after.id if context_after else None,
                "thread": threading.current_thread().name
            })
            
            results["crew_uuid"] = str(crew.id)
            results["output"] = output.raw
            
            return results
        
        # Execute crews in parallel
        with ThreadPoolExecutor(max_workers=num_crews) as executor:
            futures = []
            for i in range(num_crews):
                future = executor.submit(run_crew_with_context_check, f"crew_{i}")
                futures.append(future)
            
            results = [f.result() for f in futures]
        
        # Validate context isolation
        validation_results = []
        for result in results:
            crew_uuid = result["crew_uuid"]
            
            # Find contexts
            before_ctx = next(ctx for ctx in result["contexts"] if ctx["stage"] == "before_kickoff")
            task_ctx = next(ctx for ctx in result["contexts"] if ctx["stage"] == "task_callback")
            after_ctx = next(ctx for ctx in result["contexts"] if ctx["stage"] == "after_kickoff")
            
            validation = {
                "crew_id": result["crew_id"],
                "context_none_before": before_ctx["crew_id"] is None,
                "context_correct_during": task_ctx["crew_id"] == crew_uuid,
                "context_none_after": after_ctx["crew_id"] is None,
                "thread_pool_used": "ThreadPoolExecutor" in before_ctx["thread"]
            }
            validation_results.append(validation)
        
        return {
            "parallel_results": results,
            "validations": validation_results,
            "all_contexts_isolated": all(
                v["context_none_before"] and v["context_correct_during"] and v["context_none_after"]
                for v in validation_results
            )
        }
    
    async def test_async_crews_context_isolation(self, num_crews: int = 5) -> Dict[str, Any]:
        """Test context isolation for async crew execution"""
        async def run_crew_async(crew_id: str) -> Dict[str, Any]:
            task_context = {"crew_id": crew_id, "context": None}
            
            def capture_context(output):
                ctx = get_crew_context()
                task_context["context"] = {
                    "crew_id": ctx.id if ctx else None,
                    "crew_key": ctx.key if ctx else None
                }
                return output
            
            crew = self.create_test_crew(crew_id, task_callback=capture_context)
            output = await crew.kickoff_async()
            
            return {
                "crew_id": crew_id,
                "crew_uuid": str(crew.id),
                "output": output.raw,
                "task_context": task_context
            }
        
        # Execute async crews concurrently
        tasks = [run_crew_async(f"async_crew_{i}") for i in range(num_crews)]
        results = await asyncio.gather(*tasks)
        
        # Validate async context isolation
        validation_results = []
        for result in results:
            crew_uuid = result["crew_uuid"]
            task_ctx = result["task_context"]["context"]
            
            validation = {
                "crew_id": result["crew_id"],
                "context_exists_during_task": task_ctx is not None,
                "context_correct_during_task": task_ctx["crew_id"] == crew_uuid if task_ctx else False
            }
            validation_results.append(validation)
        
        return {
            "async_results": results,
            "validations": validation_results,
            "all_contexts_correct": all(
                v["context_exists_during_task"] and v["context_correct_during_task"]
                for v in validation_results
            )
        }
    
    def test_for_each_context_uniqueness(self, inputs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Test context uniqueness in kickoff_for_each"""
        if inputs is None:
            inputs = [{"item": f"input_{i}"} for i in range(3)]
        
        contexts_captured = []
        
        def capture_context(output):
            ctx = get_crew_context()
            contexts_captured.append({
                "context_id": ctx.id if ctx else None,
                "thread": threading.current_thread().name
            })
            return output
        
        crew = self.create_test_crew("for_each_test", task_callback=capture_context)
        results = crew.kickoff_for_each(inputs=inputs)
        
        # Validate uniqueness
        context_ids = [ctx["context_id"] for ctx in contexts_captured]
        unique_contexts = len(set(context_ids))
        
        return {
            "execution_results": results,
            "contexts_captured": contexts_captured,
            "unique_contexts": unique_contexts,
            "expected_contexts": len(inputs),
            "all_contexts_unique": unique_contexts == len(inputs)
        }
    
    def test_context_leakage_prevention(self) -> Dict[str, Any]:
        """Test that contexts don't leak between crew executions"""
        contexts = []
        
        def check_context(output):
            ctx = get_crew_context()
            contexts.append({
                "context_id": ctx.id if ctx else None,
                "context_key": ctx.key if ctx else None
            })
            return output
        
        def run_crew(name: str) -> str:
            crew = self.create_test_crew(name, task_callback=check_context)
            crew.kickoff()
            return str(crew.id)
        
        # Run two crews sequentially
        crew1_id = run_crew("First")
        crew2_id = run_crew("Second")
        
        # Validate no leakage
        return {
            "contexts": contexts,
            "crew1_id": crew1_id,
            "crew2_id": crew2_id,
            "correct_context_isolation": (
                len(contexts) == 2 and
                contexts[0]["context_id"] == crew1_id and
                contexts[1]["context_id"] == crew2_id and
                contexts[0]["context_id"] != contexts[1]["context_id"]
            )
        }


class ThreadSafetyValidator:
    """Validates thread safety test results"""
    
    def __init__(self):
        self.framework = ThreadSafetyTestFramework()
    
    def validate_parallel_execution(self, num_crews: int = 5) -> Dict[str, Any]:
        """Validate parallel crew execution safety"""
        result = self.framework.test_parallel_crews_context_isolation(num_crews)
        
        success_metrics = {
            "all_crews_executed": len(result["parallel_results"]) == num_crews,
            "all_contexts_isolated": result["all_contexts_isolated"],
            "no_context_leakage": all(
                v["context_none_before"] and v["context_none_after"] 
                for v in result["validations"]
            ),
            "correct_context_during_execution": all(
                v["context_correct_during"] for v in result["validations"]
            ),
            "thread_pool_usage": all(
                v["thread_pool_used"] for v in result["validations"]
            )
        }
        
        return {
            "test_result": result,
            "success_metrics": success_metrics,
            "overall_success": all(success_metrics.values())
        }
    
    async def validate_async_execution(self, num_crews: int = 5) -> Dict[str, Any]:
        """Validate async crew execution safety"""
        result = await self.framework.test_async_crews_context_isolation(num_crews)
        
        success_metrics = {
            "all_crews_executed": len(result["async_results"]) == num_crews,
            "all_contexts_correct": result["all_contexts_correct"],
            "contexts_exist_during_tasks": all(
                v["context_exists_during_task"] for v in result["validations"]
            )
        }
        
        return {
            "test_result": result,
            "success_metrics": success_metrics,
            "overall_success": all(success_metrics.values())
        }
    
    def validate_for_each_uniqueness(self) -> Dict[str, Any]:
        """Validate for_each execution context uniqueness"""
        result = self.framework.test_for_each_context_uniqueness()
        
        success_metrics = {
            "all_executions_completed": len(result["execution_results"]) == result["expected_contexts"],
            "all_contexts_unique": result["all_contexts_unique"],
            "correct_context_count": result["unique_contexts"] == result["expected_contexts"]
        }
        
        return {
            "test_result": result,
            "success_metrics": success_metrics,
            "overall_success": all(success_metrics.values())
        }
    
    def validate_context_leakage_prevention(self) -> Dict[str, Any]:
        """Validate prevention of context leakage"""
        result = self.framework.test_context_leakage_prevention()
        
        success_metrics = {
            "correct_isolation": result["correct_context_isolation"],
            "two_contexts_captured": len(result["contexts"]) == 2,
            "different_context_ids": result["contexts"][0]["context_id"] != result["contexts"][1]["context_id"]
        }
        
        return {
            "test_result": result,
            "success_metrics": success_metrics,
            "overall_success": all(success_metrics.values())
        }
    
    async def run_comprehensive_thread_safety_tests(self) -> Dict[str, Any]:
        """Run comprehensive thread safety validation"""
        results = {}
        
        # Test parallel execution
        results["parallel"] = self.validate_parallel_execution()
        
        # Test async execution
        results["async"] = await self.validate_async_execution()
        
        # Test for_each uniqueness
        results["for_each"] = self.validate_for_each_uniqueness()
        
        # Test context leakage prevention
        results["leakage_prevention"] = self.validate_context_leakage_prevention()
        
        # Calculate overall success
        overall_success = all(
            result.get("overall_success", False) 
            for result in results.values()
        )
        
        results["summary"] = {
            "total_test_categories": len(results) - 1,
            "passed_categories": sum(1 for k, v in results.items() 
                                   if k != "summary" and v.get("overall_success", False)),
            "overall_success": overall_success
        }
        
        return results


# Pytest integration patterns
class PyTestThreadSafetyPatterns:
    """Thread safety testing patterns for pytest"""
    
    @pytest.fixture
    def simple_agent_factory(self):
        """Create agent factory fixture"""
        def create_agent(name: str) -> MockAgent:
            return MockAgent(name)
        return create_agent
    
    @pytest.fixture
    def simple_task_factory(self):
        """Create task factory fixture"""
        def create_task(name: str, callback: Callable = None) -> MockTask:
            return MockTask(name, callback=callback)
        return create_task
    
    @pytest.fixture
    def crew_factory(self, simple_agent_factory, simple_task_factory):
        """Create crew factory fixture"""
        def create_crew(name: str, task_callback: Callable = None) -> MockCrew:
            agent = simple_agent_factory(name)
            task = simple_task_factory(name, callback=task_callback)
            task.agent = agent
            
            return MockCrew(agents=[agent], tasks=[task], verbose=False)
        return create_crew
    
    @patch("crewai.Agent.execute_task")
    def test_parallel_crews_thread_safety(self, mock_execute_task, crew_factory):
        """Test parallel crew execution with thread safety"""
        mock_execute_task.return_value = "Task completed"
        
        framework = ThreadSafetyTestFramework()
        result = framework.test_parallel_crews_context_isolation(num_crews=3)
        
        assert result["all_contexts_isolated"] is True
        assert len(result["parallel_results"]) == 3
        
        # Validate each crew had proper context isolation
        for validation in result["validations"]:
            assert validation["context_none_before"] is True
            assert validation["context_correct_during"] is True
            assert validation["context_none_after"] is True
    
    @pytest.mark.asyncio
    @patch("crewai.Agent.execute_task") 
    async def test_async_crews_thread_safety(self, mock_execute_task, crew_factory):
        """Test async crew execution with thread safety"""
        mock_execute_task.return_value = "Task completed"
        
        framework = ThreadSafetyTestFramework()
        result = await framework.test_async_crews_context_isolation(num_crews=3)
        
        assert result["all_contexts_correct"] is True
        assert len(result["async_results"]) == 3
        
        # Validate async context handling
        for validation in result["validations"]:
            assert validation["context_exists_during_task"] is True
            assert validation["context_correct_during_task"] is True
    
    @patch("crewai.Agent.execute_task")
    def test_concurrent_kickoff_for_each(self, mock_execute_task, crew_factory):
        """Test concurrent for_each execution"""
        mock_execute_task.return_value = "Task completed"
        
        framework = ThreadSafetyTestFramework()
        result = framework.test_for_each_context_uniqueness()
        
        assert result["all_contexts_unique"] is True
        assert len(result["execution_results"]) == result["expected_contexts"]
    
    @patch("crewai.Agent.execute_task")
    def test_no_context_leakage_between_crews(self, mock_execute_task, crew_factory):
        """Test no context leakage between sequential crews"""
        mock_execute_task.return_value = "Task completed"
        
        framework = ThreadSafetyTestFramework()
        result = framework.test_context_leakage_prevention()
        
        assert result["correct_context_isolation"] is True
        assert len(result["contexts"]) == 2
        assert result["contexts"][0]["context_id"] != result["contexts"][1]["context_id"]


# Export patterns for integration
__all__ = [
    'ThreadSafetyTestFramework',
    'ThreadSafetyValidator',
    'MockCrew',
    'MockAgent',
    'MockTask',
    'MockCrewContext',
    'get_crew_context',
    'PyTestThreadSafetyPatterns'
]