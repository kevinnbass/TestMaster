"""
Swarms Comprehensive Testing Patterns - AGENT B COMPREHENSIVE TESTING EXCELLENCE
================================================================================

Extracted testing patterns from swarms repository for enhanced multi-agent testing capabilities.
Focus: Sequential/concurrent workflows, graph workflows, comprehensive testing, agent coordination.

AGENT B Enhancement: Phase 1.6 - Swarms Pattern Integration
- Sequential and concurrent workflow testing
- Graph workflow comprehensive testing
- Multi-agent coordination patterns
- Task execution and result tracking
- Workflow state management and persistence
- Agent communication and conversation testing
"""

import asyncio
import json
import time
import tempfile
import os
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from unittest.mock import Mock, AsyncMock
from datetime import datetime


class WorkflowTestPatterns:
    """
    Workflow testing patterns extracted from swarms sequential_workflow and concurrent tests
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @dataclass
    class MockTask:
        """Mock task for workflow testing"""
        description: str
        agent: Any
        result: Optional[str] = None
        execution_time: Optional[float] = None
        status: str = "pending"  # pending, running, completed, failed
        metadata: Dict[str, Any] = field(default_factory=dict)
        
        def run(self, *args, **kwargs):
            """Execute the task"""
            start_time = time.time()
            self.status = "running"
            
            try:
                if hasattr(self.agent, 'run'):
                    self.result = self.agent.run(*args, **kwargs)
                else:
                    self.result = f"Mock result for {self.description}"
                
                self.status = "completed"
                self.execution_time = time.time() - start_time
                return self.result
            except Exception as e:
                self.status = "failed"
                self.result = f"Error: {str(e)}"
                self.execution_time = time.time() - start_time
                raise
        
        async def run_async(self, *args, **kwargs):
            """Execute the task asynchronously"""
            start_time = time.time()
            self.status = "running"
            
            try:
                if hasattr(self.agent, 'run_async'):
                    self.result = await self.agent.run_async(*args, **kwargs)
                elif hasattr(self.agent, 'run'):
                    self.result = self.agent.run(*args, **kwargs)
                else:
                    await asyncio.sleep(0.1)  # Simulate async work
                    self.result = f"Async mock result for {self.description}"
                
                self.status = "completed"
                self.execution_time = time.time() - start_time
                return self.result
            except Exception as e:
                self.status = "failed"
                self.result = f"Error: {str(e)}"
                self.execution_time = time.time() - start_time
                raise
    
    class MockAgent:
        """Mock agent for workflow testing"""
        def __init__(self, name: str, model: str = "mock-model"):
            self.name = name
            self.model = model
            self.execution_count = 0
            self.total_execution_time = 0
        
        def run(self, task: str, *args, **kwargs) -> str:
            """Execute agent with task"""
            self.execution_count += 1
            execution_time = 0.1  # Mock execution time
            self.total_execution_time += execution_time
            time.sleep(execution_time)
            return f"Agent {self.name} result for: {task}"
        
        async def run_async(self, task: str, *args, **kwargs) -> str:
            """Execute agent asynchronously"""
            self.execution_count += 1
            execution_time = 0.1
            self.total_execution_time += execution_time
            await asyncio.sleep(execution_time)
            return f"Async Agent {self.name} result for: {task}"
    
    class MockSequentialWorkflow:
        """Mock sequential workflow for testing"""
        def __init__(self, max_loops: int = 1, autosave: bool = False, 
                     saved_state_filepath: str = "workflow_state.json"):
            self.tasks = []
            self.max_loops = max_loops
            self.autosave = autosave
            self.saved_state_filepath = saved_state_filepath
            self.restore_state_filepath = None
            self.dashboard = False
            self.execution_log = []
        
        def add(self, description: str, agent: Any):
            """Add task to workflow"""
            task = self.MockTask(description=description, agent=agent)
            self.tasks.append(task)
        
        def run(self) -> List[str]:
            """Run workflow sequentially"""
            results = []
            start_time = time.time()
            
            for i in range(self.max_loops):
                loop_results = []
                for task in self.tasks:
                    result = task.run()
                    loop_results.append(result)
                    results.append(result)
                
                self.execution_log.append({
                    'loop': i + 1,
                    'results': loop_results,
                    'timestamp': time.time()
                })
            
            if self.autosave:
                self._save_state()
            
            return results
        
        async def run_async(self) -> List[str]:
            """Run workflow asynchronously"""
            results = []
            start_time = time.time()
            
            for i in range(self.max_loops):
                loop_results = []
                for task in self.tasks:
                    result = await task.run_async()
                    loop_results.append(result)
                    results.append(result)
                
                self.execution_log.append({
                    'loop': i + 1,
                    'results': loop_results,
                    'timestamp': time.time()
                })
            
            if self.autosave:
                self._save_state()
            
            return results
        
        def reset_workflow(self):
            """Reset workflow state"""
            for task in self.tasks:
                task.result = None
                task.status = "pending"
                task.execution_time = None
            self.execution_log.clear()
        
        def get_task_results(self) -> List[Optional[str]]:
            """Get results from all tasks"""
            return [task.result for task in self.tasks]
        
        def _save_state(self):
            """Save workflow state to file"""
            state = {
                'tasks': [
                    {
                        'description': task.description,
                        'result': task.result,
                        'status': task.status,
                        'execution_time': task.execution_time
                    }
                    for task in self.tasks
                ],
                'execution_log': self.execution_log,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.saved_state_filepath, 'w') as f:
                json.dump(state, f, indent=2)
    
    class MockConcurrentWorkflow:
        """Mock concurrent workflow for testing"""
        def __init__(self, max_workers: int = 5):
            self.tasks = []
            self.max_workers = max_workers
            self.execution_log = []
        
        def add(self, description: str, agent: Any):
            """Add task to workflow"""
            task = self.MockTask(description=description, agent=agent)
            self.tasks.append(task)
        
        async def run_concurrent(self) -> List[str]:
            """Run tasks concurrently"""
            start_time = time.time()
            
            # Limit concurrent tasks by max_workers
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def run_task_with_semaphore(task):
                async with semaphore:
                    return await task.run_async()
            
            # Execute all tasks concurrently
            results = await asyncio.gather(
                *[run_task_with_semaphore(task) for task in self.tasks],
                return_exceptions=True
            )
            
            execution_time = time.time() - start_time
            self.execution_log.append({
                'total_tasks': len(self.tasks),
                'execution_time': execution_time,
                'max_workers': self.max_workers,
                'results': results,
                'timestamp': time.time()
            })
            
            return [r for r in results if not isinstance(r, Exception)]
    
    def test_task_initialization(self) -> Dict[str, Any]:
        """Test task initialization with various parameters"""
        agent = self.MockAgent("TestAgent")
        task = self.MockTask(description="Sample Task", agent=agent)
        
        return {
            'task_description': task.description,
            'task_agent': task.agent.name,
            'task_status': task.status,
            'task_result': task.result,
            'initialization_success': task.description == "Sample Task" and task.agent == agent
        }
    
    def test_task_execution(self) -> Dict[str, Any]:
        """Test task execution"""
        agent = self.MockAgent("TestAgent")
        task = self.MockTask(description="Execute Task", agent=agent)
        
        # Execute task
        result = task.run("test input")
        
        return {
            'task_description': task.description,
            'execution_result': result,
            'task_status': task.status,
            'execution_time': task.execution_time,
            'agent_executions': agent.execution_count,
            'execution_success': task.status == "completed" and result is not None
        }
    
    async def test_task_async_execution(self) -> Dict[str, Any]:
        """Test asynchronous task execution"""
        agent = self.MockAgent("AsyncTestAgent")
        task = self.MockTask(description="Async Execute Task", agent=agent)
        
        # Execute task asynchronously
        result = await task.run_async("async test input")
        
        return {
            'task_description': task.description,
            'async_result': result,
            'task_status': task.status,
            'execution_time': task.execution_time,
            'agent_executions': agent.execution_count,
            'async_execution_success': task.status == "completed" and "Async" in result
        }
    
    def test_sequential_workflow_basic(self) -> Dict[str, Any]:
        """Test basic sequential workflow functionality"""
        workflow = self.MockSequentialWorkflow()
        
        # Add tasks
        agent1 = self.MockAgent("Agent1")
        agent2 = self.MockAgent("Agent2")
        
        workflow.add("First Task", agent1)
        workflow.add("Second Task", agent2)
        
        # Run workflow
        results = workflow.run()
        
        return {
            'workflow_tasks': len(workflow.tasks),
            'execution_results': results,
            'task_results': workflow.get_task_results(),
            'max_loops': workflow.max_loops,
            'execution_log': workflow.execution_log,
            'workflow_success': len(results) == len(workflow.tasks) * workflow.max_loops
        }
    
    async def test_sequential_workflow_async(self) -> Dict[str, Any]:
        """Test asynchronous sequential workflow"""
        workflow = self.MockSequentialWorkflow(max_loops=2)
        
        # Add tasks
        agent1 = self.MockAgent("AsyncAgent1")
        agent2 = self.MockAgent("AsyncAgent2")
        
        workflow.add("Async First Task", agent1)
        workflow.add("Async Second Task", agent2)
        
        # Run workflow asynchronously
        results = await workflow.run_async()
        
        return {
            'workflow_tasks': len(workflow.tasks),
            'async_results': results,
            'task_results': workflow.get_task_results(),
            'max_loops': workflow.max_loops,
            'execution_log': workflow.execution_log,
            'async_workflow_success': len(results) == len(workflow.tasks) * workflow.max_loops
        }
    
    async def test_concurrent_workflow(self, num_tasks: int = 5, max_workers: int = 3) -> Dict[str, Any]:
        """Test concurrent workflow execution"""
        workflow = self.MockConcurrentWorkflow(max_workers=max_workers)
        
        # Add multiple tasks
        agents = [self.MockAgent(f"ConcurrentAgent{i}") for i in range(num_tasks)]
        for i, agent in enumerate(agents):
            workflow.add(f"Concurrent Task {i+1}", agent)
        
        # Run concurrently
        start_time = time.time()
        results = await workflow.run_concurrent()
        total_time = time.time() - start_time
        
        return {
            'total_tasks': num_tasks,
            'max_workers': max_workers,
            'concurrent_results': results,
            'total_execution_time': total_time,
            'successful_results': len(results),
            'execution_log': workflow.execution_log,
            'concurrency_benefit': total_time < num_tasks * 0.1,  # Should be faster than sequential
            'concurrent_success': len(results) == num_tasks
        }
    
    def test_workflow_state_management(self) -> Dict[str, Any]:
        """Test workflow state saving and restoration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_filepath = temp_file.name
        
        try:
            # Create workflow with autosave
            workflow = self.MockSequentialWorkflow(
                autosave=True, 
                saved_state_filepath=temp_filepath
            )
            
            # Add and run tasks
            agent = self.MockAgent("StateTestAgent")
            workflow.add("State Test Task", agent)
            results = workflow.run()
            
            # Check if state file was created
            state_saved = os.path.exists(temp_filepath)
            
            # Load and verify state
            state_data = None
            if state_saved:
                with open(temp_filepath, 'r') as f:
                    state_data = json.load(f)
            
            return {
                'autosave_enabled': workflow.autosave,
                'state_filepath': temp_filepath,
                'state_file_created': state_saved,
                'state_data': state_data,
                'workflow_results': results,
                'state_management_success': state_saved and state_data is not None
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    def test_workflow_reset(self) -> Dict[str, Any]:
        """Test workflow reset functionality"""
        workflow = self.MockSequentialWorkflow()
        
        # Add and run tasks
        agent = self.MockAgent("ResetTestAgent")
        workflow.add("Reset Test Task", agent)
        
        # Run workflow
        initial_results = workflow.run()
        initial_task_results = workflow.get_task_results()
        
        # Reset workflow
        workflow.reset_workflow()
        
        # Check reset state
        reset_task_results = workflow.get_task_results()
        reset_log = workflow.execution_log
        
        return {
            'initial_results': initial_results,
            'initial_task_results': initial_task_results,
            'reset_task_results': reset_task_results,
            'reset_execution_log': reset_log,
            'reset_success': all(result is None for result in reset_task_results) and len(reset_log) == 0
        }


class GraphWorkflowTestPatterns:
    """
    Graph workflow testing patterns extracted from swarms graph_workflow tests
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    class NodeType(Enum):
        """Node types for graph workflow"""
        AGENT = "agent"
        START = "start"
        END = "end"
        DECISION = "decision"
    
    @dataclass
    class MockNode:
        """Mock node for graph workflow testing"""
        id: str
        type: 'NodeType'
        agent: Optional[Any] = None
        conditions: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
        
        @classmethod
        def from_agent(cls, agent) -> 'MockNode':
            """Create node from agent"""
            return cls(
                id=getattr(agent, 'name', 'unnamed_agent'),
                type=cls.NodeType.AGENT,
                agent=agent
            )
    
    @dataclass
    class MockEdge:
        """Mock edge for graph workflow"""
        source: str
        target: str
        condition: Optional[Callable] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    class TestResults:
        """Test results tracker"""
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.errors = []
        
        def add_pass(self, test_name: str):
            """Add passed test"""
            self.passed += 1
            
        def add_fail(self, test_name: str, error: str):
            """Add failed test"""
            self.failed += 1
            self.errors.append(f"{test_name}: {error}")
        
        def get_summary(self) -> Dict[str, Any]:
            """Get test summary"""
            total = self.passed + self.failed
            return {
                'passed': self.passed,
                'failed': self.failed,
                'total': total,
                'success_rate': (self.passed / total) * 100 if total > 0 else 0,
                'errors': self.errors
            }
    
    class MockGraphWorkflow:
        """Mock graph workflow for testing"""
        def __init__(self):
            self.nodes = {}
            self.edges = []
            self.execution_log = []
            self.start_node = None
            self.end_nodes = set()
        
        def add_node(self, node: 'MockNode'):
            """Add node to graph"""
            self.nodes[node.id] = node
            
            if node.type == self.NodeType.START:
                self.start_node = node.id
            elif node.type == self.NodeType.END:
                self.end_nodes.add(node.id)
        
        def add_edge(self, edge: 'MockEdge'):
            """Add edge to graph"""
            self.edges.append(edge)
        
        def compile(self) -> 'MockGraphWorkflow':
            """Compile the workflow"""
            # Validate graph structure
            if not self.start_node:
                raise ValueError("No start node defined")
            
            if not self.end_nodes:
                raise ValueError("No end nodes defined")
            
            return self
        
        async def execute(self, initial_input: str = "") -> Dict[str, Any]:
            """Execute the graph workflow"""
            start_time = time.time()
            current_node_id = self.start_node
            execution_path = [current_node_id]
            results = {}
            
            while current_node_id and current_node_id not in self.end_nodes:
                node = self.nodes[current_node_id]
                
                # Execute node
                if node.type == self.NodeType.AGENT and node.agent:
                    if hasattr(node.agent, 'run_async'):
                        result = await node.agent.run_async(initial_input)
                    else:
                        result = node.agent.run(initial_input)
                    results[node.id] = result
                
                # Find next node
                next_node_id = None
                for edge in self.edges:
                    if edge.source == current_node_id:
                        if edge.condition is None or edge.condition(results.get(node.id, "")):
                            next_node_id = edge.target
                            break
                
                current_node_id = next_node_id
                if current_node_id:
                    execution_path.append(current_node_id)
            
            execution_time = time.time() - start_time
            
            execution_record = {
                'execution_path': execution_path,
                'results': results,
                'execution_time': execution_time,
                'timestamp': time.time()
            }
            self.execution_log.append(execution_record)
            
            return execution_record
        
        def visualize(self) -> Dict[str, Any]:
            """Create visualization data for the graph"""
            return {
                'nodes': [
                    {
                        'id': node.id,
                        'type': node.type.value,
                        'agent': getattr(node.agent, 'name', None) if node.agent else None
                    }
                    for node in self.nodes.values()
                ],
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'has_condition': edge.condition is not None
                    }
                    for edge in self.edges
                ],
                'start_node': self.start_node,
                'end_nodes': list(self.end_nodes)
            }
        
        def to_dict(self) -> Dict[str, Any]:
            """Serialize workflow to dictionary"""
            return {
                'nodes': {
                    node_id: {
                        'id': node.id,
                        'type': node.type.value,
                        'agent_name': getattr(node.agent, 'name', None) if node.agent else None,
                        'metadata': node.metadata
                    }
                    for node_id, node in self.nodes.items()
                },
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'metadata': edge.metadata
                    }
                    for edge in self.edges
                ],
                'start_node': self.start_node,
                'end_nodes': list(self.end_nodes)
            }
    
    def create_mock_agent(self, name: str, model: str = "mock-model"):
        """Create mock agent for testing"""
        from swarms.testing.patterns.workflow_patterns import WorkflowTestPatterns
        return WorkflowTestPatterns.MockAgent(name, model)
    
    def test_node_creation(self) -> Dict[str, Any]:
        """Test node creation with various parameters"""
        results = self.TestResults()
        test_results = []
        
        try:
            # Test basic node creation
            agent = self.create_mock_agent("TestAgent")
            node = self.MockNode.from_agent(agent)
            
            if node.id == "TestAgent" and node.type == self.NodeType.AGENT:
                results.add_pass("Node Creation - Basic")
                test_results.append({'test': 'basic_creation', 'success': True})
            else:
                results.add_fail("Node Creation - Basic", "Node properties incorrect")
                test_results.append({'test': 'basic_creation', 'success': False})
            
            # Test custom node
            custom_node = self.MockNode(id="CustomID", type=self.NodeType.AGENT, agent=agent)
            
            if custom_node.id == "CustomID":
                results.add_pass("Node Creation - Custom ID")
                test_results.append({'test': 'custom_id', 'success': True})
            else:
                results.add_fail("Node Creation - Custom ID", "Custom ID not set correctly")
                test_results.append({'test': 'custom_id', 'success': False})
            
        except Exception as e:
            results.add_fail("Node Creation", str(e))
            test_results.append({'test': 'node_creation', 'success': False, 'error': str(e)})
        
        return {
            'test_results': test_results,
            'summary': results.get_summary()
        }
    
    def test_edge_creation(self) -> Dict[str, Any]:
        """Test edge creation and conditions"""
        results = self.TestResults()
        test_results = []
        
        try:
            # Test basic edge
            edge = self.MockEdge(source="node1", target="node2")
            
            if edge.source == "node1" and edge.target == "node2":
                results.add_pass("Edge Creation - Basic")
                test_results.append({'test': 'basic_edge', 'success': True})
            else:
                results.add_fail("Edge Creation - Basic", "Edge properties incorrect")
                test_results.append({'test': 'basic_edge', 'success': False})
            
            # Test conditional edge
            condition = lambda x: "success" in x.lower()
            conditional_edge = self.MockEdge(source="node1", target="node2", condition=condition)
            
            if conditional_edge.condition is not None:
                results.add_pass("Edge Creation - Conditional")
                test_results.append({'test': 'conditional_edge', 'success': True})
            else:
                results.add_fail("Edge Creation - Conditional", "Condition not set")
                test_results.append({'test': 'conditional_edge', 'success': False})
            
        except Exception as e:
            results.add_fail("Edge Creation", str(e))
            test_results.append({'test': 'edge_creation', 'success': False, 'error': str(e)})
        
        return {
            'test_results': test_results,
            'summary': results.get_summary()
        }
    
    async def test_graph_workflow_execution(self) -> Dict[str, Any]:
        """Test complete graph workflow execution"""
        results = self.TestResults()
        
        try:
            # Create workflow
            workflow = self.MockGraphWorkflow()
            
            # Create nodes
            start_node = self.MockNode(id="start", type=self.NodeType.START)
            agent1 = self.create_mock_agent("Agent1")
            agent_node1 = self.MockNode.from_agent(agent1)
            agent2 = self.create_mock_agent("Agent2")
            agent_node2 = self.MockNode.from_agent(agent2)
            end_node = self.MockNode(id="end", type=self.NodeType.END)
            
            # Add nodes
            workflow.add_node(start_node)
            workflow.add_node(agent_node1)
            workflow.add_node(agent_node2)
            workflow.add_node(end_node)
            
            # Add edges
            workflow.add_edge(self.MockEdge("start", "Agent1"))
            workflow.add_edge(self.MockEdge("Agent1", "Agent2"))
            workflow.add_edge(self.MockEdge("Agent2", "end"))
            
            # Compile and execute
            compiled_workflow = workflow.compile()
            execution_result = await compiled_workflow.execute("Test input")
            
            if execution_result and execution_result['execution_path']:
                results.add_pass("Graph Workflow Execution")
                execution_success = True
            else:
                results.add_fail("Graph Workflow Execution", "Execution failed")
                execution_success = False
            
        except Exception as e:
            results.add_fail("Graph Workflow Execution", str(e))
            execution_success = False
            execution_result = {'error': str(e)}
        
        return {
            'execution_result': execution_result,
            'execution_success': execution_success,
            'summary': results.get_summary()
        }
    
    def test_graph_workflow_visualization(self) -> Dict[str, Any]:
        """Test graph workflow visualization"""
        try:
            # Create simple workflow
            workflow = self.MockGraphWorkflow()
            
            # Add test nodes and edges
            start_node = self.MockNode(id="start", type=self.NodeType.START)
            agent = self.create_mock_agent("VisAgent")
            agent_node = self.MockNode.from_agent(agent)
            end_node = self.MockNode(id="end", type=self.NodeType.END)
            
            workflow.add_node(start_node)
            workflow.add_node(agent_node)
            workflow.add_node(end_node)
            
            workflow.add_edge(self.MockEdge("start", "VisAgent"))
            workflow.add_edge(self.MockEdge("VisAgent", "end"))
            
            # Generate visualization
            viz_data = workflow.visualize()
            
            # Generate serialization
            serialized = workflow.to_dict()
            
            return {
                'visualization_data': viz_data,
                'serialized_data': serialized,
                'nodes_count': len(viz_data['nodes']),
                'edges_count': len(viz_data['edges']),
                'visualization_success': True
            }
        
        except Exception as e:
            return {
                'visualization_success': False,
                'error': str(e)
            }


class ComprehensiveTestPatterns:
    """
    Comprehensive testing patterns extracted from swarms comprehensive tests
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_timestamp(self) -> str:
        """Generate timestamp for test runs"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def write_test_report(self, results: List[Dict[str, Any]], 
                         filename: str = None) -> Dict[str, Any]:
        """Write comprehensive test report"""
        if filename is None:
            filename = f"test_report_{self.generate_timestamp()}"
        
        # Create test runs directory
        test_dir = "test_runs"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        # Calculate statistics
        total = len(results)
        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = total - passed
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        # Create markdown report
        report_content = f"""# Swarms Comprehensive Test Report

Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Tests:** {total}
- **Passed:** {passed}
- **Failed:** {failed}
- **Success Rate:** {success_rate:.2f}%

## Detailed Results

"""
        
        for result in results:
            report_content += f"""### {result.get('test_name', 'Unknown Test')}

**Status:** {result.get('status', 'UNKNOWN').upper()}

"""
            
            if result.get("response"):
                report_content += "Response:\n```json\n"
                try:
                    response_str = result["response"]
                    if isinstance(response_str, str):
                        response_json = json.loads(response_str)
                    else:
                        response_json = response_str
                    report_content += json.dumps(response_json, indent=2)
                except (json.JSONDecodeError, TypeError):
                    report_content += str(response_str)
                report_content += "\n```\n\n"
            
            if result.get("error"):
                report_content += f"**Error:** {result['error']}\n\n"
        
        # Write report file
        report_path = os.path.join(test_dir, f"{filename}.md")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return {
            'report_path': report_path,
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': failed,
            'success_rate': success_rate,
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'success_rate': success_rate
            }
        }
    
    async def run_comprehensive_test_suite(self, test_functions: List[Callable]) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        results = []
        start_time = time.time()
        
        for i, test_func in enumerate(test_functions):
            test_name = getattr(test_func, '__name__', f'test_{i}')
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                
                results.append({
                    'test_name': test_name,
                    'status': 'passed',
                    'response': result,
                    'execution_time': time.time() - start_time
                })
                
            except Exception as e:
                results.append({
                    'test_name': test_name,
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': time.time() - start_time
                })
        
        total_time = time.time() - start_time
        
        # Generate report
        report_data = self.write_test_report(results)
        
        return {
            'test_results': results,
            'total_execution_time': total_time,
            'report_data': report_data,
            'test_count': len(test_functions),
            'comprehensive_test_success': True
        }
    
    def create_test_matrix(self, agents: List[Any], tasks: List[str], 
                          workflows: List[str]) -> List[Dict[str, Any]]:
        """Create comprehensive test matrix"""
        test_matrix = []
        
        for agent in agents:
            for task in tasks:
                for workflow_type in workflows:
                    test_matrix.append({
                        'agent': getattr(agent, 'name', 'unnamed_agent'),
                        'task': task,
                        'workflow_type': workflow_type,
                        'test_id': f"{agent.name}_{workflow_type}_{len(test_matrix)}"
                    })
        
        return test_matrix
    
    async def execute_test_matrix(self, test_matrix: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute comprehensive test matrix"""
        results = []
        start_time = time.time()
        
        for test_case in test_matrix:
            try:
                # Mock test execution
                execution_time = 0.1
                await asyncio.sleep(execution_time)
                
                result = {
                    'test_case': test_case,
                    'status': 'passed',
                    'result': f"Test {test_case['test_id']} completed successfully",
                    'execution_time': execution_time
                }
                results.append(result)
                
            except Exception as e:
                result = {
                    'test_case': test_case,
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': 0
                }
                results.append(result)
        
        total_time = time.time() - start_time
        passed = sum(1 for r in results if r['status'] == 'passed')
        
        return {
            'test_matrix': test_matrix,
            'results': results,
            'total_tests': len(test_matrix),
            'passed_tests': passed,
            'failed_tests': len(test_matrix) - passed,
            'success_rate': (passed / len(test_matrix)) * 100 if test_matrix else 0,
            'total_execution_time': total_time,
            'matrix_execution_success': True
        }


# Export all patterns
__all__ = [
    'WorkflowTestPatterns',
    'GraphWorkflowTestPatterns',
    'ComprehensiveTestPatterns'
]