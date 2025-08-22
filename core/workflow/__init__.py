"""
Intelligent Workflow Engine Package - Enterprise-grade workflow orchestration

This package provides comprehensive intelligent workflow management capabilities
including advanced scheduling, optimization, execution monitoring, and adaptive
resource management with machine learning-powered performance enhancement.

Key Components:
- Workflow Models: Core data structures and workflow definitions
- Task Scheduler: Intelligent task scheduling with load balancing
- Workflow Optimizer: Performance optimization and bottleneck resolution
- Execution Engine: Complete workflow execution orchestration

Enterprise Features:
- Multi-objective workflow optimization with Pareto analysis
- Real-time performance monitoring and adaptive execution
- Sophisticated resource management and constraint satisfaction
- Event-driven architecture with callback support
- Distributed execution with fault tolerance and recovery
- Machine learning-powered optimization recommendations
"""

from .workflow_models import (
    # Core enums
    WorkflowStatus,
    TaskStatus,
    TaskPriority,
    OptimizationObjective,
    SystemCapability,
    ResourceType,
    
    # Data models
    WorkflowDefinition,
    WorkflowExecution,
    TaskDefinition,
    TaskExecution,
    TaskResource,
    TaskConstraint,
    SystemStatus,
    OptimizationMetrics,
    
    # Factory functions
    create_task_definition,
    create_workflow_definition,
    create_system_status,
    
    # Utility functions
    calculate_task_priority_score,
    validate_workflow_consistency,
    
    # Constants
    DEFAULT_TASK_TIMEOUT,
    DEFAULT_WORKFLOW_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    PRIORITY_WEIGHTS,
    RESOURCE_LIMITS
)

from .task_scheduler import (
    TaskScheduler,
    SchedulingStrategy,
    create_task_scheduler
)

from .workflow_optimizer import (
    WorkflowOptimizer,
    OptimizationStrategy,
    BottleneckType,
    PerformanceBottleneck,
    OptimizationSuggestion,
    create_workflow_optimizer
)

from .execution_engine import (
    ExecutionEngine,
    ExecutionMode,
    ExecutionEvent,
    create_execution_engine
)

__all__ = [
    # Main components
    'TaskScheduler',
    'WorkflowOptimizer', 
    'ExecutionEngine',
    
    # Factory functions
    'create_task_scheduler',
    'create_workflow_optimizer',
    'create_execution_engine',
    
    # Core enums
    'WorkflowStatus',
    'TaskStatus',
    'TaskPriority',
    'OptimizationObjective',
    'SystemCapability',
    'ResourceType',
    'SchedulingStrategy',
    'OptimizationStrategy',
    'BottleneckType',
    'ExecutionMode',
    'ExecutionEvent',
    
    # Data models
    'WorkflowDefinition',
    'WorkflowExecution',
    'TaskDefinition',
    'TaskExecution',
    'TaskResource',
    'TaskConstraint',
    'SystemStatus',
    'OptimizationMetrics',
    'PerformanceBottleneck',
    'OptimizationSuggestion',
    
    # Model factory functions
    'create_task_definition',
    'create_workflow_definition',
    'create_system_status',
    
    # Utility functions
    'calculate_task_priority_score',
    'validate_workflow_consistency',
    
    # Constants
    'DEFAULT_TASK_TIMEOUT',
    'DEFAULT_WORKFLOW_TIMEOUT',
    'DEFAULT_MAX_RETRIES',
    'PRIORITY_WEIGHTS',
    'RESOURCE_LIMITS',
    
    # Convenience functions
    'create_simple_workflow',
    'create_data_pipeline',
    'create_batch_processor',
    'execute_simple_task',
    'monitor_workflow_execution'
]

# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Workflow Team'
__description__ = 'Enterprise-grade intelligent workflow orchestration with ML-powered optimization'


def create_simple_workflow(name: str, tasks: List[Tuple[str, Callable]], dependencies: Dict[str, List[str]] = None) -> WorkflowDefinition:
    """
    Create a simple workflow from a list of tasks
    
    Args:
        name: Workflow name
        tasks: List of (task_name, function) tuples
        dependencies: Optional task dependencies
        
    Returns:
        Configured workflow definition
        
    Example:
        >>> from core.workflow import create_simple_workflow
        >>> def task1(): return "result1"
        >>> def task2(): return "result2"
        >>> workflow = create_simple_workflow("Simple Pipeline", [
        ...     ("task1", task1),
        ...     ("task2", task2)
        ... ], {"task2": ["task1"]})
    """
    workflow = create_workflow_definition(name, f"Simple workflow with {len(tasks)} tasks")
    
    dependencies = dependencies or {}
    
    for task_name, function in tasks:
        task = create_task_definition(
            name=task_name,
            function=function,
            dependencies=dependencies.get(task_name, [])
        )
        workflow.add_task(task, dependencies.get(task_name, []))
    
    return workflow


def create_data_pipeline(name: str, 
                        data_source: Callable,
                        processors: List[Callable],
                        data_sink: Callable,
                        parallel_processing: bool = True) -> WorkflowDefinition:
    """
    Create a data processing pipeline workflow
    
    Args:
        name: Pipeline name
        data_source: Function to load data
        processors: List of data processing functions
        data_sink: Function to save processed data
        parallel_processing: Whether processors can run in parallel
        
    Returns:
        Configured data pipeline workflow
        
    Example:
        >>> from core.workflow import create_data_pipeline
        >>> def load_data(): return [1, 2, 3]
        >>> def process_data(data): return [x*2 for x in data]
        >>> def save_data(data): print(f"Saved: {data}")
        >>> pipeline = create_data_pipeline("ETL Pipeline", load_data, [process_data], save_data)
    """
    workflow = create_workflow_definition(name, "Data processing pipeline")
    
    # Create data source task
    source_task = create_task_definition(
        name="data_source",
        function=data_source,
        priority=TaskPriority.HIGH
    )
    workflow.add_task(source_task)
    
    # Create processor tasks
    processor_dependencies = ["data_source"]
    for i, processor in enumerate(processors):
        processor_task = create_task_definition(
            name=f"processor_{i+1}",
            function=processor,
            dependencies=processor_dependencies if not parallel_processing else ["data_source"],
            priority=TaskPriority.MEDIUM
        )
        workflow.add_task(processor_task, processor_task.depends_on)
        
        if not parallel_processing:
            processor_dependencies = [f"processor_{i+1}"]
    
    # Create data sink task
    sink_dependencies = [f"processor_{i+1}" for i in range(len(processors))]
    sink_task = create_task_definition(
        name="data_sink",
        function=data_sink,
        dependencies=sink_dependencies,
        priority=TaskPriority.HIGH
    )
    workflow.add_task(sink_task, sink_dependencies)
    
    return workflow


def create_batch_processor(name: str,
                          items: List[Any],
                          processor_function: Callable,
                          batch_size: int = 10,
                          max_parallel_batches: int = 3) -> WorkflowDefinition:
    """
    Create a batch processing workflow
    
    Args:
        name: Processor name
        items: Items to process
        processor_function: Function to process each batch
        batch_size: Size of each batch
        max_parallel_batches: Maximum parallel batches
        
    Returns:
        Configured batch processing workflow
        
    Example:
        >>> from core.workflow import create_batch_processor
        >>> def process_batch(batch): return [x*2 for x in batch]
        >>> items = list(range(100))
        >>> workflow = create_batch_processor("Batch Job", items, process_batch, batch_size=20)
    """
    workflow = create_workflow_definition(name, f"Batch processor for {len(items)} items")
    
    # Create batches
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    
    # Create batch processing tasks
    for i, batch in enumerate(batches):
        batch_task = create_task_definition(
            name=f"batch_{i+1}",
            function=processor_function,
            parameters={"batch": batch},
            priority=TaskPriority.MEDIUM
        )
        workflow.add_task(batch_task)
    
    # Set maximum parallel tasks
    workflow.max_parallel_tasks = min(max_parallel_batches, len(batches))
    
    return workflow


async def execute_simple_task(task_function: Callable, 
                            parameters: Dict[str, Any] = None,
                            timeout_seconds: int = 300) -> Any:
    """
    Execute a simple task with workflow infrastructure
    
    Args:
        task_function: Function to execute
        parameters: Function parameters
        timeout_seconds: Task timeout
        
    Returns:
        Task execution result
        
    Example:
        >>> import asyncio
        >>> from core.workflow import execute_simple_task
        >>> def my_task(x, y): return x + y
        >>> result = await execute_simple_task(my_task, {"x": 5, "y": 3})
        >>> print(result)  # Output: 8
    """
    # Create simple workflow with single task
    task = create_task_definition(
        name="simple_task",
        function=task_function,
        parameters=parameters or {},
        timeout_seconds=timeout_seconds
    )
    
    workflow = create_workflow_definition("Simple Task", "Single task execution")
    workflow.add_task(task)
    
    # Execute workflow
    engine = create_execution_engine(max_concurrent_workflows=1)
    try:
        execution = await engine.execute_workflow(workflow)
        
        if execution.successful_tasks > 0:
            task_execution = list(execution.task_executions.values())[0]
            return task_execution.result
        else:
            raise RuntimeError(f"Task execution failed: {execution.error_messages}")
    finally:
        engine.shutdown()


async def monitor_workflow_execution(workflow: WorkflowDefinition,
                                   monitoring_callback: Callable[[Dict[str, Any]], None] = None,
                                   monitoring_interval: float = 2.0) -> WorkflowExecution:
    """
    Execute workflow with real-time monitoring
    
    Args:
        workflow: Workflow to execute
        monitoring_callback: Optional callback for monitoring updates
        monitoring_interval: Monitoring update interval in seconds
        
    Returns:
        Workflow execution result
        
    Example:
        >>> import asyncio
        >>> from core.workflow import monitor_workflow_execution, create_simple_workflow
        >>> def monitor(status): print(f"Progress: {status['progress']['completion_percentage']:.1f}%")
        >>> workflow = create_simple_workflow("Test", [("task1", lambda: "done")])
        >>> result = await monitor_workflow_execution(workflow, monitor)
    """
    engine = create_execution_engine()
    
    try:
        # Start workflow execution
        execution_task = asyncio.create_task(engine.execute_workflow(workflow))
        
        # Monitor execution progress
        while not execution_task.done():
            await asyncio.sleep(monitoring_interval)
            
            # Get current status
            status = engine.get_execution_status()
            
            # Call monitoring callback if provided
            if monitoring_callback and status.get('active_workflows', 0) > 0:
                # Get detailed status for first active workflow
                active_workflows = list(engine.active_workflows.keys())
                if active_workflows:
                    detailed_status = engine.get_execution_status(active_workflows[0])
                    monitoring_callback(detailed_status)
        
        # Return execution result
        return await execution_task
        
    finally:
        engine.shutdown()


# Convenience factory for complete workflow suite
def create_workflow_suite(scheduler_config: Dict[str, Any] = None,
                         optimizer_config: Dict[str, Any] = None,
                         execution_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a complete workflow management suite
    
    Args:
        scheduler_config: Task scheduler configuration
        optimizer_config: Workflow optimizer configuration  
        execution_config: Execution engine configuration
        
    Returns:
        Dictionary containing configured workflow components
        
    Example:
        >>> from core.workflow import create_workflow_suite
        >>> suite = create_workflow_suite()
        >>> scheduler = suite['scheduler']
        >>> optimizer = suite['optimizer']
        >>> engine = suite['engine']
    """
    scheduler_config = scheduler_config or {}
    optimizer_config = optimizer_config or {}
    execution_config = execution_config or {}
    
    suite = {
        'scheduler': create_task_scheduler(
            max_concurrent_tasks=scheduler_config.get('max_concurrent_tasks', 10)
        ),
        'optimizer': create_workflow_optimizer(),
        'engine': create_execution_engine(
            max_concurrent_workflows=execution_config.get('max_concurrent_workflows', 5)
        )
    }
    
    return suite


# Module initialization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())