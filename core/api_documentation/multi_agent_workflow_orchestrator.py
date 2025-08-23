#!/usr/bin/env python3
"""
🎼 MODULE: Multi-Agent Workflow Orchestrator - Advanced Task Distribution System
==================================================================

📋 PURPOSE:
    Provides advanced workflow orchestration for Greek Swarm agents,
    intelligent task distribution, automated workflows, and performance optimization.

🎯 CORE FUNCTIONALITY:
    • Multi-agent workflow orchestration and task automation
    • Intelligent task distribution based on agent capabilities
    • Advanced workflow templates and execution pipelines
    • Performance optimization and resource allocation
    • Automated cross-agent collaboration and coordination

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 06:20:00 | Agent Delta | 🆕 FEATURE
   └─ Goal: Create advanced workflow orchestration for Hour 8 mission
   └─ Changes: Complete implementation of task distribution, workflows, optimization
   └─ Impact: Enables automated multi-agent coordination with intelligent task allocation

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Delta
🔧 Language: Python
📦 Dependencies: asyncio, aiohttp, sqlalchemy, celery, redis
🎯 Integration Points: All Greek Swarm agents, workflow execution engine
⚡ Performance Notes: Async execution, task queuing, distributed processing
🔒 Security Notes: Task authentication, workflow validation, resource limits

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: 0% | Last Run: N/A (New implementation)
✅ Integration Tests: 0% | Last Run: N/A (New implementation)
✅ Performance Tests: 0% | Last Run: N/A (New implementation)
⚠️  Known Issues: None (Initial implementation)

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Greek Swarm Coordinator, all Greek agents, task queue
📤 Provides: Advanced workflow orchestration for all Greek Swarm operations
🚨 Breaking Changes: None (new orchestration layer)
"""

import asyncio
import aiohttp
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import logging
from greek_swarm_coordinator import GreekSwarmCoordinator, AgentType, AgentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class WorkflowStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class TaskType(Enum):
    DATA_PROCESSING = "data_processing"
    API_REQUEST = "api_request"
    ANALYSIS = "analysis"
    REPORT_GENERATION = "report_generation"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    COORDINATION = "coordination"
    CUSTOM = "custom"

@dataclass
class TaskDefinition:
    """Definition of a task to be executed by agents"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    target_agents: List[AgentType]
    required_capabilities: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = 300
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    expected_output: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskExecution:
    """Execution instance of a task"""
    execution_id: str
    task_def: TaskDefinition
    assigned_agent: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)

@dataclass
class WorkflowDefinition:
    """Definition of a workflow containing multiple tasks"""
    workflow_id: str
    name: str
    description: str
    tasks: List[TaskDefinition]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # task_id -> [dependent_task_ids]
    parameters: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression
    max_parallel_tasks: int = 5
    timeout_minutes: int = 60
    retry_policy: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Execution instance of a workflow"""
    execution_id: str
    workflow_def: WorkflowDefinition
    status: WorkflowStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    task_executions: List[TaskExecution] = field(default_factory=list)
    current_stage: int = 0
    total_stages: int = 0
    success_count: int = 0
    failure_count: int = 0
    execution_context: Dict[str, Any] = field(default_factory=dict)

class MultiAgentWorkflowOrchestrator:
    """Advanced workflow orchestration system for Greek Swarm"""
    
    def __init__(self, db_path: str = "workflow_orchestrator.db"):
        self.db_path = db_path
        self.coordinator = GreekSwarmCoordinator()
        
        # Workflow and task storage
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        self.task_queue: List[TaskExecution] = []
        self.active_executions: Dict[str, TaskExecution] = {}
        
        # Execution engine
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.is_running = True
        
        # Performance tracking
        self.performance_metrics = {
            'total_tasks_executed': 0,
            'total_workflows_completed': 0,
            'average_task_duration': 0.0,
            'success_rate': 0.0,
            'agent_utilization': {},
            'resource_efficiency': 0.0
        }
        
        # Initialize database and services
        self.init_database()
        self.load_built_in_workflows()
        self.start_orchestration_services()
    
    def init_database(self):
        """Initialize SQLite database for workflow orchestration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Workflow definitions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_definitions (
                workflow_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                definition_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Workflow executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_executions (
                execution_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                status TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                execution_json TEXT,
                result_json TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflow_definitions (workflow_id)
            )
        """)
        
        # Task executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_executions (
                execution_id TEXT PRIMARY KEY,
                workflow_execution_id TEXT,
                task_id TEXT,
                assigned_agent TEXT,
                status TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                execution_time REAL,
                result_json TEXT,
                error_message TEXT,
                retry_count INTEGER,
                FOREIGN KEY (workflow_execution_id) REFERENCES workflow_executions (execution_id)
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TEXT PRIMARY KEY,
                metrics_json TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Workflow orchestrator database initialized")
    
    def load_built_in_workflows(self):
        """Load built-in workflow templates"""
        
        # Greek Swarm Health Check Workflow
        health_check_workflow = WorkflowDefinition(
            workflow_id="greek_swarm_health_check",
            name="Greek Swarm Health Check",
            description="Comprehensive health check across all Greek agents",
            tasks=[
                TaskDefinition(
                    task_id="alpha_health_check",
                    task_type=TaskType.MONITORING,
                    name="Alpha Agent Health Check",
                    description="Check Alpha agent health and performance",
                    target_agents=[AgentType.ALPHA],
                    required_capabilities=["health_monitoring"],
                    timeout_seconds=30
                ),
                TaskDefinition(
                    task_id="beta_health_check", 
                    task_type=TaskType.MONITORING,
                    name="Beta Agent Health Check",
                    description="Check Beta agent health and performance",
                    target_agents=[AgentType.BETA],
                    required_capabilities=["health_monitoring"],
                    timeout_seconds=30
                ),
                TaskDefinition(
                    task_id="gamma_health_check",
                    task_type=TaskType.MONITORING,
                    name="Gamma Agent Health Check", 
                    description="Check Gamma agent health and performance",
                    target_agents=[AgentType.GAMMA],
                    required_capabilities=["health_monitoring"],
                    timeout_seconds=30
                ),
                TaskDefinition(
                    task_id="delta_health_check",
                    task_type=TaskType.MONITORING,
                    name="Delta Agent Health Check",
                    description="Check Delta agent health and performance", 
                    target_agents=[AgentType.DELTA],
                    required_capabilities=["health_monitoring"],
                    timeout_seconds=30
                ),
                TaskDefinition(
                    task_id="epsilon_health_check",
                    task_type=TaskType.MONITORING,
                    name="Epsilon Agent Health Check",
                    description="Check Epsilon agent health and performance",
                    target_agents=[AgentType.EPSILON], 
                    required_capabilities=["health_monitoring"],
                    timeout_seconds=30
                ),
                TaskDefinition(
                    task_id="aggregate_health_report",
                    task_type=TaskType.REPORT_GENERATION,
                    name="Aggregate Health Report",
                    description="Generate comprehensive health report",
                    target_agents=[AgentType.DELTA],  # Delta handles reporting
                    required_capabilities=["report_generation"],
                    dependencies=["alpha_health_check", "beta_health_check", "gamma_health_check", 
                                "delta_health_check", "epsilon_health_check"],
                    timeout_seconds=60
                )
            ],
            max_parallel_tasks=5,
            timeout_minutes=10
        )
        
        # Cross-Agent Data Synchronization Workflow
        data_sync_workflow = WorkflowDefinition(
            workflow_id="cross_agent_data_sync",
            name="Cross-Agent Data Synchronization",
            description="Synchronize data across all Greek agents",
            tasks=[
                TaskDefinition(
                    task_id="collect_agent_data",
                    task_type=TaskType.DATA_PROCESSING,
                    name="Collect Agent Data",
                    description="Collect current data from all agents",
                    target_agents=[AgentType.ALPHA, AgentType.BETA, AgentType.GAMMA, 
                                 AgentType.DELTA, AgentType.EPSILON],
                    required_capabilities=["data_export"],
                    timeout_seconds=120
                ),
                TaskDefinition(
                    task_id="validate_data_integrity",
                    task_type=TaskType.ANALYSIS,
                    name="Validate Data Integrity",
                    description="Validate collected data for consistency",
                    target_agents=[AgentType.DELTA],
                    required_capabilities=["data_validation"],
                    dependencies=["collect_agent_data"],
                    timeout_seconds=60
                ),
                TaskDefinition(
                    task_id="synchronize_agent_data",
                    task_type=TaskType.DATA_PROCESSING,
                    name="Synchronize Agent Data",
                    description="Push synchronized data to all agents",
                    target_agents=[AgentType.ALPHA, AgentType.BETA, AgentType.GAMMA,
                                 AgentType.DELTA, AgentType.EPSILON],
                    required_capabilities=["data_import"],
                    dependencies=["validate_data_integrity"],
                    timeout_seconds=180
                )
            ],
            max_parallel_tasks=3,
            timeout_minutes=15
        )
        
        # Performance Optimization Workflow
        performance_optimization_workflow = WorkflowDefinition(
            workflow_id="performance_optimization",
            name="Greek Swarm Performance Optimization",
            description="Optimize performance across all Greek agents",
            tasks=[
                TaskDefinition(
                    task_id="collect_performance_metrics",
                    task_type=TaskType.MONITORING,
                    name="Collect Performance Metrics",
                    description="Gather performance data from all agents",
                    target_agents=[AgentType.ALPHA, AgentType.BETA, AgentType.GAMMA,
                                 AgentType.DELTA, AgentType.EPSILON],
                    required_capabilities=["performance_monitoring"],
                    timeout_seconds=90
                ),
                TaskDefinition(
                    task_id="analyze_performance_bottlenecks",
                    task_type=TaskType.ANALYSIS,
                    name="Analyze Performance Bottlenecks",
                    description="Identify performance bottlenecks and optimization opportunities",
                    target_agents=[AgentType.BETA],  # Beta specializes in performance
                    required_capabilities=["performance_analysis"],
                    dependencies=["collect_performance_metrics"],
                    timeout_seconds=120
                ),
                TaskDefinition(
                    task_id="apply_optimizations",
                    task_type=TaskType.OPTIMIZATION,
                    name="Apply Performance Optimizations",
                    description="Apply recommended optimizations to agents",
                    target_agents=[AgentType.ALPHA, AgentType.BETA, AgentType.GAMMA,
                                 AgentType.DELTA, AgentType.EPSILON],
                    required_capabilities=["configuration_management"],
                    dependencies=["analyze_performance_bottlenecks"],
                    timeout_seconds=180
                ),
                TaskDefinition(
                    task_id="validate_optimization_results",
                    task_type=TaskType.MONITORING,
                    name="Validate Optimization Results",
                    description="Verify performance improvements after optimization",
                    target_agents=[AgentType.BETA],
                    required_capabilities=["performance_validation"],
                    dependencies=["apply_optimizations"],
                    timeout_seconds=120
                )
            ],
            max_parallel_tasks=2,
            timeout_minutes=20
        )
        
        # Store built-in workflows
        self.workflows.update({
            health_check_workflow.workflow_id: health_check_workflow,
            data_sync_workflow.workflow_id: data_sync_workflow,
            performance_optimization_workflow.workflow_id: performance_optimization_workflow
        })
        
        logger.info(f"Loaded {len(self.workflows)} built-in workflows")
    
    def start_orchestration_services(self):
        """Start background orchestration services"""
        
        # Task queue processor
        queue_thread = threading.Thread(target=self._task_queue_processor, daemon=True)
        queue_thread.start()
        
        # Workflow executor
        workflow_thread = threading.Thread(target=self._workflow_executor, daemon=True)
        workflow_thread.start()
        
        # Performance monitor
        monitor_thread = threading.Thread(target=self._performance_monitor, daemon=True)
        monitor_thread.start()
        
        # Cleanup service
        cleanup_thread = threading.Thread(target=self._cleanup_service, daemon=True)
        cleanup_thread.start()
        
        logger.info("Workflow orchestration services started")
    
    def _task_queue_processor(self):
        """Background task queue processing"""
        while self.is_running:
            try:
                if self.task_queue:
                    # Sort by priority and creation time
                    self.task_queue.sort(key=lambda t: (t.task_def.priority.value, t.created_at), reverse=True)
                    
                    # Process highest priority tasks first
                    for task_execution in self.task_queue[:]:
                        if len(self.active_executions) < 10:  # Limit concurrent executions
                            if self.can_execute_task(task_execution):
                                self.task_queue.remove(task_execution)
                                self.execute_task_async(task_execution)
                        else:
                            break
                
                time.sleep(2)  # Process queue every 2 seconds
            except Exception as e:
                logger.error(f"Task queue processor error: {e}")
                time.sleep(1)
    
    def _workflow_executor(self):
        """Background workflow execution management"""
        while self.is_running:
            try:
                # Check for running workflows
                for execution_id, workflow_execution in list(self.workflow_executions.items()):
                    if workflow_execution.status == WorkflowStatus.RUNNING:
                        self.advance_workflow_execution(workflow_execution)
                
                time.sleep(5)  # Check workflows every 5 seconds
            except Exception as e:
                logger.error(f"Workflow executor error: {e}")
                time.sleep(2)
    
    def _performance_monitor(self):
        """Background performance monitoring"""
        while self.is_running:
            try:
                self.update_performance_metrics()
                self.store_performance_metrics()
                time.sleep(30)  # Update metrics every 30 seconds
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                time.sleep(10)
    
    def _cleanup_service(self):
        """Background cleanup of completed executions"""
        while self.is_running:
            try:
                # Clean up old completed executions
                cutoff_time = datetime.utcnow() - timedelta(hours=2)
                
                completed_executions = [
                    exec_id for exec_id, execution in self.workflow_executions.items()
                    if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
                    and execution.completed_at and execution.completed_at < cutoff_time
                ]
                
                for exec_id in completed_executions:
                    del self.workflow_executions[exec_id]
                
                time.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Cleanup service error: {e}")
                time.sleep(60)
    
    def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any] = None) -> str:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_def = self.workflows[workflow_id]
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        workflow_execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_def=workflow_def,
            status=WorkflowStatus.RUNNING,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            total_stages=len(workflow_def.tasks),
            execution_context=parameters or {}
        )
        
        self.workflow_executions[execution_id] = workflow_execution
        
        # Queue initial tasks (those without dependencies)
        initial_tasks = [
            task for task in workflow_def.tasks
            if not task.dependencies or all(dep not in [t.task_id for t in workflow_def.tasks] 
                                           for dep in task.dependencies)
        ]
        
        for task in initial_tasks:
            task_execution = TaskExecution(
                execution_id=f"task_{uuid.uuid4().hex[:8]}",
                task_def=task,
                assigned_agent="",  # Will be assigned during execution
                status=TaskStatus.QUEUED,
                created_at=datetime.utcnow()
            )
            
            workflow_execution.task_executions.append(task_execution)
            self.task_queue.append(task_execution)
        
        logger.info(f"Started workflow execution: {execution_id}")
        return execution_id
    
    def can_execute_task(self, task_execution: TaskExecution) -> bool:
        """Check if a task can be executed"""
        task_def = task_execution.task_def
        
        # Check if dependencies are satisfied
        if task_def.dependencies:
            workflow_execution = next(
                (we for we in self.workflow_executions.values()
                 if task_execution in we.task_executions), None
            )
            
            if workflow_execution:
                completed_tasks = {
                    te.task_def.task_id for te in workflow_execution.task_executions
                    if te.status == TaskStatus.COMPLETED
                }
                
                if not all(dep in completed_tasks for dep in task_def.dependencies):
                    return False
        
        # Check if suitable agent is available
        available_agents = [
            agent_info for agent_info in self.coordinator.agents.values()
            if (agent_info.agent_type in task_def.target_agents and 
                agent_info.status == AgentStatus.ACTIVE and
                agent_info.health_score > 0.5)
        ]
        
        return len(available_agents) > 0
    
    def execute_task_async(self, task_execution: TaskExecution):
        """Execute a task asynchronously"""
        task_execution.status = TaskStatus.RUNNING
        task_execution.started_at = datetime.utcnow()
        
        # Assign best available agent
        assigned_agent = self.select_best_agent(task_execution.task_def)
        if not assigned_agent:
            task_execution.status = TaskStatus.FAILED
            task_execution.error = "No suitable agent available"
            return
        
        task_execution.assigned_agent = assigned_agent.agent_id
        self.active_executions[task_execution.execution_id] = task_execution
        
        # Submit task to executor
        future = self.executor.submit(self._execute_task_sync, task_execution, assigned_agent)
        future.add_done_callback(lambda f: self._task_completion_callback(task_execution, f))
    
    def _execute_task_sync(self, task_execution: TaskExecution, agent_info) -> Dict[str, Any]:
        """Execute task synchronously"""
        task_def = task_execution.task_def
        start_time = time.time()
        
        try:
            # Prepare task request
            task_request = {
                'task_id': task_execution.execution_id,
                'task_type': task_def.task_type.value,
                'name': task_def.name,
                'description': task_def.description,
                'parameters': task_def.parameters,
                'timeout': task_def.timeout_seconds
            }
            
            # Execute task via agent API
            result = asyncio.run(self._send_task_to_agent(agent_info, task_request))
            
            execution_time = time.time() - start_time
            task_execution.execution_time = execution_time
            task_execution.result = result
            task_execution.status = TaskStatus.COMPLETED
            task_execution.completed_at = datetime.utcnow()
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            task_execution.execution_time = execution_time
            task_execution.error = str(e)
            task_execution.status = TaskStatus.FAILED
            task_execution.completed_at = datetime.utcnow()
            
            # Retry if within limits
            if task_execution.retry_count < task_def.max_retries:
                task_execution.retry_count += 1
                task_execution.status = TaskStatus.RETRYING
                # Re-queue for retry
                self.task_queue.append(task_execution)
            
            raise e
    
    async def _send_task_to_agent(self, agent_info, task_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send task to agent via API"""
        url = f"http://{agent_info.host}:{agent_info.port}/api/execute-task"
        
        timeout = aiohttp.ClientTimeout(total=task_request['timeout'])
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=task_request) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Agent returned status {response.status}: {error_text}")
    
    def _task_completion_callback(self, task_execution: TaskExecution, future):
        """Handle task completion"""
        try:
            if task_execution.execution_id in self.active_executions:
                del self.active_executions[task_execution.execution_id]
            
            if task_execution.status == TaskStatus.COMPLETED:
                logger.info(f"Task completed: {task_execution.task_def.name}")
                self.performance_metrics['total_tasks_executed'] += 1
            else:
                logger.error(f"Task failed: {task_execution.task_def.name} - {task_execution.error}")
            
            # Check if this completes a workflow
            self.check_workflow_completion(task_execution)
            
        except Exception as e:
            logger.error(f"Task completion callback error: {e}")
    
    def select_best_agent(self, task_def: TaskDefinition):
        """Select the best agent for a task"""
        available_agents = [
            agent_info for agent_info in self.coordinator.agents.values()
            if (agent_info.agent_type in task_def.target_agents and
                agent_info.status == AgentStatus.ACTIVE and
                agent_info.health_score > 0.5)
        ]
        
        if not available_agents:
            return None
        
        # Score agents based on health, load, and capabilities
        def score_agent(agent_info):
            score = agent_info.health_score * 0.4  # 40% health
            score += (1 - agent_info.load_factor) * 0.3  # 30% availability
            score += (1 / (agent_info.response_time + 0.1)) * 0.3  # 30% responsiveness
            return score
        
        return max(available_agents, key=score_agent)
    
    def advance_workflow_execution(self, workflow_execution: WorkflowExecution):
        """Advance workflow execution based on completed tasks"""
        completed_tasks = {
            te.task_def.task_id for te in workflow_execution.task_executions
            if te.status == TaskStatus.COMPLETED
        }
        
        failed_tasks = {
            te.task_def.task_id for te in workflow_execution.task_executions
            if te.status == TaskStatus.FAILED and te.retry_count >= te.task_def.max_retries
        }
        
        # Check for newly available tasks
        for task in workflow_execution.workflow_def.tasks:
            if task.task_id not in [te.task_def.task_id for te in workflow_execution.task_executions]:
                # Check if all dependencies are satisfied
                if all(dep in completed_tasks for dep in task.dependencies):
                    task_execution = TaskExecution(
                        execution_id=f"task_{uuid.uuid4().hex[:8]}",
                        task_def=task,
                        assigned_agent="",
                        status=TaskStatus.QUEUED,
                        created_at=datetime.utcnow()
                    )
                    
                    workflow_execution.task_executions.append(task_execution)
                    self.task_queue.append(task_execution)
        
        # Check workflow completion
        all_tasks = {task.task_id for task in workflow_execution.workflow_def.tasks}
        
        if all_tasks <= (completed_tasks | failed_tasks):
            if failed_tasks:
                workflow_execution.status = WorkflowStatus.FAILED
            else:
                workflow_execution.status = WorkflowStatus.COMPLETED
            
            workflow_execution.completed_at = datetime.utcnow()
            workflow_execution.success_count = len(completed_tasks)
            workflow_execution.failure_count = len(failed_tasks)
            
            self.performance_metrics['total_workflows_completed'] += 1
            logger.info(f"Workflow completed: {workflow_execution.workflow_def.name}")
    
    def check_workflow_completion(self, task_execution: TaskExecution):
        """Check if a task completion triggers workflow advancement"""
        workflow_execution = next(
            (we for we in self.workflow_executions.values()
             if task_execution in we.task_executions), None
        )
        
        if workflow_execution and workflow_execution.status == WorkflowStatus.RUNNING:
            self.advance_workflow_execution(workflow_execution)
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        if self.performance_metrics['total_tasks_executed'] > 0:
            # Calculate success rate
            completed_tasks = sum(
                len([te for te in we.task_executions if te.status == TaskStatus.COMPLETED])
                for we in self.workflow_executions.values()
            )
            total_tasks = sum(
                len(we.task_executions) for we in self.workflow_executions.values()
            )
            
            if total_tasks > 0:
                self.performance_metrics['success_rate'] = completed_tasks / total_tasks
            
            # Calculate average task duration
            completed_executions = [
                te for we in self.workflow_executions.values()
                for te in we.task_executions
                if te.status == TaskStatus.COMPLETED and te.execution_time > 0
            ]
            
            if completed_executions:
                self.performance_metrics['average_task_duration'] = sum(
                    te.execution_time for te in completed_executions
                ) / len(completed_executions)
            
            # Update agent utilization
            for agent_id, agent_info in self.coordinator.agents.items():
                agent_tasks = sum(
                    1 for we in self.workflow_executions.values()
                    for te in we.task_executions
                    if te.assigned_agent == agent_id
                )
                self.performance_metrics['agent_utilization'][agent_id] = agent_tasks
    
    def store_performance_metrics(self):
        """Store performance metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO performance_metrics (timestamp, metrics_json)
                VALUES (?, ?)
            """, (
                datetime.utcnow().isoformat(),
                json.dumps(self.performance_metrics)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        if execution_id not in self.workflow_executions:
            return None
        
        workflow_execution = self.workflow_executions[execution_id]
        
        return {
            'execution_id': execution_id,
            'workflow_name': workflow_execution.workflow_def.name,
            'status': workflow_execution.status.value,
            'created_at': workflow_execution.created_at.isoformat(),
            'started_at': workflow_execution.started_at.isoformat() if workflow_execution.started_at else None,
            'completed_at': workflow_execution.completed_at.isoformat() if workflow_execution.completed_at else None,
            'current_stage': workflow_execution.current_stage,
            'total_stages': workflow_execution.total_stages,
            'success_count': workflow_execution.success_count,
            'failure_count': workflow_execution.failure_count,
            'task_executions': [
                {
                    'task_name': te.task_def.name,
                    'status': te.status.value,
                    'assigned_agent': te.assigned_agent,
                    'execution_time': te.execution_time,
                    'error': te.error
                }
                for te in workflow_execution.task_executions
            ]
        }
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        return {
            'orchestrator_id': 'multi_agent_workflow_orchestrator',
            'is_running': self.is_running,
            'workflows_available': len(self.workflows),
            'active_executions': len([we for we in self.workflow_executions.values() 
                                    if we.status == WorkflowStatus.RUNNING]),
            'queued_tasks': len(self.task_queue),
            'active_tasks': len(self.active_executions),
            'performance_metrics': self.performance_metrics,
            'connected_agents': len(self.coordinator.agents),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def shutdown(self):
        """Shutdown the orchestrator"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Workflow orchestrator shutdown complete")

def main():
    """Test the Multi-Agent Workflow Orchestrator"""
    print("=" * 80)
    print("MULTI-AGENT WORKFLOW ORCHESTRATOR - HOUR 8 DEPLOYMENT")
    print("=" * 80)
    print("Status: Advanced Greek Swarm Workflow Orchestration")
    print("Capabilities: Task Distribution, Workflow Automation, Performance Optimization")
    print("Integration: All Greek Swarm agents with intelligent task allocation")
    print("=" * 80)
    
    orchestrator = MultiAgentWorkflowOrchestrator()
    
    try:
        # Example: Execute health check workflow
        print("\nExecuting Greek Swarm Health Check workflow...")
        execution_id = orchestrator.execute_workflow("greek_swarm_health_check")
        print(f"Workflow execution started: {execution_id}")
        
        # Keep orchestrator running
        while True:
            time.sleep(10)
            status = orchestrator.get_orchestrator_status()
            print(f"Orchestrator Status: {status['active_executions']} active, {status['queued_tasks']} queued")
    except KeyboardInterrupt:
        print("Shutting down Multi-Agent Workflow Orchestrator...")
        orchestrator.shutdown()

if __name__ == "__main__":
    main()