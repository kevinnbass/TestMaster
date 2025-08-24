"""
Analysis Coordinator

Central coordinator that manages and orchestrates different intelligence agents
and analysis systems. Provides intelligent task distribution, resource management,
and cross-system communication.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import uuid

from ...classical_analysis.analysis_orchestrator import AnalysisOrchestrator
from ..agents.documentation_agent import DocumentationIntelligenceAgent, DocumentationRequest


class AnalysisTaskType(Enum):
    """Types of analysis tasks."""
    CLASSICAL_ANALYSIS = "classical_analysis"
    DOCUMENTATION_GENERATION = "documentation_generation"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_PROFILING = "performance_profiling"
    CODE_QUALITY_REVIEW = "code_quality_review"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    CUSTOM_ANALYSIS = "custom_analysis"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AnalysisTask:
    """Represents an analysis task in the coordination system."""
    task_id: str
    task_type: AnalysisTaskType
    priority: TaskPriority
    project_path: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    requester: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # seconds
    
    # Runtime fields
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "priority": self.priority.value,
            "project_path": self.project_path,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "requester": self.requester,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "estimated_duration": self.estimated_duration,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "error": self.error
        }


@dataclass
class ResourceConstraints:
    """Resource constraints for task execution."""
    max_concurrent_tasks: int = 5
    max_memory_mb: int = 4096
    max_cpu_cores: int = 4
    max_disk_io_mb_per_sec: int = 100
    max_network_requests_per_sec: int = 10


@dataclass
class AgentCapability:
    """Describes what an agent is capable of handling."""
    agent_id: str
    supported_task_types: List[AnalysisTaskType]
    max_concurrent_tasks: int
    estimated_task_duration: Dict[AnalysisTaskType, int]  # seconds
    resource_requirements: Dict[str, int]
    current_load: float = 0.0  # 0.0 to 1.0


class AnalysisCoordinator:
    """
    Central coordinator for all intelligence analysis activities.
    
    Responsibilities:
    - Task queue management and prioritization
    - Agent coordination and load balancing
    - Resource allocation and constraint enforcement
    - Cross-system communication and data flow
    - Performance monitoring and optimization
    - Failure recovery and retry logic
    """
    
    def __init__(self, 
                 resource_constraints: Optional[ResourceConstraints] = None,
                 enable_auto_scaling: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.resource_constraints = resource_constraints or ResourceConstraints()
        self.enable_auto_scaling = enable_auto_scaling
        
        # Task management
        self.task_queue: List[AnalysisTask] = []
        self.active_tasks: Dict[str, AnalysisTask] = {}
        self.completed_tasks: Dict[str, AnalysisTask] = {}
        self.failed_tasks: Dict[str, AnalysisTask] = {}
        
        # Agent management
        self.registered_agents: Dict[str, AgentCapability] = {}
        self.agent_instances: Dict[str, Any] = {}
        
        # System state
        self.is_running = False
        self.coordinator_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_task_duration": 0.0,
            "current_throughput": 0.0,  # tasks per minute
            "resource_utilization": {
                "cpu": 0.0,
                "memory": 0.0,
                "disk_io": 0.0,
                "network": 0.0
            }
        }
        
        # Initialize core agents
        self._initialize_core_agents()
    
    def _initialize_core_agents(self):
        """Initialize core intelligence agents."""
        # Register classical analysis orchestrator
        self.register_agent(
            agent_id="classical_orchestrator",
            agent_instance=AnalysisOrchestrator(),
            capability=AgentCapability(
                agent_id="classical_orchestrator",
                supported_task_types=[
                    AnalysisTaskType.CLASSICAL_ANALYSIS,
                    AnalysisTaskType.SECURITY_AUDIT,
                    AnalysisTaskType.PERFORMANCE_PROFILING,
                    AnalysisTaskType.CODE_QUALITY_REVIEW,
                    AnalysisTaskType.DEPENDENCY_ANALYSIS
                ],
                max_concurrent_tasks=2,
                estimated_task_duration={
                    AnalysisTaskType.CLASSICAL_ANALYSIS: 300,  # 5 minutes
                    AnalysisTaskType.SECURITY_AUDIT: 180,
                    AnalysisTaskType.PERFORMANCE_PROFILING: 240,
                    AnalysisTaskType.CODE_QUALITY_REVIEW: 120,
                    AnalysisTaskType.DEPENDENCY_ANALYSIS: 60
                },
                resource_requirements={"memory_mb": 1024, "cpu_cores": 2}
            )
        )
        
        # Register documentation intelligence agent
        self.register_agent(
            agent_id="documentation_agent",
            agent_instance=DocumentationIntelligenceAgent(),
            capability=AgentCapability(
                agent_id="documentation_agent",
                supported_task_types=[AnalysisTaskType.DOCUMENTATION_GENERATION],
                max_concurrent_tasks=3,
                estimated_task_duration={
                    AnalysisTaskType.DOCUMENTATION_GENERATION: 400  # 6.7 minutes
                },
                resource_requirements={"memory_mb": 512, "cpu_cores": 1}
            )
        )
    
    async def start_coordinator(self):
        """Start the analysis coordinator."""
        if self.is_running:
            self.logger.warning("Coordinator is already running")
            return
        
        # Start registered agents
        for agent_id, agent_instance in self.agent_instances.items():
            if hasattr(agent_instance, 'start_agent'):
                await agent_instance.start_agent()
        
        self.is_running = True
        self.coordinator_task = asyncio.create_task(self._coordinator_main_loop())
        self.logger.info("Analysis Coordinator started")
    
    async def stop_coordinator(self):
        """Stop the analysis coordinator."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop coordinator loop
        if self.coordinator_task:
            self.coordinator_task.cancel()
            try:
                await self.coordinator_task
            except asyncio.CancelledError:
                pass
        
        # Stop registered agents
        for agent_id, agent_instance in self.agent_instances.items():
            if hasattr(agent_instance, 'stop_agent'):
                await agent_instance.stop_agent()
        
        self.logger.info("Analysis Coordinator stopped")
    
    def register_agent(self, 
                      agent_id: str, 
                      agent_instance: Any, 
                      capability: AgentCapability):
        """Register a new intelligence agent."""
        self.registered_agents[agent_id] = capability
        self.agent_instances[agent_id] = agent_instance
        self.logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an intelligence agent."""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
        if agent_id in self.agent_instances:
            del self.agent_instances[agent_id]
        self.logger.info(f"Unregistered agent: {agent_id}")
    
    async def submit_task(self, task: AnalysisTask) -> str:
        """Submit a new analysis task."""
        # Validate task
        if not Path(task.project_path).exists():
            raise ValueError(f"Project path does not exist: {task.project_path}")
        
        # Check if we have capable agents
        capable_agents = self._find_capable_agents(task.task_type)
        if not capable_agents:
            raise ValueError(f"No agents capable of handling task type: {task.task_type}")
        
        # Add to queue
        task.status = TaskStatus.QUEUED
        self.task_queue.append(task)
        self._prioritize_task_queue()
        
        self.logger.info(f"Task submitted: {task.task_id} ({task.task_type.value})")
        return task.task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        # Check if in queue
        for i, task in enumerate(self.task_queue):
            if task.task_id == task_id:
                task.status = TaskStatus.CANCELLED
                self.task_queue.pop(i)
                self.logger.info(f"Cancelled queued task: {task_id}")
                return True
        
        # Check if active (more complex cancellation needed)
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            # Implementation would need agent-specific cancellation
            self.logger.info(f"Cancelled active task: {task_id}")
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task."""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            return self.failed_tasks[task_id].to_dict()
        
        # Check queue
        for task in self.task_queue:
            if task.task_id == task_id:
                return task.to_dict()
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "is_running": self.is_running,
            "queue_length": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "registered_agents": len(self.registered_agents),
            "performance_metrics": self.performance_metrics.copy(),
            "resource_constraints": {
                "max_concurrent_tasks": self.resource_constraints.max_concurrent_tasks,
                "max_memory_mb": self.resource_constraints.max_memory_mb,
                "max_cpu_cores": self.resource_constraints.max_cpu_cores
            },
            "agent_status": {
                agent_id: {
                    "current_load": capability.current_load,
                    "max_concurrent": capability.max_concurrent_tasks,
                    "supported_types": [t.value for t in capability.supported_task_types]
                }
                for agent_id, capability in self.registered_agents.items()
            }
        }
    
    async def _coordinator_main_loop(self):
        """Main coordination loop."""
        while self.is_running:
            try:
                # Process task queue
                await self._process_task_queue()
                
                # Monitor active tasks
                await self._monitor_active_tasks()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check resource constraints
                self._enforce_resource_constraints()
                
                # Auto-scaling if enabled
                if self.enable_auto_scaling:
                    await self._handle_auto_scaling()
                
                # Cleanup old tasks
                self._cleanup_old_tasks()
                
                # Brief pause
                await asyncio.sleep(2.0)
                
            except Exception as e:
                self.logger.error(f"Error in coordinator main loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_task_queue(self):
        """Process tasks from the queue."""
        if not self.task_queue:
            return
        
        # Check if we can start more tasks
        if len(self.active_tasks) >= self.resource_constraints.max_concurrent_tasks:
            return
        
        # Get next task that's ready to run
        next_task = self._get_next_ready_task()
        if not next_task:
            return
        
        # Find best agent for the task
        best_agent_id = self._select_best_agent_for_task(next_task)
        if not best_agent_id:
            self.logger.warning(f"No available agent for task {next_task.task_id}")
            return
        
        # Assign and start task
        await self._assign_and_start_task(next_task, best_agent_id)
    
    def _get_next_ready_task(self) -> Optional[AnalysisTask]:
        """Get the next task that's ready to run."""
        for task in self.task_queue:
            if task.status != TaskStatus.QUEUED:
                continue
            
            # Check if dependencies are satisfied
            if self._are_dependencies_satisfied(task):
                return task
        
        return None
    
    def _are_dependencies_satisfied(self, task: AnalysisTask) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
        return True
    
    def _select_best_agent_for_task(self, task: AnalysisTask) -> Optional[str]:
        """Select the best agent for a task based on capability and load."""
        capable_agents = self._find_capable_agents(task.task_type)
        if not capable_agents:
            return None
        
        # Score agents based on load and estimated completion time
        best_agent_id = None
        best_score = float('inf')
        
        for agent_id in capable_agents:
            capability = self.registered_agents[agent_id]
            
            # Skip if agent is at capacity
            if capability.current_load >= 1.0:
                continue
            
            # Calculate score (lower is better)
            estimated_duration = capability.estimated_task_duration.get(task.task_type, 300)
            load_factor = capability.current_load
            score = estimated_duration * (1 + load_factor)
            
            if score < best_score:
                best_score = score
                best_agent_id = agent_id
        
        return best_agent_id
    
    def _find_capable_agents(self, task_type: AnalysisTaskType) -> List[str]:
        """Find agents capable of handling a specific task type."""
        return [
            agent_id for agent_id, capability in self.registered_agents.items()
            if task_type in capability.supported_task_types
        ]
    
    async def _assign_and_start_task(self, task: AnalysisTask, agent_id: str):
        """Assign a task to an agent and start execution."""
        # Remove from queue
        self.task_queue.remove(task)
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.assigned_agent = agent_id
        task.started_at = datetime.now()
        
        # Move to active tasks
        self.active_tasks[task.task_id] = task
        
        # Update agent load
        capability = self.registered_agents[agent_id]
        capability.current_load = min(1.0, capability.current_load + (1.0 / capability.max_concurrent_tasks))
        
        # Start task execution
        asyncio.create_task(self._execute_task(task, agent_id))
        
        self.logger.info(f"Started task {task.task_id} on agent {agent_id}")
    
    async def _execute_task(self, task: AnalysisTask, agent_id: str):
        """Execute a task on the specified agent."""
        try:
            agent_instance = self.agent_instances[agent_id]
            
            # Execute based on task type
            if task.task_type == AnalysisTaskType.CLASSICAL_ANALYSIS:
                result = await self._execute_classical_analysis(task, agent_instance)
            elif task.task_type == AnalysisTaskType.DOCUMENTATION_GENERATION:
                result = await self._execute_documentation_generation(task, agent_instance)
            elif task.task_type == AnalysisTaskType.SECURITY_AUDIT:
                result = await self._execute_security_audit(task, agent_instance)
            elif task.task_type == AnalysisTaskType.PERFORMANCE_PROFILING:
                result = await self._execute_performance_profiling(task, agent_instance)
            elif task.task_type == AnalysisTaskType.CODE_QUALITY_REVIEW:
                result = await self._execute_code_quality_review(task, agent_instance)
            elif task.task_type == AnalysisTaskType.DEPENDENCY_ANALYSIS:
                result = await self._execute_dependency_analysis(task, agent_instance)
            else:
                result = await self._execute_custom_analysis(task, agent_instance)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            task.progress = 1.0
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
            
            # Move to failed tasks
            self.failed_tasks[task.task_id] = task
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update agent load
            capability = self.registered_agents[agent_id]
            capability.current_load = max(0.0, capability.current_load - (1.0 / capability.max_concurrent_tasks))
    
    async def _execute_classical_analysis(self, task: AnalysisTask, agent: AnalysisOrchestrator) -> Any:
        """Execute classical analysis task."""
        return await agent.run_comprehensive_analysis(
            task.project_path,
            **task.parameters
        )
    
    async def _execute_documentation_generation(self, 
                                                task: AnalysisTask, 
                                                agent: DocumentationIntelligenceAgent) -> Any:
        """Execute documentation generation task."""
        # Convert task to documentation request
        doc_request = DocumentationRequest(
            request_id=task.task_id,
            project_path=task.project_path,
            document_types=task.parameters.get("document_types", ["readme"]),
            priority=task.priority.value,
            context_depth=task.parameters.get("context_depth", "standard"),
            style_preferences=task.parameters.get("style_preferences", {}),
            deadline=task.deadline,
            requester=task.requester
        )
        
        # Submit to documentation agent
        await agent.submit_documentation_request(doc_request)
        
        # Wait for completion (simplified)
        while True:
            status = await agent.get_request_status(task.task_id)
            if status["status"] in ["completed", "failed"]:
                return status.get("result")
            await asyncio.sleep(5.0)
    
    async def _execute_security_audit(self, task: AnalysisTask, agent: AnalysisOrchestrator) -> Any:
        """Execute security audit task."""
        return await agent.run_security_analysis(
            task.project_path,
            **task.parameters
        )
    
    async def _execute_performance_profiling(self, task: AnalysisTask, agent: AnalysisOrchestrator) -> Any:
        """Execute performance profiling task."""
        return await agent.run_performance_analysis(
            task.project_path,
            **task.parameters
        )
    
    async def _execute_code_quality_review(self, task: AnalysisTask, agent: AnalysisOrchestrator) -> Any:
        """Execute code quality review task."""
        return await agent.run_quality_analysis(
            task.project_path,
            **task.parameters
        )
    
    async def _execute_dependency_analysis(self, task: AnalysisTask, agent: AnalysisOrchestrator) -> Any:
        """Execute dependency analysis task."""
        return await agent.run_dependency_analysis(
            task.project_path,
            **task.parameters
        )
    
    async def _execute_custom_analysis(self, task: AnalysisTask, agent: Any) -> Any:
        """Execute custom analysis task."""
        # Custom analysis would need specific handling
        raise NotImplementedError("Custom analysis not yet implemented")
    
    def _prioritize_task_queue(self):
        """Sort task queue by priority and creation time."""
        self.task_queue.sort(key=lambda t: (
            t.priority.value,
            t.deadline or datetime.max,
            t.created_at
        ))
    
    async def _monitor_active_tasks(self):
        """Monitor active tasks for progress and timeouts."""
        current_time = datetime.now()
        
        for task_id, task in list(self.active_tasks.items()):
            # Check for deadline timeout
            if task.deadline and current_time > task.deadline:
                self.logger.warning(f"Task {task_id} exceeded deadline")
                # Could implement deadline handling here
            
            # Update progress if agent supports it
            agent_instance = self.agent_instances.get(task.assigned_agent)
            if hasattr(agent_instance, 'get_task_progress'):
                try:
                    progress = await agent_instance.get_task_progress(task_id)
                    task.progress = progress
                except Exception:
                    pass  # Progress tracking is optional
    
    def _update_performance_metrics(self):
        """Update system performance metrics."""
        total_completed = len(self.completed_tasks)
        total_failed = len(self.failed_tasks)
        
        self.performance_metrics.update({
            "total_tasks_processed": total_completed + total_failed,
            "successful_tasks": total_completed,
            "failed_tasks": total_failed
        })
        
        if total_completed > 0:
            # Calculate average duration
            avg_duration = sum(
                (task.completed_at - task.started_at).total_seconds()
                for task in self.completed_tasks.values()
                if task.started_at and task.completed_at
            ) / total_completed
            
            self.performance_metrics["average_task_duration"] = avg_duration
        
        # Calculate current throughput (tasks per minute)
        # This is simplified - would need proper time window tracking
        active_count = len(self.active_tasks)
        if active_count > 0:
            avg_duration = self.performance_metrics.get("average_task_duration", 300)
            throughput = (60.0 / avg_duration) * active_count
            self.performance_metrics["current_throughput"] = throughput
    
    def _enforce_resource_constraints(self):
        """Enforce resource constraints."""
        # This is a simplified implementation
        # Real implementation would monitor actual resource usage
        
        current_tasks = len(self.active_tasks)
        if current_tasks > self.resource_constraints.max_concurrent_tasks:
            self.logger.warning(f"Exceeding max concurrent tasks: {current_tasks}")
    
    async def _handle_auto_scaling(self):
        """Handle automatic scaling of resources."""
        # Simplified auto-scaling logic
        queue_length = len(self.task_queue)
        active_tasks = len(self.active_tasks)
        
        # If queue is backing up, consider scaling
        if queue_length > 10 and active_tasks < self.resource_constraints.max_concurrent_tasks:
            self.logger.info("Queue backing up, consider scaling resources")
            # Could implement dynamic agent spawning here
    
    def _cleanup_old_tasks(self):
        """Clean up old completed/failed tasks."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Keep recent tasks for analysis
        if len(self.completed_tasks) > 1000:
            old_tasks = [
                task_id for task_id, task in self.completed_tasks.items()
                if task.completed_at and task.completed_at < cutoff_time
            ]
            for task_id in old_tasks[:500]:  # Remove oldest 500
                del self.completed_tasks[task_id]
        
        if len(self.failed_tasks) > 200:
            old_tasks = [
                task_id for task_id, task in self.failed_tasks.items()
                if task.completed_at and task.completed_at < cutoff_time
            ]
            for task_id in old_tasks[:100]:  # Remove oldest 100
                del self.failed_tasks[task_id]


# Factory functions for common task types
def create_classical_analysis_task(project_path: str, 
                                   priority: TaskPriority = TaskPriority.NORMAL,
                                   **parameters) -> AnalysisTask:
    """Create a classical analysis task."""
    return AnalysisTask(
        task_id=str(uuid.uuid4()),
        task_type=AnalysisTaskType.CLASSICAL_ANALYSIS,
        priority=priority,
        project_path=project_path,
        parameters=parameters,
        estimated_duration=300
    )


def create_documentation_task(project_path: str,
                             document_types: List[str],
                             priority: TaskPriority = TaskPriority.NORMAL,
                             **parameters) -> AnalysisTask:
    """Create a documentation generation task."""
    params = {"document_types": document_types, **parameters}
    return AnalysisTask(
        task_id=str(uuid.uuid4()),
        task_type=AnalysisTaskType.DOCUMENTATION_GENERATION,
        priority=priority,
        project_path=project_path,
        parameters=params,
        estimated_duration=400
    )


def create_security_audit_task(project_path: str,
                              priority: TaskPriority = TaskPriority.HIGH,
                              **parameters) -> AnalysisTask:
    """Create a security audit task."""
    return AnalysisTask(
        task_id=str(uuid.uuid4()),
        task_type=AnalysisTaskType.SECURITY_AUDIT,
        priority=priority,
        project_path=project_path,
        parameters=parameters,
        estimated_duration=180
    )