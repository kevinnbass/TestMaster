"""
Enhanced Agent Orchestration System
===================================

Advanced orchestration system that enhances the existing agent_graph.py with
swarm-based patterns, multi-architecture support, and intelligent routing
from the unified orchestrator.

Extends core/orchestration/agent_graph.py without breaking existing functionality.

Author: TestMaster Core Orchestration Enhancement
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Callable, Literal
import random

# Import existing agent graph
try:
    from .agent_graph import TestAgent, AgentStatus, OrchestrationMode
except ImportError:
    # Fallback definitions
    from enum import Enum
    from dataclasses import dataclass, field
    
    class AgentStatus(Enum):
        IDLE = "idle"
        RUNNING = "running"
        PAUSED = "paused"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    class OrchestrationMode(Enum):
        SEQUENTIAL = "sequential"
        PARALLEL = "parallel"
        WORKFLOW = "workflow"
        ADAPTIVE = "adaptive"

logger = logging.getLogger(__name__)

# Enhanced Enums
class SwarmArchitecture(Enum):
    """Swarm architecture patterns for advanced orchestration."""
    SEQUENTIAL_WORKFLOW = "sequential_workflow"
    CONCURRENT_WORKFLOW = "concurrent_workflow"
    HIERARCHICAL_SWARM = "hierarchical_swarm"
    AGENT_REARRANGE = "agent_rearrange"
    MIXTURE_OF_AGENTS = "mixture_of_agents"
    SWARM_ROUTER = "swarm_router"
    DYNAMIC_PIPELINE = "dynamic_pipeline"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 10
    HIGH = 7
    MEDIUM = 5
    LOW = 3
    BACKGROUND = 1

class LoadBalancingStrategy(Enum):
    """Load balancing strategies for agent assignment."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    INTELLIGENT_ROUTING = "intelligent_routing"

@dataclass
class EnhancedTask:
    """Enhanced task with swarm capabilities."""
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    task_type: str = "test_execution"
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: Optional[int] = None
    max_retries: int = 3
    retry_count: int = 0
    status: str = "pending"
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority.value if isinstance(self.priority, TaskPriority) else self.priority,
            "required_capabilities": self.required_capabilities,
            "estimated_duration": self.estimated_duration,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "status": self.status,
            "assigned_agent": self.assigned_agent,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error_message": self.error_message,
            "metadata": self.metadata
        }

@dataclass
class SwarmAgent:
    """Enhanced agent with swarm capabilities."""
    agent_id: str
    name: str
    capabilities: List[str]
    status: AgentStatus = AgentStatus.IDLE
    current_tasks: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    load_factor: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    specializations: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    error_rate: float = 0.0
    average_execution_time: float = 0.0
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (self.status == AgentStatus.IDLE and 
                len(self.current_tasks) < self.max_concurrent_tasks)
    
    def can_handle_task(self, task: EnhancedTask) -> bool:
        """Check if agent can handle a specific task."""
        if not self.is_available():
            return False
        
        # Check capability requirements
        if task.required_capabilities:
            return all(cap in self.capabilities for cap in task.required_capabilities)
        
        return True
    
    def calculate_suitability_score(self, task: EnhancedTask) -> float:
        """Calculate how suitable this agent is for a task."""
        if not self.can_handle_task(task):
            return 0.0
        
        score = self.quality_score
        
        # Bonus for specializations
        if task.task_type in self.specializations:
            score *= 1.5
        
        # Penalty for high load
        score *= (1.0 - self.load_factor * 0.5)
        
        # Penalty for high error rate
        score *= (1.0 - self.error_rate * 0.3)
        
        return min(score, 10.0)  # Cap at 10

class EnhancedAgentOrchestrator:
    """
    Enhanced agent orchestration system with swarm capabilities.
    
    Extends existing agent graph functionality with:
    - Swarm-based distributed execution
    - Multiple architecture patterns
    - Intelligent load balancing
    - Advanced task routing
    - Performance optimization
    """
    
    def __init__(self,
                 max_agents: int = 10,
                 default_architecture: SwarmArchitecture = SwarmArchitecture.CONCURRENT_WORKFLOW,
                 load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT_ROUTING):
        """
        Initialize enhanced orchestrator.
        
        Args:
            max_agents: Maximum number of agents
            default_architecture: Default swarm architecture
            load_balancing: Load balancing strategy
        """
        self.max_agents = max_agents
        self.default_architecture = default_architecture
        self.load_balancing = load_balancing
        
        # Agent management
        self.agents: Dict[str, SwarmAgent] = {}
        self.agent_pool = ThreadPoolExecutor(max_workers=max_agents)
        
        # Task management
        self.task_queue: deque = deque()
        self.priority_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.active_tasks: Dict[str, EnhancedTask] = {}
        self.completed_tasks: Dict[str, EnhancedTask] = {}
        
        # Orchestration state
        self.orchestration_active = True
        self.orchestration_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            'total_tasks_processed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_completion_time': 0.0,
            'throughput_per_minute': 0.0,
            'agent_utilization': 0.0
        }
        
        # Architecture patterns
        self.architecture_strategies = {
            SwarmArchitecture.SEQUENTIAL_WORKFLOW: self._execute_sequential_workflow,
            SwarmArchitecture.CONCURRENT_WORKFLOW: self._execute_concurrent_workflow,
            SwarmArchitecture.HIERARCHICAL_SWARM: self._execute_hierarchical_swarm,
            SwarmArchitecture.AGENT_REARRANGE: self._execute_agent_rearrange,
            SwarmArchitecture.MIXTURE_OF_AGENTS: self._execute_mixture_of_agents,
            SwarmArchitecture.SWARM_ROUTER: self._execute_swarm_router,
            SwarmArchitecture.DYNAMIC_PIPELINE: self._execute_dynamic_pipeline
        }
        
        # Load balancing strategies
        self.load_balancers = {
            LoadBalancingStrategy.ROUND_ROBIN: self._assign_round_robin,
            LoadBalancingStrategy.LEAST_LOADED: self._assign_least_loaded,
            LoadBalancingStrategy.CAPABILITY_BASED: self._assign_capability_based,
            LoadBalancingStrategy.PERFORMANCE_WEIGHTED: self._assign_performance_weighted,
            LoadBalancingStrategy.INTELLIGENT_ROUTING: self._assign_intelligent_routing
        }
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Start orchestration
        self._start_orchestration()
        
        logger.info(f"Enhanced Agent Orchestrator initialized with {max_agents} max agents")
    
    def register_agent(self, agent: SwarmAgent) -> bool:
        """Register a new agent with the orchestrator."""
        with self.lock:
            if len(self.agents) >= self.max_agents:
                logger.warning(f"Cannot register agent {agent.agent_id}: max agents reached")
                return False
            
            self.agents[agent.agent_id] = agent
            logger.info(f"Registered agent {agent.agent_id} with capabilities: {agent.capabilities}")
            return True
    
    def submit_task(self, task: EnhancedTask) -> str:
        """Submit a task for execution."""
        with self.lock:
            # Add to appropriate priority queue
            self.priority_queues[task.priority].append(task)
            
            logger.debug(f"Submitted task {task.task_id} with priority {task.priority.name}")
            return task.task_id
    
    def submit_workflow(self, 
                       tasks: List[EnhancedTask],
                       architecture: SwarmArchitecture = None) -> str:
        """Submit a workflow (collection of related tasks)."""
        architecture = architecture or self.default_architecture
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        # Mark tasks as part of workflow
        for task in tasks:
            task.metadata['workflow_id'] = workflow_id
            task.metadata['architecture'] = architecture.value
        
        # Execute based on architecture
        strategy = self.architecture_strategies.get(architecture, self._execute_concurrent_workflow)
        
        try:
            result = strategy(tasks)
            logger.info(f"Workflow {workflow_id} executed with {architecture.name}")
            return workflow_id
        except Exception as e:
            logger.error(f"Workflow {workflow_id} execution failed: {e}")
            return ""
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].to_dict()
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].to_dict()
            
            # Check queues
            for priority_queue in self.priority_queues.values():
                for task in priority_queue:
                    if task.task_id == task_id:
                        return task.to_dict()
            
            return None
    
    def get_agent_status(self, agent_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of specific agent or all agents."""
        with self.lock:
            if agent_id:
                agent = self.agents.get(agent_id)
                return self._agent_to_dict(agent) if agent else None
            else:
                return [self._agent_to_dict(agent) for agent in self.agents.values()]
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration metrics."""
        with self.lock:
            # Calculate current metrics
            total_agents = len(self.agents)
            busy_agents = sum(1 for agent in self.agents.values() if agent.status != AgentStatus.IDLE)
            
            queue_sizes = {
                priority.name: len(queue) for priority, queue in self.priority_queues.items()
            }
            
            return {
                'agents': {
                    'total': total_agents,
                    'busy': busy_agents,
                    'idle': total_agents - busy_agents,
                    'utilization': (busy_agents / total_agents * 100) if total_agents > 0 else 0
                },
                'tasks': {
                    'queued': sum(len(queue) for queue in self.priority_queues.values()),
                    'active': len(self.active_tasks),
                    'completed': len(self.completed_tasks),
                    'queue_breakdown': queue_sizes
                },
                'performance': dict(self.performance_metrics),
                'architecture': self.default_architecture.value,
                'load_balancing': self.load_balancing.value,
                'timestamp': datetime.now().isoformat()
            }
    
    def _start_orchestration(self):
        """Start orchestration background threads."""
        self.orchestration_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        
        self.orchestration_thread.start()
        self.heartbeat_thread.start()
    
    def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.orchestration_active:
            try:
                with self.lock:
                    self._process_task_queues()
                    self._update_performance_metrics()
                
                time.sleep(0.1)  # 100ms cycle
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                time.sleep(1.0)
    
    def _process_task_queues(self):
        """Process tasks from priority queues."""
        # Process by priority (highest first)
        for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
            queue = self.priority_queues[priority]
            
            while queue and self._has_available_agents():
                task = queue.popleft()
                
                # Check dependencies
                if self._dependencies_satisfied(task):
                    agent = self._assign_task(task)
                    if agent:
                        self._execute_task_on_agent(task, agent)
                    else:
                        # No suitable agent, put back in queue
                        queue.appendleft(task)
                        break
                else:
                    # Dependencies not satisfied, put back in queue
                    queue.append(task)
    
    def _has_available_agents(self) -> bool:
        """Check if any agents are available."""
        return any(agent.is_available() for agent in self.agents.values())
    
    def _dependencies_satisfied(self, task: EnhancedTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        # Check if all dependency tasks are completed
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
            
            # Check if dependency task was successful
            dep_task = self.completed_tasks[dep_task_id]
            if dep_task.status != "completed":
                return False
        
        return True
    
    def _assign_task(self, task: EnhancedTask) -> Optional[SwarmAgent]:
        """Assign task to best available agent."""
        load_balancer = self.load_balancers.get(
            self.load_balancing, 
            self._assign_intelligent_routing
        )
        
        return load_balancer(task)
    
    def _assign_round_robin(self, task: EnhancedTask) -> Optional[SwarmAgent]:
        """Round-robin assignment."""
        available_agents = [agent for agent in self.agents.values() if agent.can_handle_task(task)]
        if not available_agents:
            return None
        
        # Simple round-robin based on agent_id
        sorted_agents = sorted(available_agents, key=lambda x: x.agent_id)
        return sorted_agents[0]
    
    def _assign_least_loaded(self, task: EnhancedTask) -> Optional[SwarmAgent]:
        """Assign to least loaded agent."""
        available_agents = [agent for agent in self.agents.values() if agent.can_handle_task(task)]
        if not available_agents:
            return None
        
        return min(available_agents, key=lambda x: x.load_factor)
    
    def _assign_capability_based(self, task: EnhancedTask) -> Optional[SwarmAgent]:
        """Assign based on capability match."""
        available_agents = [agent for agent in self.agents.values() if agent.can_handle_task(task)]
        if not available_agents:
            return None
        
        # Prefer agents with exact capability matches
        exact_matches = [
            agent for agent in available_agents
            if all(cap in agent.capabilities for cap in task.required_capabilities)
        ]
        
        if exact_matches:
            return min(exact_matches, key=lambda x: x.load_factor)
        
        return available_agents[0]
    
    def _assign_performance_weighted(self, task: EnhancedTask) -> Optional[SwarmAgent]:
        """Assign based on performance metrics."""
        available_agents = [agent for agent in self.agents.values() if agent.can_handle_task(task)]
        if not available_agents:
            return None
        
        return max(available_agents, key=lambda x: x.quality_score * (1.0 - x.load_factor))
    
    def _assign_intelligent_routing(self, task: EnhancedTask) -> Optional[SwarmAgent]:
        """Intelligent assignment based on multiple factors."""
        available_agents = [agent for agent in self.agents.values() if agent.can_handle_task(task)]
        if not available_agents:
            return None
        
        # Calculate suitability scores
        scored_agents = [
            (agent, agent.calculate_suitability_score(task))
            for agent in available_agents
        ]
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        return scored_agents[0][0] if scored_agents else None
    
    def _execute_task_on_agent(self, task: EnhancedTask, agent: SwarmAgent):
        """Execute task on assigned agent."""
        task.assigned_agent = agent.agent_id
        task.status = "assigned"
        task.started_at = datetime.now()
        
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        
        # Update agent state
        agent.current_tasks.append(task.task_id)
        agent.status = AgentStatus.RUNNING
        
        # Submit to thread pool for execution
        future = self.agent_pool.submit(self._run_task, task, agent)
        
        # Handle completion
        def on_completion(fut):
            try:
                result = fut.result()
                self._handle_task_completion(task, agent, result)
            except Exception as e:
                self._handle_task_error(task, agent, str(e))
        
        future.add_done_callback(on_completion)
    
    def _run_task(self, task: EnhancedTask, agent: SwarmAgent) -> Any:
        """Actually run the task (override this method)."""
        # This is a placeholder - should be overridden to execute actual tasks
        time.sleep(random.uniform(0.5, 2.0))  # Simulate work
        
        # Simulate occasional failures
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Simulated task failure")
        
        return f"Task {task.task_id} completed by {agent.agent_id}"
    
    def _handle_task_completion(self, task: EnhancedTask, agent: SwarmAgent, result: Any):
        """Handle successful task completion."""
        with self.lock:
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            # Update agent
            agent.current_tasks.remove(task.task_id)
            if not agent.current_tasks:
                agent.status = AgentStatus.IDLE
            
            # Move to completed tasks
            self.active_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = task
            
            # Update metrics
            self.performance_metrics['successful_tasks'] += 1
            self.performance_metrics['total_tasks_processed'] += 1
            
            # Update agent performance
            execution_time = (task.completed_at - task.started_at).total_seconds()
            agent.average_execution_time = (
                (agent.average_execution_time + execution_time) / 2
                if agent.average_execution_time > 0 else execution_time
            )
            
            logger.debug(f"Task {task.task_id} completed successfully on {agent.agent_id}")
    
    def _handle_task_error(self, task: EnhancedTask, agent: SwarmAgent, error_message: str):
        """Handle task execution error."""
        with self.lock:
            task.error_message = error_message
            task.retry_count += 1
            
            # Update agent
            agent.current_tasks.remove(task.task_id)
            if not agent.current_tasks:
                agent.status = AgentStatus.IDLE
            
            # Update agent error rate
            agent.error_rate = min(agent.error_rate + 0.1, 1.0)
            
            # Retry logic
            if task.retry_count <= task.max_retries:
                task.status = "pending"
                task.assigned_agent = None
                self.priority_queues[task.priority].appendleft(task)
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
            else:
                task.status = "failed"
                task.completed_at = datetime.now()
                self.completed_tasks[task.task_id] = task
                self.performance_metrics['failed_tasks'] += 1
                logger.error(f"Task {task.task_id} failed permanently: {error_message}")
            
            # Remove from active tasks
            self.active_tasks.pop(task.task_id, None)
            self.performance_metrics['total_tasks_processed'] += 1
    
    # Architecture-specific execution methods
    def _execute_sequential_workflow(self, tasks: List[EnhancedTask]) -> Dict[str, Any]:
        """Execute tasks sequentially."""
        for i, task in enumerate(tasks):
            if i > 0:
                task.dependencies = [tasks[i-1].task_id]
            self.submit_task(task)
        
        return {'workflow_type': 'sequential', 'task_count': len(tasks)}
    
    def _execute_concurrent_workflow(self, tasks: List[EnhancedTask]) -> Dict[str, Any]:
        """Execute tasks concurrently."""
        for task in tasks:
            self.submit_task(task)
        
        return {'workflow_type': 'concurrent', 'task_count': len(tasks)}
    
    def _execute_hierarchical_swarm(self, tasks: List[EnhancedTask]) -> Dict[str, Any]:
        """Execute with hierarchical task distribution."""
        # Group tasks by priority
        priority_groups = defaultdict(list)
        for task in tasks:
            priority_groups[task.priority].append(task)
        
        # Submit higher priority groups first
        for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
            for task in priority_groups[priority]:
                self.submit_task(task)
        
        return {'workflow_type': 'hierarchical', 'task_count': len(tasks)}
    
    def _execute_agent_rearrange(self, tasks: List[EnhancedTask]) -> Dict[str, Any]:
        """Execute with dynamic agent reassignment."""
        # This would implement more sophisticated agent reassignment logic
        return self._execute_concurrent_workflow(tasks)
    
    def _execute_mixture_of_agents(self, tasks: List[EnhancedTask]) -> Dict[str, Any]:
        """Execute using mixture of agent types."""
        # Group tasks by required capabilities
        capability_groups = defaultdict(list)
        for task in tasks:
            key = tuple(sorted(task.required_capabilities))
            capability_groups[key].append(task)
        
        # Submit each group
        for capability_set, group_tasks in capability_groups.items():
            for task in group_tasks:
                self.submit_task(task)
        
        return {'workflow_type': 'mixture_of_agents', 'task_count': len(tasks)}
    
    def _execute_swarm_router(self, tasks: List[EnhancedTask]) -> Dict[str, Any]:
        """Execute with intelligent routing."""
        # This implementation uses the intelligent routing assignment
        for task in tasks:
            task.metadata['routing_strategy'] = 'intelligent'
            self.submit_task(task)
        
        return {'workflow_type': 'swarm_router', 'task_count': len(tasks)}
    
    def _execute_dynamic_pipeline(self, tasks: List[EnhancedTask]) -> Dict[str, Any]:
        """Execute with dynamic pipeline adjustment."""
        # Create pipeline stages based on task dependencies
        stages = self._analyze_task_dependencies(tasks)
        
        for stage in stages:
            for task in stage:
                self.submit_task(task)
        
        return {'workflow_type': 'dynamic_pipeline', 'task_count': len(tasks), 'stages': len(stages)}
    
    def _analyze_task_dependencies(self, tasks: List[EnhancedTask]) -> List[List[EnhancedTask]]:
        """Analyze task dependencies and create execution stages."""
        # Simple dependency analysis - can be enhanced
        stages = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            current_stage = []
            
            for task in remaining_tasks[:]:
                # Check if all dependencies are in previous stages
                deps_satisfied = all(
                    any(t.task_id == dep for stage in stages for t in stage)
                    for dep in task.dependencies
                )
                
                if not task.dependencies or deps_satisfied:
                    current_stage.append(task)
                    remaining_tasks.remove(task)
            
            if current_stage:
                stages.append(current_stage)
            else:
                # Circular dependency or error - add remaining tasks
                stages.append(remaining_tasks)
                break
        
        return stages
    
    def _heartbeat_loop(self):
        """Monitor agent health and performance."""
        while self.orchestration_active:
            try:
                with self.lock:
                    current_time = datetime.now()
                    
                    for agent in self.agents.values():
                        # Update load factor
                        agent.load_factor = len(agent.current_tasks) / agent.max_concurrent_tasks
                        
                        # Check for stale agents
                        if (current_time - agent.last_heartbeat).total_seconds() > 60:
                            agent.status = AgentStatus.OFFLINE
                            logger.warning(f"Agent {agent.agent_id} appears offline")
                        
                        # Update last heartbeat
                        agent.last_heartbeat = current_time
                
                time.sleep(10)  # 10 second heartbeat
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(10)
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        total_processed = self.performance_metrics['total_tasks_processed']
        if total_processed > 0:
            success_rate = self.performance_metrics['successful_tasks'] / total_processed
            self.performance_metrics['success_rate'] = success_rate
    
    def _agent_to_dict(self, agent: SwarmAgent) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'capabilities': agent.capabilities,
            'status': agent.status.value,
            'current_tasks': agent.current_tasks,
            'max_concurrent_tasks': agent.max_concurrent_tasks,
            'load_factor': agent.load_factor,
            'quality_score': agent.quality_score,
            'error_rate': agent.error_rate,
            'average_execution_time': agent.average_execution_time,
            'last_heartbeat': agent.last_heartbeat.isoformat()
        }
    
    def shutdown(self):
        """Shutdown orchestrator."""
        self.orchestration_active = False
        
        if self.orchestration_thread:
            self.orchestration_thread.join(timeout=5)
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        
        self.agent_pool.shutdown(wait=True)
        
        logger.info("Enhanced Agent Orchestrator shutdown")

# Global enhanced orchestrator instance
enhanced_orchestrator = None