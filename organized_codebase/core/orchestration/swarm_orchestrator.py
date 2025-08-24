"""
Swarm Orchestration System
==========================

Distributed swarm-based test orchestration using Swarms patterns.
Provides intelligent task distribution and collective intelligence.

Author: TestMaster Team  
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import random
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


class SwarmTaskStatus(Enum):
    """Status of a swarm task"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class SwarmAgentState(Enum):
    """State of a swarm agent"""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class SwarmTask:
    """Task to be executed by the swarm"""
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    task_type: str = "test_execution"
    priority: int = 5  # 1-10, higher is more important
    payload: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    status: SwarmTaskStatus = SwarmTaskStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (dependencies met)"""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def is_timeout(self) -> bool:
        """Check if task has timed out"""
        if not self.started_at:
            return False
        return (datetime.now() - self.started_at).total_seconds() > self.timeout


@dataclass
class SwarmAgent:
    """Individual agent in the swarm"""
    agent_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:12]}")
    name: str = ""
    capabilities: Set[str] = field(default_factory=set)
    state: SwarmAgentState = SwarmAgentState.IDLE
    current_tasks: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    performance_score: float = 1.0  # 0-1, affects task assignment
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_accept_task(self, task: SwarmTask) -> bool:
        """Check if agent can accept a task"""
        if self.state in [SwarmAgentState.OFFLINE, SwarmAgentState.MAINTENANCE]:
            return False
        
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            return False
        
        # Check if agent has required capabilities
        required_capabilities = task.metadata.get("required_capabilities", set())
        if required_capabilities and not required_capabilities.issubset(self.capabilities):
            return False
        
        return True
    
    def update_performance(self, success: bool, execution_time: float):
        """Update agent performance metrics"""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_execution_time += execution_time
        
        # Calculate performance score (simple weighted average)
        if self.tasks_completed + self.tasks_failed > 0:
            success_rate = self.tasks_completed / (self.tasks_completed + self.tasks_failed)
            avg_time = self.total_execution_time / (self.tasks_completed + self.tasks_failed)
            
            # Normalize execution time (assume 60s is average)
            time_score = min(1.0, 60 / max(avg_time, 1))
            
            # Weighted performance score
            self.performance_score = 0.7 * success_rate + 0.3 * time_score


@dataclass
class SwarmConfig:
    """Configuration for swarm orchestration"""
    name: str = "TestSwarm"
    min_agents: int = 3
    max_agents: int = 50
    task_distribution_strategy: str = "performance_weighted"  # random, round_robin, performance_weighted, capability_based
    agent_heartbeat_timeout: int = 60  # seconds
    task_queue_size: int = 1000
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # Queue utilization
    scale_down_threshold: float = 0.2
    enable_task_stealing: bool = True  # Allow idle agents to steal tasks
    enable_collective_learning: bool = True  # Share learnings across swarm
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmOrchestrator:
    """
    Swarm-based orchestration system for distributed test execution.
    
    Features:
    - Dynamic agent management
    - Intelligent task distribution
    - Dependency-aware scheduling
    - Auto-scaling capabilities
    - Task stealing for load balancing
    - Collective intelligence and learning
    - Fault tolerance and recovery
    """
    
    def __init__(self, config: SwarmConfig = None):
        self.orchestrator_id = f"orchestrator_{uuid.uuid4().hex[:12]}"
        self.config = config or SwarmConfig()
        self.agents: Dict[str, SwarmAgent] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.task_queue: deque = deque(maxlen=self.config.task_queue_size)
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.task_dependencies: Dict[str, Set[str]] = {}  # task_id -> dependent_tasks
        self.collective_knowledge: Dict[str, Any] = {}  # Shared learnings
        self.started_at = datetime.now()
        self.statistics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "agents_joined": 0,
            "agents_left": 0,
            "total_execution_time": 0,
            "average_queue_size": 0
        }
        
        # Start background tasks
        self.scheduler_task = None
        self.monitor_task = None
        self.scaler_task = None
    
    def add_agent(self, agent: SwarmAgent) -> str:
        """Add an agent to the swarm"""
        self.agents[agent.agent_id] = agent
        self.statistics["agents_joined"] += 1
        
        logger.info(f"Agent {agent.name} ({agent.agent_id}) joined the swarm")
        return agent.agent_id
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the swarm"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Reassign any tasks from this agent
        for task_id in agent.current_tasks:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = SwarmTaskStatus.PENDING
                task.assigned_to = None
                self.task_queue.append(task_id)
        
        del self.agents[agent_id]
        self.statistics["agents_left"] += 1
        
        logger.info(f"Agent {agent.name} ({agent_id}) left the swarm")
        return True
    
    def submit_task(self, task: SwarmTask) -> str:
        """Submit a task to the swarm"""
        self.tasks[task.task_id] = task
        
        # Track dependencies
        for dep in task.dependencies:
            if dep not in self.task_dependencies:
                self.task_dependencies[dep] = set()
            self.task_dependencies[dep].add(task.task_id)
        
        # Add to queue if ready
        if task.is_ready(self.completed_tasks):
            self.task_queue.append(task.task_id)
        
        self.statistics["tasks_submitted"] += 1
        
        logger.info(f"Task {task.task_id} submitted to swarm")
        return task.task_id
    
    def submit_batch(self, tasks: List[SwarmTask]) -> List[str]:
        """Submit multiple tasks at once"""
        task_ids = []
        for task in tasks:
            task_id = self.submit_task(task)
            task_ids.append(task_id)
        return task_ids
    
    async def assign_task(self, task_id: str) -> Optional[str]:
        """Assign a task to an available agent"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        # Find suitable agent based on strategy
        agent = self._select_agent(task)
        
        if not agent:
            return None
        
        # Assign task
        task.status = SwarmTaskStatus.ASSIGNED
        task.assigned_to = agent.agent_id
        task.started_at = datetime.now()
        
        agent.current_tasks.append(task_id)
        agent.state = SwarmAgentState.BUSY if len(agent.current_tasks) >= agent.max_concurrent_tasks * 0.7 else SwarmAgentState.IDLE
        
        logger.info(f"Task {task_id} assigned to agent {agent.agent_id}")
        return agent.agent_id
    
    def _select_agent(self, task: SwarmTask) -> Optional[SwarmAgent]:
        """Select best agent for a task based on strategy"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.can_accept_task(task)
        ]
        
        if not available_agents:
            return None
        
        strategy = self.config.task_distribution_strategy
        
        if strategy == "random":
            return random.choice(available_agents)
        
        elif strategy == "round_robin":
            # Sort by number of current tasks
            return min(available_agents, key=lambda a: len(a.current_tasks))
        
        elif strategy == "performance_weighted":
            # Weighted random based on performance
            total_score = sum(a.performance_score for a in available_agents)
            if total_score == 0:
                return random.choice(available_agents)
            
            rand = random.uniform(0, total_score)
            cumulative = 0
            
            for agent in available_agents:
                cumulative += agent.performance_score
                if rand <= cumulative:
                    return agent
            
            return available_agents[-1]
        
        elif strategy == "capability_based":
            # Prefer agents with matching capabilities
            required_caps = task.metadata.get("required_capabilities", set())
            if required_caps:
                # Sort by capability match
                available_agents.sort(
                    key=lambda a: len(a.capabilities.intersection(required_caps)),
                    reverse=True
                )
            return available_agents[0]
        
        return available_agents[0]
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        agent_id = task.assigned_to
        
        # Update task
        task.status = SwarmTaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = result
        
        # Update agent
        if agent_id and agent_id in self.agents:
            agent = self.agents[agent_id]
            if task_id in agent.current_tasks:
                agent.current_tasks.remove(task_id)
            
            execution_time = (task.completed_at - task.started_at).total_seconds()
            agent.update_performance(True, execution_time)
            
            if len(agent.current_tasks) == 0:
                agent.state = SwarmAgentState.IDLE
        
        # Add to completed set
        self.completed_tasks.add(task_id)
        
        # Check and queue dependent tasks
        if task_id in self.task_dependencies:
            for dependent_id in self.task_dependencies[task_id]:
                if dependent_id in self.tasks:
                    dependent_task = self.tasks[dependent_id]
                    if dependent_task.is_ready(self.completed_tasks) and dependent_task.status == SwarmTaskStatus.PENDING:
                        self.task_queue.append(dependent_id)
        
        # Update statistics
        self.statistics["tasks_completed"] += 1
        if task.started_at:
            self.statistics["total_execution_time"] += (task.completed_at - task.started_at).total_seconds()
        
        # Share learnings if enabled
        if self.config.enable_collective_learning:
            self._update_collective_knowledge(task, result)
        
        logger.info(f"Task {task_id} completed successfully")
        return True
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        agent_id = task.assigned_to
        
        # Update agent
        if agent_id and agent_id in self.agents:
            agent = self.agents[agent_id]
            if task_id in agent.current_tasks:
                agent.current_tasks.remove(task_id)
            
            if task.started_at:
                execution_time = (datetime.now() - task.started_at).total_seconds()
                agent.update_performance(False, execution_time)
            
            if len(agent.current_tasks) == 0:
                agent.state = SwarmAgentState.IDLE
        
        # Check if we should retry
        if task.retry_count < task.max_retries:
            task.status = SwarmTaskStatus.RETRYING
            task.retry_count += 1
            task.assigned_to = None
            task.started_at = None
            self.task_queue.append(task_id)
            self.statistics["tasks_retried"] += 1
            logger.info(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
        else:
            task.status = SwarmTaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.now()
            self.failed_tasks.add(task_id)
            self.statistics["tasks_failed"] += 1
            logger.error(f"Task {task_id} failed permanently: {error}")
        
        return True
    
    def _update_collective_knowledge(self, task: SwarmTask, result: Dict[str, Any]):
        """Update collective knowledge from task results"""
        task_type = task.task_type
        
        if task_type not in self.collective_knowledge:
            self.collective_knowledge[task_type] = {
                "patterns": {},
                "optimizations": [],
                "common_errors": {},
                "performance_hints": []
            }
        
        knowledge = self.collective_knowledge[task_type]
        
        # Extract patterns from successful execution
        if "patterns" in result:
            for pattern, count in result["patterns"].items():
                if pattern not in knowledge["patterns"]:
                    knowledge["patterns"][pattern] = 0
                knowledge["patterns"][pattern] += count
        
        # Store optimization hints
        if "optimization" in result:
            knowledge["optimizations"].append({
                "timestamp": datetime.now().isoformat(),
                "hint": result["optimization"]
            })
        
        # Track common errors for future prevention
        if "errors" in result:
            for error_type in result["errors"]:
                if error_type not in knowledge["common_errors"]:
                    knowledge["common_errors"][error_type] = 0
                knowledge["common_errors"][error_type] += 1
    
    async def steal_task(self, agent_id: str) -> Optional[str]:
        """Allow an idle agent to steal a task from overloaded agent"""
        if not self.config.enable_task_stealing:
            return None
        
        if agent_id not in self.agents:
            return None
        
        stealing_agent = self.agents[agent_id]
        
        if stealing_agent.state != SwarmAgentState.IDLE:
            return None
        
        # Find overloaded agents
        overloaded_agents = [
            a for a in self.agents.values()
            if a.state == SwarmAgentState.OVERLOADED and len(a.current_tasks) > 1
        ]
        
        if not overloaded_agents:
            return None
        
        # Select agent with most tasks
        victim_agent = max(overloaded_agents, key=lambda a: len(a.current_tasks))
        
        # Steal oldest unstarted task
        for task_id in victim_agent.current_tasks:
            task = self.tasks.get(task_id)
            if task and task.status == SwarmTaskStatus.ASSIGNED and not task.started_at:
                # Transfer task
                victim_agent.current_tasks.remove(task_id)
                stealing_agent.current_tasks.append(task_id)
                task.assigned_to = agent_id
                
                # Update states
                if len(victim_agent.current_tasks) < victim_agent.max_concurrent_tasks * 0.7:
                    victim_agent.state = SwarmAgentState.BUSY
                
                logger.info(f"Agent {agent_id} stole task {task_id} from agent {victim_agent.agent_id}")
                return task_id
        
        return None
    
    async def schedule_tasks(self):
        """Main scheduling loop"""
        while True:
            try:
                # Process task queue
                tasks_to_assign = []
                
                while self.task_queue and len(tasks_to_assign) < 10:  # Batch assignment
                    task_id = self.task_queue.popleft()
                    if task_id in self.tasks:
                        task = self.tasks[task_id]
                        if task.is_ready(self.completed_tasks):
                            tasks_to_assign.append(task_id)
                        else:
                            self.task_queue.append(task_id)  # Put back if not ready
                
                # Assign tasks
                for task_id in tasks_to_assign:
                    await self.assign_task(task_id)
                
                # Check for timed out tasks
                for task in self.tasks.values():
                    if task.status == SwarmTaskStatus.IN_PROGRESS and task.is_timeout():
                        await self.fail_task(task.task_id, "Task timeout")
                
                # Update statistics
                self.statistics["average_queue_size"] = (
                    self.statistics["average_queue_size"] * 0.9 + len(self.task_queue) * 0.1
                )
                
                await asyncio.sleep(1)  # Scheduling interval
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)
    
    async def monitor_agents(self):
        """Monitor agent health and performance"""
        while True:
            try:
                current_time = datetime.now()
                
                for agent in list(self.agents.values()):
                    # Check heartbeat
                    time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.config.agent_heartbeat_timeout:
                        if agent.state != SwarmAgentState.OFFLINE:
                            agent.state = SwarmAgentState.OFFLINE
                            logger.warning(f"Agent {agent.agent_id} is offline")
                    
                    # Update agent state based on load
                    if agent.state not in [SwarmAgentState.OFFLINE, SwarmAgentState.MAINTENANCE]:
                        load_percentage = len(agent.current_tasks) / agent.max_concurrent_tasks
                        
                        if load_percentage >= 0.9:
                            agent.state = SwarmAgentState.OVERLOADED
                        elif load_percentage >= 0.5:
                            agent.state = SwarmAgentState.BUSY
                        else:
                            agent.state = SwarmAgentState.IDLE
                
                await asyncio.sleep(10)  # Monitor interval
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(10)
    
    async def auto_scale(self):
        """Auto-scale swarm based on load"""
        if not self.config.enable_auto_scaling:
            return
        
        while True:
            try:
                # Calculate current load
                queue_utilization = len(self.task_queue) / self.config.task_queue_size
                active_agents = len([a for a in self.agents.values() if a.state != SwarmAgentState.OFFLINE])
                
                # Scale up logic
                if queue_utilization > self.config.scale_up_threshold:
                    if active_agents < self.config.max_agents:
                        # Create new agent (simplified)
                        new_agent = SwarmAgent(
                            name=f"AutoAgent_{len(self.agents)}",
                            capabilities={"test_execution", "test_analysis"}
                        )
                        self.add_agent(new_agent)
                        logger.info(f"Scaled up: Added agent {new_agent.agent_id}")
                
                # Scale down logic
                elif queue_utilization < self.config.scale_down_threshold:
                    if active_agents > self.config.min_agents:
                        # Remove idle agent
                        idle_agents = [
                            a for a in self.agents.values()
                            if a.state == SwarmAgentState.IDLE and not a.current_tasks
                        ]
                        if idle_agents:
                            agent_to_remove = min(idle_agents, key=lambda a: a.performance_score)
                            self.remove_agent(agent_to_remove.agent_id)
                            logger.info(f"Scaled down: Removed agent {agent_to_remove.agent_id}")
                
                await asyncio.sleep(30)  # Scaling interval
                
            except Exception as e:
                logger.error(f"Auto-scale error: {e}")
                await asyncio.sleep(30)
    
    async def start(self):
        """Start the swarm orchestrator"""
        logger.info(f"Starting swarm orchestrator {self.orchestrator_id}")
        
        # Start background tasks
        self.scheduler_task = asyncio.create_task(self.schedule_tasks())
        self.monitor_task = asyncio.create_task(self.monitor_agents())
        
        if self.config.enable_auto_scaling:
            self.scaler_task = asyncio.create_task(self.auto_scale())
        
        # Initialize minimum agents
        for i in range(self.config.min_agents):
            agent = SwarmAgent(
                name=f"InitialAgent_{i}",
                capabilities={"test_execution", "test_analysis", "test_reporting"}
            )
            self.add_agent(agent)
        
        logger.info(f"Swarm orchestrator started with {self.config.min_agents} initial agents")
    
    async def stop(self):
        """Stop the swarm orchestrator"""
        logger.info(f"Stopping swarm orchestrator {self.orchestrator_id}")
        
        # Cancel background tasks
        if self.scheduler_task:
            self.scheduler_task.cancel()
        if self.monitor_task:
            self.monitor_task.cancel()
        if self.scaler_task:
            self.scaler_task.cancel()
        
        # Wait for tasks to complete
        for task in self.tasks.values():
            if task.status in [SwarmTaskStatus.IN_PROGRESS, SwarmTaskStatus.ASSIGNED]:
                task.status = SwarmTaskStatus.CANCELLED
        
        logger.info("Swarm orchestrator stopped")
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        active_agents = [a for a in self.agents.values() if a.state != SwarmAgentState.OFFLINE]
        
        return {
            "orchestrator_id": self.orchestrator_id,
            "uptime_seconds": (datetime.now() - self.started_at).total_seconds(),
            "config": {
                "name": self.config.name,
                "min_agents": self.config.min_agents,
                "max_agents": self.config.max_agents,
                "strategy": self.config.task_distribution_strategy
            },
            "agents": {
                "total": len(self.agents),
                "active": len(active_agents),
                "idle": len([a for a in active_agents if a.state == SwarmAgentState.IDLE]),
                "busy": len([a for a in active_agents if a.state == SwarmAgentState.BUSY]),
                "overloaded": len([a for a in active_agents if a.state == SwarmAgentState.OVERLOADED])
            },
            "tasks": {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() if t.status == SwarmTaskStatus.PENDING]),
                "in_progress": len([t for t in self.tasks.values() if t.status == SwarmTaskStatus.IN_PROGRESS]),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "queue_size": len(self.task_queue)
            },
            "performance": {
                "average_agent_score": sum(a.performance_score for a in active_agents) / len(active_agents) if active_agents else 0,
                "total_execution_time": self.statistics["total_execution_time"],
                "average_task_time": self.statistics["total_execution_time"] / self.statistics["tasks_completed"] if self.statistics["tasks_completed"] > 0 else 0
            },
            "statistics": self.statistics,
            "collective_knowledge_topics": list(self.collective_knowledge.keys())
        }
    
    def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "state": agent.state.value,
            "capabilities": list(agent.capabilities),
            "current_tasks": agent.current_tasks,
            "max_concurrent_tasks": agent.max_concurrent_tasks,
            "performance_score": agent.performance_score,
            "tasks_completed": agent.tasks_completed,
            "tasks_failed": agent.tasks_failed,
            "average_execution_time": agent.total_execution_time / (agent.tasks_completed + agent.tasks_failed) if (agent.tasks_completed + agent.tasks_failed) > 0 else 0,
            "last_heartbeat": agent.last_heartbeat.isoformat(),
            "metadata": agent.metadata
        }
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "priority": task.priority,
            "status": task.status.value,
            "assigned_to": task.assigned_to,
            "dependencies": task.dependencies,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "execution_time": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else None,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "timeout": task.timeout,
            "result": task.result,
            "error": task.error,
            "metadata": task.metadata
        }