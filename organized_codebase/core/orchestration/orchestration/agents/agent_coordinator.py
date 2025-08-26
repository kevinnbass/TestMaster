"""
Agent Coordination System
Inspired by Agency-Swarm patterns, adapted for TestMaster Intelligence Platform
"""

import asyncio
import inspect
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from collections import defaultdict
import queue


@dataclass
class AgentTask:
    """Represents a task for an agent"""
    task_id: str
    agent_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 1
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'task_type': self.task_type,
            'description': self.description,
            'parameters': self.parameters,
            'priority': self.priority,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error
        }


@dataclass
class AgentCapability:
    """Defines an agent's capabilities"""
    name: str
    description: str
    task_types: List[str]
    max_concurrent_tasks: int = 1
    estimated_task_time: float = 60.0  # seconds
    specialized_domains: List[str] = field(default_factory=list)
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if this capability can handle the given task type"""
        return task_type in self.task_types


class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class IntelligenceAgent:
    """Represents an intelligence agent in the coordination system"""
    agent_id: str
    name: str
    capabilities: List[AgentCapability]
    status: AgentStatus = AgentStatus.IDLE
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 60.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_accept_task(self, task_type: str) -> bool:
        """Check if agent can accept a new task of given type"""
        if self.status in [AgentStatus.ERROR, AgentStatus.OFFLINE]:
            return False
            
        # Check if agent has capability for this task type
        has_capability = any(cap.can_handle_task(task_type) for cap in self.capabilities)
        if not has_capability:
            return False
        
        # Check if agent has capacity
        max_concurrent = max((cap.max_concurrent_tasks for cap in self.capabilities 
                             if cap.can_handle_task(task_type)), default=1)
        
        return len(self.current_tasks) < max_concurrent
    
    def get_workload_score(self) -> float:
        """Calculate current workload score (0.0 = idle, 1.0 = fully loaded)"""
        if not self.capabilities:
            return 1.0
        
        max_capacity = sum(cap.max_concurrent_tasks for cap in self.capabilities)
        current_load = len(self.current_tasks)
        
        return min(current_load / max_capacity, 1.0) if max_capacity > 0 else 1.0


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AgentCoordinator:
    """
    Coordinates multiple intelligence agents for parallel processing
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.agents: Dict[str, IntelligenceAgent] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.failed_tasks: List[AgentTask] = []
        
        # Threading
        self.coordinator_active = True
        self.coordinator_thread = None
        self.task_threads: Dict[str, threading.Thread] = {}
        
        # Statistics
        self.stats = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_completion_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
        
    def register_agent(self, agent: IntelligenceAgent):
        """Register a new agent with the coordinator"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
        
        # Trigger agent registered event
        self._trigger_event('agent_registered', agent)
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the coordinator"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Cancel any active tasks for this agent
            tasks_to_cancel = [task for task in self.active_tasks.values() 
                             if task.agent_id == agent_id]
            
            for task in tasks_to_cancel:
                self._cancel_task(task.task_id, "Agent unregistered")
            
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent.name} ({agent_id})")
            
            # Trigger agent unregistered event
            self._trigger_event('agent_unregistered', agent)
    
    def submit_task(self, task_type: str, description: str, 
                   parameters: Dict[str, Any] = None, 
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   preferred_agent: str = None) -> str:
        """Submit a new task to the coordination system"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Find best agent for this task
        selected_agent_id = self._select_agent(task_type, preferred_agent)
        
        if not selected_agent_id:
            self.logger.warning(f"No available agent for task type: {task_type}")
            return None
        
        task = AgentTask(
            task_id=task_id,
            agent_id=selected_agent_id,
            task_type=task_type,
            description=description,
            parameters=parameters or {},
            priority=priority.value
        )
        
        # Add to queue
        self.task_queue.put((priority.value, task))
        self.stats['tasks_created'] += 1
        
        self.logger.info(f"Submitted task {task_id} to agent {selected_agent_id}")
        
        # Trigger task submitted event
        self._trigger_event('task_submitted', task)
        
        return task_id
    
    def _select_agent(self, task_type: str, preferred_agent: str = None) -> Optional[str]:
        """Select the best agent for a given task type"""
        # If preferred agent specified and available, use it
        if preferred_agent and preferred_agent in self.agents:
            agent = self.agents[preferred_agent]
            if agent.can_accept_task(task_type):
                return preferred_agent
        
        # Find all capable agents
        capable_agents = [
            agent for agent in self.agents.values() 
            if agent.can_accept_task(task_type)
        ]
        
        if not capable_agents:
            return None
        
        # Sort by workload (prefer less loaded agents)
        capable_agents.sort(key=lambda a: (a.get_workload_score(), a.average_task_time))
        
        return capable_agents[0].agent_id
    
    def start_coordination(self):
        """Start the coordination system"""
        if self.coordinator_thread is not None:
            self.logger.warning("Coordinator already running")
            return
        
        self.coordinator_active = True
        self.coordinator_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordinator_thread.start()
        
        self.logger.info("Agent coordination system started")
    
    def stop_coordination(self):
        """Stop the coordination system"""
        self.coordinator_active = False
        
        # Wait for coordinator thread to finish
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=5)
        
        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            self._cancel_task(task_id, "System shutdown")
        
        self.logger.info("Agent coordination system stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.coordinator_active:
            try:
                # Get next task from queue (block for up to 1 second)
                try:
                    priority, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Verify agent is still available
                if task.agent_id not in self.agents:
                    self.logger.warning(f"Agent {task.agent_id} no longer available for task {task.task_id}")
                    self._fail_task(task, "Agent no longer available")
                    continue
                
                agent = self.agents[task.agent_id]
                
                if not agent.can_accept_task(task.task_type):
                    # Try to reassign to another agent
                    new_agent_id = self._select_agent(task.task_type)
                    if new_agent_id:
                        task.agent_id = new_agent_id
                        agent = self.agents[new_agent_id]
                    else:
                        self.logger.warning(f"No available agent for reassigned task {task.task_id}")
                        self._fail_task(task, "No available agent")
                        continue
                
                # Assign task to agent
                self._assign_task(task, agent)
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
    
    def _assign_task(self, task: AgentTask, agent: IntelligenceAgent):
        """Assign a task to a specific agent"""
        task.status = "assigned"
        task.assigned_at = datetime.now()
        
        # Update agent state
        agent.current_tasks.append(task.task_id)
        if len(agent.current_tasks) >= sum(cap.max_concurrent_tasks for cap in agent.capabilities):
            agent.status = AgentStatus.OVERLOADED
        else:
            agent.status = AgentStatus.BUSY
        
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        
        # Start task execution in separate thread
        task_thread = threading.Thread(
            target=self._execute_task,
            args=(task, agent),
            daemon=True
        )
        self.task_threads[task.task_id] = task_thread
        task_thread.start()
        
        self.logger.info(f"Assigned task {task.task_id} to agent {agent.name}")
        
        # Trigger task assigned event
        self._trigger_event('task_assigned', task, agent)
    
    def _execute_task(self, task: AgentTask, agent: IntelligenceAgent):
        """Execute a task (run in separate thread)"""
        start_time = datetime.now()
        
        try:
            task.status = "running"
            
            # Trigger task started event
            self._trigger_event('task_started', task, agent)
            
            # Simulate task execution (replace with actual agent execution)
            result = self._simulate_task_execution(task, agent)
            
            # Task completed successfully
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            execution_time = (task.completed_at - start_time).total_seconds()
            
            # Update statistics
            agent.completed_tasks += 1
            agent.average_task_time = (
                (agent.average_task_time * (agent.completed_tasks - 1) + execution_time) /
                agent.completed_tasks
            )
            
            self.stats['tasks_completed'] += 1
            self.stats['total_processing_time'] += execution_time
            self.stats['average_completion_time'] = (
                self.stats['total_processing_time'] / self.stats['tasks_completed']
            )
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            
            self.logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            
            # Trigger task completed event
            self._trigger_event('task_completed', task, agent)
            
        except Exception as e:
            # Task failed
            task.status = "failed"
            task.completed_at = datetime.now()
            task.error = str(e)
            
            agent.failed_tasks += 1
            self.stats['tasks_failed'] += 1
            
            # Move to failed tasks
            self.failed_tasks.append(task)
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            # Trigger task failed event
            self._trigger_event('task_failed', task, agent)
        
        finally:
            # Clean up
            self._cleanup_task(task, agent)
    
    def _simulate_task_execution(self, task: AgentTask, agent: IntelligenceAgent) -> Dict[str, Any]:
        """Simulate task execution (replace with actual agent-specific logic)"""
        import time
        import random
        
        # Simulate processing time based on task type and agent capability
        base_time = 1.0
        for cap in agent.capabilities:
            if cap.can_handle_task(task.task_type):
                base_time = cap.estimated_task_time
                break
        
        # Add some randomness
        processing_time = base_time * random.uniform(0.5, 1.5)
        time.sleep(min(processing_time, 5.0))  # Cap at 5 seconds for testing
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated task failure")
        
        # Return simulated result
        return {
            'task_id': task.task_id,
            'status': 'success',
            'processing_time': processing_time,
            'agent_id': agent.agent_id,
            'result_data': f"Processed {task.task_type} task: {task.description}"
        }
    
    def _cleanup_task(self, task: AgentTask, agent: IntelligenceAgent):
        """Clean up after task completion"""
        # Remove from active tasks
        self.active_tasks.pop(task.task_id, None)
        
        # Remove from agent's current tasks
        if task.task_id in agent.current_tasks:
            agent.current_tasks.remove(task.task_id)
        
        # Update agent status
        if not agent.current_tasks:
            agent.status = AgentStatus.IDLE
        elif len(agent.current_tasks) < sum(cap.max_concurrent_tasks for cap in agent.capabilities):
            agent.status = AgentStatus.BUSY
        
        # Clean up thread
        self.task_threads.pop(task.task_id, None)
    
    def _cancel_task(self, task_id: str, reason: str = "Cancelled"):
        """Cancel a task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "cancelled"
            task.error = reason
            task.completed_at = datetime.now()
            
            # Find agent and clean up
            if task.agent_id in self.agents:
                agent = self.agents[task.agent_id]
                self._cleanup_task(task, agent)
            
            # Move to failed tasks
            self.failed_tasks.append(task)
            
            self.logger.info(f"Cancelled task {task_id}: {reason}")
    
    def _fail_task(self, task: AgentTask, reason: str):
        """Fail a task"""
        task.status = "failed"
        task.error = reason
        task.completed_at = datetime.now()
        
        self.failed_tasks.append(task)
        self.stats['tasks_failed'] += 1
        
        self.logger.warning(f"Failed task {task.task_id}: {reason}")
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'status': agent.status.value,
            'current_tasks': len(agent.current_tasks),
            'completed_tasks': agent.completed_tasks,
            'failed_tasks': agent.failed_tasks,
            'average_task_time': agent.average_task_time,
            'workload_score': agent.get_workload_score(),
            'capabilities': [cap.name for cap in agent.capabilities],
            'last_heartbeat': agent.last_heartbeat.isoformat()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'active_agents': len([a for a in self.agents.values() if a.status != AgentStatus.OFFLINE]),
            'total_agents': len(self.agents),
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'statistics': dict(self.stats),
            'agent_summary': {
                agent_id: {
                    'name': agent.name,
                    'status': agent.status.value,
                    'workload': agent.get_workload_score()
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register an event callback"""
        self.event_callbacks[event_type].append(callback)
    
    def _trigger_event(self, event_type: str, *args):
        """Trigger event callbacks"""
        for callback in self.event_callbacks[event_type]:
            try:
                callback(*args)
            except Exception as e:
                self.logger.error(f"Error in event callback {event_type}: {e}")


# Export
__all__ = [
    'AgentCoordinator', 
    'IntelligenceAgent', 
    'AgentTask', 
    'AgentCapability', 
    'AgentStatus', 
    'TaskPriority'
]