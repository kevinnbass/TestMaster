"""
Swarm Orchestration Engine
=========================

Swarm-based orchestration with intelligent task distribution.
Extracted from unified_orchestrator.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import logging
import time
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from .data_models import SwarmAgent, SwarmTask, SwarmAgentState, SwarmTaskStatus


class SwarmOrchestrationEngine:
    """Swarm-based orchestration with intelligent task distribution."""
    
    def __init__(self):
        self.agents: Dict[str, SwarmAgent] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.swarm_statistics = {
            "total_tasks_processed": 0,
            "average_task_duration": 0.0,
            "agent_utilization": 0.0,
            "success_rate": 0.0
        }
        
    def register_agent(self, agent: SwarmAgent) -> bool:
        """Register agent in the swarm."""
        try:
            self.agents[agent.agent_id] = agent
            logging.info(f"Agent registered: {agent.agent_id} ({agent.agent_type})")
            return True
        except Exception as e:
            logging.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    def submit_task(self, task: SwarmTask) -> bool:
        """Submit task to swarm."""
        try:
            self.tasks[task.task_id] = task
            self.task_queue.append(task.task_id)
            logging.info(f"Task submitted: {task.task_id} ({task.task_type})")
            return True
        except Exception as e:
            logging.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def find_best_agent(self, task: SwarmTask) -> Optional[SwarmAgent]:
        """Find best agent for task using intelligent matching."""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state == SwarmAgentState.IDLE
        ]
        
        if not available_agents:
            return None
        
        # Score agents based on capabilities and performance
        scored_agents = []
        for agent in available_agents:
            score = 0.0
            
            # Capability match score
            if task.required_capabilities:
                matched_capabilities = len(set(task.required_capabilities) & set(agent.capabilities))
                score += (matched_capabilities / len(task.required_capabilities)) * 0.6
            else:
                score += 0.6  # No specific requirements
            
            # Performance score
            score += agent.performance_score * 0.3
            
            # Agent type preference
            if agent.agent_type == task.task_type:
                score += 0.1
            
            scored_agents.append((agent, score))
        
        # Sort by score and return best
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0] if scored_agents else None
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign task to specific agent."""
        if task_id not in self.tasks or agent_id not in self.agents:
            return False
        
        task = self.tasks[task_id]
        agent = self.agents[agent_id]
        
        if agent.state != SwarmAgentState.IDLE:
            return False
        
        try:
            # Update task
            task.status = SwarmTaskStatus.ASSIGNED
            task.assigned_agent = agent_id
            
            # Update agent
            agent.state = SwarmAgentState.BUSY
            agent.current_task = task_id
            
            logging.info(f"Task assigned: {task_id} -> {agent_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to assign task {task_id} to {agent_id}: {e}")
            return False
    
    def execute_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Execute task (can be overridden for actual execution)."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        agent_id = task.assigned_agent
        
        if not agent_id or agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        
        try:
            # Update task status
            task.status = SwarmTaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            
            # Simulate task execution
            execution_time = random.uniform(1.0, 5.0)
            time.sleep(execution_time)
            
            # Simulate success/failure (90% success rate)
            success = random.random() < 0.9
            
            if success:
                # Task completed successfully
                task.status = SwarmTaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = {"status": "success", "execution_time": execution_time}
                
                # Update agent
                agent.state = SwarmAgentState.IDLE
                agent.current_task = None
                agent.total_tasks_completed += 1
                
                # Update performance
                if agent.total_tasks_completed > 0:
                    total_time = agent.average_execution_time * (agent.total_tasks_completed - 1) + execution_time
                    agent.average_execution_time = total_time / agent.total_tasks_completed
                
                self.completed_tasks.append(task_id)
                logging.info(f"Task completed: {task_id} by {agent_id}")
                
                return {"status": "completed", "result": task.result}
            else:
                # Task failed
                task.status = SwarmTaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error_message = "Simulated task failure"
                
                # Update agent
                agent.state = SwarmAgentState.IDLE
                agent.current_task = None
                agent.total_tasks_failed += 1
                
                # Check for retry
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = SwarmTaskStatus.RETRYING
                    task.assigned_agent = None
                    self.task_queue.appendleft(task_id)  # High priority for retry
                    logging.info(f"Task queued for retry: {task_id} (attempt {task.retry_count + 1})")
                else:
                    self.failed_tasks.append(task_id)
                    logging.error(f"Task failed permanently: {task_id}")
                
                return {"status": "failed", "error": task.error_message}
                
        except Exception as e:
            # Handle execution error
            task.status = SwarmTaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            agent.state = SwarmAgentState.IDLE
            agent.current_task = None
            agent.total_tasks_failed += 1
            
            logging.error(f"Task execution error: {task_id} - {e}")
            return {"status": "error", "error": str(e)}
    
    def start_swarm(self) -> bool:
        """Start swarm processing."""
        if self.running:
            return False
        
        self.running = True
        logging.info("Swarm orchestration started")
        
        # Start task processing in background
        self.executor.submit(self._process_tasks)
        return True
    
    def stop_swarm(self) -> bool:
        """Stop swarm processing."""
        if not self.running:
            return False
        
        self.running = False
        logging.info("Swarm orchestration stopped")
        return True
    
    def _process_tasks(self):
        """Background task processing."""
        while self.running:
            try:
                if not self.task_queue:
                    time.sleep(0.1)
                    continue
                
                task_id = self.task_queue.popleft()
                task = self.tasks.get(task_id)
                
                if not task or task.status not in [SwarmTaskStatus.PENDING, SwarmTaskStatus.RETRYING]:
                    continue
                
                # Find best agent
                best_agent = self.find_best_agent(task)
                if not best_agent:
                    # No available agents, put task back
                    self.task_queue.append(task_id)
                    time.sleep(0.5)
                    continue
                
                # Assign and execute task
                if self.assign_task(task_id, best_agent.agent_id):
                    # Execute in separate thread
                    self.executor.submit(self.execute_task, task_id)
                
            except Exception as e:
                logging.error(f"Error in task processing: {e}")
                time.sleep(1.0)
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status."""
        total_agents = len(self.agents)
        idle_agents = len([a for a in self.agents.values() if a.state == SwarmAgentState.IDLE])
        busy_agents = len([a for a in self.agents.values() if a.state == SwarmAgentState.BUSY])
        
        total_tasks = len(self.tasks)
        pending_tasks = len([t for t in self.tasks.values() if t.status == SwarmTaskStatus.PENDING])
        completed_tasks = len(self.completed_tasks)
        failed_tasks = len(self.failed_tasks)
        
        success_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        utilization = (busy_agents / total_agents * 100) if total_agents > 0 else 0
        
        return {
            "running": self.running,
            "agents": {
                "total": total_agents,
                "idle": idle_agents,
                "busy": busy_agents,
                "utilization": utilization
            },
            "tasks": {
                "total": total_tasks,
                "pending": pending_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks,
                "in_queue": len(self.task_queue),
                "success_rate": success_rate
            },
            "statistics": self.swarm_statistics
        }


__all__ = ['SwarmOrchestrationEngine']