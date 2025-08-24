"""
Orchestration Data Models
========================

Core data structures and enums for the orchestration system.
Extracted from unified_orchestrator.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
import uuid


# ============================================================================
# ENUMS
# ============================================================================

class GraphExecutionMode(Enum):
    """Graph execution modes for DAG workflows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class NodeState(Enum):
    """State of individual nodes in execution graph."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class SwarmTaskStatus(Enum):
    """Status of a swarm task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class SwarmAgentState(Enum):
    """State of a swarm agent."""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class SwarmArchitecture(Enum):
    """Swarm architecture patterns."""
    SEQUENTIAL_WORKFLOW = "sequential_workflow"
    CONCURRENT_WORKFLOW = "concurrent_workflow"
    HIERARCHICAL_SWARM = "hierarchical_swarm"
    AGENT_REARRANGE = "agent_rearrange"
    MIXTURE_OF_AGENTS = "mixture_of_agents"
    SWARM_ROUTER = "swarm_router"


class OrchestrationStrategy(Enum):
    """Unified orchestration strategies."""
    GRAPH_BASED = "graph_based"
    SWARM_BASED = "swarm_based"
    HYBRID_ORCHESTRATION = "hybrid_orchestration"
    INTELLIGENT_ROUTING = "intelligent_routing"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class GraphNode:
    """Represents a node in the execution graph."""
    node_id: str
    agent_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    state: NodeState = NodeState.PENDING
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "agent_type": self.agent_type,
            "config": self.config,
            "dependencies": self.dependencies,
            "outputs": self.outputs,
            "state": self.state.value,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "priority": self.priority
        }


@dataclass
class SwarmTask:
    """Task to be executed by the swarm."""
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    task_type: str = "test_execution"
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: Optional[int] = None
    max_retries: int = 3
    retry_count: int = 0
    status: SwarmTaskStatus = SwarmTaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "required_capabilities": self.required_capabilities,
            "estimated_duration": self.estimated_duration,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error_message": self.error_message
        }


@dataclass
class SwarmAgent:
    """Represents a swarm agent."""
    agent_id: str
    agent_type: str
    capabilities: List[str] = field(default_factory=list)
    state: SwarmAgentState = SwarmAgentState.IDLE
    current_task: Optional[str] = None
    performance_score: float = 1.0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_execution_time: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "state": self.state.value,
            "current_task": self.current_task,
            "performance_score": self.performance_score,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "average_execution_time": self.average_execution_time,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class OrchestrationConfiguration:
    """Unified orchestration configuration."""
    orchestration_id: str
    name: str
    strategy: OrchestrationStrategy = OrchestrationStrategy.HYBRID_ORCHESTRATION
    graph_config: Optional[Dict[str, Any]] = None
    swarm_config: Optional[Dict[str, Any]] = None
    routing_config: Optional[Dict[str, Any]] = None
    execution_mode: GraphExecutionMode = GraphExecutionMode.ADAPTIVE
    swarm_architecture: SwarmArchitecture = SwarmArchitecture.HIERARCHICAL_SWARM
    timeout_seconds: int = 3600
    max_concurrent_tasks: int = 10
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "orchestration_id": self.orchestration_id,
            "name": self.name,
            "strategy": self.strategy.value,
            "graph_config": self.graph_config,
            "swarm_config": self.swarm_config,
            "routing_config": self.routing_config,
            "execution_mode": self.execution_mode.value,
            "swarm_architecture": self.swarm_architecture.value,
            "timeout_seconds": self.timeout_seconds,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "retry_policy": self.retry_policy
        }


__all__ = [
    'GraphExecutionMode',
    'NodeState',
    'SwarmTaskStatus',
    'SwarmAgentState',
    'SwarmArchitecture',
    'OrchestrationStrategy',
    'GraphNode',
    'SwarmTask',
    'SwarmAgent',
    'OrchestrationConfiguration'
]