"""
Swarm Patterns
=============

Swarm orchestration patterns for distributed agent coordination,
collective intelligence, and emergent behavior management.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from collections import defaultdict
import random
import math


class SwarmBehavior(Enum):
    """Swarm behavior patterns."""
    COLLECTIVE = "collective"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"
    HIERARCHICAL = "hierarchical"
    EMERGENT = "emergent"
    ADAPTIVE = "adaptive"


class AgentRole(Enum):
    """Agent roles in swarm."""
    LEADER = "leader"
    FOLLOWER = "follower"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    SCOUT = "scout"
    WORKER = "worker"


class CommunicationMode(Enum):
    """Communication modes between agents."""
    BROADCAST = "broadcast"
    UNICAST = "unicast"
    MULTICAST = "multicast"
    GOSSIP = "gossip"
    HIERARCHICAL = "hierarchical"


@dataclass
class SwarmAgent:
    """Agent in swarm orchestration."""
    agent_id: str
    name: str
    role: AgentRole
    capabilities: Set[str] = field(default_factory=set)
    position: Tuple[float, float] = (0.0, 0.0)  # Virtual position for spatial algorithms
    status: str = "active"
    load: float = 0.0
    performance_score: float = 1.0
    neighbors: Set[str] = field(default_factory=set)
    communication_range: float = 1.0
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmMessage:
    """Message between swarm agents."""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: str
    content: Any
    timestamp: datetime
    ttl: int = 5  # Time to live for message propagation
    priority: int = 0


class SwarmPattern(ABC):
    """Abstract base class for swarm patterns."""
    
    def __init__(self, pattern_name: str, behavior: SwarmBehavior = SwarmBehavior.COLLECTIVE):
        self.pattern_name = pattern_name
        self.behavior = behavior
        self.agents: Dict[str, SwarmAgent] = {}
        self.message_queue: List[SwarmMessage] = []
        self.communication_mode = CommunicationMode.BROADCAST
        self.coordination_rules: Dict[str, Callable] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    @abstractmethod
    async def execute_swarm(self, task: Any) -> Dict[str, Any]:
        """Execute task using swarm coordination."""
        pass
    
    @abstractmethod
    def coordinate_agents(self) -> Dict[str, Any]:
        """Coordinate agent behaviors."""
        pass
    
    def add_agent(self, agent: SwarmAgent):
        """Add agent to swarm."""
        self.agents[agent.agent_id] = agent
        self._emit_event("agent_joined", {"agent_id": agent.agent_id})
    
    def remove_agent(self, agent_id: str):
        """Remove agent from swarm."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self._emit_event("agent_left", {"agent_id": agent_id})
    
    def send_message(self, message: SwarmMessage):
        """Send message in swarm."""
        self.message_queue.append(message)
        self._emit_event("message_sent", {"message": message})
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit event to handlers."""
        for handler in self.event_handlers[event_type]:
            try:
                handler(event_type, event_data)
            except Exception:
                pass


class CoordinationPattern(SwarmPattern):
    """
    Coordination pattern for distributed task execution.
    
    Implements distributed coordination algorithms including:
    - Leader election
    - Consensus protocols
    - Load balancing
    - Task distribution
    """
    
    def __init__(self, coordination_name: str = "coordination_swarm"):
        super().__init__(coordination_name, SwarmBehavior.COLLABORATIVE)
        self.leader_id: Optional[str] = None
        self.task_queue: List[Any] = []
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.load_balancing_algorithm = "round_robin"
        self.consensus_threshold = 0.6
    
    async def execute_swarm(self, task: Any) -> Dict[str, Any]:
        """Execute task using coordinated swarm."""
        try:
            # Ensure we have a leader
            if not self.leader_id:
                await self._elect_leader()
            
            # Decompose task into subtasks
            subtasks = await self._decompose_task(task)
            
            # Distribute subtasks to agents
            assignments = await self._distribute_tasks(subtasks)
            
            # Execute tasks in parallel
            results = await self._execute_distributed_tasks(assignments)
            
            # Aggregate results
            final_result = await self._aggregate_results(results)
            
            return {
                "status": "completed",
                "result": final_result,
                "assignments": assignments,
                "execution_time": datetime.now()
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def coordinate_agents(self) -> Dict[str, Any]:
        """Coordinate agent behaviors."""
        coordination_status = {
            "leader_id": self.leader_id,
            "active_agents": len([a for a in self.agents.values() if a.status == "active"]),
            "load_distribution": {},
            "communication_efficiency": 0.0
        }
        
        # Calculate load distribution
        for agent_id, agent in self.agents.items():
            coordination_status["load_distribution"][agent_id] = agent.load
        
        # Update agent neighborhoods
        self._update_agent_neighborhoods()
        
        # Process pending messages
        self._process_messages()
        
        return coordination_status
    
    async def _elect_leader(self):
        """Elect leader using performance-based election."""
        if not self.agents:
            return
        
        # Simple election: highest performance score wins
        best_agent = max(
            self.agents.values(),
            key=lambda a: a.performance_score if a.status == "active" else 0
        )
        
        self.leader_id = best_agent.agent_id
        best_agent.role = AgentRole.LEADER
        
        # Notify agents of new leader
        leader_message = SwarmMessage(
            message_id=f"election_{datetime.now().timestamp()}",
            sender_id="system",
            recipient_id=None,  # Broadcast
            message_type="leader_elected",
            content={"leader_id": self.leader_id},
            timestamp=datetime.now()
        )
        self.send_message(leader_message)
        
        self._emit_event("leader_elected", {"leader_id": self.leader_id})
    
    async def _decompose_task(self, task: Any) -> List[Any]:
        """Decompose task into subtasks."""
        # Simple decomposition - would be more sophisticated in practice
        if hasattr(task, 'subtasks'):
            return task.subtasks
        elif hasattr(task, 'data') and isinstance(task.data, list):
            # Split data-based tasks
            chunk_size = max(1, len(task.data) // len(self.agents))
            return [
                {"type": "data_chunk", "data": task.data[i:i+chunk_size]}
                for i in range(0, len(task.data), chunk_size)
            ]
        else:
            # Single task
            return [task]
    
    async def _distribute_tasks(self, subtasks: List[Any]) -> Dict[str, List[Any]]:
        """Distribute tasks to agents using load balancing."""
        assignments = defaultdict(list)
        
        if self.load_balancing_algorithm == "round_robin":
            active_agents = [a for a in self.agents.values() if a.status == "active"]
            for i, subtask in enumerate(subtasks):
                agent = active_agents[i % len(active_agents)]
                assignments[agent.agent_id].append(subtask)
                
        elif self.load_balancing_algorithm == "least_loaded":
            for subtask in subtasks:
                # Find least loaded agent
                agent = min(
                    [a for a in self.agents.values() if a.status == "active"],
                    key=lambda a: a.load
                )
                assignments[agent.agent_id].append(subtask)
                agent.load += 0.1  # Increment load estimate
        
        return dict(assignments)
    
    async def _execute_distributed_tasks(self, assignments: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Execute tasks distributed across agents."""
        execution_tasks = []
        
        for agent_id, subtasks in assignments.items():
            task = asyncio.create_task(
                self._execute_agent_tasks(agent_id, subtasks)
            )
            execution_tasks.append((agent_id, task))
        
        results = {}
        for agent_id, task in execution_tasks:
            try:
                result = await task
                results[agent_id] = result
            except Exception as e:
                results[agent_id] = {"status": "failed", "error": str(e)}
        
        return results
    
    async def _execute_agent_tasks(self, agent_id: str, subtasks: List[Any]) -> Any:
        """Execute tasks for specific agent."""
        # Simulate task execution
        await asyncio.sleep(0.1 * len(subtasks))
        return {
            "agent_id": agent_id,
            "tasks_completed": len(subtasks),
            "results": [f"Result for task {i}" for i in range(len(subtasks))]
        }
    
    async def _aggregate_results(self, results: Dict[str, Any]) -> Any:
        """Aggregate results from all agents."""
        successful_results = [r for r in results.values() if r.get("status") != "failed"]
        total_tasks = sum(r.get("tasks_completed", 0) for r in successful_results)
        
        return {
            "total_tasks_completed": total_tasks,
            "participating_agents": len(successful_results),
            "aggregated_data": [r.get("results", []) for r in successful_results]
        }
    
    def _update_agent_neighborhoods(self):
        """Update agent neighborhoods for communication."""
        for agent_id, agent in self.agents.items():
            agent.neighbors.clear()
            
            for other_id, other_agent in self.agents.items():
                if agent_id != other_id and other_agent.status == "active":
                    # Calculate distance (for spatial algorithms)
                    distance = math.sqrt(
                        (agent.position[0] - other_agent.position[0]) ** 2 +
                        (agent.position[1] - other_agent.position[1]) ** 2
                    )
                    
                    if distance <= agent.communication_range:
                        agent.neighbors.add(other_id)
    
    def _process_messages(self):
        """Process pending messages in swarm."""
        processed_messages = []
        
        for message in self.message_queue:
            if self._should_process_message(message):
                self._handle_message(message)
                processed_messages.append(message)
        
        # Remove processed messages
        for message in processed_messages:
            self.message_queue.remove(message)
    
    def _should_process_message(self, message: SwarmMessage) -> bool:
        """Check if message should be processed."""
        return (datetime.now() - message.timestamp).seconds < message.ttl * 60
    
    def _handle_message(self, message: SwarmMessage):
        """Handle specific message."""
        if message.message_type == "leader_elected":
            # Update local knowledge of leader
            self.leader_id = message.content.get("leader_id")
        elif message.message_type == "task_assignment":
            # Handle task assignment
            pass
        elif message.message_type == "status_update":
            # Update agent status
            agent_id = message.sender_id
            if agent_id in self.agents:
                self.agents[agent_id].status = message.content.get("status", "active")


class EmergentPattern(SwarmPattern):
    """
    Emergent behavior pattern for self-organizing swarms.
    
    Implements emergent behavior algorithms including:
    - Flocking/herding behaviors
    - Ant colony optimization
    - Particle swarm optimization
    - Self-organization patterns
    """
    
    def __init__(self, emergent_name: str = "emergent_swarm"):
        super().__init__(emergent_name, SwarmBehavior.EMERGENT)
        self.pheromone_trails: Dict[str, float] = {}
        self.global_best_solution: Optional[Any] = None
        self.global_best_fitness: float = float('inf')
        self.optimization_target: Optional[Callable] = None
    
    async def execute_swarm(self, task: Any) -> Dict[str, Any]:
        """Execute task using emergent swarm behavior."""
        try:
            # Set optimization target if provided
            if hasattr(task, 'fitness_function'):
                self.optimization_target = task.fitness_function
            
            # Initialize swarm positions and velocities
            self._initialize_swarm_state()
            
            # Run swarm optimization iterations
            iterations = getattr(task, 'max_iterations', 100)
            for iteration in range(iterations):
                await self._swarm_iteration(iteration)
                
                # Check convergence
                if self._check_convergence():
                    break
            
            return {
                "status": "completed",
                "best_solution": self.global_best_solution,
                "best_fitness": self.global_best_fitness,
                "iterations": iteration + 1
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def coordinate_agents(self) -> Dict[str, Any]:
        """Coordinate emergent agent behaviors."""
        coordination_info = {
            "swarm_cohesion": self._calculate_cohesion(),
            "swarm_alignment": self._calculate_alignment(),
            "swarm_separation": self._calculate_separation(),
            "exploration_exploitation": self._calculate_exploration_ratio()
        }
        
        # Update pheromone trails (decay)
        self._update_pheromone_trails()
        
        # Apply flocking behaviors
        self._apply_flocking_behaviors()
        
        return coordination_info
    
    def _initialize_swarm_state(self):
        """Initialize swarm state for optimization."""
        for agent in self.agents.values():
            # Random initial position
            agent.position = (random.uniform(-10, 10), random.uniform(-10, 10))
            
            # Initialize agent-specific optimization state
            agent.metadata.update({
                'velocity': (random.uniform(-1, 1), random.uniform(-1, 1)),
                'personal_best_position': agent.position,
                'personal_best_fitness': float('inf'),
                'exploration_factor': random.uniform(0.1, 0.9)
            })
    
    async def _swarm_iteration(self, iteration: int):
        """Execute one iteration of swarm optimization."""
        for agent in self.agents.values():
            if agent.status != "active":
                continue
            
            # Evaluate current position
            if self.optimization_target:
                fitness = self.optimization_target(agent.position)
                
                # Update personal best
                if fitness < agent.metadata['personal_best_fitness']:
                    agent.metadata['personal_best_fitness'] = fitness
                    agent.metadata['personal_best_position'] = agent.position
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = agent.position
            
            # Update agent velocity and position
            self._update_agent_movement(agent, iteration)
        
        # Update pheromone trails based on solutions
        self._deposit_pheromones()
    
    def _update_agent_movement(self, agent: SwarmAgent, iteration: int):
        """Update agent movement using PSO-like algorithm."""
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Current velocity
        vx, vy = agent.metadata['velocity']
        
        # Personal best influence
        pbx, pby = agent.metadata['personal_best_position']
        personal_force_x = c1 * random.random() * (pbx - agent.position[0])
        personal_force_y = c1 * random.random() * (pby - agent.position[1])
        
        # Global best influence
        if self.global_best_solution:
            gbx, gby = self.global_best_solution
            social_force_x = c2 * random.random() * (gbx - agent.position[0])
            social_force_y = c2 * random.random() * (gby - agent.position[1])
        else:
            social_force_x = social_force_y = 0
        
        # Update velocity
        new_vx = w * vx + personal_force_x + social_force_x
        new_vy = w * vy + personal_force_y + social_force_y
        
        # Limit velocity
        max_velocity = 2.0
        new_vx = max(-max_velocity, min(max_velocity, new_vx))
        new_vy = max(-max_velocity, min(max_velocity, new_vy))
        
        agent.metadata['velocity'] = (new_vx, new_vy)
        
        # Update position
        new_x = agent.position[0] + new_vx
        new_y = agent.position[1] + new_vy
        
        # Apply boundaries
        new_x = max(-10, min(10, new_x))
        new_y = max(-10, min(10, new_y))
        
        agent.position = (new_x, new_y)
    
    def _calculate_cohesion(self) -> float:
        """Calculate swarm cohesion metric."""
        if len(self.agents) < 2:
            return 1.0
        
        center_x = sum(a.position[0] for a in self.agents.values()) / len(self.agents)
        center_y = sum(a.position[1] for a in self.agents.values()) / len(self.agents)
        
        total_distance = sum(
            math.sqrt((a.position[0] - center_x)**2 + (a.position[1] - center_y)**2)
            for a in self.agents.values()
        )
        
        return 1.0 / (1.0 + total_distance / len(self.agents))
    
    def _calculate_alignment(self) -> float:
        """Calculate swarm alignment metric."""
        if len(self.agents) < 2:
            return 1.0
        
        velocities = [a.metadata.get('velocity', (0, 0)) for a in self.agents.values()]
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        # Calculate alignment as similarity to average velocity
        alignment_sum = 0
        for vx, vy in velocities:
            dot_product = vx * avg_vx + vy * avg_vy
            magnitude = math.sqrt(vx**2 + vy**2) * math.sqrt(avg_vx**2 + avg_vy**2)
            if magnitude > 0:
                alignment_sum += dot_product / magnitude
        
        return alignment_sum / len(velocities) if velocities else 0
    
    def _calculate_separation(self) -> float:
        """Calculate swarm separation metric."""
        if len(self.agents) < 2:
            return 1.0
        
        min_distance = float('inf')
        for agent1 in self.agents.values():
            for agent2 in self.agents.values():
                if agent1.agent_id != agent2.agent_id:
                    distance = math.sqrt(
                        (agent1.position[0] - agent2.position[0])**2 +
                        (agent1.position[1] - agent2.position[1])**2
                    )
                    min_distance = min(min_distance, distance)
        
        return min_distance / 10.0  # Normalize
    
    def _calculate_exploration_ratio(self) -> float:
        """Calculate exploration vs exploitation ratio."""
        exploration_agents = sum(
            1 for a in self.agents.values()
            if a.metadata.get('exploration_factor', 0.5) > 0.5
        )
        return exploration_agents / len(self.agents) if self.agents else 0
    
    def _update_pheromone_trails(self):
        """Update pheromone trails with decay."""
        decay_rate = 0.9
        for trail_id in list(self.pheromone_trails.keys()):
            self.pheromone_trails[trail_id] *= decay_rate
            if self.pheromone_trails[trail_id] < 0.01:
                del self.pheromone_trails[trail_id]
    
    def _deposit_pheromones(self):
        """Deposit pheromones based on solution quality."""
        for agent in self.agents.values():
            if agent.status == "active":
                # Create trail identifier from position
                trail_id = f"{int(agent.position[0]*10)}_{int(agent.position[1]*10)}"
                
                # Deposit pheromone based on fitness (lower fitness = more pheromone)
                fitness = agent.metadata.get('personal_best_fitness', float('inf'))
                if fitness != float('inf'):
                    pheromone_amount = 1.0 / (1.0 + fitness)
                    self.pheromone_trails[trail_id] = (
                        self.pheromone_trails.get(trail_id, 0) + pheromone_amount
                    )
    
    def _apply_flocking_behaviors(self):
        """Apply flocking behaviors to agents."""
        for agent in self.agents.values():
            if agent.status != "active":
                continue
            
            # Find neighbors
            neighbors = [
                other for other in self.agents.values()
                if other.agent_id != agent.agent_id and
                math.sqrt(
                    (agent.position[0] - other.position[0])**2 +
                    (agent.position[1] - other.position[1])**2
                ) <= agent.communication_range
            ]
            
            if neighbors:
                # Apply separation, alignment, and cohesion forces
                sep_force = self._calculate_separation_force(agent, neighbors)
                align_force = self._calculate_alignment_force(agent, neighbors)
                coh_force = self._calculate_cohesion_force(agent, neighbors)
                
                # Combine forces
                total_force_x = sep_force[0] + align_force[0] + coh_force[0]
                total_force_y = sep_force[1] + align_force[1] + coh_force[1]
                
                # Apply to velocity (small increment)
                vx, vy = agent.metadata.get('velocity', (0, 0))
                agent.metadata['velocity'] = (
                    vx + total_force_x * 0.1,
                    vy + total_force_y * 0.1
                )
    
    def _calculate_separation_force(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Tuple[float, float]:
        """Calculate separation force to avoid crowding."""
        force_x = force_y = 0
        
        for neighbor in neighbors:
            dx = agent.position[0] - neighbor.position[0]
            dy = agent.position[1] - neighbor.position[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0 and distance < 2.0:  # Separation range
                force_x += dx / distance
                force_y += dy / distance
        
        return (force_x, force_y)
    
    def _calculate_alignment_force(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Tuple[float, float]:
        """Calculate alignment force to match neighbor velocities."""
        if not neighbors:
            return (0, 0)
        
        avg_vx = sum(n.metadata.get('velocity', (0, 0))[0] for n in neighbors) / len(neighbors)
        avg_vy = sum(n.metadata.get('velocity', (0, 0))[1] for n in neighbors) / len(neighbors)
        
        vx, vy = agent.metadata.get('velocity', (0, 0))
        return (avg_vx - vx, avg_vy - vy)
    
    def _calculate_cohesion_force(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Tuple[float, float]:
        """Calculate cohesion force to move toward neighbor center."""
        if not neighbors:
            return (0, 0)
        
        center_x = sum(n.position[0] for n in neighbors) / len(neighbors)
        center_y = sum(n.position[1] for n in neighbors) / len(neighbors)
        
        return (center_x - agent.position[0], center_y - agent.position[1])
    
    def _check_convergence(self) -> bool:
        """Check if swarm has converged."""
        if len(self.agents) < 2:
            return True
        
        # Check if all agents are close to global best
        if not self.global_best_solution:
            return False
        
        convergence_threshold = 0.1
        converged_agents = 0
        
        for agent in self.agents.values():
            distance = math.sqrt(
                (agent.position[0] - self.global_best_solution[0])**2 +
                (agent.position[1] - self.global_best_solution[1])**2
            )
            if distance < convergence_threshold:
                converged_agents += 1
        
        return converged_agents / len(self.agents) > 0.8


# Export key classes
__all__ = [
    'SwarmBehavior',
    'AgentRole',
    'CommunicationMode',
    'SwarmAgent',
    'SwarmMessage',
    'SwarmPattern',
    'CoordinationPattern',
    'EmergentPattern'
]