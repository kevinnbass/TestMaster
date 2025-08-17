"""
Agent Coordination for Multi-Agent TestMaster System

Coordinates multiple test generation agents, manages voting, and orchestrates
consensus-driven decision making across the TestMaster ecosystem.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
import time
import threading
from datetime import datetime, timedelta
import uuid

from .consensus_engine import ConsensusEngine, AgentVote, ConsensusResult, ConsensusStrategy, VotingMethod
from ...core.shared_state import get_shared_state


class AgentRole(Enum):
    """Roles for different types of agents."""
    TEST_GENERATOR = "test_generator"
    PLAN_EVALUATOR = "plan_evaluator"
    QUALITY_ASSESSOR = "quality_assessor"
    SECURITY_SCANNER = "security_scanner"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    INTEGRATION_COORDINATOR = "integration_coordinator"


class CoordinationStatus(Enum):
    """Status of coordination processes."""
    INITIALIZING = "initializing"
    COLLECTING_VOTES = "collecting_votes"
    REACHING_CONSENSUS = "reaching_consensus"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentInfo:
    """Information about an agent in the coordination system."""
    agent_id: str
    role: AgentRole
    weight: float = 1.0
    reliability_score: float = 1.0
    specialization: List[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    performance_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationConfig:
    """Configuration for agent coordination."""
    voting_timeout: float = 30.0  # seconds
    min_participants: int = 2
    consensus_threshold: float = 0.6
    max_coordination_rounds: int = 3
    enable_learning: bool = True
    weight_adjustment_factor: float = 0.1


@dataclass
class CoordinationTask:
    """A task requiring multi-agent coordination."""
    task_id: str
    description: str
    required_roles: Set[AgentRole]
    context: Dict[str, Any] = field(default_factory=dict)
    votes: List[AgentVote] = field(default_factory=list)
    status: CoordinationStatus = CoordinationStatus.INITIALIZING
    result: Optional[ConsensusResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentCoordinator:
    """Coordinates multiple agents for collaborative decision making."""
    
    def __init__(self, config: CoordinationConfig = None):
        self.config = config or CoordinationConfig()
        self.consensus_engine = ConsensusEngine(
            min_participants=self.config.min_participants,
            consensus_threshold=self.config.consensus_threshold
        )
        
        # Agent registry
        self.registered_agents: Dict[str, AgentInfo] = {}
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.completed_tasks: List[CoordinationTask] = []
        
        # Shared state integration
        self.shared_state = get_shared_state()
        
        # Threading
        self.coordination_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "tasks_coordinated": 0,
            "successful_consensus": 0,
            "failed_consensus": 0,
            "total_agents_coordinated": 0,
            "average_consensus_time": 0.0
        }
        
        print("Agent Coordinator initialized")
        print(f"   Voting timeout: {self.config.voting_timeout}s")
        print(f"   Min participants: {self.config.min_participants}")
        print(f"   Consensus threshold: {self.config.consensus_threshold}")
    
    def register_agent(self, agent_id: str, role: AgentRole, 
                      weight: float = 1.0, specialization: List[str] = None) -> bool:
        """Register an agent for coordination."""
        
        with self.coordination_lock:
            agent_info = AgentInfo(
                agent_id=agent_id,
                role=role,
                weight=weight,
                specialization=specialization or [],
                performance_history=[]
            )
            
            self.registered_agents[agent_id] = agent_info
            
            # Store in shared state for persistence
            self.shared_state.set(f"agent_registry_{agent_id}", {
                "role": role.value,
                "weight": weight,
                "specialization": specialization or [],
                "registered_at": datetime.now().isoformat()
            })
            
            print(f"Registered agent: {agent_id} ({role.value})")
            return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        
        with self.coordination_lock:
            if agent_id in self.registered_agents:
                del self.registered_agents[agent_id]
                self.shared_state.delete(f"agent_registry_{agent_id}")
                print(f"Unregistered agent: {agent_id}")
                return True
            return False
    
    def create_coordination_task(self, description: str, 
                               required_roles: Set[AgentRole],
                               context: Dict[str, Any] = None,
                               timeout: float = None) -> str:
        """Create a new coordination task."""
        
        task_id = str(uuid.uuid4())[:8]
        timeout = timeout or self.config.voting_timeout
        
        task = CoordinationTask(
            task_id=task_id,
            description=description,
            required_roles=required_roles,
            context=context or {},
            deadline=datetime.now() + timedelta(seconds=timeout)
        )
        
        with self.coordination_lock:
            self.active_tasks[task_id] = task
            self.stats["tasks_coordinated"] += 1
        
        print(f"Created coordination task: {task_id}")
        print(f"   Description: {description}")
        print(f"   Required roles: {[r.value for r in required_roles]}")
        
        return task_id
    
    def submit_vote(self, task_id: str, agent_id: str, choice: Any,
                   confidence: float = 1.0, reasoning: str = "") -> bool:
        """Submit a vote for a coordination task."""
        
        with self.coordination_lock:
            # Validate task exists
            if task_id not in self.active_tasks:
                print(f"Task {task_id} not found")
                return False
            
            task = self.active_tasks[task_id]
            
            # Check if task is still accepting votes
            if task.status != CoordinationStatus.COLLECTING_VOTES and task.status != CoordinationStatus.INITIALIZING:
                print(f"Task {task_id} not accepting votes (status: {task.status.value})")
                return False
            
            # Check deadline
            if task.deadline and datetime.now() > task.deadline:
                print(f"Task {task_id} voting deadline passed")
                task.status = CoordinationStatus.FAILED
                return False
            
            # Validate agent
            if agent_id not in self.registered_agents:
                print(f"Agent {agent_id} not registered")
                return False
            
            agent_info = self.registered_agents[agent_id]
            
            # Check if agent role is required for this task
            if agent_info.role not in task.required_roles:
                print(f"Agent {agent_id} role {agent_info.role.value} not required for this task")
                return False
            
            # Check if agent already voted
            existing_vote = next((v for v in task.votes if v.agent_id == agent_id), None)
            if existing_vote:
                # Update existing vote
                existing_vote.choice = choice
                existing_vote.confidence = confidence
                existing_vote.reasoning = reasoning
                existing_vote.timestamp = datetime.now()
                print(f"Updated vote from agent {agent_id}")
            else:
                # Create new vote
                vote = AgentVote(
                    agent_id=agent_id,
                    choice=choice,
                    confidence=confidence,
                    weight=agent_info.weight * agent_info.reliability_score,
                    reasoning=reasoning
                )
                task.votes.append(vote)
                print(f"Received vote from agent {agent_id}: {choice} (confidence: {confidence:.2f})")
            
            # Update task status
            if task.status == CoordinationStatus.INITIALIZING:
                task.status = CoordinationStatus.COLLECTING_VOTES
            
            # Check if we have enough votes to proceed
            self._check_and_process_consensus(task_id)
            
            return True
    
    def get_coordination_result(self, task_id: str) -> Optional[ConsensusResult]:
        """Get the result of a coordination task."""
        
        with self.coordination_lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.status == CoordinationStatus.COMPLETED:
                    return task.result
            
            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return task.result
            
            return None
    
    def _check_and_process_consensus(self, task_id: str):
        """Check if consensus can be reached and process if ready."""
        
        task = self.active_tasks[task_id]
        
        # Check if we have votes from all required roles
        voted_roles = {self.registered_agents[v.agent_id].role for v in task.votes}
        missing_roles = task.required_roles - voted_roles
        
        # Check if minimum participants met
        if len(task.votes) >= self.config.min_participants and not missing_roles:
            self._process_consensus(task_id)
        elif task.deadline and datetime.now() > task.deadline:
            # Deadline passed, try consensus with available votes
            if len(task.votes) >= self.config.min_participants:
                print(f"Deadline passed for task {task_id}, processing with {len(task.votes)} votes")
                self._process_consensus(task_id)
            else:
                print(f"Task {task_id} failed: insufficient votes by deadline")
                task.status = CoordinationStatus.FAILED
    
    def _process_consensus(self, task_id: str):
        """Process consensus for a task."""
        
        task = self.active_tasks[task_id]
        task.status = CoordinationStatus.REACHING_CONSENSUS
        
        print(f"\nProcessing consensus for task {task_id}")
        
        try:
            # Determine best consensus strategy based on votes
            strategy = self._select_consensus_strategy(task.votes, task.context)
            
            # Reach consensus
            consensus_result = self.consensus_engine.reach_consensus(
                task.votes,
                strategy=strategy,
                max_rounds=self.config.max_coordination_rounds
            )
            
            # Store result
            task.result = consensus_result
            task.status = CoordinationStatus.COMPLETED
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            
            # Update statistics
            self.stats["successful_consensus"] += 1
            self.stats["total_agents_coordinated"] += len(task.votes)
            
            # Update agent performance if learning enabled
            if self.config.enable_learning:
                self._update_agent_performance(task.votes, consensus_result)
            
            print(f"Consensus reached for task {task_id}: {consensus_result.decision}")
            
            # Store result in shared state
            self.shared_state.set(f"consensus_result_{task_id}", {
                "decision": consensus_result.decision,
                "confidence": consensus_result.confidence,
                "support_ratio": consensus_result.support_ratio,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Consensus failed for task {task_id}: {e}")
            task.status = CoordinationStatus.FAILED
            self.stats["failed_consensus"] += 1
    
    def _select_consensus_strategy(self, votes: List[AgentVote], context: Dict[str, Any]) -> ConsensusStrategy:
        """Select the best consensus strategy based on the situation."""
        
        # Check if all votes are numeric
        numeric_votes = 0
        for vote in votes:
            try:
                float(vote.choice)
                numeric_votes += 1
            except (ValueError, TypeError):
                pass
        
        # If mostly numeric, use weighted average
        if numeric_votes / len(votes) > 0.8:
            return ConsensusStrategy.WEIGHTED_AVERAGE
        
        # If high confidence votes, use first past post
        avg_confidence = sum(v.confidence for v in votes) / len(votes)
        if avg_confidence > 0.8:
            return ConsensusStrategy.FIRST_PAST_POST
        
        # If many agents, use Byzantine fault tolerant
        if len(votes) >= 4:
            return ConsensusStrategy.BYZANTINE_FAULT_TOLERANT
        
        # Default to weighted average
        return ConsensusStrategy.WEIGHTED_AVERAGE
    
    def _update_agent_performance(self, votes: List[AgentVote], result: ConsensusResult):
        """Update agent performance based on consensus result."""
        
        # Update reliability scores based on how agents voted
        for vote in votes:
            agent_info = self.registered_agents[vote.agent_id]
            
            # Calculate performance score (how well this vote aligned with consensus)
            if str(vote.choice) == str(result.decision):
                # Vote matched consensus
                performance_score = vote.confidence * result.confidence
            else:
                # Vote didn't match consensus
                performance_score = (1 - vote.confidence) * result.confidence
            
            # Update performance history
            agent_info.performance_history.append(performance_score)
            
            # Keep only recent history
            if len(agent_info.performance_history) > 10:
                agent_info.performance_history = agent_info.performance_history[-10:]
            
            # Update reliability score
            if len(agent_info.performance_history) >= 3:
                avg_performance = sum(agent_info.performance_history) / len(agent_info.performance_history)
                
                # Adjust reliability score
                adjustment = (avg_performance - 0.5) * self.config.weight_adjustment_factor
                agent_info.reliability_score = max(0.1, min(2.0, agent_info.reliability_score + adjustment))
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        with self.coordination_lock:
            active_tasks = len(self.active_tasks)
            completed_tasks = len(self.completed_tasks)
            registered_agents = len(self.registered_agents)
            
            # Calculate average consensus time
            if self.completed_tasks:
                total_time = sum((task.result.convergence_time for task in self.completed_tasks 
                                if task.result))
                avg_time = total_time / len([t for t in self.completed_tasks if t.result])
            else:
                avg_time = 0.0
            
            return {
                "active_tasks": active_tasks,
                "completed_tasks": completed_tasks,
                "registered_agents": registered_agents,
                "consensus_stats": self.stats,
                "average_consensus_time": avg_time,
                "agent_roles": {role.value: len([a for a in self.registered_agents.values() 
                                               if a.role == role]) 
                              for role in AgentRole},
                "success_rate": (self.stats["successful_consensus"] / 
                               max(self.stats["tasks_coordinated"], 1)) * 100
            }
    
    def cleanup_expired_tasks(self) -> int:
        """Clean up expired tasks."""
        current_time = datetime.now()
        expired_count = 0
        
        with self.coordination_lock:
            expired_tasks = []
            for task_id, task in self.active_tasks.items():
                if task.deadline and current_time > task.deadline:
                    task.status = CoordinationStatus.FAILED
                    expired_tasks.append(task_id)
                    expired_count += 1
            
            # Move expired tasks to completed
            for task_id in expired_tasks:
                task = self.active_tasks[task_id]
                self.completed_tasks.append(task)
                del self.active_tasks[task_id]
        
        if expired_count > 0:
            print(f"Cleaned up {expired_count} expired coordination tasks")
        
        return expired_count


def test_agent_coordination():
    """Test the agent coordination system."""
    print("\n" + "="*60)
    print("Testing Agent Coordination System")
    print("="*60)
    
    # Create coordinator
    config = CoordinationConfig(
        voting_timeout=5.0,
        min_participants=2,
        consensus_threshold=0.6
    )
    coordinator = AgentCoordinator(config)
    
    # Register test agents
    coordinator.register_agent("generator_1", AgentRole.TEST_GENERATOR, weight=1.0)
    coordinator.register_agent("generator_2", AgentRole.TEST_GENERATOR, weight=1.2)
    coordinator.register_agent("evaluator_1", AgentRole.PLAN_EVALUATOR, weight=1.1)
    coordinator.register_agent("quality_1", AgentRole.QUALITY_ASSESSOR, weight=0.9)
    
    # Create coordination task
    task_id = coordinator.create_coordination_task(
        description="Select best test generation strategy",
        required_roles={AgentRole.TEST_GENERATOR, AgentRole.PLAN_EVALUATOR},
        context={"module": "test_module.py", "complexity": "high"}
    )
    
    # Submit votes
    print("\n1. Submitting votes...")
    coordinator.submit_vote(task_id, "generator_1", "comprehensive", 0.9, "High complexity requires comprehensive testing")
    coordinator.submit_vote(task_id, "generator_2", "basic", 0.7, "Start with basic coverage")
    coordinator.submit_vote(task_id, "evaluator_1", "comprehensive", 0.85, "Comprehensive plan shows better coverage potential")
    
    # Wait for consensus
    time.sleep(1)
    
    # Get result
    result = coordinator.get_coordination_result(task_id)
    if result:
        print(f"\n2. Consensus Result:")
        print(f"   Decision: {result.decision}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Support: {result.support_ratio:.1%}")
    
    # Test numeric consensus
    task_id_2 = coordinator.create_coordination_task(
        description="Evaluate test quality score",
        required_roles={AgentRole.QUALITY_ASSESSOR, AgentRole.TEST_GENERATOR}
    )
    
    print("\n3. Testing numeric consensus...")
    coordinator.submit_vote(task_id_2, "generator_1", 0.85, 0.9, "Good coverage achieved")
    coordinator.submit_vote(task_id_2, "quality_1", 0.78, 0.8, "Some areas need improvement")
    
    time.sleep(1)
    
    result_2 = coordinator.get_coordination_result(task_id_2)
    if result_2:
        print(f"   Average Score: {result_2.decision:.3f}")
        print(f"   Confidence: {result_2.confidence:.2f}")
    
    # Get statistics
    stats = coordinator.get_coordination_stats()
    print(f"\n4. Coordination Statistics:")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Active tasks: {stats['active_tasks']}")
    print(f"   Completed tasks: {stats['completed_tasks']}")
    print(f"   Registered agents: {stats['registered_agents']}")
    
    print("\n✅ Agent Coordination test completed successfully!")
    return True


if __name__ == "__main__":
    test_agent_coordination()