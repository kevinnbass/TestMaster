"""
Testing Supervisor
==================

Intelligent supervisor for coordinating multi-agent testing activities.
Inspired by LangGraph-Supervisor patterns with TestMaster specialization.

Author: TestMaster Team
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from core.observability import global_observability

class SupervisorMode(Enum):
    """Supervision modes for different testing scenarios"""
    AUTONOMOUS = "autonomous"      # Minimal supervision, agents self-coordinate
    GUIDED = "guided"             # Moderate supervision with guidance
    DIRECTED = "directed"         # High supervision with specific directions
    COLLABORATIVE = "collaborative"  # Democratic decision-making
    HIERARCHICAL = "hierarchical"   # Strict hierarchical control

class DecisionType(Enum):
    """Types of supervisor decisions"""
    TASK_ASSIGNMENT = "task_assignment"
    PRIORITY_ADJUSTMENT = "priority_adjustment"
    RESOURCE_ALLOCATION = "resource_allocation"
    WORKFLOW_MODIFICATION = "workflow_modification"
    QUALITY_GATE = "quality_gate"
    ESCALATION = "escalation"

@dataclass
class DecisionCriteria:
    """Criteria for supervisor decision-making"""
    performance_threshold: float = 80.0
    quality_threshold: float = 75.0
    time_threshold: float = 300.0  # 5 minutes
    error_threshold: int = 3
    collaboration_score: float = 70.0
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "quality": 0.4,
        "performance": 0.3,
        "timeliness": 0.2,
        "collaboration": 0.1
    })

@dataclass
class SupervisorDecision:
    """Represents a supervisor decision"""
    id: str = field(default_factory=lambda: f"decision_{uuid.uuid4().hex[:12]}")
    decision_type: DecisionType = DecisionType.TASK_ASSIGNMENT
    target_agents: List[str] = field(default_factory=list)
    action_required: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False
    result: Optional[Dict[str, Any]] = None

class TestingSupervisor:
    """
    Intelligent supervisor for coordinating multi-agent testing activities.
    
    Provides:
    - Task assignment and prioritization
    - Performance monitoring and optimization
    - Quality gate enforcement
    - Resource allocation decisions
    - Conflict resolution
    - Workflow adaptation
    """
    
    def __init__(
        self,
        mode: SupervisorMode = SupervisorMode.GUIDED,
        criteria: Optional[DecisionCriteria] = None
    ):
        self.mode = mode
        self.criteria = criteria or DecisionCriteria()
        
        # State management
        self.supervisor_id = f"supervisor_{uuid.uuid4().hex[:12]}"
        self.active = False
        self.session_id = None
        
        # Agent management
        self.managed_agents: Dict[str, Any] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        self.agent_workload: Dict[str, int] = {}
        
        # Decision tracking
        self.decisions: List[SupervisorDecision] = []
        self.pending_decisions: List[SupervisorDecision] = []
        
        # Workflow state
        self.current_workflow: Optional[Dict[str, Any]] = None
        self.workflow_metrics: Dict[str, Any] = {}
        
        # Performance tracking
        self.supervision_metrics = {
            "decisions_made": 0,
            "successful_decisions": 0,
            "agent_coordination_score": 0.0,
            "workflow_efficiency": 0.0,
            "quality_score": 0.0,
            "total_supervision_time": 0.0
        }
        
        # Logging
        self.logger = logging.getLogger(f'TestingSupervisor.{self.supervisor_id}')
        
        self.logger.info(f"Testing Supervisor initialized in {mode.value} mode")
    
    async def start_supervision(
        self, 
        agents: Dict[str, Any],
        workflow: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ):
        """Start supervising a group of agents"""
        self.active = True
        self.session_id = session_id
        self.managed_agents = agents
        self.current_workflow = workflow
        
        # Initialize agent tracking
        for agent_name, agent in agents.items():
            self.agent_performance[agent_name] = {
                "success_rate": 100.0,
                "average_response_time": 0.0,
                "quality_score": 100.0,
                "collaboration_score": 100.0
            }
            self.agent_workload[agent_name] = 0
        
        if self.session_id:
            global_observability.track_agent_action(
                self.session_id,
                "TestingSupervisor",
                "supervision_started",
                {
                    "mode": self.mode.value,
                    "agent_count": len(agents),
                    "workflow_defined": workflow is not None
                }
            )
        
        self.logger.info(f"Started supervision of {len(agents)} agents")
        
        # Start supervision loop
        asyncio.create_task(self._supervision_loop())
    
    async def stop_supervision(self):
        """Stop supervision and generate final report"""
        self.active = False
        
        # Generate final metrics
        final_metrics = await self._generate_final_metrics()
        
        if self.session_id:
            global_observability.track_agent_action(
                self.session_id,
                "TestingSupervisor",
                "supervision_ended",
                {
                    "final_metrics": final_metrics,
                    "decisions_made": len(self.decisions)
                }
            )
        
        self.logger.info("Supervision ended")
        return final_metrics
    
    async def _supervision_loop(self):
        """Main supervision loop"""
        while self.active:
            try:
                # Monitor agent performance
                await self._monitor_agents()
                
                # Make decisions based on current state
                decisions = await self._make_decisions()
                
                # Execute approved decisions
                for decision in decisions:
                    await self._execute_decision(decision)
                
                # Update workflow if needed
                await self._update_workflow()
                
                # Wait before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in supervision loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _monitor_agents(self):
        """Monitor agent performance and status"""
        for agent_name, agent in self.managed_agents.items():
            try:
                # Get agent status
                status = agent.get_status() if hasattr(agent, 'get_status') else {}
                
                # Update performance metrics
                if "performance_metrics" in status:
                    metrics = status["performance_metrics"]
                    self.agent_performance[agent_name].update({
                        "success_rate": metrics.get("success_rate", 100.0),
                        "average_response_time": metrics.get("average_action_time", 0.0)
                    })
                
                # Update workload
                self.agent_workload[agent_name] = status.get("active_actions", 0)
                
            except Exception as e:
                self.logger.warning(f"Failed to monitor agent {agent_name}: {e}")
    
    async def _make_decisions(self) -> List[SupervisorDecision]:
        """Make supervision decisions based on current state"""
        decisions = []
        
        # Check for performance issues
        performance_decisions = await self._check_performance_issues()
        decisions.extend(performance_decisions)
        
        # Check for workload imbalances
        workload_decisions = await self._check_workload_balance()
        decisions.extend(workload_decisions)
        
        # Check quality gates
        quality_decisions = await self._check_quality_gates()
        decisions.extend(quality_decisions)
        
        # Check for collaboration issues
        collaboration_decisions = await self._check_collaboration()
        decisions.extend(collaboration_decisions)
        
        return decisions
    
    async def _check_performance_issues(self) -> List[SupervisorDecision]:
        """Check for agent performance issues"""
        decisions = []
        
        for agent_name, performance in self.agent_performance.items():
            if performance["success_rate"] < self.criteria.performance_threshold:
                decision = SupervisorDecision(
                    decision_type=DecisionType.PRIORITY_ADJUSTMENT,
                    target_agents=[agent_name],
                    action_required="reduce_workload",
                    reasoning=f"Agent {agent_name} success rate ({performance['success_rate']:.1f}%) below threshold",
                    confidence=0.8
                )
                decisions.append(decision)
            
            if performance["average_response_time"] > self.criteria.time_threshold:
                decision = SupervisorDecision(
                    decision_type=DecisionType.RESOURCE_ALLOCATION,
                    target_agents=[agent_name],
                    action_required="optimize_performance",
                    reasoning=f"Agent {agent_name} response time ({performance['average_response_time']:.1f}s) too high",
                    confidence=0.7
                )
                decisions.append(decision)
        
        return decisions
    
    async def _check_workload_balance(self) -> List[SupervisorDecision]:
        """Check for workload imbalance between agents"""
        decisions = []
        
        if not self.agent_workload:
            return decisions
        
        workloads = list(self.agent_workload.values())
        avg_workload = sum(workloads) / len(workloads)
        max_workload = max(workloads)
        min_workload = min(workloads)
        
        # If imbalance is significant
        if max_workload - min_workload > 2 and avg_workload > 1:
            overloaded = [name for name, load in self.agent_workload.items() if load > avg_workload + 1]
            underloaded = [name for name, load in self.agent_workload.items() if load < avg_workload - 1]
            
            if overloaded and underloaded:
                decision = SupervisorDecision(
                    decision_type=DecisionType.TASK_ASSIGNMENT,
                    target_agents=overloaded + underloaded,
                    action_required="rebalance_workload",
                    reasoning=f"Workload imbalance detected: {max_workload} vs {min_workload}",
                    confidence=0.9,
                    metadata={
                        "overloaded_agents": overloaded,
                        "underloaded_agents": underloaded
                    }
                )
                decisions.append(decision)
        
        return decisions
    
    async def _check_quality_gates(self) -> List[SupervisorDecision]:
        """Check quality gates and standards"""
        decisions = []
        
        # Calculate overall quality score
        quality_scores = [
            perf.get("quality_score", 100.0)
            for perf in self.agent_performance.values()
        ]
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            if avg_quality < self.criteria.quality_threshold:
                decision = SupervisorDecision(
                    decision_type=DecisionType.QUALITY_GATE,
                    target_agents=list(self.managed_agents.keys()),
                    action_required="improve_quality",
                    reasoning=f"Overall quality score ({avg_quality:.1f}%) below threshold",
                    confidence=0.9,
                    metadata={"quality_score": avg_quality}
                )
                decisions.append(decision)
        
        return decisions
    
    async def _check_collaboration(self) -> List[SupervisorDecision]:
        """Check for collaboration issues between agents"""
        decisions = []
        
        # This is a simplified check - in a real implementation,
        # you'd analyze message patterns, response times, etc.
        collaboration_scores = [
            perf.get("collaboration_score", 100.0)
            for perf in self.agent_performance.values()
        ]
        
        if collaboration_scores:
            avg_collaboration = sum(collaboration_scores) / len(collaboration_scores)
            
            if avg_collaboration < self.criteria.collaboration_score:
                decision = SupervisorDecision(
                    decision_type=DecisionType.WORKFLOW_MODIFICATION,
                    target_agents=list(self.managed_agents.keys()),
                    action_required="improve_collaboration",
                    reasoning=f"Collaboration score ({avg_collaboration:.1f}%) needs improvement",
                    confidence=0.6
                )
                decisions.append(decision)
        
        return decisions
    
    async def _execute_decision(self, decision: SupervisorDecision):
        """Execute a supervisor decision"""
        try:
            if decision.action_required == "reduce_workload":
                await self._reduce_agent_workload(decision.target_agents)
            
            elif decision.action_required == "optimize_performance":
                await self._optimize_agent_performance(decision.target_agents)
            
            elif decision.action_required == "rebalance_workload":
                await self._rebalance_workload(decision)
            
            elif decision.action_required == "improve_quality":
                await self._improve_quality(decision.target_agents)
            
            elif decision.action_required == "improve_collaboration":
                await self._improve_collaboration(decision.target_agents)
            
            decision.executed = True
            decision.result = {"status": "success"}
            self.supervision_metrics["successful_decisions"] += 1
            
        except Exception as e:
            decision.executed = True
            decision.result = {"status": "failed", "error": str(e)}
            self.logger.error(f"Failed to execute decision {decision.id}: {e}")
        
        self.decisions.append(decision)
        self.supervision_metrics["decisions_made"] += 1
        
        self.logger.info(f"Executed decision: {decision.action_required} for agents {decision.target_agents}")
    
    async def _reduce_agent_workload(self, agent_names: List[str]):
        """Reduce workload for specified agents"""
        for agent_name in agent_names:
            if agent_name in self.managed_agents:
                # In a real implementation, you'd pause task assignment
                # or redistribute tasks to other agents
                self.logger.info(f"Reducing workload for agent {agent_name}")
    
    async def _optimize_agent_performance(self, agent_names: List[str]):
        """Optimize performance for specified agents"""
        for agent_name in agent_names:
            if agent_name in self.managed_agents:
                # In a real implementation, you'd apply performance optimizations
                self.logger.info(f"Optimizing performance for agent {agent_name}")
    
    async def _rebalance_workload(self, decision: SupervisorDecision):
        """Rebalance workload between agents"""
        metadata = decision.metadata
        overloaded = metadata.get("overloaded_agents", [])
        underloaded = metadata.get("underloaded_agents", [])
        
        self.logger.info(f"Rebalancing workload: {overloaded} -> {underloaded}")
    
    async def _improve_quality(self, agent_names: List[str]):
        """Improve quality for specified agents"""
        for agent_name in agent_names:
            if agent_name in self.managed_agents:
                # In a real implementation, you'd adjust quality parameters
                self.logger.info(f"Improving quality for agent {agent_name}")
    
    async def _improve_collaboration(self, agent_names: List[str]):
        """Improve collaboration between agents"""
        self.logger.info(f"Improving collaboration for agents: {agent_names}")
    
    async def _update_workflow(self):
        """Update workflow based on current conditions"""
        if not self.current_workflow:
            return
        
        # Calculate workflow efficiency
        total_agents = len(self.managed_agents)
        active_agents = sum(1 for load in self.agent_workload.values() if load > 0)
        
        if total_agents > 0:
            efficiency = active_agents / total_agents
            self.workflow_metrics["efficiency"] = efficiency
            self.supervision_metrics["workflow_efficiency"] = efficiency
    
    async def _generate_final_metrics(self) -> Dict[str, Any]:
        """Generate final supervision metrics"""
        # Calculate overall scores
        if self.agent_performance:
            avg_success_rate = sum(
                perf["success_rate"] for perf in self.agent_performance.values()
            ) / len(self.agent_performance)
            
            avg_quality = sum(
                perf.get("quality_score", 100.0) for perf in self.agent_performance.values()
            ) / len(self.agent_performance)
            
            self.supervision_metrics["agent_coordination_score"] = avg_success_rate
            self.supervision_metrics["quality_score"] = avg_quality
        
        # Calculate decision success rate
        if self.supervision_metrics["decisions_made"] > 0:
            success_rate = (
                self.supervision_metrics["successful_decisions"] / 
                self.supervision_metrics["decisions_made"] * 100
            )
            self.supervision_metrics["decision_success_rate"] = success_rate
        
        return {
            "supervisor_id": self.supervisor_id,
            "mode": self.mode.value,
            "supervision_metrics": self.supervision_metrics,
            "agent_performance": self.agent_performance,
            "decisions_made": len(self.decisions),
            "workflow_metrics": self.workflow_metrics
        }
    
    def get_supervision_status(self) -> Dict[str, Any]:
        """Get current supervision status"""
        return {
            "supervisor_id": self.supervisor_id,
            "mode": self.mode.value,
            "active": self.active,
            "managed_agents": len(self.managed_agents),
            "pending_decisions": len(self.pending_decisions),
            "total_decisions": len(self.decisions),
            "current_metrics": self.supervision_metrics,
            "agent_workload": self.agent_workload
        }

# Export components
__all__ = [
    'TestingSupervisor',
    'SupervisorMode',
    'SupervisorDecision',
    'DecisionCriteria',
    'DecisionType'
]