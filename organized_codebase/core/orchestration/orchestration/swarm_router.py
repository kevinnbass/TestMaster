"""
Swarm Router Module
==================

Intelligent routing system for dynamic swarm organization.
Extracted from unified_orchestrator.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from .data_models import SwarmTask, SwarmAgent


class SwarmRouter:
    """Intelligent routing system for dynamic swarm organization."""
    
    def __init__(self):
        self.routing_strategies: Dict[str, Callable] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.routing_decisions: List[Dict[str, Any]] = []
        
    def register_routing_strategy(self, strategy_name: str, strategy_func: Callable):
        """Register custom routing strategy."""
        self.routing_strategies[strategy_name] = strategy_func
        logging.info(f"Routing strategy registered: {strategy_name}")
    
    def route_task(self, task: SwarmTask, available_agents: List[SwarmAgent], 
                  strategy: str = "performance_based") -> Optional[SwarmAgent]:
        """Route task to best agent using specified strategy."""
        if strategy in self.routing_strategies:
            return self.routing_strategies[strategy](task, available_agents)
        
        # Default performance-based routing
        return self._performance_based_routing(task, available_agents)
    
    def _performance_based_routing(self, task: SwarmTask, agents: List[SwarmAgent]) -> Optional[SwarmAgent]:
        """Route based on agent performance and capability match."""
        if not agents:
            return None
        
        scored_agents = []
        
        for agent in agents:
            score = 0.0
            
            # Historical performance
            if agent.agent_id in self.performance_history:
                recent_performance = self.performance_history[agent.agent_id][-10:]  # Last 10 tasks
                avg_performance = sum(recent_performance) / len(recent_performance)
                score += avg_performance * 0.4
            else:
                score += agent.performance_score * 0.4
            
            # Capability matching
            if task.required_capabilities:
                capability_score = len(set(task.required_capabilities) & set(agent.capabilities))
                capability_score /= len(task.required_capabilities)
                score += capability_score * 0.3
            else:
                score += 0.3
            
            # Agent specialization
            if agent.agent_type == task.task_type:
                score += 0.2
            
            # Load balancing
            if agent.current_task is None:
                score += 0.1
            
            scored_agents.append((agent, score))
        
        # Return best agent
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        selected_agent = scored_agents[0][0]
        
        # Record routing decision
        self.routing_decisions.append({
            "timestamp": datetime.now().isoformat(),
            "task_id": task.task_id,
            "selected_agent": selected_agent.agent_id,
            "strategy": "performance_based",
            "score": scored_agents[0][1]
        })
        
        return selected_agent
    
    def update_performance(self, agent_id: str, task_id: str, performance_score: float):
        """Update agent performance based on task completion."""
        self.performance_history[agent_id].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[agent_id]) > 50:
            self.performance_history[agent_id] = self.performance_history[agent_id][-50:]
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get routing analytics and insights."""
        total_decisions = len(self.routing_decisions)
        
        if total_decisions == 0:
            return {"total_decisions": 0, "analytics": "No routing decisions recorded"}
        
        # Agent selection frequency
        agent_frequency = defaultdict(int)
        for decision in self.routing_decisions:
            agent_frequency[decision["selected_agent"]] += 1
        
        # Strategy usage
        strategy_usage = defaultdict(int)
        for decision in self.routing_decisions:
            strategy_usage[decision["strategy"]] += 1
        
        return {
            "total_decisions": total_decisions,
            "agent_selection_frequency": dict(agent_frequency),
            "strategy_usage": dict(strategy_usage),
            "recent_decisions": self.routing_decisions[-10:],
            "performance_tracking": {
                "agents_tracked": len(self.performance_history),
                "total_performance_records": sum(len(history) for history in self.performance_history.values())
            }
        }


__all__ = ['SwarmRouter']