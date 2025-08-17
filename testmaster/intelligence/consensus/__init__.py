"""
Consensus Mechanisms for Multi-Agent Coordination

Implements consensus algorithms for coordinating multiple test generation agents,
plan evaluation, and result aggregation across the TestMaster system.
"""

from .consensus_engine import (
    ConsensusEngine,
    ConsensusStrategy,
    VotingMethod,
    ConsensusResult
)

from .agent_coordination import (
    AgentCoordinator,
    AgentVote,
    CoordinationConfig
)

__all__ = [
    'ConsensusEngine',
    'ConsensusStrategy',
    'VotingMethod', 
    'ConsensusResult',
    'AgentCoordinator',
    'AgentVote',
    'CoordinationConfig'
]