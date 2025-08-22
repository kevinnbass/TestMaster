"""
Prescriptive Types and Data Structures
======================================

Core type definitions and data structures for the Prescriptive Intelligence Engine.
Provides enterprise-grade type safety for decision optimization, strategy generation,
and outcome maximization systems.

This module contains all Enum definitions and dataclass structures used throughout
the prescriptive analytics system, implementing advanced decision science patterns.

Author: Agent A - Hours 40-50
Created: 2025-08-22
Module: prescriptive_types.py (120 lines)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ActionType(Enum):
    """Types of prescriptive actions for decision optimization"""
    IMMEDIATE = "immediate"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    OPTIMIZATION = "optimization"
    STRATEGIC = "strategic"


class OptimizationObjective(Enum):
    """Optimization objectives for multi-criteria decision making"""
    MAXIMIZE_VALUE = "maximize_value"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_TRADEOFFS = "balance_tradeoffs"
    PARETO_OPTIMAL = "pareto_optimal"
    NASH_EQUILIBRIUM = "nash_equilibrium"
    ROBUST_OPTIMIZATION = "robust_optimization"


class StrategyType(Enum):
    """Types of strategic approaches for decision implementation"""
    GREEDY = "greedy"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    MINIMAX = "minimax"
    MONTE_CARLO = "monte_carlo"
    GAME_THEORETIC = "game_theoretic"


@dataclass
class PrescriptiveAction:
    """Represents a prescriptive action with comprehensive metadata"""
    action_id: str
    action_type: ActionType
    description: str
    parameters: Dict[str, Any]
    expected_outcome: Dict[str, float]
    confidence: float
    risk_level: float
    priority: int
    dependencies: List[str]
    constraints: List[Dict[str, Any]]
    timestamp: datetime


@dataclass
class OptimalDecision:
    """Represents an optimal decision with risk-adjusted metrics"""
    decision_id: str
    objective: OptimizationObjective
    actions: List[PrescriptiveAction]
    expected_value: float
    risk_adjusted_value: float
    success_probability: float
    implementation_time: float
    resource_requirements: Dict[str, float]


@dataclass
class Strategy:
    """Represents a strategic plan with game-theoretic components"""
    strategy_id: str
    strategy_type: StrategyType
    objectives: List[OptimizationObjective]
    action_sequence: List[PrescriptiveAction]
    decision_tree: nx.DiGraph
    payoff_matrix: np.ndarray
    optimal_path: List[str]
    expected_utility: float


@dataclass
class Outcome:
    """Represents a predicted outcome with probabilistic metrics"""
    outcome_id: str
    scenario: str
    probability: float
    value: float
    risk: float
    time_to_outcome: float
    contributing_actions: List[str]


# Export all prescriptive types
__all__ = [
    'ActionType',
    'OptimizationObjective',
    'StrategyType',
    'PrescriptiveAction',
    'OptimalDecision',
    'Strategy',
    'Outcome'
]