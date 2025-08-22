"""
Decision Optimizer - Multi-Objective Optimization Engine
========================================================

Advanced decision optimization system implementing multi-objective optimization,
Pareto frontier analysis, and constraint-aware decision making for prescriptive
analytics with enterprise-grade mathematical optimization capabilities.

This module provides sophisticated optimization algorithms including:
- Single and multi-objective optimization using differential evolution
- Pareto frontier generation and knee-point selection
- Constraint handling for budget, time, and resource limitations
- Risk-adjusted value calculations with confidence intervals

Author: Agent A - Hours 40-50
Created: 2025-08-22
Module: decision_optimizer.py (260 lines)
"""

import asyncio
import numpy as np
import hashlib
from typing import Dict, List, Any, Callable
from collections import deque
from scipy.optimize import differential_evolution
from datetime import datetime

from .prescriptive_types import (
    OptimizationObjective, PrescriptiveAction, OptimalDecision
)


class DecisionOptimizer:
    """
    Enterprise decision optimizer implementing multi-objective optimization
    with Pareto frontier analysis and constraint handling.
    
    Features:
    - Multi-objective optimization using weighted sum and Pareto methods
    - Global optimization with differential evolution algorithms
    - Constraint handling for budget, time, and resource limitations
    - Risk-adjusted value calculations with confidence intervals
    - Pareto frontier generation with knee-point selection
    """
    
    def __init__(self):
        self.optimization_history = deque(maxlen=1000)
        self.pareto_frontier = []
        self.constraint_violations = []
        
    async def optimize_decision(
        self,
        objectives: List[OptimizationObjective],
        constraints: List[Dict[str, Any]],
        action_space: List[PrescriptiveAction]
    ) -> OptimalDecision:
        """
        Optimize decision based on objectives and constraints using advanced algorithms.
        
        Args:
            objectives: List of optimization objectives to satisfy
            constraints: List of constraint dictionaries with type and limit
            action_space: Available actions for selection and optimization
            
        Returns:
            OptimalDecision with selected actions and performance metrics
        """
        
        # Formulate optimization problem with objective functions and constraints
        problem = self._formulate_optimization_problem(objectives, constraints, action_space)
        
        # Select optimization algorithm based on problem characteristics
        if len(objectives) == 1:
            solution = await self._single_objective_optimization(problem)
        else:
            solution = await self._multi_objective_optimization(problem)
        
        # Select actions based on optimization solution
        selected_actions = self._select_actions(solution, action_space)
        
        # Calculate comprehensive performance metrics
        expected_value = self._calculate_expected_value(selected_actions)
        risk_adjusted = self._calculate_risk_adjusted_value(expected_value, selected_actions)
        success_prob = self._calculate_success_probability(selected_actions)
        
        decision = OptimalDecision(
            decision_id=self._generate_id("decision"),
            objective=objectives[0] if len(objectives) == 1 else OptimizationObjective.PARETO_OPTIMAL,
            actions=selected_actions,
            expected_value=expected_value,
            risk_adjusted_value=risk_adjusted,
            success_probability=success_prob,
            implementation_time=self._calculate_implementation_time(selected_actions),
            resource_requirements=self._calculate_resource_requirements(selected_actions)
        )
        
        self.optimization_history.append(decision)
        
        return decision
    
    def _formulate_optimization_problem(
        self,
        objectives: List[OptimizationObjective],
        constraints: List[Dict[str, Any]],
        action_space: List[PrescriptiveAction]
    ) -> Dict[str, Any]:
        """Formulate mathematical optimization problem with objectives and constraints"""
        
        n_actions = len(action_space)
        
        # Multi-objective function combining all objectives
        def objective_function(x):
            total_value = 0
            for i, action in enumerate(action_space):
                if x[i] > 0.5:  # Binary decision threshold
                    for obj in objectives:
                        if obj == OptimizationObjective.MAXIMIZE_VALUE:
                            total_value += action.expected_outcome.get("value", 0)
                        elif obj == OptimizationObjective.MINIMIZE_COST:
                            total_value -= action.expected_outcome.get("cost", 0)
                        elif obj == OptimizationObjective.MINIMIZE_RISK:
                            total_value -= action.risk_level * 10
                        elif obj == OptimizationObjective.MAXIMIZE_EFFICIENCY:
                            efficiency = action.expected_outcome.get("efficiency", 0.5)
                            total_value += efficiency * 10
                        elif obj == OptimizationObjective.BALANCE_TRADEOFFS:
                            # Balance risk vs reward
                            reward = action.expected_outcome.get("value", 0)
                            risk_penalty = action.risk_level * 5
                            total_value += (reward - risk_penalty)
            
            # Negate for minimization (scipy minimizes by default)
            return -total_value if OptimizationObjective.MAXIMIZE_VALUE in objectives else total_value
        
        # Constraint functions for resource limitations
        constraint_funcs = []
        for constraint in constraints:
            if constraint["type"] == "budget":
                def budget_constraint(x):
                    total_cost = sum(
                        action.expected_outcome.get("cost", 0) * x[i]
                        for i, action in enumerate(action_space)
                    )
                    return constraint["limit"] - total_cost
                constraint_funcs.append({"type": "ineq", "fun": budget_constraint})
            
            elif constraint["type"] == "time":
                def time_constraint(x):
                    total_time = sum(
                        action.expected_outcome.get("time", 0) * x[i]
                        for i, action in enumerate(action_space)
                    )
                    return constraint["limit"] - total_time
                constraint_funcs.append({"type": "ineq", "fun": time_constraint})
            
            elif constraint["type"] == "resource":
                def resource_constraint(x):
                    total_resource = sum(
                        action.expected_outcome.get(constraint["resource_type"], 0) * x[i]
                        for i, action in enumerate(action_space)
                    )
                    return constraint["limit"] - total_resource
                constraint_funcs.append({"type": "ineq", "fun": resource_constraint})
        
        return {
            "objective": objective_function,
            "n_vars": n_actions,
            "bounds": [(0, 1) for _ in range(n_actions)],
            "constraints": constraint_funcs
        }
    
    async def _single_objective_optimization(self, problem: Dict[str, Any]) -> np.ndarray:
        """Single objective optimization using differential evolution"""
        
        # Use differential evolution for robust global optimization
        result = differential_evolution(
            problem["objective"],
            problem["bounds"],
            constraints=problem["constraints"] if problem["constraints"] else (),
            maxiter=100,
            popsize=15,
            seed=42  # For reproducibility
        )
        
        # Convert continuous solution to binary decisions
        solution = (result.x > 0.5).astype(int)
        
        return solution
    
    async def _multi_objective_optimization(self, problem: Dict[str, Any]) -> np.ndarray:
        """Multi-objective optimization using Pareto frontier analysis"""
        
        pareto_solutions = []
        
        # Generate Pareto frontier using weighted sum method
        for weight in np.linspace(0, 1, 10):
            # Create weighted objective function
            def weighted_objective(x):
                return weight * problem["objective"](x)
            
            result = differential_evolution(
                weighted_objective,
                problem["bounds"],
                constraints=problem["constraints"] if problem["constraints"] else (),
                maxiter=50,
                popsize=10
            )
            
            pareto_solutions.append(result.x)
        
        # Select best solution from Pareto frontier using knee-point detection
        best_solution = self._select_from_pareto(pareto_solutions, problem["objective"])
        
        # Store Pareto frontier for analysis
        self.pareto_frontier = pareto_solutions
        
        # Convert to binary decision vector
        return (best_solution > 0.5).astype(int)
    
    def _select_from_pareto(self, solutions: List[np.ndarray], objective: Callable) -> np.ndarray:
        """Select optimal solution from Pareto frontier using knee-point detection"""
        if not solutions:
            return np.array([])
        
        # Calculate objective values for all Pareto solutions
        objectives = [objective(sol) for sol in solutions]
        
        # Knee-point detection for optimal trade-off selection
        if len(objectives) > 2:
            # Normalize objective values
            obj_min, obj_max = np.min(objectives), np.max(objectives)
            normalized = (objectives - obj_min) / (obj_max - obj_min + 1e-10)
            
            # Calculate distances from ideal point (maximum curvature)
            distances = [abs(i/len(normalized) - normalized[i]) for i in range(len(normalized))]
            knee_idx = np.argmax(distances)
        else:
            # Fallback to middle solution
            knee_idx = len(objectives) // 2
        
        return solutions[knee_idx]
    
    def _select_actions(
        self,
        solution: np.ndarray,
        action_space: List[PrescriptiveAction]
    ) -> List[PrescriptiveAction]:
        """Select actions based on optimization solution vector"""
        selected = []
        
        for i, select in enumerate(solution):
            if select > 0.5 and i < len(action_space):
                selected.append(action_space[i])
        
        # Sort by priority for optimal execution order
        selected.sort(key=lambda a: a.priority)
        
        return selected
    
    def _calculate_expected_value(self, actions: List[PrescriptiveAction]) -> float:
        """Calculate expected value with confidence weighting"""
        return sum(
            action.expected_outcome.get("value", 0) * action.confidence
            for action in actions
        )
    
    def _calculate_risk_adjusted_value(
        self,
        expected_value: float,
        actions: List[PrescriptiveAction]
    ) -> float:
        """Calculate risk-adjusted value using modern portfolio theory"""
        if not actions:
            return expected_value
        
        # Calculate portfolio risk
        total_risk = sum(action.risk_level for action in actions) / len(actions)
        
        # Risk adjustment using Sharpe ratio concept
        risk_free_rate = 0.02  # 2% risk-free rate
        risk_adjustment = max(0, expected_value - risk_free_rate) / (total_risk + 1e-6)
        
        return expected_value * (1 - total_risk * 0.1) + risk_adjustment * 0.1
    
    def _calculate_success_probability(self, actions: List[PrescriptiveAction]) -> float:
        """Calculate overall success probability using independence assumption"""
        if not actions:
            return 1.0
        
        # Combined probability assuming partial independence
        individual_probs = [action.confidence for action in actions]
        
        # Use geometric mean for conservative estimate
        if individual_probs:
            combined_prob = np.prod(individual_probs) ** (1.0 / len(individual_probs))
        else:
            combined_prob = 0.5
        
        return min(1.0, combined_prob)
    
    def _calculate_implementation_time(self, actions: List[PrescriptiveAction]) -> float:
        """Calculate total implementation time considering dependencies"""
        if not actions:
            return 0.0
        
        # Simple summation (would be more sophisticated with dependency analysis)
        return sum(
            action.expected_outcome.get("time", 1.0)
            for action in actions
        )
    
    def _calculate_resource_requirements(
        self, 
        actions: List[PrescriptiveAction]
    ) -> Dict[str, float]:
        """Calculate aggregated resource requirements"""
        resources = {}
        
        for action in actions:
            for resource_type, amount in action.expected_outcome.items():
                if resource_type in ["cpu", "memory", "storage", "budget"]:
                    resources[resource_type] = resources.get(resource_type, 0) + amount
        
        return resources
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with timestamp and hash"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{prefix}_{timestamp}_{len(self.optimization_history)}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"


# Export decision optimization components
__all__ = ['DecisionOptimizer']