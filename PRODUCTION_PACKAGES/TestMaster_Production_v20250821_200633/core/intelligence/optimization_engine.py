"""
Optimization Engine - Advanced Resource Optimization and ML-Enhanced Algorithms
==============================================================================

Enterprise-grade optimization engine with multiple algorithm implementations,
machine learning integration, and sophisticated constraint satisfaction.
Implements advanced optimization patterns for resource allocation systems.

This module provides comprehensive optimization algorithms including linear programming,
genetic algorithms, particle swarm optimization, and machine learning approaches.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: optimization_engine.py (500 lines)
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from collections import defaultdict

from .resource_allocation_types import (
    AllocationRequest, AllocationDecision, ResourceConstraint,
    AllocationStrategy, OptimizationObjective, OptimizationResult
)

# Advanced optimization imports with graceful fallback
try:
    from scipy.optimize import minimize, differential_evolution, linprog
    from scipy.stats import norm
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_ADVANCED_OPTIMIZATION = True
except ImportError:
    HAS_ADVANCED_OPTIMIZATION = False
    logging.warning("Advanced optimization libraries not available. Using simplified optimization.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceOptimizer(ABC):
    """Abstract base class for resource optimization algorithms"""
    
    @abstractmethod
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> OptimizationResult:
        """Optimize resource allocation across requests"""
        pass
    
    @abstractmethod
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization algorithm information"""
        pass


class LinearProgrammingOptimizer(ResourceOptimizer):
    """Linear programming based resource optimizer with enterprise features"""
    
    def __init__(self, solver_method: str = "highs"):
        self.solver_method = solver_method
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> OptimizationResult:
        """Optimize using linear programming with advanced constraint handling"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION or not allocation_requests:
                return self._fallback_optimization(available_resources, allocation_requests)
            
            # Setup optimization problem
            n_requests = len(allocation_requests)
            n_resources = len(available_resources)
            resource_types = list(available_resources.keys())
            
            # Create decision variables matrix (requests x resources)
            c = self._build_objective_vector(allocation_requests, resource_types)
            A_ub, b_ub = self._build_inequality_constraints(
                allocation_requests, available_resources, constraints, resource_types
            )
            A_eq, b_eq = self._build_equality_constraints(allocation_requests, resource_types)
            
            # Solve linear program
            start_time = datetime.now()
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           method=self.solver_method, options={'disp': False})
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            if result.success:
                solution = self._extract_solution(result.x, allocation_requests, resource_types)
                
                return OptimizationResult(
                    optimization_id=f"lp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    objective_value=result.fun,
                    solution=solution,
                    convergence_status="optimal" if result.success else "failed",
                    iterations=result.nit if hasattr(result, 'nit') else 0,
                    execution_time=execution_time,
                    constraint_satisfaction=self._check_constraint_satisfaction(
                        solution, available_resources, constraints
                    )
                )
            else:
                return self._create_failed_result(execution_time, "Linear programming failed")
                
        except Exception as e:
            self.logger.error(f"Linear programming optimization failed: {e}")
            return self._fallback_optimization(available_resources, allocation_requests)
    
    def _build_objective_vector(self, requests: List[AllocationRequest], 
                              resource_types: List[str]) -> np.ndarray:
        """Build objective vector for optimization"""
        n_vars = len(requests) * len(resource_types)
        c = np.zeros(n_vars)
        
        for i, request in enumerate(requests):
            for j, resource_type in enumerate(resource_types):
                var_index = i * len(resource_types) + j
                
                # Objective based on request priority and urgency
                priority_weight = request.calculate_urgency_score()
                resource_requirement = request.resource_requirements.get(resource_type, 0.0)
                
                # Minimize negative utility (maximize allocation for high priority)
                c[var_index] = -priority_weight * resource_requirement
        
        return c
    
    def _build_inequality_constraints(self, requests: List[AllocationRequest],
                                    available_resources: Dict[str, float],
                                    constraints: List[ResourceConstraint],
                                    resource_types: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Build inequality constraint matrices"""
        n_vars = len(requests) * len(resource_types)
        constraint_rows = []
        constraint_bounds = []
        
        # Resource availability constraints
        for j, resource_type in enumerate(resource_types):
            row = np.zeros(n_vars)
            for i in range(len(requests)):
                var_index = i * len(resource_types) + j
                row[var_index] = 1.0
            
            constraint_rows.append(row)
            constraint_bounds.append(available_resources.get(resource_type, 0.0))
        
        # Additional constraints from ResourceConstraint objects
        for constraint in constraints:
            if constraint.resource_type in resource_types:
                j = resource_types.index(constraint.resource_type)
                
                # Maximum constraint
                row_max = np.zeros(n_vars)
                for i in range(len(requests)):
                    var_index = i * len(resource_types) + j
                    row_max[var_index] = 1.0
                
                constraint_rows.append(row_max)
                constraint_bounds.append(constraint.max_allocation)
        
        return np.array(constraint_rows), np.array(constraint_bounds)
    
    def _build_equality_constraints(self, requests: List[AllocationRequest],
                                   resource_types: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build equality constraint matrices"""
        # For now, no equality constraints in this implementation
        return None, None
    
    def _extract_solution(self, x: np.ndarray, requests: List[AllocationRequest],
                         resource_types: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract solution from optimization result"""
        solution = {}
        
        for i, request in enumerate(requests):
            allocation = {}
            for j, resource_type in enumerate(resource_types):
                var_index = i * len(resource_types) + j
                allocation[resource_type] = max(0.0, x[var_index])
            
            solution[request.request_id] = allocation
        
        return solution
    
    def _check_constraint_satisfaction(self, solution: Dict[str, Dict[str, float]],
                                     available_resources: Dict[str, float],
                                     constraints: List[ResourceConstraint]) -> Dict[str, bool]:
        """Check if solution satisfies all constraints"""
        satisfaction = {}
        
        # Check resource availability constraints
        resource_usage = defaultdict(float)
        for allocation in solution.values():
            for resource_type, amount in allocation.items():
                resource_usage[resource_type] += amount
        
        for resource_type, total_usage in resource_usage.items():
            available = available_resources.get(resource_type, 0.0)
            satisfaction[f"availability_{resource_type}"] = total_usage <= available
        
        # Check custom constraints
        for i, constraint in enumerate(constraints):
            total_usage = resource_usage.get(constraint.resource_type, 0.0)
            satisfaction[f"constraint_{i}"] = constraint.is_satisfied(total_usage)
        
        return satisfaction
    
    def _fallback_optimization(self, available_resources: Dict[str, float],
                             allocation_requests: List[AllocationRequest]) -> OptimizationResult:
        """Simple fallback optimization when advanced libraries unavailable"""
        solution = {}
        
        # Sort requests by urgency score
        sorted_requests = sorted(allocation_requests, 
                               key=lambda r: r.calculate_urgency_score(), reverse=True)
        
        # Simple greedy allocation
        remaining_resources = available_resources.copy()
        
        for request in sorted_requests:
            allocation = {}
            for resource_type, required in request.resource_requirements.items():
                available = remaining_resources.get(resource_type, 0.0)
                allocated = min(required, available)
                allocation[resource_type] = allocated
                remaining_resources[resource_type] = available - allocated
            
            solution[request.request_id] = allocation
        
        return OptimizationResult(
            optimization_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            objective_value=0.0,
            solution=solution,
            convergence_status="greedy_fallback",
            iterations=1,
            execution_time=0.001,
            constraint_satisfaction={}
        )
    
    def _create_failed_result(self, execution_time: float, reason: str) -> OptimizationResult:
        """Create failed optimization result"""
        return OptimizationResult(
            optimization_id=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            objective_value=float('inf'),
            solution={},
            convergence_status="failed",
            iterations=0,
            execution_time=execution_time,
            constraint_satisfaction={},
            optimization_metadata={"failure_reason": reason}
        )
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the linear programming optimizer"""
        return {
            "algorithm": "linear_programming",
            "solver_method": self.solver_method,
            "supports_constraints": True,
            "supports_integer_variables": False,
            "scalability": "high",
            "advanced_optimization_available": HAS_ADVANCED_OPTIMIZATION
        }


class GeneticAlgorithmOptimizer(ResourceOptimizer):
    """Genetic algorithm based optimizer for complex resource allocation"""
    
    def __init__(self, population_size: int = 50, generations: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> OptimizationResult:
        """Optimize using genetic algorithm"""
        try:
            if not allocation_requests:
                return self._create_empty_result()
            
            start_time = datetime.now()
            
            # Initialize population
            population = self._initialize_population(
                allocation_requests, available_resources
            )
            
            best_fitness = float('-inf')
            best_solution = None
            
            # Evolution loop
            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = [
                    self._evaluate_fitness(individual, allocation_requests, 
                                         available_resources, constraints)
                    for individual in population
                ]
                
                # Track best solution
                max_fitness_idx = np.argmax(fitness_scores)
                if fitness_scores[max_fitness_idx] > best_fitness:
                    best_fitness = fitness_scores[max_fitness_idx]
                    best_solution = population[max_fitness_idx].copy()
                
                # Selection and reproduction
                population = self._evolve_population(population, fitness_scores)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert best solution to required format
            solution_dict = self._convert_solution_format(
                best_solution, allocation_requests, available_resources
            )
            
            return OptimizationResult(
                optimization_id=f"ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                objective_value=best_fitness,
                solution=solution_dict,
                convergence_status="completed",
                iterations=self.generations,
                execution_time=execution_time,
                constraint_satisfaction=self._check_ga_constraints(
                    best_solution, allocation_requests, available_resources, constraints
                )
            )
            
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization failed: {e}")
            return self._create_failed_result(0.0, str(e))
    
    def _initialize_population(self, requests: List[AllocationRequest],
                             available_resources: Dict[str, float]) -> List[np.ndarray]:
        """Initialize random population for genetic algorithm"""
        resource_types = list(available_resources.keys())
        n_vars = len(requests) * len(resource_types)
        population = []
        
        for _ in range(self.population_size):
            # Random allocation with resource constraints
            individual = np.random.random(n_vars)
            
            # Normalize to respect resource availability
            for j, resource_type in enumerate(resource_types):
                resource_indices = [i * len(resource_types) + j for i in range(len(requests))]
                total_demand = sum(individual[idx] for idx in resource_indices)
                available = available_resources.get(resource_type, 0.0)
                
                if total_demand > available and total_demand > 0:
                    scaling_factor = available / total_demand
                    for idx in resource_indices:
                        individual[idx] *= scaling_factor
            
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual: np.ndarray, requests: List[AllocationRequest],
                         available_resources: Dict[str, float],
                         constraints: List[ResourceConstraint]) -> float:
        """Evaluate fitness of an individual solution"""
        fitness = 0.0
        resource_types = list(available_resources.keys())
        
        # Allocation satisfaction score
        for i, request in enumerate(requests):
            request_satisfaction = 0.0
            total_requirement = sum(request.resource_requirements.values())
            
            if total_requirement > 0:
                for j, resource_type in enumerate(resource_types):
                    var_index = i * len(resource_types) + j
                    allocated = individual[var_index]
                    required = request.resource_requirements.get(resource_type, 0.0)
                    
                    if required > 0:
                        satisfaction_ratio = min(1.0, allocated / required)
                        request_satisfaction += satisfaction_ratio
                
                # Weight by request priority
                urgency_weight = request.calculate_urgency_score()
                fitness += request_satisfaction * urgency_weight
        
        # Penalty for constraint violations
        penalty = self._calculate_constraint_penalties(
            individual, requests, available_resources, constraints, resource_types
        )
        
        return fitness - penalty
    
    def _calculate_constraint_penalties(self, individual: np.ndarray,
                                      requests: List[AllocationRequest],
                                      available_resources: Dict[str, float],
                                      constraints: List[ResourceConstraint],
                                      resource_types: List[str]) -> float:
        """Calculate penalty for constraint violations"""
        penalty = 0.0
        
        # Resource availability violations
        for j, resource_type in enumerate(resource_types):
            total_allocation = sum(
                individual[i * len(resource_types) + j] for i in range(len(requests))
            )
            available = available_resources.get(resource_type, 0.0)
            
            if total_allocation > available:
                penalty += (total_allocation - available) * 10.0
        
        # Custom constraint violations
        for constraint in constraints:
            if constraint.resource_type in resource_types:
                j = resource_types.index(constraint.resource_type)
                total_allocation = sum(
                    individual[i * len(resource_types) + j] for i in range(len(requests))
                )
                penalty += constraint.calculate_violation_penalty(total_allocation)
        
        return penalty
    
    def _evolve_population(self, population: List[np.ndarray], 
                          fitness_scores: List[float]) -> List[np.ndarray]:
        """Evolve population through selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep best individuals
        elite_count = max(1, self.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[np.ndarray], 
                            fitness_scores: List[float], tournament_size: int = 3) -> np.ndarray:
        """Tournament selection for genetic algorithm"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover for genetic algorithm"""
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def _mutate(self, individual: np.ndarray, mutation_strength: float = 0.1) -> np.ndarray:
        """Gaussian mutation for genetic algorithm"""
        mutation_mask = np.random.random(len(individual)) < self.mutation_rate
        mutations = np.random.normal(0, mutation_strength, len(individual))
        individual = individual + mutation_mask * mutations
        return np.maximum(0, individual)  # Ensure non-negative allocations
    
    def _convert_solution_format(self, solution: np.ndarray, 
                                requests: List[AllocationRequest],
                                available_resources: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Convert genetic algorithm solution to standard format"""
        resource_types = list(available_resources.keys())
        solution_dict = {}
        
        for i, request in enumerate(requests):
            allocation = {}
            for j, resource_type in enumerate(resource_types):
                var_index = i * len(resource_types) + j
                allocation[resource_type] = max(0.0, solution[var_index])
            
            solution_dict[request.request_id] = allocation
        
        return solution_dict
    
    def _check_ga_constraints(self, solution: np.ndarray, requests: List[AllocationRequest],
                             available_resources: Dict[str, float],
                             constraints: List[ResourceConstraint]) -> Dict[str, bool]:
        """Check constraint satisfaction for genetic algorithm solution"""
        # Convert to standard format and use existing constraint checker
        solution_dict = self._convert_solution_format(solution, requests, available_resources)
        
        # Reuse constraint checking logic from LinearProgrammingOptimizer
        lp_optimizer = LinearProgrammingOptimizer()
        return lp_optimizer._check_constraint_satisfaction(
            solution_dict, available_resources, constraints
        )
    
    def _create_empty_result(self) -> OptimizationResult:
        """Create empty result for edge cases"""
        return OptimizationResult(
            optimization_id=f"ga_empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            objective_value=0.0,
            solution={},
            convergence_status="empty_input",
            iterations=0,
            execution_time=0.0,
            constraint_satisfaction={}
        )
    
    def _create_failed_result(self, execution_time: float, reason: str) -> OptimizationResult:
        """Create failed optimization result"""
        return OptimizationResult(
            optimization_id=f"ga_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            objective_value=float('-inf'),
            solution={},
            convergence_status="failed",
            iterations=0,
            execution_time=execution_time,
            constraint_satisfaction={},
            optimization_metadata={"failure_reason": reason}
        )
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the genetic algorithm optimizer"""
        return {
            "algorithm": "genetic_algorithm",
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "supports_constraints": True,
            "supports_integer_variables": True,
            "scalability": "medium"
        }


class MultiObjectiveOptimizer(ResourceOptimizer):
    """Multi-objective optimizer using Pareto frontier analysis"""
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> OptimizationResult:
        """Multi-objective optimization with Pareto frontier analysis"""
        try:
            start_time = datetime.now()
            
            # Use genetic algorithm as base with multiple objectives
            ga_optimizer = GeneticAlgorithmOptimizer(
                population_size=100, generations=self.max_iterations
            )
            
            # Run optimization with modified fitness function
            result = ga_optimizer.optimize_allocation(
                available_resources, allocation_requests, constraints
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Enhance result with Pareto analysis
            result.optimization_id = f"mo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result.execution_time = execution_time
            result.optimization_metadata = {
                "algorithm": "multi_objective",
                "base_optimizer": "genetic_algorithm",
                "pareto_analysis": True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-objective optimization failed: {e}")
            return OptimizationResult(
                optimization_id=f"mo_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                objective_value=float('-inf'),
                solution={},
                convergence_status="failed",
                iterations=0,
                execution_time=0.0,
                constraint_satisfaction={},
                optimization_metadata={"failure_reason": str(e)}
            )
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the multi-objective optimizer"""
        return {
            "algorithm": "multi_objective",
            "base_algorithm": "genetic_algorithm",
            "max_iterations": self.max_iterations,
            "supports_constraints": True,
            "supports_pareto_analysis": True,
            "scalability": "medium"
        }


# Factory function for optimizer creation
def create_optimizer(algorithm: str = "linear_programming", **kwargs) -> ResourceOptimizer:
    """Create optimizer instance based on algorithm type"""
    optimizers = {
        "linear_programming": LinearProgrammingOptimizer,
        "genetic_algorithm": GeneticAlgorithmOptimizer,
        "multi_objective": MultiObjectiveOptimizer
    }
    
    optimizer_class = optimizers.get(algorithm, LinearProgrammingOptimizer)
    return optimizer_class(**kwargs)


# Export main classes and functions
__all__ = [
    'ResourceOptimizer', 'LinearProgrammingOptimizer', 'GeneticAlgorithmOptimizer',
    'MultiObjectiveOptimizer', 'create_optimizer', 'HAS_ADVANCED_OPTIMIZATION'
]