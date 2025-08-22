"""
Resource Intelligence Optimizers
===============================

Advanced optimization algorithms including Linear Programming and Genetic Algorithms.
Extracted from intelligent_resource_allocator.py for enterprise modular architecture.

Agent D Implementation - Hour 10-11: Revolutionary Intelligence Modularization
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any

from .data_models import AllocationRequest, ResourceConstraint

# Advanced optimization imports
try:
    from scipy.optimize import minimize, differential_evolution, linprog
    from scipy.stats import norm
    HAS_ADVANCED_OPTIMIZATION = True
except ImportError:
    HAS_ADVANCED_OPTIMIZATION = False
    logging.warning("Advanced optimization libraries not available. Using simplified optimization.")


class ResourceOptimizer(ABC):
    """Abstract base class for resource optimization algorithms"""
    
    @abstractmethod
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> Dict[str, Dict[str, float]]:
        """Optimize resource allocation across requests"""
        pass
    
    @abstractmethod
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization algorithm information"""
        pass


class LinearProgrammingOptimizer(ResourceOptimizer):
    """Linear programming based resource optimizer using scipy.optimize.linprog"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> Dict[str, Dict[str, float]]:
        """Optimize using linear programming with highs method"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION or not allocation_requests:
                return self._fallback_optimization(available_resources, allocation_requests)
            
            # Build constraint matrix and objective function
            resource_types = list(available_resources.keys())
            num_requests = len(allocation_requests)
            num_resources = len(resource_types)
            
            # Variables: allocation[request_i][resource_j]
            num_vars = num_requests * num_resources
            
            # Objective: maximize priority-weighted allocation satisfaction
            c = []
            for req in allocation_requests:
                for resource_type in resource_types:
                    # Negative because linprog minimizes
                    weight = -req.priority * req.urgency
                    c.append(weight)
            
            # Constraints
            A_ub = []
            b_ub = []
            
            # Resource capacity constraints
            for j, resource_type in enumerate(resource_types):
                constraint_row = [0] * num_vars
                for i in range(num_requests):
                    var_index = i * num_resources + j
                    constraint_row[var_index] = 1
                A_ub.append(constraint_row)
                b_ub.append(available_resources[resource_type])
            
            # Request requirement constraints (as bounds)
            bounds = []
            for i, req in enumerate(allocation_requests):
                for j, resource_type in enumerate(resource_types):
                    required = req.resource_requirements.get(resource_type, 0)
                    bounds.append((0, min(required * 1.5, available_resources[resource_type])))
            
            # Solve linear program
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                # Parse solution
                allocations = {}
                for i, req in enumerate(allocation_requests):
                    allocations[req.request_id] = {}
                    for j, resource_type in enumerate(resource_types):
                        var_index = i * num_resources + j
                        allocations[req.request_id][resource_type] = result.x[var_index]
                
                self.logger.info(f"Linear programming optimization successful. Objective value: {result.fun}")
                return allocations
            else:
                self.logger.warning(f"Linear programming failed: {result.message}")
                return self._fallback_optimization(available_resources, allocation_requests)
                
        except Exception as e:
            self.logger.error(f"Error in linear programming optimization: {e}")
            return self._fallback_optimization(available_resources, allocation_requests)
    
    def _fallback_optimization(self, available_resources: Dict[str, float], 
                             allocation_requests: List[AllocationRequest]) -> Dict[str, Dict[str, float]]:
        """Fallback proportional allocation when optimization fails"""
        if not allocation_requests:
            return {}
        
        allocations = {}
        
        # Calculate total demand per resource
        total_demand = {}
        for resource_type in available_resources:
            total_demand[resource_type] = sum(
                req.resource_requirements.get(resource_type, 0) 
                for req in allocation_requests
            )
        
        # Proportional allocation based on priority
        for req in allocation_requests:
            allocations[req.request_id] = {}
            
            for resource_type, available in available_resources.items():
                requested = req.resource_requirements.get(resource_type, 0)
                
                if total_demand[resource_type] > 0:
                    # Weight by priority and urgency
                    weight = req.priority * req.urgency
                    total_weight = sum(r.priority * r.urgency for r in allocation_requests)
                    
                    if total_weight > 0:
                        proportion = weight / total_weight
                        allocated = min(requested, available * proportion)
                    else:
                        allocated = min(requested, available / len(allocation_requests))
                else:
                    allocated = 0
                
                allocations[req.request_id][resource_type] = allocated
        
        return allocations
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get linear programming optimizer information"""
        return {
            'algorithm': 'Linear Programming',
            'method': 'highs',
            'supports_constraints': True,
            'supports_priorities': True,
            'complexity': 'Polynomial',
            'requires_scipy': True,
            'optimal_solution': True
        }


class GeneticAlgorithmOptimizer(ResourceOptimizer):
    """Genetic algorithm based resource optimizer using scipy differential_evolution"""
    
    def __init__(self, population_size: int = 50, max_generations: int = 100, 
                 convergence_tolerance: float = 1e-4):
        self.population_size = population_size
        self.max_generations = max_generations
        self.convergence_tolerance = convergence_tolerance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> Dict[str, Dict[str, float]]:
        """Optimize using genetic algorithm via differential evolution"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION or not allocation_requests:
                return self._fallback_optimization(available_resources, allocation_requests)
            
            resource_types = list(available_resources.keys())
            num_requests = len(allocation_requests)
            num_resources = len(resource_types)
            num_vars = num_requests * num_resources
            
            # Bounds for variables (0 to available resource)
            bounds = []
            for i, req in enumerate(allocation_requests):
                for j, resource_type in enumerate(resource_types):
                    max_alloc = min(
                        req.resource_requirements.get(resource_type, 0) * 1.2,
                        available_resources[resource_type]
                    )
                    bounds.append((0, max_alloc))
            
            # Define fitness function
            def fitness_function(x):
                return -self._evaluate_allocation_fitness(x, allocation_requests, 
                                                        available_resources, resource_types)
            
            # Run differential evolution
            result = differential_evolution(
                fitness_function,
                bounds,
                maxiter=self.max_generations,
                popsize=self.population_size,
                tol=self.convergence_tolerance,
                seed=42  # For reproducible results
            )
            
            if result.success:
                # Parse solution
                allocations = self._allocation_vector_to_dict(
                    result.x, allocation_requests, resource_types
                )
                
                self.logger.info(f"Genetic algorithm optimization successful. Fitness: {-result.fun}")
                return allocations
            else:
                self.logger.warning(f"Genetic algorithm failed: {result.message}")
                return self._fallback_optimization(available_resources, allocation_requests)
                
        except Exception as e:
            self.logger.error(f"Error in genetic algorithm optimization: {e}")
            return self._fallback_optimization(available_resources, allocation_requests)
    
    def _evaluate_allocation_fitness(self, allocation_vector: np.ndarray,
                                   allocation_requests: List[AllocationRequest],
                                   available_resources: Dict[str, float],
                                   resource_types: List[str]) -> float:
        """Evaluate fitness of allocation solution"""
        try:
            fitness = 0.0
            num_resources = len(resource_types)
            
            # Resource usage tracking
            resource_usage = {rt: 0.0 for rt in resource_types}
            
            # Evaluate each request
            for i, req in enumerate(allocation_requests):
                req_fitness = 0.0
                
                for j, resource_type in enumerate(resource_types):
                    var_index = i * num_resources + j
                    allocated = allocation_vector[var_index]
                    requested = req.resource_requirements.get(resource_type, 0)
                    
                    # Resource efficiency (how well we meet the request)
                    if requested > 0:
                        efficiency = min(allocated / requested, 1.0)
                        req_fitness += efficiency * 0.4
                    
                    # Track resource usage
                    resource_usage[resource_type] += allocated
                
                # Request satisfaction (weighted by priority and urgency)
                satisfaction = req_fitness
                weight = req.priority * req.urgency
                fitness += satisfaction * weight * 0.3
            
            # Resource constraint penalties
            constraint_penalty = 0.0
            for resource_type, used in resource_usage.items():
                available = available_resources[resource_type]
                if used > available:
                    # Heavy penalty for exceeding capacity
                    constraint_penalty += (used - available) / available * 10
            
            fitness -= constraint_penalty
            
            # Priority satisfaction bonus
            total_priority = sum(req.priority for req in allocation_requests)
            if total_priority > 0:
                priority_satisfaction = sum(
                    req.priority * min(1.0, sum(
                        allocation_vector[i * num_resources + j] / 
                        max(req.resource_requirements.get(resource_types[j], 1), 1)
                        for j in range(num_resources)
                    ) / num_resources)
                    for i, req in enumerate(allocation_requests)
                ) / total_priority
                fitness += priority_satisfaction * 0.3
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error evaluating allocation fitness: {e}")
            return -1000.0  # Poor fitness for invalid solutions
    
    def _allocation_vector_to_dict(self, allocation_vector: np.ndarray,
                                 allocation_requests: List[AllocationRequest],
                                 resource_types: List[str]) -> Dict[str, Dict[str, float]]:
        """Convert allocation vector to dictionary format"""
        allocations = {}
        num_resources = len(resource_types)
        
        for i, req in enumerate(allocation_requests):
            allocations[req.request_id] = {}
            for j, resource_type in enumerate(resource_types):
                var_index = i * num_resources + j
                allocations[req.request_id][resource_type] = allocation_vector[var_index]
        
        return allocations
    
    def _fallback_optimization(self, available_resources: Dict[str, float], 
                             allocation_requests: List[AllocationRequest]) -> Dict[str, Dict[str, float]]:
        """Fallback proportional allocation when optimization fails"""
        if not allocation_requests:
            return {}
        
        allocations = {}
        
        # Simple proportional allocation
        for req in allocation_requests:
            allocations[req.request_id] = {}
            for resource_type, available in available_resources.items():
                requested = req.resource_requirements.get(resource_type, 0)
                # Simple proportional based on priority
                total_priority = sum(r.priority for r in allocation_requests)
                if total_priority > 0:
                    proportion = req.priority / total_priority
                    allocated = min(requested, available * proportion)
                else:
                    allocated = min(requested, available / len(allocation_requests))
                allocations[req.request_id][resource_type] = allocated
        
        return allocations
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get genetic algorithm optimizer information"""
        return {
            'algorithm': 'Genetic Algorithm (Differential Evolution)',
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'convergence_tolerance': self.convergence_tolerance,
            'supports_constraints': True,
            'supports_priorities': True,
            'complexity': 'Heuristic',
            'requires_scipy': True,
            'optimal_solution': False,
            'global_optimization': True
        }