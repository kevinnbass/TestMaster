"""
Resource Optimization Algorithms
=================================

Optimization algorithms for resource allocation including Linear Programming
and Genetic Algorithm optimizers.
"""

import logging
import numpy as np
from typing import Dict, List, Any
from abc import ABC, abstractmethod

from .data_models import AllocationRequest, ResourceConstraint

# Advanced optimization imports
try:
    from scipy.optimize import differential_evolution, linprog
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
    """Linear programming based resource optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> Dict[str, Dict[str, float]]:
        """Optimize using linear programming"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION or not allocation_requests:
                return self._fallback_optimization(available_resources, allocation_requests)
            
            # Prepare optimization problem
            resource_types = list(available_resources.keys())
            n_requests = len(allocation_requests)
            n_resources = len(resource_types)
            
            # Objective function coefficients (maximize priority-weighted satisfaction)
            c = []
            for request in allocation_requests:
                for resource_type in resource_types:
                    priority_weight = request.priority / 10.0
                    urgency_weight = request.urgency
                    c.append(-(priority_weight * urgency_weight))  # Negative for maximization
            
            # Constraint matrix (resource capacity constraints)
            A_ub = []
            b_ub = []
            
            # Resource capacity constraints
            for i, resource_type in enumerate(resource_types):
                constraint_row = [0.0] * (n_requests * n_resources)
                for j in range(n_requests):
                    constraint_row[j * n_resources + i] = 1.0
                A_ub.append(constraint_row)
                b_ub.append(available_resources[resource_type])
            
            # Request requirement constraints (simplified)
            A_eq = []
            b_eq = []
            
            # Bounds for variables (non-negative allocations)
            bounds = []
            for request in allocation_requests:
                for resource_type in resource_types:
                    required = request.resource_requirements.get(resource_type, 0.0)
                    max_useful = required * 2.0  # Allow up to 2x requested amount
                    bounds.append((0.0, max_useful))
            
            # Solve linear program
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                # Extract allocation decisions
                allocations = {}
                for i, request in enumerate(allocation_requests):
                    request_allocations = {}
                    for j, resource_type in enumerate(resource_types):
                        idx = i * n_resources + j
                        request_allocations[resource_type] = float(result.x[idx])
                    allocations[request.request_id] = request_allocations
                
                return allocations
            else:
                self.logger.warning("Linear programming optimization failed, using fallback")
                return self._fallback_optimization(available_resources, allocation_requests)
                
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return self._fallback_optimization(available_resources, allocation_requests)
    
    def _fallback_optimization(self, 
                             available_resources: Dict[str, float],
                             allocation_requests: List[AllocationRequest]) -> Dict[str, Dict[str, float]]:
        """Fallback optimization using simple proportional allocation"""
        allocations = {}
        
        if not allocation_requests:
            return allocations
        
        # Sort requests by priority and urgency
        sorted_requests = sorted(
            allocation_requests, 
            key=lambda r: (r.priority, r.urgency), 
            reverse=True
        )
        
        # Track remaining resources
        remaining = dict(available_resources)
        
        for request in sorted_requests:
            request_allocations = {}
            
            for resource_type, required_amount in request.resource_requirements.items():
                available_amount = remaining.get(resource_type, 0.0)
                allocated_amount = min(required_amount, available_amount)
                
                request_allocations[resource_type] = allocated_amount
                remaining[resource_type] = max(0.0, available_amount - allocated_amount)
            
            allocations[request.request_id] = request_allocations
        
        return allocations
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization algorithm information"""
        return {
            'algorithm': 'LinearProgrammingOptimizer',
            'method': 'highs' if HAS_ADVANCED_OPTIMIZATION else 'fallback_proportional',
            'supports_constraints': True,
            'optimization_type': 'exact'
        }


class GeneticAlgorithmOptimizer(ResourceOptimizer):
    """Genetic algorithm based resource optimizer for complex scenarios"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_allocation(self, 
                          available_resources: Dict[str, float],
                          allocation_requests: List[AllocationRequest],
                          constraints: List[ResourceConstraint]) -> Dict[str, Dict[str, float]]:
        """Optimize using genetic algorithm"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION or not allocation_requests:
                return self._simple_allocation(available_resources, allocation_requests)
            
            # Define the optimization problem
            def objective_function(allocation_vector):
                return self._evaluate_allocation_fitness(
                    allocation_vector, available_resources, allocation_requests, constraints
                )
            
            # Define bounds for the optimization variables
            bounds = []
            for request in allocation_requests:
                for resource_type in available_resources.keys():
                    max_allocation = available_resources[resource_type]
                    bounds.append((0.0, max_allocation))
            
            # Run genetic algorithm optimization
            result = differential_evolution(
                objective_function, 
                bounds, 
                maxiter=self.generations,
                popsize=15,  # Smaller population for speed
                seed=42,
                atol=1e-6,
                tol=1e-6
            )
            
            if result.success:
                # Convert result back to allocation dictionary
                allocations = self._vector_to_allocations(
                    result.x, available_resources, allocation_requests
                )
                return allocations
            else:
                self.logger.warning("Genetic algorithm optimization failed, using simple allocation")
                return self._simple_allocation(available_resources, allocation_requests)
                
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization error: {e}")
            return self._simple_allocation(available_resources, allocation_requests)
    
    def _evaluate_allocation_fitness(self, 
                                   allocation_vector: np.ndarray,
                                   available_resources: Dict[str, float],
                                   allocation_requests: List[AllocationRequest],
                                   constraints: List[ResourceConstraint]) -> float:
        """Evaluate fitness of an allocation solution"""
        try:
            allocations = self._vector_to_allocations(
                allocation_vector, available_resources, allocation_requests
            )
            
            total_fitness = 0.0
            
            # Resource utilization efficiency
            resource_efficiency = self._calculate_resource_efficiency(allocations, available_resources)
            total_fitness += resource_efficiency * 0.3
            
            # Request satisfaction
            request_satisfaction = self._calculate_request_satisfaction(allocations, allocation_requests)
            total_fitness += request_satisfaction * 0.4
            
            # Priority weighting
            priority_satisfaction = self._calculate_priority_satisfaction(allocations, allocation_requests)
            total_fitness += priority_satisfaction * 0.2
            
            # Constraint satisfaction penalty
            constraint_penalty = self._calculate_constraint_penalty(allocations, constraints)
            total_fitness -= constraint_penalty * 0.1
            
            return -total_fitness  # Negative because differential_evolution minimizes
            
        except Exception as e:
            self.logger.error(f"Fitness evaluation error: {e}")
            return float('inf')  # Bad fitness for invalid solutions
    
    def _vector_to_allocations(self, 
                             allocation_vector: np.ndarray,
                             available_resources: Dict[str, float],
                             allocation_requests: List[AllocationRequest]) -> Dict[str, Dict[str, float]]:
        """Convert optimization vector back to allocation dictionary"""
        allocations = {}
        resource_types = list(available_resources.keys())
        n_resources = len(resource_types)
        
        for i, request in enumerate(allocation_requests):
            request_allocations = {}
            for j, resource_type in enumerate(resource_types):
                idx = i * n_resources + j
                if idx < len(allocation_vector):
                    request_allocations[resource_type] = max(0.0, float(allocation_vector[idx]))
                else:
                    request_allocations[resource_type] = 0.0
            allocations[request.request_id] = request_allocations
        
        return allocations
    
    def _calculate_resource_efficiency(self, 
                                     allocations: Dict[str, Dict[str, float]],
                                     available_resources: Dict[str, float]) -> float:
        """Calculate resource utilization efficiency"""
        if not available_resources:
            return 0.0
        
        total_efficiency = 0.0
        
        for resource_type, available in available_resources.items():
            if available <= 0:
                continue
            
            total_allocated = sum(
                request_allocations.get(resource_type, 0.0)
                for request_allocations in allocations.values()
            )
            
            utilization = min(total_allocated / available, 1.0)
            
            # Efficiency function - good utilization without over-allocation
            if utilization <= 0.8:
                efficiency = utilization
            elif utilization <= 1.0:
                efficiency = 0.8 - (utilization - 0.8) * 2  # Penalty for over-utilization
            else:
                efficiency = -1.0  # Heavy penalty for exceeding capacity
            
            total_efficiency += efficiency
        
        return total_efficiency / len(available_resources)
    
    def _calculate_request_satisfaction(self, 
                                      allocations: Dict[str, Dict[str, float]],
                                      allocation_requests: List[AllocationRequest]) -> float:
        """Calculate how well requests are satisfied"""
        if not allocation_requests:
            return 1.0
        
        total_satisfaction = 0.0
        
        for request in allocation_requests:
            request_allocations = allocations.get(request.request_id, {})
            request_satisfaction = 0.0
            resource_count = 0
            
            for resource_type, required in request.resource_requirements.items():
                if required > 0:
                    allocated = request_allocations.get(resource_type, 0.0)
                    satisfaction_ratio = min(allocated / required, 1.0)
                    request_satisfaction += satisfaction_ratio
                    resource_count += 1
            
            if resource_count > 0:
                request_satisfaction /= resource_count
            
            total_satisfaction += request_satisfaction
        
        return total_satisfaction / len(allocation_requests)
    
    def _calculate_priority_satisfaction(self, 
                                       allocations: Dict[str, Dict[str, float]],
                                       allocation_requests: List[AllocationRequest]) -> float:
        """Calculate priority-weighted satisfaction"""
        if not allocation_requests:
            return 1.0
        
        total_weighted_satisfaction = 0.0
        total_weight = 0.0
        
        for request in allocation_requests:
            request_allocations = allocations.get(request.request_id, {})
            weight = request.priority * request.urgency
            
            satisfaction = 0.0
            for resource_type, required in request.resource_requirements.items():
                if required > 0:
                    allocated = request_allocations.get(resource_type, 0.0)
                    satisfaction += min(allocated / required, 1.0)
            
            if request.resource_requirements:
                satisfaction /= len(request.resource_requirements)
            
            total_weighted_satisfaction += weight * satisfaction
            total_weight += weight
        
        return total_weighted_satisfaction / total_weight if total_weight > 0 else 0.0
    
    def _calculate_constraint_penalty(self, 
                                    allocations: Dict[str, Dict[str, float]],
                                    constraints: List[ResourceConstraint]) -> float:
        """Calculate penalty for constraint violations"""
        penalty = 0.0
        
        for constraint in constraints:
            total_allocation = sum(
                request_allocations.get(constraint.resource_type, 0.0)
                for request_allocations in allocations.values()
            )
            
            if total_allocation < constraint.min_allocation:
                penalty += (constraint.min_allocation - total_allocation) / constraint.min_allocation
            elif total_allocation > constraint.max_allocation:
                penalty += (total_allocation - constraint.max_allocation) / constraint.max_allocation
        
        return penalty
    
    def _simple_allocation(self, 
                         available_resources: Dict[str, float],
                         allocation_requests: List[AllocationRequest]) -> Dict[str, Dict[str, float]]:
        """Simple proportional allocation fallback"""
        return LinearProgrammingOptimizer()._fallback_optimization(
            available_resources, allocation_requests
        )
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization algorithm information"""
        return {
            'algorithm': 'GeneticAlgorithmOptimizer',
            'population_size': self.population_size,
            'generations': self.generations,
            'supports_constraints': True,
            'optimization_type': 'heuristic'
        }


__all__ = [
    'ResourceOptimizer',
    'LinearProgrammingOptimizer',
    'GeneticAlgorithmOptimizer'
]