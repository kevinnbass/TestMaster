"""
Optimization Algorithms for Multi-Objective Optimization

Implements various optimization algorithms.
Adapted from Swarm intelligence patterns in Agency Swarm and PraisonAI.
"""

from typing import List, Tuple, Optional
import numpy as np
import random
import math
from dataclasses import dataclass

from .multi_objective_optimizer import (
    MultiObjectiveOptimizer, Solution, OptimizationObjective,
    OptimizationConfig, OptimizationResult
)


class NSGAIIOptimizer(MultiObjectiveOptimizer):
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II).
    One of the most popular multi-objective optimization algorithms.
    """
    
    def __init__(self, objectives: List[OptimizationObjective], config: OptimizationConfig = None):
        if config:
            config.algorithm = "nsga2"
        super().__init__(objectives, config)
        
        print(f"NSGA-II Optimizer initialized")
    
    def _environmental_selection(self, combined: List[Solution]) -> List[Solution]:
        """NSGA-II environmental selection using non-dominated sorting and crowding distance."""
        
        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined)
        
        selected = []
        current_front = 0
        
        # Add fronts until we exceed population size
        while len(selected) < self.config.population_size and current_front < len(fronts):
            front = fronts[current_front]
            
            if len(selected) + len(front) <= self.config.population_size:
                # Add entire front
                selected.extend(front)
            else:
                # Need to select from this front based on crowding distance
                remaining_slots = self.config.population_size - len(selected)
                
                # Calculate crowding distance
                self._calculate_crowding_distance(front)
                
                # Sort by crowding distance (descending) and select
                front.sort(key=lambda s: s.crowding_distance, reverse=True)
                selected.extend(front[:remaining_slots])
            
            current_front += 1
        
        return selected
    
    def _fast_non_dominated_sort(self, population: List[Solution]) -> List[List[Solution]]:
        """Fast non-dominated sorting algorithm."""
        fronts = []
        
        # Initialize domination counts and dominated solutions
        domination_count = {}
        dominated_solutions = {}
        
        for p in population:
            dominated_solutions[p.id] = []
            domination_count[p.id] = 0
        
        # Calculate domination relationships
        for i, p in enumerate(population):
            for q in population[i+1:]:
                if p.dominates(q):
                    dominated_solutions[p.id].append(q)
                    domination_count[q.id] += 1
                elif q.dominates(p):
                    dominated_solutions[q.id].append(p)
                    domination_count[p.id] += 1
        
        # Find first front (non-dominated solutions)
        current_front = []
        for p in population:
            if domination_count[p.id] == 0:
                p.rank = 0
                current_front.append(p)
        
        fronts.append(current_front)
        
        # Find subsequent fronts
        front_rank = 0
        while current_front:
            next_front = []
            
            for p in current_front:
                for q in dominated_solutions[p.id]:
                    domination_count[q.id] -= 1
                    
                    if domination_count[q.id] == 0:
                        q.rank = front_rank + 1
                        next_front.append(q)
            
            if next_front:
                fronts.append(next_front)
            
            current_front = next_front
            front_rank += 1
        
        return fronts
    
    def _calculate_crowding_distance(self, front: List[Solution]):
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for solution in front:
            solution.crowding_distance = 0.0
        
        # Calculate distance for each objective
        for objective in self.objectives:
            # Sort by this objective
            front.sort(key=lambda s: s.objectives.get(objective.name, 0))
            
            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_min = front[0].objectives.get(objective.name, 0)
            obj_max = front[-1].objectives.get(objective.name, 0)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate crowding distance
            for i in range(1, len(front) - 1):
                prev_val = front[i-1].objectives.get(objective.name, 0)
                next_val = front[i+1].objectives.get(objective.name, 0)
                
                front[i].crowding_distance += (next_val - prev_val) / obj_range


class MOEADOptimizer(MultiObjectiveOptimizer):
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D).
    Decomposes multi-objective problem into scalar subproblems.
    """
    
    def __init__(self, objectives: List[OptimizationObjective], config: OptimizationConfig = None):
        if config:
            config.algorithm = "moead"
        super().__init__(objectives, config)
        
        # MOEA/D specific parameters
        self.weight_vectors = self._generate_weight_vectors()
        self.neighborhoods = self._define_neighborhoods()
        self.reference_point = self._initialize_reference_point()
        
        print(f"MOEA/D Optimizer initialized")
        print(f"   Subproblems: {len(self.weight_vectors)}")
    
    def _generate_weight_vectors(self) -> List[np.ndarray]:
        """Generate uniformly distributed weight vectors."""
        num_objectives = len(self.objectives)
        num_vectors = self.config.population_size
        
        # Simple uniform distribution (can be improved with simplex-lattice design)
        vectors = []
        for i in range(num_vectors):
            vector = np.random.dirichlet(np.ones(num_objectives))
            vectors.append(vector)
        
        return vectors
    
    def _define_neighborhoods(self, size: int = 5) -> List[List[int]]:
        """Define neighborhood structure based on weight vector distances."""
        neighborhoods = []
        
        for i, wi in enumerate(self.weight_vectors):
            # Calculate distances to all other weight vectors
            distances = []
            for j, wj in enumerate(self.weight_vectors):
                if i != j:
                    dist = np.linalg.norm(wi - wj)
                    distances.append((j, dist))
            
            # Sort by distance and select nearest neighbors
            distances.sort(key=lambda x: x[1])
            neighbors = [idx for idx, _ in distances[:size]]
            neighborhoods.append(neighbors)
        
        return neighborhoods
    
    def _initialize_reference_point(self) -> np.ndarray:
        """Initialize reference point for scalarization."""
        # Use ideal point (best value for each objective)
        ref_point = np.zeros(len(self.objectives))
        
        for i, obj in enumerate(self.objectives):
            if obj.type.value == "minimize":
                ref_point[i] = obj.min_value if obj.min_value is not None else 0
            else:
                ref_point[i] = obj.max_value if obj.max_value is not None else 100
        
        return ref_point
    
    def _tchebycheff_scalarization(self, solution: Solution, weight: np.ndarray) -> float:
        """Tchebycheff scalarization function."""
        max_val = 0.0
        
        for i, obj in enumerate(self.objectives):
            obj_val = solution.objectives.get(obj.name, 0)
            ref_val = self.reference_point[i]
            
            # Normalize
            normalized = abs(obj_val - ref_val)
            
            # Weighted Tchebycheff
            weighted = weight[i] * normalized
            max_val = max(max_val, weighted)
        
        return max_val


class ParticleSwarmOptimizer(MultiObjectiveOptimizer):
    """
    Multi-Objective Particle Swarm Optimization (MOPSO).
    Inspired by swarm intelligence.
    """
    
    def __init__(self, objectives: List[OptimizationObjective], config: OptimizationConfig = None):
        if config:
            config.algorithm = "pso"
        super().__init__(objectives, config)
        
        # PSO specific parameters
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.velocities = {}
        self.personal_best = {}
        self.global_best = None
        
        print(f"Particle Swarm Optimizer initialized")
    
    def _initialize_population(self) -> List[Solution]:
        """Initialize particles with positions and velocities."""
        population = super()._initialize_population()
        
        # Initialize velocities and personal best
        for solution in population:
            # Random initial velocity
            velocity = [random.uniform(-1, 1) for _ in solution.genes]
            self.velocities[solution.id] = velocity
            
            # Initial position is personal best
            self.personal_best[solution.id] = solution
        
        return population
    
    def _update_particles(self, population: List[Solution]):
        """Update particle positions and velocities."""
        
        # Update global best
        if self.global_best is None or max(population, key=lambda s: s.fitness).fitness > self.global_best.fitness:
            self.global_best = max(population, key=lambda s: s.fitness)
        
        # Update each particle
        for solution in population:
            velocity = self.velocities[solution.id]
            pbest = self.personal_best[solution.id]
            
            # Update velocity
            for i in range(len(solution.genes)):
                r1, r2 = random.random(), random.random()
                
                cognitive = self.cognitive_weight * r1 * (pbest.genes[i] - solution.genes[i])
                social = self.social_weight * r2 * (self.global_best.genes[i] - solution.genes[i])
                
                velocity[i] = self.inertia_weight * velocity[i] + cognitive + social
                
                # Clamp velocity
                velocity[i] = max(-1, min(1, velocity[i]))
            
            # Update position
            for i in range(len(solution.genes)):
                solution.genes[i] += velocity[i]
                
                # Ensure within bounds [0, 1]
                solution.genes[i] = max(0, min(1, solution.genes[i]))
            
            # Update personal best
            if solution.fitness > pbest.fitness:
                self.personal_best[solution.id] = solution


class SimulatedAnnealingOptimizer(MultiObjectiveOptimizer):
    """
    Multi-Objective Simulated Annealing (MOSA).
    Uses temperature-based acceptance criteria.
    """
    
    def __init__(self, objectives: List[OptimizationObjective], config: OptimizationConfig = None):
        if config:
            config.algorithm = "simulated_annealing"
        super().__init__(objectives, config)
        
        # SA specific parameters
        self.initial_temperature = 100.0
        self.cooling_rate = 0.95
        self.current_temperature = self.initial_temperature
        
        print(f"Simulated Annealing Optimizer initialized")
        print(f"   Initial temperature: {self.initial_temperature}")
    
    def _accept_solution(self, current: Solution, candidate: Solution) -> bool:
        """Determine whether to accept a candidate solution."""
        if candidate.fitness >= current.fitness:
            return True  # Always accept better solutions
        
        # Accept worse solutions with probability based on temperature
        delta = candidate.fitness - current.fitness
        probability = math.exp(delta / self.current_temperature)
        
        return random.random() < probability
    
    def _cool_down(self):
        """Reduce temperature."""
        self.current_temperature *= self.cooling_rate
    
    def _perturb_solution(self, solution: Solution) -> Solution:
        """Create a neighbor solution by perturbation."""
        import copy
        
        neighbor = copy.deepcopy(solution)
        neighbor.id = f"{solution.id}_perturbed_{self.generation}"
        
        # Perturb genes
        for i in range(len(neighbor.genes)):
            if random.random() < 0.3:  # 30% chance to perturb each gene
                perturbation = random.gauss(0, 0.1)
                neighbor.genes[i] += perturbation
                neighbor.genes[i] = max(0, min(1, neighbor.genes[i]))
        
        return neighbor


class GeneticAlgorithmOptimizer(MultiObjectiveOptimizer):
    """
    Multi-Objective Genetic Algorithm (MOGA).
    Classic evolutionary approach with genetic operators.
    """
    
    def __init__(self, objectives: List[OptimizationObjective], config: OptimizationConfig = None):
        if config:
            config.algorithm = "genetic"
        super().__init__(objectives, config)
        
        print(f"Genetic Algorithm Optimizer initialized")
    
    def _crossover(self, parents: List[Solution]) -> List[Solution]:
        """Apply various crossover operators."""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            if random.random() < self.config.crossover_rate:
                # Choose crossover method
                method = random.choice(['single_point', 'two_point', 'uniform'])
                
                if method == 'single_point':
                    child1_genes, child2_genes = self._single_point_crossover(
                        parent1.genes, parent2.genes
                    )
                elif method == 'two_point':
                    child1_genes, child2_genes = self._two_point_crossover(
                        parent1.genes, parent2.genes
                    )
                else:  # uniform
                    child1_genes, child2_genes = self._uniform_crossover(
                        parent1.genes, parent2.genes
                    )
                
                child1 = Solution(
                    id=f"sol_{self.generation}_{len(offspring)}",
                    genes=child1_genes
                )
                child2 = Solution(
                    id=f"sol_{self.generation}_{len(offspring)+1}",
                    genes=child2_genes
                )
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def _two_point_crossover(self, genes1: List, genes2: List) -> Tuple[List, List]:
        """Two-point crossover."""
        if len(genes1) <= 2:
            return genes1.copy(), genes2.copy()
        
        # Select two crossover points
        points = sorted(random.sample(range(1, len(genes1)), 2))
        
        # Create offspring
        child1_genes = genes1[:points[0]] + genes2[points[0]:points[1]] + genes1[points[1]:]
        child2_genes = genes2[:points[0]] + genes1[points[0]:points[1]] + genes2[points[1]:]
        
        return child1_genes, child2_genes
    
    def _uniform_crossover(self, genes1: List, genes2: List) -> Tuple[List, List]:
        """Uniform crossover."""
        child1_genes = []
        child2_genes = []
        
        for g1, g2 in zip(genes1, genes2):
            if random.random() < 0.5:
                child1_genes.append(g1)
                child2_genes.append(g2)
            else:
                child1_genes.append(g2)
                child2_genes.append(g1)
        
        return child1_genes, child2_genes
    
    def _mutation(self, offspring: List[Solution]) -> List[Solution]:
        """Apply various mutation operators."""
        for solution in offspring:
            if random.random() < self.config.mutation_rate:
                # Choose mutation method
                method = random.choice(['gaussian', 'uniform', 'boundary'])
                
                if method == 'gaussian':
                    solution.genes = self._gaussian_mutation(solution.genes)
                elif method == 'uniform':
                    solution.genes = self._uniform_mutation(solution.genes)
                else:  # boundary
                    solution.genes = self._boundary_mutation(solution.genes)
        
        return offspring
    
    def _gaussian_mutation(self, genes: List) -> List:
        """Gaussian mutation."""
        mutated = genes.copy()
        
        for i in range(len(mutated)):
            if random.random() < 0.2:  # 20% chance per gene
                mutated[i] += random.gauss(0, 0.1)
                mutated[i] = max(0, min(1, mutated[i]))
        
        return mutated
    
    def _uniform_mutation(self, genes: List) -> List:
        """Uniform mutation."""
        mutated = genes.copy()
        
        for i in range(len(mutated)):
            if random.random() < 0.2:
                mutated[i] = random.random()
        
        return mutated
    
    def _boundary_mutation(self, genes: List) -> List:
        """Boundary mutation."""
        mutated = genes.copy()
        
        for i in range(len(mutated)):
            if random.random() < 0.2:
                mutated[i] = 0.0 if random.random() < 0.5 else 1.0
        
        return mutated