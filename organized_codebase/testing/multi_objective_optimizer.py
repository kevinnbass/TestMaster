"""
Multi-Objective Optimization Core

Implements multi-objective optimization for test generation.
Adapted from Agency Swarm's task optimization and PraisonAI's goal balancing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from enum import Enum
import numpy as np
from datetime import datetime
import json
import math
from abc import ABC, abstractmethod


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"  # Optimize towards a specific value
    CONSTRAINT = "constraint"  # Must satisfy constraint


@dataclass
class OptimizationObjective:
    """Represents a single optimization objective."""
    name: str
    type: ObjectiveType
    weight: float = 1.0
    target_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    evaluator: Optional[Callable] = None
    description: str = ""
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate the objective for a given solution."""
        if self.evaluator:
            value = self.evaluator(solution)
        else:
            value = 0.0
        
        # Apply constraints
        if self.min_value is not None and value < self.min_value:
            value = self.min_value
        if self.max_value is not None and value > self.max_value:
            value = self.max_value
        
        return value
    
    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        if self.max_value is not None and self.min_value is not None:
            range_val = self.max_value - self.min_value
            if range_val > 0:
                return (value - self.min_value) / range_val
        return value
    
    def fitness(self, value: float) -> float:
        """Calculate fitness based on objective type."""
        normalized = self.normalize(value)
        
        if self.type == ObjectiveType.MAXIMIZE:
            return normalized * self.weight
        elif self.type == ObjectiveType.MINIMIZE:
            return (1.0 - normalized) * self.weight
        elif self.type == ObjectiveType.TARGET:
            if self.target_value is not None:
                distance = abs(value - self.target_value)
                return (1.0 / (1.0 + distance)) * self.weight
        elif self.type == ObjectiveType.CONSTRAINT:
            # Binary: satisfied (1.0) or not (0.0)
            if self.min_value is not None and value < self.min_value:
                return 0.0
            if self.max_value is not None and value > self.max_value:
                return 0.0
            return 1.0 * self.weight
        
        return normalized * self.weight


@dataclass
class Solution:
    """Represents a solution in the optimization space."""
    id: str
    genes: List[Any]  # Solution representation
    objectives: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    rank: int = 0  # For Pareto ranking
    crowding_distance: float = 0.0  # For diversity preservation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def dominates(self, other: 'Solution') -> bool:
        """Check if this solution dominates another (Pareto dominance)."""
        better_in_at_least_one = False
        
        for obj_name in self.objectives:
            if obj_name not in other.objectives:
                continue
            
            if self.objectives[obj_name] < other.objectives[obj_name]:
                return False  # Worse in at least one objective
            elif self.objectives[obj_name] > other.objectives[obj_name]:
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'genes': self.genes,
            'objectives': self.objectives,
            'fitness': self.fitness,
            'rank': self.rank,
            'crowding_distance': self.crowding_distance,
            'metadata': self.metadata
        }


@dataclass
class ParetoFront:
    """Represents the Pareto optimal front."""
    solutions: List[Solution] = field(default_factory=list)
    generation: int = 0
    
    def add(self, solution: Solution):
        """Add a solution if it's non-dominated."""
        # Check if solution is dominated by any existing solution
        for existing in self.solutions:
            if existing.dominates(solution):
                return  # Don't add dominated solution
        
        # Remove solutions dominated by the new solution
        self.solutions = [s for s in self.solutions if not solution.dominates(s)]
        
        # Add the new solution
        self.solutions.append(solution)
    
    def get_best(self, objective_name: str) -> Optional[Solution]:
        """Get the best solution for a specific objective."""
        if not self.solutions:
            return None
        
        return max(self.solutions, key=lambda s: s.objectives.get(objective_name, 0))
    
    def get_balanced(self) -> Optional[Solution]:
        """Get the most balanced solution (closest to ideal point)."""
        if not self.solutions:
            return None
        
        # Calculate ideal point (best value for each objective)
        ideal_point = {}
        for solution in self.solutions:
            for obj_name, value in solution.objectives.items():
                if obj_name not in ideal_point or value > ideal_point[obj_name]:
                    ideal_point[obj_name] = value
        
        # Find solution closest to ideal point
        best_solution = None
        min_distance = float('inf')
        
        for solution in self.solutions:
            distance = 0.0
            for obj_name, ideal_value in ideal_point.items():
                obj_value = solution.objectives.get(obj_name, 0)
                distance += (ideal_value - obj_value) ** 2
            distance = math.sqrt(distance)
            
            if distance < min_distance:
                min_distance = distance
                best_solution = solution
        
        return best_solution


@dataclass
class OptimizationConfig:
    """Configuration for multi-objective optimization."""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    
    # Algorithm-specific parameters
    algorithm: str = "nsga2"  # nsga2, moead, pso, etc.
    selection_method: str = "tournament"
    tournament_size: int = 3
    
    # Convergence criteria
    convergence_threshold: float = 0.001
    patience: int = 10  # Generations without improvement
    
    # Diversity preservation
    maintain_diversity: bool = True
    diversity_threshold: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elitism_rate': self.elitism_rate,
            'algorithm': self.algorithm,
            'selection_method': self.selection_method,
            'tournament_size': self.tournament_size,
            'convergence_threshold': self.convergence_threshold,
            'patience': self.patience,
            'maintain_diversity': self.maintain_diversity,
            'diversity_threshold': self.diversity_threshold
        }


@dataclass
class OptimizationResult:
    """Result of multi-objective optimization."""
    pareto_front: ParetoFront
    all_solutions: List[Solution]
    best_solutions: Dict[str, Solution]  # Best for each objective
    balanced_solution: Optional[Solution]
    
    # Metrics
    generations_run: int = 0
    convergence_achieved: bool = False
    optimization_time: float = 0.0
    
    # Statistics
    average_fitness: float = 0.0
    fitness_improvement: float = 0.0
    diversity_score: float = 0.0
    hypervolume: float = 0.0  # Measure of Pareto front quality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pareto_front_size': len(self.pareto_front.solutions),
            'total_solutions': len(self.all_solutions),
            'best_solutions': {k: v.to_dict() for k, v in self.best_solutions.items()},
            'balanced_solution': self.balanced_solution.to_dict() if self.balanced_solution else None,
            'generations_run': self.generations_run,
            'convergence_achieved': self.convergence_achieved,
            'optimization_time': self.optimization_time,
            'average_fitness': self.average_fitness,
            'fitness_improvement': self.fitness_improvement,
            'diversity_score': self.diversity_score,
            'hypervolume': self.hypervolume
        }


class MultiObjectiveOptimizer:
    """Main multi-objective optimization engine."""
    
    def __init__(self, 
                 objectives: List[OptimizationObjective],
                 config: OptimizationConfig = None):
        self.objectives = objectives
        self.config = config or OptimizationConfig()
        
        # Population
        self.population: List[Solution] = []
        self.pareto_front = ParetoFront()
        
        # Tracking
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.no_improvement_count = 0
        
        print(f"Multi-Objective Optimizer initialized")
        print(f"   Objectives: {len(objectives)}")
        print(f"   Algorithm: {self.config.algorithm}")
        print(f"   Population: {self.config.population_size}")
    
    def optimize(self, initial_solutions: List[Solution] = None) -> OptimizationResult:
        """Run multi-objective optimization."""
        start_time = datetime.now()
        
        print(f"\nStarting multi-objective optimization...")
        print(f"   Max generations: {self.config.max_generations}")
        
        # Initialize population
        if initial_solutions:
            self.population = initial_solutions[:self.config.population_size]
        else:
            self.population = self._initialize_population()
        
        # Evaluate initial population
        self._evaluate_population(self.population)
        
        # Main optimization loop
        for gen in range(self.config.max_generations):
            self.generation = gen
            
            # Select parents
            parents = self._selection(self.population)
            
            # Generate offspring
            offspring = self._crossover(parents)
            offspring = self._mutation(offspring)
            
            # Evaluate offspring
            self._evaluate_population(offspring)
            
            # Environmental selection (survivor selection)
            self.population = self._environmental_selection(
                self.population + offspring
            )
            
            # Update Pareto front
            self._update_pareto_front()
            
            # Track metrics
            self._track_metrics()
            
            # Check convergence
            if self._check_convergence():
                print(f"   Convergence achieved at generation {gen}")
                break
            
            # Progress report
            if gen % 10 == 0:
                best_fitness = max(s.fitness for s in self.population)
                avg_fitness = sum(s.fitness for s in self.population) / len(self.population)
                print(f"   Generation {gen}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, "
                      f"Pareto size={len(self.pareto_front.solutions)}")
        
        # Create result
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = self._create_result(optimization_time)
        
        print(f"\nOptimization complete:")
        print(f"   Generations: {self.generation}")
        print(f"   Pareto front size: {len(self.pareto_front.solutions)}")
        print(f"   Time: {optimization_time:.2f}s")
        
        return result
    
    def _initialize_population(self) -> List[Solution]:
        """Initialize random population."""
        population = []
        
        for i in range(self.config.population_size):
            # Create random solution
            genes = self._create_random_genes()
            
            solution = Solution(
                id=f"sol_{i}",
                genes=genes
            )
            
            population.append(solution)
        
        return population
    
    def _create_random_genes(self) -> List[Any]:
        """Create random genes for a solution."""
        # This should be overridden by specific implementations
        # Default: random binary string
        import random
        return [random.random() for _ in range(10)]
    
    def _evaluate_population(self, population: List[Solution]):
        """Evaluate all solutions in population."""
        for solution in population:
            # Evaluate each objective
            for objective in self.objectives:
                value = objective.evaluate(solution)
                solution.objectives[objective.name] = value
            
            # Calculate overall fitness
            solution.fitness = self._calculate_fitness(solution)
    
    def _calculate_fitness(self, solution: Solution) -> float:
        """Calculate overall fitness from objectives."""
        total_fitness = 0.0
        
        for objective in self.objectives:
            value = solution.objectives.get(objective.name, 0)
            total_fitness += objective.fitness(value)
        
        return total_fitness
    
    def _selection(self, population: List[Solution]) -> List[Solution]:
        """Select parents for reproduction."""
        if self.config.selection_method == "tournament":
            return self._tournament_selection(population)
        elif self.config.selection_method == "roulette":
            return self._roulette_selection(population)
        else:
            # Default to tournament
            return self._tournament_selection(population)
    
    def _tournament_selection(self, population: List[Solution]) -> List[Solution]:
        """Tournament selection."""
        import random
        
        selected = []
        num_parents = len(population) // 2
        
        for _ in range(num_parents):
            # Select tournament participants
            tournament = random.sample(population, self.config.tournament_size)
            
            # Winner is the one with best fitness
            winner = max(tournament, key=lambda s: s.fitness)
            selected.append(winner)
        
        return selected
    
    def _roulette_selection(self, population: List[Solution]) -> List[Solution]:
        """Roulette wheel selection."""
        import random
        
        # Calculate selection probabilities
        total_fitness = sum(s.fitness for s in population)
        if total_fitness == 0:
            # Equal probability if all have zero fitness
            probabilities = [1.0 / len(population)] * len(population)
        else:
            probabilities = [s.fitness / total_fitness for s in population]
        
        # Select parents
        num_parents = len(population) // 2
        selected = np.random.choice(
            population, 
            size=num_parents,
            p=probabilities,
            replace=True
        ).tolist()
        
        return selected
    
    def _crossover(self, parents: List[Solution]) -> List[Solution]:
        """Apply crossover to create offspring."""
        import random
        
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            if random.random() < self.config.crossover_rate:
                # Perform crossover
                child1_genes, child2_genes = self._single_point_crossover(
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
                # No crossover, copy parents
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def _single_point_crossover(self, genes1: List[Any], genes2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Single-point crossover."""
        import random
        
        if len(genes1) <= 1:
            return genes1.copy(), genes2.copy()
        
        # Select crossover point
        point = random.randint(1, len(genes1) - 1)
        
        # Create offspring
        child1_genes = genes1[:point] + genes2[point:]
        child2_genes = genes2[:point] + genes1[point:]
        
        return child1_genes, child2_genes
    
    def _mutation(self, offspring: List[Solution]) -> List[Solution]:
        """Apply mutation to offspring."""
        import random
        
        for solution in offspring:
            if random.random() < self.config.mutation_rate:
                # Mutate genes
                solution.genes = self._mutate_genes(solution.genes)
        
        return offspring
    
    def _mutate_genes(self, genes: List[Any]) -> List[Any]:
        """Mutate genes."""
        import random
        
        mutated = genes.copy()
        
        # Random mutation: change one gene
        if mutated:
            index = random.randint(0, len(mutated) - 1)
            mutated[index] = random.random()  # Simple random mutation
        
        return mutated
    
    def _environmental_selection(self, combined: List[Solution]) -> List[Solution]:
        """Select survivors for next generation."""
        # Sort by fitness
        combined.sort(key=lambda s: s.fitness, reverse=True)
        
        # Elitism: keep best solutions
        num_elite = int(self.config.population_size * self.config.elitism_rate)
        elite = combined[:num_elite]
        
        # Fill rest with diversity preservation
        if self.config.maintain_diversity:
            remaining = self._select_diverse(
                combined[num_elite:],
                self.config.population_size - num_elite
            )
        else:
            remaining = combined[num_elite:self.config.population_size]
        
        return elite + remaining
    
    def _select_diverse(self, candidates: List[Solution], num_select: int) -> List[Solution]:
        """Select diverse solutions."""
        if not candidates or num_select <= 0:
            return []
        
        selected = []
        remaining = candidates.copy()
        
        # Select solutions that maximize diversity
        while len(selected) < num_select and remaining:
            # Find solution most different from selected
            best_candidate = None
            max_distance = -1
            
            for candidate in remaining:
                if not selected:
                    # First selection, choose best fitness
                    distance = candidate.fitness
                else:
                    # Calculate minimum distance to selected solutions
                    min_dist = min(
                        self._solution_distance(candidate, s) for s in selected
                    )
                    distance = min_dist
                
                if distance > max_distance:
                    max_distance = distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _solution_distance(self, sol1: Solution, sol2: Solution) -> float:
        """Calculate distance between two solutions."""
        # Objective space distance
        distance = 0.0
        
        for obj_name in sol1.objectives:
            if obj_name in sol2.objectives:
                diff = sol1.objectives[obj_name] - sol2.objectives[obj_name]
                distance += diff * diff
        
        return math.sqrt(distance)
    
    def _update_pareto_front(self):
        """Update Pareto front with current population."""
        for solution in self.population:
            self.pareto_front.add(solution)
    
    def _track_metrics(self):
        """Track optimization metrics."""
        # Best fitness
        best_fitness = max(s.fitness for s in self.population)
        self.best_fitness_history.append(best_fitness)
        
        # Diversity
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)
        
        # Check improvement
        if len(self.best_fitness_history) > 1:
            improvement = best_fitness - self.best_fitness_history[-2]
            if improvement < self.config.convergence_threshold:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        # Average pairwise distance
        total_distance = 0.0
        count = 0
        
        for i, sol1 in enumerate(self.population):
            for sol2 in self.population[i+1:]:
                total_distance += self._solution_distance(sol1, sol2)
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        # Check patience
        if self.no_improvement_count >= self.config.patience:
            return True
        
        # Check diversity
        if self.diversity_history:
            current_diversity = self.diversity_history[-1]
            if current_diversity < self.config.diversity_threshold:
                return True
        
        return False
    
    def _create_result(self, optimization_time: float) -> OptimizationResult:
        """Create optimization result."""
        # Find best solutions for each objective
        best_solutions = {}
        for objective in self.objectives:
            best = self.pareto_front.get_best(objective.name)
            if best:
                best_solutions[objective.name] = best
        
        # Get balanced solution
        balanced = self.pareto_front.get_balanced()
        
        # Calculate metrics
        avg_fitness = sum(s.fitness for s in self.population) / len(self.population)
        
        fitness_improvement = 0.0
        if self.best_fitness_history:
            fitness_improvement = (
                self.best_fitness_history[-1] - self.best_fitness_history[0]
            )
        
        diversity = self._calculate_diversity()
        
        # Calculate hypervolume (simplified)
        hypervolume = self._calculate_hypervolume()
        
        return OptimizationResult(
            pareto_front=self.pareto_front,
            all_solutions=self.population,
            best_solutions=best_solutions,
            balanced_solution=balanced,
            generations_run=self.generation,
            convergence_achieved=(self.no_improvement_count >= self.config.patience),
            optimization_time=optimization_time,
            average_fitness=avg_fitness,
            fitness_improvement=fitness_improvement,
            diversity_score=diversity,
            hypervolume=hypervolume
        )
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume indicator for Pareto front."""
        if not self.pareto_front.solutions:
            return 0.0
        
        # Simplified hypervolume calculation
        # (proper implementation would use WFG algorithm)
        total_volume = 1.0
        
        for solution in self.pareto_front.solutions:
            volume = 1.0
            for value in solution.objectives.values():
                volume *= max(0.0, value)  # Assume reference point at origin
            total_volume *= (1.0 + volume)
        
        return math.log(total_volume)