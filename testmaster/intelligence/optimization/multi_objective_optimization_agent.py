"""
Multi-Objective Optimization Agent

Optimizes test generation across multiple competing objectives including coverage,
performance, security, maintainability, and resource efficiency using intelligent
optimization algorithms and consensus-driven decision making.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from enum import Enum
from datetime import datetime
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..hierarchical_planning import (
    HierarchicalTestPlanner, 
    PlanningNode, 
    TestPlanGenerator, 
    TestPlanEvaluator,
    EvaluationCriteria,
    get_best_planner
)
from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ..security import SecurityIntelligenceAgent
from ...core.shared_state import get_shared_state, cache_test_result, get_cached_test_result


class OptimizationObjective(Enum):
    """Optimization objectives for test generation."""
    COVERAGE_MAXIMIZATION = "coverage_maximization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_ENHANCEMENT = "security_enhancement"
    MAINTAINABILITY = "maintainability"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    EXECUTION_SPEED = "execution_speed"
    RELIABILITY = "reliability"
    COMPREHENSIVENESS = "comprehensiveness"


class OptimizationStrategy(Enum):
    """Multi-objective optimization strategies."""
    PARETO_OPTIMIZATION = "pareto_optimization"
    WEIGHTED_SUM = "weighted_sum"
    LEXICOGRAPHIC = "lexicographic"
    NSGA_II = "nsga_ii"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"


@dataclass
class ObjectiveWeights:
    """Weights for different optimization objectives."""
    coverage: float = 0.25
    performance: float = 0.20
    security: float = 0.20
    maintainability: float = 0.15
    resource_efficiency: float = 0.10
    execution_speed: float = 0.05
    reliability: float = 0.03
    comprehensiveness: float = 0.02
    
    def normalize(self) -> 'ObjectiveWeights':
        """Normalize weights to sum to 1.0."""
        total = (self.coverage + self.performance + self.security + 
                self.maintainability + self.resource_efficiency + 
                self.execution_speed + self.reliability + self.comprehensiveness)
        
        if total == 0:
            return ObjectiveWeights()  # Default weights
        
        return ObjectiveWeights(
            coverage=self.coverage / total,
            performance=self.performance / total,
            security=self.security / total,
            maintainability=self.maintainability / total,
            resource_efficiency=self.resource_efficiency / total,
            execution_speed=self.execution_speed / total,
            reliability=self.reliability / total,
            comprehensiveness=self.comprehensiveness / total
        )


@dataclass
class OptimizationCandidate:
    """A candidate solution for optimization."""
    solution_id: str
    test_plan: Dict[str, Any]
    objective_scores: Dict[str, float] = field(default_factory=dict)
    fitness_score: float = 0.0
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of multi-objective optimization."""
    best_solutions: List[OptimizationCandidate]
    pareto_front: List[OptimizationCandidate]
    optimization_statistics: Dict[str, Any]
    convergence_history: List[Dict[str, Any]]
    total_time: float
    iterations: int
    strategy_used: OptimizationStrategy


class MultiObjectiveOptimizer:
    """Core multi-objective optimization engine."""
    
    def __init__(self, 
                 objectives: List[OptimizationObjective] = None,
                 weights: ObjectiveWeights = None,
                 strategy: OptimizationStrategy = OptimizationStrategy.NSGA_II):
        
        self.objectives = objectives or [
            OptimizationObjective.COVERAGE_MAXIMIZATION,
            OptimizationObjective.PERFORMANCE_OPTIMIZATION,
            OptimizationObjective.SECURITY_ENHANCEMENT,
            OptimizationObjective.MAINTAINABILITY
        ]
        
        self.weights = weights or ObjectiveWeights()
        self.strategy = strategy
        
        # Optimization parameters
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.convergence_threshold = 1e-6
        
        print("Multi-Objective Optimizer initialized")
        print(f"   Objectives: {[obj.value for obj in self.objectives]}")
        print(f"   Strategy: {strategy.value}")
        print(f"   Population size: {self.population_size}")
    
    def optimize(self, 
                initial_solutions: List[Dict[str, Any]], 
                context: Dict[str, Any]) -> OptimizationResult:
        """Perform multi-objective optimization."""
        
        start_time = time.time()
        print(f"\nMulti-Objective Optimization")
        print(f"   Initial solutions: {len(initial_solutions)}")
        print(f"   Objectives: {len(self.objectives)}")
        
        # Initialize population
        population = self._initialize_population(initial_solutions, context)
        
        # Evaluate initial population
        self._evaluate_population(population, context)
        
        convergence_history = []
        best_fitness_history = []
        
        for generation in range(self.max_generations):
            # Selection, crossover, mutation
            if self.strategy == OptimizationStrategy.NSGA_II:
                population = self._nsga_ii_step(population, context)
            elif self.strategy == OptimizationStrategy.PARETO_OPTIMIZATION:
                population = self._pareto_optimization_step(population, context)
            elif self.strategy == OptimizationStrategy.WEIGHTED_SUM:
                population = self._weighted_sum_step(population, context)
            else:
                population = self._evolutionary_step(population, context)
            
            # Track convergence
            best_fitness = max(candidate.fitness_score for candidate in population)
            best_fitness_history.append(best_fitness)
            
            convergence_info = {
                'generation': generation,
                'best_fitness': best_fitness,
                'population_diversity': self._calculate_diversity(population),
                'pareto_front_size': len([c for c in population if c.pareto_rank == 0])
            }
            convergence_history.append(convergence_info)
            
            # Check convergence
            if len(best_fitness_history) >= 10:
                recent_improvement = (best_fitness_history[-1] - best_fitness_history[-10])
                if recent_improvement < self.convergence_threshold:
                    print(f"   Converged at generation {generation}")
                    break
        
        # Extract results
        pareto_front = [c for c in population if c.pareto_rank == 0]
        best_solutions = sorted(population, key=lambda c: c.fitness_score, reverse=True)[:10]
        
        total_time = time.time() - start_time
        
        print(f"   Optimization completed in {total_time:.2f}s")
        print(f"   Pareto front size: {len(pareto_front)}")
        print(f"   Best fitness: {best_solutions[0].fitness_score:.3f}")
        
        return OptimizationResult(
            best_solutions=best_solutions,
            pareto_front=pareto_front,
            optimization_statistics={
                'final_generation': generation,
                'population_size': len(population),
                'convergence_rate': recent_improvement if 'recent_improvement' in locals() else 0.0,
                'diversity': self._calculate_diversity(population)
            },
            convergence_history=convergence_history,
            total_time=total_time,
            iterations=generation + 1,
            strategy_used=self.strategy
        )
    
    def _initialize_population(self, 
                             initial_solutions: List[Dict[str, Any]], 
                             context: Dict[str, Any]) -> List[OptimizationCandidate]:
        """Initialize optimization population."""
        
        population = []
        
        # Add initial solutions
        for i, solution in enumerate(initial_solutions):
            candidate = OptimizationCandidate(
                solution_id=f"initial_{i}",
                test_plan=solution,
                generation=0
            )
            population.append(candidate)
        
        # Generate additional random solutions to fill population
        while len(population) < self.population_size:
            random_solution = self._generate_random_solution(context)
            candidate = OptimizationCandidate(
                solution_id=f"random_{len(population)}",
                test_plan=random_solution,
                generation=0
            )
            population.append(candidate)
        
        return population
    
    def _evaluate_population(self, population: List[OptimizationCandidate], context: Dict[str, Any]):
        """Evaluate fitness for entire population."""
        
        # Parallel evaluation for performance
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._evaluate_candidate, candidate, context): candidate 
                for candidate in population
            }
            
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    objective_scores, fitness = future.result()
                    candidate.objective_scores = objective_scores
                    candidate.fitness_score = fitness
                except Exception as e:
                    print(f"Error evaluating candidate {candidate.solution_id}: {e}")
                    candidate.fitness_score = 0.0
        
        # Calculate Pareto ranks
        self._calculate_pareto_ranks(population)
    
    def _evaluate_candidate(self, candidate: OptimizationCandidate, context: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        """Evaluate a single candidate solution."""
        
        test_plan = candidate.test_plan
        objective_scores = {}
        
        # Evaluate each objective
        for objective in self.objectives:
            if objective == OptimizationObjective.COVERAGE_MAXIMIZATION:
                score = self._evaluate_coverage(test_plan, context)
            elif objective == OptimizationObjective.PERFORMANCE_OPTIMIZATION:
                score = self._evaluate_performance(test_plan, context)
            elif objective == OptimizationObjective.SECURITY_ENHANCEMENT:
                score = self._evaluate_security(test_plan, context)
            elif objective == OptimizationObjective.MAINTAINABILITY:
                score = self._evaluate_maintainability(test_plan, context)
            elif objective == OptimizationObjective.RESOURCE_EFFICIENCY:
                score = self._evaluate_resource_efficiency(test_plan, context)
            elif objective == OptimizationObjective.EXECUTION_SPEED:
                score = self._evaluate_execution_speed(test_plan, context)
            elif objective == OptimizationObjective.RELIABILITY:
                score = self._evaluate_reliability(test_plan, context)
            elif objective == OptimizationObjective.COMPREHENSIVENESS:
                score = self._evaluate_comprehensiveness(test_plan, context)
            else:
                score = 0.5  # Default score
            
            objective_scores[objective.value] = score
        
        # Calculate weighted fitness
        weights = self.weights.normalize()
        fitness = (
            objective_scores.get('coverage_maximization', 0) * weights.coverage +
            objective_scores.get('performance_optimization', 0) * weights.performance +
            objective_scores.get('security_enhancement', 0) * weights.security +
            objective_scores.get('maintainability', 0) * weights.maintainability +
            objective_scores.get('resource_efficiency', 0) * weights.resource_efficiency +
            objective_scores.get('execution_speed', 0) * weights.execution_speed +
            objective_scores.get('reliability', 0) * weights.reliability +
            objective_scores.get('comprehensiveness', 0) * weights.comprehensiveness
        )
        
        return objective_scores, fitness
    
    def _evaluate_coverage(self, test_plan: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate coverage objective."""
        test_scenarios = test_plan.get('test_scenarios', [])
        expected_coverage = test_plan.get('expected_coverage', 0.5)
        
        # Score based on number of scenarios and expected coverage
        scenario_score = min(1.0, len(test_scenarios) / 20.0)  # Normalize to max 20 scenarios
        coverage_score = expected_coverage
        
        return (scenario_score + coverage_score) / 2.0
    
    def _evaluate_performance(self, test_plan: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate performance objective."""
        estimated_time = test_plan.get('estimated_time', 60.0)
        test_scenarios = test_plan.get('test_scenarios', [])
        
        # Prefer shorter execution times but sufficient coverage
        time_score = max(0.1, 1.0 - (estimated_time / 120.0))  # Normalize to 2 hour max
        efficiency_score = len(test_scenarios) / max(estimated_time / 60.0, 1.0)  # scenarios per minute
        efficiency_score = min(1.0, efficiency_score / 10.0)  # Normalize
        
        return (time_score + efficiency_score) / 2.0
    
    def _evaluate_security(self, test_plan: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate security objective."""
        security_plan = test_plan.get('security_plan', {})
        target_vulnerabilities = test_plan.get('target_vulnerabilities', [])
        
        # Score based on security coverage
        vuln_count_score = min(1.0, len(target_vulnerabilities) / 10.0)
        
        # Check if security-focused strategy
        strategy = test_plan.get('strategy', '')
        strategy_score = 1.0 if 'security' in strategy else 0.5
        
        return (vuln_count_score + strategy_score) / 2.0
    
    def _evaluate_maintainability(self, test_plan: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate maintainability objective."""
        test_scenarios = test_plan.get('test_scenarios', [])
        
        # Score based on test organization and clarity
        scenario_complexity = sum(
            len(scenario.get('description', '').split()) 
            for scenario in test_scenarios
        ) / max(len(test_scenarios), 1)
        
        # Prefer moderate complexity (not too simple, not too complex)
        optimal_complexity = 15  # words per description
        complexity_score = 1.0 - abs(scenario_complexity - optimal_complexity) / optimal_complexity
        complexity_score = max(0.1, complexity_score)
        
        # Prefer organized test structure
        has_clear_structure = bool(test_plan.get('strategy'))
        structure_score = 1.0 if has_clear_structure else 0.5
        
        return (complexity_score + structure_score) / 2.0
    
    def _evaluate_resource_efficiency(self, test_plan: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate resource efficiency objective."""
        estimated_time = test_plan.get('estimated_time', 60.0)
        expected_coverage = test_plan.get('expected_coverage', 0.5)
        
        # Efficiency = coverage per unit time
        efficiency = expected_coverage / max(estimated_time / 60.0, 0.1)  # coverage per minute
        efficiency_score = min(1.0, efficiency / 2.0)  # Normalize to max 2.0 coverage/minute
        
        return efficiency_score
    
    def _evaluate_execution_speed(self, test_plan: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate execution speed objective."""
        estimated_time = test_plan.get('estimated_time', 60.0)
        
        # Prefer faster execution
        speed_score = max(0.1, 1.0 - (estimated_time / 300.0))  # Normalize to 5 minute max
        return speed_score
    
    def _evaluate_reliability(self, test_plan: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate reliability objective."""
        priority_score = test_plan.get('priority_score', 0.5)
        expected_coverage = test_plan.get('expected_coverage', 0.5)
        
        # Combine priority and coverage as reliability indicators
        return (priority_score + expected_coverage) / 2.0
    
    def _evaluate_comprehensiveness(self, test_plan: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate comprehensiveness objective."""
        test_scenarios = test_plan.get('test_scenarios', [])
        target_vulnerabilities = test_plan.get('target_vulnerabilities', [])
        
        # Score based on breadth of testing
        scenario_breadth = min(1.0, len(test_scenarios) / 15.0)
        vulnerability_breadth = min(1.0, len(target_vulnerabilities) / 8.0)
        
        return (scenario_breadth + vulnerability_breadth) / 2.0
    
    def _nsga_ii_step(self, population: List[OptimizationCandidate], context: Dict[str, Any]) -> List[OptimizationCandidate]:
        """NSGA-II optimization step."""
        
        # Calculate crowding distances
        self._calculate_crowding_distances(population)
        
        # Generate offspring through selection, crossover, mutation
        offspring = []
        
        for _ in range(self.population_size):
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if statistics.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2, context)
            else:
                child = parent1
            
            # Mutation
            if statistics.random() < self.mutation_rate:
                child = self._mutate(child, context)
            
            offspring.append(child)
        
        # Combine parent and offspring populations
        combined_population = population + offspring
        
        # Evaluate new candidates
        new_candidates = [c for c in offspring if not c.objective_scores]
        if new_candidates:
            for candidate in new_candidates:
                objective_scores, fitness = self._evaluate_candidate(candidate, context)
                candidate.objective_scores = objective_scores
                candidate.fitness_score = fitness
        
        # Calculate Pareto ranks for combined population
        self._calculate_pareto_ranks(combined_population)
        self._calculate_crowding_distances(combined_population)
        
        # Select next generation using NSGA-II selection
        next_generation = self._nsga_ii_selection(combined_population, self.population_size)
        
        return next_generation
    
    def _calculate_pareto_ranks(self, population: List[OptimizationCandidate]):
        """Calculate Pareto ranks for population."""
        
        # Initialize domination counts and dominated sets
        for candidate in population:
            candidate.pareto_rank = 0
            candidate.metadata['dominated_solutions'] = set()
            candidate.metadata['domination_count'] = 0
        
        # Calculate domination relationships
        fronts = [[] for _ in range(len(population))]
        
        for i, candidate_i in enumerate(population):
            for j, candidate_j in enumerate(population):
                if i != j:
                    if self._dominates(candidate_i, candidate_j):
                        candidate_i.metadata['dominated_solutions'].add(j)
                    elif self._dominates(candidate_j, candidate_i):
                        candidate_i.metadata['domination_count'] += 1
            
            if candidate_i.metadata['domination_count'] == 0:
                candidate_i.pareto_rank = 0
                fronts[0].append(i)
        
        # Build subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            for i in fronts[front_index]:
                candidate_i = population[i]
                for j in candidate_i.metadata['dominated_solutions']:
                    candidate_j = population[j]
                    candidate_j.metadata['domination_count'] -= 1
                    if candidate_j.metadata['domination_count'] == 0:
                        candidate_j.pareto_rank = front_index + 1
                        next_front.append(j)
            
            front_index += 1
            if next_front:
                fronts[front_index] = next_front
            else:
                break
    
    def _dominates(self, candidate_a: OptimizationCandidate, candidate_b: OptimizationCandidate) -> bool:
        """Check if candidate A dominates candidate B."""
        
        at_least_one_better = False
        
        for objective in self.objectives:
            score_a = candidate_a.objective_scores.get(objective.value, 0)
            score_b = candidate_b.objective_scores.get(objective.value, 0)
            
            if score_a < score_b:
                return False  # A is worse in at least one objective
            elif score_a > score_b:
                at_least_one_better = True
        
        return at_least_one_better
    
    def _calculate_crowding_distances(self, population: List[OptimizationCandidate]):
        """Calculate crowding distances for diversity preservation."""
        
        # Initialize distances
        for candidate in population:
            candidate.crowding_distance = 0.0
        
        # Group by Pareto rank
        fronts = {}
        for candidate in population:
            rank = candidate.pareto_rank
            if rank not in fronts:
                fronts[rank] = []
            fronts[rank].append(candidate)
        
        # Calculate crowding distance for each front
        for front in fronts.values():
            if len(front) <= 2:
                for candidate in front:
                    candidate.crowding_distance = float('inf')
                continue
            
            # For each objective
            for objective in self.objectives:
                # Sort by objective value
                front.sort(key=lambda c: c.objective_scores.get(objective.value, 0))
                
                # Set boundary solutions to infinite distance
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                
                # Calculate distance for intermediate solutions
                objective_range = (front[-1].objective_scores.get(objective.value, 0) - 
                                 front[0].objective_scores.get(objective.value, 0))
                
                if objective_range > 0:
                    for i in range(1, len(front) - 1):
                        distance = (front[i + 1].objective_scores.get(objective.value, 0) - 
                                  front[i - 1].objective_scores.get(objective.value, 0)) / objective_range
                        front[i].crowding_distance += distance
    
    def _tournament_selection(self, population: List[OptimizationCandidate]) -> OptimizationCandidate:
        """Tournament selection for parent selection."""
        import random
        
        tournament_size = 3
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Select best candidate from tournament
        best_candidate = tournament[0]
        for candidate in tournament[1:]:
            if (candidate.pareto_rank < best_candidate.pareto_rank or
                (candidate.pareto_rank == best_candidate.pareto_rank and 
                 candidate.crowding_distance > best_candidate.crowding_distance)):
                best_candidate = candidate
        
        return best_candidate
    
    def _crossover(self, parent1: OptimizationCandidate, parent2: OptimizationCandidate, 
                  context: Dict[str, Any]) -> OptimizationCandidate:
        """Crossover operation to create offspring."""
        
        # Simple blend crossover for test plans
        child_plan = parent1.test_plan.copy()
        
        # Blend numeric values
        if 'estimated_time' in parent1.test_plan and 'estimated_time' in parent2.test_plan:
            child_plan['estimated_time'] = (parent1.test_plan['estimated_time'] + 
                                          parent2.test_plan['estimated_time']) / 2.0
        
        if 'expected_coverage' in parent1.test_plan and 'expected_coverage' in parent2.test_plan:
            child_plan['expected_coverage'] = (parent1.test_plan['expected_coverage'] + 
                                             parent2.test_plan['expected_coverage']) / 2.0
        
        # Combine test scenarios
        scenarios1 = parent1.test_plan.get('test_scenarios', [])
        scenarios2 = parent2.test_plan.get('test_scenarios', [])
        child_plan['test_scenarios'] = scenarios1[:len(scenarios1)//2] + scenarios2[len(scenarios2)//2:]
        
        child = OptimizationCandidate(
            solution_id=f"child_{parent1.solution_id}_{parent2.solution_id}",
            test_plan=child_plan,
            generation=max(parent1.generation, parent2.generation) + 1
        )
        
        return child
    
    def _mutate(self, candidate: OptimizationCandidate, context: Dict[str, Any]) -> OptimizationCandidate:
        """Mutation operation."""
        import random
        
        mutated_plan = candidate.test_plan.copy()
        
        # Mutate estimated time
        if 'estimated_time' in mutated_plan:
            factor = random.uniform(0.8, 1.2)
            mutated_plan['estimated_time'] = mutated_plan['estimated_time'] * factor
        
        # Mutate expected coverage
        if 'expected_coverage' in mutated_plan:
            delta = random.uniform(-0.1, 0.1)
            mutated_plan['expected_coverage'] = max(0.1, min(1.0, 
                mutated_plan['expected_coverage'] + delta))
        
        mutated_candidate = OptimizationCandidate(
            solution_id=f"mutated_{candidate.solution_id}",
            test_plan=mutated_plan,
            generation=candidate.generation + 1
        )
        
        return mutated_candidate
    
    def _nsga_ii_selection(self, population: List[OptimizationCandidate], target_size: int) -> List[OptimizationCandidate]:
        """NSGA-II selection for next generation."""
        
        # Sort by Pareto rank and crowding distance
        population.sort(key=lambda c: (c.pareto_rank, -c.crowding_distance))
        
        return population[:target_size]
    
    def _pareto_optimization_step(self, population: List[OptimizationCandidate], context: Dict[str, Any]) -> List[OptimizationCandidate]:
        """Simple Pareto optimization step."""
        # For now, use NSGA-II as implementation
        return self._nsga_ii_step(population, context)
    
    def _weighted_sum_step(self, population: List[OptimizationCandidate], context: Dict[str, Any]) -> List[OptimizationCandidate]:
        """Weighted sum optimization step."""
        # Sort by fitness score and select top performers
        population.sort(key=lambda c: c.fitness_score, reverse=True)
        
        # Keep top 50% and generate new solutions
        elite_size = len(population) // 2
        elite = population[:elite_size]
        
        # Generate new solutions
        new_solutions = []
        while len(elite) + len(new_solutions) < self.population_size:
            new_solution = self._generate_random_solution(context)
            candidate = OptimizationCandidate(
                solution_id=f"new_{len(new_solutions)}",
                test_plan=new_solution,
                generation=max(c.generation for c in elite) + 1
            )
            new_solutions.append(candidate)
        
        # Evaluate new solutions
        for candidate in new_solutions:
            objective_scores, fitness = self._evaluate_candidate(candidate, context)
            candidate.objective_scores = objective_scores
            candidate.fitness_score = fitness
        
        return elite + new_solutions
    
    def _evolutionary_step(self, population: List[OptimizationCandidate], context: Dict[str, Any]) -> List[OptimizationCandidate]:
        """Basic evolutionary optimization step."""
        return self._weighted_sum_step(population, context)
    
    def _generate_random_solution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a random test plan solution."""
        import random
        
        strategies = ['comprehensive', 'basic', 'security_focused', 'performance_optimized']
        
        return {
            'strategy': random.choice(strategies),
            'estimated_time': random.uniform(15.0, 120.0),
            'expected_coverage': random.uniform(0.3, 0.95),
            'priority_score': random.uniform(0.3, 0.9),
            'test_scenarios': [
                {'test': f'scenario_{i}', 'description': f'Test scenario {i}'}
                for i in range(random.randint(3, 15))
            ]
        }
    
    def _calculate_diversity(self, population: List[OptimizationCandidate]) -> float:
        """Calculate population diversity metric."""
        if len(population) < 2:
            return 0.0
        
        fitness_values = [c.fitness_score for c in population]
        return statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0.0


class OptimizationPlanGenerator(TestPlanGenerator):
    """Generates optimization-focused test plans."""
    
    def __init__(self, optimizer: MultiObjectiveOptimizer = None):
        self.optimizer = optimizer or MultiObjectiveOptimizer()
        print("Optimization Plan Generator initialized")
    
    def generate(self, parent_node: PlanningNode, context: Dict[str, Any]) -> List[PlanningNode]:
        """Generate optimization-focused test plans."""
        
        module_path = context.get('module_path', '')
        
        # Create base solutions for optimization
        base_solutions = self._create_base_solutions(context)
        
        # Optimize solutions
        optimization_result = self.optimizer.optimize(base_solutions, context)
        
        children = []
        for i, solution in enumerate(optimization_result.best_solutions[:5]):  # Top 5 solutions
            child_node = PlanningNode(
                id=f"{parent_node.id}_optimized_{i}",
                content={
                    'optimization_result': solution.test_plan,
                    'fitness_score': solution.fitness_score,
                    'objective_scores': solution.objective_scores,
                    'pareto_rank': solution.pareto_rank,
                    'optimization_strategy': optimization_result.strategy_used.value,
                    'generation': solution.generation
                },
                depth=parent_node.depth + 1
            )
            children.append(child_node)
        
        return children
    
    def _create_base_solutions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create base solutions for optimization."""
        
        return [
            {  # Comprehensive strategy
                'strategy': 'comprehensive',
                'estimated_time': 90.0,
                'expected_coverage': 0.9,
                'priority_score': 0.8,
                'test_scenarios': [
                    {'test': 'unit_tests', 'description': 'Comprehensive unit testing'},
                    {'test': 'integration_tests', 'description': 'Integration test coverage'},
                    {'test': 'edge_cases', 'description': 'Edge case validation'},
                    {'test': 'error_handling', 'description': 'Error condition testing'}
                ]
            },
            {  # Performance-focused strategy
                'strategy': 'performance_optimized',
                'estimated_time': 45.0,
                'expected_coverage': 0.7,
                'priority_score': 0.9,
                'test_scenarios': [
                    {'test': 'performance_tests', 'description': 'Performance benchmarking'},
                    {'test': 'load_tests', 'description': 'Load testing'},
                    {'test': 'stress_tests', 'description': 'Stress testing'}
                ]
            },
            {  # Security-focused strategy
                'strategy': 'security_focused',
                'estimated_time': 60.0,
                'expected_coverage': 0.8,
                'priority_score': 0.95,
                'test_scenarios': [
                    {'test': 'vulnerability_tests', 'description': 'Security vulnerability testing'},
                    {'test': 'penetration_tests', 'description': 'Penetration testing'},
                    {'test': 'compliance_tests', 'description': 'Compliance validation'}
                ]
            },
            {  # Basic strategy
                'strategy': 'basic',
                'estimated_time': 30.0,
                'expected_coverage': 0.6,
                'priority_score': 0.6,
                'test_scenarios': [
                    {'test': 'basic_functionality', 'description': 'Basic functionality tests'},
                    {'test': 'happy_path', 'description': 'Happy path testing'}
                ]
            }
        ]


class OptimizationPlanEvaluator(TestPlanEvaluator):
    """Evaluates optimization-focused test plans."""
    
    def __init__(self):
        self.evaluation_weights = {
            'fitness_score': 0.4,
            'pareto_optimality': 0.3,
            'objective_balance': 0.2,
            'feasibility': 0.1
        }
        print("Optimization Plan Evaluator initialized")
    
    def evaluate(self, node: PlanningNode, criteria: List[EvaluationCriteria]) -> float:
        """Evaluate optimization plan."""
        
        content = node.content
        
        # Get optimization metrics
        fitness_score = content.get('fitness_score', 0.0)
        objective_scores = content.get('objective_scores', {})
        pareto_rank = content.get('pareto_rank', float('inf'))
        
        # Evaluate components
        fitness_component = fitness_score
        pareto_component = 1.0 / (1.0 + pareto_rank)  # Lower rank = better
        
        # Objective balance (how well balanced across objectives)
        if objective_scores:
            scores = list(objective_scores.values())
            balance_component = 1.0 - (max(scores) - min(scores)) if scores else 0.5
        else:
            balance_component = 0.5
        
        # Feasibility (based on estimated time and resources)
        optimization_result = content.get('optimization_result', {})
        estimated_time = optimization_result.get('estimated_time', 60.0)
        feasibility_component = max(0.1, 1.0 - (estimated_time / 180.0))  # 3 hour max
        
        # Calculate weighted score
        aggregate_score = (
            fitness_component * self.evaluation_weights['fitness_score'] +
            pareto_component * self.evaluation_weights['pareto_optimality'] +
            balance_component * self.evaluation_weights['objective_balance'] +
            feasibility_component * self.evaluation_weights['feasibility']
        )
        
        # Store individual scores
        node.update_score('fitness_score', fitness_component)
        node.update_score('pareto_optimality', pareto_component)
        node.update_score('objective_balance', balance_component)
        node.update_score('feasibility', feasibility_component)
        
        return aggregate_score


class MultiObjectiveOptimizationAgent:
    """Main multi-objective optimization agent."""
    
    def __init__(self, 
                 coordinator: AgentCoordinator = None,
                 security_agent: SecurityIntelligenceAgent = None):
        
        self.coordinator = coordinator
        self.security_agent = security_agent
        self.shared_state = get_shared_state()
        
        # Initialize components
        self.optimizer = MultiObjectiveOptimizer()
        self.plan_generator = OptimizationPlanGenerator(self.optimizer)
        self.plan_evaluator = OptimizationPlanEvaluator()
        
        # Register with coordinator if provided
        if self.coordinator:
            self.coordinator.register_agent(
                "multi_objective_optimization_agent",
                AgentRole.PERFORMANCE_OPTIMIZER,
                weight=1.3,  # Higher weight for optimization expertise
                specialization=["multi_objective_optimization", "performance_tuning", "resource_efficiency"]
            )
        
        print("Multi-Objective Optimization Agent initialized")
        print("   Components: optimizer, planner, evaluator")
    
    def optimize_test_generation(self, 
                                module_path: str, 
                                context: Dict[str, Any] = None,
                                objectives: List[OptimizationObjective] = None,
                                weights: ObjectiveWeights = None) -> Dict[str, Any]:
        """Optimize test generation for multiple objectives."""
        
        context = context or {}
        print(f"\nMulti-Objective Optimization: {module_path}")
        
        # Check cache first
        cached_result = get_cached_test_result(f"optimization_{module_path}")
        if cached_result:
            print("Using cached optimization result")
            return cached_result
        
        start_time = time.time()
        
        # Update optimizer configuration if provided
        if objectives:
            self.optimizer.objectives = objectives
        if weights:
            self.optimizer.weights = weights
        
        # Get security analysis if security agent available
        security_context = {}
        if self.security_agent:
            try:
                security_analysis = self.security_agent.analyze_security(module_path, context)
                security_context = {
                    'security_findings': security_analysis.get('findings', []),
                    'vulnerability_count': security_analysis.get('vulnerability_count', 0),
                    'compliance_score': security_analysis.get('compliance_report', {}).get('compliance_score', 0.5)
                }
                context.update(security_context)
            except Exception as e:
                print(f"Security analysis failed: {e}")
        
        # Generate optimized test plan using hierarchical planning
        optimization_plan = self._generate_optimization_plan(module_path, context)
        
        # Coordinate with other agents for consensus if coordinator available
        consensus_result = None
        if self.coordinator:
            consensus_result = self._coordinate_optimization_consensus(
                module_path, optimization_plan, context
            )
        
        optimization_time = time.time() - start_time
        
        result = {
            'module_path': module_path,
            'optimization_plan': optimization_plan,
            'consensus_result': consensus_result,
            'security_context': security_context,
            'optimization_time': optimization_time,
            'objectives_used': [obj.value for obj in self.optimizer.objectives],
            'strategy_used': self.optimizer.strategy.value,
            'recommendations': self._generate_optimization_recommendations(optimization_plan),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        cache_test_result(f"optimization_{module_path}", result, 90.0)
        
        print(f"   Optimization completed in {optimization_time:.2f}s")
        if optimization_plan:
            best_solution = optimization_plan.get('best_solutions', [{}])[0]
            print(f"   Best fitness: {best_solution.get('fitness_score', 0):.3f}")
        
        return result
    
    def _generate_optimization_plan(self, module_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization plan using hierarchical planning."""
        
        try:
            # Use hierarchical planner for optimization planning
            planner = get_best_planner(prefer_llm=False)  # Use template-based for optimization
            planner.plan_generator = self.plan_generator
            planner.plan_evaluator = self.plan_evaluator
            
            # Create planning context
            planning_context = {
                'module_path': module_path,
                'optimization_objectives': [obj.value for obj in self.optimizer.objectives],
                'objective_weights': self.optimizer.weights.__dict__,
                **context
            }
            
            # Initial optimization request
            initial_plan = {
                'objective': 'multi_objective_test_optimization',
                'module_path': module_path,
                'optimization_strategy': self.optimizer.strategy.value
            }
            
            # Execute planning
            planning_tree = planner.plan(initial_plan, planning_context)
            best_plan = planning_tree.get_best_plan()
            
            if best_plan:
                return {
                    'best_solutions': [node.content for node in best_plan],
                    'planning_tree_size': len(planning_tree.nodes),
                    'planning_time': planning_tree.metadata.get('planning_time', 0)
                }
            else:
                return self._create_fallback_optimization_plan()
                
        except Exception as e:
            print(f"Optimization planning failed: {e}")
            return self._create_fallback_optimization_plan()
    
    def _create_fallback_optimization_plan(self) -> Dict[str, Any]:
        """Create fallback optimization plan."""
        
        # Create simple optimization result
        base_solutions = [
            {
                'strategy': 'balanced',
                'estimated_time': 60.0,
                'expected_coverage': 0.75,
                'priority_score': 0.7,
                'fitness_score': 0.7,
                'objective_scores': {
                    'coverage_maximization': 0.75,
                    'performance_optimization': 0.7,
                    'security_enhancement': 0.6,
                    'maintainability': 0.8
                }
            }
        ]
        
        return {
            'best_solutions': base_solutions,
            'fallback': True,
            'message': 'Using fallback optimization plan'
        }
    
    def _coordinate_optimization_consensus(self, 
                                         module_path: str, 
                                         optimization_plan: Dict[str, Any], 
                                         context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Coordinate with other agents for optimization consensus."""
        
        if not self.coordinator:
            return None
        
        try:
            # Create coordination task
            task_id = self.coordinator.create_coordination_task(
                description=f"Optimize test generation for {module_path}",
                required_roles={AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.QUALITY_ASSESSOR},
                context={
                    'module_path': module_path,
                    'optimization_plan': optimization_plan,
                    **context
                }
            )
            
            # Submit optimization assessment vote
            best_solution = optimization_plan.get('best_solutions', [{}])[0]
            optimization_score = best_solution.get('fitness_score', 0.5)
            
            self.coordinator.submit_vote(
                task_id=task_id,
                agent_id="multi_objective_optimization_agent",
                choice=optimization_score,
                confidence=0.9,
                reasoning="Multi-objective optimization analysis with Pareto front evaluation"
            )
            
            # Wait for consensus (in real implementation, this would be event-driven)
            time.sleep(2)
            
            result = self.coordinator.get_coordination_result(task_id)
            return result.to_dict() if result else None
            
        except Exception as e:
            print(f"Optimization consensus coordination failed: {e}")
            return None
    
    def _generate_optimization_recommendations(self, optimization_plan: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        best_solutions = optimization_plan.get('best_solutions', [])
        if not best_solutions:
            recommendations.append("Unable to generate optimized solutions - consider simpler objectives")
            return recommendations
        
        best_solution = best_solutions[0]
        objective_scores = best_solution.get('objective_scores', {})
        
        # Analyze objective scores for recommendations
        coverage_score = objective_scores.get('coverage_maximization', 0)
        performance_score = objective_scores.get('performance_optimization', 0)
        security_score = objective_scores.get('security_enhancement', 0)
        maintainability_score = objective_scores.get('maintainability', 0)
        
        if coverage_score < 0.7:
            recommendations.append("Consider increasing test coverage focus for better quality assurance")
        
        if performance_score < 0.6:
            recommendations.append("Optimize test execution time and resource usage")
        
        if security_score < 0.7:
            recommendations.append("Enhance security testing coverage and vulnerability detection")
        
        if maintainability_score < 0.6:
            recommendations.append("Improve test organization and documentation for better maintainability")
        
        # Strategy-specific recommendations
        strategy = best_solution.get('strategy', '')
        if strategy == 'comprehensive':
            recommendations.append("Comprehensive strategy selected - ensure adequate time allocation")
        elif strategy == 'basic':
            recommendations.append("Basic strategy may miss edge cases - consider hybrid approach")
        
        if not recommendations:
            recommendations.append("Optimization appears well-balanced across all objectives")
        
        return recommendations


def test_multi_objective_optimization():
    """Test the multi-objective optimization agent."""
    print("\n" + "="*60)
    print("Testing Multi-Objective Optimization Agent")
    print("="*60)
    
    # Create test context
    context = {
        'module_path': 'test_module.py',
        'module_analysis': {
            'complexity': 12,
            'function_count': 8,
            'class_count': 2
        },
        'performance_requirements': {
            'max_execution_time': 60.0,
            'target_coverage': 0.85
        }
    }
    
    # Create optimization agent
    agent = MultiObjectiveOptimizationAgent()
    
    # Test optimization
    print("\n1. Running multi-objective optimization...")
    result = agent.optimize_test_generation('test_module.py', context)
    
    print(f"\n2. Optimization Results:")
    print(f"   Optimization time: {result['optimization_time']:.2f}s")
    print(f"   Strategy used: {result['strategy_used']}")
    print(f"   Objectives: {', '.join(result['objectives_used'])}")
    
    # Show optimization plan
    optimization_plan = result.get('optimization_plan', {})
    best_solutions = optimization_plan.get('best_solutions', [])
    
    if best_solutions:
        print(f"\n3. Best Solution:")
        best = best_solutions[0]
        print(f"   Fitness score: {best.get('fitness_score', 0):.3f}")
        
        objective_scores = best.get('objective_scores', {})
        if objective_scores:
            print(f"   Objective scores:")
            for obj, score in objective_scores.items():
                print(f"     {obj}: {score:.3f}")
    
    # Show recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        print(f"\n4. Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    print("\nMulti-Objective Optimization Agent test completed successfully!")
    return True


if __name__ == "__main__":
    test_multi_objective_optimization()