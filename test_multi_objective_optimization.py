"""
Test Multi-Objective Universal Optimization System
"""

import sys
from pathlib import Path
import numpy as np
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from testmaster.intelligence.optimization import (
    MultiObjectiveOptimizer,
    OptimizationObjective,
    ObjectiveType,
    OptimizationConfig,
    Solution,
    NSGAIIOptimizer,
    CoverageObjective,
    QualityObjective,
    PerformanceObjective,
    SecurityObjective,
    MaintainabilityObjective,
    BalancedTestObjective
)
from testmaster.core.ast_abstraction import UniversalASTAbstractor
from testmaster.core.framework_abstraction import UniversalTestSuite, UniversalTestCase, UniversalTest, TestMetadata


def create_test_suite_solution(id: str, num_tests: int = 10) -> Solution:
    """Create a solution with a test suite."""
    # Create test suite
    test_suite = UniversalTestSuite(
        name=f"TestSuite_{id}",
        metadata=TestMetadata(
            tags=["optimization", "test"],
            category="unit"
        )
    )
    
    # Add test cases
    test_case = UniversalTestCase(
        name=f"TestCase_{id}",
        description="Optimized test case"
    )
    
    # Add tests
    for i in range(num_tests):
        test = UniversalTest(
            name=f"test_{id}_{i}",
            test_function=f"result = function_{i}()",
            description=f"Test {i}"
        )
        
        # Add random assertions
        from testmaster.core.framework_abstraction import TestAssertion, AssertionType
        for _ in range(random.randint(1, 5)):
            test.add_assertion(TestAssertion(
                assertion_type=random.choice(list(AssertionType)),
                actual="result"
            ))
        
        # Randomly add security tag
        if random.random() < 0.3:
            test.metadata.tags.append("security")
        
        test_case.add_test(test)
    
    test_suite.add_test_case(test_case)
    
    # Create solution
    solution = Solution(
        id=id,
        genes=[random.random() for _ in range(6)]  # 6 genes for 6 objectives
    )
    solution.test_suite = test_suite
    
    return solution


def test_basic_optimization():
    """Test basic multi-objective optimization."""
    print("\n" + "="*80)
    print("Testing Basic Multi-Objective Optimization")
    print("="*80)
    
    # Define objectives
    objectives = [
        OptimizationObjective(
            name="maximize_test",
            type=ObjectiveType.MAXIMIZE,
            weight=2.0,
            min_value=0.0,
            max_value=100.0,
            evaluator=lambda s: s.genes[0] * 100 if hasattr(s, 'genes') else 0
        ),
        OptimizationObjective(
            name="minimize_cost",
            type=ObjectiveType.MINIMIZE,
            weight=1.0,
            min_value=0.0,
            max_value=100.0,
            evaluator=lambda s: s.genes[1] * 100 if hasattr(s, 'genes') and len(s.genes) > 1 else 50
        ),
        OptimizationObjective(
            name="target_quality",
            type=ObjectiveType.TARGET,
            weight=1.5,
            target_value=75.0,
            min_value=0.0,
            max_value=100.0,
            evaluator=lambda s: s.genes[2] * 100 if hasattr(s, 'genes') and len(s.genes) > 2 else 50
        )
    ]
    
    # Configure optimization
    config = OptimizationConfig(
        population_size=20,
        max_generations=30,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # Create optimizer
    optimizer = MultiObjectiveOptimizer(objectives, config)
    
    # Run optimization
    result = optimizer.optimize()
    
    # Display results
    print(f"\nOptimization Results:")
    print(f"   Pareto front size: {len(result.pareto_front.solutions)}")
    print(f"   Generations run: {result.generations_run}")
    print(f"   Convergence achieved: {result.convergence_achieved}")
    print(f"   Average fitness: {result.average_fitness:.3f}")
    print(f"   Fitness improvement: {result.fitness_improvement:.3f}")
    print(f"   Diversity score: {result.diversity_score:.3f}")
    
    # Show best solutions
    print(f"\nBest Solutions by Objective:")
    for obj_name, solution in result.best_solutions.items():
        value = solution.objectives.get(obj_name, 0)
        print(f"   {obj_name}: {value:.2f} (fitness: {solution.fitness:.3f})")
    
    # Show balanced solution
    if result.balanced_solution:
        print(f"\nBalanced Solution:")
        for obj_name, value in result.balanced_solution.objectives.items():
            print(f"   {obj_name}: {value:.2f}")
        print(f"   Overall fitness: {result.balanced_solution.fitness:.3f}")
    
    print("\n[SUCCESS] Basic optimization test completed!")
    return result


def test_nsga2_optimization():
    """Test NSGA-II optimization algorithm."""
    print("\n" + "="*80)
    print("Testing NSGA-II Multi-Objective Optimization")
    print("="*80)
    
    # Build AST for a small project
    print("\n1. Building Universal AST...")
    ast_builder = UniversalASTAbstractor()
    project_path = Path(__file__).parent / "testmaster" / "intelligence"
    universal_ast = ast_builder.create_project_ast(str(project_path))
    print(f"   AST built: {universal_ast.total_functions} functions")
    
    # Create test objectives
    print("\n2. Creating test optimization objectives...")
    objectives = [
        CoverageObjective(target_coverage=80.0, universal_ast=universal_ast),
        QualityObjective(min_quality_score=70.0),
        PerformanceObjective(max_time_ms=5000.0),
        SecurityObjective(min_security_tests=5)
    ]
    
    # Configure NSGA-II
    config = OptimizationConfig(
        population_size=30,
        max_generations=20,
        algorithm="nsga2"
    )
    
    # Create NSGA-II optimizer
    print("\n3. Running NSGA-II optimization...")
    optimizer = NSGAIIOptimizer(objectives, config)
    
    # Create initial population with test suites
    initial_solutions = [
        create_test_suite_solution(f"sol_{i}", num_tests=random.randint(5, 20))
        for i in range(config.population_size)
    ]
    
    # Run optimization
    result = optimizer.optimize(initial_solutions)
    
    # Display results
    print(f"\nNSGA-II Results:")
    print(f"   Pareto front size: {len(result.pareto_front.solutions)}")
    print(f"   Generations run: {result.generations_run}")
    print(f"   Hypervolume: {result.hypervolume:.3f}")
    
    # Analyze Pareto front
    print(f"\nPareto Front Analysis:")
    for i, solution in enumerate(result.pareto_front.solutions[:5]):  # Show top 5
        print(f"\n   Solution {i+1} (Rank {solution.rank}):")
        print(f"      Coverage: {solution.objectives.get('coverage', 0):.1f}%")
        print(f"      Quality: {solution.objectives.get('quality', 0):.1f}")
        print(f"      Performance: {solution.objectives.get('performance', 0):.1f}ms")
        print(f"      Security: {solution.objectives.get('security', 0):.1f}%")
        print(f"      Crowding distance: {solution.crowding_distance:.3f}")
    
    print("\n[SUCCESS] NSGA-II optimization test completed!")
    return result


def test_balanced_optimization():
    """Test optimization with balanced compound objective."""
    print("\n" + "="*80)
    print("Testing Balanced Multi-Objective Optimization")
    print("="*80)
    
    # Build AST
    print("\n1. Building Universal AST...")
    ast_builder = UniversalASTAbstractor()
    project_path = Path(__file__).parent / "testmaster" / "core"
    universal_ast = ast_builder.create_project_ast(str(project_path))
    
    # Create balanced objective
    print("\n2. Creating balanced objective...")
    balanced_objective = BalancedTestObjective(universal_ast)
    
    # Show sub-objectives
    print(f"   Compound objective with {len(balanced_objective.sub_objectives)} components:")
    for obj in balanced_objective.sub_objectives:
        print(f"      - {obj.name} (weight: {obj.weight})")
    
    # Configure optimization
    config = OptimizationConfig(
        population_size=25,
        max_generations=15,
        maintain_diversity=True,
        diversity_threshold=0.1
    )
    
    # Run optimization
    print("\n3. Running balanced optimization...")
    optimizer = MultiObjectiveOptimizer([balanced_objective], config)
    
    # Create initial solutions
    initial_solutions = [
        create_test_suite_solution(f"balanced_{i}", num_tests=random.randint(10, 30))
        for i in range(config.population_size)
    ]
    
    result = optimizer.optimize(initial_solutions)
    
    # Display results
    print(f"\nBalanced Optimization Results:")
    print(f"   Final fitness: {result.average_fitness:.3f}")
    print(f"   Improvement: {result.fitness_improvement:.3f}")
    print(f"   Diversity maintained: {result.diversity_score:.3f}")
    
    # Show best solution
    if result.balanced_solution:
        print(f"\nBest Balanced Solution:")
        print(f"   Compound score: {result.balanced_solution.objectives.get('balanced_testing', 0):.2f}")
        print(f"   Overall fitness: {result.balanced_solution.fitness:.3f}")
        
        # Analyze test suite
        if hasattr(result.balanced_solution, 'test_suite'):
            suite = result.balanced_solution.test_suite
            print(f"\n   Test Suite Analysis:")
            print(f"      Test cases: {len(suite.test_cases)}")
            print(f"      Total tests: {suite.count_tests()}")
            print(f"      Total assertions: {suite.count_assertions()}")
    
    print("\n[SUCCESS] Balanced optimization test completed!")
    return result


def compare_algorithms():
    """Compare different optimization algorithms."""
    print("\n" + "="*80)
    print("Comparing Optimization Algorithms")
    print("="*80)
    
    # Simple objectives for comparison
    objectives = [
        OptimizationObjective(
            name="obj1",
            type=ObjectiveType.MAXIMIZE,
            weight=1.0,
            evaluator=lambda s: s.genes[0] * 100 if hasattr(s, 'genes') else 0
        ),
        OptimizationObjective(
            name="obj2",
            type=ObjectiveType.MAXIMIZE,
            weight=1.0,
            evaluator=lambda s: s.genes[1] * 100 if hasattr(s, 'genes') and len(s.genes) > 1 else 0
        )
    ]
    
    # Test different algorithms
    algorithms = ["genetic", "nsga2"]
    results = {}
    
    for algo in algorithms:
        print(f"\nTesting {algo}...")
        
        config = OptimizationConfig(
            population_size=20,
            max_generations=10,
            algorithm=algo
        )
        
        if algo == "nsga2":
            optimizer = NSGAIIOptimizer(objectives, config)
        else:
            optimizer = MultiObjectiveOptimizer(objectives, config)
        
        result = optimizer.optimize()
        results[algo] = {
            'pareto_size': len(result.pareto_front.solutions),
            'fitness': result.average_fitness,
            'diversity': result.diversity_score,
            'time': result.optimization_time
        }
    
    # Compare results
    print("\n" + "="*80)
    print("Algorithm Comparison:")
    print("="*80)
    print(f"{'Algorithm':<15} {'Pareto Size':<12} {'Avg Fitness':<12} {'Diversity':<12} {'Time(s)':<10}")
    print("-"*70)
    
    for algo, metrics in results.items():
        print(f"{algo:<15} {metrics['pareto_size']:<12} {metrics['fitness']:<12.3f} "
              f"{metrics['diversity']:<12.3f} {metrics['time']:<10.3f}")
    
    print("="*70)


if __name__ == "__main__":
    # Test basic optimization
    basic_result = test_basic_optimization()
    
    print("\n" * 2)
    
    # Test NSGA-II
    nsga2_result = test_nsga2_optimization()
    
    print("\n" * 2)
    
    # Test balanced optimization
    balanced_result = test_balanced_optimization()
    
    print("\n" * 2)
    
    # Compare algorithms
    compare_algorithms()
    
    print("\n\n[SUCCESS] All multi-objective optimization tests completed successfully!")