"""
Universal Tree-of-Thought Test Generation Integration

Integrates ToT reasoning with the universal test generation framework.
Combines patterns from OpenAI Swarm, Agency Swarm, and PraisonAI.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from .tot_reasoning import (
    TreeOfThoughtReasoner, ReasoningStrategy, EvaluationCriteria,
    ThoughtTree, ThoughtNode
)
from .test_thought_generator import (
    TestThoughtGenerator, TestThoughtEvaluator,
    TestGenerationThought, TestStrategyType
)
from ...core.ast_abstraction import UniversalAST
from ...core.framework_abstraction import (
    UniversalTest, UniversalTestCase, UniversalTestSuite,
    TestAssertion, AssertionType, TestMetadata,
    UniversalTestGenerator, TestGenerationConfig
)


@dataclass
class ToTGenerationConfig:
    """Configuration for Tree-of-Thought test generation."""
    # ToT parameters
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.BEST_FIRST
    max_reasoning_depth: int = 5
    max_iterations: int = 50
    beam_width: int = 3
    
    # Test generation parameters
    target_coverage: float = 80.0
    generate_all_strategies: bool = False
    prioritize_complex: bool = True
    prioritize_security: bool = True
    
    # Output parameters
    max_tests_per_function: int = 5
    combine_similar_tests: bool = True
    
    # Quality thresholds
    min_test_quality: float = 0.7
    min_confidence: float = 0.6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'reasoning_strategy': self.reasoning_strategy.value,
            'max_reasoning_depth': self.max_reasoning_depth,
            'max_iterations': self.max_iterations,
            'beam_width': self.beam_width,
            'target_coverage': self.target_coverage,
            'generate_all_strategies': self.generate_all_strategies,
            'prioritize_complex': self.prioritize_complex,
            'prioritize_security': self.prioritize_security,
            'max_tests_per_function': self.max_tests_per_function,
            'combine_similar_tests': self.combine_similar_tests,
            'min_test_quality': self.min_test_quality,
            'min_confidence': self.min_confidence
        }


@dataclass
class ToTGenerationResult:
    """Result of Tree-of-Thought test generation."""
    test_suite: UniversalTestSuite
    thought_tree: ThoughtTree
    best_path: List[ThoughtNode]
    
    # Metrics
    total_thoughts_generated: int = 0
    total_thoughts_evaluated: int = 0
    reasoning_depth_achieved: int = 0
    reasoning_time: float = 0.0
    
    # Quality metrics
    confidence_score: float = 0.0
    coverage_estimate: float = 0.0
    test_quality_score: float = 0.0
    
    # Insights
    key_insights: List[str] = field(default_factory=list)
    recommended_improvements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_suite_name': self.test_suite.name,
            'total_tests': self.test_suite.count_tests(),
            'total_assertions': self.test_suite.count_assertions(),
            'best_path_length': len(self.best_path),
            'total_thoughts_generated': self.total_thoughts_generated,
            'total_thoughts_evaluated': self.total_thoughts_evaluated,
            'reasoning_depth_achieved': self.reasoning_depth_achieved,
            'reasoning_time': self.reasoning_time,
            'confidence_score': self.confidence_score,
            'coverage_estimate': self.coverage_estimate,
            'test_quality_score': self.test_quality_score,
            'key_insights': self.key_insights,
            'recommended_improvements': self.recommended_improvements
        }


class UniversalToTTestGenerator:
    """Universal test generator using Tree-of-Thought reasoning."""
    
    def __init__(self, config: ToTGenerationConfig = None):
        self.config = config or ToTGenerationConfig()
        
        print(f"Universal ToT Test Generator initialized")
        print(f"   Strategy: {self.config.reasoning_strategy.value}")
        print(f"   Target coverage: {self.config.target_coverage}%")
        print(f"   Max depth: {self.config.max_reasoning_depth}")
    
    def generate_tests(self, universal_ast: UniversalAST) -> ToTGenerationResult:
        """Generate tests using Tree-of-Thought reasoning."""
        start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"Starting Tree-of-Thought Test Generation")
        print(f"   Project: {universal_ast.project_path}")
        print(f"   Modules: {len(universal_ast.modules)}")
        print(f"   Functions: {universal_ast.total_functions}")
        print(f"   Classes: {universal_ast.total_classes}")
        print(f"{'='*60}\n")
        
        # Initialize ToT components
        thought_generator = TestThoughtGenerator(universal_ast)
        thought_evaluator = TestThoughtEvaluator(self.config.target_coverage)
        
        # Create reasoner
        reasoner = TreeOfThoughtReasoner(
            thought_generator=thought_generator,
            thought_evaluator=thought_evaluator,
            strategy=self.config.reasoning_strategy,
            max_depth=self.config.max_reasoning_depth,
            max_iterations=self.config.max_iterations,
            beam_width=self.config.beam_width
        )
        
        # Add evaluation criteria
        self._add_evaluation_criteria(reasoner)
        
        # Initial thought: analyze the codebase
        initial_thought = {
            'task': 'generate_comprehensive_tests',
            'target': universal_ast.project_path,
            'modules': len(universal_ast.modules),
            'functions': universal_ast.total_functions,
            'classes': universal_ast.total_classes
        }
        
        # Execute reasoning
        print("Executing Tree-of-Thought reasoning...")
        thought_tree = reasoner.reason(initial_thought, {'ast': universal_ast})
        
        # Get best path through tree
        best_path = thought_tree.get_best_path()
        
        print(f"\nReasoning complete. Best path: {len(best_path)} thoughts")
        
        # Extract test cases from best path
        test_suite = self._extract_test_suite(best_path, universal_ast)
        
        # Calculate metrics
        reasoning_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = ToTGenerationResult(
            test_suite=test_suite,
            thought_tree=thought_tree,
            best_path=best_path,
            total_thoughts_generated=reasoner.nodes_generated,
            total_thoughts_evaluated=reasoner.nodes_evaluated,
            reasoning_depth_achieved=thought_tree.max_depth,
            reasoning_time=reasoning_time
        )
        
        # Calculate quality metrics
        self._calculate_quality_metrics(result, universal_ast)
        
        # Extract insights
        self._extract_insights(result, thought_tree)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _add_evaluation_criteria(self, reasoner: TreeOfThoughtReasoner):
        """Add evaluation criteria for test thoughts."""
        
        # Coverage impact criterion
        def coverage_evaluator(node: ThoughtNode) -> float:
            if 'coverage_impact' in node.content:
                return node.content['coverage_impact'] / 10.0
            elif 'current_coverage' in node.content:
                return node.content['current_coverage'] / 100.0
            return 0.5
        
        reasoner.add_criterion(EvaluationCriteria(
            name="coverage_impact",
            weight=2.0,
            evaluator=coverage_evaluator,
            description="Impact on test coverage"
        ))
        
        # Test quality criterion
        def quality_evaluator(node: ThoughtNode) -> float:
            if 'quality_score' in node.content:
                return node.content['quality_score'] / 100.0
            elif 'test_cases' in node.content:
                # More test cases = higher quality (up to a point)
                num_tests = len(node.content['test_cases'])
                return min(num_tests / 10.0, 1.0)
            return 0.5
        
        reasoner.add_criterion(EvaluationCriteria(
            name="test_quality",
            weight=1.5,
            evaluator=quality_evaluator,
            description="Quality of generated tests"
        ))
        
        # Strategy appropriateness criterion
        def strategy_evaluator(node: ThoughtNode) -> float:
            if 'confidence' in node.content:
                return node.content['confidence']
            elif 'strategy' in node.content:
                # Prioritize certain strategies
                strategy = node.content.get('strategy')
                if strategy == TestStrategyType.HAPPY_PATH.value:
                    return 0.9
                elif strategy == TestStrategyType.EDGE_CASES.value:
                    return 0.85
                elif strategy == TestStrategyType.ERROR_HANDLING.value:
                    return 0.8
                elif strategy == TestStrategyType.SECURITY.value:
                    return 0.9 if self.config.prioritize_security else 0.7
            return 0.5
        
        reasoner.add_criterion(EvaluationCriteria(
            name="strategy_appropriateness",
            weight=1.0,
            evaluator=strategy_evaluator,
            description="Appropriateness of testing strategy"
        ))
        
        # Complexity handling criterion
        def complexity_evaluator(node: ThoughtNode) -> float:
            if 'complexity' in node.content:
                complexity = node.content['complexity']
                # Prefer moderate complexity
                if 1.5 <= complexity <= 2.5:
                    return 1.0
                elif complexity < 1.5:
                    return 0.7
                else:
                    return 0.5
            return 0.6
        
        reasoner.add_criterion(EvaluationCriteria(
            name="complexity_handling",
            weight=0.8,
            evaluator=complexity_evaluator,
            description="Handling of test complexity"
        ))
    
    def _extract_test_suite(self, best_path: List[ThoughtNode], 
                           universal_ast: UniversalAST) -> UniversalTestSuite:
        """Extract test suite from best path through thought tree."""
        
        # Create test suite
        test_suite = UniversalTestSuite(
            name=f"ToT_TestSuite_{universal_ast.project_path.split('/')[-1]}",
            metadata=TestMetadata(
                tags=["tot-generated", "intelligent"],
                category="comprehensive"
            )
        )
        
        # Collect all test cases from path
        all_test_cases = {}  # Group by target
        
        for node in best_path:
            if 'test_cases' in node.content:
                test_cases = node.content['test_cases']
                target = node.content.get('target_function') or node.content.get('target_class', 'general')
                
                if target not in all_test_cases:
                    all_test_cases[target] = []
                
                all_test_cases[target].extend(test_cases)
        
        # Convert to UniversalTests
        for target, test_cases in all_test_cases.items():
            if not test_cases:
                continue
            
            # Create test case for target
            test_case = UniversalTestCase(
                name=f"Test_{target}",
                description=f"ToT-generated tests for {target}",
                metadata=TestMetadata(
                    tags=["tot", target.lower()],
                    category="unit"
                )
            )
            
            # Convert each test case to UniversalTest
            for i, tc in enumerate(test_cases[:self.config.max_tests_per_function]):
                test = self._convert_to_universal_test(tc, i)
                test_case.add_test(test)
            
            test_suite.add_test_case(test_case)
        
        # If no specific tests were generated, create basic tests
        if not test_suite.test_cases:
            test_suite = self._generate_fallback_tests(universal_ast)
        
        # Calculate metrics
        test_suite.calculate_metrics()
        
        return test_suite
    
    def _convert_to_universal_test(self, test_case: Dict[str, Any], index: int) -> UniversalTest:
        """Convert a test case from thought to UniversalTest."""
        test = UniversalTest(
            name=test_case.get('name', f'test_{index}'),
            test_function=self._generate_test_function(test_case),
            description=test_case.get('description', f"Test case {index}"),
            metadata=TestMetadata(
                tags=[test_case.get('type', 'general')],
                category=test_case.get('type', 'unit')
            )
        )
        
        # Add assertions based on test case
        assertions = test_case.get('assertions', [])
        for assertion in assertions:
            test.add_assertion(self._create_assertion(assertion))
        
        return test
    
    def _generate_test_function(self, test_case: Dict[str, Any]) -> str:
        """Generate test function code."""
        target = test_case.get('target', 'unknown_function')
        inputs = test_case.get('inputs', [])
        
        # Simple test function generation
        if inputs:
            input_str = ', '.join(str(i) for i in inputs[:3])  # Limit inputs
            return f"result = {target}({input_str})"
        else:
            return f"result = {target}()"
    
    def _create_assertion(self, assertion_type: str) -> TestAssertion:
        """Create assertion from type string."""
        assertion_map = {
            'not_null': AssertionType.NOT_NULL,
            'correct_type': AssertionType.INSTANCE_OF,
            'equals_expected': AssertionType.EQUAL,
            'no_crash': AssertionType.NOT_THROWS,
            'handle_none': AssertionType.NOT_THROWS,
            'throws_correct_exception': AssertionType.THROWS,
            'input_sanitized': AssertionType.NOT_CONTAINS
        }
        
        assertion_type_enum = assertion_map.get(assertion_type, AssertionType.NOT_NULL)
        
        return TestAssertion(
            assertion_type=assertion_type_enum,
            actual="result",
            message=f"Check {assertion_type}"
        )
    
    def _generate_fallback_tests(self, universal_ast: UniversalAST) -> UniversalTestSuite:
        """Generate fallback tests if ToT didn't produce specific tests."""
        print("Generating fallback tests...")
        
        # Use standard test generator as fallback
        config = TestGenerationConfig(
            target_coverage=self.config.target_coverage,
            generate_edge_cases=True,
            generate_error_tests=True
        )
        
        generator = UniversalTestGenerator(config)
        result = generator.generate_tests_from_ast(universal_ast)
        
        return result.test_suite
    
    def _calculate_quality_metrics(self, result: ToTGenerationResult, universal_ast: UniversalAST):
        """Calculate quality metrics for the result."""
        
        # Confidence score from best path
        if result.best_path:
            confidences = [n.content.get('confidence', 0.5) for n in result.best_path 
                          if 'confidence' in n.content]
            result.confidence_score = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Coverage estimate
        total_functions = universal_ast.total_functions
        tested_functions = len(set(
            test.name.replace('test_', '').split('_')[0]
            for test_case in result.test_suite.test_cases
            for test in test_case.tests
        ))
        result.coverage_estimate = (tested_functions / total_functions * 100) if total_functions > 0 else 0
        
        # Test quality score
        result.test_quality_score = result.test_suite.test_quality_score
    
    def _extract_insights(self, result: ToTGenerationResult, thought_tree: ThoughtTree):
        """Extract insights from the reasoning process."""
        
        # Analyze thought distribution
        strategy_counts = {}
        for node in thought_tree.nodes.values():
            if 'strategy' in node.content:
                strategy = node.content['strategy']
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Key insights
        if strategy_counts:
            most_common = max(strategy_counts, key=strategy_counts.get)
            result.key_insights.append(
                f"Most explored strategy: {most_common} ({strategy_counts[most_common]} thoughts)"
            )
        
        result.key_insights.append(
            f"Reasoning depth achieved: {thought_tree.max_depth} levels"
        )
        
        result.key_insights.append(
            f"Average branching factor: {thought_tree.get_statistics()['average_branching_factor']:.2f}"
        )
        
        # Recommendations
        if result.coverage_estimate < self.config.target_coverage:
            result.recommended_improvements.append(
                f"Increase test coverage from {result.coverage_estimate:.1f}% to {self.config.target_coverage}%"
            )
        
        if result.test_quality_score < 70:
            result.recommended_improvements.append(
                "Improve test quality by adding more assertions and edge cases"
            )
        
        pruned = thought_tree.pruned_branches
        if pruned > 10:
            result.recommended_improvements.append(
                f"Consider more focused reasoning (pruned {pruned} branches)"
            )
    
    def _print_summary(self, result: ToTGenerationResult):
        """Print summary of generation results."""
        print(f"\n{'='*60}")
        print(f"Tree-of-Thought Test Generation Summary")
        print(f"{'='*60}")
        
        print(f"\n[STATS] Test Suite Generated:")
        print(f"   Test cases: {len(result.test_suite.test_cases)}")
        print(f"   Total tests: {result.test_suite.count_tests()}")
        print(f"   Total assertions: {result.test_suite.count_assertions()}")
        
        print(f"\n[TREE] Reasoning Process:")
        print(f"   Thoughts generated: {result.total_thoughts_generated}")
        print(f"   Thoughts evaluated: {result.total_thoughts_evaluated}")
        print(f"   Reasoning depth: {result.reasoning_depth_achieved}")
        print(f"   Best path length: {len(result.best_path)}")
        print(f"   Time taken: {result.reasoning_time:.2f}s")
        
        print(f"\n[METRICS] Quality Metrics:")
        print(f"   Confidence score: {result.confidence_score:.2f}")
        print(f"   Coverage estimate: {result.coverage_estimate:.1f}%")
        print(f"   Test quality score: {result.test_quality_score:.1f}")
        
        if result.key_insights:
            print(f"\n[INSIGHTS] Key Insights:")
            for insight in result.key_insights:
                print(f"   * {insight}")
        
        if result.recommended_improvements:
            print(f"\n[RECOMMENDATIONS] Recommendations:")
            for rec in result.recommended_improvements:
                print(f"   * {rec}")
        
        print(f"\n{'='*60}\n")