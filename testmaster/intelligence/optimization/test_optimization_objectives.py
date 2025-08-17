"""
Test-Specific Optimization Objectives

Defines optimization objectives for test generation.
Adapted from Agency Swarm's goal definitions and PraisonAI's metrics.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import math

from .multi_objective_optimizer import OptimizationObjective, ObjectiveType
from ...core.framework_abstraction import UniversalTestSuite
from ...core.ast_abstraction import UniversalAST


class CoverageObjective(OptimizationObjective):
    """Optimize for maximum code coverage."""
    
    def __init__(self, target_coverage: float = 80.0, universal_ast: UniversalAST = None):
        self.universal_ast = universal_ast
        
        super().__init__(
            name="coverage",
            type=ObjectiveType.MAXIMIZE,
            weight=2.0,  # High priority
            min_value=0.0,
            max_value=100.0,
            target_value=target_coverage,
            description="Maximize code coverage percentage"
        )
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate coverage for a test suite solution."""
        if hasattr(solution, 'test_suite'):
            test_suite = solution.test_suite
            
            if self.universal_ast:
                # Calculate actual coverage
                covered_functions = self._get_covered_functions(test_suite)
                total_functions = self.universal_ast.total_functions
                
                if total_functions > 0:
                    coverage = (len(covered_functions) / total_functions) * 100
                else:
                    coverage = 0.0
            else:
                # Estimate coverage based on test count
                test_count = test_suite.count_tests() if hasattr(test_suite, 'count_tests') else 0
                coverage = min(test_count * 2, 100)  # Rough estimate: 2% per test
            
            return coverage
        
        # For other solution types, extract coverage from genes
        if hasattr(solution, 'genes'):
            # Assume first gene represents coverage
            return solution.genes[0] * 100 if solution.genes else 0.0
        
        return 0.0
    
    def _get_covered_functions(self, test_suite: UniversalTestSuite) -> set:
        """Get set of functions covered by test suite."""
        covered = set()
        
        for test_case in test_suite.test_cases:
            for test in test_case.tests:
                # Extract function name from test
                if hasattr(test, 'test_function'):
                    # Parse function name from test code
                    import re
                    matches = re.findall(r'(\w+)\(', test.test_function)
                    covered.update(matches)
        
        return covered


class PerformanceObjective(OptimizationObjective):
    """Optimize for test execution performance."""
    
    def __init__(self, max_time_ms: float = 5000.0):
        super().__init__(
            name="performance",
            type=ObjectiveType.MINIMIZE,
            weight=1.0,
            min_value=0.0,
            max_value=max_time_ms,
            description="Minimize test execution time"
        )
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate performance of test suite."""
        if hasattr(solution, 'test_suite'):
            test_suite = solution.test_suite
            
            # Estimate execution time based on test complexity
            total_time = 0.0
            
            for test_case in test_suite.test_cases:
                for test in test_case.tests:
                    # Base time per test
                    base_time = 10.0  # 10ms base
                    
                    # Add time for assertions
                    assertion_time = len(test.assertions) * 2.0 if hasattr(test, 'assertions') else 0
                    
                    # Add time for async tests
                    async_time = 20.0 if hasattr(test, 'is_async') and test.is_async else 0
                    
                    # Add time for fixtures
                    fixture_time = len(test.fixtures) * 5.0 if hasattr(test, 'fixtures') else 0
                    
                    total_time += base_time + assertion_time + async_time + fixture_time
            
            return total_time
        
        # For other solution types
        if hasattr(solution, 'genes') and len(solution.genes) > 1:
            return solution.genes[1] * self.max_value
        
        return 100.0  # Default moderate time


class QualityObjective(OptimizationObjective):
    """Optimize for test quality."""
    
    def __init__(self, min_quality_score: float = 70.0):
        super().__init__(
            name="quality",
            type=ObjectiveType.MAXIMIZE,
            weight=1.5,
            min_value=0.0,
            max_value=100.0,
            target_value=min_quality_score,
            description="Maximize test quality score"
        )
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate test quality."""
        if hasattr(solution, 'test_suite'):
            test_suite = solution.test_suite
            
            quality_factors = []
            
            # Factor 1: Assertion density
            test_count = test_suite.count_tests() if hasattr(test_suite, 'count_tests') else 1
            assertion_count = test_suite.count_assertions() if hasattr(test_suite, 'count_assertions') else 0
            
            if test_count > 0:
                assertion_density = assertion_count / test_count
                if 3 <= assertion_density <= 5:
                    quality_factors.append(1.0)
                elif 1 <= assertion_density < 3:
                    quality_factors.append(0.7)
                elif 5 < assertion_density <= 10:
                    quality_factors.append(0.8)
                else:
                    quality_factors.append(0.5)
            
            # Factor 2: Test diversity (different test types)
            test_types = set()
            for test_case in test_suite.test_cases:
                for test in test_case.tests:
                    if hasattr(test, 'metadata') and hasattr(test.metadata, 'category'):
                        test_types.add(test.metadata.category)
            
            diversity_score = min(len(test_types) / 5.0, 1.0)  # Up to 5 different types
            quality_factors.append(diversity_score)
            
            # Factor 3: Documentation (tests with descriptions)
            documented_tests = 0
            for test_case in test_suite.test_cases:
                for test in test_case.tests:
                    if hasattr(test, 'description') and test.description:
                        documented_tests += 1
            
            doc_score = documented_tests / test_count if test_count > 0 else 0
            quality_factors.append(doc_score)
            
            # Calculate overall quality
            if quality_factors:
                quality = sum(quality_factors) / len(quality_factors) * 100
            else:
                quality = 50.0  # Default medium quality
            
            return quality
        
        # For other solution types
        if hasattr(solution, 'genes') and len(solution.genes) > 2:
            return solution.genes[2] * 100
        
        return 50.0


class SecurityObjective(OptimizationObjective):
    """Optimize for security test coverage."""
    
    def __init__(self, min_security_tests: int = 10):
        super().__init__(
            name="security",
            type=ObjectiveType.MAXIMIZE,
            weight=1.2,
            min_value=0.0,
            max_value=100.0,
            description="Maximize security test coverage"
        )
        self.min_security_tests = min_security_tests
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate security testing coverage."""
        if hasattr(solution, 'test_suite'):
            test_suite = solution.test_suite
            
            security_score = 0.0
            security_tests = 0
            
            # Count security-related tests
            for test_case in test_suite.test_cases:
                for test in test_case.tests:
                    # Check if test is security-related
                    is_security = False
                    
                    # Check by name
                    if hasattr(test, 'name'):
                        security_keywords = ['security', 'auth', 'injection', 'xss', 'csrf', 
                                           'sql', 'sanitize', 'validate', 'encrypt']
                        if any(keyword in test.name.lower() for keyword in security_keywords):
                            is_security = True
                    
                    # Check by tags
                    if hasattr(test, 'metadata') and hasattr(test.metadata, 'tags'):
                        if 'security' in test.metadata.tags:
                            is_security = True
                    
                    if is_security:
                        security_tests += 1
            
            # Calculate score
            if security_tests >= self.min_security_tests:
                security_score = 100.0
            else:
                security_score = (security_tests / self.min_security_tests) * 100
            
            return security_score
        
        # For other solution types
        if hasattr(solution, 'genes') and len(solution.genes) > 3:
            return solution.genes[3] * 100
        
        return 0.0


class MaintainabilityObjective(OptimizationObjective):
    """Optimize for test maintainability."""
    
    def __init__(self):
        super().__init__(
            name="maintainability",
            type=ObjectiveType.MAXIMIZE,
            weight=0.8,
            min_value=0.0,
            max_value=100.0,
            description="Maximize test maintainability"
        )
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate test maintainability."""
        if hasattr(solution, 'test_suite'):
            test_suite = solution.test_suite
            
            maintainability_factors = []
            
            # Factor 1: Modularity (tests organized in test cases)
            test_cases = len(test_suite.test_cases) if hasattr(test_suite, 'test_cases') else 0
            tests = test_suite.count_tests() if hasattr(test_suite, 'count_tests') else 1
            
            if tests > 0 and test_cases > 0:
                modularity = min(test_cases / (tests / 10), 1.0)  # Ideal: 10 tests per case
                maintainability_factors.append(modularity)
            
            # Factor 2: Use of fixtures (reduces duplication)
            fixtures_used = 0
            for test_case in test_suite.test_cases:
                if hasattr(test_case, 'fixtures'):
                    fixtures_used += len(test_case.fixtures)
                for test in test_case.tests:
                    if hasattr(test, 'fixtures'):
                        fixtures_used += len(test.fixtures)
            
            fixture_score = min(fixtures_used / 10, 1.0)  # Up to 10 fixtures
            maintainability_factors.append(fixture_score)
            
            # Factor 3: Setup/teardown usage
            setup_teardown_score = 0.0
            for test_case in test_suite.test_cases:
                if hasattr(test_case, 'setup_method') and test_case.setup_method:
                    setup_teardown_score += 0.25
                if hasattr(test_case, 'teardown_method') and test_case.teardown_method:
                    setup_teardown_score += 0.25
                if hasattr(test_case, 'setup_class') and test_case.setup_class:
                    setup_teardown_score += 0.25
                if hasattr(test_case, 'teardown_class') and test_case.teardown_class:
                    setup_teardown_score += 0.25
            
            maintainability_factors.append(min(setup_teardown_score, 1.0))
            
            # Calculate overall maintainability
            if maintainability_factors:
                maintainability = sum(maintainability_factors) / len(maintainability_factors) * 100
            else:
                maintainability = 50.0
            
            return maintainability
        
        # For other solution types
        if hasattr(solution, 'genes') and len(solution.genes) > 4:
            return solution.genes[4] * 100
        
        return 50.0


class CostObjective(OptimizationObjective):
    """Optimize for test generation and maintenance cost."""
    
    def __init__(self, max_api_calls: int = 100):
        super().__init__(
            name="cost",
            type=ObjectiveType.MINIMIZE,
            weight=0.7,
            min_value=0.0,
            max_value=float(max_api_calls),
            description="Minimize test generation cost"
        )
        self.max_api_calls = max_api_calls
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate cost of test generation."""
        if hasattr(solution, 'test_suite'):
            test_suite = solution.test_suite
            
            # Estimate API calls needed
            api_calls = 0
            
            # Base cost per test
            test_count = test_suite.count_tests() if hasattr(test_suite, 'count_tests') else 0
            api_calls += test_count * 0.5  # 0.5 calls per test on average
            
            # Additional cost for complex tests
            for test_case in test_suite.test_cases:
                for test in test_case.tests:
                    # Parameterized tests cost more
                    if hasattr(test, 'parameters') and test.parameters:
                        api_calls += len(test.parameters) * 0.2
                    
                    # Async tests cost more
                    if hasattr(test, 'is_async') and test.is_async:
                        api_calls += 0.3
            
            return min(api_calls, self.max_api_calls)
        
        # For other solution types
        if hasattr(solution, 'genes') and len(solution.genes) > 5:
            return solution.genes[5] * self.max_api_calls
        
        return 10.0  # Default moderate cost


@dataclass
class CompoundObjective(OptimizationObjective):
    """Combines multiple objectives into one."""
    
    def __init__(self, objectives: List[OptimizationObjective], name: str = "compound"):
        self.sub_objectives = objectives
        
        # Calculate combined weight
        total_weight = sum(obj.weight for obj in objectives)
        
        super().__init__(
            name=name,
            type=ObjectiveType.MAXIMIZE,
            weight=total_weight / len(objectives) if objectives else 1.0,
            min_value=0.0,
            max_value=100.0,
            description=f"Compound objective combining {len(objectives)} objectives"
        )
    
    def evaluate(self, solution: Any) -> float:
        """Evaluate compound objective."""
        scores = []
        weights = []
        
        for obj in self.sub_objectives:
            value = obj.evaluate(solution)
            normalized = obj.normalize(value)
            scores.append(normalized)
            weights.append(obj.weight)
        
        # Weighted average
        if weights:
            total_weight = sum(weights)
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            return (weighted_sum / total_weight) * 100 if total_weight > 0 else 0.0
        
        return 0.0


class BalancedTestObjective(CompoundObjective):
    """Balanced objective for comprehensive test generation."""
    
    def __init__(self, universal_ast: UniversalAST = None):
        objectives = [
            CoverageObjective(target_coverage=80.0, universal_ast=universal_ast),
            QualityObjective(min_quality_score=70.0),
            PerformanceObjective(max_time_ms=5000.0),
            SecurityObjective(min_security_tests=10),
            MaintainabilityObjective(),
            CostObjective(max_api_calls=100)
        ]
        
        super().__init__(
            objectives=objectives,
            name="balanced_testing"
        )