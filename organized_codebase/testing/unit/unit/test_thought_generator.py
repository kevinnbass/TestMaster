"""
Test-Specific Tree-of-Thought Generator

Generates test-specific thoughts for intelligent test case creation.
Adapted from Swarm and PraisonAI's agent reasoning patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json
import hashlib
from datetime import datetime

from .tot_reasoning import (
    ThoughtNode, ThoughtGenerator, ThoughtEvaluator,
    EvaluationCriteria
)
from ...core.ast_abstraction import UniversalAST, UniversalFunction, UniversalClass


class TestStrategyType(Enum):
    """Types of testing strategies to consider."""
    HAPPY_PATH = "happy_path"              # Normal expected behavior
    EDGE_CASES = "edge_cases"              # Boundary conditions
    ERROR_HANDLING = "error_handling"      # Exception scenarios
    PERFORMANCE = "performance"            # Performance testing
    SECURITY = "security"                  # Security testing
    INTEGRATION = "integration"            # Integration with dependencies
    REGRESSION = "regression"              # Prevent regression
    PROPERTY_BASED = "property_based"      # Property-based testing
    MUTATION = "mutation"                  # Mutation testing
    FUZZING = "fuzzing"                    # Fuzz testing


@dataclass
class TestGenerationThought:
    """A thought about test generation."""
    strategy: TestStrategyType
    target_function: Optional[str] = None
    target_class: Optional[str] = None
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    coverage_impact: float = 0.0
    complexity: float = 0.0
    priority: int = 1
    rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy': self.strategy.value,
            'target_function': self.target_function,
            'target_class': self.target_class,
            'test_cases': self.test_cases,
            'coverage_impact': self.coverage_impact,
            'complexity': self.complexity,
            'priority': self.priority,
            'rationale': self.rationale
        }


@dataclass
class TestStrategyThought:
    """A thought about which testing strategy to use."""
    strategies: List[TestStrategyType] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    applicable_to: List[str] = field(default_factory=list)  # Function/class names
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategies': [s.value for s in self.strategies],
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'applicable_to': self.applicable_to
        }


@dataclass
class TestCoverageThought:
    """A thought about test coverage."""
    current_coverage: float = 0.0
    target_coverage: float = 0.0
    uncovered_functions: List[str] = field(default_factory=list)
    uncovered_branches: List[str] = field(default_factory=list)
    coverage_gaps: List[Dict[str, Any]] = field(default_factory=list)
    improvement_strategy: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_coverage': self.current_coverage,
            'target_coverage': self.target_coverage,
            'uncovered_functions': self.uncovered_functions,
            'uncovered_branches': self.uncovered_branches,
            'coverage_gaps': self.coverage_gaps,
            'improvement_strategy': self.improvement_strategy
        }


class TestThoughtGenerator(ThoughtGenerator):
    """Generates test-specific thoughts in the tree."""
    
    def __init__(self, universal_ast: UniversalAST = None):
        self.ast = universal_ast
        self.generated_thoughts: Set[str] = set()
        
        # Strategy weights based on importance
        self.strategy_weights = {
            TestStrategyType.HAPPY_PATH: 1.0,
            TestStrategyType.EDGE_CASES: 0.9,
            TestStrategyType.ERROR_HANDLING: 0.8,
            TestStrategyType.INTEGRATION: 0.7,
            TestStrategyType.PERFORMANCE: 0.6,
            TestStrategyType.SECURITY: 0.8,
            TestStrategyType.REGRESSION: 0.5,
            TestStrategyType.PROPERTY_BASED: 0.4,
            TestStrategyType.MUTATION: 0.3,
            TestStrategyType.FUZZING: 0.3
        }
    
    def generate(self, parent_node: ThoughtNode, context: Dict[str, Any]) -> List[ThoughtNode]:
        """Generate child test thoughts from parent."""
        children = []
        
        # Determine generation strategy based on parent content
        if parent_node.id == "root":
            # Initial thoughts: which strategies to use
            children = self._generate_strategy_thoughts(parent_node, context)
        elif "strategy" in parent_node.content:
            # Strategy selected: generate specific test cases
            children = self._generate_test_case_thoughts(parent_node, context)
        elif "test_cases" in parent_node.content:
            # Test cases generated: think about coverage
            children = self._generate_coverage_thoughts(parent_node, context)
        elif "coverage" in parent_node.content:
            # Coverage analyzed: think about optimization
            children = self._generate_optimization_thoughts(parent_node, context)
        
        return children
    
    def _generate_strategy_thoughts(self, parent: ThoughtNode, context: Dict[str, Any]) -> List[ThoughtNode]:
        """Generate thoughts about which testing strategies to use."""
        thoughts = []
        
        # Analyze the AST to determine applicable strategies
        if self.ast:
            # Strategy 1: Focus on complex functions
            complex_functions = self._find_complex_functions()
            if complex_functions:
                thought = TestStrategyThought(
                    strategies=[TestStrategyType.EDGE_CASES, TestStrategyType.ERROR_HANDLING],
                    reasoning="Complex functions require thorough edge case and error testing",
                    confidence=0.9,
                    applicable_to=complex_functions[:5]  # Top 5 complex functions
                )
                thoughts.append(self._create_thought_node(
                    f"{parent.id}_strategy_complex",
                    thought.to_dict(),
                    "Focus on complex functions"
                ))
            
            # Strategy 2: Focus on public API
            public_api = self._find_public_api()
            if public_api:
                thought = TestStrategyThought(
                    strategies=[TestStrategyType.HAPPY_PATH, TestStrategyType.INTEGRATION],
                    reasoning="Public API functions need comprehensive happy path and integration tests",
                    confidence=0.95,
                    applicable_to=public_api[:10]
                )
                thoughts.append(self._create_thought_node(
                    f"{parent.id}_strategy_api",
                    thought.to_dict(),
                    "Focus on public API"
                ))
            
            # Strategy 3: Security-sensitive functions
            security_functions = self._find_security_sensitive()
            if security_functions:
                thought = TestStrategyThought(
                    strategies=[TestStrategyType.SECURITY, TestStrategyType.FUZZING],
                    reasoning="Security-sensitive functions require security and fuzz testing",
                    confidence=0.85,
                    applicable_to=security_functions
                )
                thoughts.append(self._create_thought_node(
                    f"{parent.id}_strategy_security",
                    thought.to_dict(),
                    "Focus on security"
                ))
        
        # Default strategy if no AST
        if not thoughts:
            thought = TestStrategyThought(
                strategies=[TestStrategyType.HAPPY_PATH, TestStrategyType.EDGE_CASES],
                reasoning="Standard testing approach with happy path and edge cases",
                confidence=0.7,
                applicable_to=[]
            )
            thoughts.append(self._create_thought_node(
                f"{parent.id}_strategy_default",
                thought.to_dict(),
                "Default testing strategy"
            ))
        
        return thoughts
    
    def _generate_test_case_thoughts(self, parent: ThoughtNode, context: Dict[str, Any]) -> List[ThoughtNode]:
        """Generate specific test case thoughts based on selected strategy."""
        thoughts = []
        
        if "strategies" not in parent.content:
            return thoughts
        
        strategies = parent.content.get("strategies", [])
        applicable_to = parent.content.get("applicable_to", [])
        
        for strategy_str in strategies[:2]:  # Limit to 2 strategies per branch
            strategy = TestStrategyType(strategy_str)
            
            if strategy == TestStrategyType.HAPPY_PATH:
                thoughts.extend(self._generate_happy_path_tests(parent, applicable_to))
            elif strategy == TestStrategyType.EDGE_CASES:
                thoughts.extend(self._generate_edge_case_tests(parent, applicable_to))
            elif strategy == TestStrategyType.ERROR_HANDLING:
                thoughts.extend(self._generate_error_tests(parent, applicable_to))
            elif strategy == TestStrategyType.SECURITY:
                thoughts.extend(self._generate_security_tests(parent, applicable_to))
        
        return thoughts
    
    def _generate_coverage_thoughts(self, parent: ThoughtNode, context: Dict[str, Any]) -> List[ThoughtNode]:
        """Generate thoughts about test coverage improvements."""
        thoughts = []
        
        # Analyze current test cases
        test_cases = parent.content.get("test_cases", [])
        
        # Calculate estimated coverage
        if self.ast:
            total_functions = self.ast.total_functions
            covered_functions = len(set(tc.get("target") for tc in test_cases if tc.get("target")))
            coverage = (covered_functions / total_functions * 100) if total_functions > 0 else 0
            
            # Find gaps
            all_functions = self._get_all_functions()
            covered = set(tc.get("target") for tc in test_cases if tc.get("target"))
            uncovered = [f for f in all_functions if f not in covered]
            
            thought = TestCoverageThought(
                current_coverage=coverage,
                target_coverage=80.0,
                uncovered_functions=uncovered[:10],  # Top 10 uncovered
                improvement_strategy="Focus on high-priority uncovered functions"
            )
            
            thoughts.append(self._create_thought_node(
                f"{parent.id}_coverage_analysis",
                thought.to_dict(),
                "Coverage analysis"
            ))
            
            # If coverage is low, suggest more tests
            if coverage < 60:
                thought = TestCoverageThought(
                    current_coverage=coverage,
                    target_coverage=80.0,
                    uncovered_functions=uncovered[:20],
                    improvement_strategy="Generate additional tests for critical gaps"
                )
                thoughts.append(self._create_thought_node(
                    f"{parent.id}_coverage_expansion",
                    thought.to_dict(),
                    "Expand test coverage"
                ))
        
        return thoughts
    
    def _generate_optimization_thoughts(self, parent: ThoughtNode, context: Dict[str, Any]) -> List[ThoughtNode]:
        """Generate thoughts about test optimization."""
        thoughts = []
        
        # Terminal thought: final optimized test suite
        thought_content = {
            'optimization': 'complete',
            'final_coverage': parent.content.get('current_coverage', 0),
            'total_tests': len(parent.content.get('test_cases', [])),
            'quality_score': self._calculate_quality_score(parent.content)
        }
        
        node = self._create_thought_node(
            f"{parent.id}_final",
            thought_content,
            "Final optimized test suite"
        )
        node.is_terminal = True
        thoughts.append(node)
        
        return thoughts
    
    def _generate_happy_path_tests(self, parent: ThoughtNode, targets: List[str]) -> List[ThoughtNode]:
        """Generate happy path test cases."""
        thoughts = []
        
        for target in targets[:3]:  # Limit to 3 targets
            test_cases = []
            
            # Generate basic happy path test
            test_cases.append({
                'name': f'test_{target}_basic',
                'target': target,
                'type': 'happy_path',
                'inputs': self._generate_typical_inputs(target),
                'expected': 'success',
                'assertions': ['not_null', 'correct_type']
            })
            
            # Generate with valid inputs
            test_cases.append({
                'name': f'test_{target}_valid_inputs',
                'target': target,
                'type': 'happy_path',
                'inputs': self._generate_valid_inputs(target),
                'expected': 'success',
                'assertions': ['equals_expected', 'performance_acceptable']
            })
            
            thought = TestGenerationThought(
                strategy=TestStrategyType.HAPPY_PATH,
                target_function=target,
                test_cases=test_cases,
                coverage_impact=5.0,
                complexity=1.0,
                priority=1,
                rationale=f"Happy path tests for {target}"
            )
            
            thoughts.append(self._create_thought_node(
                f"{parent.id}_happy_{target}",
                thought.to_dict(),
                f"Happy path: {target}"
            ))
        
        return thoughts
    
    def _generate_edge_case_tests(self, parent: ThoughtNode, targets: List[str]) -> List[ThoughtNode]:
        """Generate edge case test cases."""
        thoughts = []
        
        for target in targets[:3]:
            test_cases = []
            
            # Boundary values
            test_cases.append({
                'name': f'test_{target}_boundaries',
                'target': target,
                'type': 'edge_case',
                'inputs': self._generate_boundary_inputs(target),
                'expected': 'handle_gracefully',
                'assertions': ['no_crash', 'valid_output']
            })
            
            # Empty/null inputs
            test_cases.append({
                'name': f'test_{target}_empty_inputs',
                'target': target,
                'type': 'edge_case',
                'inputs': self._generate_empty_inputs(target),
                'expected': 'handle_gracefully',
                'assertions': ['handle_none', 'handle_empty']
            })
            
            thought = TestGenerationThought(
                strategy=TestStrategyType.EDGE_CASES,
                target_function=target,
                test_cases=test_cases,
                coverage_impact=8.0,
                complexity=2.0,
                priority=2,
                rationale=f"Edge case tests for {target}"
            )
            
            thoughts.append(self._create_thought_node(
                f"{parent.id}_edge_{target}",
                thought.to_dict(),
                f"Edge cases: {target}"
            ))
        
        return thoughts
    
    def _generate_error_tests(self, parent: ThoughtNode, targets: List[str]) -> List[ThoughtNode]:
        """Generate error handling test cases."""
        thoughts = []
        
        for target in targets[:2]:
            test_cases = []
            
            # Invalid inputs
            test_cases.append({
                'name': f'test_{target}_invalid_inputs',
                'target': target,
                'type': 'error_handling',
                'inputs': self._generate_invalid_inputs(target),
                'expected': 'raise_exception',
                'assertions': ['throws_correct_exception', 'error_message_clear']
            })
            
            thought = TestGenerationThought(
                strategy=TestStrategyType.ERROR_HANDLING,
                target_function=target,
                test_cases=test_cases,
                coverage_impact=6.0,
                complexity=2.5,
                priority=2,
                rationale=f"Error handling tests for {target}"
            )
            
            thoughts.append(self._create_thought_node(
                f"{parent.id}_error_{target}",
                thought.to_dict(),
                f"Error handling: {target}"
            ))
        
        return thoughts
    
    def _generate_security_tests(self, parent: ThoughtNode, targets: List[str]) -> List[ThoughtNode]:
        """Generate security test cases."""
        thoughts = []
        
        for target in targets[:2]:
            test_cases = []
            
            # SQL injection attempts
            test_cases.append({
                'name': f'test_{target}_sql_injection',
                'target': target,
                'type': 'security',
                'inputs': ["'; DROP TABLE users; --", "<script>alert('XSS')</script>"],
                'expected': 'sanitized',
                'assertions': ['input_sanitized', 'no_injection']
            })
            
            thought = TestGenerationThought(
                strategy=TestStrategyType.SECURITY,
                target_function=target,
                test_cases=test_cases,
                coverage_impact=7.0,
                complexity=3.0,
                priority=1,
                rationale=f"Security tests for {target}"
            )
            
            thoughts.append(self._create_thought_node(
                f"{parent.id}_security_{target}",
                thought.to_dict(),
                f"Security: {target}"
            ))
        
        return thoughts
    
    def _create_thought_node(self, node_id: str, content: Dict[str, Any], description: str) -> ThoughtNode:
        """Create a thought node."""
        # Avoid duplicates
        thought_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
        if thought_hash in self.generated_thoughts:
            node_id = f"{node_id}_dup_{thought_hash[:8]}"
        self.generated_thoughts.add(thought_hash)
        
        return ThoughtNode(
            id=node_id,
            content={**content, 'description': description}
        )
    
    def _find_complex_functions(self) -> List[str]:
        """Find complex functions in the AST."""
        if not self.ast:
            return []
        
        complex_functions = []
        for module in self.ast.modules:
            for func in module.functions:
                if func.cyclomatic_complexity > 5 or func.lines_of_code > 20:
                    complex_functions.append(func.name)
        
        # Sort by complexity
        return sorted(complex_functions, key=lambda f: -self._get_function_complexity(f))[:10]
    
    def _find_public_api(self) -> List[str]:
        """Find public API functions."""
        if not self.ast:
            return []
        
        public_functions = []
        for module in self.ast.modules:
            for func in module.functions:
                if not func.name.startswith('_'):
                    public_functions.append(func.name)
            for cls in module.classes:
                for method in cls.methods:
                    if not method.name.startswith('_'):
                        public_functions.append(f"{cls.name}.{method.name}")
        
        return public_functions
    
    def _find_security_sensitive(self) -> List[str]:
        """Find security-sensitive functions."""
        if not self.ast:
            return []
        
        security_keywords = ['auth', 'login', 'password', 'token', 'encrypt', 'decrypt',
                           'hash', 'validate', 'sanitize', 'sql', 'query', 'exec']
        
        sensitive_functions = []
        for module in self.ast.modules:
            for func in module.functions:
                if any(keyword in func.name.lower() for keyword in security_keywords):
                    sensitive_functions.append(func.name)
        
        return sensitive_functions
    
    def _get_all_functions(self) -> List[str]:
        """Get all function names from AST."""
        if not self.ast:
            return []
        
        functions = []
        for module in self.ast.modules:
            functions.extend(func.name for func in module.functions)
            for cls in module.classes:
                functions.extend(f"{cls.name}.{method.name}" for method in cls.methods)
        
        return functions
    
    def _get_function_complexity(self, func_name: str) -> int:
        """Get complexity of a function."""
        if not self.ast:
            return 1
        
        for module in self.ast.modules:
            for func in module.functions:
                if func.name == func_name:
                    return func.cyclomatic_complexity
        return 1
    
    def _calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """Calculate quality score for test suite."""
        coverage = content.get('current_coverage', 0)
        test_count = len(content.get('test_cases', []))
        
        # Simple quality formula
        quality = (coverage * 0.6 + min(test_count / 10, 1.0) * 40)
        return min(quality, 100.0)
    
    def _generate_typical_inputs(self, target: str) -> List[Any]:
        """Generate typical inputs for a function."""
        return ["test_value", 123, True, ["item1", "item2"]]
    
    def _generate_valid_inputs(self, target: str) -> List[Any]:
        """Generate valid inputs for a function."""
        return ["valid_string", 42, False, {"key": "value"}]
    
    def _generate_boundary_inputs(self, target: str) -> List[Any]:
        """Generate boundary inputs for a function."""
        return [0, -1, "", None, [], {}, float('inf'), -float('inf')]
    
    def _generate_empty_inputs(self, target: str) -> List[Any]:
        """Generate empty inputs for a function."""
        return [None, "", [], {}, 0]
    
    def _generate_invalid_inputs(self, target: str) -> List[Any]:
        """Generate invalid inputs for a function."""
        return ["invalid\x00", -999999, {"invalid": None}, ["broken", None]]


class TestThoughtEvaluator(ThoughtEvaluator):
    """Evaluates test generation thoughts."""
    
    def __init__(self, target_coverage: float = 80.0):
        self.target_coverage = target_coverage
    
    def evaluate(self, node: ThoughtNode, criteria: List[EvaluationCriteria]) -> float:
        """Evaluate a test thought node."""
        content = node.content
        
        # Base score from criteria
        score = 0.0
        if criteria:
            total_weight = sum(c.weight for c in criteria)
            for criterion in criteria:
                score += criterion.evaluate(node) * (criterion.weight / total_weight)
        
        # Additional test-specific scoring
        if 'coverage_impact' in content:
            score += content['coverage_impact'] / 100.0
        
        if 'priority' in content:
            score += (5 - content['priority']) / 5.0
        
        if 'complexity' in content:
            # Prefer moderate complexity (not too simple, not too complex)
            complexity = content['complexity']
            if 1.5 <= complexity <= 2.5:
                score += 0.5
            elif complexity < 1.5:
                score += 0.3
            else:
                score += 0.2
        
        if 'confidence' in content:
            score += content['confidence'] * 0.5
        
        if 'quality_score' in content:
            score += content['quality_score'] / 100.0
        
        # Depth penalty (prefer shallower solutions)
        score -= node.depth * 0.1
        
        return max(0.0, min(score, 10.0))  # Clamp to [0, 10]